"""
run_experiments.py
==================

Driver for the survival-aware MPC experiments (Exp 0, A-E) of
``Survival_Aware_iSHIPMENT_Formulation.docx``.

Every experiment runs on the frozen strategic network (one solve per demand
scale at alpha = $500K, shared by all policies -- Exp C is the sole exception),
with manufacturing failures ON, N_rep replications over yield seeds, and common
random numbers across policies.

All figures are written as PNGs to ``figures/``; nothing here calls
``plt.show()``.  Every replication records the seed that produced it.

Reported metrics are only the five the doc names: expected high-risk patients
lost (primary), total expected patients lost, total cost / cost per therapy,
mean hold by tier, and the share of the best_achievable gap closed.

Usage
    python run_experiments.py --exp A1 --scale 50      # the Gantt sanity check
    python run_experiments.py --exp all --scale 200
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")                    # never rely on plt.show()
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

import cart_data as cd                   # noqa: E402
import policies as pol                   # noqa: E402
import simulation as sim                 # noqa: E402
import strategic as st                   # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "results", "mpc")

N_REP = 30                               # confirmed: 30 yield seeds
SEEDS = list(range(N_REP))

# ---------------------------------------------------------------------------
# Palette.  Validated for colour-vision deficiency and contrast (adjacent-pair
# dE >= 8 under deutan/protan/tritan, normal-vision dE >= 15, >= 3:1 against
# the surface); tiers are ordered dark-red -> blue by severity.
# ---------------------------------------------------------------------------
TIER_COLOR = {"H": "#B2182B", "M": "#EF8A62", "L": "#2166AC"}
TIER_LABEL = {"H": "High risk", "M": "Medium risk", "L": "Low risk"}
POLICY_COLOR = {"fifo": "#B07A00", "survival_index": "#2166AC",
                "static_survival": "#00A39A", "adaptive_mpc": "#C2185B"}
BOUND_COLOR = "#3A3A38"                  # best_achievable: a reference, not a series
INK, MUTED, GRIDC = "#1c1c1a", "#6b6b66", "#d8d8d2"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 9,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRIDC, "grid.linewidth": 0.6,
})


# ---------------------------------------------------------------------------
# Replication driver
# ---------------------------------------------------------------------------
def run_policy(plan, name, seeds=SEEDS, cfg=None, **kw):
    """Run one policy over ``seeds`` on a frozen plan; returns the SimResults."""
    cfg = cfg or sim.SimConfig()
    out = []
    for s in seeds:
        policy = pol.build(name, **kw)
        t0 = time.time()
        res = sim.simulate(plan, policy, seed=s, cfg=cfg)
        res.metrics["wall_s"] = round(time.time() - t0, 2)
        res.metrics["seed"] = s
        res.metrics["policy"] = name
        if name == "best_achievable":
            res.metrics["bound_status"] = policy.status
            res.metrics["bound_objective"] = policy.bound
        out.append(res)
    return out


def mean_metrics(results, keys=None):
    """Average the per-replication metrics, with a standard error for each."""
    keys = keys or [k for k, v in results[0].metrics.items()
                    if isinstance(v, (int, float)) and v is not None]
    agg = {}
    for k in keys:
        vals = [r.metrics.get(k) for r in results]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue
        agg[k] = statistics.fmean(vals)
        agg[k + "_se"] = (statistics.stdev(vals) / len(vals) ** 0.5
                          if len(vals) > 1 else 0.0)
    agg["n_rep"] = len(results)
    return agg


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def save_csv(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"  [csv] {path}")


def run_all(plan, names, seeds=SEEDS, cfg=None, **kw):
    """Run several policies over the same seeds -- common random numbers."""
    out = {}
    for name in names:
        t0 = time.time()
        out[name] = run_policy(plan, name, seeds=seeds, cfg=cfg, **kw)
        m = mean_metrics(out[name])
        print(f"    {name:16s} E[lost]={m['expected_lost']:.3f} "
              f"(H {m['expected_lost_H']:.3f})  cost=${m['total_cost']:,.0f}  "
              f"spill={m['spillover']:.1f}  [{time.time() - t0:.0f}s]", flush=True)
    return out


def gap_closed(mean_by_policy, key="expected_lost_H", ref="fifo",
               bound="best_achievable"):
    """Share of the achievable gap each policy closes, on ``key``.

    ``fifo`` -- serve in arrival order, no objective -- is the do-nothing
    reference; ``best_achievable`` is the perfect-information floor.  A policy
    that reaches the floor scores 1.0.
    """
    lo = mean_by_policy[bound][key]
    hi = mean_by_policy[ref][key]
    span = hi - lo
    if abs(span) < 1e-12:
        return {p: None for p in mean_by_policy}
    return {p: (hi - m[key]) / span for p, m in mean_by_policy.items()}


def rescale_arrivals(plan, span, window_start=1):
    """Exp B: stretch or compress the arrival window, holding N and the
    network fixed, so that offered load = demand density / capacity moves.

    Only the collection days change: the strategic network, m(i), the modes and
    the static serving order are the ones the single strategic solve fixed.
    """
    out = copy.deepcopy(plan)
    days = [p.t0 for p in plan.patients.values()]
    lo, hi = min(days), max(days)
    for p in out.patients.values():
        frac = (p.t0 - lo) / (hi - lo)
        p.t0 = int(window_start + round(frac * (span - 1)))
    return out


def span_for_load(plan, load, tmfe=7):
    """Arrival span that puts the instance at the requested offered load."""
    cap = sum(plan.fcap[m] for m in plan.opened) / tmfe    # starts per day
    return max(2, int(round(plan.n / (load * cap))))


# ---------------------------------------------------------------------------
# Exp A -- figA1: the emergent prioritisation mechanism
# ---------------------------------------------------------------------------
def fig_a1_gantt(plan, seed=0, cfg=None, stem="figA1_schedule_gantt_fifo_vs_mpc"):
    """Manufacturing-start schedule, fifo vs adaptive_mpc, patients by tier.

    One replication (the seed is printed on the figure) so that the two panels
    show the SAME realised failures -- any difference in the schedule is the
    policy, not luck.
    """
    cfg = cfg or sim.SimConfig()
    runs = {name: sim.simulate(plan, pol.build(name), seed=seed, cfg=cfg)
            for name in ("fifo", "adaptive_mpc")}

    # Rows: tier block (H, M, L), then arrival day -- the same order in both
    # panels so a patient sits on one line across the figure.
    order = sorted(plan.patients.values(),
                   key=lambda p: (cd.TIER_ORDER.index(p.tier), p.t0, p.pid))
    row = {p.pid: i for i, p in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.6), sharey=True)
    for ax, name in zip(axes, ("fifo", "adaptive_mpc")):
        res = runs[name]
        for p in order:
            r, y = res.records[p.pid], row[p.pid]
            col = TIER_COLOR[p.tier]
            ax.plot([p.t0], [y], marker="|", ms=5, color=col, alpha=.75)
            for k, ready in enumerate(r.ready_days):
                starts = [s for s in r.starts if s["attempt"] == k + 1]
                end = starts[0]["day"] if starts else ready
                ax.plot([ready, end], [y, y], color=col, lw=0.8, alpha=.40,
                        solid_capstyle="butt")           # the queue wait (hold)
            for s in r.starts:
                ax.barh(y, cfg.tmfe, left=s["day"], height=0.74, color=col,
                        edgecolor="#fcfcfb", linewidth=0.6,
                        alpha=1.0 if s["outcome"] != "fail" else 0.45,
                        hatch="" if s["outcome"] != "fail" else "///")
            if r.status == "lost":
                ax.plot([cfg.horizon + 4], [y], marker="x", ms=4.5,
                        color=BOUND_COLOR, mew=1.1)
        m = res.metrics
        ax.set_title(
            f"{name}\nhigh-risk E[lost] {m['expected_lost_H']:.2f}   "
            f"mean hold H {m['mean_hold_H']:.1f} d / L {m['mean_hold_L']:.1f} d",
            fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Day")
        ax.grid(axis="x", alpha=.55)
        ax.set_axisbelow(True)
        ax.set_xlim(0, cfg.horizon + 8)

    # tier bands on the shared y-axis
    axes[0].set_ylabel("Patients, grouped by risk tier")
    ticks, labels = [], []
    for u in cd.TIER_ORDER:
        rows = [row[p.pid] for p in order if p.tier == u]
        ticks.append(statistics.fmean(rows))
        labels.append(f"{TIER_LABEL[u]}\n(n={len(rows)})")
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels, fontsize=9, color=INK)
    axes[0].invert_yaxis()

    handles = [Patch(facecolor=TIER_COLOR[u], label=TIER_LABEL[u])
               for u in cd.TIER_ORDER]
    handles += [Patch(facecolor="#999993", hatch="///", alpha=.45,
                      label="failed batch (remade)"),
                plt.Line2D([], [], color="#999993", lw=0.8, alpha=.6,
                           label="waiting for a slot (hold)"),
                plt.Line2D([], [], color=BOUND_COLOR, marker="x", ls="none",
                           label="lost")]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Manufacturing-start schedule, N = {plan.n} "
        f"({'+'.join(plan.opened)}, offered load {plan.offered_load():.2f}), "
        f"yield seed {seed}",
        fontsize=11.5, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, stem + ".png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  [fig] {out}")
    return out, runs


# ---------------------------------------------------------------------------
# Exp 0 -- tier survival calibration (setup, not a result)
# ---------------------------------------------------------------------------
def fig0_tier_survival(stem="fig0_tier_survival_curves"):
    """Survival vs turnaround by tier, and the marginal one-day survival loss.

    The right panel is the life-value-weighted signal the index rule (P8)
    actually ranks on; the unweighted decline is drawn faintly behind it for
    reference.
    """
    days = list(cd.D_RANGE)
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.5))

    for u in cd.TIER_ORDER:
        s = [cd.survival(u, d) for d in days]
        ax[0].plot(days, s, color=TIER_COLOR[u], lw=2,
                   label=f"{TIER_LABEL[u]}  (w={cd.W_RISK[u]:.2f})")
        ax[0].annotate(f"{s[-1]:.3f}", (days[-1], s[-1]), fontsize=8.5,
                       color=TIER_COLOR[u], xytext=(5, -3),
                       textcoords="offset points")
        raw = [cd.survival(u, d) - cd.survival(u, d + 1) for d in days]
        ax[1].plot(days, raw, color=TIER_COLOR[u], lw=1.1, alpha=.32, ls=":")
        ax[1].plot(days, [cd.ALPHA_W[u] * r for r in raw], color=TIER_COLOR[u],
                   lw=2,
                   label=f"{TIER_LABEL[u]}  ($\\alpha_u$=\\${cd.ALPHA_TIER[u]/1e6:g}M)")

    ax[0].set_xlabel("Turnaround time TRT [days]")
    ax[0].set_ylabel("Survival probability $S_u$(TRT)")
    ax[0].set_title("Survival declines faster for higher-risk tiers",
                    fontsize=10, loc="left")
    ax[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    ax[1].set_xlabel("Turnaround time TRT [days]")
    ax[1].set_ylabel(r"$(\alpha_u/\alpha_{\mathrm{ref}})\,"
                     r"[\,S_u(t) - S_u(t{+}1)\,]$")
    ax[1].set_title("Marginal cost of one more day of waiting\n"
                    "(solid: weighted, the P8 priority signal; dotted: unweighted)",
                    fontsize=10, loc="left")
    ax[1].legend(fontsize=8.5, frameon=False)
    for a in ax:
        a.grid(alpha=.5)
        a.set_axisbelow(True)
        a.margins(x=.04)

    fig.suptitle("Tier survival calibration: "
                 r"$S_u(t)=(1-w_u)^{t/42}$, $\gamma=1$, $\eta=42$, $\kappa=1$",
                 fontsize=11.5, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, stem)


# ---------------------------------------------------------------------------
# Exp A -- figA2: where the waiting goes
# ---------------------------------------------------------------------------
def fig_a2_holdtime(runs, plan, stem="figA2_holdtime_by_tier"):
    """Hold-time distribution by risk tier, fifo vs adaptive_mpc.

    Hold is the total time a patient spends queueing for a manufacturing slot,
    summed over its attempts, pooled over every replication.
    """
    names = ["fifo", "adaptive_mpc"]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    pos, ticks, labels = [], [], []
    for i, u in enumerate(cd.TIER_ORDER):
        for k, name in enumerate(names):
            holds = [r.hold_total for res in runs[name]
                     for r in res.records.values() if r.tier == u]
            y = i * 3 + k
            bp = ax.boxplot([holds], positions=[y], widths=.72,
                            patch_artist=True, showfliers=False, orientation="horizontal",
                            medianprops=dict(color="#fcfcfb", lw=1.4),
                            whiskerprops=dict(color=TIER_COLOR[u], lw=1),
                            capprops=dict(color=TIER_COLOR[u], lw=1))
            box = bp["boxes"][0]
            box.set_facecolor(TIER_COLOR[u])
            box.set_edgecolor(TIER_COLOR[u])
            if name == "fifo":                    # secondary encoding, not colour
                box.set_facecolor("none")
                box.set_hatch("///")
            ax.plot([statistics.fmean(holds)], [y], marker="D", ms=5,
                    color=TIER_COLOR[u], mec="#fcfcfb", mew=.8, zorder=3)
            pos.append(y)
            labels.append(f"{TIER_LABEL[u]} - {name}")
            ticks.append(y)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Hold: days waiting for a manufacturing slot (all attempts)")
    ax.grid(axis="x", alpha=.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor="none", edgecolor=MUTED, hatch="///", label="fifo"),
               Patch(facecolor=MUTED, label="adaptive_mpc"),
               plt.Line2D([], [], marker="D", ls="none", color=MUTED, label="mean")]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title(f"Survival awareness moves the waiting onto the low-risk tier\n"
                 f"N = {plan.n}, {N_REP} yield seeds, offered load "
                 f"{plan.offered_load():.2f}", fontsize=10.5, loc="left")
    fig.tight_layout()
    return _save(fig, stem)


# ---------------------------------------------------------------------------
# Exp B -- congestion
# ---------------------------------------------------------------------------
LOADS = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]


def exp_b(plan, names, seeds=SEEDS, loads=LOADS, cfg=None,
          stem="figB1_clinical_loss_vs_offered_load"):
    """Expected high-risk clinical loss vs offered load, by policy.

    Load is varied by compressing or expanding the arrival window; N and the
    strategic network are held fixed, so one strategic solve serves every
    point and every policy.
    """
    rows, curves = [], {n: [] for n in names}
    natural = plan.offered_load()
    for load in loads:
        span = span_for_load(plan, load)
        scaled = rescale_arrivals(plan, span)
        actual = scaled.offered_load()
        print(f"  load {load:.2f} (span {span} d, realised {actual:.2f})", flush=True)
        runs = run_all(scaled, names, seeds=seeds, cfg=cfg)
        for name in names:
            m = mean_metrics(runs[name])
            curves[name].append((actual, m))
            rows.append({"target_load": load, "offered_load": round(actual, 4),
                         "arrival_span_days": span, "policy": name,
                         **{k: m[k] for k in REPORT_KEYS if k in m}})

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, name in enumerate(names):
        xs = [p[0] for p in curves[name]]
        ys = [p[1]["expected_lost_H"] for p in curves[name]]
        es = [p[1]["expected_lost_H_se"] for p in curves[name]]
        _series(ax, xs, ys, es, name, i, len(names))
    ax.axvline(natural, color=MUTED, ls=":", lw=1)
    ax.annotate(f"natural operating point ({natural:.2f})", (natural, ax.get_ylim()[1]),
                fontsize=8.5, color=MUTED, rotation=90, va="top",
                xytext=(-12, -6), textcoords="offset points")
    ax.set_xlabel("Offered load  (arrivals per day / sustainable starts per day)")
    ax.set_ylabel("Expected high-risk patients lost")
    ax.set_title(f"Survival-aware scheduling pays off once the system is loaded\n"
                 f"N = {plan.n}, {len(seeds)} yield seeds, common random numbers",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=.5)
    ax.set_axisbelow(True)
    ax.margins(x=.12)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save_csv(os.path.join(OUTDIR, f"expB_offered_load_N{plan.n}.csv"), rows)
    return _save(fig, stem), rows


# ---------------------------------------------------------------------------
# Exp C -- value of life
# ---------------------------------------------------------------------------
# The sweep starts at the tie-break value, not 0: at alpha = 0 the strategic
# schedule is degenerate, so the frozen plan -- and every operational result
# built on it -- would be an arbitrary pick from a large tie set.
VALUE_OF_LIFE = [cd.ALPHA_TIEBREAK, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6]
OPERATING_ALPHA = st.ALPHA_STRATEGIC


def exp_c(net, inst, seeds=SEEDS, alphas=VALUE_OF_LIFE, cfg=None,
          policy="adaptive_mpc", stem="figC1_value_of_life_frontier"):
    """Cost vs expected clinical loss as the value of a life varies.

    The ONLY experiment that re-solves the strategic model: each alpha gets its
    own network, and the simulated outcome of running ``policy`` on that
    network is overlaid on the strategic frontier.
    """
    rows = []
    for a in alphas:
        plan = st.frozen_plan(net, inst, alpha=a, time_limit=2400)
        runs = run_policy(plan, policy, seeds=seeds, cfg=cfg)
        m = mean_metrics(runs)
        rows.append({"alpha": a, "opened": "+".join(plan.opened),
                     "capacity": sum(plan.fcap[x] for x in plan.opened),
                     "facility_cost": plan.facility_cost,
                     "offered_load": round(plan.offered_load(), 4),
                     "policy": policy,
                     **{k: m[k] for k in REPORT_KEYS if k in m}})
        print(f"  alpha ${a:,.0f}: {'+'.join(plan.opened)}  "
              f"E[lost]={m['expected_lost']:.2f} (H {m['expected_lost_H']:.2f})  "
              f"cost=${m['total_cost']:,.0f}", flush=True)

    flips = [i for i in range(1, len(rows))
             if rows[i]["opened"] != rows[i - 1]["opened"]]
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))

    # (a) the frontier the simulation actually realises
    xs = [r["expected_lost"] for r in rows]
    ys = [r["total_cost"] / 1e6 for r in rows]
    ax[0].plot(xs, ys, "-", color=MUTED, lw=1, zorder=1)
    # alpha rises right-to-left along the frontier and the high-alpha points
    # bunch up, so the labels are fanned out instead of all sitting up-right
    for i, (r, x, y) in enumerate(zip(rows, xs, ys)):
        op = abs(r["alpha"] - OPERATING_ALPHA) < 1e-9
        ax[0].plot([x], [y], marker="o" if not op else "*",
                   ms=7 if not op else 16,
                   color=POLICY_COLOR["adaptive_mpc"] if op else BOUND_COLOR,
                   zorder=3)
        dx, dy = [(7, 6), (7, -13), (-34, 6), (-34, -13)][i % 4]
        ax[0].annotate(_money(r["alpha"]), (x, y), fontsize=8,
                       color=INK if op else MUTED, fontweight="bold" if op else None,
                       xytext=(dx, dy), textcoords="offset points")
    ax[0].set_xlabel("Expected patients lost  $\\Sigma_p (1 - S_p)$")
    ax[0].set_ylabel("Total realised cost [$M]")
    ax[0].set_title("Cost-lives frontier (simulated, adaptive_mpc)",
                    fontsize=10, loc="left")

    # (b) where the network design changes
    a_pos = [max(r["alpha"], 2e4) for r in rows]
    ax[1].semilogx(a_pos, [r["expected_lost_H"] for r in rows], "-o",
                   color=TIER_COLOR["H"], lw=2, label="high-risk E[lost]")
    ax[1].semilogx(a_pos, [r["expected_lost"] for r in rows], "-s",
                   color=MUTED, lw=1.4, label="all patients E[lost]")
    for i in flips:
        ax[1].axvline(a_pos[i], color=BOUND_COLOR, ls="--", lw=1.2)
        ax[1].annotate(f"design flips\n{rows[i - 1]['opened']} -> {rows[i]['opened']}",
                       (a_pos[i], max(r["expected_lost"] for r in rows)),
                       fontsize=8.5, color=BOUND_COLOR, ha="right",
                       xytext=(-6, -4), textcoords="offset points")
    ax[1].axvline(OPERATING_ALPHA, color=POLICY_COLOR["adaptive_mpc"], lw=1.2)
    ax[1].annotate("$500K operating point",
                   (OPERATING_ALPHA, ax[1].get_ylim()[0]),
                   fontsize=8.5, color=POLICY_COLOR["adaptive_mpc"], rotation=90,
                   va="bottom", xytext=(4, 4), textcoords="offset points")
    ax[1].set_xlabel("Value of a life $\\alpha$  [\\$ per life; "
                     "$\\alpha=0$ plotted at $2{\\times}10^{4}$]")
    ax[1].set_ylabel("Expected patients lost")
    ax[1].set_title("Where the strategic design changes", fontsize=10, loc="left")
    ax[1].legend(fontsize=8.5, frameon=False)
    for a in ax:
        a.grid(alpha=.5)
        a.set_axisbelow(True)
        a.margins(x=.12, y=.14)
    fig.suptitle(f"Value-of-life sensitivity, N = {inst.n} "
                 f"(strategic model re-solved at every $\\alpha$)",
                 fontsize=11.5, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_csv(os.path.join(OUTDIR, f"expC_value_of_life_N{inst.n}.csv"), rows)
    return _save(fig, stem), rows


# ---------------------------------------------------------------------------
# Exp D -- policy benchmark against the best-achievable bound
# ---------------------------------------------------------------------------
def exp_d(plan, runs=None, stem="figD1_policy_cost_vs_loss",
          stem2="figD2_gap_to_best_achievable", means=None, bound_label=None):
    """Total cost vs expected clinical loss for every policy, plus the bound.

    The left panel uses the CLINICAL LOSS of objective (1), sum_p alpha_u(p)
    (1 - S_p) -- the quantity best_achievable actually minimises, so its line
    is a genuine lower bound there.  The right panel repeats the exercise on
    the study's primary metric, expected high-risk patients lost; the
    perfect-information solve is drawn there as a reference, since it optimises
    the life-value-weighted total rather than the high-risk tier alone.
    """
    # ``means`` lets the figures be re-rendered from the saved aggregates
    # without re-solving 30 perfect-information MILPs
    means = means or {name: mean_metrics(res) for name, res in runs.items()}
    label = bound_label or _bound_label(runs or {})
    online = [n for n in pol.POLICY_NAMES if n in means and n != "best_achievable"]

    panels = [("weighted_loss",
               "Expected clinical loss  "
               "$\\Sigma_p\\,(\\alpha_{u(p)}/\\alpha_{\\mathrm{ref}})(1 - S_p)$",
               "lower bound"),
              ("expected_lost_H", "Expected high-risk patients lost",
               "reference")]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for ax, (key, xlabel, role) in zip(axes, panels):
        # survival_index and adaptive_mpc land on top of each other whenever
        # (P1)-(P6) collapses to (P8), so their labels are stacked vertically
        for i, name in enumerate(online):
            m = means[name]
            ax.errorbar([m[key]], [m["total_cost"] / 1e6],
                        xerr=[m[key + "_se"]], yerr=[m["total_cost_se"] / 1e6],
                        marker="o", ms=9, capsize=3, lw=1.4,
                        color=POLICY_COLOR[name], label=name,
                        zorder=2 + i)
            ax.annotate(name, (m[key], m["total_cost"] / 1e6), fontsize=9,
                        color=POLICY_COLOR[name],
                        xytext=(11, 6 if i % 2 == 0 else -12),
                        textcoords="offset points", va="center")
        if "best_achievable" in means:
            b = means["best_achievable"]
            ax.axvline(b[key], color=BOUND_COLOR, ls="--", lw=1.4)
            ax.annotate(f"best_achievable\n({role}, {label})",
                        (b[key], ax.get_ylim()[1]), fontsize=8.5,
                        color=BOUND_COLOR, ha="left", va="top",
                        xytext=(5, -6), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Total realised cost [$M]")
        ax.grid(alpha=.5)
        ax.set_axisbelow(True)
        ax.margins(x=.22, y=.16)
    axes[0].set_title("Clinical loss -- the objective's own metric",
                      fontsize=10, loc="left")
    axes[1].set_title("High-risk patients lost -- the primary metric",
                      fontsize=10, loc="left")
    fig.suptitle(f"Policy benchmark, N = {plan.n}: survival awareness buys "
                 f"high-risk lives for slightly less money\n"
                 f"{N_REP} yield seeds, common random numbers, frozen "
                 f"{'+'.join(plan.opened)} network",
                 fontsize=11, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out1 = _save(fig, stem)

    out2 = None
    if "best_achievable" in means:
        share_w = gap_closed(means, "weighted_loss")
        share_h = gap_closed(means, "expected_lost_H")
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        bars = [n for n in online if n != "fifo"]
        y = range(len(bars))
        ax.barh([i - .19 for i in y], [share_h[n] for n in bars], height=.36,
                color=TIER_COLOR["H"], label="high-risk patients lost")
        ax.barh([i + .19 for i in y], [share_w[n] for n in bars], height=.36,
                color=MUTED,
                label=r"clinical loss $\Sigma(\alpha_u/\alpha_{\mathrm{ref}})(1-S)$")
        for i, n in enumerate(bars):
            ax.annotate(f"{share_h[n]:.0%}", (share_h[n], i - .19), fontsize=8.5,
                        va="center", xytext=(4, 0), textcoords="offset points")
            ax.annotate(f"{share_w[n]:.0%}", (share_w[n], i + .19), fontsize=8.5,
                        va="center", xytext=(4, 0), textcoords="offset points")
        ax.set_yticks(list(y))
        ax.set_yticklabels(bars)
        ax.invert_yaxis()
        ax.set_xlabel("Share of the fifo -> best_achievable gap closed")
        ax.axvline(1.0, color=BOUND_COLOR, ls="--", lw=1.2)
        ax.annotate("best_achievable", xy=(1.0, 1.0),
                    xycoords=("data", "axes fraction"), fontsize=8.5,
                    color=BOUND_COLOR, ha="right", va="bottom",
                    xytext=(0, 4), textcoords="offset points")
        ax.grid(axis="x", alpha=.5)
        ax.set_axisbelow(True)
        ax.margins(x=.10)
        ax.set_title(f"How much of the achievable gain each policy captures\n"
                     f"N = {plan.n}, {N_REP} yield seeds", fontsize=10.5, loc="left")
        fig.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        out2 = _save(fig, stem2)
    return out1, out2, means


# ---------------------------------------------------------------------------
# Exp E -- value of adapting under manufacturing failure
# ---------------------------------------------------------------------------
FAIL_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def exp_e(plan, names=("fifo", "static_survival", "adaptive_mpc"), seeds=SEEDS,
          rates=FAIL_RATES, stem="figE1_expected_loss_vs_failure_rate"):
    """Expected patients lost vs a common manufacturing failure rate.

    Every facility is overridden to the same (1 - p) so the x-axis is clean;
    the realistic per-facility baseline is marked, weighted by the capacity of
    the opened network.
    """
    rows, curves = [], {n: [] for n in names}
    for q in rates:
        cfg = sim.SimConfig(fail_rate=q)
        print(f"  failure rate {q:.2f}", flush=True)
        runs = run_all(plan, names, seeds=seeds, cfg=cfg)
        for name in names:
            m = mean_metrics(runs[name])
            curves[name].append(m)
            rows.append({"fail_rate": q, "policy": name,
                         **{k: m[k] for k in REPORT_KEYS if k in m}})

    base = sum(plan.fcap[m] * (1 - st.YIELD_P[m]) for m in plan.opened) \
        / sum(plan.fcap[m] for m in plan.opened)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for panel, key, lab in ((0, "expected_lost_H", "Expected high-risk patients lost"),
                            (1, "expected_lost", "Expected patients lost, all tiers")):
        for i, name in enumerate(names):
            ys = [m[key] for m in curves[name]]
            es = [m[key + "_se"] for m in curves[name]]
            _series(ax[panel], rates, ys, es, name, i, len(names), label=False)
        ax[panel].axvline(base, color=MUTED, ls=":", lw=1)
        ax[panel].annotate(f"per-facility baseline ({base:.3f})",
                           (base, ax[panel].get_ylim()[1]), fontsize=8.5,
                           color=MUTED, rotation=90, va="top",
                           xytext=(-12, -6), textcoords="offset points")
        ax[panel].set_xlabel("Manufacturing failure rate  $1 - p_m$ (all facilities)")
        ax[panel].set_ylabel(lab)
        ax[panel].grid(alpha=.5)
        ax[panel].set_axisbelow(True)
        ax[panel].margins(x=.08)
        ax[panel].legend(fontsize=8.5, frameon=False, loc="upper left")
    ax[0].set_title("Adapting matters more as batches fail more often",
                    fontsize=10, loc="left")
    ax[1].set_title("Total expected loss", fontsize=10, loc="left")
    fig.suptitle(f"Value of adaptive prioritisation under manufacturing failure, "
                 f"N = {plan.n}, {len(seeds)} yield seeds",
                 fontsize=11.5, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_csv(os.path.join(OUTDIR, f"expE_failure_rate_N{plan.n}.csv"), rows)
    return _save(fig, stem), rows


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
REPORT_KEYS = [
    # the only metrics the doc reports
    "expected_lost_H", "expected_lost_H_se", "expected_lost", "expected_lost_se",
    # the clinical-loss term of objective (1) -- what best_achievable bounds
    "weighted_loss", "weighted_loss_se",
    "total_cost", "total_cost_se", "cost_per_therapy", "cost_per_therapy_se",
    "mean_hold_H", "mean_hold_M", "mean_hold_L",
    # provenance / diagnostics
    "treated", "lost", "failures", "recollections", "spillover",
    "spillover_share", "lost_gate_recollection", "lost_k_remake",
    "lost_backstop", "idle_slot_days", "n_rep",
]


def _pcolor(name):
    return POLICY_COLOR.get(name, BOUND_COLOR)


def _pstyle(name):
    return "--" if name == "best_achievable" else "-"


_MARKER = {"fifo": "o", "survival_index": "s", "static_survival": "^",
           "adaptive_mpc": "D", "best_achievable": "x"}


def _series(ax, xs, ys, es, name, i, n, label=True):
    """One policy curve, drawn so that coinciding policies stay readable.

    survival_index and adaptive_mpc often land on identical values -- (P1)-(P6)
    collapses to (P8) whenever a single slot frees at a time -- so earlier
    series get a wider, paler stroke that shows as a halo under later ones, the
    markers differ, and the direct labels are staggered.
    """
    ax.errorbar(xs, ys, yerr=es, marker=_MARKER.get(name, "o"), ms=6,
                lw=2 + 1.4 * (n - 1 - i), alpha=1.0 if i == n - 1 else 0.72,
                capsize=3, color=_pcolor(name), ls=_pstyle(name), label=name,
                zorder=2 + i)
    if label:
        ax.annotate(name, (xs[-1], ys[-1]), fontsize=8.5, color=_pcolor(name),
                    xytext=(8, (i - (n - 1) / 2) * 11), textcoords="offset points",
                    va="center")


def _money(a):
    if a == 0:
        return "$0"
    return f"${a / 1e6:g}M" if a >= 1e6 else f"${a / 1e3:g}K"


def _bound_label(runs):
    st_ = {r.metrics.get("bound_status") for r in runs.get("best_achievable", [])}
    return "perfect information" if st_ == {"optimal"} else "/".join(sorted(map(str, st_)))


def _save(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, stem + ".png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  [fig] {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="A1",
                    choices=["A1", "0", "A", "B", "C", "D", "E", "all"])
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--bound", default="auto",
                    choices=["auto", "exact", "proxy", "off"],
                    help="best_achievable: exact perfect-information solve, "
                         "labelled failures-revealed proxy, or skip")
    ap.add_argument("--bound-time-limit", type=int, default=900)
    ap.add_argument("--dat", default="Data200_profileA.dat")
    ap.add_argument("--render-only", action="store_true",
                    help="Exp D: redraw the figures from the cached aggregates "
                         "instead of re-running the replications")
    args = ap.parse_args(argv)

    seeds = list(range(args.n_rep))
    suffix = f"_N{args.scale}"

    if args.exp == "0":                     # calibration only -- no simulation
        fig0_tier_survival()
        return 0

    net, inst, plan = st.load_scale(args.scale,
                                    dat=os.path.join(ROOT, args.dat))
    print(f"frozen strategic plan: N={plan.n} alpha=${plan.alpha:,.0f} "
          f"opened={'+'.join(plan.opened)} "
          f"capacity={sum(plan.fcap[m] for m in plan.opened)} slots "
          f"offered load={plan.offered_load():.2f}  [{plan.source}]", flush=True)
    provenance = {"n": plan.n, "alpha": plan.alpha, "opened": plan.opened,
                  "source": plan.source, "offered_load": plan.offered_load(),
                  "seeds": seeds, "n_rep": len(seeds),
                  "solver_note": plan.solve}

    if args.exp == "A1":
        _, runs = fig_a1_gantt(plan, seed=args.seed)
        rows = {name: r.metrics for name, r in runs.items()}
        save_json(os.path.join(OUTDIR, f"figA1_run_N{plan.n}_seed{args.seed}.json"),
                  {"plan": provenance, "seed": args.seed, "metrics": rows})
        for name, m in rows.items():
            print(f"  {name:16s} E[lost]={m['expected_lost']:.2f} "
                  f"(H {m['expected_lost_H']:.2f})  lost={m['lost']}  "
                  f"failures={m['failures']}  "
                  f"hold H/M/L={m['mean_hold_H']:.1f}/{m['mean_hold_M']:.1f}/"
                  f"{m['mean_hold_L']:.1f}  cost=${m['total_cost']:,.0f}  "
                  f"spill={m['spillover']}")
        return 0

    if args.exp in ("A", "all"):
        print("\n=== Exp A -- emergent prioritisation ===", flush=True)
        fig_a1_gantt(plan, seed=args.seed,
                     stem=f"figA1_schedule_gantt_fifo_vs_mpc{suffix}"
                     if args.scale != 50 else
                     "figA1_schedule_gantt_fifo_vs_mpc")
        runs = run_all(plan, ["fifo", "adaptive_mpc"], seeds=seeds)
        fig_a2_holdtime(runs, plan, stem=f"figA2_holdtime_by_tier{suffix}")
        save_csv(os.path.join(OUTDIR, f"expA_holds{suffix}.csv"),
                 [{"policy": n, **{k: v for k, v in mean_metrics(r).items()
                                   if k in REPORT_KEYS
                                   or k.startswith("mean_hold")}}
                  for n, r in runs.items()])

    if args.exp in ("B", "all"):
        print("\n=== Exp B -- offered load ===", flush=True)
        exp_b(plan, ["fifo", "survival_index", "static_survival", "adaptive_mpc"],
              seeds=seeds, stem=f"figB1_clinical_loss_vs_offered_load{suffix}")

    if args.exp in ("C", "all"):
        print("\n=== Exp C -- value of life ===", flush=True)
        exp_c(net, inst, seeds=seeds, stem=f"figC1_value_of_life_frontier{suffix}")

    if args.exp in ("D", "all"):
        print("\n=== Exp D -- policy benchmark ===", flush=True)
        cache = os.path.join(OUTDIR, f"expD_aggregates{suffix}.json")
        if args.render_only:
            blob = json.load(open(cache))
            exp_d(plan, means=blob["means"], bound_label=blob["bound_label"],
                  stem=f"figD1_policy_cost_vs_loss{suffix}",
                  stem2=f"figD2_gap_to_best_achievable{suffix}")
            return 0
        names = ["fifo", "survival_index", "static_survival", "adaptive_mpc"]
        runs = run_all(plan, names, seeds=seeds)
        if args.bound != "off":
            runs["best_achievable"] = run_policy(
                plan, "best_achievable", seeds=seeds,
                time_limit=args.bound_time_limit,
                proxy_on_failure=args.bound != "exact")
            b = mean_metrics(runs["best_achievable"])
            print(f"    best_achievable  E[lost]={b['expected_lost']:.3f} "
                  f"(H {b['expected_lost_H']:.3f})  "
                  f"[{_bound_label(runs)}]", flush=True)
        _, _, means = exp_d(plan, runs,
                            stem=f"figD1_policy_cost_vs_loss{suffix}",
                            stem2=f"figD2_gap_to_best_achievable{suffix}")
        # cache the aggregates so the figures can be re-rendered later without
        # re-solving 30 perfect-information MILPs
        save_json(os.path.join(OUTDIR, f"expD_aggregates{suffix}.json"),
                  {"means": means, "bound_label": _bound_label(runs),
                   "plan": provenance})
        share_h = (gap_closed(means, "expected_lost_H")
                   if "best_achievable" in means else {})
        share_w = (gap_closed(means, "weighted_loss")
                   if "best_achievable" in means else {})
        save_csv(os.path.join(OUTDIR, f"expD_policies{suffix}.csv"),
                 [{"policy": n, "gap_closed_high_risk": share_h.get(n),
                   "gap_closed_clinical_loss": share_w.get(n),
                   **{k: m[k] for k in REPORT_KEYS if k in m}}
                  for n, m in means.items()])
        save_json(os.path.join(OUTDIR, f"expD_provenance{suffix}.json"), provenance)

    if args.exp in ("E", "all"):
        print("\n=== Exp E -- manufacturing failure rate ===", flush=True)
        exp_e(plan, seeds=seeds,
              stem=f"figE1_expected_loss_vs_failure_rate{suffix}")

    if args.exp == "all":
        fig0_tier_survival()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
