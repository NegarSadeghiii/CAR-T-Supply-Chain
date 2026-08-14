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

Usage
    python run_experiments.py --exp A1 --scale 50      # the Gantt sanity check
    python run_experiments.py --exp all --scale 200
"""

from __future__ import annotations

import argparse
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
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="A1", choices=["A1"])
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dat", default="Data200_profileA.dat")
    args = ap.parse_args(argv)

    net, inst, plan = st.load_scale(args.scale,
                                    dat=os.path.join(ROOT, args.dat))
    print(f"frozen strategic plan: N={plan.n} alpha=${plan.alpha:,.0f} "
          f"opened={'+'.join(plan.opened)} "
          f"capacity={sum(plan.fcap[m] for m in plan.opened)} slots "
          f"offered load={plan.offered_load():.2f}  [{plan.source}]")

    if args.exp == "A1":
        out, runs = fig_a1_gantt(plan, seed=args.seed)
        rows = {name: r.metrics for name, r in runs.items()}
        save_json(os.path.join(OUTDIR, f"figA1_run_N{plan.n}_seed{args.seed}.json"),
                  {"plan": {"n": plan.n, "alpha": plan.alpha,
                            "opened": plan.opened, "source": plan.source,
                            "offered_load": plan.offered_load()},
                   "seed": args.seed, "metrics": rows})
        for name, m in rows.items():
            print(f"  {name:16s} E[lost]={m['expected_lost']:.2f} "
                  f"(H {m['expected_lost_H']:.2f})  lost={m['lost']}  "
                  f"failures={m['failures']}  "
                  f"hold H/M/L={m['mean_hold_H']:.1f}/{m['mean_hold_M']:.1f}/"
                  f"{m['mean_hold_L']:.1f}  cost=${m['total_cost']:,.0f}  "
                  f"spill={m['spillover']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
