"""
Three "value" figures for the survival extension, built entirely from the
Phase-3 CSVs already on disk.  Nothing is re-solved and no model is imported
for solving -- this script is read-only over ``results/``.

    value_summary_N{n}.png   Figure 1 (who bears the wait)
                             + Figure 2 (same cost, fewer high-risk losses)
    schedule_gantt_N{n}.png  Figure 3 (patient ordering, cost vs survival)

Usage
    python value_figures.py                 # N = 200, 100, 500
    python value_figures.py --scales 200
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

import cart_data as cd                                            # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGDIR = os.path.join(RESULTS, "figures")

# Scenario colours
COST_C, SURV_C = "#1b6a6a", "#5b2a86"
DESIGNS = [("cost", "Scenario 1 - cost design (cost-minimising)", COST_C),
           ("survival", "Scenario 2 - survival design (alpha > 0)", SURV_C)]

# Tier colours
TIER_C = {"H": "#c0392b", "M": "#e08e0b", "L": "#2a8f6a"}
TIER_NAME = {"H": "High risk", "M": "Medium risk", "L": "Low risk"}
TIERS = ["H", "M", "L"]

# CON1 actually configured per scale (mirrors ishipment_survival's mapping);
# kept local so this script never imports the solver stack.
CAP_BY_SCALE = {100: 2, 200: 2, 500: 3}


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def cap_for(n):
    return CAP_BY_SCALE.get(n, 2)


def patient_csv(n, design):
    """Prefer the facility-budget-tagged file for the scale's configured CON1;
    fall back to the untagged name for scales solved at a single budget."""
    tagged = os.path.join(RESULTS,
                          f"phase3_patients_N{n}_{design}_fac{cap_for(n)}.csv")
    plain = os.path.join(RESULTS, f"phase3_patients_N{n}_{design}.csv")
    return tagged if os.path.exists(tagged) else plain


def load(n):
    """tier stats, per-patient rows and total costs for scale n, at its
    configured CON1.  Returns None if anything needed is missing."""
    tier_all = read_csv(os.path.join(RESULTS, "phase3_by_tier.csv")) or []
    cap = str(cap_for(n))
    tiers = {}
    for d, _, _ in DESIGNS:
        rows = [r for r in tier_all
                if r["n_patients_total"] == str(n) and r["design"] == d
                and r.get("max_facilities", cap) == cap and r["tier"] in TIERS]
        if len(rows) != 3:
            return None
        tiers[d] = {r["tier"]: r for r in rows}

    pat = {}
    for d, _, _ in DESIGNS:
        rows = read_csv(patient_csv(n, d))
        if not rows:
            return None
        for r in rows:
            for k in ("ms_arrival", "start", "hold", "delivery", "TRT"):
                r[k] = float(r[k])
            r["survival"] = float(r["survival"])
        pat[d] = rows

    comp = read_csv(os.path.join(RESULTS, "phase3_design_comparison.csv")) or []
    costs = {}
    for d, _, _ in DESIGNS:
        m = [r for r in comp if r["n"] == str(n) and r["design"] == d
             and r.get("max_facilities", cap) == cap]
        costs[d] = float(m[0]["total_cost"]) if m else None
    network = next((r["opened"] for r in comp
                    if r["n"] == str(n) and r.get("max_facilities", cap) == cap),
                   "")
    return {"n": n, "cap": cap_for(n), "tiers": tiers, "patients": pat,
            "costs": costs, "network": network,
            "source": {d: os.path.basename(patient_csv(n, d))
                       for d, _, _ in DESIGNS}}


# --------------------------------------------------------------------------
# Figures 1 + 2
# --------------------------------------------------------------------------
def value_summary(data, path):
    n = data["n"]
    t = data["tiers"]
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.6))
    width, xs = 0.36, range(len(TIERS))

    # ---- Figure 1: mean hold by tier ------------------------------------
    a = ax[0]
    for k, (d, label, colour) in enumerate(DESIGNS):
        vals = [float(t[d][u]["mean_HOLD"]) for u in TIERS]
        bars = a.bar([x + (k - 0.5) * width for x in xs], vals, width,
                     color=colour, label=label, zorder=2,
                     edgecolor="white", linewidth=.6)
        for b_, v in zip(bars, vals):
            a.annotate(f"{v:.2f}", (b_.get_x() + b_.get_width() / 2, v),
                       textcoords="offset points", xytext=(0, 3),
                       ha="center", fontsize=8, color=colour,
                       fontweight="bold")

    cost_h = {u: float(t["cost"][u]["mean_HOLD"]) for u in TIERS}
    surv_h = {u: float(t["survival"][u]["mean_HOLD"]) for u in TIERS}
    worst = max(cost_h, key=cost_h.get)
    best_surv = min(surv_h, key=surv_h.get)

    if worst == "H":
        callout = ("Cost-optimal scheduling gives the sickest patients\n"
                   "the longest wait.")
    else:
        callout = ("Cost-optimal scheduling ignores risk: high-risk patients\n"
                   f"wait {cost_h['H']:.1f} d, low-risk only {cost_h['L']:.1f} d "
                   f"(longest of all is {TIER_NAME[worst].lower()},"
                   f" {cost_h[worst]:.1f} d).")
    callout += ("\nWeighting clinical loss reverses it: high risk drops to "
                f"{surv_h['H']:.2f} d,\nthe shortest wait of any tier."
                if best_surv == "H" else "")
    a.text(.03, .97, callout, transform=a.transAxes, va="top", ha="left",
           fontsize=8.5, color="#333333",
           bbox=dict(boxstyle="round,pad=0.45", fc="#f7f4fb", ec="#b9a8cf"))

    a.set_xticks(list(xs))
    a.set_xticklabels([f"{TIER_NAME[u]}\n(n={t['cost'][u]['n']})" for u in TIERS])
    a.set_ylabel("Mean hold  [days]")
    a.set_title(f"(a) Who bears the manufacturing wait - N = {n}",
                loc="left", fontweight="bold")
    a.grid(axis="y", alpha=.3, zorder=0)
    a.set_axisbelow(True)
    a.margins(y=.42)

    # ---- Figure 2: expected losses by tier ------------------------------
    b = ax[1]
    for k, (d, label, colour) in enumerate(DESIGNS):
        vals = [float(t[d][u]["expected_lost"]) for u in TIERS]
        bars = b.bar([x + (k - 0.5) * width for x in xs], vals, width,
                     color=colour, label=label, zorder=2,
                     edgecolor="white", linewidth=.6)
        for b_, v in zip(bars, vals):
            b.annotate(f"{v:.2f}", (b_.get_x() + b_.get_width() / 2, v),
                       textcoords="offset points", xytext=(0, 3),
                       ha="center", fontsize=8, color=colour,
                       fontweight="bold")

    saved = {u: float(t["cost"][u]["expected_lost"])
             - float(t["survival"][u]["expected_lost"]) for u in TIERS}
    c0, c1 = data["costs"]["cost"], data["costs"]["survival"]
    if c0 and c1:
        pct = 100 * (c1 - c0) / c0
        msg = (f"Total cost: ${c0/1e6:.3f}M (cost) vs ${c1/1e6:.3f}M (survival)"
               f"\n{pct:+.1f}% - same network ({data['network']}).\n"
               f"Buys back {saved['H']:.2f} expected high-risk patients"
               f" ({sum(saved.values()):.2f} overall).")
        b.text(.03, .97, msg, transform=b.transAxes, va="top", ha="left",
               fontsize=8.5, color="#333333",
               bbox=dict(boxstyle="round,pad=0.45", fc="#eef6f6",
                         ec="#8fb8b8"))

    b.set_xticks(list(xs))
    b.set_xticklabels([f"{TIER_NAME[u]}\n(\\${cd.ALPHA_TIER[u]/1e6:g}M)"
                       for u in TIERS])
    b.set_ylabel("Expected patients lost   sum (1 - S[p])")
    b.set_title("(b) Same cost, fewer high-risk losses",
                loc="left", fontweight="bold")
    b.grid(axis="y", alpha=.3, zorder=0)
    b.set_axisbelow(True)
    b.margins(y=.42)

    handles = [Patch(facecolor=c, label=lab) for _, lab, c in DESIGNS]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle(f"Value of survival-aware scheduling - N = {n} "
                 f"(CON1 <= {data['cap']}, network {data['network']})",
                 fontweight="bold", y=.99)
    fig.tight_layout(rect=(0, .055, 1, .955))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("[figure]", path)


# --------------------------------------------------------------------------
# Figure 3 - patient schedule Gantt
# --------------------------------------------------------------------------
def schedule_gantt(data, path):
    n = data["n"]
    height = max(6.5, min(18.0, 0.055 * n + 3.0))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, height), sharey=True)

    xmax = max(r["delivery"] for d in data["patients"]
               for r in data["patients"][d]) + 3

    for col, (d, label, colour) in enumerate(DESIGNS):
        ax = axes[col]
        rows = data["patients"][d]
        ordered, bounds, y = [], [], 0
        for u in TIERS:                       # H block on top -> plot last
            block = sorted([r for r in rows if r["tier"] == u],
                           key=lambda r: (r["start"], r["ms_arrival"]))
            bounds.append((u, y, y + len(block)))
            ordered += block
            y += len(block)

        ys = list(range(len(ordered)))
        bar_h = 0.85
        # waiting (hold) segment: arrival -> start
        ax.barh(ys, [r["hold"] for r in ordered], height=bar_h,
                left=[r["ms_arrival"] for r in ordered],
                color=[TIER_C[r["tier"]] for r in ordered], alpha=.22,
                hatch="////", edgecolor="white", linewidth=0, zorder=2)
        # processing + transport: start -> delivery
        ax.barh(ys, [r["delivery"] - r["start"] for r in ordered], height=bar_h,
                left=[r["start"] for r in ordered],
                color=[TIER_C[r["tier"]] for r in ordered], zorder=3,
                edgecolor="none")

        for u, lo, hi in bounds:
            if hi > lo:
                ax.axhline(hi - .5, color="#555555", lw=.9, ls="-", zorder=4)
                mh = sum(r["hold"] for r in ordered[lo:hi]) / (hi - lo)
                ax.text(xmax * .995, (lo + hi) / 2 - .5,
                        f"{TIER_NAME[u]}\nmean hold {mh:.2f} d",
                        ha="right", va="center", fontsize=8.5,
                        color=TIER_C[u], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=TIER_C[u], alpha=.85))

        ax.set_xlim(0, xmax)
        ax.set_ylim(-1, len(ordered))
        # y grows downward so the first block plotted (High risk) sits on top
        ax.invert_yaxis()
        ax.set_xlabel("Day")
        ax.set_title(f"({'ab'[col]}) {label}", loc="left", fontweight="bold",
                     color=colour)
        ax.grid(axis="x", alpha=.3, zorder=0)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel(f"Patients, grouped by risk tier "
                          f"(n = {n}, sorted by start day)")
        ax.set_yticks([])

    handles = [Patch(facecolor=TIER_C[u], label=TIER_NAME[u]) for u in TIERS]
    handles += [Patch(facecolor="#999999", alpha=.28, hatch="////",
                      label="waiting in queue (hold)"),
                Patch(facecolor="#999999", label="manufacture + QC + transport")]
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(.5, .004))

    fig.suptitle(f"Patient schedule by design - who is served first "
                 f"(N = {n}, CON1 <= {data['cap']}, network {data['network']})",
                 fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, .035, 1, .975))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[figure]", path)


# --------------------------------------------------------------------------
def crossover_table(loaded):
    hdr = (f"{'N':>5} {'CON1':>5} {'design':>9} " +
           " ".join(f"{'hold ' + u:>9}" for u in TIERS) +
           f" {'H rank':>7}  {'E[lost] H':>10}")
    print("\nCrossover - mean hold by tier (days)\n" + hdr)
    print("-" * len(hdr))
    for d_ in loaded:
        for design, _, _ in DESIGNS:
            t = d_["tiers"][design]
            holds = {u: float(t[u]["mean_HOLD"]) for u in TIERS}
            rank = sorted(TIERS, key=lambda u: -holds[u]).index("H") + 1
            print(f"{d_['n']:>5} {d_['cap']:>5} {design:>9} " +
                  " ".join(f"{holds[u]:>9.3f}" for u in TIERS) +
                  f" {rank:>4}/3  {float(t['H']['expected_lost']):>10.3f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=int, default=[200, 100, 500])
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    loaded = []
    for n in args.scales:
        data = load(n)
        if data is None:
            print(f"[skip] N = {n}: Phase-3 per-patient or tier CSVs missing")
            continue
        print(f"[N={n}] CON1<={data['cap']}  sources: "
              + ", ".join(f"{k}->{v}" for k, v in data["source"].items()))
        loaded.append(data)
        value_summary(data, os.path.join(FIGDIR, f"value_summary_N{n}.png"))
        schedule_gantt(data, os.path.join(FIGDIR, f"schedule_gantt_N{n}.png"))

    if loaded:
        crossover_table(sorted(loaded, key=lambda d: d["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
