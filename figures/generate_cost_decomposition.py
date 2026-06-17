"""
figures/generate_cost_decomposition.py — Figure 14: cost composition donuts.

Three donut charts (one per plan) showing the percentage breakdown of the
six cost components.  Each donut's hollow center displays the plan's total
mean cost (M USD).  Segment labels appear outside the ring for all
components whose share is >= 3%.

Six cost components:
  Stage 1 (committed):
    1. Facility opening  — f[m] * z[m]
    2. Capacity          — pi[m] * C[m]
    3. Primary mfg       — c[m] * sum_i x[i,m]

  Stage 2 (expected recourse, averaged over 2,000 OOS scenarios):
    4. Re-manufacture    — (rho_leuk + rho_remfg[m]) * r_remfg[i,m]
    5. Subcontract       — (rho_leuk + rho_sub[m,mp]) * r_sub[i,m,mp]
    6. Cancellation      — rho_cancel[i] * r_cancel[i]

Run: python figures/generate_cost_decomposition.py
Source: results/out_of_sample_results.json + results/out_of_sample_recourse.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))

from figure_style import COLORS, setup_style, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

setup_style()

_RESULT = _REPO / "results" / "out_of_sample_results.json"
_NPZ    = _REPO / "results" / "out_of_sample_recourse.npz"

# Instance parameters (calibrated case-study, fixed across all plans)
_F   = np.array([0.5, 2.0, 3.0, 2.0])
_PI  = np.array([0.04, 0.06, 0.09, 0.06])
_C   = np.array([0.20, 0.18, 0.15, 0.18])
_RHO_LEUK   = 0.005
_RHO_REMFG  = _C.copy()
_RHO_SUB    = np.zeros((4, 4))
for _m in range(4):
    for _mp in range(4):
        if _m != _mp:
            _RHO_SUB[_m, _mp] = _C[_mp] * 1.15

_RHO_CANCEL = np.array([6.00] * 10 + [2.00] * 25 + [0.75] * 15)

PLAN_KEYS   = ["naive_deterministic", "expected_cost_deterministic", "stochastic_plan"]
PLAN_LABELS = ["Deterministic plan", "Expected-parameter det.", "Stochastic plan"]
PLAN_LABELS_CENTER = [
    "Deterministic\nplan",
    "Expected-parameter\ndet. plan",
    "Stochastic\nplan",
]
PLAN_COLORS = [COLORS["naive_det"], COLORS["expected_cost_det"], COLORS["sp"]]

COMP_LABELS = [
    "Facility opening",
    "Capacity",
    "Primary mfg",
    "Re-manufacture",
    "Subcontract",
    "Cancellation",
]
COMP_COLORS = [
    "#404040",               # Facility opening — dark gray
    COLORS["stage1"],        # Capacity — light blue
    "#808080",               # Primary mfg — medium gray
    COLORS["remfg"],         # Re-manufacture — green
    COLORS["subcontract"],   # Subcontract — amber
    COLORS["cancel"],        # Cancellation — red
]

_LABEL_THRESHOLD = 3.0   # % — segments below this get no outside label


def _load() -> dict[str, list[float]]:
    with open(_RESULT) as f:
        d = json.load(f)

    npz = np.load(str(_NPZ))
    results = {}

    for key in PLAN_KEYS:
        pr = d["plans"][key]

        # Stage-1 decomposition
        fs  = pr["first_stage"]
        z   = np.array(fs["z"])
        C   = np.array(fs["C"])
        x   = np.array(fs["x"])

        open_cost = float((_F  * z).sum())
        cap_cost  = float((_PI * C).sum())
        mfg_cost  = float((_C  * x.sum(axis=0)).sum())

        # Stage-2 decomposition (mean over scenarios)
        pfx      = pr["_recourse_npz_key"]
        r_remfg  = npz[f"{pfx}__r_remfg"]
        r_sub    = npz[f"{pfx}__r_sub"]
        r_cancel = npz[f"{pfx}__r_cancel"]

        rho_r = _RHO_LEUK + _RHO_REMFG
        rho_s = _RHO_LEUK + _RHO_SUB

        s2_remfg  = (r_remfg  * rho_r[np.newaxis, np.newaxis, :]).sum(axis=(1, 2))
        s2_sub    = (r_sub    * rho_s[np.newaxis, np.newaxis, :, :]).sum(axis=(1, 2, 3))
        s2_cancel = (r_cancel * _RHO_CANCEL[np.newaxis, :]).sum(axis=1)

        results[key] = [
            open_cost,
            cap_cost,
            mfg_cost,
            float(s2_remfg.mean()),
            float(s2_sub.mean()),
            float(s2_cancel.mean()),
        ]

    return results


def generate_figure14() -> None:
    data = _load()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.02, right=0.98,
                        wspace=0.10)

    for ax, key, plan_center_lbl, plan_color in zip(
        axes, PLAN_KEYS, PLAN_LABELS_CENTER, PLAN_COLORS
    ):
        costs = data[key]
        total = sum(costs)
        pcts  = [c / total * 100.0 for c in costs]

        # Build per-segment labels: non-empty only for segments >= threshold
        pie_labels = [
            f"{comp}:\n{pct:.1f}%" if pct >= _LABEL_THRESHOLD else ""
            for comp, pct in zip(COMP_LABELS, pcts)
        ]

        wedges, texts = ax.pie(
            costs,
            labels=pie_labels,
            labeldistance=1.32,
            colors=COMP_COLORS,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.40, "edgecolor": "white", "linewidth": 0.8},
        )

        # Style the leader-line texts
        for text, pct in zip(texts, pcts):
            text.set_fontsize(8)
            # Align text based on x-position
            xp = text.get_position()[0]
            text.set_ha("right" if xp < -0.02 else ("left" if xp > 0.02 else "center"))

        # Center: total cost (large bold) + plan name (smaller, colored)
        ax.text(0,  0.14, f"{total:.2f} M USD",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="#111111")
        ax.text(0, -0.16, plan_center_lbl,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=plan_color)

        ax.set_aspect("equal")

    # Shared horizontal legend below all donuts
    legend_handles = [
        mpatches.Patch(facecolor=c, label=l)
        for l, c in zip(COMP_LABELS, COMP_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        title="Cost component",
        title_fontsize=9,
    )

    # Title + subtitle
    fig.suptitle(
        "Cost composition by plan (2,000 OOS scenarios)",
        fontsize=13, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.905,
        "Total cost (center) and per-component percentages (outer ring).",
        ha="center", fontsize=10, fontstyle="italic", color="#555555",
    )

    save_figure(fig, "figure14_cost_decomposition")


def generate_bar_figure() -> None:
    data = _load()

    x_labels = [
        "Deterministic\nplan",
        "Expected-parameter\ndet.",
        "Stochastic\nplan",
    ]
    x = range(len(PLAN_KEYS))
    width = 0.52

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(bottom=0.28)

    bottoms = [0.0] * len(PLAN_KEYS)
    MIN_LABEL = 0.40   # M USD — skip segments smaller than this
    for comp_idx, (comp_label, comp_color) in enumerate(zip(COMP_LABELS, COMP_COLORS)):
        vals = [data[k][comp_idx] for k in PLAN_KEYS]
        ax.bar(x, vals, width, bottom=bottoms,
               color=comp_color, zorder=3, label=comp_label)
        # Per-segment value labels
        for xi, (val, bot) in enumerate(zip(vals, bottoms)):
            if val >= MIN_LABEL:
                ax.text(
                    xi, bot + val / 2, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    path_effects=[
                        pe.Stroke(linewidth=2, foreground="black"),
                        pe.Normal(),
                    ],
                    zorder=5,
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Total cost annotation above each bar
    for xi, total in enumerate(bottoms):
        ax.text(xi, total + 0.15, f"{total:.3f} M",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#111111")

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Mean total cost (M USD)")
    ax.set_ylim(0, max(bottoms) * 1.15)
    ax.set_title(
        "Cost decomposition across planning strategies\n"
        "(Stage-1 committed + expected Stage-2 recourse, 2,000 OOS scenarios)",
        fontsize=11,
    )
    ax.tick_params(axis="x", length=0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=6, frameon=True, fontsize=8)

    save_figure(fig, "figure14_cost_decomposition_bar")


def _print_table(data: dict) -> dict[str, str]:
    """Print per-plan component tables and return the headline observation."""
    print()
    print("=== Component breakdown ===")
    for key, label in zip(PLAN_KEYS, PLAN_LABELS):
        costs = data[key]
        total = sum(costs)
        print(f"\n  {label} — total = {total:.4f} M USD")
        print(f"  {'Component':<22} {'M USD':>8}  {'% of total':>10}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*10}")
        for comp, cost in zip(COMP_LABELS, costs):
            print(f"  {comp:<22} {cost:>8.4f}  {cost/total*100:>9.1f}%")

    # Headline observation about cancellation
    cancel_naive = data["naive_deterministic"][5]
    cancel_sp    = data["stochastic_plan"][5]
    tot_naive    = sum(data["naive_deterministic"])
    tot_sp       = sum(data["stochastic_plan"])
    pct_naive    = cancel_naive / tot_naive * 100
    pct_sp       = cancel_sp   / tot_sp    * 100
    print()
    print(f"  Headline: Deterministic cancellation = {pct_naive:.1f}% of total cost; "
          f"SP = {pct_sp:.1f}% — savings concentrated in cancellation reduction.")
    return {
        "cancel_naive_pct": round(pct_naive, 1),
        "cancel_sp_pct":    round(pct_sp, 1),
    }


def update_readme() -> None:
    import re as _re
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    entry_donut = (
        "| `figure14_cost_decomposition.png` | "
        "Three donut charts decomposing mean total cost (2,000 OOS scenarios) "
        "into six components per plan: facility opening, capacity, primary manufacturing "
        "(Stage 1 committed), and re-manufacture, subcontract, cancellation "
        "(expected Stage 2 recourse). "
        "Each donut's center shows the plan total in M USD. "
        "Segment labels appear outside the ring for components >= 3%. "
        "| `results/out_of_sample_results.json` + `results/out_of_sample_recourse.npz` |\n"
    )
    entry_bar = (
        "| `figure14_cost_decomposition_bar.png` | "
        "Stacked bar chart companion to the donut chart, showing absolute M USD values "
        "for each of the six cost components across the three planning strategies. "
        "Stage-1 components (facility opening, capacity, primary mfg) appear in the "
        "bottom portion; Stage-2 recourse components (re-manufacture, subcontract, "
        "cancellation) appear in the top portion. Total cost is annotated above each bar. "
        "| `results/out_of_sample_results.json` + `results/out_of_sample_recourse.npz` |\n"
    )

    # Remove any existing entries for figure14_cost_decomposition
    cleaned = _re.sub(r"\| `figure(?:14_cost_decomposition|_cost_decomposition)[^|]+\|[^|]+\|\n?",
                      "", existing)
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry_donut + entry_bar)
    print(f"  Updated: {readme}")


def main() -> None:
    print("=== Figure 14 — cost decomposition donuts ===\n")

    data = _load()
    _print_table(data)

    print("\n[1/2] Generating figure14_cost_decomposition (donuts) …")
    generate_figure14()

    print("\n[2/2] Generating figure14_cost_decomposition_bar (stacked bar) …")
    generate_bar_figure()

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for name in ("figure14_cost_decomposition", "figure14_cost_decomposition_bar"):
        for ext in ("png", "pdf"):
            p = _HERE / ext / f"{name}.{ext}"
            if p.exists():
                print(f"  [OK]      {p.name}  ({p.stat().st_size // 1024} KB)")
            else:
                print(f"  [MISSING] {p.name}")
                ok = False

    if not ok:
        sys.exit(1)
    print("\nFigure 14 (both variants) generated successfully.")


if __name__ == "__main__":
    main()
