"""
figures/generate_phase2.py — Phase 2 figure: cost composition across three plans.

Generates Figure 07 — stacked bar chart showing Stage-1 and Stage-2 cost
breakdown for three planning strategies.

Run: python figures/generate_phase2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import COLORS, setup_style, single_column, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

setup_style()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open(_REPO / "case_study_results.json") as f:
    CS = json.load(f)

with open(_REPO / "results" / "expected_cost_baseline_results.json") as f:
    ECD = json.load(f)


# ---------------------------------------------------------------------------
# Figure 07 — Cost composition stacked bar chart
# ---------------------------------------------------------------------------

def fig07_cost_composition() -> None:
    """
    Stacked bar chart: Stage-1 (committed) + Stage-2 (expected recourse)
    for three planning strategies.

    Data sources:
      Deterministic plan:
        Stage-1 = case_study_results.json → ev.ev_cost  (14.50 M)
        Stage-2 = case_study_results.json → eev.stage2_cost  (5.98 M)
        Total   = case_study_results.json → eev.eev_cost  (20.48 M)

      Expected-cost deterministic:
        Stage-1 = results/expected_cost_baseline_results.json → stage1_cost
        Stage-2 = results/expected_cost_baseline_results.json → expected_stage2_cost
        Total   = results/expected_cost_baseline_results.json → total_eev_smart

      Stochastic plan:
        Stage-1 = case_study_results.json → rp.stage1_cost  (15.50 M)
        Stage-2 = case_study_results.json → rp.stage2_cost  (2.37 M)
        Total   = case_study_results.json → rp.rp_cost  (17.87 M)
    """
    plans = {
        "Deterministic\nplan": {
            "s1":    CS["ev"]["ev_cost"],
            "s2":    CS["eev"]["stage2_cost"],
            "total": CS["eev"]["eev_cost"],
        },
        "Expected-parameter\ndeterministic": {
            "s1":    ECD["stage1_cost"],
            "s2":    ECD["expected_stage2_cost"],
            "total": ECD["total_eev_smart"],
        },
        "Stochastic\nplan": {
            "s1":    CS["rp"]["stage1_cost"],
            "s2":    CS["rp"]["stage2_cost"],
            "total": CS["rp"]["rp_cost"],
        },
    }

    labels = list(plans.keys())
    s1_vals = [plans[l]["s1"]    for l in labels]
    s2_vals = [plans[l]["s2"]    for l in labels]
    totals  = [plans[l]["total"] for l in labels]

    x     = range(len(labels))
    width = 0.52

    fig = single_column()
    ax  = fig.add_subplot(111)

    # Bottom segment — Stage-1 (light blue, committed cost)
    bars_s1 = ax.bar(x, s1_vals, width,
                     color=COLORS["stage1"], zorder=3,
                     label="Stage-1 (committed)")

    # Top segment — Stage-2 (dark blue, expected recourse)
    bars_s2 = ax.bar(x, s2_vals, width, bottom=s1_vals,
                     color=COLORS["stage2"], zorder=3,
                     label="Expected Stage-2 (recourse)")

    # --- Annotations ---
    # Total cost label above each bar
    for xi, total in enumerate(totals):
        ax.text(xi, total + 0.15, f"{total:.2f} M",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#111111")

    # Stage-1 value inside bottom segment
    MIN_LABEL_HEIGHT = 1.0   # only annotate if segment is ≥ 1.0 M tall
    for xi, s1 in enumerate(s1_vals):
        if s1 >= MIN_LABEL_HEIGHT:
            ax.text(xi, s1 / 2, f"{s1:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

    # Stage-2 value inside top segment
    for xi, (s1, s2) in enumerate(zip(s1_vals, s2_vals)):
        if s2 >= MIN_LABEL_HEIGHT:
            ax.text(xi, s1 + s2 / 2, f"{s2:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

    # --- Axes ---
    max_total = max(totals)
    ax.set_ylim(0, max_total * 1.18)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total expected cost (M USD)")
    ax.set_title("Cost composition: committed vs. recourse\nacross three planning strategies",
                 fontsize=11)
    ax.tick_params(axis="x", length=0)

    # Legend upper right
    leg_patches = [
        mpatches.Patch(color=COLORS["stage1"], label="Stage-1 (committed)"),
        mpatches.Patch(color=COLORS["stage2"], label="Expected Stage-2 (recourse)"),
    ]
    ax.legend(handles=leg_patches, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()
    save_figure(fig, "figure07_cost_composition")

    # Report headline numbers
    print("  Figure 07 values:")
    for l, p in plans.items():
        print(f"    {l.replace(chr(10), ' '):30s}: "
              f"Stage-1={p['s1']:.2f}  Stage-2={p['s2']:.2f}  "
              f"Total={p['total']:.2f}")


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme() -> None:
    readme = _HERE / "README.md"
    existing = readme.read_text()

    fig07_entry = (
        "| `figure07_cost_composition.png` | "
        "Total expected cost decomposed into Stage-1 (committed) and Stage-2 (recourse) "
        "components across three planning strategies. "
        "The deterministic plan pays a low committed cost but absorbs substantial recourse "
        "cost when its decisions are deployed under realistic yield uncertainty. "
        "The expected-cost-deterministic baseline anticipates expected failures by adjusting "
        "per-batch costs, shifting more cost into Stage-1 and reducing expected recourse. "
        "The stochastic plan minimizes the total by trading a moderate increase in Stage-1 "
        "commitment for a substantially lower expected recourse cost, capturing structural "
        "value (scenario-specific recourse decisions, capacity hedging, and eligibility-tier "
        "coupling) that no deterministic formulation can represent. "
        "| `case_study_results.json`, `results/expected_cost_baseline_results.json` |"
    )

    if "figure07" not in existing:
        # Insert before the trailing newline / end of table
        updated = existing.rstrip("\n") + "\n" + fig07_entry + "\n"
        readme.write_text(updated)
        print(f"  Updated: {readme}")
    else:
        print(f"  README already contains figure07 entry — skipped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Phase 2 figure generation ===")
    print()

    print("[1/1] Figure 07 — Cost composition")
    fig07_cost_composition()

    print()
    update_readme()

    print()
    png = _HERE / "png" / "figure07_cost_composition.png"
    pdf = _HERE / "pdf" / "figure07_cost_composition.pdf"
    status = "OK" if (png.exists() and pdf.exists()) else "MISSING"
    print(f"  [{status}] figure07_cost_composition")
    if status == "OK":
        print("Figure 07 generated successfully.")
    else:
        print("WARNING: output file missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
