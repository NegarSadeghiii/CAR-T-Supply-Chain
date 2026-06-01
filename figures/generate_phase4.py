"""
figures/generate_phase4.py — Phase 4 figures: OOS CDF and probability-of-target.

Figure 09 — three-panel CDF of cost, tier-H cancellation, and service level
            evaluated on 2,000 held-out scenarios (seed 4242).
Figure 10 — grouped bar chart of probability-of-target with 90% bootstrap CIs.

Run: python figures/generate_phase4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import (
    COLORS, setup_style,
    triple_column, single_column,
    save_figure,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

setup_style()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open(_REPO / "results" / "out_of_sample_results.json") as f:
    OOS = json.load(f)

PLAN_KEYS   = ["naive_deterministic", "expected_cost_deterministic", "stochastic_plan"]
PLAN_LABELS = ["Deterministic plan", "Expected-cost det.", "Stochastic plan"]
PLAN_COLORS = [COLORS["naive_det"], COLORS["expected_cost_det"], COLORS["sp"]]
PLAN_STYLES = ["-", "--", "-"]        # solid for outer plans, dashed for middle

BUDGET      = OOS["budget_threshold_M_USD"]
TH_TH       = OOS["tier_H_threshold_pct"]
SV_TH       = OOS["service_level_threshold_pct"]


# ---------------------------------------------------------------------------
# Figure 09 — Three-panel CDF
# ---------------------------------------------------------------------------

def _cdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_x, cumulative_probability) for empirical CDF."""
    s = np.sort(arr)
    p = np.arange(1, len(s) + 1) / len(s)
    return s, p


def fig09_oos_cdf() -> None:
    """
    Three-panel empirical CDF: (a) total cost, (b) tier-H cancellation rate,
    (c) overall service level.  All evaluated on 2,000 held-out scenarios.
    SP CDF lies left in (a) and (b), right in (c) — confirming dominance.
    """
    fig = triple_column()
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(wspace=0.30, bottom=0.22)

    # Shared line-width
    lw = 1.6

    # ---- Panel (a): Cost CDF ----
    ax = axes[0]
    for key, label, color, ls in zip(PLAN_KEYS, PLAN_LABELS, PLAN_COLORS, PLAN_STYLES):
        data = np.array(OOS["plans"][key]["per_scenario_total_cost"])
        xs, ps = _cdf(data)
        ax.plot(xs, ps, color=color, linewidth=lw, linestyle=ls, label=label)

    ax.axvline(BUDGET, color=COLORS["target_line"], linestyle="--", linewidth=1.0)
    ax.text(BUDGET + 0.2, 0.04, f"Budget\n{BUDGET:.2f} M",
            fontsize=7.5, color="#333333", va="bottom", ha="left")

    # Inline P(cost ≤ budget) annotation
    probs = [OOS["probability_of_target"][k]["P_cost_below_budget"]["value"]
             for k in PLAN_KEYS]
    ann_txt = (f"Det.: {probs[0]:.2f}\n"
               f"Exp-cost: {probs[1]:.2f}\n"
               f"SP: {probs[2]:.2f}")
    ax.text(0.97, 0.08, ann_txt, transform=ax.transAxes,
            fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))

    ax.set_xlabel("Total realized cost (M USD)", fontsize=9)
    ax.set_ylabel("Cumulative probability", fontsize=9)
    ax.set_title("(a) Cost distribution", fontsize=10)
    ax.set_ylim(0, 1.05)

    # ---- Panel (b): Tier-H cancellation CDF ----
    ax = axes[1]
    for key, label, color, ls in zip(PLAN_KEYS, PLAN_LABELS, PLAN_COLORS, PLAN_STYLES):
        data = np.array(OOS["plans"][key]["per_scenario_tier_H_cancel_rate"])
        xs, ps = _cdf(data)
        ax.plot(xs, ps, color=color, linewidth=lw, linestyle=ls, label=label)

    ax.axvline(TH_TH, color=COLORS["target_line"], linestyle="--", linewidth=1.0)
    ax.text(TH_TH + 0.3, 0.04, f"5% clinical\nthreshold",
            fontsize=7.5, color="#333333", va="bottom", ha="left")

    probs_th = [OOS["probability_of_target"][k]["P_tier_H_below_5pct"]["value"]
                for k in PLAN_KEYS]
    ann_txt_th = (f"Det.: {probs_th[0]:.2f}\n"
                  f"Exp-cost: {probs_th[1]:.2f}\n"
                  f"SP: {probs_th[2]:.2f}")
    ax.text(0.97, 0.08, ann_txt_th, transform=ax.transAxes,
            fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))

    ax.set_xlabel("Tier-H cancellation rate (%)", fontsize=9)
    ax.set_ylabel("Cumulative probability", fontsize=9)
    ax.set_title("(b) Tier-H cancellation rate", fontsize=10)
    ax.set_ylim(0, 1.05)

    # ---- Panel (c): Service level CDF ----
    ax = axes[2]
    for key, label, color, ls in zip(PLAN_KEYS, PLAN_LABELS, PLAN_COLORS, PLAN_STYLES):
        data = np.array(OOS["plans"][key]["per_scenario_service_level"])
        xs, ps = _cdf(data)
        ax.plot(xs, ps, color=color, linewidth=lw, linestyle=ls, label=label)

    ax.axvline(SV_TH, color=COLORS["target_line"], linestyle="--", linewidth=1.0)
    ax.text(SV_TH - 0.3, 0.97, f"95% service\ntarget",
            fontsize=7.5, color="#333333", va="top", ha="right")

    probs_sv = [OOS["probability_of_target"][k]["P_service_above_95pct"]["value"]
                for k in PLAN_KEYS]
    ann_txt_sv = (f"Det.: {probs_sv[0]:.2f}\n"
                  f"Exp-cost: {probs_sv[1]:.2f}\n"
                  f"SP: {probs_sv[2]:.2f}")
    ax.text(0.03, 0.92, ann_txt_sv, transform=ax.transAxes,
            fontsize=7, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))

    ax.set_xlabel("Service level (%)", fontsize=9)
    ax.set_ylabel("Cumulative probability", fontsize=9)
    ax.set_title("(c) Service level", fontsize=10)
    ax.set_ylim(0, 1.05)

    # Shared legend below all panels
    handles = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(PLAN_COLORS, PLAN_LABELS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.0))

    save_figure(fig, "figure09_oos_cdf")

    print("  Figure 09 panel probabilities:")
    for lbl, pc, pt, ps in zip(PLAN_LABELS, probs, probs_th, probs_sv):
        print(f"    {lbl:25s}  P(cost≤budget)={pc:.3f}  "
              f"P(TH≤5%)={pt:.3f}  P(SvcLv≥95%)={ps:.3f}")


# ---------------------------------------------------------------------------
# Figure 10 — Probability-of-target bar chart with bootstrap CIs
# ---------------------------------------------------------------------------

def fig10_prob_of_target() -> None:
    """
    Grouped bar chart: three plans × three targets.
    Bar colors: gray=cost, red-tinted=tier-H, green-tinted=service level.
    Error bars show 90% bootstrap CI.
    """
    # Target palette
    TARGET_LABELS = [
        f"P(cost ≤ {BUDGET:.2f} M)",
        "P(tier-H rate ≤ 5%)",
        "P(service level ≥ 95%)",
    ]
    TARGET_COLORS = ["#888888", "#C0504D", "#4A8C4A"]   # gray, red, green
    TARGET_KEYS   = [
        "P_cost_below_budget",
        "P_tier_H_below_5pct",
        "P_service_above_95pct",
    ]

    n_plans   = len(PLAN_KEYS)
    n_targets = len(TARGET_KEYS)
    group_w   = 0.75
    bar_w     = group_w / n_targets
    x_centers = np.arange(n_plans)

    fig = single_column()
    ax  = fig.add_subplot(111)

    for ti, (tk, tl, tc) in enumerate(zip(TARGET_KEYS, TARGET_LABELS, TARGET_COLORS)):
        x_pos  = x_centers + (ti - 1) * bar_w   # center, left, right
        vals   = [OOS["probability_of_target"][pk][tk]["value"]   for pk in PLAN_KEYS]
        ci_lo  = [OOS["probability_of_target"][pk][tk]["ci_low"]  for pk in PLAN_KEYS]
        ci_hi  = [OOS["probability_of_target"][pk][tk]["ci_high"] for pk in PLAN_KEYS]

        err_lo = [v - lo for v, lo in zip(vals, ci_lo)]
        err_hi = [hi - v for v, hi in zip(vals, ci_hi)]

        bars = ax.bar(x_pos, vals, bar_w * 0.92,
                      color=tc, alpha=0.85, label=tl, zorder=3)

        ax.errorbar(x_pos, vals,
                    yerr=[err_lo, err_hi],
                    fmt="none", color="black", linewidth=1.0,
                    capsize=4, capthick=1.0, zorder=5)

        # Value labels above bars
        for xi, v in zip(x_pos, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(PLAN_LABELS, fontsize=9)
    ax.set_ylabel("Probability of meeting target", fontsize=9)
    ax.set_title("Out-of-sample probability of meeting operational targets\n"
                 "(90% bootstrap CI, N=2000 held-out scenarios)",
                 fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="x", length=0)

    ax.legend(title="Target", fontsize=7.5, title_fontsize=8,
              loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()
    save_figure(fig, "figure10_probability_of_target")

    print("  Figure 10 target probabilities:")
    for pk, pl in zip(PLAN_KEYS, PLAN_LABELS):
        row = OOS["probability_of_target"][pk]
        print(f"    {pl:25s}  "
              + "  ".join(
                  f"{tk.split('_')[1]}={row[tk]['value']:.3f}"
                  f"[{row[tk]['ci_low']:.3f},{row[tk]['ci_high']:.3f}]"
                  for tk in TARGET_KEYS
              ))


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme() -> None:
    readme = _HERE / "README.md"
    existing = readme.read_text()

    if "figure09" in existing and "figure10" in existing:
        print("  README already contains figure09/10 entries — skipping.")
        return

    fig09_entry = (
        "| `figure09_oos_cdf.png` | "
        "Out-of-sample distribution of three operational outcomes — "
        "total realized cost (a), tier-H cancellation rate (b), and overall "
        "service level (c) — evaluated on 2,000 held-out scenarios independent "
        "of the optimization scenario set (seed 4242 vs. seed 0). "
        "The stochastic plan's CDF lies to the left in panels (a) and (b) and "
        "to the right in panel (c), indicating stochastic dominance over both "
        "deterministic baselines on all three operational outcomes. "
        "Vertical reference lines mark the budget threshold (a), the clinically "
        "defensible 5% cancellation rate (b), and the 95% service-level target (c). "
        f"| `results/out_of_sample_results.json` |"
    )
    fig10_entry = (
        "| `figure10_probability_of_target.png` | "
        "Probability of meeting each operational target — cost below budget "
        f"threshold ({BUDGET:.2f} M USD), tier-H cancellation rate below 5%, "
        "overall service level above 95% — under 2,000 held-out scenarios with "
        "90% bootstrap confidence intervals (500 bootstrap resamples, seed 100). "
        "The stochastic plan dominates both deterministic baselines on all three "
        "targets, with the largest margin on the tier-H cancellation target where "
        "its protection of vulnerable patients is structurally absent from the "
        "deterministic formulations. "
        f"| `results/out_of_sample_results.json` |"
    )

    updated = existing.rstrip("\n") + "\n" + fig09_entry + "\n" + fig10_entry + "\n"
    readme.write_text(updated)
    print(f"  Updated: {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Phase 4 figure generation ===\n")

    print("[1/2] Figure 09 — OOS CDF (three panels)")
    fig09_oos_cdf()

    print()
    print("[2/2] Figure 10 — Probability-of-target bar chart")
    fig10_prob_of_target()

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for name in ["figure09_oos_cdf", "figure10_probability_of_target"]:
        png = _HERE / "png" / f"{name}.png"
        pdf = _HERE / "pdf" / f"{name}.pdf"
        status = "OK" if png.exists() and pdf.exists() else "MISSING"
        print(f"  [{status}] {name}")
        if status != "OK":
            ok = False

    if ok:
        print("Phase 4 figures generated successfully.")
    else:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
