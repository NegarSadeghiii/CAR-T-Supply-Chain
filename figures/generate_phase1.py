"""
figures/generate_phase1.py — generate all six Phase 1 paper figures.

Reads data from:
  ../case_study_results.json
  ../results/per_tier_cancel_rates.json   (all per-tier cancellation rates)

Run from any directory:
    python figures/generate_phase1.py
or from inside figures/:
    python generate_phase1.py

Label conventions (Revision 1):
  "Deterministic plan" — EV first-stage decisions deployed under stochastic yields
  "Stochastic plan"    — full two-stage SP optimum (RP)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import (
    COLORS, setup_style, single_column, double_column, save_figure,
)

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

with open(_REPO / "results" / "per_tier_cancel_rates.json") as f:
    TIER_CANCEL = json.load(f)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Patient tier indices: TIERS = ["H"]*10 + ["M"]*25 + ["L"]*15
TIER_IDX = {"H": list(range(0, 10)),
             "M": list(range(10, 35)),
             "L": list(range(35, 50))}

EV_X = np.array(CS["ev"]["x"])   # (50, 4) first-stage assignment — deterministic plan
RP_X = np.array(CS["rp"]["x"])   # (50, 4) first-stage assignment — stochastic plan


# ---------------------------------------------------------------------------
# Figure 01 — Methodology placeholder
# ---------------------------------------------------------------------------

def fig01_placeholder() -> None:
    """White placeholder PNG for the hand-drawn methodology SVG."""
    img  = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, 397, 197], outline="#AAAAAA", width=2)
    draw.text((20, 80), "Methodology diagram — hand-drawn SVG", fill="#888888")
    png_path = _HERE / "png" / "figure01_methodology.png"
    img.save(png_path)
    print(f"  Saved: {png_path}")


# ---------------------------------------------------------------------------
# Figure 02 — Empirical OOS context
# ---------------------------------------------------------------------------

def fig02_oos_context() -> None:
    """
    Bar chart of real-world CAR-T manufacturing failure rates.
    Source: Patel et al. 2024 (JCO abstract 7044).
    Reference line: UK National CAR T Panel real-world aggregate (Dulobdas 2025).
    """
    products = ["axicabtagene\nciloleucel\n(axi-cel)",
                "tisagenlecleucel\n(tisa-cel)",
                "lisocabtagene\nmaraleucel\n(liso-cel)"]
    rates   = [4.0, 17.0, 28.0]
    uk_rate = 3.87

    fig = single_column()
    ax  = fig.add_subplot(111)

    bars = ax.bar(products, rates,
                  color=COLORS["expected_cost_det"], width=0.5, zorder=3)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(uk_rate, color=COLORS["target_line"],
               linestyle="--", linewidth=1.0, zorder=4)
    ax.text(2.48, uk_rate + 0.8,
            f"UK National CAR T Panel\nreal-world rate {uk_rate}%\n(Dulobdas 2025)",
            ha="right", va="bottom", fontsize=7.5, color=COLORS["target_line"])

    ax.set_ylabel("Manufacturing failure rate (%)")
    ax.set_ylim(0, 35)
    ax.set_title("Real-world CAR-T out-of-specification rates\n(Patel et al. 2024, JCO abstract 7044)",
                 fontsize=11)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save_figure(fig, "figure02_oos_context")


# ---------------------------------------------------------------------------
# Figure 03 — Tier-dependent eligibility profile
# ---------------------------------------------------------------------------

def fig03_eligibility_profile() -> None:
    """
    Bar chart of re-collection eligibility probabilities β per tier.
    Calibrated from Locke 2022, Bachy 2022, Lulla 2024.
    """
    tiers  = ["Tier H\n(high urgency)", "Tier M\n(medium urgency)", "Tier L\n(low urgency)"]
    betas  = [0.55, 0.78, 0.92]
    colors = [COLORS["tier_H"], COLORS["tier_M"], COLORS["tier_L"]]

    fig = single_column()
    ax  = fig.add_subplot(111)

    bars = ax.bar(tiers, betas, color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"β = {val:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate("",
                xy=(2.25, 0.65), xytext=(-0.25, 0.65),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    ax.text(1.0, 0.67,
            "Higher clinical urgency → lower re-collection\n"
            "eligibility after 14–21 day manufacturing delay",
            ha="center", va="bottom", fontsize=7.5, color="#444444", style="italic")

    ax.set_ylabel("Re-collection eligibility probability βᵤ")
    ax.set_ylim(0, 1.05)
    ax.set_title("Tier-dependent re-collection eligibility", fontsize=11)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save_figure(fig, "figure03_eligibility_profile")


# ---------------------------------------------------------------------------
# Figure 04 — Tier-stratified cancellation rate (simulated, all tiers)
# ---------------------------------------------------------------------------

def fig04_tier_cancellation() -> None:
    """
    Grouped bar chart: per-tier cancellation rate under each plan.

    All six rates (3 tiers × 2 plans) come from full simulation stored in
    results/per_tier_cancel_rates.json.  No analytical estimates used.

    Deterministic plan: EV first-stage fixed, second stage optimised over
    200 stochastic scenarios (seed=0).
    Stochastic plan: full two-stage SP optimum evaluated on the same scenarios.
    """
    det_rates = TIER_CANCEL["deterministic_plan"]  # {"H": %, "M": %, "L": %}
    sp_rates  = TIER_CANCEL["stochastic_plan"]

    tier_keys  = ["H", "M", "L"]
    tier_labels = ["Tier H", "Tier M", "Tier L"]
    det_vals   = [det_rates[t] for t in tier_keys]
    sp_vals    = [sp_rates[t]  for t in tier_keys]
    reductions = [det_rates[t] - sp_rates[t] for t in tier_keys]

    x     = np.arange(len(tier_labels))
    width = 0.35

    fig = single_column()
    ax  = fig.add_subplot(111)

    ax.bar(x - width / 2, det_vals, width,
           label="Deterministic plan",
           color=COLORS["naive_det"], zorder=3)
    ax.bar(x + width / 2, sp_vals, width,
           label="Stochastic plan",
           color=COLORS["sp"], zorder=3)

    for i, (det, sp, red) in enumerate(zip(det_vals, sp_vals, reductions)):
        ax.text(x[i], max(det, sp) + 0.12, f"−{red:.1f} pp",
                ha="center", va="bottom", fontsize=8,
                color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel("Realized cancellation rate (%)")
    ax.set_ylim(0, max(det_vals) * 1.35)
    ax.set_title("Tier-stratified cancellation rate:\ndeterministic vs. stochastic plan",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save_figure(fig, "figure04_tier_cancellation")

    print("  Figure 04 values (all simulated):")
    for t, tl in zip(tier_keys, tier_labels):
        print(f"    {tl}: Det={det_rates[t]:.4f}%  SP={sp_rates[t]:.4f}%  "
              f"reduction={det_rates[t]-sp_rates[t]:.4f} pp")


# ---------------------------------------------------------------------------
# Figure 05 — Recourse mix breakdown
# ---------------------------------------------------------------------------

def fig05_recourse_mix() -> None:
    """
    Horizontal stacked bar chart of recourse action mix.

    Data from case_study_results.json:
      Deterministic plan: eev.recourse_mix
      Stochastic plan:    rp.recourse_mix
    """
    plans = {
        "Deterministic plan": CS["eev"]["recourse_mix"],
        "Stochastic plan":    CS["rp"]["recourse_mix"],
    }
    labels      = list(plans.keys())
    remfg_vals  = [plans[l]["re-manufacture"] for l in labels]
    sub_vals    = [plans[l]["subcontract"]    for l in labels]
    cancel_vals = [plans[l]["cancel"]         for l in labels]

    fig = single_column()
    ax  = fig.add_subplot(111)

    y_pos  = [1, 0]
    height = 0.4

    def _hbar(lefts, vals, color, label):
        bars = ax.barh(y_pos, vals, height, left=lefts,
                       color=color, label=label, zorder=3)
        for bar, val, left in zip(bars, vals, lefts):
            if val > 5.0:
                cx = left + val / 2
                cy = bar.get_y() + bar.get_height() / 2
                tc = "white" if color != COLORS["subcontract"] else "#333333"
                ax.text(cx, cy, f"{val:.1f}%",
                        ha="center", va="center", fontsize=8,
                        color=tc, fontweight="bold")
        return [l + v for l, v in zip(lefts, vals)]

    lefts = [0.0, 0.0]
    lefts = _hbar(lefts, remfg_vals,  COLORS["remfg"],      "Re-manufacture in-house")
    lefts = _hbar(lefts, sub_vals,    COLORS["subcontract"], "Subcontract to partner facility")
    lefts = _hbar(lefts, cancel_vals, COLORS["cancel"],      "Cancel patient")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Share of failed batches (%)")
    ax.set_title("Recourse action composition per plan", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, fontsize=7.5, frameon=True)
    fig.tight_layout()
    save_figure(fig, "figure05_recourse_mix")

    print("  Figure 05 values:")
    for l in labels:
        m = plans[l]
        print(f"    {l}: remfg={m['re-manufacture']:.1f}%  "
              f"sub={m['subcontract']:.1f}%  cancel={m['cancel']:.1f}%")


# ---------------------------------------------------------------------------
# Figure 06 — Stage-1 facility allocation by tier (two panels)
# ---------------------------------------------------------------------------

def fig06_facility_allocation() -> None:
    """
    Two-panel stacked bar chart: patients per facility × tier.
    Left panel: Deterministic plan.  Right panel: Stochastic plan.
    Shared y-axis (sharey=True) so scales are comparable.

    Allocation verified against case study output:
      Det plan: m0 (H=6, M=23, L=11), m1 (H=4, M=2, L=4)
      SP plan:  m0 (H=0, M=3,  L=7),  m2 (H=10, M=22, L=8)
    """
    tiers       = ["H", "M", "L"]
    tier_colors = [COLORS["tier_H"], COLORS["tier_M"], COLORS["tier_L"]]
    tier_labels = ["Tier H", "Tier M", "Tier L"]
    n_fac       = 4
    fac_labels  = [f"m$_{m}$" for m in range(n_fac)]

    def _counts(x_mat: np.ndarray) -> np.ndarray:
        """Count patients per (facility, tier); returns (n_fac, 3) array."""
        counts = np.zeros((n_fac, len(tiers)), dtype=int)
        for ti, tier in enumerate(tiers):
            counts[:, ti] = x_mat[TIER_IDX[tier]].sum(axis=0)
        return counts

    ev_counts = _counts(EV_X)
    rp_counts = _counts(RP_X)

    fig = double_column()
    fig.subplots_adjust(bottom=0.22, wspace=0.08)
    axes = fig.subplots(1, 2, sharey=True)

    x     = np.arange(n_fac)
    width = 0.55

    for ax, counts, title in zip(axes,
                                  [ev_counts, rp_counts],
                                  ["Deterministic plan", "Stochastic plan"]):
        bottom = np.zeros(n_fac)
        for ti, (tier, color, t_label) in enumerate(
                zip(tiers, tier_colors, tier_labels)):
            vals = counts[:, ti].astype(float)
            ax.bar(x, vals, width, bottom=bottom,
                   color=color, zorder=3, label=t_label)
            for fi in range(n_fac):
                if vals[fi] > 0:
                    tc = "white" if color != COLORS["tier_M"] else "#333333"
                    ax.text(x[fi], bottom[fi] + vals[fi] / 2,
                            str(int(vals[fi])),
                            ha="center", va="center", fontsize=8,
                            color=tc, fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(fac_labels, fontsize=10)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Manufacturing facility", fontsize=10)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Number of patients assigned")
    axes[0].set_ylim(0, 52)
    # Remove left spine on right panel to avoid double-line between panels
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False)

    # Single shared legend below both panels
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(tier_colors, tier_labels)]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    # Annotation between panels
    fig.text(0.5, 0.13,
             "Under the stochastic plan, high-urgency-tier patients are reassigned to a\n"
             "higher-yield facility — enabled by constraint (10b) tying recourse to the\n"
             "primary assignment.",
             ha="center", va="bottom", fontsize=8, color="#333333", style="italic")

    save_figure(fig, "figure06_facility_allocation")

    print("  Figure 06 allocation:")
    for fi in range(n_fac):
        print(f"    m{fi}:  Det={ev_counts[fi].tolist()}  SP={rp_counts[fi].tolist()}")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def write_readme() -> None:
    readme = _HERE / "README.md"
    lines = [
        "# Phase 1 Figures — CAR-T Supply Chain Paper",
        "",
        "All figures generated by `generate_phase1.py` from `case_study_results.json`",
        "and `results/per_tier_cancel_rates.json`.",
        "PNG (300 DPI) and PDF (vector) outputs are in `png/` and `pdf/` respectively.",
        "",
        "## Label convention",
        "",
        "**Deterministic plan** — the optimization is solved assuming no batches fail",
        "(expected-value problem). The cost shown in figures reflects realized outcomes",
        "when this plan is deployed under realistic stochastic yields.",
        "",
        "**Stochastic plan** — optimizes with manufacturing yield modeled explicitly as",
        "a Bernoulli random variable per batch (full two-stage stochastic program).",
        "",
        "## Figure index",
        "",
        "| File | Caption | Source |",
        "|------|---------|--------|",
        "| `figure01_methodology.png` | Methodology diagram — hand-drawn SVG placeholder. | (hand-drawn) |",
        "| `figure02_oos_context.png` | Real-world out-of-specification rates for commercially approved CAR-T products span 4% to 28%, with the UK National CAR T Panel reporting an aggregate failure rate of 3.87%. The variance across products and the binary clinical consequence per patient motivates a stochastic-yield modeling framework. | Patel et al. 2024; Dulobdas 2025 |",
        "| `figure03_eligibility_profile.png` | Tier-dependent re-collection eligibility probabilities used in the model, calibrated from clinical literature on patient deterioration during CAR-T manufacturing delays (Locke 2022, Bachy 2022, Lulla 2024). Tier-H patients face the lowest probability of remaining eligible for a re-collection attempt. | case_study.py calibration |",
        "| `figure04_tier_cancellation.png` | Realized cancellation rate by patient urgency tier. The deterministic plan is obtained by solving the optimization assuming no batches fail; the cost shown reflects realized outcomes when the plan is deployed under realistic stochastic yields. The stochastic plan optimizes with manufacturing yield modeled as a Bernoulli random variable per batch. High-urgency-tier cancellation falls by more than half, with the largest absolute reduction concentrated in the most clinically vulnerable patients. | `results/per_tier_cancel_rates.json` (all rates simulated) |",
        "| `figure05_recourse_mix.png` | Composition of recourse actions selected by each plan, expressed as a percentage of all failed batches. The stochastic optimum shifts cases away from cancellation toward in-house re-manufacture, with the residual subcontracting share reflecting cross-facility hedging. | `case_study_results.json` |",
        "| `figure06_facility_allocation.png` | Stage-1 patient-to-facility allocation by urgency tier. The deterministic plan (left) concentrates patients at the two opened facilities (m0, m1). The stochastic plan (right) opens a different facility set (m0, m2) and reassigns high-urgency patients to the higher-yield site, reflecting the constraint that recourse feasibility depends on the patient's primary assignment (Equation 10b). | `case_study_results.json` |",
        "",
    ]
    readme.write_text("\n".join(lines))
    print(f"  Saved: {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Phase 1 figure generation ===")
    print()

    print("[1/6] Figure 01 — Methodology placeholder")
    fig01_placeholder()

    print("[2/6] Figure 02 — OOS context")
    fig02_oos_context()

    print("[3/6] Figure 03 — Eligibility profile")
    fig03_eligibility_profile()

    print("[4/6] Figure 04 — Tier-stratified cancellation (simulated)")
    fig04_tier_cancellation()

    print("[5/6] Figure 05 — Recourse mix")
    fig05_recourse_mix()

    print("[6/6] Figure 06 — Facility allocation (split panels)")
    fig06_facility_allocation()

    print()
    write_readme()

    print()
    print("Output verification:")
    names = [
        "figure01_methodology",
        "figure02_oos_context",
        "figure03_eligibility_profile",
        "figure04_tier_cancellation",
        "figure05_recourse_mix",
        "figure06_facility_allocation",
    ]
    all_ok = True
    for name in names:
        png    = _HERE / "png" / f"{name}.png"
        pdf    = _HERE / "pdf" / f"{name}.pdf"
        png_ok = png.exists()
        pdf_ok = pdf.exists() if name != "figure01_methodology" else True
        status = "OK" if (png_ok and pdf_ok) else "MISSING"
        if not (png_ok and pdf_ok):
            all_ok = False
        print(f"  [{status}] {name}")

    print()
    if all_ok:
        print("All figures generated successfully.")
    else:
        print("WARNING: Some files are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
