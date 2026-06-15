"""
figures/generate_phase6.py — Figure 15: β_H sensitivity sweep.

Three-panel figure showing how VSS, Tier-H cancellation rate, and
SP facility selection vary with β_H ∈ {0.40, 0.55, 0.70, 0.85}.

Source: results/sensitivity_beta_results.json

Run: python figures/generate_phase6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import COLORS, setup_style, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

setup_style()

_RESULT = _REPO / "results" / "sensitivity_beta_results.json"

C_NAIVE = COLORS["naive_det"]
C_ECD   = COLORS["expected_cost_det"]
C_SP    = COLORS["sp"]
C_REF   = "#888888"

BETA_H_CENTRAL = 0.55
FAC_LABELS = ["m₀", "m₁", "m₂", "m₃"]


# ── load data ──────────────────────────────────────────────────────────────────

def _load():
    with open(_RESULT) as f:
        d = json.load(f)
    pts = d["sweep"]

    betas      = [p["beta_H"]                              for p in pts]
    vss_naive  = [p["vss_vs_naive_pct"]                    for p in pts]
    vss_ecd    = [p["vss_vs_ecd_pct"]                      for p in pts]
    h_naive    = [p["naive"]["tier_h_pct"]                 for p in pts]
    h_ecd      = [p["expected_cost_det"]["tier_h_pct"]     for p in pts]
    h_sp       = [p["stochastic"]["tier_h_pct"]            for p in pts]

    # Facility selections: list of open-facility index sets per sweep point
    sp_opens  = [set(int(f[1]) for f in p["stochastic"]["facilities"]) for p in pts]
    ecd_opens = [set(int(f[1]) for f in p["expected_cost_det"]["facilities"]) for p in pts]

    return dict(
        betas=betas,
        vss_naive=vss_naive, vss_ecd=vss_ecd,
        h_naive=h_naive, h_ecd=h_ecd, h_sp=h_sp,
        sp_opens=sp_opens, ecd_opens=ecd_opens,
    )


# ── figure ─────────────────────────────────────────────────────────────────────

def generate_figure15() -> None:
    d = _load()
    betas = d["betas"]

    fig, axes = plt.subplots(3, 1, figsize=(7, 9),
                              gridspec_kw={"hspace": 0.52})

    # ── (a) VSS curves ────────────────────────────────────────────────────────
    ax = axes[0]

    ax.plot(betas, d["vss_naive"], "o-",
            color=C_NAIVE, lw=2, ms=7, label="vs Deterministic")
    ax.plot(betas, d["vss_ecd"], "o-",
            color=C_ECD,   lw=2, ms=7, label="vs Expected-parameter det.")

    ax.axhline(0, color="#CCCCCC", lw=0.8, linestyle=":")

    # Shade below-4% warning zone for VSS vs ECD
    ax.axhspan(0, 4, alpha=0.06, color="orange", zorder=0)
    ax.text(0.84, 2.2, "< 4% threshold", fontsize=7, color="darkorange",
            fontstyle="italic", va="center")

    ax.set_ylabel("VSS (%)", fontsize=10)
    ax.set_title("(a) Value of Stochastic Solution vs β_H", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_xlim(0.36, 0.89)
    ax.set_xticks(betas)

    # Calibrated-value reference line + label (after axis limits are set)
    ax.axvline(BETA_H_CENTRAL, color=C_REF, lw=0.9, linestyle="--")
    y_top = max(max(d["vss_naive"]), max(d["vss_ecd"]))
    ax.text(BETA_H_CENTRAL + 0.007, y_top * 0.98,
            "calibrated", fontsize=7.5, fontstyle="italic",
            color=C_REF, va="top")

    # ── (b) Tier-H cancellation rate ─────────────────────────────────────────
    ax = axes[1]

    ax.plot(betas, d["h_naive"], "o-",
            color=C_NAIVE, lw=2, ms=7, label="Deterministic")
    ax.plot(betas, d["h_ecd"],   "o-",
            color=C_ECD,   lw=2, ms=7, label="Expected-parameter det.")
    ax.plot(betas, d["h_sp"],    "o-",
            color=C_SP,    lw=2, ms=7, label="Stochastic plan")

    ax.axhline(5.0, color=COLORS["target_line"], lw=1.0, linestyle="--")
    ax.text(0.89, 5.0 + 0.12, "5% clinical threshold",
            fontsize=7.5, color=COLORS["target_line"],
            ha="right", va="bottom")
    ax.axvline(BETA_H_CENTRAL, color=C_REF, lw=0.9, linestyle="--")

    ax.set_ylabel("Tier-H cancellation rate (%)", fontsize=10)
    ax.set_title("(b) Tier-H cancellation rate vs β_H", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_xlim(0.36, 0.89)
    ax.set_xticks(betas)

    # ── (c) Facility selection ────────────────────────────────────────────────
    ax = axes[2]

    n_fac = 4
    fac_ys = list(range(n_fac))   # y positions: 0=m0, 1=m1, 2=m2, 3=m3

    jitter = 0.012   # horizontal offset so SP and ECD markers don't overlap

    for k, bH in enumerate(betas):
        # SP: triangle markers
        for m in d["sp_opens"][k]:
            ax.plot(bH + jitter, fac_ys[m], "^",
                    color=C_SP, ms=11, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.5)
        # ECD: square markers (slightly left)
        for m in d["ecd_opens"][k]:
            ax.plot(bH - jitter, fac_ys[m], "s",
                    color=C_ECD, ms=9, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.5)

    ax.axvline(BETA_H_CENTRAL, color=C_REF, lw=0.9, linestyle="--")

    ax.set_yticks(fac_ys)
    ax.set_yticklabels(FAC_LABELS, fontsize=9)
    ax.set_ylim(-0.5, n_fac - 0.5)
    ax.set_xticks(betas)
    ax.set_xlim(0.36, 0.89)

    ax.set_ylabel("Facility", fontsize=10)
    ax.set_title("(c) Facility selection by plan vs β_H", fontsize=11)

    # Legend
    leg_sp  = mlines.Line2D([], [], marker="^", color=C_SP,
                             ms=9, linestyle="none", label="Stochastic plan")
    leg_ecd = mlines.Line2D([], [], marker="s", color=C_ECD,
                             ms=8, linestyle="none", label="Expected-parameter det.")
    ax.legend(handles=[leg_sp, leg_ecd], fontsize=8.5, loc="upper right")

    # Shared x-axis label on bottom panel
    ax.set_xlabel("β_H  (tier-H re-collection eligibility)", fontsize=10)

    save_figure(fig, "figure15_beta_sensitivity")


# ── README update ──────────────────────────────────────────────────────────────

def update_readme() -> None:
    import re as _re
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    entry = (
        "| `figure15_beta_sensitivity.png` | "
        "Sensitivity of the contribution claim to the tier-H eligibility parameter β_H. "
        "(a) The Value of Stochastic Solution remains positive across the calibrated range, "
        "both against the naive deterministic baseline and against the expected-cost "
        "deterministic baseline that anticipates expected failure cost; at β_H ≥ 0.70 the "
        "VSS vs ECD narrows below 4%, reflecting that when re-collection is almost always "
        "feasible the ECD approximation captures most of the structural benefit. "
        "(b) The stochastic plan maintains the lowest tier-H cancellation rate across the "
        "full β_H range; the deterministic plans' protection improves with higher β_H but "
        "never matches the SP. "
        "(c) The SP's facility selection adapts to β_H — across β_H ∈ {0.40, 0.55, 0.70} "
        "the SP consistently chooses m₀+m₂ (highest-yield facility for tier-H patients); "
        "at β_H = 0.85 it switches to m₀+m₃, reflecting that when tier-H re-collection "
        "feasibility is high the cost trade-off shifts. "
        "Vertical dashed line marks the calibrated central value β_H = 0.55. "
        "| `results/sensitivity_beta_results.json` |\n"
    )

    cleaned = _re.sub(r"\| `figure15[^|]+\|[^|]+\|\n?", "", existing)
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Figure 15 — β_H sensitivity sweep ===\n")
    print("[1/2] Generating figure15_beta_sensitivity …")
    generate_figure15()

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p = _HERE / ext / f"figure15_beta_sensitivity.{ext}"
        if p.exists():
            print(f"  [OK]      {p.name}  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  [MISSING] {p.name}")
            ok = False

    if not ok:
        sys.exit(1)
    print("\nFigure 15 generated successfully.")


if __name__ == "__main__":
    main()
