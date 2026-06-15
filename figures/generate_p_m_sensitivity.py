"""
figures/generate_p_m_sensitivity.py — Figure 16: p_m sensitivity heatmap.

8×6 in heatmap of VSS-vs-ECD (%) over the 5×3 grid:
  p_shift  ∈ {-0.10, -0.05, 0.00, +0.05, +0.10}  (rows)
  p_spread ∈ {0.5, 1.0, 1.5}                       (columns)

RdYlGn colormap, centered at 4% (contribution threshold).
Black marker at the calibrated central value (shift=0, spread=1.0).

Run: python figures/generate_p_m_sensitivity.py
Source: results/sensitivity_p_m_results.json
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
import matplotlib.colors as mcolors

setup_style()

_RESULT = _REPO / "results" / "sensitivity_p_m_results.json"

P_SHIFT_VALS  = [-0.10, -0.05, 0.00, +0.05, +0.10]
P_SPREAD_VALS = [0.5, 1.0, 1.5]


def _load():
    with open(_RESULT) as f:
        d = json.load(f)
    pts = d["sweep"]

    # Build (n_shift × n_spread) grid of VSS/ECD values
    n_shift  = len(P_SHIFT_VALS)
    n_spread = len(P_SPREAD_VALS)
    vss_ecd   = np.full((n_shift, n_spread), np.nan)
    vss_naive = np.full((n_shift, n_spread), np.nan)
    sp_fac    = [[None] * n_spread for _ in range(n_shift)]

    for pt in pts:
        si = P_SHIFT_VALS.index(round(pt["p_shift"], 4))
        ki = P_SPREAD_VALS.index(round(pt["p_spread"], 4))
        vss_ecd[si, ki]   = pt["vss_vs_ecd_pct"]
        vss_naive[si, ki] = pt["vss_vs_naive_pct"]
        sp_fac[si][ki]    = "+".join(pt["stochastic"]["facilities"])

    return vss_ecd, vss_naive, sp_fac


def generate_figure16() -> None:
    vss_ecd, vss_naive, sp_fac = _load()

    n_shift  = len(P_SHIFT_VALS)
    n_spread = len(P_SPREAD_VALS)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6),
                              gridspec_kw={"wspace": 0.35})

    THRESHOLD = 4.0   # contribution threshold

    for ax_idx, (ax, data, title, add_star) in enumerate(zip(
        axes,
        [vss_ecd, vss_naive],
        ["(a) VSS vs Expected-cost det. (%)",
         "(b) VSS vs Naive det. (%)"],
        [True, False],
    )):
        vmin = max(0.0, float(np.nanmin(data)) - 1.0)
        vmax = float(np.nanmax(data)) + 1.0

        # Centre the colormap at THRESHOLD (for VSS/ECD panel)
        if add_star:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=THRESHOLD, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        cmap = plt.cm.RdYlGn
        im = ax.imshow(data, cmap=cmap, norm=norm,
                       aspect="auto", origin="lower")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("VSS (%)", fontsize=9)

        # 4% threshold contour (contribution boundary)
        if add_star:
            try:
                ax.contour(np.arange(n_spread), np.arange(n_shift),
                           data, levels=[THRESHOLD],
                           colors=["#333333"], linewidths=[1.2], linestyles=["--"])
            except Exception:
                pass
            # Annotation for threshold line
            ax.text(n_spread - 0.45, 0.35, f"4% threshold",
                    fontsize=7, color="#333333", fontstyle="italic", ha="right")

        # Cell text: VSS value + facility label
        for si in range(n_shift):
            for ki in range(n_spread):
                v = data[si, ki]
                fac = sp_fac[si][ki] if sp_fac[si][ki] else ""
                txt_col = "black" if 0.3 < norm(v) < 0.7 else "white"
                ax.text(ki, si, f"{v:.1f}%\n{fac}",
                        ha="center", va="center",
                        fontsize=7.5, color=txt_col)

        # Calibrated-point star marker
        if add_star or True:
            # calibrated: shift=0.00 (index 2), spread=1.0 (index 1)
            ax.plot(1, 2, "k*", ms=14, zorder=10, markeredgewidth=0.5)

        # Axes
        ax.set_xticks(range(n_spread))
        ax.set_xticklabels([f"{v:.1f}" for v in P_SPREAD_VALS], fontsize=9)
        ax.set_yticks(range(n_shift))
        ax.set_yticklabels([f"{v:+.2f}" for v in P_SHIFT_VALS], fontsize=9)
        ax.set_xlabel("p_spread (deviation multiplier)", fontsize=10)
        ax.set_ylabel("p_shift (additive shift)", fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)

        # Calibrated-point label
        ax.annotate("calibrated", xy=(1, 2), xytext=(1.55, 2.55),
                    fontsize=7.5, fontstyle="italic", color="#444444",
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))

    fig.suptitle("Sensitivity of VSS to manufacturing success probability p_m",
                 fontsize=12, y=1.02)

    save_figure(fig, "figure16_p_m_sensitivity")


def update_readme() -> None:
    import re as _re
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    entry = (
        "| `figure16_p_m_sensitivity.png` | "
        "Sensitivity of the contribution claim to manufacturing success probability p_m. "
        "Two-panel heatmap (5 shift values × 3 spread values): "
        "(a) VSS vs Expected-cost det. — the contribution claim holds (VSS > 4%) across "
        "the majority of the grid; the 4% threshold contour marks the boundary. "
        "(b) VSS vs Naive det. — uniformly higher, confirming robustness. "
        "Star marks the calibrated central value (shift=0, spread=1.0). "
        "| `results/sensitivity_p_m_results.json` |\n"
    )

    cleaned = _re.sub(r"\| `figure16[^|]+\|[^|]+\|\n?", "", existing)
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


def main() -> None:
    print("=== Figure 16 — p_m sensitivity heatmap ===\n")
    print("[1/2] Generating figure16_p_m_sensitivity …")
    generate_figure16()

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p = _HERE / ext / f"figure16_p_m_sensitivity.{ext}"
        if p.exists():
            print(f"  [OK]      {p.name}  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  [MISSING] {p.name}")
            ok = False

    if not ok:
        sys.exit(1)
    print("\nFigure 16 generated successfully.")


if __name__ == "__main__":
    main()
