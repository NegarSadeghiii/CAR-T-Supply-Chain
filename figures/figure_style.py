"""
figures/figure_style.py — shared matplotlib style for all paper figures.

Import and call setup_style() at the top of every figure script.
Uses Liberation Sans (Arial-compatible, available on Linux) with DejaVu Sans
as fallback.  All rcParams are applied globally so subsequent figure scripts
inherit them automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    # Planning strategies (consistent across all paper figures)
    "naive_det":         "#B0B0B0",  # light gray  — naive deterministic (Y=1)
    "expected_cost_det": "#707070",  # medium gray — smarter deterministic
    "sp":                "#2A6FB8",  # blue        — stochastic SP optimum

    # Cost components
    "stage1":            "#9EC5E8",  # light blue  — Stage-1 committed cost
    "stage2":            "#1A4F8B",  # dark blue   — Stage-2 / recourse cost

    # Recourse actions
    "remfg":             "#4A8C4A",  # green       — re-manufacture
    "subcontract":       "#D4A744",  # amber       — subcontract
    "cancel":            "#B83A3A",  # red         — cancel

    # Patient tiers (urgency color-coding, consistent throughout paper)
    "tier_H":            "#B83A3A",  # red         — high urgency
    "tier_M":            "#D4A744",  # amber       — medium urgency
    "tier_L":            "#4A8C4A",  # green       — low urgency

    # Reference / threshold lines
    "target_line":       "#000000",  # black, dashed, 1pt
}

# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).parent


def setup_style() -> None:
    """Apply rcParams globally. Call once at the top of every figure script."""
    plt.rcParams.update({
        # Font
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Liberation Sans", "Arial", "Helvetica",
                                 "DejaVu Sans"],
        "mathtext.fontset":     "dejavuserif",
        "font.size":            10,

        # Specific roles
        "axes.labelsize":       11,
        "axes.titlesize":       12,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      8,

        # Lines
        "lines.linewidth":      1.5,
        "grid.linewidth":       0.8,
        "axes.linewidth":       1.0,

        # Spines
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # Grid
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           "#E5E5E5",
        "grid.linestyle":       "-",

        # Background
        "axes.facecolor":       "white",
        "figure.facecolor":     "white",

        # DPI
        "figure.dpi":           300,

        # Legend
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
    })


def single_column() -> Figure:
    """Return a 6×4 inch Figure with the style applied."""
    setup_style()
    return plt.figure(figsize=(6, 4))


def double_column() -> Figure:
    """Return a 9×5 inch Figure with the style applied."""
    setup_style()
    return plt.figure(figsize=(9, 5))


def triple_column() -> Figure:
    """Return a 13×4.5 inch Figure for 1×3 panel layouts."""
    setup_style()
    return plt.figure(figsize=(13, 4.5))


def save_figure(fig: Figure, name: str) -> None:
    """
    Save fig to figures/png/{name}.png (300 DPI) and figures/pdf/{name}.pdf.
    Creates output directories if they don't exist.
    """
    png_dir = _BASE_DIR / "png"
    pdf_dir = _BASE_DIR / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f"{name}.png"
    pdf_path = pdf_dir / f"{name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_style()
    fig = single_column()
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 2], [0, 1, 0.5], color=COLORS["sp"], label="SP")
    ax.plot([0, 1, 2], [0, 0.8, 0.3], color=COLORS["expected_cost_det"],
            label="Det")
    ax.set_xlabel("x axis label")
    ax.set_ylabel("y axis label")
    ax.set_title("Style smoke test")
    ax.legend()
    save_figure(fig, "style_test")
    print("Style test passed.")
