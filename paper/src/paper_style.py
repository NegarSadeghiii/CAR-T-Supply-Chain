"""
paper/src/paper_style.py — one shared matplotlib style for every manuscript
figure (Health Care Management Science / healthcare-OR house style).

- Colorblind-safe palette (Okabe-Ito).
- Journal column widths: single ~3.3in, double ~6.9in.
- Clean: thin axes, no top/right spines, light grid, ~9-11pt, tight layout.
- Every figure saved as VECTOR PDF + 300-dpi PNG preview.

Plain patient-and-cost axis labels only on main figures (no kappa/VoE/surrogate).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Okabe-Ito colorblind-safe palette.
OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
    "yellow": "#F0E442", "black": "#000000", "grey": "#8C8C8C",
}

# Semantic roles used across figures (kept consistent throughout the paper).
C = {
    "normal_wait": OKABE["blue"],       # lost during the normal wait
    "after_fail": OKABE["vermillion"],  # lost after a manufacturing failure
    "on_time": OKABE["vermillion"],     # on-time plan
    "decline_aware": OKABE["orange"],   # decline-aware (spare capacity)
    "sickest": OKABE["green"],          # sickest-first
    "saved": OKABE["green"],
    "cost": OKABE["purple"],
    "ref": OKABE["black"],
    "grey": OKABE["grey"],
}

SINGLE_W = 3.3   # inches (single journal column)
DOUBLE_W = 6.9   # inches (double column / full text width)


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.dpi": 300,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def fig(width="double", height_ratio=0.62, ncols=1) -> Figure:
    setup()
    w = DOUBLE_W if width == "double" else SINGLE_W
    if ncols > 1:
        w = DOUBLE_W
    return plt.figure(figsize=(w, w * height_ratio))


def save(fig: Figure, name: str, outdir: Path) -> None:
    png = Path(outdir) / "png"
    pdf = Path(outdir) / "pdf"
    png.mkdir(parents=True, exist_ok=True)
    pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f"{name}.png", dpi=300)
    fig.savefig(pdf / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.pdf + {name}.png")
