"""
figures/generate_methodology.py — Figure 13 methodology schematic.

Figure 13 — Planning vs. Evaluation: three planners, one stochastic world.
Non-data conceptual schematic showing Phase A (three planners, different
views of uncertainty) and Phase B (all plans evaluated on the same OOS world).

Run: python figures/generate_methodology.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import COLORS, setup_style, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

setup_style()

# ---------------------------------------------------------------------------
# Coordinate system: data units = inches  (figure 12 × 6.5 in)
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 12.0, 6.5

# ── Phase A: planner boxes ────────────────────────────────────────────────
_BW   = 3.00    # box width
_BH   = 2.05    # box height
_BTH  = 0.48    # title-bar height
_BCY  = 4.98    # box center y
_BXS  = [2.0, 6.0, 10.0]
_BOUT_Y = _BCY - _BH / 2 - 0.24   # output-label center y  ≈ 3.715

# ── Divider ───────────────────────────────────────────────────────────────
_DIV_Y = 3.42

# ── Phase B: sampler ─────────────────────────────────────────────────────
_SM_CX  = 6.0
_SM_CY  = 2.42
_SM_W   = 4.30
_SM_H   = 0.92
_SM_BOT = _SM_CY - _SM_H / 2   # bottom edge of sampler box ≈ 1.96

# ── Phase B: eval boxes ───────────────────────────────────────────────────
_EW   = 2.80
_EH   = 1.12
_ETH  = 0.34
_ECY  = 0.76
_EXS  = [2.0, 6.0, 10.0]
_E_TOP = _ECY + _EH / 2         # top edge of eval boxes     ≈ 1.32

# Clearance from sampler bottom to eval top: _SM_BOT - _E_TOP ≈ 0.64 in > 0.15 ✓

# ── Plan palette ─────────────────────────────────────────────────────────
_PLAN_FC = [COLORS["naive_det"], COLORS["expected_cost_det"], COLORS["sp"]]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _rounded_box(ax, cx, cy, w, h, fc, ec, lw=1.4, zorder=2,
                 alpha=1.0, clip=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=zorder, alpha=alpha,
        clip_on=clip,
    ))


def _plan_box(ax, cx: float, fc: str, title: str,
              body: list[tuple[str, float]]) -> None:
    cy  = _BCY
    w, h, th = _BW, _BH, _BTH

    _rounded_box(ax, cx, cy, w, h, fc, fc, lw=1.5, zorder=2)
    _rounded_box(ax, cx, cy - th / 2, w - 0.08, h - th,
                 "white", "none", lw=0, zorder=3)

    ax.text(cx, cy + h / 2 - th / 2, title,
            ha="center", va="center",
            fontsize=9.0, fontweight="bold", color="white", zorder=4)

    top = cy + h / 2 - th - 0.10
    bot = cy - h / 2 + 0.12
    spc = (top - bot) / (len(body) + 0.2)
    for i, (line, fs) in enumerate(body):
        ax.text(cx, top - (i + 0.65) * spc, line,
                ha="center", va="center",
                fontsize=fs, color="#1A1A1A", zorder=4)

    ax.text(cx, _BOUT_Y,
            r"Plan: $(\mathbf{z}^*,\,\mathbf{C}^*,\,\mathbf{x}^*)$",
            ha="center", va="center",
            fontsize=8.5, fontstyle="italic", color="#333333", zorder=4,
            bbox=dict(boxstyle="round,pad=0.22",
                      fc="#F6F6F6", ec=fc, lw=0.9))


def _sampler_box(ax) -> None:
    cx, cy = _SM_CX, _SM_CY
    w, h = _SM_W, _SM_H
    fc = COLORS["stage2"]

    _rounded_box(ax, cx, cy, w, h, fc, "white", lw=1.8, zorder=4)

    ax.text(cx, cy + h / 2 - 0.15, "Stochastic sampler",
            ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="white", zorder=5)

    lines = [
        (r"$Y_{im}(\omega)\sim\mathrm{Bernoulli}(p_m)$,"
         r"  $B_i(\omega)\sim\mathrm{Bernoulli}(\beta_{u(i)})$", 8.0),
        (r"$|\Omega|=2{,}000$ held-out scenarios (seed 4242,"
         r" independent of planning seed 0)", 7.6),
    ]
    top = cy + h / 2 - 0.30
    bot = cy - h / 2 + 0.08
    spc = (top - bot) / (len(lines) + 0.1)
    for i, (line, fs) in enumerate(lines):
        ax.text(cx, top - (i + 0.6) * spc, line,
                ha="center", va="center",
                fontsize=fs, color="white", zorder=5)


def _eval_box(ax, cx: float, fc: str, title: str,
              body: list[tuple[str, float]]) -> None:
    """
    Eval boxes use zorder ≥ 10 so they always render on top of every arrow
    (sampler arrows have zorder 6; frozen-plan arrows have zorder 1).
    """
    cy  = _ECY
    w, h, th = _EW, _EH, _ETH

    _rounded_box(ax, cx, cy, w, h, fc, fc, lw=1.2, zorder=10)
    _rounded_box(ax, cx, cy - th / 2, w - 0.08, h - th,
                 "#F8F8F8", "none", lw=0, zorder=11)

    ax.text(cx, cy + h / 2 - th / 2, title,
            ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="white", zorder=12)

    top = cy + h / 2 - th - 0.08
    bot = cy - h / 2 + 0.08
    spc = (top - bot) / (len(body) + 0.2)
    for i, (line, fs) in enumerate(body):
        ax.text(cx, top - (i + 0.65) * spc, line,
                ha="center", va="center",
                fontsize=fs, color="#1A1A1A", zorder=12)


# ---------------------------------------------------------------------------
# Arrows
# ---------------------------------------------------------------------------

def _frozen_plan_arrows(ax) -> None:
    """
    Thin gray-dashed arrows from each plan's output label straight down
    to the top of the corresponding eval box.
    """
    src_y = _BOUT_Y - 0.17
    dst_y = _E_TOP

    for cx in _BXS:
        ax.annotate(
            "",
            xy=(cx, dst_y), xytext=(cx, src_y),
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-|>",
                color="#AAAAAA", lw=1.0,
                linestyle="dashed",
                mutation_scale=10,
            ),
            zorder=1,
        )

    # Single label alongside the left arrow
    mid_y = (src_y + dst_y) / 2 + 0.18
    ax.text(_BXS[0] - 0.20, mid_y, "frozen\nfirst-stage",
            ha="right", va="center",
            fontsize=7.5, fontstyle="italic", color="#999999", zorder=2)


def _sampler_arrows(ax) -> None:
    """
    Three dark-blue arrows from the sampler box bottom edge to the eval boxes.

    Key geometry per the spec:
    • Each arrow starts at a distinct point on the sampler bottom edge
      (not all from the same center point) so arrows fan outward without
      bowing back into the sampler body text.
    • Vertical clearance between _SM_BOT and _E_TOP is ≈ 0.64 in (> 0.15).
    • rad=0.0 (center/straight), ±0.20 (left/right), mutation_scale=18, lw=2.0.
    """
    sm_color = COLORS["stage2"]

    # Arrow start x positions spread 1/3 of sampler half-width apart
    spread   = _SM_W / 3.5          # ≈ 1.23 in from center
    x_starts = [_SM_CX - spread, _SM_CX, _SM_CX + spread]
    rads     = [0.20, 0.0, -0.20]

    for (x_src, rad), ex in zip(zip(x_starts, rads), _EXS):
        ax.annotate(
            "",
            xy=(ex, _E_TOP), xytext=(x_src, _SM_BOT),
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-|>",
                color=sm_color, lw=2.0,
                mutation_scale=18,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=6,
        )

    # Label alongside the right arrow
    mid_y = (_SM_BOT + _E_TOP) / 2
    ax.text(_SM_CX + _SM_W / 2 + 0.20, mid_y + 0.08,
            "OOS scenarios\n(shared across\nall three plans)",
            ha="left", va="center",
            fontsize=7.5, fontstyle="italic", color=sm_color, zorder=6)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def generate_figure13() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Phase A header ────────────────────────────────────────────────
    ax.text(FIG_W / 2, 6.32,
            "Phase A — Planning  "
            "(each planner sees a different view of uncertainty)",
            ha="center", va="top",
            fontsize=11, fontweight="bold", fontstyle="italic",
            color="#1A1A1A")

    # ── Planner boxes ─────────────────────────────────────────────────
    titles = [
        "Deterministic\nplanner",
        "Expected-parameter\ndeterministic planner",
        "Stochastic\nplanner",
    ]
    bodies = [
        [
            (r"$Y=1,\;B=1$", 9.0),
            ("1 scenario", 9.0),
            (r"cost: $c_m$", 8.5),
        ],
        [
            (r"$Y=1,\;B=1$", 9.0),
            ("1 scenario", 9.0),
            (r"cost: $c_m^{\mathrm{eff}}=c_m+(1-p_m)\cdot\mathbb{E}[\rho]$", 7.5),
        ],
        [
            (r"$Y,B\sim\mathrm{Bernoulli}$", 9.0),
            ("200 scenarios (seed 0)", 9.0),
            (r"$\min\,\mathbb{E}_\omega[\text{Stage-1}+\text{Stage-2}(\omega)]$",
             8.0),
        ],
    ]
    for cx, fc, title, body in zip(_BXS, _PLAN_FC, titles, bodies):
        _plan_box(ax, cx, fc, title, body)

    # ── Divider + punchline ───────────────────────────────────────────
    ax.plot([0.40, FIG_W - 0.40], [_DIV_Y, _DIV_Y],
            color="#C8C8C8", lw=0.9, zorder=1)
    ax.text(FIG_W / 2, _DIV_Y,
            "  Different views during planning;  shared world during evaluation.  ",
            ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#444444",
            bbox=dict(boxstyle="round,pad=0.30",
                      fc="white", ec="#C8C8C8", lw=0.8),
            zorder=3)

    # ── Phase B header ────────────────────────────────────────────────
    ax.text(FIG_W / 2, _DIV_Y - 0.07,
            "Phase B — Evaluation  "
            "(all three plans face the same stochastic world)",
            ha="center", va="top",
            fontsize=11, fontweight="bold", fontstyle="italic",
            color="#1A1A1A")

    # ── Drawing order matters for z-axis clarity ──────────────────────
    # Step 1: frozen-plan arrows (lowest layer, zorder 1)
    _frozen_plan_arrows(ax)

    # Step 2: sampler box and its text (zorder 4-5)
    _sampler_box(ax)

    # Step 3: sampler→eval arrows (zorder 6) — travel through empty space
    #         between sampler bottom (_SM_BOT ≈ 1.96) and eval top
    #         (_E_TOP ≈ 1.32); no content in that gap.
    _sampler_arrows(ax)

    # Step 4: eval boxes on top (zorder 10-12) — arrows cannot cover their text
    eval_titles = [
        "Deterministic plan evaluated",
        "Expected-parameter plan evaluated",
        "Stochastic plan evaluated",
    ]
    eval_bodies = [
        [("Mean cost: 20.58 M USD", 8.5), ("P(service ≥ 95%): 0.80", 8.5)],
        [("Mean cost: 19.15 M USD", 8.5), ("P(service ≥ 95%): 0.91", 8.5)],
        [("Mean cost: 18.16 M USD", 8.5), ("P(service ≥ 95%): 0.97", 8.5)],
    ]
    for cx, fc, title, body in zip(_EXS, _PLAN_FC, eval_titles, eval_bodies):
        _eval_box(ax, cx, fc, title, body)

    save_figure(fig, "figure13_methodology_schematic")


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme() -> None:
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    entry = (
        "| `figure13_methodology_schematic.png` | "
        "Methodological structure of the comparison framework. Phase A (top): "
        "each of the three planners optimizes under a different view of uncertainty "
        "— the deterministic and expected-parameter deterministic planners assume no "
        "failures (Y=1, B=1) in a single scenario, while the stochastic planner sees "
        "200 sampled (Y, B) realizations. Phase B (bottom): all three planners' frozen "
        "first-stage decisions are deployed under the same stochastic sampler, with "
        "2,000 held-out scenarios drawn from the same Bernoulli distributions used in "
        "Phase A but with an independent seed. The comparison is policy-vs-policy on a "
        "shared environment; the differences reported in Figures 7–10 reflect the "
        "structural value of optimizing under uncertainty rather than ignoring it. "
        "| none (schematic) |\n"
    )
    import re as _re
    cleaned = _re.sub(
        r"\| `figure13_methodology[^|]+\| none[^|]+\|\n?", "", existing
    )
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Figure 13 — methodology schematic ===\n")
    print("[1/1] Generating figure13_methodology_schematic …")
    generate_figure13()

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p = _HERE / ext / f"figure13_methodology_schematic.{ext}"
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}")
        if status != "OK":
            ok = False

    if not ok:
        import sys as _sys
        _sys.exit(1)
    print("\nFigure 13 generated successfully.")


if __name__ == "__main__":
    main()
