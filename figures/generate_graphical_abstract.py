"""
figures/generate_graphical_abstract.py — Graphical abstract (sparse redesign).

14 × 6 in, 300 DPI → 4200 × 1800 px (Elsevier min: 531 × 1328 px).
Three panels: The challenge | The approach | The result.
No takeaway strips. Large icons, large numbers, generous whitespace.

Run: python figures/generate_graphical_abstract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import COLORS, setup_style

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon

setup_style()

# ── palette ────────────────────────────────────────────────────────────────────
C_NAV = COLORS["stage2"]        # dark navy  — title / sampler
C_S1  = COLORS["stage1"]        # light blue — Stage-1 card border
C_DET = COLORS["naive_det"]     # light gray — deterministic / before
C_SP  = COLORS["sp"]            # blue       — stochastic / after
C_REM = COLORS["remfg"]         # green
C_SUB = COLORS["subcontract"]   # amber
C_CAN = COLORS["cancel"]        # red
C_H   = COLORS["tier_H"]        # red — Tier-H

# ── axes layout (figure-fraction coordinates) ─────────────────────────────────
# Title strip: top 0.5 in of 6 in = 8.3 %
_TITLE_B = 0.917       # bottom of title strip in fig coords
_TITLE_H = 0.083

_PAN_B = 0.04          # panel bottom
_PAN_H = _TITLE_B - _PAN_B - 0.015   # ≈ 0.862

# Three panel left edges + widths (figure fractions)
_L1, _W1 = 0.030, 0.28
_L2, _W2 = 0.345, 0.40
_L3, _W3 = 0.775, 0.21

# Separator x positions
_SEP = [_L2 - 0.015, _L3 - 0.015]

# Panel physical sizes (inches) — for sizing icons
_FIG_W, _FIG_H = 14.0, 6.0
_P1W_IN = _W1 * _FIG_W   # 3.92 in
_P1H_IN = _PAN_H * _FIG_H  # 5.17 in
_P2W_IN = _W2 * _FIG_W   # 5.60 in
_P2H_IN = _P1H_IN
_P3W_IN = _W3 * _FIG_W   # 2.94 in
_P3H_IN = _P1H_IN


# ── drawing helpers ────────────────────────────────────────────────────────────

def _fancy_rect(ax, cx, cy, w, h, fc, ec, lw=1.0, zorder=3, alpha=1.0, pad=0.01):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha,
    ))


def draw_person(ax, cx, cy, h, color):
    """Person silhouette: circle head + trapezoid body. h in axes units."""
    head_r  = h * 0.20
    head_cy = cy + h * 0.30

    # Head
    ax.add_patch(plt.Circle((cx, head_cy), head_r,
                             fc=color, ec=color, zorder=4))
    # Body (trapezoid)
    bw_top, bw_bot = h * 0.30, h * 0.44
    body_top = cy + h * 0.07
    body_bot = cy - h * 0.50
    pts = [
        (cx - bw_bot / 2, body_bot), (cx + bw_bot / 2, body_bot),
        (cx + bw_top / 2, body_top), (cx - bw_top / 2, body_top),
    ]
    ax.add_patch(MplPolygon(pts, fc=color, ec=color, zorder=4))


def draw_factory(ax, cx, cy, h, color):
    """Factory: rect base + triangle roof + chimney. h in axes units."""
    bw   = h * 0.64     # base width
    bh   = h * 0.52     # base height
    base_bot = cy - h / 2
    base_top = base_bot + bh
    # Base rectangle
    ax.add_patch(FancyBboxPatch(
        (cx - bw / 2, base_bot), bw, bh,
        boxstyle="round,pad=0.003",
        fc=color, ec=color, lw=0, zorder=4,
    ))
    # Roof triangle
    roof_h = h - bh        # remaining height
    roof = [
        (cx - bw / 2 - h * 0.02, base_top),
        (cx + bw / 2 + h * 0.02, base_top),
        (cx, base_top + roof_h),
    ]
    ax.add_patch(MplPolygon(roof, fc=color, ec=color, lw=0, zorder=4))
    # Chimney
    cw, ch = h * 0.12, h * 0.14
    ax.add_patch(FancyBboxPatch(
        (cx + h * 0.10, base_top - ch * 0.3), cw, ch,
        boxstyle="round,pad=0.001",
        fc=color, ec=color, lw=0, zorder=4,
    ))


def draw_card(ax, cx, cy, w, h, border_color, title, subtitle, body_lines):
    """
    Rounded-rectangle card with a colored top border.
    body_lines: list of (text, fontsize, color, fontweight) tuples.
    """
    # Card background
    _fancy_rect(ax, cx, cy, w, h, "#FAFAFA", "#DDDDDD", lw=0.8, zorder=3, pad=0.008)
    # Colored top border (thin filled rect at top)
    border_h = h * 0.07
    top_y = cy + h / 2 - border_h / 2
    _fancy_rect(ax, cx, top_y, w, border_h, border_color, border_color,
                lw=0, zorder=4, pad=0.004)
    # Title
    ax.text(cx, cy + h / 2 - border_h - 0.040, title,
            ha="center", va="top", fontsize=12, fontweight="bold",
            color="#1A1A1A", zorder=5)
    # Subtitle
    if subtitle:
        ax.text(cx, cy + h / 2 - border_h - 0.115, subtitle,
                ha="center", va="top", fontsize=10,
                fontstyle="italic", color="#666666", zorder=5)
    # Body lines
    body_top = cy + h / 2 - border_h - 0.190
    line_gap = 0.073
    for k, (txt, fs, col, fw) in enumerate(body_lines):
        ax.text(cx, body_top - k * line_gap, txt,
                ha="center", va="top",
                fontsize=fs, color=col, fontweight=fw, zorder=5)


def draw_recourse_icons(ax, cx, cy, icon_size=0.036):
    """Three recourse icons (R / arrow / ×) centered at (cx, cy)."""
    spacing = 0.12
    xs = [cx - spacing, cx, cx + spacing]

    # Green circle with R
    ax.add_patch(plt.Circle((xs[0], cy), icon_size,
                             fc=C_REM, ec=C_REM, zorder=5))
    ax.text(xs[0], cy, "R", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=6)

    # Amber forward arrow
    ax.annotate("", xy=(xs[1] + icon_size, cy),
                xytext=(xs[1] - icon_size, cy),
                arrowprops=dict(arrowstyle="-|>", color=C_SUB,
                                lw=2.5, mutation_scale=16),
                annotation_clip=False, zorder=5)

    # Red ×
    d = icon_size * 0.80
    for sgn in [1, -1]:
        ax.plot([xs[2] - d, xs[2] + d], [cy + sgn * d, cy - sgn * d],
                color=C_CAN, lw=2.5, solid_capstyle="round", zorder=5)


def draw_stat_callout(ax, xc, yc, val_before, val_after, label, annotation,
                      c_before=C_DET, c_after=C_SP, fontsize_val=22):
    """
    Three-row callout:
      Row 1 (yc+0.06): val_before → val_after  [large, split color]
      Row 2 (yc-0.02): annotation              [11pt bold, c_after]
      Row 3 (yc-0.10): label                   [10pt gray]
    """
    y_nums = yc + 0.06
    y_ann  = yc - 0.03
    y_lbl  = yc - 0.105

    # Numbers: three separate elements positioned in thirds of panel width
    ax.text(0.22, y_nums, val_before,
            ha="center", va="center",
            fontsize=fontsize_val, fontweight="bold", color=c_before, zorder=5)
    ax.text(0.50, y_nums, "→",
            ha="center", va="center",
            fontsize=fontsize_val - 4, color="#555555", zorder=5)
    ax.text(0.78, y_nums, val_after,
            ha="center", va="center",
            fontsize=fontsize_val, fontweight="bold", color=c_after, zorder=5)

    # Annotation
    ax.text(xc, y_ann, annotation,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=c_after, zorder=5)

    # Label
    ax.text(xc, y_lbl, label,
            ha="center", va="center",
            fontsize=10, color="#666666", zorder=5)


# ── Panel 1: The challenge ────────────────────────────────────────────────────

def _build_panel1(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel headline
    ax.text(0.50, 0.950, "The challenge",
            ha="center", va="top",
            fontsize=14, fontweight="bold", color="#1A1A1A", zorder=5)

    # ── Big number ────────────────────────────────────────────────────────────
    ax.text(0.50, 0.790, "4–28%",
            ha="center", va="center",
            fontsize=48, fontweight="bold", color=C_CAN, zorder=5)

    ax.text(0.50, 0.660, "batch failure rate",
            ha="center", va="center",
            fontsize=12, color="#444444", zorder=5)

    ax.text(0.50, 0.600, "(axi-cel, tisa-cel, liso-cel)",
            ha="center", va="center",
            fontsize=11, color="#666666", zorder=5)

    ax.text(0.50, 0.540, "UK National CAR-T Panel: 3.87% (Dulobdas 2025)",
            ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#888888", zorder=5)

    # ── Supply chain row ──────────────────────────────────────────────────────
    icon_y  = 0.23
    icon_h  = 0.175     # in axes units ≈ 0.905 in ✓ (> 0.8 in spec)
    xs_icon = [0.15, 0.50, 0.85]

    draw_person(ax, xs_icon[0], icon_y, icon_h, C_H)
    draw_factory(ax, xs_icon[1], icon_y, icon_h + 0.01, "#888888")
    draw_person(ax, xs_icon[2], icon_y, icon_h, "#4A8C4A")  # green = success

    # Small ✓ above infusion person (outcome marker)
    chk_y = icon_y + icon_h * 0.5 + 0.04
    d = 0.022
    ax.plot([xs_icon[2] - d * 0.8, xs_icon[2] - d * 0.1, xs_icon[2] + d * 0.8],
            [chk_y, chk_y - d * 0.7, chk_y + d * 0.7],
            color="#4A8C4A", lw=2.0, solid_capstyle="round", zorder=6)

    # Arrows between icons
    person_rw = icon_h * 0.22   # approx half-width
    factory_lw = icon_h * 0.33
    arrow_kw = dict(arrowstyle="-|>", color="#AAAAAA",
                    lw=1.2, mutation_scale=14)
    ax.annotate("", xy=(xs_icon[1] - factory_lw, icon_y),
                xytext=(xs_icon[0] + person_rw, icon_y),
                arrowprops=arrow_kw, annotation_clip=False, zorder=3)
    ax.annotate("", xy=(xs_icon[2] - person_rw, icon_y),
                xytext=(xs_icon[1] + factory_lw, icon_y),
                arrowprops=arrow_kw, annotation_clip=False, zorder=3)

    # Icon labels — well below icons
    label_y = icon_y - icon_h / 2 - 0.050
    for x, lbl in zip(xs_icon, ["Patient", "Manufacturing", "Infusion"]):
        ax.text(x, label_y, lbl,
                ha="center", va="top",
                fontsize=10, color="#333333", zorder=5)

    # Batch-failure mark: small red circle + × above factory, off to side
    fail_cx, fail_cy = 0.73, 0.32
    fail_r = 0.030
    ax.add_patch(plt.Circle((fail_cx, fail_cy), fail_r,
                             fc="#FFE8E8", ec=C_CAN, lw=1.3, zorder=5))
    d = fail_r * 0.62
    for sgn in [1, -1]:
        ax.plot([fail_cx - d, fail_cx + d],
                [fail_cy + sgn * d, fail_cy - sgn * d],
                color=C_CAN, lw=1.5, solid_capstyle="round", zorder=6)

    # Dashed line from factory upper-right to fail mark
    factory_tip_x = xs_icon[1] + icon_h * 0.22
    factory_tip_y = icon_y + icon_h * 0.35
    ax.plot([factory_tip_x, fail_cx - fail_r - 0.01],
            [factory_tip_y, fail_cy],
            color="#CCCCCC", lw=0.8, linestyle="--", zorder=3)

    # "failure" label to the right of circle
    ax.text(fail_cx + fail_r + 0.025, fail_cy, "failure",
            ha="left", va="center",
            fontsize=9, color=C_CAN, fontstyle="italic", zorder=5)


# ── Panel 2: The approach ─────────────────────────────────────────────────────

def _build_panel2(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel headline + subtitle
    ax.text(0.50, 0.950, "The approach",
            ha="center", va="top",
            fontsize=14, fontweight="bold", color="#1A1A1A", zorder=5)
    ax.text(0.50, 0.880, "Two-stage stochastic program with three-action recourse",
            ha="center", va="top",
            fontsize=11, fontstyle="italic", color="#555555", zorder=5)

    # Three cards: cx, cy, w, h (axes coords)
    # Panel 2 axes = 1.0 × 1.0 ≡ 5.60 × 5.17 in
    # Cards: w = 0.27 (1.51 in), h = 0.47 (2.43 in)
    cx_cards = [0.155, 0.500, 0.845]
    cy_card  = 0.44
    cw, ch   = 0.272, 0.470

    # Card A — Stage 1
    draw_card(ax, cx_cards[0], cy_card, cw, ch,
              border_color=C_S1,
              title="Stage 1",
              subtitle="plan",
              body_lines=[
                  ("Open facilities",    10, "#333333", "normal"),
                  ("Contract capacity",  10, "#333333", "normal"),
                  ("Assign patients",    10, "#333333", "normal"),
              ])

    # Card B — Realize
    draw_card(ax, cx_cards[1], cy_card, cw, ch,
              border_color=C_NAV,
              title="Realize",
              subtitle="sample uncertainty",
              body_lines=[
                  ("Y, B ~ Bernoulli",   13, "#1A1A1A", "bold"),
                  ("200 scenarios",      10, "#555555", "normal"),
              ])

    # Card C — Stage 2
    draw_card(ax, cx_cards[2], cy_card, cw, ch,
              border_color=C_SP,
              title="Stage 2",
              subtitle="recourse",
              body_lines=[])     # icons drawn separately

    # Recourse icons inside Card C
    icon_cx = cx_cards[2]
    icon_cy = cy_card - 0.03
    draw_recourse_icons(ax, icon_cx, icon_cy, icon_size=0.038)
    ax.text(icon_cx, icon_cy - 0.105,
            "gated by patient eligibility βᵤ",
            ha="center", va="top",
            fontsize=9, fontstyle="italic", color="#777777", zorder=5)

    # Arrows between cards
    gap_l = cx_cards[0] + cw / 2 + 0.010
    gap_r = cx_cards[1] - cw / 2 - 0.010
    ax.annotate("", xy=(gap_r, cy_card), xytext=(gap_l, cy_card),
                arrowprops=dict(arrowstyle="-|>", color=C_NAV,
                                lw=2.0, mutation_scale=22),
                annotation_clip=False, zorder=4)

    gap_l2 = cx_cards[1] + cw / 2 + 0.010
    gap_r2 = cx_cards[2] - cw / 2 - 0.010
    ax.annotate("", xy=(gap_r2, cy_card), xytext=(gap_l2, cy_card),
                arrowprops=dict(arrowstyle="-|>", color=C_NAV,
                                lw=2.0, mutation_scale=22),
                annotation_clip=False, zorder=4)


# ── Panel 3: The result ───────────────────────────────────────────────────────

def _build_panel3(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.950, "The result",
            ha="center", va="top",
            fontsize=14, fontweight="bold", color="#1A1A1A", zorder=5)

    # Three stat callouts — centered at y = 0.74, 0.47, 0.20
    callout_ys = [0.74, 0.47, 0.20]

    # Callout 1 — Tier-H cancellation
    draw_stat_callout(ax, 0.50, callout_ys[0],
                      "5.40%", "2.15%",
                      "Tier-H cancellation rate",
                      "−60%",        # −60%
                      fontsize_val=22)

    # Callout 2 — Service target
    draw_stat_callout(ax, 0.50, callout_ys[1],
                      "80%", "97%",
                      "Scenarios meeting 95% service target",
                      "+17 pp",
                      fontsize_val=22)

    # Callout 3 — Cost reduction (single value, no before/after)
    y3 = callout_ys[2]
    ax.text(0.50, y3 + 0.06,
            "2.42 M USD",
            ha="center", va="center",
            fontsize=26, fontweight="bold", color=C_SP, zorder=5)
    ax.text(0.50, y3 - 0.02,
            "VSS = 14.6%",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_SP, zorder=5)
    ax.text(0.50, y3 - 0.095,
            "Mean cost reduction vs. deterministic",
            ha="center", va="center",
            fontsize=10, color="#666666", zorder=5)

    # Thin separator + validation note
    ax.axhline(0.052, xmin=0.05, xmax=0.95,
               color="#CCCCCC", lw=0.7, zorder=3)
    ax.text(0.50, 0.036,
            "Validated on 2,000 out-of-sample scenarios with 90% bootstrap CIs.",
            ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#888888", zorder=5)


# ── Main assembly ──────────────────────────────────────────────────────────────

def generate_graphical_abstract() -> None:
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    fig.patch.set_facecolor("white")

    # Title strip
    ax_t = fig.add_axes([0.0, _TITLE_B, 1.0, _TITLE_H])
    ax_t.axis("off")
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
    ax_t.add_patch(mpatches.Rectangle((0, 0), 1, 1, fc=C_NAV, ec="none",
                                       zorder=1, transform=ax_t.transAxes))
    ax_t.text(0.50, 0.50,
              "Two-stage stochastic programming for autologous CAR-T "
              "supply chain under batch-failure uncertainty",
              ha="center", va="center",
              fontsize=13, fontweight="bold", color="white", zorder=3,
              transform=ax_t.transAxes)

    # Three panel axes
    ax1 = fig.add_axes([_L1, _PAN_B, _W1, _PAN_H])
    ax2 = fig.add_axes([_L2, _PAN_B, _W2, _PAN_H])
    ax3 = fig.add_axes([_L3, _PAN_B, _W3, _PAN_H])

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _build_panel1(ax1)
    _build_panel2(ax2)
    _build_panel3(ax3)

    # Vertical separator rules
    for sx in _SEP:
        fig.add_artist(mlines.Line2D(
            [sx, sx], [_PAN_B + 0.01, _TITLE_B - 0.01],
            transform=fig.transFigure,
            color="#CCCCCC", lw=1.0, alpha=0.40, zorder=0))

    # Save
    out_png = _HERE / "png" / "graphical_abstract.png"
    out_pdf = _HERE / "pdf" / "graphical_abstract.pdf"
    (_HERE / "png").mkdir(parents=True, exist_ok=True)
    (_HERE / "pdf").mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,            bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# ── README update ──────────────────────────────────────────────────────────────

def update_readme() -> None:
    import re as _re
    readme   = _HERE / "README.md"
    existing = readme.read_text()
    entry = (
        "| `graphical_abstract.png` | "
        "Journal graphical abstract (Elsevier-compliant, 14×6 in, 300 DPI). "
        "Panel 1 — The challenge: 4–28% real-world batch-failure rates with "
        "autologous CAR-T supply chain schematic. "
        "Panel 2 — The approach: three-card Stage 1 → Realize → Stage 2 flow "
        "showing the two-stage stochastic program structure. "
        "Panel 3 — The result: Tier-H cancellation halved (5.40%→2.15%), "
        "service target met in 97% vs 80% of OOS scenarios, "
        "cost reduction 2.42 M USD (VSS=14.6%). "
        "| none (schematic) |\n"
    )
    cleaned = _re.sub(r"\| `graphical_abstract[^|]+\| none[^|]+\|\n?", "", existing)
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Graphical abstract (sparse redesign) ===\n")
    print("[1/2] Generating graphical_abstract …")
    generate_graphical_abstract()
    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p = _HERE / ext / f"graphical_abstract.{ext}"
        if p.exists():
            print(f"  [OK]      {p.name}  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  [MISSING] {p.name}")
            ok = False

    try:
        from PIL import Image
        img  = Image.open(_HERE / "png" / "graphical_abstract.png")
        w, h = img.size
        print(f"\n  PNG: {w} × {h} px  |  aspect {w/h:.2f}:1")
        print(f"  Elsevier min: 1328 × 531 px  "
              f"{'PASS' if w >= 1328 and h >= 531 else 'FAIL'}")
    except ImportError:
        print("\n  (PIL not available — skipping pixel check)")

    if not ok:
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
