"""
figures/generate_graphical_abstract.py — Graphical abstract.

Single landscape figure (12 × 5 in, 300 DPI) with three panels:
  Panel 1 (left,  ~28%): The challenge
  Panel 2 (middle,~44%): The approach
  Panel 3 (right, ~28%): The result

Elsevier spec: minimum 531 × 1328 px, aspect ~1:2.5.
At 300 DPI the 12 × 5 output is 3600 × 1500 px — well above the minimum.

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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

setup_style()

# ── layout constants ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 12.0, 5.0

# Title strip: top 7% of figure height
TITLE_H = 0.07

# Panel left edges and widths (fraction of figure width)
P1_L, P1_W = 0.01,  0.27
P2_L, P2_W = 0.285, 0.435
P3_L, P3_W = 0.725, 0.27

# Panel bottom and height (below title strip)
PAN_B = 0.02
PAN_H = 1.0 - TITLE_H - PAN_B - 0.01   # ≈ 0.84

# Separator x positions
SEP_X = [P2_L - 0.005, P3_L - 0.005]

# ── color shortcuts ───────────────────────────────────────────────────────────
C_NAV  = COLORS["stage2"]        # dark navy  — title strip / sampler
C_DET  = COLORS["naive_det"]     # light gray — deterministic plan
C_SP   = COLORS["sp"]            # blue       — stochastic plan
C_REM  = COLORS["remfg"]         # green      — re-manufacture
C_SUB  = COLORS["subcontract"]   # amber      — subcontract
C_CAN  = COLORS["cancel"]        # red        — cancel
C_H    = COLORS["tier_H"]        # red        — Tier-H


# ── drawing helpers ───────────────────────────────────────────────────────────

def _fancy_rect(ax, cx, cy, w, h, fc, ec, lw=1.0, zorder=3,
                alpha=1.0, pad=0.02, radius=None):
    style = f"round,pad={pad}" if radius is None else f"round,pad={pad}"
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style,
        fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha,
        transform=ax.transData,
    ))


def draw_hospital(ax, cx, cy, label="", label_pos="below", fontsize=9):
    """Rounded rectangle with a red + cross."""
    w, h = 0.12, 0.10
    _fancy_rect(ax, cx, cy, w, h, fc="#F0F8FF", ec="#4477AA", lw=1.3, zorder=4, pad=0.008)
    # Red cross
    bar_l, bar_w = 0.034, 0.010
    ax.add_patch(mpatches.Rectangle(
        (cx - bar_l / 2, cy - bar_w / 2), bar_l, bar_w,
        fc="#CC2222", ec="none", zorder=5))
    ax.add_patch(mpatches.Rectangle(
        (cx - bar_w / 2, cy - bar_l / 2), bar_w, bar_l,
        fc="#CC2222", ec="none", zorder=5))
    if label:
        ly = cy - h / 2 - 0.038 if label_pos == "below" else cy + h / 2 + 0.025
        ax.text(cx, ly, label, ha="center", va="top" if label_pos == "below" else "bottom",
                fontsize=fontsize, color="#333333", zorder=6)


def draw_factory(ax, cx, cy, label="", fill_color="#707070", label_pos="below", fontsize=9):
    """Square base + triangular roof + chimney."""
    w, h = 0.12, 0.09
    # Base
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004",
        fc=fill_color, ec=fill_color, lw=1.0, zorder=4))
    # Roof
    roof = [(cx - w / 2 - 0.01, cy + h / 2),
            (cx + w / 2 + 0.01, cy + h / 2),
            (cx, cy + h / 2 + 0.055)]
    ax.add_patch(Polygon(roof, fc=fill_color, ec=fill_color, lw=0.8, zorder=4))
    # Chimney
    ax.add_patch(FancyBboxPatch(
        (cx + 0.015, cy + h / 2), 0.020, 0.032,
        boxstyle="round,pad=0.001",
        fc=fill_color, ec=fill_color, lw=0.5, zorder=4))
    if label:
        ly = cy - h / 2 - 0.038 if label_pos == "below" else cy + h / 2 + 0.020
        ax.text(cx, ly, label, ha="center", va="top" if label_pos == "below" else "bottom",
                fontsize=fontsize, color="#333333", zorder=6)


def draw_recourse_icon(ax, cx, cy, action, size=0.055):
    """Colored circle (remfg), amber rounded arrow (subcontract), red × (cancel)."""
    if action == "remfg":
        ax.add_patch(plt.Circle((cx, cy), size, fc=C_REM, ec=C_REM,
                                zorder=5, lw=0))
        ax.text(cx, cy, "R", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)
    elif action == "subcontract":
        ax.add_patch(FancyArrowPatch(
            (cx - size, cy), (cx + size, cy),
            arrowstyle="-|>",
            color=C_SUB,
            mutation_scale=14, lw=2.5, zorder=5,
        ))
    elif action == "cancel":
        d = size * 0.7
        for dx, dy in [((-d, d), (d, -d)), ((-d, -d), (d, d))]:
            ax.plot([cx + dx[0], cx + dx[1]], [cy + dy[0], cy + dy[1]],
                    color=C_CAN, lw=2.5, solid_capstyle="round", zorder=5)


def draw_horizontal_bar(ax, x_left, cy, length, color, label_right="",
                        bar_h=0.038, fontsize=9):
    """Filled horizontal bar; label placed to the right."""
    ax.add_patch(FancyBboxPatch(
        (x_left, cy - bar_h / 2), length, bar_h,
        boxstyle="round,pad=0.003",
        fc=color, ec=color, lw=0, zorder=4))
    if label_right:
        ax.text(x_left + length + 0.012, cy, label_right,
                ha="left", va="center",
                fontsize=fontsize, fontweight="bold", color=color, zorder=5)


def _arrow(ax, x0, y0, x1, y1, color="#666666", lw=1.2, label="",
           label_side="right", fontsize=8.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10),
                zorder=4, annotation_clip=False)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        offset_x = 0.028 if label_side == "right" else -0.028
        ax.text(mx + offset_x, my, label,
                ha="left" if label_side == "right" else "right",
                va="center", fontsize=fontsize, color=color,
                fontstyle="italic", zorder=5)


def _panel_headline(ax, text, fontsize=12):
    ax.text(0.50, 0.965, text,
            ha="center", va="top",
            fontsize=fontsize, fontweight="bold", color="#1A1A1A",
            transform=ax.transAxes, zorder=6)


def _takeaway(ax, text, fontsize=10):
    ax.text(0.50, 0.025, text,
            ha="center", va="bottom",
            fontsize=fontsize, color="#555555",
            transform=ax.transAxes,
            wrap=True, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="#F8F8F8",
                      ec="#DDDDDD", lw=0.7))


# ── Panel 1: The challenge ────────────────────────────────────────────────────

def _build_panel1(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel_headline(ax, "The challenge")

    # CAR-T flow schematic — centered at x=0.38, top half of panel
    cx_flow = 0.36
    hosp_top_y  = 0.84
    factory_y   = 0.63
    hosp_bot_y  = 0.42

    draw_hospital(ax, cx_flow, hosp_top_y, "Patient", fontsize=8)
    _arrow(ax, cx_flow, hosp_top_y - 0.055, cx_flow, factory_y + 0.060,
           label="leukapheresis", label_side="right", fontsize=7.5)
    draw_factory(ax, cx_flow, factory_y, "Manufacturing\nfacility",
                 fill_color="#707070", fontsize=8)
    _arrow(ax, cx_flow, factory_y - 0.058, cx_flow, hosp_bot_y + 0.058,
           label="product return", label_side="right", fontsize=7.5)
    draw_hospital(ax, cx_flow, hosp_bot_y, "Infusion", fontsize=8)

    # Batch-failure annotation to the right of factory
    fail_x, fail_y = cx_flow + 0.27, factory_y
    # thin dashed line from factory right edge to ×
    ax.annotate("", xy=(fail_x - 0.04, fail_y),
                xytext=(cx_flow + 0.08, factory_y + 0.01),
                arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                lw=0.8, linestyle="dashed"),
                annotation_clip=False, zorder=3)
    # red × glyph
    d = 0.026
    for dx in [(-d, d), (-d, -d)]:
        ax.plot([fail_x + dx[0], fail_x - dx[0]],
                [fail_y + dx[1], fail_y - dx[1]],
                color=C_CAN, lw=2.2, solid_capstyle="round", zorder=5)
    ax.text(fail_x, fail_y - 0.065, "batch\nfailure",
            ha="center", va="top", fontsize=7.5, color=C_CAN,
            fontweight="bold", zorder=5)

    # Failure-rate bars — bottom half
    bar_y_base = 0.29
    bar_ys     = [bar_y_base, bar_y_base - 0.09, bar_y_base - 0.18]
    max_len    = 0.50
    items = [
        ("axi-cel: 4%",   0.04 / 0.28, "#999999"),
        ("tisa-cel: 17%", 0.17 / 0.28, "#808080"),
        ("liso-cel: 28%", 1.00,         "#505050"),
    ]
    x_bar_l = 0.17
    for (lbl, frac, col), by in zip(items, bar_ys):
        ax.text(x_bar_l - 0.02, by, lbl, ha="right", va="center",
                fontsize=8, color="#333333", zorder=5)
        draw_horizontal_bar(ax, x_bar_l, by, frac * max_len, col,
                            bar_h=0.033, fontsize=8)

    ax.text(0.50, bar_ys[-1] - 0.065,
            "Real-world batch-failure rates\n(Dulobdas 2025, UK National CAR-T Panel)",
            ha="center", va="top", fontsize=8, color="#888888",
            fontstyle="italic", transform=ax.transAxes if False else None,
            zorder=5)

    _takeaway(ax,
              "Batch failures cause cancellations among high-urgency patients —\n"
              "modeled as a binary Bernoulli r.v. per patient–facility pair.")


# ── Panel 2: The approach ─────────────────────────────────────────────────────

def _build_panel2(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel_headline(ax, "The approach: two-stage stochastic program\nwith three-action recourse",
                    fontsize=10.5)

    # Stage-1 row
    s1_y   = 0.73
    s1_xs  = [0.18, 0.50, 0.82]
    s1_cols = [C_DET, "#888888", C_SP]
    s1_icons = ["facility", "bar", "arrow"]
    s1_labels = ["open facilities\n(z)", "contract\ncapacity (C)", "assign\npatients (x)"]

    ax.text(0.06, s1_y + 0.005,
            "Stage 1 — here-and-now",
            ha="left", va="center",
            fontsize=9, fontstyle="italic", color="#444444", zorder=5)

    for ix, (ix_x, col, icon, lbl) in enumerate(
            zip(s1_xs, s1_cols, s1_icons, s1_labels)):

        if icon == "facility":
            draw_factory(ax, ix_x, s1_y, fill_color=col, fontsize=8)
        elif icon == "bar":
            # simple bar-chart icon: 3 small vertical bars
            for k, bh in enumerate([0.040, 0.065, 0.050]):
                bx = ix_x - 0.032 + k * 0.032
                ax.add_patch(mpatches.Rectangle(
                    (bx - 0.008, s1_y - bh / 2), 0.016, bh,
                    fc=col, ec="none", zorder=4))
        elif icon == "arrow":
            ax.add_patch(FancyArrowPatch(
                (ix_x - 0.055, s1_y), (ix_x + 0.045, s1_y),
                arrowstyle="-|>",
                color=col, mutation_scale=14, lw=2.0, zorder=4))

        ax.text(ix_x, s1_y - 0.095, lbl,
                ha="center", va="top", fontsize=8, color="#333333", zorder=5)

        # connector arrows between icons
        if ix < 2:
            gap_x = (s1_xs[ix] + s1_xs[ix + 1]) / 2
            _arrow(ax, s1_xs[ix] + 0.08, s1_y,
                   s1_xs[ix + 1] - 0.08, s1_y,
                   color="#AAAAAA", lw=1.0)

    # Big right-arrow from Stage-1 into sampler
    _arrow(ax, 0.91, s1_y, 0.91, 0.575,
           color="#AAAAAA", lw=1.0)

    # Stochastic sampler box
    samp_cx, samp_cy = 0.50, 0.535
    samp_w, samp_h   = 0.56, 0.075
    _fancy_rect(ax, samp_cx, samp_cy, samp_w, samp_h,
                fc=C_NAV, ec=C_NAV, lw=0, zorder=4, pad=0.01)
    ax.text(samp_cx, samp_cy,
            r"observe $Y(\omega)$, $B(\omega)$",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=5)
    _arrow(ax, samp_cx, samp_cy - samp_h / 2 - 0.005,
           samp_cx, samp_cy - samp_h / 2 - 0.065,
           color=C_NAV, lw=1.2)

    # Stage-2 row
    s2_y   = 0.35
    s2_xs  = [0.18, 0.50, 0.82]
    s2_actions = ["remfg", "subcontract", "cancel"]
    s2_labels  = ["re-manufacture", "subcontract", "cancel"]
    s2_cols    = [C_REM, C_SUB, C_CAN]

    ax.text(0.06, s2_y + 0.005,
            "Stage 2 — wait-and-see, per scenario ω",
            ha="left", va="center",
            fontsize=9, fontstyle="italic", color="#444444", zorder=5)

    # Eligibility annotation above Stage-2 icons
    ax.text(0.50, s2_y + 0.115,
            "recourse choice gated by patient clinical eligibility β_u (tier-dependent)",
            ha="center", va="bottom",
            fontsize=8.5, fontstyle="italic", color="#666666", zorder=5)

    for ix, (ix_x, action, lbl, col) in enumerate(
            zip(s2_xs, s2_actions, s2_labels, s2_cols)):
        draw_recourse_icon(ax, ix_x, s2_y, action, size=0.052)
        ax.text(ix_x, s2_y - 0.085, lbl,
                ha="center", va="top", fontsize=8.5,
                color=col, fontweight="bold", zorder=5)

    _takeaway(ax,
              "The stochastic optimum minimizes expected total cost across 200 sampled\n"
              "scenarios; the structural advantage manifests in tier-aware routing and capacity reservation.")


# ── Panel 3: The result ───────────────────────────────────────────────────────

def _build_panel3(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel_headline(ax, "The result")

    bar_l = 0.20   # left start of bars (axis coords)
    max_w = 0.52   # full-width = 100%

    # ── Comparison 1: Tier-H cancellation rate ────────────────────────────────
    c1_y_lbl = 0.83
    ax.text(0.50, c1_y_lbl,
            "High-urgency-tier cancellation rate",
            ha="center", va="center", fontsize=9, color="#333333",
            fontweight="bold", zorder=5)

    # Bars for comparison 1 (rates relative to max 5.40 → proportion of bar space)
    max_val_c1 = 7.0
    c1_rows = [
        (0.76, 5.40, C_DET, "5.40%", "Deterministic"),
        (0.66, 2.15, C_SP,  "2.15%", "Stochastic SP"),
    ]
    for row_y, val, col, val_lbl, plan_lbl in c1_rows:
        ax.text(bar_l - 0.02, row_y, plan_lbl,
                ha="right", va="center", fontsize=8, color=col, zorder=5)
        draw_horizontal_bar(ax, bar_l, row_y, val / max_val_c1 * max_w, col,
                            label_right=val_lbl, bar_h=0.038, fontsize=8.5)

    # Improvement annotation
    ann_x = bar_l + 5.40 / max_val_c1 * max_w + 0.01
    ax.annotate("−60%",
                xy=(bar_l + 2.15 / max_val_c1 * max_w + 0.01, 0.66),
                xytext=(ann_x + 0.05, 0.71),
                fontsize=8.5, fontweight="bold", color=C_SP, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=C_SP, lw=0.9,
                                mutation_scale=8))

    # ── Comparison 2: Service target met ──────────────────────────────────────
    c2_y_lbl = 0.56
    ax.text(0.50, c2_y_lbl,
            r'Service target (95%) met',
            ha="center", va="center", fontsize=9, color="#333333",
            fontweight="bold", zorder=5)

    max_val_c2 = 100.0
    c2_rows = [
        (0.49, 80, C_DET, "80%",  "Deterministic"),
        (0.39, 97, C_SP,  "97%",  "Stochastic SP"),
    ]
    for row_y, val, col, val_lbl, plan_lbl in c2_rows:
        ax.text(bar_l - 0.02, row_y, plan_lbl,
                ha="right", va="center", fontsize=8, color=col, zorder=5)
        draw_horizontal_bar(ax, bar_l, row_y, val / max_val_c2 * max_w, col,
                            label_right=val_lbl, bar_h=0.038, fontsize=8.5)

    ax.annotate("+17 pp",
                xy=(bar_l + 97 / max_val_c2 * max_w + 0.01, 0.39),
                xytext=(bar_l + 80 / max_val_c2 * max_w + 0.08, 0.44),
                fontsize=8.5, fontweight="bold", color=C_SP, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=C_SP, lw=0.9,
                                mutation_scale=8))

    # ── Headline stats callout ────────────────────────────────────────────────
    callout_cy = 0.235
    _fancy_rect(ax, 0.50, callout_cy, 0.90, 0.145,
                fc="#F4F8FF", ec=C_NAV, lw=1.0, zorder=3, pad=0.01)

    stats = [
        ("Mean cost reduction:", "2.42 M USD  (−11.8%)"),
        ("VSS vs deterministic:", "14.6%"),
        ("P(cost below budget):", "0.75  vs  0.49"),
    ]
    top_y = callout_cy + 0.043
    for k, (desc, val) in enumerate(stats):
        row_y = top_y - k * 0.044
        ax.text(0.17, row_y, desc,
                ha="left", va="center", fontsize=8.5, color="#555555", zorder=5)
        ax.text(0.87, row_y, val,
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=C_NAV, zorder=5)

    _takeaway(ax,
              "Validated on 2,000 out-of-sample scenarios;\n"
              "all gaps hold with 90% bootstrap confidence intervals.")


# ── Main assembly ──────────────────────────────────────────────────────────────

def generate_graphical_abstract() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    # Title strip
    ax_title = fig.add_axes([0.0, 1.0 - TITLE_H, 1.0, TITLE_H])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis("off")
    ax_title.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, fc=C_NAV, ec="none", zorder=1,
        transform=ax_title.transAxes))
    ax_title.text(0.50, 0.50,
                  "Two-stage stochastic programming for autologous CAR-T supply chain"
                  " under batch-failure uncertainty",
                  ha="center", va="center",
                  fontsize=12, fontweight="bold", color="white", zorder=3,
                  transform=ax_title.transAxes)

    # Three panels
    ax1 = fig.add_axes([P1_L, PAN_B, P1_W, PAN_H])
    ax2 = fig.add_axes([P2_L, PAN_B, P2_W, PAN_H])
    ax3 = fig.add_axes([P3_L, PAN_B, P3_W, PAN_H])

    _build_panel1(ax1)
    _build_panel2(ax2)
    _build_panel3(ax3)

    # Vertical separator lines
    for sx in SEP_X:
        fig.add_artist(mlines.Line2D(
            [sx, sx], [PAN_B + 0.01, 1.0 - TITLE_H - 0.01],
            transform=fig.transFigure,
            color="#CCCCCC", lw=0.9, alpha=0.5, zorder=0))

    # Save
    out_png = _HERE / "png" / "graphical_abstract.png"
    out_pdf = _HERE / "pdf" / "graphical_abstract.pdf"
    (_HERE / "png").mkdir(parents=True, exist_ok=True)
    (_HERE / "pdf").mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
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
        "Journal graphical abstract — three-panel landscape summary of the paper's "
        "contribution. Panel 1 (The challenge): autologous CAR-T supply chain flow "
        "with real-world batch-failure rates (4%–28%). Panel 2 (The approach): "
        "two-stage stochastic program with three-action recourse (re-manufacture, "
        "subcontract, cancel) gated by tier-dependent clinical eligibility. "
        "Panel 3 (The result): Tier-H cancellation rate halved (5.40% → 2.15%), "
        "service target met in 97% vs 80% of OOS scenarios, mean cost reduction "
        "2.42 M USD (−11.8%). Elsevier-compliant: 3600 × 1500 px at 300 DPI. "
        "| none (schematic) |\n"
    )

    cleaned = _re.sub(r"\| `graphical_abstract[^|]+\| none[^|]+\|\n?", "", existing)
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Graphical abstract ===\n")
    print("[1/2] Generating graphical_abstract …")
    generate_graphical_abstract()

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p = _HERE / ext / f"graphical_abstract.{ext}"
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"  [OK]      {p.name}  ({size_kb} KB)")
        else:
            print(f"  [MISSING] {p.name}")
            ok = False

    # Check pixel dimensions
    try:
        from PIL import Image
        img = Image.open(_HERE / "png" / "graphical_abstract.png")
        w, h = img.size
        print(f"\n  PNG dimensions: {w} × {h} px")
        print(f"  Elsevier min:   1328 × 531 px  ({'PASS' if w >= 1328 and h >= 531 else 'FAIL'})")
        print(f"  Aspect ratio:   {w/h:.2f}:1 (target ≥ 2.0:1)")
    except ImportError:
        print("\n  (PIL not installed — skipping pixel-dimension check)")

    if not ok:
        sys.exit(1)
    print("\nGraphical abstract generated successfully.")


if __name__ == "__main__":
    main()
