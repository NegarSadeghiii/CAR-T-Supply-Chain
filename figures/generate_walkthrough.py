"""
figures/generate_walkthrough.py — Figure 14 supply-chain walkthrough.

Figure 14 — Side-by-side, 5-row walkthrough comparing how the Deterministic
and Stochastic plans handle the same patient cohort through a representative
out-of-sample scenario (seed 4242).

Rows:
  1. Patient cohort (identical in both columns): 50 icons, tier-coloured
  2. First-stage manufacturing plan (different per column)
  3. Primary-assignment arrows + FROZEN badge (between rows 1 and 2)
  4. Scenario ω: same stochastic realization, per-patient failure markers
  5. Recourse decisions: per-patient outcome markers (✓ / R / ✕)

Run: python figures/generate_walkthrough.py
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
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon

setup_style()

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

FIG_W, FIG_H = 14.0, 10.0

_CL = 0.25       # left column left edge
_CR = 7.25       # right column left edge
_CW = 6.50       # column width
_DIV_X = 7.0     # vertical divider

# Facility positions within a column (offsets from col_l)
_FX_OFF = [0.60, 2.10, 3.60, 5.10]

# Patient icon radius
_PR = 0.085

# Factory icon dimensions
_FW_FAC = 0.64    # factory base width
_FH_FAC = 0.50    # factory base height
_FR_H   = 0.27    # roof height
_FC_W   = 0.10    # chimney width
_FC_H   = 0.14    # chimney height

# Row 1 patient sub-row y-coords (H, M1, M2, L1, L2)
_R1_YS = [9.26, 8.98, 8.73, 8.50, 8.27]
# Row 4 patient sub-row y-coords
_R4_YS = [5.20, 4.93, 4.68, 4.45, 4.22]
# Row 5 recourse sub-row y-coords
_R5_YS = [3.82, 3.55, 3.30, 3.07, 2.84]

# Facility center y (Row 2)
_FAC_Y = 6.90

# Phase divider y
_DIV_Y = 5.60

# Plan colours (text and accents)
_DET_COL = "#303030"
_SP_COL  = COLORS["sp"]

# Tier colours
_TC = {"H": COLORS["tier_H"], "M": COLORS["tier_M"], "L": COLORS["tier_L"]}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load():
    with open(_REPO / "case_study_results.json") as f:
        d = json.load(f)
    inst = d["instance"]
    return {
        "tiers":      inst["tiers"],          # list[str], len 50
        "p":          inst["p"],              # facility yield probs [0.85,0.92,0.95,0.92]
        "beta":       inst["beta"],           # per-patient batch probs
        "rho_cancel": inst["rho_cancel"],
        "rho_remfg":  inst["rho_remfg"],
        "c":          inst["c"],
        "z_det": d["ev"]["z"],   "C_det": d["ev"]["C"],
        "x_det": np.array(d["ev"]["x"]),
        "z_sp":  d["rp"]["z"],   "C_sp":  d["rp"]["C"],
        "x_sp":  np.array(d["rp"]["x"]),
    }

# ---------------------------------------------------------------------------
# Greedy recourse simulation
# ---------------------------------------------------------------------------

def _greedy_recourse(data, x, C, z, Y, B):
    """
    Greedy recourse for one scenario.
    Returns list of 50 strings: "ok" | "remfg" | "cancel".
    """
    n_p = len(data["tiers"])
    outcomes = ["ok"] * n_p

    # Spare capacity = C[m] - (original assignments that SUCCEED)
    spare = [0] * 4
    for m in range(4):
        if z[m] == 0:
            spare[m] = 0
        else:
            used = sum(x[i, m] * Y[i, m] for i in range(n_p))
            spare[m] = max(0.0, C[m] - used)

    # Process failures ordered by cancel cost (H-tier first)
    rho_c = data["rho_cancel"]
    order = sorted(range(n_p), key=lambda i: -rho_c[i])

    for i in order:
        m_orig = int(np.argmax(x[i]))
        if Y[i, m_orig] == 1.0:
            outcomes[i] = "ok"
            continue
        # Failed at original facility
        if B[i] == 0.0:
            outcomes[i] = "cancel"
            continue
        # Try remfg: prefer lowest-cost open facility with spare capacity
        fac_pref = sorted(range(4),
                          key=lambda m: data["rho_remfg"][m] if z[m] else 1e9)
        placed = False
        for m_dst in fac_pref:
            if z[m_dst] == 0:
                continue
            if spare[m_dst] >= 1:
                outcomes[i] = "remfg"
                spare[m_dst] -= 1
                placed = True
                break
        if not placed:
            outcomes[i] = "cancel"

    return outcomes


def _find_scenario(data):
    """
    Scan OOS scenarios (seed 4242) for the first one with ≥3 more H-tier
    cancellations in the deterministic plan than the stochastic plan.
    Returns (Y, B, det_outcomes, sp_outcomes).
    """
    rng = np.random.default_rng(4242)
    p    = np.array(data["p"])
    beta = np.array(data["beta"])
    n_p  = len(data["tiers"])

    for _ in range(2000):
        Y = rng.binomial(1, p[np.newaxis, np.newaxis, :],
                         size=(1, n_p, 4)).astype(float)[0]   # (n_p, 4)
        B = rng.binomial(1, beta[np.newaxis, :],
                         size=(1, n_p)).astype(float)[0]       # (n_p,)

        det_out = _greedy_recourse(data, data["x_det"], data["C_det"], data["z_det"], Y, B)
        sp_out  = _greedy_recourse(data, data["x_sp"],  data["C_sp"],  data["z_sp"],  Y, B)

        n_hc_det = sum(1 for i in range(10) if det_out[i] == "cancel")
        n_hc_sp  = sum(1 for i in range(10) if sp_out[i]  == "cancel")

        if n_hc_det - n_hc_sp >= 3 and 3 <= n_hc_det <= 7:
            return Y, B, det_out, sp_out

    # Fallback: first scenario regardless
    rng2 = np.random.default_rng(4242)
    Y = rng2.binomial(1, p[np.newaxis, np.newaxis, :],
                      size=(1, n_p, 4)).astype(float)[0]
    B = rng2.binomial(1, beta[np.newaxis, :],
                      size=(1, n_p)).astype(float)[0]
    det_out = _greedy_recourse(data, data["x_det"], data["C_det"], data["z_det"], Y, B)
    sp_out  = _greedy_recourse(data, data["x_sp"],  data["C_sp"],  data["z_sp"],  Y, B)
    return Y, B, det_out, sp_out

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _row_band(ax, y_top, y_bot, fc="#F8F8F8"):
    """Lightly shaded background band for a row."""
    ax.add_patch(Rectangle((0.0, y_bot), FIG_W, y_top - y_bot,
                            fc=fc, ec="none", zorder=0))


def _patient_positions(col_l):
    """Return dict: patient_idx -> (x, y) for the 5 sub-rows."""
    pos = {}
    ys = _R1_YS
    # H: 10 patients
    for k in range(10):
        pos[k] = (col_l + _CW * (k + 0.5) / 10, ys[0])
    # M: 13 + 12
    for k in range(13):
        pos[10 + k] = (col_l + _CW * (k + 0.5) / 13, ys[1])
    for k in range(12):
        pos[23 + k] = (col_l + _CW * (k + 0.5) / 12, ys[2])
    # L: 8 + 7
    for k in range(8):
        pos[35 + k] = (col_l + _CW * (k + 0.5) / 8, ys[3])
    for k in range(7):
        pos[43 + k] = (col_l + _CW * (k + 0.5) / 7, ys[4])
    return pos


def _draw_patient_grid(ax, col_l, row_ys, tiers,
                       mark_fail=None, mark_outcome=None, alpha=1.0):
    """
    Draw 50 patient icons at positions derived from row_ys.
    mark_fail: set of patient indices to overlay with red ×
    mark_outcome: dict idx -> "ok"|"remfg"|"cancel"
    """
    counts = [10, 13, 12, 8, 7]    # sub-row counts
    starts = [0,  10, 23, 35, 43]  # starting index

    for row_i, (n, start, y) in enumerate(zip(counts, starts, row_ys)):
        for k in range(n):
            i = start + k
            x = col_l + _CW * (k + 0.5) / n
            tier = tiers[i]
            fc = _TC[tier]

            if mark_outcome is not None:
                out = mark_outcome[i]
                if out == "ok":
                    fc = COLORS["remfg"]
                elif out == "remfg":
                    fc = COLORS["remfg"]
                elif out == "cancel":
                    fc = COLORS["cancel"]

            ax.add_patch(Circle((x, y), _PR,
                                fc=fc, ec="white", lw=0.4,
                                alpha=alpha, zorder=5))

            # Outcome markers drawn as lines (no Unicode)
            if mark_outcome is not None:
                out = mark_outcome[i]
                d = _PR * 0.52
                if out == "ok":
                    # Draw check-mark: two strokes
                    ax.plot([x - d * 0.6, x - d * 0.05, x + d * 0.7],
                            [y - d * 0.0, y - d * 0.55, y + d * 0.6],
                            color="white", lw=1.0, zorder=7, solid_capstyle="round")
                elif out == "remfg":
                    ax.text(x, y, "R", ha="center", va="center",
                            fontsize=5.5, color="white", fontweight="bold", zorder=7)
                elif out == "cancel":
                    ax.plot([x - d, x + d], [y + d, y - d],
                            color="white", lw=1.2, zorder=7,
                            solid_capstyle="round")
                    ax.plot([x - d, x + d], [y - d, y + d],
                            color="white", lw=1.2, zorder=7,
                            solid_capstyle="round")

            # Failure cross overlay (row 4)
            if mark_fail is not None and i in mark_fail:
                d = _PR * 0.68
                ax.plot([x - d, x + d], [y + d, y - d],
                        color="#CC0000", lw=1.4, zorder=8, solid_capstyle="round")
                ax.plot([x - d, x + d], [y - d, y + d],
                        color="#CC0000", lw=1.4, zorder=8, solid_capstyle="round")


def _facility_x(col_l, m):
    return col_l + _FX_OFF[m]


def _draw_factory(ax, cx, cy, fc, closed=False, label="", cap_str="", p_str=""):
    """Draw a factory icon: rect base + triangle roof + chimney."""
    alpha = 0.25 if closed else 1.0
    ec    = "#AAAAAA" if closed else fc
    lc    = "#BBBBBB" if closed else "#333333"

    bw, bh = _FW_FAC, _FH_FAC
    # Base rectangle
    ax.add_patch(Rectangle((cx - bw / 2, cy - bh / 2), bw, bh,
                            fc=fc if not closed else "#E0E0E0",
                            ec=ec, lw=1.1, alpha=alpha, zorder=3))
    # Roof triangle
    roof_pts = [[cx - bw / 2 - 0.06, cy + bh / 2],
                [cx + bw / 2 + 0.06, cy + bh / 2],
                [cx,                  cy + bh / 2 + _FR_H]]
    ax.add_patch(Polygon(roof_pts,
                         fc=fc if not closed else "#D8D8D8",
                         ec=ec, lw=1.1, alpha=alpha, zorder=3))
    # Chimney
    chim_x = cx + 0.15
    ax.add_patch(Rectangle((chim_x - _FC_W / 2, cy + bh / 2),
                            _FC_W, _FC_H,
                            fc=fc if not closed else "#D0D0D0",
                            ec=ec, lw=0.8, alpha=alpha, zorder=3))

    # Label below
    ax.text(cx, cy - bh / 2 - 0.09, label,
            ha="center", va="top",
            fontsize=8.5, fontweight="bold",
            color=lc, zorder=4)
    if cap_str:
        ax.text(cx, cy - bh / 2 - 0.32, cap_str,
                ha="center", va="top", fontsize=7.5, color=lc, zorder=4)
    if p_str:
        ax.text(cx, cy - bh / 2 - 0.52, p_str,
                ha="center", va="top", fontsize=7.2,
                color="#888888" if closed else "#555555", zorder=4)


def _draw_fail_marker(ax, cx, cy, failed=True):
    """Overlay ✕ or ✓ on a facility icon."""
    if failed:
        r = 0.25
        ax.add_patch(Circle((cx, cy), r,
                             fc="#CC0000", ec="white", lw=0.8, alpha=0.88, zorder=6))
        ax.text(cx, cy, "✕", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold", zorder=7)
    else:
        r = 0.25
        ax.add_patch(Circle((cx, cy), r,
                             fc=COLORS["remfg"], ec="white", lw=0.8, alpha=0.85, zorder=6))
        ax.text(cx, cy, "✓", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold", zorder=7)


def _rounded_box(ax, cx, cy, w, h, fc, ec, lw=1.2, zorder=3, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha,
    ))

# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def generate_figure14(data, Y, B, det_out, sp_out):
    tiers = data["tiers"]

    # Determine which patients fail at their assigned facility
    x_det = data["x_det"]
    x_sp  = data["x_sp"]
    z_det = data["z_det"]
    z_sp  = data["z_sp"]
    C_det = data["C_det"]
    C_sp  = data["C_sp"]

    fail_det = {i for i in range(50) if Y[i, int(np.argmax(x_det[i]))] == 0}
    fail_sp  = {i for i in range(50) if Y[i, int(np.argmax(x_sp[i]))]  == 0}

    # Count outcomes
    def _counts(outcomes, n=50):
        return (sum(o == "ok"     for o in outcomes[:n]),
                sum(o == "remfg"  for o in outcomes[:n]),
                sum(o == "cancel" for o in outcomes[:n]))

    det_ok, det_rem, det_can = _counts(det_out)
    sp_ok,  sp_rem,  sp_can  = _counts(sp_out)
    det_h_can = sum(1 for i in range(10) if det_out[i] == "cancel")
    sp_h_can  = sum(1 for i in range(10) if sp_out[i]  == "cancel")

    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    p_vals = data["p"]

    # ── Row bands (light shading) ──────────────────────────────────────────
    _row_band(ax, 9.44, 8.18, "#F5F8FF")
    _row_band(ax, 7.98, 6.20, "#F8FFF8")
    _row_band(ax, 5.42, 4.15, "#FFF8F5")
    _row_band(ax, 4.05, 2.70, "#F5FFFE")

    # ── Vertical divider ──────────────────────────────────────────────────
    ax.plot([_DIV_X, _DIV_X], [0.40, 9.80],
            color="#CCCCCC", lw=0.8, zorder=1)

    det_col_cx = _CL + _CW / 2
    sp_col_cx  = _CR + _CW / 2

    # ── Column headers ────────────────────────────────────────────────────
    _rounded_box(ax, det_col_cx, 9.80, 3.20, 0.34,
                 fc="#F0F0F0", ec=COLORS["naive_det"], lw=1.8, zorder=3)
    ax.text(det_col_cx, 9.80, "Deterministic plan",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=_DET_COL, zorder=5)

    _rounded_box(ax, sp_col_cx, 9.80, 3.20, 0.34,
                 fc="#EEF4FF", ec=_SP_COL, lw=1.8, zorder=3)
    ax.text(sp_col_cx, 9.80, "Stochastic plan",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=_SP_COL, zorder=5)

    # ── Phase A sub-header ─────────────────────────────────────────────────
    ax.text(FIG_W / 2, 9.58,
            "Phase A — Planning  "
            "(each plan selects facilities z, capacities C, and assignments x "
            "before any uncertainty is revealed)",
            ha="center", va="top",
            fontsize=8.5, fontstyle="italic", color="#555555", zorder=6)

    # ── Row 1: Patient cohort ────────────────────────────────────────────
    ax.text(0.15, np.mean(_R1_YS), "Row 1",
            ha="center", va="center",
            fontsize=6.5, fontstyle="italic", color="#888888",
            rotation=90)
    ax.text(FIG_W / 2, 9.44,
            "Patient cohort (identical in both columns)  "
            "— Tier H (×10) ● red,  Tier M (×25) ● amber,  Tier L (×15) ● green",
            ha="center", va="top",
            fontsize=8, color="#444444")

    for col_l in (_CL, _CR):
        _draw_patient_grid(ax, col_l, _R1_YS, tiers)

    # ── Row 2: Facilities ─────────────────────────────────────────────────
    ax.text(0.15, np.mean([6.20, 7.98]), "Row 2",
            ha="center", va="center",
            fontsize=6.5, fontstyle="italic", color="#888888",
            rotation=90)
    ax.text(FIG_W / 2, 7.98,
            "First-stage plan  — facility selection z and capacity C",
            ha="center", va="top",
            fontsize=8, color="#444444")

    fac_label = ["m₀", "m₁", "m₂", "m₃"]
    p_labels  = [f"p={p_vals[m]:.2f}" for m in range(4)]

    for col_l, z, C, plan_col in [
            (_CL, z_det, C_det, "#787878"),   # medium gray for det open facilities
            (_CR, z_sp,  C_sp,  _SP_COL)]:
        for m in range(4):
            cx = _facility_x(col_l, m)
            opened = bool(z[m])
            fc = plan_col if opened else "#D8D8D8"
            cap_str = f"C={C[m]}" if opened else "closed"
            _draw_factory(ax, cx, _FAC_Y, fc,
                          closed=not opened,
                          label=fac_label[m],
                          cap_str=cap_str,
                          p_str=p_labels[m])

    # ── Assignment arrows (Row 1 patients → Row 2 facilities) ────────────
    fac_top_y = _FAC_Y + _FH_FAC / 2 + _FR_H + _FC_H  # ≈ 7.55

    for col_l, x_mat, tiers_list in [
            (_CL, x_det, tiers),
            (_CR, x_sp,  tiers)]:
        pat_pos = _patient_positions(col_l)
        for i in range(50):
            m = int(np.argmax(x_mat[i]))
            px, py = pat_pos[i]
            fx = _facility_x(col_l, m)
            fy = fac_top_y
            tier = tiers[i]
            ax.plot([px, fx], [py - _PR, fy],
                    color=_TC[tier], lw=0.45, alpha=0.22, zorder=2)

    # ── "FROZEN first-stage" badge ────────────────────────────────────────
    for cx_badge in (det_col_cx, sp_col_cx):
        ax.text(cx_badge, 7.78,
                "  [FROZEN first-stage decisions]  ",
                ha="center", va="center",
                fontsize=7.8, fontstyle="italic", color="#2255AA",
                bbox=dict(boxstyle="round,pad=0.22",
                          fc="#EEF4FF", ec="#6688CC", lw=0.9),
                zorder=5)

    # ── Phase divider ─────────────────────────────────────────────────────
    ax.plot([0.35, FIG_W - 0.35], [_DIV_Y, _DIV_Y],
            color="#BBBBBB", lw=0.9, zorder=1)
    ax.text(FIG_W / 2, _DIV_Y,
            "  Different first-stage decisions;  same out-of-sample world.  ",
            ha="center", va="center",
            fontsize=8.5, fontstyle="italic", color="#444444",
            bbox=dict(boxstyle="round,pad=0.28",
                      fc="white", ec="#BBBBBB", lw=0.8), zorder=4)

    ax.text(FIG_W / 2, _DIV_Y - 0.06,
            "Phase B — Evaluation  (frozen plans meet the same stochastic world)",
            ha="center", va="top",
            fontsize=9, fontstyle="italic", color="#444444",
            bbox=dict(boxstyle="round,pad=0.20",
                      fc="white", ec="#C8C8C8", lw=0.7), zorder=4)

    # ── Row 4: Scenario ω ─────────────────────────────────────────────────
    ax.text(0.15, np.mean(_R4_YS), "Row 4",
            ha="center", va="center",
            fontsize=6.5, fontstyle="italic", color="#888888",
            rotation=90)
    n_fail_det = len(fail_det)
    n_fail_sp  = len(fail_sp)
    ax.text(FIG_W / 2, 5.44,
            f"Scenario ω — same realization in both columns  "
            f"(red X = patient failed at assigned facility:  "
            f"det {n_fail_det} failures,  SP {n_fail_sp} failures)",
            ha="center", va="top",
            fontsize=7.8, color="#444444")

    _draw_patient_grid(ax, _CL, _R4_YS, tiers, mark_fail=fail_det, alpha=0.85)
    _draw_patient_grid(ax, _CR, _R4_YS, tiers, mark_fail=fail_sp,  alpha=0.85)

    # ── Row 5: Recourse ───────────────────────────────────────────────────
    ax.text(0.15, np.mean(_R5_YS), "Row 5",
            ha="center", va="center",
            fontsize=6.5, fontstyle="italic", color="#888888",
            rotation=90)
    ax.text(FIG_W / 2, 4.07,
            "Recourse decisions  — check = treated,  R = re-manufactured,  X = cancelled",
            ha="center", va="top",
            fontsize=7.8, color="#444444")

    det_outcome_map = {i: det_out[i] for i in range(50)}
    sp_outcome_map  = {i: sp_out[i]  for i in range(50)}

    _draw_patient_grid(ax, _CL, _R5_YS, tiers, mark_outcome=det_outcome_map, alpha=0.90)
    _draw_patient_grid(ax, _CR, _R5_YS, tiers, mark_outcome=sp_outcome_map,  alpha=0.90)

    # ── Summary banner ────────────────────────────────────────────────────
    sum_y = 2.40
    sum_h = 0.70

    for cx, n_ok, n_rem, n_can, h_can, plan_str, ec in [
            (det_col_cx, det_ok, det_rem, det_can, det_h_can,
             "Deterministic", COLORS["naive_det"]),
            (sp_col_cx,  sp_ok,  sp_rem,  sp_can,  sp_h_can,
             "Stochastic",    _SP_COL)]:
        _rounded_box(ax, cx, sum_y, _CW - 0.20, sum_h,
                     fc="#FAFAFA", ec=ec, lw=1.2)
        ax.text(cx, sum_y + sum_h / 2 - 0.10,
                f"{plan_str} plan — scenario outcome",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=ec)
        ax.text(cx, sum_y - 0.02,
                f"Treated: {n_ok}   Re-mfg: {n_rem}   Cancelled: {n_can}   "
                f"(Tier-H cancelled: {h_can}/10)",
                ha="center", va="center",
                fontsize=8.0, color="#222222")

    # ── Legend ────────────────────────────────────────────────────────────
    leg_y = 1.58
    leg_items = [
        (COLORS["tier_H"], "Tier H — high urgency (×10, β=0.55)"),
        (COLORS["tier_M"], "Tier M — medium urgency (×25, β=0.78)"),
        (COLORS["tier_L"], "Tier L — low urgency (×15, β=0.92)"),
    ]
    ax.text(1.40, leg_y + 0.22,
            "Patient tiers:", ha="left", va="center",
            fontsize=8, fontweight="bold", color="#333333")
    for k, (fc, lbl) in enumerate(leg_items):
        xk = 1.40 + k * 2.40
        ax.add_patch(Circle((xk + 0.12, leg_y - 0.02), 0.10,
                             fc=fc, ec="white", lw=0.5, zorder=4))
        ax.text(xk + 0.30, leg_y - 0.02, lbl,
                ha="left", va="center", fontsize=7.8, color="#333333")

    rec_items = [
        (COLORS["remfg"],  "check",  "Treated (facility succeeded)"),
        (COLORS["remfg"],  "R",      "Re-manufactured (spare capacity used)"),
        (COLORS["cancel"], "cross",  "Cancelled (no spare capacity / batch fail)"),
    ]
    ax.text(1.40, leg_y - 0.42,
            "Recourse markers:", ha="left", va="center",
            fontsize=8, fontweight="bold", color="#333333")
    for k, (fc, sym, lbl) in enumerate(rec_items):
        xk = 1.40 + k * 3.50
        cx_leg = xk + 0.12
        cy_leg = leg_y - 0.68
        ax.add_patch(Circle((cx_leg, cy_leg), 0.10,
                             fc=fc, ec="white", lw=0.5, zorder=4))
        d = 0.052
        if sym == "check":
            ax.plot([cx_leg - d * 0.6, cx_leg - d * 0.05, cx_leg + d * 0.7],
                    [cy_leg - d * 0.0, cy_leg - d * 0.55, cy_leg + d * 0.6],
                    color="white", lw=0.9, zorder=6, solid_capstyle="round")
        elif sym == "R":
            ax.text(cx_leg, cy_leg, "R", ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold", zorder=6)
        elif sym == "cross":
            ax.plot([cx_leg - d, cx_leg + d], [cy_leg + d, cy_leg - d],
                    color="white", lw=1.0, zorder=6, solid_capstyle="round")
            ax.plot([cx_leg - d, cx_leg + d], [cy_leg - d, cy_leg + d],
                    color="white", lw=1.0, zorder=6, solid_capstyle="round")
        ax.text(xk + 0.32, cy_leg, lbl,
                ha="left", va="center", fontsize=7.8, color="#333333")

    # ── Scenario footnote ─────────────────────────────────────────────────
    ax.text(FIG_W / 2, 0.28,
            "Scenario drawn from OOS sample (seed 4242, N=2000).  "
            "Recourse via greedy LP-proxy: remfg at cheapest open facility "
            "with spare capacity, else cancel.",
            ha="center", va="center",
            fontsize=6.5, fontstyle="italic", color="#888888")

    save_figure(fig, "figure14_supply_chain_walkthrough")


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme(det_h_can, sp_h_can, det_can, sp_can):
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    entry = (
        "| `figure14_supply_chain_walkthrough.png` | "
        "Side-by-side five-row supply-chain walkthrough comparing the "
        "Deterministic and Stochastic plans on a representative out-of-sample "
        "scenario (seed 4242). "
        "Row 1 (patient cohort, identical in both columns): 50 patients colour-coded "
        "by urgency tier (H red ×10, M amber ×25, L green ×15). "
        "Row 2 (first-stage plan): factory icons show which facilities are opened — "
        "Det opens m₀ (cap 40) + m₁ (cap 10); SP opens m₀ (cap 10) + m₂ (cap 40, "
        "p=0.95 highest-yield). "
        "Thin tier-coloured lines connect each patient to their assigned facility "
        "(frozen first-stage). "
        "Row 4 (scenario ω): same stochastic realization overlaid on both columns; "
        "red ✕ marks patients who fail at their assigned facility. "
        "Row 5 (recourse): ✓ treated, R re-manufactured (spare capacity used), "
        "✕ cancelled. "
        f"In this scenario Det cancels {det_can} patients ({det_h_can} Tier-H); "
        f"SP cancels {sp_can} ({sp_h_can} Tier-H). "
        "| none (schematic, OOS seed 4242) |\n"
    )

    import re as _re
    cleaned = _re.sub(
        r"\| `figure14_supply_chain[^|]+\| none[^|]+\|\n?", "", existing
    )
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Figure 14 — supply-chain walkthrough ===\n")

    print("[1/3] Loading data …")
    data = _load()

    print("[2/3] Finding illustrative OOS scenario …")
    Y, B, det_out, sp_out = _find_scenario(data)

    n_hc_det = sum(1 for i in range(10) if det_out[i] == "cancel")
    n_hc_sp  = sum(1 for i in range(10) if sp_out[i]  == "cancel")
    n_can_det = sum(o == "cancel" for o in det_out)
    n_can_sp  = sum(o == "cancel" for o in sp_out)
    print(f"  Scenario: Det H-cancel={n_hc_det}, SP H-cancel={n_hc_sp}")
    print(f"           Det total-cancel={n_can_det}, SP total-cancel={n_can_sp}")

    print("[3/3] Generating figure14_supply_chain_walkthrough …")
    generate_figure14(data, Y, B, det_out, sp_out)

    print()
    update_readme(n_hc_det, n_hc_sp, n_can_det, n_can_sp)

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p = _HERE / ext / f"figure14_supply_chain_walkthrough.{ext}"
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}")
        if status != "OK":
            ok = False

    if not ok:
        import sys as _sys
        _sys.exit(1)
    print("\nFigure 14 generated successfully.")


if __name__ == "__main__":
    main()
