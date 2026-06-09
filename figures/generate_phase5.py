"""
figures/generate_phase5.py — Phase 5 figures.

Figure 11 — Monte Carlo realization plot: 2,000 OOS per-scenario costs sorted
            by rank for each plan, with mean and P5-P95 band.

Figure 12 — Alluvial patient-routing chart: tier → primary facility → realized
            outcome for naive deterministic plan vs. stochastic plan. All
            tier-H patients are routed to m2 (p=0.95) under the SP but split
            across m0/m1 under the naive plan — the structural protection of
            high-urgency patients is visible without reading any numbers.

Run: python figures/generate_phase5.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))

from figure_style import (
    COLORS, setup_style,
    triple_column, double_column,
    save_figure,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

setup_style()

# Re-use infrastructure from out_of_sample_evaluation
from scipy.optimize import linprog
from out_of_sample_evaluation import (
    _load_plans,
    _build_stage2_matrices,
    N_OOS,
    SEED_OOS,
)
from case_study import build_case_study_instance, TIERS, TIER_IDX
from yield_sp_v1 import sample_scenarios

# ---------------------------------------------------------------------------
# Load OOS results
# ---------------------------------------------------------------------------

with open(_REPO / "results" / "out_of_sample_results.json") as _f:
    OOS = json.load(_f)

_PLAN_KEYS   = ["naive_deterministic", "expected_cost_deterministic", "stochastic_plan"]
_PLAN_LABELS = ["Naive deterministic", "Expected-cost deterministic", "Stochastic plan"]
_PLAN_COLORS = [COLORS["naive_det"], COLORS["expected_cost_det"], COLORS["sp"]]


# ---------------------------------------------------------------------------
# Figure 11 — Monte Carlo realization plot
# ---------------------------------------------------------------------------

def fig11_monte_carlo() -> None:
    """
    Three sorted-rank scatter panels with shared y-axis.
    Each panel: sorted per-scenario costs + mean line + P5–P95 shaded band.
    """
    # Gather data
    all_data = {}
    for key in _PLAN_KEYS:
        arr = np.sort(np.array(OOS["plans"][key]["per_scenario_total_cost"]))
        all_data[key] = arr

    # Shared y-limits
    y_lo = min(np.percentile(d, 5)  for d in all_data.values()) * 0.97
    y_hi = max(np.percentile(d, 95) for d in all_data.values()) * 1.05

    fig = triple_column()
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0.08, bottom=0.16, top=0.88)

    for ax, key, label, color in zip(axes, _PLAN_KEYS, _PLAN_LABELS, _PLAN_COLORS):
        arr   = all_data[key]
        N     = len(arr)
        ranks = np.arange(1, N + 1)
        mean  = arr.mean()
        p5    = np.percentile(arr, 5)
        p95   = np.percentile(arr, 95)
        spread = p95 - p5

        # P5–P95 shaded band
        ax.axhspan(p5, p95, color=color, alpha=0.12, zorder=1)
        # P5 and P95 lines
        ax.axhline(p5,  color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(p95, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        # Edge labels P5 / P95
        ax.text(N * 1.01, p5,  "P5",  va="center", ha="left", fontsize=7,
                color=color, alpha=0.85)
        ax.text(N * 1.01, p95, "P95", va="center", ha="left", fontsize=7,
                color=color, alpha=0.85)

        # Scatter — individual scenario costs
        ax.scatter(ranks, arr, s=6, color=color, alpha=0.22, linewidths=0, zorder=2)

        # Mean line
        ax.axhline(mean, color="#111111", linewidth=1.2, linestyle="--", zorder=4)
        ax.text(N * 0.99, mean + (p95 - mean) * 0.12,
                f"Mean\n{mean:.2f} M", va="bottom", ha="right",
                fontsize=7.5, color="#111111", fontweight="bold")

        # Callout box (upper-left)
        callout = (
            f"Mean:  {mean:.2f} M USD\n"
            f"P5:      {p5:.2f} M USD\n"
            f"P95:    {p95:.2f} M USD\n"
            f"Spread: {spread:.2f} M USD"
        )
        ax.text(0.03, 0.97, callout, transform=ax.transAxes,
                fontsize=7, va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#CCCCCC", alpha=0.92))

        ax.set_title(label, fontsize=9.5, pad=5)
        ax.set_xlabel("Scenario rank", fontsize=9)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(0, N * 1.08)
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_ylabel("Realized total cost (M USD)", fontsize=9)

    fig.suptitle(
        "Per-scenario realized cost: 2,000 held-out scenarios sorted by rank",
        fontsize=10.5, y=0.97
    )
    save_figure(fig, "figure11_monte_carlo_realizations")

    # Report callout values
    print("  Figure 11 callout values:")
    for key, label in zip(_PLAN_KEYS, _PLAN_LABELS):
        arr   = all_data[key]
        mean  = arr.mean()
        p5    = np.percentile(arr, 5)
        p95   = np.percentile(arr, 95)
        print(f"    {label:32s}: mean={mean:.4f}  P5={p5:.4f}  "
              f"P95={p95:.4f}  spread={p95-p5:.4f}")


# ---------------------------------------------------------------------------
# Figure 12 — Alluvial patient-routing chart
# ---------------------------------------------------------------------------

# ---- Mini OOS evaluation for per-facility routing data ----

def _solve_scenario_routing(mats: dict, x_mat: np.ndarray,
                             C_arr: np.ndarray,
                             Y_ω: np.ndarray, B_ω: np.ndarray):
    """
    Solve per-scenario Stage-2 LP and return per-facility outcome counts.
    Returns: fac_success, fac_remfg, fac_sub_from, fac_cancel  (each shape n_f)
    """
    n_p = mats["n_p"]
    n_f = mats["n_f"]
    n_vars_pp = mats["n_vars_pp"]
    n_sub_pp  = n_f * (n_f - 1)
    vc  = mats["vc"]

    # Reconstruct index helpers (not stored in mats to keep it lightweight)
    def _soff(m: int, mp: int) -> int:
        return m * (n_f - 1) + (mp if mp < m else mp - 1)
    def vr(i: int, m: int) -> int:
        return i * n_vars_pp + m
    def vs(i: int, m: int, mp: int) -> int:
        return i * n_vars_pp + n_f + _soff(m, mp)

    m_assign = x_mat.argmax(axis=1)

    b_eq = 1.0 - Y_ω[np.arange(n_p), m_assign]
    b_ub = mats["b_ub_fixed"].copy()
    b_ub[:n_p] = B_ω.astype(float)
    rs = mats["row_cap_start"]
    for m in range(n_f):
        mask = (m_assign == m)
        b_ub[rs + m] = float(C_arr[m]) - float(Y_ω[mask, m].sum())

    bounds = [(0.0, 1.0)] * mats["n_vars"]
    res = linprog(mats["c_obj"],
                  A_ub=mats["A_ub"], b_ub=b_ub,
                  A_eq=mats["A_eq"], b_eq=b_eq,
                  bounds=bounds, method="highs-ds")

    if res.status != 0:
        raise RuntimeError(f"LP failed (status {res.status}): {res.message}")

    sol = res.x
    fac_success  = np.zeros(n_f)
    fac_remfg    = np.zeros(n_f)
    fac_sub_from = np.zeros(n_f)
    fac_cancel   = np.zeros(n_f)

    for i in range(n_p):
        m_i = m_assign[i]
        fac_success[m_i]  += Y_ω[i, m_i]
        fac_remfg[m_i]    += round(sol[vr(i, m_i)])
        fac_sub_from[m_i] += sum(round(sol[vs(i, m_i, mp)])
                                  for mp in range(n_f) if mp != m_i)
        fac_cancel[m_i]   += round(sol[vc(i)])

    return fac_success, fac_remfg, fac_sub_from, fac_cancel


def _compute_alluvial_data(inst, plan: dict, Y_oos: np.ndarray, B_oos: np.ndarray):
    """
    Compute tier→facility counts (static) and facility→outcome averages (OOS).
    """
    n_p = inst.n_patients
    n_f = inst.n_facilities
    N   = Y_oos.shape[0]
    x_mat = plan["x_mat"]
    C_arr = plan["C"]

    # Tier → facility (deterministic from x_mat)
    tier_fac = {"H": np.zeros(n_f), "M": np.zeros(n_f), "L": np.zeros(n_f)}
    for i in range(n_p):
        m_i = x_mat[i].argmax()
        tier_fac[TIERS[i]][m_i] += 1

    # Facility → outcome (OOS averages)
    acc_success  = np.zeros(n_f)
    acc_remfg    = np.zeros(n_f)
    acc_sub      = np.zeros(n_f)
    acc_cancel   = np.zeros(n_f)

    mats = _build_stage2_matrices(inst, x_mat)
    t0   = time.perf_counter()
    for ω in range(N):
        s, r, sb, c = _solve_scenario_routing(
            mats, x_mat, C_arr, Y_oos[ω], B_oos[ω])
        acc_success += s
        acc_remfg   += r
        acc_sub     += sb
        acc_cancel  += c
    elapsed = time.perf_counter() - t0
    print(f"    Routing eval: {elapsed:.1f}s ({elapsed/N*1000:.1f}ms/scen)")

    return {
        "tier_fac":    tier_fac,
        "fac_success": acc_success / N,
        "fac_remfg":   acc_remfg   / N,
        "fac_sub":     acc_sub     / N,
        "fac_cancel":  acc_cancel  / N,
        "x_mat":       x_mat,
        "C":           C_arr,
        "z":           plan["z"],
    }


# ---- Alluvial drawing helpers ----

def _bezier_band(ax, x0: float, y_src_lo: float, y_src_hi: float,
                 x1: float, y_tgt_lo: float, y_tgt_hi: float,
                 color: str, alpha: float = 0.5) -> None:
    """Draw a filled bezier ribbon connecting two vertical intervals."""
    ctrl_x = (x0 + x1) / 2
    verts = [
        (x0, y_src_lo),
        (ctrl_x, y_src_lo), (ctrl_x, y_tgt_lo), (x1, y_tgt_lo),
        (x1, y_tgt_hi),
        (ctrl_x, y_tgt_hi), (ctrl_x, y_src_hi), (x0, y_src_hi),
        (x0, y_src_lo),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    patch = PathPatch(MPath(verts, codes),
                      facecolor=color, edgecolor="none", alpha=alpha, zorder=2)
    ax.add_patch(patch)


def _node_box(ax, cx: float, y_lo: float, y_hi: float,
              color: str, label: str, node_w: float = 0.06,
              fontsize: float = 7.5, text_outside: str | None = None) -> None:
    """Draw a filled rectangle node with centred label."""
    rect = plt.Rectangle(
        (cx - node_w / 2, y_lo), node_w, y_hi - y_lo,
        facecolor=color, edgecolor="white", linewidth=0.6, zorder=3
    )
    ax.add_patch(rect)
    height = y_hi - y_lo
    # Inside label only when tall enough
    if height >= 1.5:
        ax.text(cx, (y_lo + y_hi) / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white", zorder=4, clip_on=True)
    if text_outside:
        ax.text(cx, y_hi + 0.3, text_outside,
                ha="center", va="bottom", fontsize=6.5,
                color="#444444", zorder=4)


def _draw_alluvial_panel(ax, data: dict, title: str,
                         oos_stats: dict) -> None:
    """
    Draw a full alluvial panel (3 layers: tier → facility → outcome).

    Layer x-positions (in data units):
      Layer 1 (tiers):     x = 0.0
      Layer 2 (facilities): x = 0.45
      Layer 3 (outcomes):  x = 0.90
    Node width = 0.06; flows connect right edge → left edge.
    Y-axis: patient counts (0–50); gaps between nodes = 1.0.
    """
    NODE_W = 0.06
    LX     = [0.0, 0.45, 0.90]   # layer center x
    GAP    = 1.0                  # vertical gap between nodes in same layer
    PAD    = 2.0                  # top/bottom padding

    n_f = 4
    tier_names   = ["H", "M", "L"]
    tier_colors  = [COLORS["tier_H"], COLORS["tier_M"], COLORS["tier_L"]]
    tier_lbl     = ["Tier H", "Tier M", "Tier L"]
    tier_counts  = [len(TIER_IDX["H"]), len(TIER_IDX["M"]), len(TIER_IDX["L"])]

    fac_color    = COLORS["expected_cost_det"]   # medium gray for facility nodes
    outcome_colors = {
        "success": COLORS["sp"],
        "remfg":   COLORS["remfg"],
        "sub":     COLORS["subcontract"],
        "cancel":  COLORS["cancel"],
    }
    outcome_labels = {
        "success": "Success",
        "remfg":   "Re-mfg",
        "sub":     "Sub-\ncontract",
        "cancel":  "Cancel",
    }

    # ---- Layer 1: tier nodes ----
    # Stack: H top, M middle, L bottom (top of y-axis = high urgency)
    # Build from top down
    l1_nodes = list(zip(tier_names, tier_counts, tier_colors, tier_lbl))
    l1_pos = {}   # tier → (y_lo, y_hi)
    y = PAD
    # Stack bottom-up: L, M, H so H ends at top
    for t_name, t_count, t_col, t_lbl in reversed(l1_nodes):
        l1_pos[t_name] = (y, y + t_count)
        y += t_count + GAP

    # ---- Layer 2: facility nodes ----
    # Only open (non-zero) facilities
    tier_fac = data["tier_fac"]
    fac_counts = np.array([sum(tier_fac[t][m] for t in tier_names) for m in range(n_f)])
    open_facs  = [m for m in range(n_f) if fac_counts[m] > 0.5]

    l2_pos = {}   # m → (y_lo, y_hi)
    # Align vertically with layer 1: center on same midpoint
    l1_mid  = (PAD + y - GAP) / 2
    total_l2 = sum(fac_counts[m] for m in open_facs) + GAP * (len(open_facs) - 1)
    y2_start = l1_mid - total_l2 / 2
    y = y2_start
    for m in open_facs:
        l2_pos[m] = (y, y + fac_counts[m])
        y += fac_counts[m] + GAP

    # ---- Layer 3: outcome nodes ----
    outcome_names = ["success", "remfg", "sub", "cancel"]
    outcome_totals = {
        "success": sum(data["fac_success"]),
        "remfg":   sum(data["fac_remfg"]),
        "sub":     sum(data["fac_sub"]),
        "cancel":  sum(data["fac_cancel"]),
    }
    # Only non-tiny outcomes
    open_outcomes = [o for o in outcome_names if outcome_totals[o] > 0.02]

    l3_pos = {}
    total_l3 = (sum(outcome_totals[o] for o in open_outcomes)
                + GAP * (len(open_outcomes) - 1))
    y3_start = l1_mid - total_l3 / 2
    y = y3_start
    for o in open_outcomes:
        l3_pos[o] = (y, y + outcome_totals[o])
        y += outcome_totals[o] + GAP

    # ---- Draw nodes ----
    # Layer 1
    for t_name, _, t_col, t_lbl in l1_nodes:
        y_lo, y_hi = l1_pos[t_name]
        n_txt = f"{int(tier_counts[tier_names.index(t_name)])} pts"
        _node_box(ax, LX[0], y_lo, y_hi, t_col, t_lbl, NODE_W, fontsize=7)

    # Layer 2
    for m in open_facs:
        y_lo, y_hi = l2_pos[m]
        lbl = f"$m_{m}$\n(p={[0.85,0.92,0.95,0.92][m]:.2f})"
        _node_box(ax, LX[1], y_lo, y_hi, fac_color, lbl, NODE_W, fontsize=6.5)

    # Layer 3
    for o in open_outcomes:
        y_lo, y_hi = l3_pos[o]
        n_txt = f"{outcome_totals[o]:.1f}"
        _node_box(ax, LX[2], y_lo, y_hi, outcome_colors[o],
                  outcome_labels[o], NODE_W, fontsize=6.5)

    # ---- Flows: Layer 1 → Layer 2 ----
    # Stacking order within facility node: L at bottom, M middle, H at top
    # Stacking order within tier node: by facility index (low at bottom)
    l1_cursor = {t: l1_pos[t][0] for t in tier_names}     # current bottom of remaining outflow
    l2_cursor_in = {m: l2_pos[m][0] for m in open_facs}   # current bottom of incoming fill

    # Iterate in order that fills facility nodes from bottom: L, M, H
    for t_name in ["L", "M", "H"]:
        t_color = tier_colors[tier_names.index(t_name)]
        for m in open_facs:
            w = tier_fac[t_name][m]
            if w < 0.05:
                continue
            src_lo = l1_cursor[t_name]
            src_hi = src_lo + w
            l1_cursor[t_name] = src_hi

            tgt_lo = l2_cursor_in[m]
            tgt_hi = tgt_lo + w
            l2_cursor_in[m] = tgt_hi

            _bezier_band(ax,
                         LX[0] + NODE_W / 2, src_lo, src_hi,
                         LX[1] - NODE_W / 2, tgt_lo, tgt_hi,
                         t_color, alpha=0.50)

    # ---- Flows: Layer 2 → Layer 3 ----
    l2_cursor_out = {m: l2_pos[m][0] for m in open_facs}
    l3_cursor_in  = {o: l3_pos[o][0] for o in open_outcomes}

    # Map outcome → per-facility contribution
    outcome_per_fac = {
        "success": data["fac_success"],
        "remfg":   data["fac_remfg"],
        "sub":     data["fac_sub"],
        "cancel":  data["fac_cancel"],
    }

    for m in open_facs:
        for o in open_outcomes:
            w = outcome_per_fac[o][m]
            if w < 0.02:
                continue
            src_lo = l2_cursor_out[m]
            src_hi = src_lo + w
            l2_cursor_out[m] = src_hi

            tgt_lo = l3_cursor_in[o]
            tgt_hi = tgt_lo + w
            l3_cursor_in[o] = tgt_hi

            _bezier_band(ax,
                         LX[1] + NODE_W / 2, src_lo, src_hi,
                         LX[2] - NODE_W / 2, tgt_lo, tgt_hi,
                         outcome_colors[o], alpha=0.48)

    # ---- Layer labels ----
    y_label = y3_start - 2.2
    for lx, lbl in zip(LX, ["Urgency tier", "Facility", "Outcome"]):
        ax.text(lx, y_label, lbl, ha="center", va="top",
                fontsize=8, fontweight="bold", color="#333333")

    # ---- OOS statistics caption ----
    stats = oos_stats
    cap = (f"Cancel rates (OOS): "
           f"Tier H={stats['th']:.1f}%  "
           f"Tier M={stats['tm']:.1f}%  "
           f"Tier L={stats['tl']:.1f}%")
    ax.text(0.5, 0.01, cap, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, color="#444444",
            style="italic")

    # ---- Panel styling ----
    ax.set_xlim(-0.12, 1.05)
    total_y = PAD + sum(tier_counts) + GAP * (len(tier_names) - 1) + PAD
    ax.set_ylim(-3.5, total_y + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)


def fig12_alluvial(inst, plans: dict, Y_oos: np.ndarray, B_oos: np.ndarray) -> None:
    """
    Two-panel alluvial diagram: naive deterministic (left) vs. stochastic plan (right).
    """
    print("  Computing alluvial routing data for Naive plan …")
    naive_data = _compute_alluvial_data(inst, plans["naive_deterministic"], Y_oos, B_oos)

    print("  Computing alluvial routing data for Stochastic plan …")
    sp_data = _compute_alluvial_data(inst, plans["stochastic_plan"], Y_oos, B_oos)

    # OOS cancel stats per plan (from stored results)
    def _oos_stats(key):
        th = np.mean(OOS["plans"][key]["per_scenario_tier_H_cancel_rate"])
        tm = np.mean(OOS["plans"][key]["per_scenario_tier_M_cancel_rate"])
        tl = np.mean(OOS["plans"][key]["per_scenario_tier_L_cancel_rate"])
        return {"th": th, "tm": tm, "tl": tl}

    naive_stats = _oos_stats("naive_deterministic")
    sp_stats    = _oos_stats("stochastic_plan")

    # Print routing table for report
    print("\n  Tier-H routing flows:")
    for plan_name, data, stats in [
        ("Naive", naive_data, naive_stats),
        ("SP",    sp_data,    sp_stats),
    ]:
        tf = data["tier_fac"]["H"]
        main_fac = tf.argmax()
        print(f"    {plan_name}: H→ fac assignment {tf.tolist()}")
        # Facility→outcome for tier-H patients' primary facility
        fo_success = data["fac_success"][main_fac]
        fo_remfg   = data["fac_remfg"][main_fac]
        fo_sub     = data["fac_sub"][main_fac]
        fo_cancel  = data["fac_cancel"][main_fac]
        print(f"    {plan_name}: m{main_fac}→ "
              f"success={fo_success:.2f}  remfg={fo_remfg:.2f}  "
              f"sub={fo_sub:.2f}  cancel={fo_cancel:.2f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98,
                        top=0.90, bottom=0.08)

    _draw_alluvial_panel(axes[0], naive_data,
                         "(a) Naive deterministic plan", naive_stats)
    _draw_alluvial_panel(axes[1], sp_data,
                         "(b) Stochastic plan", sp_stats)

    # Legend for tier colors
    tier_patches = [
        mpatches.Patch(color=COLORS["tier_H"], label="Tier H (high urgency)"),
        mpatches.Patch(color=COLORS["tier_M"], label="Tier M (medium urgency)"),
        mpatches.Patch(color=COLORS["tier_L"], label="Tier L (low urgency)"),
    ]
    outcome_patches = [
        mpatches.Patch(color=COLORS["sp"],          label="Successful primary"),
        mpatches.Patch(color=COLORS["remfg"],        label="Re-manufactured"),
        mpatches.Patch(color=COLORS["subcontract"],  label="Subcontracted"),
        mpatches.Patch(color=COLORS["cancel"],       label="Cancelled"),
    ]
    fig.legend(handles=tier_patches + outcome_patches,
               loc="lower center", ncol=7, fontsize=7.5,
               frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Patient routing: urgency tier → primary facility → realized outcome",
        fontsize=11, y=0.97
    )
    save_figure(fig, "figure12_alluvial_patient_routing")


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme() -> None:
    readme = _HERE / "README.md"
    existing = readme.read_text()

    if "figure11" in existing and "figure12" in existing:
        print("  README already contains figure11/12 entries — skipping.")
        return

    budget = OOS["budget_threshold_M_USD"]

    fig11_entry = (
        "| `figure11_monte_carlo_realizations.png` | "
        "Per-scenario realized total cost across 2,000 held-out scenarios for "
        "each of the three plans, sorted by rank. Light scatter shows individual "
        "scenario realizations; horizontal dashed line marks the mean; shaded band "
        "marks the P5-P95 range. The stochastic plan exhibits both a lower mean "
        "realized cost and a tighter spread, demonstrating that the value of the "
        "stochastic solution manifests in both central tendency and tail-risk reduction. "
        "| `results/out_of_sample_results.json` |"
    )
    fig12_entry = (
        "| `figure12_alluvial_patient_routing.png` | "
        "Alluvial flow of patients from urgency tier (left) through primary facility "
        "assignment (middle) to realized outcome (right), comparing the naive "
        "deterministic plan (a) and the stochastic plan (b). Flow widths proportional "
        "to patient counts: layer 1→2 from frozen Stage-1 assignments; layer 2→3 "
        "averaged across 2,000 held-out scenarios. Under the stochastic plan, all "
        "tier-H patients are routed to the highest-yield facility (m_2, p=0.95), "
        "producing a visibly thinner cancellation slice in the outcome layer. "
        "| `results/out_of_sample_results.json` |"
    )

    updated = existing.rstrip("\n") + "\n" + fig11_entry + "\n" + fig12_entry + "\n"
    readme.write_text(updated)
    print(f"  Updated: {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall = time.perf_counter()
    print("=== Phase 5 figure generation ===\n")

    print("[1/2] Figure 11 — Monte Carlo realization plot")
    fig11_monte_carlo()

    print()
    print("[2/2] Figure 12 — Alluvial patient-routing chart")
    inst  = build_case_study_instance()
    plans = _load_plans(inst)
    print(f"  Generating {N_OOS} OOS scenarios (seed {SEED_OOS}) …")
    Y_oos, B_oos = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
    fig12_alluvial(inst, plans, Y_oos, B_oos)

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for name in ["figure11_monte_carlo_realizations",
                 "figure12_alluvial_patient_routing"]:
        png = _HERE / "png" / f"{name}.png"
        pdf = _HERE / "pdf" / f"{name}.pdf"
        status = "OK" if png.exists() and pdf.exists() else "MISSING"
        print(f"  [{status}] {name}")
        if status != "OK":
            ok = False

    print(f"\nTotal elapsed: {time.perf_counter() - t_wall:.1f}s")
    if not ok:
        sys.exit(1)
    print("Phase 5 figures generated successfully.")


if __name__ == "__main__":
    main()
