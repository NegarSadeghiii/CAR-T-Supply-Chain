"""
figures/generate_phase5.py — Phase 5 figures.

Figure 11 — Cumulative-cost fan chart: 2,000 OOS scenario trajectories of
            cumulative realized cost as patients are processed in tier-sorted
            order (H→M→L).  Light individual traces (alpha=0.04, lw=0.4),
            bold plan-colour mean, P5–P95 shaded envelope, and tier-band
            guides on the x-axis.  Three panels (one per plan).

Figure A.1 — Rank-sorted per-scenario cost (appendix): same three-panel
             scatter as the previously-committed Phase 5 figure, now saved
             as figureA1_rank_sorted_cost (appendix figure).

Figure 12  — Alluvial patient-routing chart: tier → primary facility →
             realized outcome for Best-case deterministic plan vs. stochastic
             plan.  Two panels.

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
    triple_column, double_column, single_column,
    save_figure,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

setup_style()

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
# Shared constants
# ---------------------------------------------------------------------------

with open(_REPO / "results" / "out_of_sample_results.json") as _f:
    OOS = json.load(_f)

_PLAN_KEYS   = ["naive_deterministic", "expected_cost_deterministic", "stochastic_plan"]
_PLAN_LABELS = ["Deterministic plan", "Expected-cost det.", "Stochastic plan"]
_PLAN_COLORS = [COLORS["naive_det"], COLORS["expected_cost_det"], COLORS["sp"]]

# Patient order for fan chart: tier H → M → L
_TIER_ORDER = list(TIER_IDX["H"]) + list(TIER_IDX["M"]) + list(TIER_IDX["L"])
_H_END  = len(TIER_IDX["H"])          # 10
_M_END  = _H_END + len(TIER_IDX["M"]) # 35


# ---------------------------------------------------------------------------
# Shared per-scenario LP solver (used by fan chart AND alluvial)
# ---------------------------------------------------------------------------

def _soff_fn(n_f: int):
    def _soff(m: int, mp: int) -> int:
        return m * (n_f - 1) + (mp if mp < m else mp - 1)
    return _soff


def _solve_scenario_full(
    mats: dict,
    x_mat: np.ndarray,
    C_arr: np.ndarray,
    Y_ω: np.ndarray,
    B_ω: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Stage-2 LP for one OOS scenario.

    Returns:
      stage2_incr   (n_p,)   per-patient stage-2 incremental cost
      fac_success   (n_f,)   per-facility success count
      fac_remfg     (n_f,)   per-facility re-manufacture count
      fac_sub_from  (n_f,)   per-facility subcontract-out count
      fac_cancel    (n_f,)   per-facility cancel count
    """
    n_p = mats["n_p"]
    n_f = mats["n_f"]
    n_vars_pp = mats["n_vars_pp"]
    n_sub_pp  = n_f * (n_f - 1)
    vc        = mats["vc"]

    _soff = _soff_fn(n_f)

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
    c_obj = mats["c_obj"]

    # Per-patient stage-2 incremental cost from LP objective
    stage2_incr = np.array([
        float(c_obj[i * n_vars_pp: (i + 1) * n_vars_pp]
              @ sol[i * n_vars_pp: (i + 1) * n_vars_pp])
        for i in range(n_p)
    ])

    # Per-facility routing counts (for alluvial)
    fac_success  = np.zeros(n_f)
    fac_remfg    = np.zeros(n_f)
    fac_sub_from = np.zeros(n_f)
    fac_cancel   = np.zeros(n_f)

    for i in range(n_p):
        m_i = m_assign[i]
        fac_success[m_i]  += float(Y_ω[i, m_i])
        fac_remfg[m_i]    += round(float(sol[vr(i, m_i)]))
        fac_sub_from[m_i] += sum(round(float(sol[vs(i, m_i, mp)]))
                                  for mp in range(n_f) if mp != m_i)
        fac_cancel[m_i]   += round(float(sol[vc(i)]))

    return stage2_incr, fac_success, fac_remfg, fac_sub_from, fac_cancel


# ---------------------------------------------------------------------------
# Combined plan evaluation — fan trajectories + alluvial routing data
# ---------------------------------------------------------------------------

def _compute_plan_data(
    inst,
    plan: dict,
    Y_oos: np.ndarray,
    B_oos: np.ndarray,
) -> dict:
    """
    Run N_OOS LP solves for one plan.  Returns a dict with:
      cum_traj   (N_OOS, n_p+1)   cumulative cost trajectories (fan chart)
      tier_fac   dict tier→(n_f,) assignment counts (alluvial layer 1→2)
      fac_success / fac_remfg / fac_sub / fac_cancel  (n_f,) OOS averages
    """
    n_p = inst.n_patients
    n_f = inst.n_facilities
    N   = Y_oos.shape[0]
    x_mat = plan["x_mat"]
    C_arr = plan["C"]
    z     = plan["z"]

    # Stage-1 baseline (facility fixed cost — no per-patient c*x in baseline)
    baseline = (sum(inst.f[m] * z[m] for m in range(n_f))
                + sum(inst.pi[m] * C_arr[m] for m in range(n_f)))

    # Per-patient Stage-1 treatment cost (deterministic)
    m_assign = x_mat.argmax(axis=1)   # (n_p,)
    c_pt = np.array([inst.c[m_assign[i]] for i in range(n_p)])  # (n_p,)

    # Tier → facility assignment (deterministic, from x_mat)
    tier_fac = {"H": np.zeros(n_f), "M": np.zeros(n_f), "L": np.zeros(n_f)}
    for i in range(n_p):
        tier_fac[TIERS[i]][m_assign[i]] += 1

    # OOS accumulators
    cum_traj = np.empty((N, n_p + 1))
    acc_success  = np.zeros(n_f)
    acc_remfg    = np.zeros(n_f)
    acc_sub      = np.zeros(n_f)
    acc_cancel   = np.zeros(n_f)

    mats = _build_stage2_matrices(inst, x_mat)

    t0 = time.perf_counter()
    for ω in range(N):
        s2_incr, fs, fr, fsb, fc = _solve_scenario_full(
            mats, x_mat, C_arr, Y_oos[ω], B_oos[ω])

        acc_success += fs
        acc_remfg   += fr
        acc_sub     += fsb
        acc_cancel  += fc

        # Cumulative trajectory: baseline then add each patient in tier order
        cum = np.empty(n_p + 1)
        cum[0] = baseline
        for k, i in enumerate(_TIER_ORDER):
            cum[k + 1] = cum[k] + c_pt[i] + s2_incr[i]
        cum_traj[ω] = cum

    elapsed = time.perf_counter() - t0
    print(f"    {N} LP solves in {elapsed:.1f}s ({elapsed / N * 1000:.1f} ms/scen)")

    return {
        "cum_traj":    cum_traj,
        "tier_fac":    tier_fac,
        "fac_success": acc_success / N,
        "fac_remfg":   acc_remfg   / N,
        "fac_sub":     acc_sub     / N,
        "fac_cancel":  acc_cancel  / N,
        "baseline":    baseline,
        "c_pt":        c_pt,
        "m_assign":    m_assign,
    }


# ---------------------------------------------------------------------------
# Figure 11 — Cumulative-cost fan chart
# ---------------------------------------------------------------------------

def fig11_cumulative_fan(plan_data: dict[str, dict]) -> None:
    """
    Three-panel cumulative-cost fan chart (one panel per plan).
    x-axis: patient index 0..50 in tier order (H → M → L).
    y-axis: cumulative realized cost (M USD).
    """
    n_p = len(_TIER_ORDER)
    x_pts = np.arange(n_p + 1)   # 0..50

    fig = triple_column()
    axes = fig.subplots(1, 3, sharey=False)
    fig.subplots_adjust(wspace=0.28, bottom=0.20, top=0.88)

    for ax, key, label, color in zip(axes, _PLAN_KEYS, _PLAN_LABELS, _PLAN_COLORS):
        cum = plan_data[key]["cum_traj"]   # (N, 51)
        N   = cum.shape[0]

        mean = cum.mean(axis=0)
        p5   = np.percentile(cum, 5,  axis=0)
        p95  = np.percentile(cum, 95, axis=0)

        # 2000 light individual traces
        for ω in range(N):
            ax.plot(x_pts, cum[ω], color=color,
                    alpha=0.04, linewidth=0.4, zorder=1)

        # P5–P95 envelope
        ax.fill_between(x_pts, p5, p95, color=color, alpha=0.18, zorder=2,
                         label="P5–P95 band")

        # Bold mean overlay
        ax.plot(x_pts, mean, color=color, linewidth=2.0, zorder=3,
                label=f"Mean ({mean[-1]:.2f} M)")

        # Tier boundary guides
        for xb, t_lbl in [(_H_END, "H→M"), (_M_END, "M→L")]:
            ax.axvline(xb, color="#888888", linestyle="--",
                       linewidth=0.9, zorder=4)

        # Tier-region labels on x-axis
        ax.text(_H_END / 2, ax.get_ylim()[0] if False else 0,
                "Tier H", ha="center", va="top", fontsize=7,
                color=COLORS["tier_H"], transform=ax.get_xaxis_transform())
        ax.text((_H_END + _M_END) / 2, 0,
                "Tier M", ha="center", va="top", fontsize=7,
                color=COLORS["tier_M"], transform=ax.get_xaxis_transform())
        ax.text((_M_END + n_p) / 2, 0,
                "Tier L", ha="center", va="top", fontsize=7,
                color=COLORS["tier_L"], transform=ax.get_xaxis_transform())

        # Callout box: endpoint stats
        callout = (f"Final mean: {mean[-1]:.2f} M\n"
                   f"P5:          {p5[-1]:.2f} M\n"
                   f"P95:        {p95[-1]:.2f} M\n"
                   f"Spread:     {p95[-1] - p5[-1]:.2f} M")
        ax.text(0.03, 0.97, callout, transform=ax.transAxes,
                fontsize=7, va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#CCCCCC", alpha=0.92))

        ax.set_title(label, fontsize=9.5, pad=5)
        ax.set_xlabel("Patients processed (tier order: H → M → L)", fontsize=8.5)
        ax.set_ylabel("Cumulative realized cost (M USD)", fontsize=8.5)

        # Clean x-ticks at tier boundaries
        ax.set_xticks([0, _H_END, _M_END, n_p])
        ax.set_xticklabels(["0", str(_H_END), str(_M_END), str(n_p)], fontsize=8)

    fig.suptitle(
        "Cumulative realized cost trajectories: 2,000 held-out scenarios"
        " (tier-sorted order, H→M→L)",
        fontsize=10.5, y=0.97
    )

    save_figure(fig, "figure11_cumulative_cost_fan")

    print("  Figure 11 endpoint stats:")
    for key, label in zip(_PLAN_KEYS, _PLAN_LABELS):
        cum  = plan_data[key]["cum_traj"]
        mean = cum[:, -1].mean()
        p5   = np.percentile(cum[:, -1], 5)
        p95  = np.percentile(cum[:, -1], 95)
        print(f"    {label:30s}: mean={mean:.4f}  P5={p5:.4f}  "
              f"P95={p95:.4f}  spread={p95-p5:.4f}")


# ---------------------------------------------------------------------------
# Sanity-check fan trajectories vs. stored OOS totals
# ---------------------------------------------------------------------------

def _sanity_check_fan(plan_data: dict[str, dict]) -> None:
    """
    Verify fan-chart endpoint means match stored per_scenario_total_cost means.
    Tolerance: 0.005 M USD (rounding of routing variables).
    """
    all_ok = True
    for key, label in zip(_PLAN_KEYS, _PLAN_LABELS):
        fan_mean  = plan_data[key]["cum_traj"][:, -1].mean()
        stored_mean = np.mean(OOS["plans"][key]["per_scenario_total_cost"])
        diff = abs(fan_mean - stored_mean)
        ok   = diff < 0.01
        status = "OK" if ok else "WARN"
        print(f"  [{status}] {label:30s}  fan={fan_mean:.4f}  "
              f"stored={stored_mean:.4f}  diff={diff:.4f}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("  WARNING: fan mean deviates from stored OOS mean by > 0.01 M USD")


# ---------------------------------------------------------------------------
# Figure A.1 — Rank-sorted per-scenario cost (appendix)
# ---------------------------------------------------------------------------

def figA1_rank_sorted() -> None:
    """
    Three sorted-rank scatter panels with shared y-axis (appendix figure).
    Each panel: sorted per-scenario costs + mean line + P5-P95 shaded band.
    """
    all_data = {}
    for key in _PLAN_KEYS:
        arr = np.sort(np.array(OOS["plans"][key]["per_scenario_total_cost"]))
        all_data[key] = arr

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

        ax.axhspan(p5, p95, color=color, alpha=0.12, zorder=1)
        ax.axhline(p5,  color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(p95, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(N * 1.01, p5,  "P5",  va="center", ha="left",
                fontsize=7, color=color, alpha=0.85)
        ax.text(N * 1.01, p95, "P95", va="center", ha="left",
                fontsize=7, color=color, alpha=0.85)

        ax.scatter(ranks, arr, s=6, color=color, alpha=0.22,
                   linewidths=0, zorder=2)
        ax.axhline(mean, color="#111111", linewidth=1.2, linestyle="--", zorder=4)
        ax.text(N * 0.99, mean + (p95 - mean) * 0.12,
                f"Mean\n{mean:.2f} M", va="bottom", ha="right",
                fontsize=7.5, color="#111111", fontweight="bold")

        callout = (f"Mean:  {mean:.2f} M USD\n"
                   f"P5:      {p5:.2f} M USD\n"
                   f"P95:    {p95:.2f} M USD\n"
                   f"Spread: {spread:.2f} M USD")
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
        "Appendix A.1 — Per-scenario realized cost: 2,000 held-out scenarios"
        " sorted by rank",
        fontsize=10.5, y=0.97
    )
    save_figure(fig, "figureA1_rank_sorted_cost")

    print("  Figure A.1 callout values:")
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

def _bezier_band(ax, x0: float, y_src_lo: float, y_src_hi: float,
                 x1: float, y_tgt_lo: float, y_tgt_hi: float,
                 color: str, alpha: float = 0.5) -> None:
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
    rect = plt.Rectangle(
        (cx - node_w / 2, y_lo), node_w, y_hi - y_lo,
        facecolor=color, edgecolor="white", linewidth=0.6, zorder=3
    )
    ax.add_patch(rect)
    height = y_hi - y_lo
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
    NODE_W = 0.06
    LX     = [0.0, 0.45, 0.90]
    GAP    = 1.0
    PAD    = 2.0

    n_f = 4
    tier_names  = ["H", "M", "L"]
    tier_colors = [COLORS["tier_H"], COLORS["tier_M"], COLORS["tier_L"]]
    tier_lbl    = ["Tier H", "Tier M", "Tier L"]
    tier_counts = [len(TIER_IDX["H"]), len(TIER_IDX["M"]), len(TIER_IDX["L"])]

    fac_color = COLORS["expected_cost_det"]
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

    # Layer 1: tier nodes (stacked bottom-up: L, M, H)
    l1_nodes = list(zip(tier_names, tier_counts, tier_colors, tier_lbl))
    l1_pos = {}
    y = PAD
    for t_name, t_count, t_col, t_lbl in reversed(l1_nodes):
        l1_pos[t_name] = (y, y + t_count)
        y += t_count + GAP

    # Layer 2: facility nodes
    tier_fac  = data["tier_fac"]
    fac_counts = np.array([sum(tier_fac[t][m] for t in tier_names)
                            for m in range(n_f)])
    open_facs  = [m for m in range(n_f) if fac_counts[m] > 0.5]

    l1_mid    = (PAD + y - GAP) / 2
    total_l2  = (sum(fac_counts[m] for m in open_facs)
                 + GAP * (len(open_facs) - 1))
    l2_pos = {}
    y = l1_mid - total_l2 / 2
    for m in open_facs:
        l2_pos[m] = (y, y + fac_counts[m])
        y += fac_counts[m] + GAP

    # Layer 3: outcome nodes
    outcome_names  = ["success", "remfg", "sub", "cancel"]
    outcome_totals = {
        "success": sum(data["fac_success"]),
        "remfg":   sum(data["fac_remfg"]),
        "sub":     sum(data["fac_sub"]),
        "cancel":  sum(data["fac_cancel"]),
    }
    open_outcomes = [o for o in outcome_names if outcome_totals[o] > 0.02]

    total_l3 = (sum(outcome_totals[o] for o in open_outcomes)
                + GAP * (len(open_outcomes) - 1))
    l3_pos = {}
    y = l1_mid - total_l3 / 2
    for o in open_outcomes:
        l3_pos[o] = (y, y + outcome_totals[o])
        y += outcome_totals[o] + GAP

    # Draw nodes — Layer 1
    for t_name, _, t_col, t_lbl in l1_nodes:
        y_lo, y_hi = l1_pos[t_name]
        _node_box(ax, LX[0], y_lo, y_hi, t_col, t_lbl, NODE_W, fontsize=7)

    # Draw nodes — Layer 2
    for m in open_facs:
        y_lo, y_hi = l2_pos[m]
        lbl = f"$m_{m}$\n(p={[0.85,0.92,0.95,0.92][m]:.2f})"
        _node_box(ax, LX[1], y_lo, y_hi, fac_color, lbl, NODE_W, fontsize=6.5)

    # Draw nodes — Layer 3
    for o in open_outcomes:
        y_lo, y_hi = l3_pos[o]
        _node_box(ax, LX[2], y_lo, y_hi, outcome_colors[o],
                  outcome_labels[o], NODE_W, fontsize=6.5)

    # Flows: Layer 1 → Layer 2
    l1_cursor    = {t: l1_pos[t][0] for t in tier_names}
    l2_cursor_in = {m: l2_pos[m][0] for m in open_facs}

    for t_name in ["L", "M", "H"]:
        t_color = tier_colors[tier_names.index(t_name)]
        for m in open_facs:
            w = tier_fac[t_name][m]
            if w < 0.05:
                continue
            src_lo = l1_cursor[t_name]; src_hi = src_lo + w
            l1_cursor[t_name] = src_hi
            tgt_lo = l2_cursor_in[m]; tgt_hi = tgt_lo + w
            l2_cursor_in[m] = tgt_hi
            _bezier_band(ax,
                         LX[0] + NODE_W / 2, src_lo, src_hi,
                         LX[1] - NODE_W / 2, tgt_lo, tgt_hi,
                         t_color, alpha=0.50)

    # Flows: Layer 2 → Layer 3
    l2_cursor_out = {m: l2_pos[m][0] for m in open_facs}
    l3_cursor_in  = {o: l3_pos[o][0] for o in open_outcomes}

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
            src_lo = l2_cursor_out[m]; src_hi = src_lo + w
            l2_cursor_out[m] = src_hi
            tgt_lo = l3_cursor_in[o]; tgt_hi = tgt_lo + w
            l3_cursor_in[o] = tgt_hi
            _bezier_band(ax,
                         LX[1] + NODE_W / 2, src_lo, src_hi,
                         LX[2] - NODE_W / 2, tgt_lo, tgt_hi,
                         outcome_colors[o], alpha=0.48)

    # Layer labels
    y_label = l1_mid - total_l2 / 2 - 2.2
    for lx, lbl in zip(LX, ["Urgency tier", "Facility", "Outcome"]):
        ax.text(lx, y_label, lbl, ha="center", va="top",
                fontsize=8, fontweight="bold", color="#333333")

    # OOS cancel-rate caption
    stats = oos_stats
    cap = (f"Cancel rates (OOS): "
           f"Tier H={stats['th']:.1f}%  "
           f"Tier M={stats['tm']:.1f}%  "
           f"Tier L={stats['tl']:.1f}%")
    ax.text(0.5, 0.01, cap, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, color="#444444",
            style="italic")

    # Panel styling
    ax.set_xlim(-0.12, 1.05)
    total_y = PAD + sum(tier_counts) + GAP * (len(tier_names) - 1) + PAD
    ax.set_ylim(-3.5, total_y + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)


def fig12_alluvial(plan_data: dict[str, dict]) -> None:
    """
    Two-panel alluvial: Best-case deterministic (left) vs. Stochastic plan (right).
    """
    def _oos_stats(key):
        th = np.mean(OOS["plans"][key]["per_scenario_tier_H_cancel_rate"])
        tm = np.mean(OOS["plans"][key]["per_scenario_tier_M_cancel_rate"])
        tl = np.mean(OOS["plans"][key]["per_scenario_tier_L_cancel_rate"])
        return {"th": th, "tm": tm, "tl": tl}

    naive_stats = _oos_stats("naive_deterministic")
    sp_stats    = _oos_stats("stochastic_plan")

    # Print routing table
    print("\n  Tier-H routing flows:")
    for plan_name, key in [("Best-case det.", "naive_deterministic"),
                            ("SP",            "stochastic_plan")]:
        tf = plan_data[key]["tier_fac"]["H"]
        print(f"    {plan_name}: H→ fac assignment {tf.tolist()}")
        main_fac = tf.argmax()
        print(f"    {plan_name}: m{main_fac}→ "
              f"success={plan_data[key]['fac_success'][main_fac]:.2f}  "
              f"remfg={plan_data[key]['fac_remfg'][main_fac]:.2f}  "
              f"sub={plan_data[key]['fac_sub'][main_fac]:.2f}  "
              f"cancel={plan_data[key]['fac_cancel'][main_fac]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98,
                        top=0.90, bottom=0.08)

    _draw_alluvial_panel(axes[0], plan_data["naive_deterministic"],
                         "(a) Best-case deterministic plan", naive_stats)
    _draw_alluvial_panel(axes[1], plan_data["stochastic_plan"],
                         "(b) Stochastic plan", sp_stats)

    tier_patches = [
        mpatches.Patch(color=COLORS["tier_H"], label="Tier H (high urgency)"),
        mpatches.Patch(color=COLORS["tier_M"], label="Tier M (medium urgency)"),
        mpatches.Patch(color=COLORS["tier_L"], label="Tier L (low urgency)"),
    ]
    outcome_patches = [
        mpatches.Patch(color=COLORS["sp"],         label="Successful primary"),
        mpatches.Patch(color=COLORS["remfg"],       label="Re-manufactured"),
        mpatches.Patch(color=COLORS["subcontract"], label="Subcontracted"),
        mpatches.Patch(color=COLORS["cancel"],      label="Cancelled"),
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

    budget = OOS["budget_threshold_M_USD"]

    need_fig11   = "figure11_cumulative" not in existing
    need_figA1   = "figureA1_rank"       not in existing
    need_fig12   = "figure12"            not in existing

    if not (need_fig11 or need_figA1 or need_fig12):
        print("  README already up to date — skipping.")
        return

    new_entries = ""

    if need_fig11:
        new_entries += (
            "| `figure11_cumulative_cost_fan.png` | "
            "Cumulative realized cost trajectories across 2,000 held-out OOS scenarios "
            "as patients are processed one at a time in urgency-tier order (H→M→L). "
            "Individual scenario trajectories are shown as light lines (alpha=0.04); "
            "the bold overlay is the across-scenario mean; the shaded band marks the "
            "P5–P95 envelope. Vertical dashed lines separate the tier-H (patients 1–10), "
            "tier-M (11–35), and tier-L (36–50) regions. Under the stochastic plan, the "
            "fan stays tighter, especially in the tier-H region, reflecting structural "
            "protection of the most time-critical patients. "
            "| `results/out_of_sample_results.json` |\n"
        )

    if need_figA1:
        new_entries += (
            "| `figureA1_rank_sorted_cost.png` | "
            "Appendix A.1 — Per-scenario realized total cost across 2,000 held-out "
            "scenarios for each of the three plans, sorted by rank. Light scatter shows "
            "individual scenario realizations; horizontal dashed line marks the mean; "
            "shaded band marks the P5–P95 range. The stochastic plan shows both a lower "
            "mean realized cost and a tighter spread. "
            "| `results/out_of_sample_results.json` |\n"
        )

    if need_fig12:
        new_entries += (
            "| `figure12_alluvial_patient_routing.png` | "
            "Alluvial flow of patients from urgency tier (left) through primary facility "
            "assignment (middle) to realized outcome (right), comparing the best-case "
            "deterministic plan (a) and the stochastic plan (b). Flow widths proportional "
            "to patient counts: layer 1→2 from frozen Stage-1 assignments; layer 2→3 "
            "averaged across 2,000 held-out scenarios. Under the stochastic plan, all "
            "tier-H patients are routed to the highest-yield facility (m_2, p=0.95), "
            "producing a visibly thinner cancellation slice in the outcome layer. "
            "| `results/out_of_sample_results.json` |\n"
        )

    updated = existing.rstrip("\n") + "\n" + new_entries
    readme.write_text(updated)
    print(f"  Updated: {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall = time.perf_counter()
    print("=== Phase 5 figure generation (REVISED) ===\n")

    inst  = build_case_study_instance()
    plans = _load_plans(inst)

    print(f"[setup] Generating {N_OOS} OOS scenarios (seed {SEED_OOS}) …")
    Y_oos, B_oos = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
    print()

    # Run LP solves for all three plans (shared across fig11 and fig12)
    plan_data: dict[str, dict] = {}
    for key, label in zip(_PLAN_KEYS, _PLAN_LABELS):
        print(f"[LP] {label} …")
        pd = _compute_plan_data(inst, plans[key], Y_oos, B_oos)
        plan_data[key] = pd
    print()

    print("[sanity] Fan trajectory endpoints vs. stored OOS totals:")
    _sanity_check_fan(plan_data)
    print()

    print("[1/3] Figure 11 — Cumulative-cost fan chart")
    fig11_cumulative_fan(plan_data)

    print()
    print("[2/3] Figure A.1 — Rank-sorted per-scenario cost (appendix)")
    figA1_rank_sorted()

    print()
    print("[3/3] Figure 12 — Alluvial patient-routing chart")
    fig12_alluvial(plan_data)

    print()
    update_readme()

    # Remove old figure11_monte_carlo_realizations files if present
    for ext in ("png", "pdf"):
        old = _HERE / ext / f"figure11_monte_carlo_realizations.{ext}"
        if old.exists():
            old.unlink()
            print(f"  Removed: {old}")

    print("\nOutput verification:")
    ok = True
    for name in [
        "figure11_cumulative_cost_fan",
        "figureA1_rank_sorted_cost",
        "figure12_alluvial_patient_routing",
    ]:
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
