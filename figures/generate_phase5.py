"""
figures/generate_phase5.py — Phase 5 figures.

Figure 11  — Per-scenario cost distribution (violins): three side-by-side
             violin plots (one per plan) showing the distribution of per-
             scenario realized total cost across 2,000 OOS scenarios, with
             mean diamond, median line, P5/P95 whiskers, and a mean-reduction
             annotation arrow.

Figure A.1 — Worst-5% stress test (appendix): for each plan, the full
             2,000-scenario cost distribution as a light background with the
             100 worst (5% tail) scenarios highlighted, plus vertical markers
             for overall mean and CVaR-95 (worst-5% mean).

Figure 12  — Alluvial patient-routing chart: tier → primary facility →
             realized outcome, side-by-side for best-case deterministic vs.
             stochastic plan.

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
from scipy.stats import gaussian_kde

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
_PLAN_LABELS = ["Best-case\ndeterministic", "Expected-cost\ndeterministic", "Stochastic\nplan"]
_PLAN_LABELS_SHORT = ["Best-case det.", "Expected-cost det.", "Stochastic plan"]
_PLAN_COLORS = [COLORS["naive_det"], COLORS["expected_cost_det"], COLORS["sp"]]

# Dark versions for visibility on white backgrounds
_PLAN_COLORS_DARK = ["#606060", "#404040", "#1A5090"]

_RNG = np.random.default_rng(seed=17)   # reproducible jitter


# ---------------------------------------------------------------------------
# Figure 11 — Per-scenario cost distribution (violins)
# ---------------------------------------------------------------------------

def fig11_cost_violin() -> None:
    """
    Three violin plots (best-case det., expected-cost det., SP) on a single
    panel.  Violin shape = empirical density; overlaid markers = mean diamond,
    median white line, P5/P95 whiskers.  Annotation arrow shows mean-cost
    reduction from best-case to SP.
    """
    arrays = [
        np.array(OOS["plans"][k]["per_scenario_total_cost"])
        for k in _PLAN_KEYS
    ]

    # Summary stats
    stats = [
        {
            "mean":   float(a.mean()),
            "median": float(np.median(a)),
            "p5":     float(np.percentile(a, 5)),
            "p95":    float(np.percentile(a, 95)),
            "iqr":    float(np.percentile(a, 75) - np.percentile(a, 25)),
        }
        for a in arrays
    ]

    y_lo = min(s["p5"]  for s in stats) * 0.95
    y_hi = max(s["p95"] for s in stats) * 1.05

    fig = single_column()
    fig.set_size_inches(7, 5.5)
    ax  = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.14, right=0.95)

    positions = [0, 1, 2]

    # --- Violin bodies ---
    vp = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.55,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(vp["bodies"], _PLAN_COLORS):
        body.set_facecolor(color)
        body.set_alpha(0.50)
        body.set_edgecolor("none")

    # --- Background scatter (jitter) ---
    for pos, arr, color in zip(positions, arrays, _PLAN_COLORS):
        jitter = _RNG.uniform(-0.12, 0.12, len(arr))
        ax.scatter(
            pos + jitter, arr,
            s=3.5, color=color, alpha=0.07, linewidths=0, zorder=1,
        )

    # --- P5 / P95 whiskers ---
    cap_w = 0.10
    for pos, s in zip(positions, stats):
        ax.vlines(pos, s["p5"], s["p95"], colors="#333333",
                  linewidths=1.2, zorder=4)
        ax.hlines([s["p5"], s["p95"]],
                  [pos - cap_w, pos - cap_w],
                  [pos + cap_w, pos + cap_w],
                  colors="#333333", linewidths=1.2, zorder=4)
        # P5 / P95 text
        ax.text(pos + cap_w + 0.03, s["p5"],
                f"P5: {s['p5']:.2f}", va="center", ha="left",
                fontsize=6.5, color="#555555")
        ax.text(pos + cap_w + 0.03, s["p95"],
                f"P95: {s['p95']:.2f}", va="center", ha="left",
                fontsize=6.5, color="#555555")

    # --- Median line (white, inside violin) ---
    med_w = 0.16
    for pos, s in zip(positions, stats):
        ax.hlines(s["median"], pos - med_w, pos + med_w,
                  colors="white", linewidths=2.0, zorder=6)

    # --- Mean diamond ---
    for pos, s, color in zip(positions, stats, _PLAN_COLORS_DARK):
        ax.scatter(pos, s["mean"], s=100, marker="D",
                   color="black", zorder=7, linewidths=0)

    # --- Annotation arrow: best-case mean → SP mean ---
    naive_mean = stats[0]["mean"]
    sp_mean    = stats[2]["mean"]
    reduction  = naive_mean - sp_mean
    pct        = reduction / naive_mean * 100

    # Arrow mid-height slightly above SP mean
    arrow_y = sp_mean - 0.40
    ax.annotate(
        "",
        xy=(2, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(
            arrowstyle="<->",
            color="#222222",
            lw=1.3,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=8,
    )
    ax.text(
        1.0, arrow_y - 0.55,
        f"Mean cost reduction: {reduction:.2f} M USD (−{pct:.1f}%)",
        ha="center", va="top", fontsize=8, color="#222222",
        fontweight="bold",
    )

    # --- Per-violin sub-labels below x-ticks ---
    sub_labels = []
    for s, lbl in zip(stats, _PLAN_LABELS):
        sub_labels.append(
            f"{lbl}\nMean: {s['mean']:.2f} M USD\n"
            f"P5–P95: {s['p5']:.2f}→{s['p95']:.2f}"
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(sub_labels, fontsize=8.5)

    ax.set_ylabel("Per-scenario realized total cost (M USD)", fontsize=9.5)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.6, 2.6)
    ax.set_title(
        "Per-scenario cost distribution: 2,000 held-out OOS scenarios",
        fontsize=10.5, pad=8,
    )

    # Mean/median legend
    legend_handles = [
        mpatches.Patch(color="none", label=""),   # spacer
        plt.Line2D([0], [0], marker="D", color="black", markersize=7,
                   linewidth=0, label="Mean"),
        plt.Line2D([0], [0], color="white", linewidth=2, label="Median",
                   markerfacecolor="white"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.2,
                   marker="|", markersize=8, label="P5 / P95"),
    ]
    ax.legend(handles=legend_handles[1:], loc="upper right",
              fontsize=8, framealpha=0.92, edgecolor="#CCCCCC")

    save_figure(fig, "figure11_cost_violin")

    # Report
    print("  Figure 11 violin summary:")
    for lbl, s in zip(_PLAN_LABELS_SHORT, stats):
        print(f"    {lbl:30s}: mean={s['mean']:.4f}  median={s['median']:.4f}  "
              f"P5={s['p5']:.4f}  P95={s['p95']:.4f}  IQR={s['iqr']:.4f}")
    print(f"\n  Arrow annotation: reduction={reduction:.4f} M USD  "
          f"pct={pct:.2f}%")


# ---------------------------------------------------------------------------
# Figure A.1 — Worst-5% stress test (appendix)
# ---------------------------------------------------------------------------

def figA1_worst5pct() -> None:
    """
    Three-panel appendix figure.  Each panel shows the full 2,000-scenario
    cost distribution (light KDE background) with the 100 worst (5% tail)
    scenarios highlighted, plus vertical markers for overall mean and CVaR-95.
    """
    arrays = {k: np.array(OOS["plans"][k]["per_scenario_total_cost"])
              for k in _PLAN_KEYS}
    N_tail = 100   # 5 % of 2000

    # Shared x-range
    x_lo = min(a.min() for a in arrays.values()) * 0.98
    x_hi = max(a.max() for a in arrays.values()) * 1.01
    x_eval = np.linspace(x_lo, x_hi, 400)

    fig = triple_column()
    axes = fig.subplots(1, 3, sharey=False)
    fig.subplots_adjust(wspace=0.28, bottom=0.16, top=0.86)

    for ax, key, label, color, dark in zip(
            axes, _PLAN_KEYS, _PLAN_LABELS_SHORT, _PLAN_COLORS, _PLAN_COLORS_DARK):

        arr      = arrays[key]
        worst100 = np.sort(arr)[-N_tail:]
        all_mean = float(arr.mean())
        tail_mean = float(worst100.mean())
        premium   = tail_mean - all_mean
        prem_pct  = premium / all_mean * 100

        # Full distribution KDE (background)
        kde_full = gaussian_kde(arr, bw_method=0.12)
        ax.fill_between(x_eval, kde_full(x_eval),
                        color=color, alpha=0.25, zorder=1)
        ax.plot(x_eval, kde_full(x_eval),
                color=color, linewidth=0.8, alpha=0.5, zorder=2)

        # Worst-5% KDE (foreground)
        kde_tail = gaussian_kde(worst100, bw_method=0.15)
        # Scale to represent 5% of total probability mass
        ax.fill_between(x_eval, kde_tail(x_eval) * 0.05,
                        color=color, alpha=0.80, zorder=3)
        ax.plot(x_eval, kde_tail(x_eval) * 0.05,
                color=dark, linewidth=1.4, zorder=4)

        # Vertical: overall mean
        ax.axvline(all_mean, color="black", linestyle="--",
                   linewidth=1.3, zorder=5)
        ax.text(all_mean, ax.get_ylim()[1] if False else 0,
                f" Mean\n {all_mean:.2f} M",
                color="#222222", fontsize=7, va="bottom",
                transform=ax.get_xaxis_transform(), zorder=6)

        # Vertical: worst-5% mean
        ax.axvline(tail_mean, color="#C0392B", linestyle="--",
                   linewidth=1.3, zorder=5)
        ax.text(tail_mean, 0,
                f" CVaR-95\n {tail_mean:.2f} M",
                color="#C0392B", fontsize=7, va="bottom",
                transform=ax.get_xaxis_transform(), zorder=6)

        # Connecting annotation arrow
        y_arr = kde_full(all_mean).item() * 0.30
        ax.annotate(
            "",
            xy=(tail_mean, y_arr), xytext=(all_mean, y_arr),
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
            zorder=7,
        )
        ax.text((all_mean + tail_mean) / 2, y_arr + 0.002,
                f"+{premium:.2f} M\n(+{prem_pct:.0f}%)",
                ha="center", va="bottom", fontsize=7, color="#C0392B")

        # Callout box
        callout = (f"Mean (all):      {all_mean:.2f} M USD\n"
                   f"Mean (worst 5%): {tail_mean:.2f} M USD\n"
                   f"Tail premium:    {premium:.2f} M USD (+{prem_pct:.0f}%)")
        ax.text(0.97, 0.97, callout, transform=ax.transAxes,
                fontsize=6.5, va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#CCCCCC", alpha=0.93))

        ax.set_title(label, fontsize=9.5, pad=5)
        ax.set_xlabel("Realized total cost (M USD)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(x_lo, x_hi)

    fig.suptitle(
        "Appendix A.1 — Worst-5% tail-risk stress test: full distribution "
        "(light) vs. worst-5% (bold); CVaR-95 in red",
        fontsize=9.5, y=0.97
    )

    save_figure(fig, "figureA1_worst5pct_stress_test")

    print("  Figure A.1 tail metrics:")
    for key, label in zip(_PLAN_KEYS, _PLAN_LABELS_SHORT):
        arr       = arrays[key]
        worst100  = np.sort(arr)[-N_tail:]
        all_mean  = float(arr.mean())
        tail_mean = float(worst100.mean())
        premium   = tail_mean - all_mean
        prem_pct  = premium / all_mean * 100
        print(f"    {label:30s}: overall_mean={all_mean:.4f}  "
              f"CVaR95={tail_mean:.4f}  premium={premium:.4f} (+{prem_pct:.1f}%)")


# ---------------------------------------------------------------------------
# Figure 12 — Alluvial patient-routing chart
# ---------------------------------------------------------------------------

def _soff_fn(n_f: int):
    def _soff(m: int, mp: int) -> int:
        return m * (n_f - 1) + (mp if mp < m else mp - 1)
    return _soff


def _solve_scenario_routing(
    mats: dict,
    x_mat: np.ndarray,
    C_arr: np.ndarray,
    Y_ω: np.ndarray,
    B_ω: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Stage-2 LP for one OOS scenario.
    Returns (fac_success, fac_remfg, fac_sub_from, fac_cancel) each (n_f,).
    """
    n_p = mats["n_p"]
    n_f = mats["n_f"]
    n_vars_pp = mats["n_vars_pp"]
    vc = mats["vc"]

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
    fac_success  = np.zeros(n_f)
    fac_remfg    = np.zeros(n_f)
    fac_sub_from = np.zeros(n_f)
    fac_cancel   = np.zeros(n_f)

    for i in range(n_p):
        m_i = m_assign[i]
        fac_success[m_i]  += float(Y_ω[i, m_i])
        fac_remfg[m_i]    += round(float(sol[vr(i, m_i)]))
        fac_sub_from[m_i] += sum(
            round(float(sol[vs(i, m_i, mp)]))
            for mp in range(n_f) if mp != m_i
        )
        fac_cancel[m_i]   += round(float(sol[vc(i)]))

    return fac_success, fac_remfg, fac_sub_from, fac_cancel


def _compute_alluvial_data(
    inst,
    plan: dict,
    Y_oos: np.ndarray,
    B_oos: np.ndarray,
) -> dict:
    n_p = inst.n_patients
    n_f = inst.n_facilities
    N   = Y_oos.shape[0]
    x_mat = plan["x_mat"]
    C_arr = plan["C"]

    # Tier → facility (deterministic)
    tier_fac = {"H": np.zeros(n_f), "M": np.zeros(n_f), "L": np.zeros(n_f)}
    m_assign = x_mat.argmax(axis=1)
    for i in range(n_p):
        tier_fac[TIERS[i]][m_assign[i]] += 1

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
    print(f"    Routing eval: {elapsed:.1f}s ({elapsed/N*1000:.1f} ms/scen)")

    return {
        "tier_fac":    tier_fac,
        "fac_success": acc_success / N,
        "fac_remfg":   acc_remfg   / N,
        "fac_sub":     acc_sub     / N,
        "fac_cancel":  acc_cancel  / N,
    }


# ---- Alluvial drawing helpers -----------------------------------------------

def _bezier_band(ax, x0, y_src_lo, y_src_hi,
                 x1, y_tgt_lo, y_tgt_hi,
                 color, alpha=0.5) -> None:
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
    ax.add_patch(PathPatch(
        MPath(verts, codes),
        facecolor=color, edgecolor="none", alpha=alpha, zorder=2,
    ))


def _node_box(ax, cx, y_lo, y_hi, color, label,
              node_w=0.06, fontsize=7.5) -> None:
    ax.add_patch(plt.Rectangle(
        (cx - node_w / 2, y_lo), node_w, y_hi - y_lo,
        facecolor=color, edgecolor="white", linewidth=0.6, zorder=3,
    ))
    if (y_hi - y_lo) >= 1.5:
        ax.text(cx, (y_lo + y_hi) / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white", zorder=4, clip_on=True)


def _draw_alluvial_panel(ax, data, title, oos_stats) -> None:
    NODE_W = 0.06
    LX     = [0.0, 0.45, 0.90]
    GAP    = 1.0
    PAD    = 2.0
    n_f    = 4

    tier_names  = ["H", "M", "L"]
    tier_colors = [COLORS["tier_H"], COLORS["tier_M"], COLORS["tier_L"]]
    tier_lbl    = ["Tier H", "Tier M", "Tier L"]
    tier_counts = [len(TIER_IDX["H"]), len(TIER_IDX["M"]), len(TIER_IDX["L"])]
    fac_color   = COLORS["expected_cost_det"]
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

    # Layer 1: tiers (stacked bottom-up: L, M, H)
    l1_pos = {}
    y = PAD
    for t_name, t_count in zip(reversed(tier_names), reversed(tier_counts)):
        l1_pos[t_name] = (y, y + t_count)
        y += t_count + GAP

    # Layer 2: open facilities
    tier_fac   = data["tier_fac"]
    fac_counts = np.array([sum(tier_fac[t][m] for t in tier_names)
                            for m in range(n_f)])
    open_facs  = [m for m in range(n_f) if fac_counts[m] > 0.5]

    l1_mid   = (PAD + y - GAP) / 2
    total_l2 = sum(fac_counts[m] for m in open_facs) + GAP * (len(open_facs) - 1)
    l2_pos = {}
    y = l1_mid - total_l2 / 2
    for m in open_facs:
        l2_pos[m] = (y, y + fac_counts[m])
        y += fac_counts[m] + GAP

    # Layer 3: outcomes
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

    # Draw nodes
    for t_name, _, t_col, t_lbl in zip(tier_names, tier_counts,
                                        tier_colors, tier_lbl):
        y_lo, y_hi = l1_pos[t_name]
        _node_box(ax, LX[0], y_lo, y_hi, t_col, t_lbl, NODE_W, fontsize=7)

    for m in open_facs:
        y_lo, y_hi = l2_pos[m]
        lbl = f"$m_{m}$\n(p={[0.85,0.92,0.95,0.92][m]:.2f})"
        _node_box(ax, LX[1], y_lo, y_hi, fac_color, lbl, NODE_W, fontsize=6.5)

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
                         outcome_colors[o], alpha=0.55)

    # Layer labels
    y_label = l1_mid - total_l2 / 2 - 2.2
    for lx, lbl in zip(LX, ["Urgency tier", "Facility", "Outcome"]):
        ax.text(lx, y_label, lbl, ha="center", va="top",
                fontsize=8, fontweight="bold", color="#333333")

    # OOS cancel-rate caption
    cap = (f"Cancel rates (OOS): "
           f"Tier H={oos_stats['th']:.1f}%  "
           f"Tier M={oos_stats['tm']:.1f}%  "
           f"Tier L={oos_stats['tl']:.1f}%")
    ax.text(0.5, 0.01, cap, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, color="#444444",
            style="italic")

    # Panel styling
    ax.set_xlim(-0.12, 1.05)
    total_y = PAD + sum(tier_counts) + GAP * (len(tier_names) - 1) + PAD
    ax.set_ylim(-3.5, total_y + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)


def fig12_alluvial(inst, plans, Y_oos, B_oos) -> None:
    """
    Two-panel alluvial: Best-case deterministic (left) vs. Stochastic plan (right).
    """
    print("  Computing routing data for Best-case det. plan …")
    naive_data = _compute_alluvial_data(
        inst, plans["naive_deterministic"], Y_oos, B_oos)

    print("  Computing routing data for Stochastic plan …")
    sp_data = _compute_alluvial_data(
        inst, plans["stochastic_plan"], Y_oos, B_oos)

    def _oos_stats(key):
        return {
            "th": float(np.mean(OOS["plans"][key]["per_scenario_tier_H_cancel_rate"])),
            "tm": float(np.mean(OOS["plans"][key]["per_scenario_tier_M_cancel_rate"])),
            "tl": float(np.mean(OOS["plans"][key]["per_scenario_tier_L_cancel_rate"])),
        }

    naive_stats = _oos_stats("naive_deterministic")
    sp_stats    = _oos_stats("stochastic_plan")

    # Report tier-H routing table
    print("\n  Tier-H routing flows:")
    for name, data in [("Best-case det.", naive_data), ("SP", sp_data)]:
        tf = data["tier_fac"]["H"]
        mf = int(tf.argmax())
        print(f"    {name}: H → {tf.tolist()}")
        print(f"    {name}: m{mf} → "
              f"success={data['fac_success'][mf]:.2f}  "
              f"remfg={data['fac_remfg'][mf]:.2f}  "
              f"sub={data['fac_sub'][mf]:.2f}  "
              f"cancel={data['fac_cancel'][mf]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98,
                        top=0.90, bottom=0.08)

    _draw_alluvial_panel(axes[0], naive_data,
                         "(a) Best-case deterministic plan", naive_stats)
    _draw_alluvial_panel(axes[1], sp_data,
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
        fontsize=11, y=0.97,
    )
    save_figure(fig, "figure12_alluvial_patient_routing")


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme() -> None:
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    need_fig11  = "figure11_cost_violin"       not in existing
    need_figA1  = "figureA1_worst5pct"          not in existing
    need_fig12  = "figure12_alluvial"           not in existing

    if not (need_fig11 or need_figA1 or need_fig12):
        print("  README already up to date — skipping.")
        return

    budget = OOS["budget_threshold_M_USD"]
    new_entries = ""

    if need_fig11:
        new_entries += (
            "| `figure11_cost_violin.png` | "
            "Distribution of per-scenario realized total cost across 2,000 held-out "
            "scenarios, shown as violin plots with overlaid mean (black diamond), "
            "median (white line), and P5–P95 whiskers. The stochastic plan's distribution "
            "is shifted left and narrower than the two deterministic baselines, "
            "demonstrating that the value of the stochastic solution manifests in both "
            "lower mean cost and tighter spread — a separate operational claim from cost "
            "reduction that matters for risk-averse decision-makers. "
            "| `results/out_of_sample_results.json` |\n"
        )

    if need_figA1:
        new_entries += (
            "| `figureA1_worst5pct_stress_test.png` | "
            "Tail-risk view (Appendix A.1): the worst-5% of held-out scenarios "
            "highlighted against the full per-scenario cost distribution, per plan. "
            "Black dashed line marks the overall mean; red dashed line marks the "
            "worst-5% mean (the CVaR-95 value). The stochastic plan exhibits both "
            "the lowest overall mean and the lowest worst-5% mean, with the smallest "
            "tail premium — demonstrating that the SP's value extends from average "
            "performance to worst-case robustness. "
            "| `results/out_of_sample_results.json` |\n"
        )

    if need_fig12:
        new_entries += (
            "| `figure12_alluvial_patient_routing.png` | "
            "Alluvial flow of patients from urgency tier (left) through primary facility "
            "assignment (middle) to realized outcome (right), comparing the best-case "
            "deterministic plan (a) and the stochastic plan (b). Flow widths are "
            "proportional to patient counts: layer 1→2 from frozen Stage-1 assignments; "
            "layer 2→3 averaged across 2,000 held-out scenarios. Under the stochastic "
            "plan, tier-H patients are re-routed to a higher-yield facility (m_2), "
            "producing a visibly thinner cancellation slice in the outcome layer. "
            "| `results/out_of_sample_results.json` |\n"
        )

    readme.write_text(existing.rstrip("\n") + "\n" + new_entries)
    print(f"  Updated: {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall = time.perf_counter()
    print("=== Phase 5 figure generation ===\n")

    # ---- Figures 11 and A.1 — no LP re-run needed ----
    print("[1/3] Figure 11 — Per-scenario cost violin")
    fig11_cost_violin()

    print()
    print("[2/3] Figure A.1 — Worst-5% stress test (appendix)")
    figA1_worst5pct()

    # ---- Figure 12 — alluvial needs LP solves ----
    print()
    print("[3/3] Figure 12 — Alluvial patient-routing chart")
    inst  = build_case_study_instance()
    plans = _load_plans(inst)
    print(f"  Generating {N_OOS} OOS scenarios (seed {SEED_OOS}) …")
    Y_oos, B_oos = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
    fig12_alluvial(inst, plans, Y_oos, B_oos)

    print()
    update_readme()

    # Remove superseded Phase-5 files from previous commit
    for stem, ext in [
        ("figure11_cumulative_cost_fan", "png"),
        ("figure11_cumulative_cost_fan", "pdf"),
        ("figureA1_rank_sorted_cost",    "png"),
        ("figureA1_rank_sorted_cost",    "pdf"),
    ]:
        old = _HERE / ext / f"{stem}.{ext}"
        if old.exists():
            old.unlink()
            print(f"  Removed: {old}")

    print("\nOutput verification:")
    ok = True
    for name in [
        "figure11_cost_violin",
        "figureA1_worst5pct_stress_test",
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
