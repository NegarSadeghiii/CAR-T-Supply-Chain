"""
causes_priority_figures.py — plain-language figures for STEP 1 (causes) and
STEP 2 (sickest-first). Main figures -> figures/ (no technical jargon);
any technical figure -> figures/technical/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_FIG = _HERE / "figures"
sys.path.insert(0, str(_FIG))
from figure_style import COLORS, setup_style, double_column, save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

NORMAL_WAIT = COLORS["sp"]        # lost during the normal wait (blue)
AFTER_FAIL = COLORS["cancel"]     # lost after a failure (red)
ON_TIME = COLORS["cancel"]
DECLINE_AWARE = COLORS["subcontract"]
SICKEST = COLORS["remfg"]
CALIB = 1.0
DECLINE_AXIS = "How fast patients decline while waiting"


def _row_at(sweep, kappa):
    return min(sweep["rows"], key=lambda r: abs(r["kappa"] - kappa))


def fig_causes(payload):
    """STEP 1 headline: why high-urgency patients are lost (normal wait vs failure)."""
    busy = _row_at(payload["busy"], CALIB)["on_time"]
    low = _row_at(payload["low"], CALIB)["on_time"]
    settings = [("Busy network\n(150 patients)", busy), ("Low-demand\n(50 patients)", low)]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    xpos = np.arange(len(settings))
    a = [s[1]["cause_a_by_tier"]["H"] for s in settings]
    b = [s[1]["cause_b_by_tier"]["H"] for s in settings]
    ax.bar(xpos, a, 0.5, color=NORMAL_WAIT, label="lost during the normal wait (cells were made)")
    ax.bar(xpos, b, 0.5, bottom=a, color=AFTER_FAIL, label="lost after a manufacturing failure")
    tops = [a[i] + b[i] for i in range(len(settings))]
    ax.set_ylim(top=max(tops) * 1.35)
    for i, s in enumerate(settings):
        tot = tops[i]
        ax.annotate(f"{100*a[i]/tot:.0f}% from\nthe wait", (xpos[i], tot), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels([s[0] for s in settings])
    ax.set_ylabel("High-urgency patients lost per cohort")
    ax.set_title("Why high-urgency patients are lost: normal wait vs after a failure",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); save_figure(fig, "figureP1_why_high_urgency_lost")


def fig_causes_by_tier(payload):
    """Context: the same split for medium and low urgency (busy network)."""
    row = _row_at(payload["busy"], CALIB)["on_time"]
    tiers = ["H", "M", "L"]; names = {"H": "High", "M": "Medium", "L": "Low"}
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    xpos = np.arange(len(tiers))
    a = [row["cause_a_by_tier"][t] for t in tiers]
    b = [row["cause_b_by_tier"][t] for t in tiers]
    ax.bar(xpos, a, 0.55, color=NORMAL_WAIT, label="lost during the normal wait")
    ax.bar(xpos, b, 0.55, bottom=a, color=AFTER_FAIL, label="lost after a failure")
    ax.set_xticks(xpos); ax.set_xticklabels([f"{names[t]} urgency" for t in tiers])
    ax.set_ylabel("Patients lost per cohort")
    ax.set_title("Cause of loss by urgency (busy network)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); save_figure(fig, "figureP2_cause_by_urgency")


def fig_three_plans(payload):
    """STEP 2 headline: high-urgency lost under the three plans (full network)."""
    plans = payload["three_plans"]["plans"]
    order = [("On-time plan", "on_time", ON_TIME),
             ("Decline-aware\n(spare capacity)", "decline_aware", DECLINE_AWARE),
             ("Decline-aware\n+ sickest first", "sickest_first", SICKEST)]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    xpos = np.arange(len(order))
    vals = [plans[k]["high_urgency_lost"] for _, k, _ in order]
    ax.bar(xpos, vals, 0.55, color=[c for _, _, c in order])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (xpos[i], v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.set_xticks(xpos); ax.set_xticklabels([n for n, _, _ in order])
    ax.set_ylabel("High-urgency patients lost per cohort")
    ax.set_title("High-urgency patients lost under three plans (full network)", fontweight="bold")
    fig.tight_layout(); save_figure(fig, "figureP3_three_plans")


def fig_tradeoff(payload):
    """STEP 2: the price of prioritizing — high-urgency saved vs lower-urgency delayed."""
    plans = payload["three_plans"]["plans"]
    base, pri = plans["decline_aware"], plans["sickest_first"]
    h_saved = base["high_urgency_lost"] - pri["high_urgency_lost"]
    low_base = base["lost_by_tier"]["M"] + base["lost_by_tier"]["L"]
    low_pri = pri["lost_by_tier"]["M"] + pri["lost_by_tier"]["L"]
    low_extra = low_pri - low_base
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    ax.bar([0], [h_saved], 0.5, color=SICKEST, label="high-urgency patients saved")
    ax.bar([1], [low_extra], 0.5, color=AFTER_FAIL, label="extra lower-urgency patients lost")
    for x, v in [(0, h_saved), (1, low_extra)]:
        ax.annotate(f"{v:+.2f}", (x, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["high-urgency\nsaved", "lower-urgency\nlost"])
    ax.set_ylabel("Patients per cohort")
    ax.set_title("The price of putting the sickest first (full network)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); save_figure(fig, "figureP4_tradeoff")


def fig_decline_sweep(payload):
    """High-urgency lost vs decline speed: on-time vs sickest-first (busy network)."""
    rows = sorted(payload["busy"]["rows"], key=lambda r: r["kappa"])
    k = [r["kappa"] for r in rows]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    ax.plot(k, [r["on_time"]["high_urgency_lost"] for r in rows], "-o", color=ON_TIME, ms=4,
            label="on-time / decline-aware plan")
    ax.plot(k, [r["sickest_first"]["high_urgency_lost"] for r in rows], "-s", color=SICKEST, ms=4,
            label="decline-aware + sickest first")
    ax.axvline(CALIB, color="#444", ls="--", lw=1.1)
    ymax = ax.get_ylim()[1]
    ax.annotate("real decline rate\n(matched to survival data)", xy=(CALIB, 0.15 * ymax),
                xytext=(CALIB + 0.12, 0.30 * ymax), fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.set_xlabel(DECLINE_AXIS); ax.set_ylabel("High-urgency patients lost per cohort")
    ax.set_title("Putting the sickest first protects high-urgency patients", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); save_figure(fig, "figureP5_priority_vs_decline_speed")


def fig_cap(payload):
    """Removing the artificial capacity cap (on-time plan, real rate)."""
    rows = sorted(payload["cap_sweep"]["rows"], key=lambda r: r["s_max"])
    smax = [r["s_max"] for r in rows]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    xpos = np.arange(len(smax))
    ax.bar(xpos, [r["on_time"]["high_urgency_lost"] for r in rows], 0.5, color=ON_TIME)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['on_time']['high_urgency_lost']:.2f}",
                    (xpos[i], r["on_time"]["high_urgency_lost"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{s} slots{' (artificial cap)' if s == 40 else ' (real limit)' if s == 75 else ''}"
                        for s in smax])
    ax.set_xlabel("Capacity limit per facility")
    ax.set_ylabel("High-urgency patients lost per cohort")
    ax.set_title("Removing the artificial capacity cap", fontweight="bold")
    fig.tight_layout(); save_figure(fig, "figureP6_capacity_cap")


def make_all(payload):
    fig_causes(payload)
    fig_causes_by_tier(payload)
    fig_three_plans(payload)
    fig_tradeoff(payload)
    fig_decline_sweep(payload)
    fig_cap(payload)
    print("Main figures written to figures/ (figureP1-P6).")


if __name__ == "__main__":
    p = json.loads((_HERE / "results" / "causes_priority_results.json").read_text())
    make_all(p)
