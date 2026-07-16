"""
deterioration_figures.py — figures for the capacity-tight deterioration rerun.

Two buckets:
  * MAIN  -> figures/           plain healthcare language ONLY (no technical names).
  * TECH  -> figures/technical/ validation + internal-parameter curves.

Called by deterioration_experiment.py (or run standalone on the saved JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_FIG = _HERE / "figures"
_FIGT = _FIG / "technical"
sys.path.insert(0, str(_FIG))
from figure_style import COLORS, setup_style, double_column, save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Plain-language axis label for the decline-speed knob and the calibrated marker.
DECLINE_AXIS = "How fast patients decline while waiting"
CALIB_LABEL = "matched to real survival data"

ON_TIME = COLORS["cancel"]        # plan assuming manufacturing succeeds on time
DECLINE_AWARE = COLORS["remfg"]   # plan accounting for patient decline
NEUTRAL = COLORS["sp"]


def _save_main(fig, name):
    save_figure(fig, name)        # figures/png, figures/pdf


def _save_tech(fig, name):
    # Temporarily point save_figure at figures/technical via its _BASE_DIR.
    import figure_style as fs
    (_FIGT / "png").mkdir(parents=True, exist_ok=True)
    (_FIGT / "pdf").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = _FIGT / ext / f"{name}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def _mark_calibrated(ax, x=1.0):
    ax.axvline(x, color="#444", ls="--", lw=1.2)
    ymin, ymax = ax.get_ylim()
    ax.annotate(f"real decline rate\n({CALIB_LABEL})", xy=(x, ymin + 0.06 * (ymax - ymin)),
                xytext=(x + 0.12, ymin + 0.18 * (ymax - ymin)), fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1))


# ===========================================================================
# MAIN figures (plain language)
# ===========================================================================

def fig_high_urgency_deaths(sweep, tag=""):
    rows = sorted(sweep["rows"], key=lambda r: r["kappa"])
    k = [r["kappa"] for r in rows]
    true_on_time = [r["on_time_plan"]["high_urgency_deaths"] for r in rows]
    decline_aware = [r["decline_aware_plan"]["high_urgency_deaths"] for r in rows]
    assumed = sweep["naive_predicted"]["high_urgency_deaths"]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    ax.axhline(assumed, color="#888", ls=":", lw=1.4,
               label="what the on-time plan assumes")
    ax.plot(k, true_on_time, "-o", color=ON_TIME, ms=4,
            label="on-time plan (what actually happens)")
    ax.plot(k, decline_aware, "-s", color=DECLINE_AWARE, ms=4,
            label="decline-aware plan")
    ax.set_xlabel(DECLINE_AXIS)
    ax.set_ylabel("High-urgency patients lost per cohort")
    ax.set_title("High-urgency patients lost while waiting for treatment", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    _mark_calibrated(ax)
    fig.tight_layout(); _save_main(fig, f"figureH1_high_urgency_lost{tag}")


def fig_cost_per_patient(sweep, tag=""):
    rows = sorted(sweep["rows"], key=lambda r: r["kappa"])
    k = [r["kappa"] for r in rows]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    ax.plot(k, [r["on_time_plan"]["cost_per_treated"] for r in rows], "-o",
            color=ON_TIME, ms=4, label="on-time plan")
    ax.plot(k, [r["decline_aware_plan"]["cost_per_treated"] for r in rows], "-s",
            color=DECLINE_AWARE, ms=4, label="decline-aware plan")
    ax.set_xlabel(DECLINE_AXIS)
    ax.set_ylabel("Cost per patient treated (M USD)")
    ax.set_title("Cost per patient treated", fontweight="bold")
    ax.legend(fontsize=8)
    _mark_calibrated(ax)
    fig.tight_layout(); _save_main(fig, f"figureH2_cost_per_patient{tag}")


def fig_treated_in_time(sweep, calib_kappa=1.0, tag=""):
    # Bar chart at the real decline rate: share treated in time by urgency tier.
    row = min(sweep["rows"], key=lambda r: abs(r["kappa"] - calib_kappa))
    tiers = ["H", "M", "L"]
    names = {"H": "High urgency", "M": "Medium urgency", "L": "Low urgency"}
    ot = [row["on_time_plan"]["treated_share_by_tier"][t] * 100 for t in tiers]
    da = [row["decline_aware_plan"]["treated_share_by_tier"][t] * 100 for t in tiers]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    xpos = np.arange(len(tiers)); w = 0.38
    ax.bar(xpos - w / 2, ot, w, color=ON_TIME, label="on-time plan")
    ax.bar(xpos + w / 2, da, w, color=DECLINE_AWARE, label="decline-aware plan")
    ax.set_xticks(xpos); ax.set_xticklabels([names[t] for t in tiers])
    ax.set_ylabel("Patients treated within clinical window (%)")
    ax.set_title("Share of patients treated in time, by urgency (real decline rate)",
                 fontweight="bold")
    ax.set_ylim(0, 100); ax.legend(fontsize=8)
    fig.tight_layout(); _save_main(fig, f"figureH3_treated_in_time{tag}")


def fig_capacity_cap(cap_sweep):
    rows = sorted(cap_sweep["rows"], key=lambda r: r["s_max"])
    smax = [r["s_max"] for r in rows]
    ot = [r["on_time_plan"]["high_urgency_deaths"] for r in rows]
    da = [r["decline_aware_plan"]["high_urgency_deaths"] for r in rows]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    xpos = np.arange(len(smax)); w = 0.38
    ax.bar(xpos - w / 2, ot, w, color=ON_TIME, label="on-time plan")
    ax.bar(xpos + w / 2, da, w, color=DECLINE_AWARE, label="decline-aware plan")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{s} slots{' (artificial cap)' if s == 40 else ' (real limit)' if s == 75 else ''}"
                        for s in smax])
    ax.set_xlabel("Capacity limit per facility")
    ax.set_ylabel("High-urgency patients lost per cohort")
    ax.set_title("Whether the best facility can grow drives high-urgency survival",
                 fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); _save_main(fig, "figureH4_capacity_cap_effect")


def fig_network_decision(sweep, calib_kappa=1.0):
    row = min(sweep["rows"], key=lambda r: abs(r["kappa"] - calib_kappa))
    plans = [("on-time plan", row["on_time_plan"], ON_TIME),
             ("decline-aware plan", row["decline_aware_plan"], DECLINE_AWARE)]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    xpos = np.arange(len(plans)); w = 0.5
    spare = [p[1]["spare_capacity_built"] for p in plans]
    ax.bar(xpos, spare, w, color=[p[2] for p in plans])
    for i, (nm, m, _) in enumerate(plans):
        ax.annotate(f"{m['facilities_open']} facilities open\n{int(m['spare_capacity_built'])} spare slots",
                    (xpos[i], spare[i]), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels([p[0] for p in plans])
    ax.set_ylabel("Spare capacity built (slots)")
    ax.set_title("Network decision: spare capacity built (real decline rate)", fontweight="bold")
    ax.set_ylim(0, max(spare) * 1.3 + 1)
    fig.tight_layout(); _save_main(fig, "figureH5_spare_capacity_built")


# ===========================================================================
# TECHNICAL figures
# ===========================================================================

def fig_tech_nesting(sweep):
    """decline speed = 0 reproduces the original (exogenous) model exactly."""
    row0 = next(r for r in sweep["rows"] if r["kappa"] == 0.0)
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    labels = ["total cost (M USD)", "high-urgency deaths", "spare slots"]
    exo = [row0["on_time_plan"]["total_cost"], row0["on_time_plan"]["high_urgency_deaths"],
           row0["on_time_plan"]["spare_capacity_built"]]
    endo = [row0["decline_aware_plan"]["total_cost"], row0["decline_aware_plan"]["high_urgency_deaths"],
            row0["decline_aware_plan"]["spare_capacity_built"]]
    xpos = np.arange(len(labels)); w = 0.38
    ax.bar(xpos - w / 2, exo, w, color=ON_TIME, label="exogenous (v1) design D_exo")
    ax.bar(xpos + w / 2, endo, w, color=DECLINE_AWARE, label="endogenous design D_endo (kappa=0)")
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_title("Nesting check: kappa=0 => D_endo == D_exo (reproduces v1)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); _save_tech(fig, "figureT1_nesting_kappa0")


def fig_tech_planned_vs_sim(pv):
    pts = pv["points"]
    spare = [p["spare_at_busiest"] for p in pts]
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    ax.plot(spare, [p["planned_delay_death_cost"] for p in pts], "-o", color=NEUTRAL,
            ms=4, label="planning model estimate (surrogate)")
    ax.plot(spare, [p["simulated_delay_cost"] for p in pts], "-s", color=ON_TIME,
            ms=4, label="detailed patient simulation")
    ax.set_xlabel(f"spare capacity added at busiest facility (m{pv['busiest_facility']})")
    ax.set_ylabel("delay-caused cancellation cost (M USD)")
    ax.set_title(f"Planned vs simulated crowding deaths (kappa={pv['kappa']}, gamma={pv['gamma']})",
                 fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); _save_tech(fig, "figureT2_planned_vs_simulated")


def fig_tech_inverted_u(main_sweep, gamma_sweeps):
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    gcolors = {1.0: COLORS["sp"], 1.5: COLORS["subcontract"], 2.0: COLORS["cancel"]}
    for sw in [main_sweep] + list(gamma_sweeps):
        rows = sorted(sw["rows"], key=lambda r: r["kappa"])
        g = sw["gamma"]
        ax.plot([r["kappa"] for r in rows], [r["voe"] for r in rows], "-o",
                color=gcolors.get(g, COLORS["sp"]), ms=4, label=f"gamma={g}")
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.axvline(1.0, color="#444", ls="--", lw=1.0)
    ax.set_xlabel("delay-sensitivity kappa")
    ax.set_ylabel("Value of Endogeneity = cost(D_exo) - cost(D_endo) (M USD)")
    ax.set_title("Internal benefit-vs-decline-speed (inverted-U)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); _save_tech(fig, "figureT3_value_of_endogeneity")


# ===========================================================================
# Driver
# ===========================================================================

def make_all_figures(payload):
    """
    TECHNICAL validation figures only (figures/technical/). The plain-language MAIN
    figures are now produced by causes_priority_figures.py (figureP1-P6), which
    carries the loss-cause split and the sickest-first lever; the earlier H-series
    main figures are superseded and no longer generated here.
    """
    tight = payload["main_tight"]
    low = payload["low_demand"]

    fig_tech_nesting(tight)
    fig_tech_planned_vs_sim(payload["planned_vs_sim"])
    # Inverted-U appears where the design has headroom to build spare (low-demand);
    # the saturated tight setting has VoE==0 by construction.
    fig_tech_inverted_u(low, payload.get("tech_gamma", []))
    print("Technical validation figures written to figures/technical/.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(_RES := _HERE / "results" / "deterioration_results.json"))
    args = ap.parse_args()
    payload = json.loads(Path(args.json).read_text())
    make_all_figures(payload)
