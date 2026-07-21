"""
paper/src/figures.py — publication figures (F1-F5) and technical/validation
figures, built from the master results payload. All figures use paper_style.

Main figures (paper/figures/): plain patient-and-cost axes only.
  F1  high-urgency patients lost vs decline speed (on-time vs sickest-first),
      with the real-decline-rate marker.
  F2  why high-urgency patients are lost: normal wait vs after a failure,
      busy vs low-demand.
  F3  three-plan comparison at the full network (high-urgency lost + equity).
  F4  capacity-cap sweep.
  F5  robustness of the two headline numbers to the normal-wait mortality anchor.

Technical figures (paper/figures/technical/):
  T1  planning-model estimate vs detailed simulation.
  T2  decline-shape (gamma) sensitivity.
  T3  tier hazard-ratio spread sensitivity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import paper_style as ps
import matplotlib.pyplot as plt

CALIB = 1.0


def _row_at(rows, k):
    return min(rows, key=lambda r: abs(r["kappa"] - k))


# ---------------------------------------------------------------- main figures

def f1_decline_sweep(P, outdir):
    rows = sorted(P["busy"]["rows"], key=lambda r: r["kappa"])
    k = [r["kappa"] for r in rows]
    fig = ps.fig("double", 0.5); ax = fig.add_subplot(111)
    ax.plot(k, [r["on_time"]["high_urgency_lost"] for r in rows], "-o",
            color=ps.C["on_time"], label="on-time / decline-aware plan")
    ax.plot(k, [r["sickest_first"]["high_urgency_lost"] for r in rows], "-s",
            color=ps.C["sickest"], label="decline-aware + sickest first")
    ax.axvline(CALIB, color=ps.C["ref"], ls="--", lw=1.0)
    ymax = ax.get_ylim()[1]
    ax.annotate("real decline rate\n(matched to survival data)", xy=(CALIB, 0.12 * ymax),
                xytext=(CALIB + 0.15, 0.32 * ymax), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=ps.C["ref"], lw=0.8))
    ax.set_xlabel("How fast patients decline while waiting")
    ax.set_ylabel("High-urgency patients lost\nper cohort")
    ax.legend(loc="upper left")
    fig.tight_layout(); ps.save(fig, "F1_high_urgency_lost_vs_decline", outdir)


def f2_causes(P, outdir):
    busy = _row_at(P["busy"]["rows"], CALIB)["on_time"]
    low = _row_at(P["low"]["rows"], CALIB)["on_time"]
    settings = [("Busy network\n(150 patients)", busy), ("Low-demand\n(50 patients)", low)]
    fig = ps.fig("single", 0.85); ax = fig.add_subplot(111)
    xpos = np.arange(len(settings))
    a = [s[1]["cause_a_by_tier"]["H"] for s in settings]
    b = [s[1]["cause_b_by_tier"]["H"] for s in settings]
    ax.bar(xpos, a, 0.55, color=ps.C["normal_wait"], label="during the normal wait")
    ax.bar(xpos, b, 0.55, bottom=a, color=ps.C["after_fail"], label="after a failure")
    ax.set_ylim(top=max(a[i] + b[i] for i in range(len(settings))) * 1.32)
    for i in range(len(settings)):
        tot = a[i] + b[i]
        ax.annotate(f"{100*a[i]/tot:.0f}%\nfrom wait", (xpos[i], tot),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5)
    ax.set_xticks(xpos); ax.set_xticklabels([s[0] for s in settings])
    ax.set_ylabel("High-urgency patients lost\nper cohort")
    ax.legend(loc="upper right")
    fig.tight_layout(); ps.save(fig, "F2_why_high_urgency_lost", outdir)


def f3_three_plans(P, outdir):
    plans = P["three_plans"]["plans"]
    order = [("On-time", "on_time", ps.C["on_time"]),
             ("Decline-aware\n(spare capacity)", "decline_aware", ps.C["decline_aware"]),
             ("Decline-aware\n+ sickest first", "sickest_first", ps.C["sickest"])]
    fig = ps.fig("double", 0.46, ncols=2)
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
    xpos = np.arange(len(order))
    vals = [plans[k]["high_urgency_lost"] for _, k, _ in order]
    ax1.bar(xpos, vals, 0.6, color=[c for _, _, c in order])
    for i, v in enumerate(vals):
        ax1.annotate(f"{v:.1f}", (xpos[i], v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8.5)
    ax1.set_ylim(top=max(vals) * 1.2)
    ax1.set_xticks(xpos); ax1.set_xticklabels([n for n, _, _ in order], fontsize=7.5)
    ax1.set_ylabel("High-urgency patients lost\nper cohort")
    ax1.set_title("(a) High-urgency patients lost, by plan", fontsize=9)
    # panel (b): treated within deadline by tier
    tiers = ["H", "M", "L"]; names = {"H": "High", "M": "Medium", "L": "Low"}
    width = 0.38
    xb = np.arange(len(tiers))
    base = plans["on_time"]["treated_share_by_tier"]
    pri = plans["sickest_first"]["treated_share_by_tier"]
    ax2.bar(xb - width/2, [100*base[t] for t in tiers], width,
            color=ps.C["on_time"], label="on-time")
    ax2.bar(xb + width/2, [100*pri[t] for t in tiers], width,
            color=ps.C["sickest"], label="sickest first")
    ax2.set_xticks(xb); ax2.set_xticklabels([names[t] for t in tiers])
    ax2.set_ylabel("Treated within deadline (%)")
    ax2.set_ylim(70, 101)
    ax2.set_title("(b) Share treated within deadline, by class", fontsize=9)
    ax2.legend(loc="upper center", ncol=2, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(); ps.save(fig, "F3_three_plans", outdir)


def f4_capacity(P, outdir):
    rows = sorted(P["cap_sweep"]["rows"], key=lambda r: r["s_max"])
    smax = [r["s_max"] for r in rows]
    fig = ps.fig("single", 0.85); ax = fig.add_subplot(111)
    xpos = np.arange(len(smax))
    vals = [r["on_time"]["high_urgency_lost"] for r in rows]
    ax.bar(xpos, vals, 0.55, color=ps.C["on_time"])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (xpos[i], v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.set_ylim(top=max(vals) * 1.18)
    lbl = {40: "40\n(artificial cap)", 55: "55", 75: "75\n(real limit)"}
    ax.set_xticks(xpos); ax.set_xticklabels([lbl.get(s, str(s)) for s in smax])
    ax.set_xlabel("Capacity limit per facility (slots)")
    ax.set_ylabel("High-urgency patients lost\nper cohort")
    fig.tight_layout(); ps.save(fig, "F4_capacity_cap", outdir)


def f5_robustness(P, outdir):
    rows = sorted(P["e5_bwd"]["rows"], key=lambda r: r["multiplier"])
    hH = [100 * r["base_wait_death"]["H"] for r in rows]     # x-axis: 6-wk H mortality (%)
    share = [100 * r["share_from_normal_wait_H"] for r in rows]
    red = [r["sickest_first_reduction"] for r in rows]
    fig = ps.fig("double", 0.46, ncols=2)
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
    ax1.plot(hH, share, "-o", color=ps.C["normal_wait"])
    ax1.axhline(50, color=ps.C["grey"], ls=":", lw=0.9)
    ax1.set_xlabel("Assumed 6-week mortality,\nhigh-urgency (%)")
    ax1.set_ylabel("Share of high-urgency losses\nfrom the normal wait (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Share of high-urgency losses from the nominal wait", fontsize=9)
    ax2.plot(hH, red, "-s", color=ps.C["sickest"])
    ax2.set_xlabel("Assumed 6-week mortality,\nhigh-urgency (%)")
    ax2.set_ylabel("High-urgency patients saved by\nsickest-first, per cohort")
    ax2.set_ylim(bottom=0)
    ax2.set_title("(b) High-urgency patients saved by prioritization", fontsize=9)
    fig.tight_layout(); ps.save(fig, "F5_robustness", outdir)


# ----------------------------------------------------------- technical figures

def t1_planned_vs_sim(P, outdir):
    pvs = P["validation"].get("planned_vs_simulated", {})
    pts = pvs.get("points", [])
    if not pts:
        return
    spare = [p["spare_at_busiest"] for p in pts]
    planned = [p["planned_delay_death_cost"] for p in pts]
    sim = [p["simulated_delay_cost"] for p in pts]
    fig = ps.fig("single", 0.8); ax = fig.add_subplot(111)
    ax.plot(spare, planned, "-o", color=ps.C["decline_aware"], label="planning-model estimate")
    ax.plot(spare, sim, "-s", color=ps.C["normal_wait"], label="detailed simulation")
    ax.set_xlabel("Spare slots added at busiest facility")
    ax.set_ylabel("Delay-caused cost (M USD)")
    ax.legend()
    fig.tight_layout(); ps.save(fig, "T1_planned_vs_simulated", outdir)


def t2_gamma(P, outdir):
    rows = sorted(P["gamma_grid"]["rows"], key=lambda r: r["gamma"])
    g = [r["gamma"] for r in rows]
    fig = ps.fig("single", 0.8); ax = fig.add_subplot(111)
    ax.plot(g, [r["on_time_H_lost"] for r in rows], "-o", color=ps.C["on_time"], label="on-time")
    ax.plot(g, [r["sickest_first_H_lost"] for r in rows], "-s", color=ps.C["sickest"], label="sickest first")
    ax.set_xlabel(r"Decline-curve shape $\gamma$ (swept)")
    ax.set_ylabel("High-urgency lost per cohort")
    ax.legend()
    fig.tight_layout(); ps.save(fig, "T2_decline_shape_gamma", outdir)


def t3_hr(P, outdir):
    order = ["narrow", "central", "wide"]
    rows = {r["spread"]: r for r in P["hr_grid"]["rows"]}
    fig = ps.fig("single", 0.8); ax = fig.add_subplot(111)
    xpos = np.arange(len(order)); w = 0.38
    ax.bar(xpos - w/2, [rows[s]["on_time_H_lost"] for s in order], w,
           color=ps.C["on_time"], label="on-time")
    ax.bar(xpos + w/2, [rows[s]["sickest_first_H_lost"] for s in order], w,
           color=ps.C["sickest"], label="sickest first")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{s}\n{rows[s]['hr_tier']['H']}/{rows[s]['hr_tier']['M']}/{rows[s]['hr_tier']['L']}"
                        for s in order], fontsize=7.5)
    ax.set_xlabel("Tier hazard-ratio spread (H/M/L)")
    ax.set_ylabel("High-urgency lost per cohort")
    ax.legend()
    fig.tight_layout(); ps.save(fig, "T3_hazard_ratio_spread", outdir)


def f6a_survival(P, outdir):
    """F6a: survival rate at therapy across the rules (mean with min-max range)."""
    e6 = P.get("e6")
    if not e6:
        return
    rules = e6["rules"]
    labels = [r["rule"].replace("Threshold-", "Th-") for r in rules]
    avg = [100 * r["avg_survival"] for r in rules]
    lo = [100 * r["min_survival"] for r in rules]
    hi = [100 * r["max_survival"] for r in rules]
    x = np.arange(len(rules))
    fig = ps.fig("double", 0.5); ax = fig.add_subplot(111)
    yerr = np.vstack([np.array(avg) - np.array(lo), np.array(hi) - np.array(avg)])
    ax.errorbar(x, avg, yerr=yerr, fmt="o", color=ps.C["sickest"], ecolor=ps.C["grey"],
                elinewidth=1.0, capsize=3, ms=5)
    cut = 100 * e6["eligibility_cutoff"]
    ax.axhline(cut, color=ps.C["ref"], ls="--", lw=1.0)
    ax.set_ylim(cut - 1.5, max(hi) + 1.5)
    ax.text(0.02, cut + 0.4, "75% eligibility line (Tseng)", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=ps.C["ref"], va="bottom")
    ax.text(0.02, 0.97, "point = mean   •   bar = min–max", transform=ax.transAxes,
            fontsize=7.5, color="#444", va="top")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Survival rate at therapy (%)")
    ax.set_xlabel("Priority rule")
    fig.tight_layout(); ps.save(fig, "F6a_survival_across_rules", outdir)


def f6b_benefit(P, outdir):
    """F6b: who benefits vs who pays across the rules. (a) number with higher vs
    lower survival rate than FIFO; (b) the worst-off (minimum) survival rate,
    which rises as the rules protect the sickest. At the real rate no patient
    crosses the 75% eligibility line (become-eligible / -ineligible are both 0,
    reported in the table); the rules lift the floor instead."""
    e6 = P.get("e6")
    if not e6:
        return
    nonf = [r for r in e6["rules"] if r["rule"] != "FIFO"]
    labels = [r["rule"].replace("Threshold-", "Th-") for r in nonf]
    x = np.arange(len(nonf)); w = 0.38
    fig = ps.fig("double", 0.5, ncols=2)
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
    ax1.bar(x - w/2, [r["n_increased"] for r in nonf], w, color=ps.C["sickest"],
            label="higher survival")
    ax1.bar(x + w/2, [-r["n_decreased"] for r in nonf], w, color=ps.C["after_fail"],
            label="lower survival")
    ax1.axhline(0, color=ps.C["grey"], lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("Patients vs FIFO")
    ax1.set_title("(a) Patients with increased vs. decreased survival rate", fontsize=9)
    ax1.legend(loc="upper left")
    # (b) worst-off (minimum) survival rate across all rules incl. FIFO.
    allr = e6["rules"]
    xall = np.arange(len(allr))
    labels_all = [r["rule"].replace("Threshold-", "Th-").replace("FIFO", "FIFO") for r in allr]
    mins = [100 * r["min_survival"] for r in allr]
    cut = 100 * e6["eligibility_cutoff"]
    ax2.plot(xall, mins, "-o", color=ps.C["sickest"])
    ax2.axhline(cut, color=ps.C["ref"], ls="--", lw=1.0)
    ax2.set_ylim(cut - 0.8, max(mins) + 0.8)
    ax2.text(0.03, cut + 0.25, "75% eligibility line (no patient crosses it)",
             transform=ax2.get_yaxis_transform(), fontsize=7, color=ps.C["ref"], va="bottom")
    ax2.set_xticks(xall); ax2.set_xticklabels(labels_all, fontsize=7)
    ax2.set_ylabel("Worst-off survival rate (%)")
    ax2.set_title("(b) Worst-off (minimum) survival rate", fontsize=9)
    fig.tight_layout(); ps.save(fig, "F6b_who_benefits", outdir)


def f7_failure_wait_heatmap(P, outdir):
    """F7 (Issue 6): where waiting dominates vs where failure dominates — share of
    high-urgency losses from the normal wait, over failure rate x normal wait."""
    ft = P.get("failure_tv2v")
    if not ft:
        return
    fr = ft["failure_grid"]; tv = ft["t_v2v_grid"]
    M = np.zeros((len(fr), len(tv)))
    by = {(c["failure_rate"], c["t_v2v"]): c for c in ft["cells"]}
    for i, f in enumerate(fr):
        for j, t in enumerate(tv):
            M[i, j] = 100 * by[(f, t)]["share_from_normal_wait_H"]
    fig = ps.fig("single", 0.85); ax = fig.add_subplot(111)
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdYlBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(tv))); ax.set_xticklabels([f"{t}" for t in tv])
    ax.set_yticks(range(len(fr))); ax.set_yticklabels([f"{100*f:.0f}%" for f in fr])
    ax.set_xlabel("Normal vein-to-vein wait (days)")
    ax.set_ylabel("Manufacturing failure rate")
    for i in range(len(fr)):
        for j in range(len(tv)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    color="black" if 25 < M[i, j] < 75 else "white", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% of high-urgency losses from the normal wait", fontsize=8)
    ax.set_title("Share of high-urgency losses from the nominal wait", fontsize=9)
    fig.tight_layout(); ps.save(fig, "F7_failure_vs_wait_heatmap", outdir)


def make_all(P, main_dir, tech_dir):
    print("Main figures:")
    f1_decline_sweep(P, main_dir)
    f2_causes(P, main_dir)
    f3_three_plans(P, main_dir)
    f4_capacity(P, main_dir)
    f5_robustness(P, main_dir)
    f6a_survival(P, main_dir)
    f6b_benefit(P, main_dir)
    f7_failure_wait_heatmap(P, main_dir)
    print("Technical figures:")
    t1_planned_vs_sim(P, tech_dir)
    t2_gamma(P, tech_dir)
    t3_hr(P, tech_dir)
