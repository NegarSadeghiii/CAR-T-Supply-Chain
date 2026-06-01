"""
figures/generate_multi_instance.py
====================================
Generates per-size figures (04-07) for N ∈ {50, 100, 200, 500} patients,
plus Figure 08 (2×2 multi-instance summary).

Run: python figures/generate_multi_instance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import COLORS, setup_style, single_column, double_column, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

setup_style()

SIZES = [50, 100, 200, 500]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load(n: int) -> dict:
    p = _REPO / "results" / f"case_study_results_N{n}.json"
    with open(p) as f:
        return json.load(f)

DATA = {n: _load(n) for n in SIZES}

with open(_REPO / "results" / "multi_instance_results.json") as f:
    SUMMARY = json.load(f)


# ---------------------------------------------------------------------------
# Figure 04 — Tier-stratified cancellation (3 plans)
# ---------------------------------------------------------------------------

def fig04(n: int) -> None:
    d = DATA[n]
    tc = d["tier_cancel_rates"]
    tiers = ["Tier H", "Tier M", "Tier L"]
    keys  = ["H", "M", "L"]
    naive_vals = [tc["naive"][k] for k in keys]
    smart_vals = [tc["smart"][k] for k in keys]
    sp_vals    = [tc["sp"][k]    for k in keys]

    x = np.arange(len(tiers))
    w = 0.26

    fig = single_column()
    ax  = fig.add_subplot(111)

    ax.bar(x - w, naive_vals, w, label="Deterministic plan",
           color=COLORS["naive_det"], zorder=3)
    ax.bar(x,     smart_vals, w, label="Expected-cost deterministic",
           color=COLORS["expected_cost_det"], zorder=3)
    ax.bar(x + w, sp_vals,    w, label="Stochastic plan",
           color=COLORS["sp"], zorder=3)

    # Relative reduction naive→SP above each tier group
    for i, (nv, sv) in enumerate(zip(naive_vals, sp_vals)):
        rel = round((nv - sv) / nv * 100) if nv > 0 else 0
        top = max(nv, smart_vals[i], sv) + 0.08
        ax.text(x[i], top, f"−{rel}%",
                ha="center", va="bottom", fontsize=7.5,
                color="#111111", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Realized cancellation rate (%)")
    ax.set_ylim(0, max(naive_vals) * 1.45)
    ax.set_title(f"Tier-stratified cancellation rate (N = {n} patients)", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save_figure(fig, f"figure04_tier_cancellation_N{n}")
    print(f"    Tier cancel: naive={[f'{v:.2f}' for v in naive_vals]}  "
          f"smart={[f'{v:.2f}' for v in smart_vals]}  "
          f"sp={[f'{v:.2f}' for v in sp_vals]}")


# ---------------------------------------------------------------------------
# Figure 05 — Recourse mix (3 plans, horizontal stacked bars)
# ---------------------------------------------------------------------------

def fig05(n: int) -> None:
    d    = DATA[n]
    rm   = d["recourse_mix"]
    lbls = ["Deterministic plan", "Exp-cost deterministic", "Stochastic plan"]
    keys = ["naive", "smart", "sp"]

    remfg  = [rm[k]["remfg"]  for k in keys]
    sub    = [rm[k]["sub"]    for k in keys]
    cancel = [rm[k]["cancel"] for k in keys]

    fig = single_column()
    ax  = fig.add_subplot(111)
    y   = [2, 1, 0]
    h   = 0.35

    def _hbar(lefts, vals, color, label):
        bars = ax.barh(y, vals, h, left=lefts, color=color, label=label, zorder=3)
        for bar, v, lft in zip(bars, vals, lefts):
            if v > 5:
                tc = "white" if color != COLORS["subcontract"] else "#333333"
                ax.text(lft + v/2, bar.get_y() + h/2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color=tc, fontweight="bold")
        return [l + v for l, v in zip(lefts, vals)]

    lefts = [0.0]*3
    lefts = _hbar(lefts, remfg,  COLORS["remfg"],      "Re-manufacture in-house")
    lefts = _hbar(lefts, sub,    COLORS["subcontract"], "Subcontract to partner")
    lefts = _hbar(lefts, cancel, COLORS["cancel"],      "Cancel patient")

    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Share of failed batches (%)")
    ax.set_title(f"Recourse action composition (N = {n} patients)", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, fontsize=7, frameon=True)
    fig.tight_layout()
    save_figure(fig, f"figure05_recourse_mix_N{n}")


# ---------------------------------------------------------------------------
# Figure 06 — Facility allocation split panels (naive | SP)
# ---------------------------------------------------------------------------

def fig06(n: int) -> None:
    d    = DATA[n]
    fa   = d["facility_allocation"]
    tiers       = ["H", "M", "L"]
    tier_colors = [COLORS["tier_H"], COLORS["tier_M"], COLORS["tier_L"]]
    tier_labels = ["Tier H", "Tier M", "Tier L"]
    n_fac = 4
    fac_labels = [f"m$_{m}$" for m in range(n_fac)]

    def _counts(plan_key):
        alloc = fa[plan_key]
        arr = np.zeros((n_fac, 3), dtype=int)
        for mi in range(n_fac):
            row = alloc.get(f"m_{mi}", {})
            for ti, t in enumerate(tiers):
                arr[mi, ti] = row.get(t, 0)
        return arr

    ev_c  = _counts("naive")
    rp_c  = _counts("sp")

    fig = double_column()
    fig.subplots_adjust(bottom=0.16, wspace=0.08)
    axes = fig.subplots(1, 2, sharey=True)
    x    = np.arange(n_fac)
    w    = 0.55

    for ax, counts, title in zip(axes, [ev_c, rp_c],
                                  ["Deterministic plan", "Stochastic plan"]):
        bottom = np.zeros(n_fac)
        for ti, (t, color, lbl) in enumerate(zip(tiers, tier_colors, tier_labels)):
            vals = counts[:, ti].astype(float)
            ax.bar(x, vals, w, bottom=bottom, color=color, zorder=3, label=lbl)
            for fi in range(n_fac):
                if vals[fi] > 0:
                    tc = "white" if color != COLORS["tier_M"] else "#333333"
                    ax.text(x[fi], bottom[fi] + vals[fi]/2, str(int(vals[fi])),
                            ha="center", va="center", fontsize=7,
                            color=tc, fontweight="bold")
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(fac_labels, fontsize=10)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Manufacturing facility", fontsize=10)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Number of patients assigned")
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False)

    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(tier_colors, tier_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Stage-1 allocation by tier (N = {n} patients)", fontsize=11, y=1.01)
    save_figure(fig, f"figure06_facility_allocation_N{n}")


# ---------------------------------------------------------------------------
# Figure 07 — Cost composition (3 plans, cost per patient for comparability)
# ---------------------------------------------------------------------------

def fig07(n: int) -> None:
    d = DATA[n]
    plans = {
        "Deterministic\nplan":            (d["naive_eev"]["stage1"], d["naive_eev"]["stage2"]),
        "Expected-cost\ndeterministic":   (d["smart_eev"]["stage1"], d["smart_eev"]["stage2"]),
        "Stochastic\nplan":               (d["rp"]["stage1"],        d["rp"]["stage2"]),
    }

    lbls = list(plans.keys())
    s1   = [v[0]/n for v in plans.values()]    # cost per patient
    s2   = [v[1]/n for v in plans.values()]
    tots = [a+b for a,b in zip(s1,s2)]

    x = range(len(lbls))
    w = 0.52

    fig = single_column()
    ax  = fig.add_subplot(111)

    ax.bar(x, s1, w, color=COLORS["stage1"], zorder=3, label="Stage-1 (committed)")
    ax.bar(x, s2, w, bottom=s1, color=COLORS["stage2"], zorder=3,
           label="Expected Stage-2 (recourse)")

    MIN = max(tots) * 0.06
    for xi, (a, b, tot) in enumerate(zip(s1, s2, tots)):
        ax.text(xi, tot + max(tots)*0.01, f"{tot:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        if a >= MIN:
            ax.text(xi, a/2, f"{a:.3f}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
        if b >= MIN:
            ax.text(xi, a + b/2, f"{b:.3f}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")

    ax.set_ylim(0, max(tots)*1.18)
    ax.set_xticks(list(x))
    ax.set_xticklabels(lbls, fontsize=8.5)
    ax.set_ylabel("Expected cost per patient (M USD)")
    ax.set_title(f"Cost composition per patient (N = {n})", fontsize=11)
    ax.tick_params(axis="x", length=0)

    leg_patches = [
        mpatches.Patch(color=COLORS["stage1"], label="Stage-1 (committed)"),
        mpatches.Patch(color=COLORS["stage2"], label="Expected Stage-2 (recourse)"),
    ]
    ax.legend(handles=leg_patches, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#CCCCCC")
    fig.tight_layout()
    save_figure(fig, f"figure07_cost_composition_N{n}")


# ---------------------------------------------------------------------------
# Figure 08 — Multi-instance 2×2 summary
# ---------------------------------------------------------------------------

def fig08_summary() -> None:
    sizes     = [r["size"]          for r in SUMMARY]
    vss_naive = [r["vss_naive_pct"] for r in SUMMARY]
    vss_smart = [r["vss_smart_pct"] for r in SUMMARY]
    th_naive  = [r["tier_h_naive"]  for r in SUMMARY]
    th_smart  = [r["tier_h_smart"]  for r in SUMMARY]
    th_sp     = [r["tier_h_sp"]     for r in SUMMARY]
    sp_remfg  = [r["sp_remfg_pct"]  for r in SUMMARY]
    sp_sub    = [r["sp_sub_pct"]    for r in SUMMARY]
    sp_cancel = [r["sp_cancel_pct"] for r in SUMMARY]
    sp_times  = [r["sp_solve_time"] for r in SUMMARY]
    naive_t   = [r["naive_time"]    for r in SUMMARY]
    smart_t   = [r["smart_time"]    for r in SUMMARY]

    fig = double_column()
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    m_kw  = dict(marker="o", markersize=5, linewidth=1.5)
    x_log = np.log10(sizes)

    # --- (a) VSS % ---
    ax = axes[0, 0]
    ax.plot(x_log, vss_naive, color=COLORS["naive_det"], label="vs Naive det.",  **m_kw)
    ax.plot(x_log, vss_smart, color=COLORS["expected_cost_det"],
            label="vs Exp-cost det.", **m_kw)
    ax.axhline(4, color="#AAAAAA", linestyle=":", linewidth=0.8)
    ax.set_title("(a) VSS (%)", fontsize=10)
    ax.set_xlabel("|I| (patients)", fontsize=9)
    ax.set_ylabel("VSS (%)", fontsize=9)
    ax.set_xticks(x_log)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(bottom=0)

    # --- (b) Tier-H cancellation ---
    ax = axes[0, 1]
    ax.plot(x_log, th_naive, color=COLORS["naive_det"],         label="Naive",  **m_kw)
    ax.plot(x_log, th_smart, color=COLORS["expected_cost_det"], label="Smart",  **m_kw)
    ax.plot(x_log, th_sp,    color=COLORS["sp"],                label="SP",     **m_kw)
    ax.set_title("(b) Tier-H cancellation rate (%)", fontsize=10)
    ax.set_xlabel("|I| (patients)", fontsize=9)
    ax.set_ylabel("Cancellation rate (%)", fontsize=9)
    ax.set_xticks(x_log)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=8)
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    # --- (c) SP recourse mix ---
    ax = axes[1, 0]
    bottom = np.zeros(len(sizes))
    for vals, color, lbl in [
        (sp_remfg,  COLORS["remfg"],      "Re-mfg"),
        (sp_sub,    COLORS["subcontract"], "Subcontract"),
        (sp_cancel, COLORS["cancel"],      "Cancel"),
    ]:
        ax.bar(x_log, vals, 0.18, bottom=bottom, color=color, label=lbl)
        bottom = bottom + np.array(vals)
    ax.set_title("(c) SP recourse mix (%)", fontsize=10)
    ax.set_xlabel("|I| (patients)", fontsize=9)
    ax.set_ylabel("Share of failed batches (%)", fontsize=9)
    ax.set_xticks(x_log)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=8)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=7, loc="upper right")

    # --- (d) Solve time log-log ---
    ax = axes[1, 1]
    ax.scatter(x_log, naive_t, color=COLORS["naive_det"],
               marker="o", s=40, label="Naive EEV", zorder=4)
    ax.scatter(x_log, smart_t, color=COLORS["expected_cost_det"],
               marker="s", s=40, label="Smart EEV", zorder=4)
    ax.scatter(x_log, sp_times, color=COLORS["sp"],
               marker="^", s=50, label="SP (RP)", zorder=4)
    ax.set_yscale("log")
    ax.axhline(1800, color="#B83A3A", linestyle="--", linewidth=0.8,
               label="30-min limit")
    ax.set_title("(d) Solve time (s)", fontsize=10)
    ax.set_xlabel("|I| (patients)", fontsize=9)
    ax.set_ylabel("Solve time (s, log scale)", fontsize=9)
    ax.set_xticks(x_log)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=8)
    ax.legend(fontsize=7)

    fig.suptitle("Multi-instance results: N = 50, 100, 200, 500 patients",
                 fontsize=11, y=1.01)
    save_figure(fig, "figure08_multi_instance_summary")


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme() -> None:
    readme = _HERE / "README.md"
    existing = readme.read_text()

    if "Multi-instance comparison" in existing:
        print("  README multi-instance section already present — skipping.")
        return

    table_rows = []
    for r in SUMMARY:
        n    = r["size"]
        d    = DATA[n]
        row  = (
            f"| {n} "
            f"| {r['naive_total']:.2f} "
            f"| {r['smart_total']:.2f} "
            f"| {r['rp_total']:.2f} "
            f"| {r['vss_naive_pct']:.1f}% "
            f"| {r['vss_smart_pct']:.1f}% "
            f"| {r['tier_h_naive']:.2f} / {r['tier_h_smart']:.2f} / {r['tier_h_sp']:.2f} "
            f"| {r['sp_solve_time']:.1f}s "
            f"| {r['sp_gap_pct']:.3f}% |"
        )
        table_rows.append(row)

    section = "\n## Multi-instance comparison\n\n"
    section += (
        "Results across patient cohort sizes. Costs in M USD. "
        "Tier-H rates shown as Naive / Smart / SP.\n\n"
    )
    section += ("| `|I|` | Naive EEV | Smart EEV | RP | "
                "VSS naive | VSS smart | Tier-H (N/S/SP) | SP time | SP gap |\n")
    section += "|---|---|---|---|---|---|---|---|---|\n"
    section += "\n".join(table_rows) + "\n\n"

    # Per-size figure entries
    for n in SIZES:
        for fi, cap in [
            ("04", f"Tier-stratified cancellation (N={n} patients)."),
            ("05", f"Recourse mix across three plans (N={n} patients)."),
            ("06", f"Stage-1 facility allocation (N={n} patients)."),
            ("07", f"Cost composition per patient (N={n} patients)."),
        ]:
            name = f"figure{fi}_*_N{n}"
            section += f"- `{name}.png/.pdf` — {cap}\n"

    section += "\n- `figure08_multi_instance_summary.png/.pdf` — 2×2 panel summary across all sizes.\n"

    readme.write_text(existing.rstrip("\n") + "\n" + section)
    print(f"  Updated: {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Multi-instance figure generation ===\n")

    for n in SIZES:
        print(f"[N={n}] Figures 04-07")
        fig04(n)
        fig05(n)
        fig06(n)
        fig07(n)

    print("\n[Figure 08] Multi-instance summary")
    fig08_summary()

    print()
    update_readme()

    # Verify
    print("\nOutput verification:")
    all_ok = True
    for n in SIZES:
        for fi in ["04_tier_cancellation", "05_recourse_mix",
                   "06_facility_allocation", "07_cost_composition"]:
            name = f"figure{fi}_N{n}"
            ok = (_HERE/"png"/f"{name}.png").exists() and (_HERE/"pdf"/f"{name}.pdf").exists()
            if not ok:
                print(f"  [MISSING] {name}")
                all_ok = False
    for name in ["figure08_multi_instance_summary"]:
        ok = (_HERE/"png"/f"{name}.png").exists() and (_HERE/"pdf"/f"{name}.pdf").exists()
        if not ok:
            print(f"  [MISSING] {name}")
            all_ok = False

    if all_ok:
        print(f"  All {len(SIZES)*4 + 1} figures OK.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
