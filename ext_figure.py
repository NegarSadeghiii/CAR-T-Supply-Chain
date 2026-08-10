"""
i-SHIPMENT-style 6-panel figure for the survival extension.

Two scenarios, solved on the SAME frozen instance and the SAME model
(``ishipment_survival.build_extension`` is reused unchanged -- only the ALPHA
and ND arguments differ):

    Scenario 1  "cost design"      ALPHA = 0                (teal)
    Scenario 2  "survival design"  ALPHA = ALPHA_SURVIVAL   (purple)

Panels
    (a) N=200  cost per therapy vs ND, stacked by Transport / Manufacturing /
               Quality control, one bar per scenario
    (b) N=500  same
    (c) N=200  average cost per therapy vs achieved mean return time
    (d) N=500  same
    (e) N=200  % utilisation of each opened facility over time, Scenario 1, ND=42
    (f) N=200  same, Scenario 2

Usage
    python ext_figure.py --sweep-only      # just run/cache the MILPs
    python ext_figure.py                   # sweep (cached) + build the figure
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import cart_data as cd
import ishipment_survival as ish

FIGDIR = os.path.join(ish.RESULTS, "figures")

ND_SWEEP = [18, 22, 26, 30, 34, 38, 42]

SCENARIOS = [
    ("Scenario 1", "cost", 0.0),
    ("Scenario 2", "survival", ish.ALPHA_SURVIVAL),
]

# ---- styling -------------------------------------------------------------
# teal = Scenario 1, purple = Scenario 2; components separated by shade+hatch
SCEN_COLOR = {"cost": "#1b7f79", "survival": "#6a3d9a"}
SHADES = {
    "cost":     {"Manufacturing": "#12615c",
                 "Transport": "#57b8b0",
                 "Quality control": "#9fd8d3"},
    "survival": {"Manufacturing": "#4e2b73",
                 "Transport": "#9a72c4",
                 "Quality control": "#c9b3e0"},
}
HATCH = {"Manufacturing": "", "Transport": "///", "Quality control": "..."}
COMPONENTS = ["Manufacturing", "Transport", "Quality control"]

FACILITY_COLOR = {"m1": "#1b7f79", "m2": "#6a3d9a", "m3": "#e08a2e",
                  "m4": "#57b8b0", "m5": "#9a72c4", "m6": "#c25b5b"}
FACILITY_HATCH = {"m1": "", "m2": "///", "m3": "...", "m4": "xxx",
                  "m5": "\\\\\\", "m6": "+++"}


# --------------------------------------------------------------------------
# 1. ND sweep
# --------------------------------------------------------------------------
def sweep(net, inst, nds=ND_SWEEP, time_limit=1800, use_cache=True):
    """Solve both scenarios at every ND. Returns one record per (scenario, ND)."""
    n = inst.n
    qc = net.CQC * n
    mat = net.C_material * n
    out = []
    for label, key, alpha in SCENARIOS:
        for nd in nds:
            blob = ish.run_design(net, inst, "extension", alpha=alpha, nd=nd,
                                  time_limit=time_limit, use_cache=use_cache)
            rec = {"n": n, "scenario": label, "design": key, "alpha": alpha,
                   "ND": nd, "status": blob["solve"]["status"],
                   "mip_gap": blob["solve"]["gap"],
                   "wall_s": round(blob["solve"]["wall_s"] or 0, 1)}
            if "summary" in blob:
                s = blob["summary"]
                man = s["facility_cost"] + mat
                rec.update({
                    "opened": "+".join(s["opened"]),
                    "capacity_opened": s["capacity_opened"],
                    "transport_cost": round(s["transport_cost"], 2),
                    "quality_control": round(qc, 2),
                    "manufacturing": round(man, 2),
                    "facility_cost": round(s["facility_cost"], 2),
                    "material_cost": round(mat, 2),
                    "total_cost": round(s["total_cost"], 2),
                    "cost_per_therapy": round(s["total_cost"] / n, 2),
                    "mean_return_time": round(s["mean_TRT"], 3),
                    "max_return_time": s["max_TRT"],
                    "mean_HOLD": round(s["mean_HOLD"], 3),
                    "mean_survival": round(s["mean_survival"], 5),
                    "expected_lost": round(s["expected_lost"], 3),
                })
                # the three plotted components must reconstruct the total
                assert abs(rec["transport_cost"] + rec["quality_control"]
                           + rec["manufacturing"] - rec["total_cost"]) < 1.0
            out.append(rec)
    return out


def occupancy(rows, net, opened):
    """util%[m][t] = 100 * (#jobs occupying m on day t) / FCAP[m].

    MSBnew puts DURV[p,m,t] = 1 exactly on t in [start+1, start+TMFE], so a job
    started on day s occupies its facility on those TMFE days.
    """
    occ = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for t in range(r["start"] + 1, r["start"] + net.TMFE + 1):
            occ[r["facility"]][t] += 1
    tmax = max((max(d) for d in occ.values()), default=1)
    days = list(range(1, tmax + 1))
    series = {m: [100.0 * occ[m].get(t, 0) / net.FCAP[m] for t in days]
              for m in opened}
    return days, series


# --------------------------------------------------------------------------
# 2. Figure
# --------------------------------------------------------------------------
def build_figure(sweeps, util_panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "hatch.linewidth": 0.6})
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.0))
    LETTER = [["a", "b"], ["c", "d"], ["e", "f"]]

    def tag(row, col, text):
        axes[row][col].set_title(f"({LETTER[row][col]}) {text}", loc="left",
                                 fontweight="bold")

    # ---- (a),(b) stacked cost-per-therapy bars ---------------------------
    for col, n in enumerate(sorted(sweeps)):
        ax = axes[0][col]
        recs = [r for r in sweeps[n] if r.get("cost_per_therapy") is not None]
        nds = sorted({r["ND"] for r in recs})
        width = 0.38
        for k, (label, key, _) in enumerate(SCENARIOS):
            offs = (k - 0.5) * width
            for i, nd in enumerate(nds):
                m = [r for r in recs if r["ND"] == nd and r["design"] == key]
                if not m:
                    continue
                r, bottom = m[0], 0.0
                for comp in COMPONENTS:
                    v = r[{"Manufacturing": "manufacturing",
                           "Transport": "transport_cost",
                           "Quality control": "quality_control"}[comp]] / n
                    ax.bar(i + offs, v, width, bottom=bottom,
                           color=SHADES[key][comp], hatch=HATCH[comp],
                           edgecolor="white", linewidth=.5, zorder=2)
                    bottom += v
        ax.set_xticks(range(len(nds)))
        ax.set_xticklabels([str(x) for x in nds])
        ax.set_xlabel("Maximum return time ND  [days]")
        ax.set_ylabel("Cost per therapy  [$]")
        ax.grid(axis="y", alpha=.3, zorder=0)
        ax.set_axisbelow(True)
        tag(0, col, f"Cost breakdown per therapy, N = {n}")
        missing = [r for r in sweeps[n] if r.get("cost_per_therapy") is None]
        if missing:
            ax.text(.02, .96, "infeasible: " + ", ".join(
                f"{r['design']} ND={r['ND']}" for r in missing),
                transform=ax.transAxes, va="top", fontsize=7.5, color="#b03030")

    handles = [Patch(facecolor="#9a9a9a", hatch=HATCH[c], edgecolor="white",
                     label=c) for c in COMPONENTS]
    handles += [Patch(facecolor=SCEN_COLOR[k], label=f"{lab} ({k} design)")
                for lab, k, _ in SCENARIOS]
    for col in range(min(2, len(sweeps))):
        axes[0][col].legend(handles=handles, fontsize=7, loc="upper right",
                            ncol=2, framealpha=.95)
        axes[0][col].margins(y=.22)

    # ---- (c),(d) cost per therapy vs achieved mean return time -----------
    for col, n in enumerate(sorted(sweeps)):
        ax = axes[1][col]
        all_pts = {}
        for label, key, _ in SCENARIOS:
            pts = sorted(((r["mean_return_time"], r["cost_per_therapy"], r["ND"])
                          for r in sweeps[n]
                          if r["design"] == key
                          and r.get("cost_per_therapy") is not None))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                    color=SCEN_COLOR[key], label=f"{label} ({key} design)",
                    ms=5, lw=1.8)
            all_pts.setdefault(key, []).extend(pts)
        # Several ND values often reach the identical schedule, so their points
        # coincide. Cluster in normalised axis space and label each cluster once.
        flat = [p for v in all_pts.values() for p in v]
        if flat:
            xr = max(p[0] for p in flat) - min(p[0] for p in flat) or 1
            yr = max(p[1] for p in flat) - min(p[1] for p in flat) or 1
        for key, pts in all_pts.items():
            clusters = []
            for x, y, nd in sorted(pts):
                for cl in clusters:
                    if (abs(x - cl[0][0]) / xr < .035
                            and abs(y - cl[0][1]) / yr < .035):
                        cl[1].append(nd)
                        break
                else:
                    clusters.append([(x, y), [nd]])
            dx, dy, ha = (7, 8, "left") if key == "cost" else (-7, -15, "right")
            clusters.sort(key=lambda c: c[0][0])
            for i, ((x, y), nds_) in enumerate(clusters):
                # stagger consecutive labels so neighbouring ND markers on the
                # same near-horizontal stretch cannot run into each other
                step = 11 if i % 2 else 0
                ax.annotate("ND=" + ",".join(str(v) for v in sorted(nds_)),
                            (x, y), textcoords="offset points",
                            xytext=(dx, dy + (step if dy > 0 else -step)),
                            fontsize=6.5, ha=ha, color=SCEN_COLOR[key])

        ax.set_xlabel("Achieved mean return time  [days]")
        ax.set_ylabel("Average cost per therapy  [$]")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.margins(x=.14, y=.18)
        tag(1, col, f"Cost vs achieved return time, N = {n}")

    # ---- (e),(f) facility utilisation over time --------------------------
    for col in range(2):
        if col >= len(util_panels):
            axes[2][col].axis("off")
            continue
        panel = util_panels[col]
        ax = axes[2][col]
        days, series, opened, title = (panel["days"], panel["series"],
                                       panel["opened"], panel["title"])
        stack = ax.stackplot(days, [series[m] for m in opened],
                             labels=opened,
                             colors=[FACILITY_COLOR[m] for m in opened],
                             edgecolor="white", linewidth=.3)
        for poly, m in zip(stack, opened):
            poly.set_hatch(FACILITY_HATCH[m])
        for k in range(1, len(opened) + 1):
            ax.axhline(100 * k, color="#666666", ls="--", lw=.7, alpha=.7,
                       zorder=3)
        ax.set_xlabel("Time  [days]")
        ax.set_ylabel("Utilisation of opened facilities  [%, stacked]")
        ax.set_xlim(1, max(p["days"][-1] for p in util_panels))
        ax.set_ylim(0, 100 * max(len(p["opened"]) for p in util_panels) * 1.04)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7.5, loc="upper right", title="facility",
                  title_fontsize=7.5)
        tag(2, col, title)

    for row, col in [(0, 1), (1, 1)]:
        if col >= len(sweeps):
            axes[row][col].axis("off")

    fig.suptitle("Survival-aware i-SHIPMENT extension - "
                 "Scenario 1 (cost, ALPHA = 0) vs Scenario 2 "
                 f"(survival, ALPHA = {ish.ALPHA_SURVIVAL:g})",
                 fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .985))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("[figure]", path)


# --------------------------------------------------------------------------
def print_table(recs):
    cols = ["n", "scenario", "ND", "status", "opened", "transport_cost",
            "quality_control", "manufacturing", "total_cost",
            "cost_per_therapy", "mean_return_time", "mean_HOLD",
            "expected_lost", "wall_s"]
    ish._print_table([{c: r.get(c, "-") for c in cols} for r in recs], cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=int, default=[200, 500])
    ap.add_argument("--nd", nargs="+", type=int, default=ND_SWEEP)
    ap.add_argument("--util-scale", type=int, default=200)
    ap.add_argument("--util-nd", type=int, default=42)
    ap.add_argument("--time-limit", type=int, default=0,
                    help="0 = scale-dependent (200 -> 1800s, 500 -> 3600s)")
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    dat = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Data200_profileA.dat")
    net = cd.load_network(dat)

    sweeps, all_recs = {}, []
    for n in args.scales:
        inst = cd.build_instance(dat, mult=n // 50)
        tl = args.time_limit or (3600 if n >= 500 else 1800)
        recs = sweep(net, inst, args.nd, time_limit=tl,
                     use_cache=not args.no_cache)
        sweeps[n] = recs
        all_recs += recs
        ish.write_csv(os.path.join(FIGDIR, f"ext_sweep_N{n}.csv"), recs)
        print(f"\n--- ND sweep, N = {n} ---")
        print_table(recs)

    infeasible = [r for r in all_recs if r["status"] != "optimal"]
    print("\nNon-optimal runs:",
          ", ".join(f"N={r['n']} {r['design']} ND={r['ND']} -> {r['status']}"
                    for r in infeasible) or "none")

    if args.sweep_only:
        return 0

    # utilisation panels come from the ND=42 solutions at --util-scale
    inst = cd.build_instance(dat, mult=args.util_scale // 50)
    panels = []
    for label, key, alpha in SCENARIOS:
        blob = ish.run_design(net, inst, "extension", alpha=alpha,
                              nd=args.util_nd,
                              time_limit=args.time_limit or 1800,
                              use_cache=not args.no_cache)
        if "rows" not in blob:
            continue
        opened = blob["summary"]["opened"]
        days, series = occupancy(blob["rows"], net, opened)
        panels.append({
            "days": days, "series": series, "opened": opened,
            "title": (f"{label} ({key} design), N = {args.util_scale}, "
                      f"ND = {args.util_nd}\nmean return time "
                      f"{blob['summary']['mean_TRT']:.2f} d, "
                      f"mean hold {blob['summary']['mean_HOLD']:.2f} d, "
                      f"E[lost] {blob['summary']['expected_lost']:.2f}"),
        })
        ish.write_csv(
            os.path.join(FIGDIR, f"ext_utilisation_N{args.util_scale}_{key}.csv"),
            [{"day": d, **{m: round(series[m][i], 4) for m in opened}}
             for i, d in enumerate(days)])

    build_figure(sweeps, panels,
                 os.path.join(FIGDIR, "extension_ishipment_style.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
