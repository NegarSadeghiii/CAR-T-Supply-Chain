"""
Ladder of interventions at N = 200 -- experiment (a), re-run clean.

Four rungs, each a superset of the previous one's freedom:

  0. FIFO reference   plants + assignment from the cost model, cheapest
                      shipping, and a first-come-first-served schedule that is
                      NOT optimised at all (constructed, not solved).
  1. + Reorder queue  same plants, same assignment, same cheapest shipping;
                      only the start times are chosen, for survival.
  2. + Expedite       same plants and assignment; both transport modes allowed.
  3. + Open a plant   plants, assignment and shipping all free.

Every rung is solved fresh (no cache is read or written) and must reach proven
optimality; two rungs are only compared when both did.

Usage:  python ladder.py            # N = 200, CON1 <= 2
"""
from __future__ import annotations

import argparse
import os
import statistics
from collections import defaultdict

import cart_data as cd
import ishipment_survival as ish

OUT = ish.RESULTS
ALPHA_BIG = 1e7          # "alpha large": clinical loss dominates the objective
MIP_GAP = 1e-6           # tighter than the study default: the ladder compares
                         # costs that differ by only a few hundred dollars
LADDER_C = ["#6b7280", "#1b6a6a", "#2a8f6a", "#5b2a86"]


# --------------------------------------------------------------------------
def fifo_schedule(net, inst, opened, assign, mode):
    """First-come-first-served: each patient gets the earliest start >= its
    arrival at the plant that keeps every 7-day rolling window within FCAP.

    This is a *constructed* schedule, not an optimised one -- it is the
    reference a plant would run without any survival model at all.
    """
    TLS = int(net.TLS)
    lead = net.TMFE + net.TQC
    sig = cd.sigma_table()
    pat = {p.pid: p for p in inst.patients}

    arrivals = {pid: pat[pid].t0 + TLS + net.TT1[mode] for pid in assign}
    starts_at = defaultdict(list)          # plant -> chosen start days
    rows = []

    def fits(m, s):
        """adding a start on day s keeps every 7-day window <= FCAP[m]"""
        days = starts_at[m] + [s]
        for w in range(s - net.TMFE + 1, s + 1):
            if sum(1 for d in days if w <= d <= w + net.TMFE - 1) > net.FCAP[m]:
                return False
        return True

    order = sorted(assign, key=lambda pid: (arrivals[pid], pid))
    for pid in order:
        m = assign[pid]
        s = arrivals[pid]
        while not fits(m, s):
            s += 1
        starts_at[m].append(s)
        p = pat[pid]
        delivery = s + lead + net.TT3[mode]
        trt = int(round(delivery - p.t0))
        rows.append({"pid": pid, "tier": p.tier, "c": p.c, "h": p.h,
                     "t0": p.t0, "facility": m, "mode_in": mode,
                     "mode_out": mode, "ms_arrival": arrivals[pid],
                     "start": s, "hold": s - arrivals[pid],
                     "delivery": delivery, "TRT": trt,
                     "survival": sig[min(trt, cd.D_MAX)][p.tier],
                     "TRT_over_ND": max(0, trt - ish.ND_EXT)})
    for r in rows:
        r["expected_loss"] = 1 - r["survival"]
        r["weighted_loss"] = cd.RHO[r["tier"]] * r["expected_loss"]

    fc = ish.RESULTS and {m: net.horizon * (net.CIM[m] + net.CVM[m])
                          for m in net.m}
    cost = (sum(fc[m] for m in opened)
            + sum(net.U1[(r["c"], r["facility"], mode)]
                  + net.U3[(r["facility"], r["h"], mode)] for r in rows)
            + (net.C_material + net.CQC) * len(rows))
    return rows, cost


def summarise(rows, cost, label, status="constructed", gap=None, wall=None):
    rec = {"rung": label, "status": status,
           "mip_gap": gap, "wall_s": None if wall is None else round(wall, 1),
           "expected_lost": sum(r["expected_loss"] for r in rows),
           "weighted_loss": sum(r["weighted_loss"] for r in rows),
           "total_cost": cost,
           "mean_TRT": statistics.fmean(r["TRT"] for r in rows),
           "mean_HOLD": statistics.fmean(r["hold"] for r in rows),
           "max_TRT": max(r["TRT"] for r in rows),
           "n_over_ND": sum(1 for r in rows if r["TRT"] > ish.ND_EXT)}
    for u in cd.TIER_ORDER:
        sub = [r for r in rows if r["tier"] == u]
        rec[f"mean_HOLD_{u}"] = statistics.fmean(r["hold"] for r in sub)
        rec[f"expected_lost_{u}"] = sum(r["expected_loss"] for r in sub)
    return rec


def solve_rung(net, inst, label, cap, time_limit, **kw):
    """Fresh solve -- run_design/cache are deliberately bypassed."""
    print(f"  [solve] {label} ...", flush=True)
    mdl = ish.build_extension(net, inst, alpha=ALPHA_BIG,
                              max_facilities=cap, **kw)
    out = ish.solve(mdl, time_limit=time_limit, mip_gap=MIP_GAP)
    if out.status not in ("optimal", "feasible"):
        print(f"    -> {out.status}")
        return None, None, out
    rows, s = ish.extract(mdl, inst, net)
    rec = summarise(rows, s["total_cost"], label, out.status, out.gap, out.wall)
    rec["opened"] = "+".join(s["opened"])
    print(f"    -> {out.status} gap={out.gap:.2e} E[lost]={rec['expected_lost']:.4f} "
          f"cost=${rec['total_cost']:,.0f} ({out.wall:.0f}s)")
    return rows, rec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=200)
    ap.add_argument("--time-limit", type=int, default=2400)
    args = ap.parse_args()

    dat = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Data200_profileA.dat")
    net = cd.load_network(dat)
    inst = cd.build_instance(dat, mult=args.scale // 50)
    cap = ish.max_facilities_for(args.scale)
    n = inst.n
    print(f"Ladder of interventions, N = {n}, CON1 <= {cap}, "
          f"ALPHA = {ALPHA_BIG:g}, MIP gap target {MIP_GAP:g}")
    print("All solves are FRESH (cache bypassed).\n")

    # ---- rung 0: the cost model, then a FIFO schedule on top of it --------
    print("  [solve] cost model (alpha = 0) -> plants + assignment ...",
          flush=True)
    mdl0 = ish.build_extension(net, inst, alpha=0.0, max_facilities=cap)
    out0 = ish.solve(mdl0, time_limit=args.time_limit, mip_gap=MIP_GAP)
    rows0, s0 = ish.extract(mdl0, inst, net)
    print(f"    -> {out0.status} gap={out0.gap:.2e} opened={'+'.join(s0['opened'])} "
          f"({out0.wall:.0f}s)")
    if out0.status != "optimal":
        print("Cost model did not reach proven optimality; aborting.")
        return 1
    F0 = s0["opened"]
    assign = {r["pid"]: r["facility"] for r in rows0}
    cheap = min(net.j, key=lambda j: max(
        [net.U1[(c, m, j)] for c in net.c for m in net.m]
        + [net.U3[(m, h, j)] for m in net.m for h in net.h]))
    print(f"    cheapest transport mode = {cheap} "
          f"(TT1={net.TT1[cheap]}d, TT3={net.TT3[cheap]}d)")

    fifo_rows, fifo_cost = fifo_schedule(net, inst, F0, assign, cheap)
    rec0 = summarise(fifo_rows, fifo_cost, "FIFO reference")
    rec0["opened"] = "+".join(F0)
    print(f"  [build] FIFO reference: E[lost]={rec0['expected_lost']:.4f} "
          f"cost=${rec0['total_cost']:,.0f} mean hold={rec0['mean_HOLD']:.2f}d "
          f"max TRT={rec0['max_TRT']}d over-ND={rec0['n_over_ND']}")

    recs, all_rows = [rec0], {"FIFO reference": fifo_rows}

    # ---- rungs 1-3 ---------------------------------------------------------
    ladder = [
        ("+ Reorder queue", dict(fixed_facilities=F0, fixed_assignment=assign,
                                 allow_expedite=False)),
        ("+ Expedite shipping", dict(fixed_facilities=F0,
                                     fixed_assignment=assign,
                                     allow_expedite=True)),
        ("+ Open a plant", dict(allow_expedite=True)),
    ]
    for label, kw in ladder:
        rows, rec, out = solve_rung(net, inst, label, cap, args.time_limit, **kw)
        if rec is None:
            print(f"  {label}: no solution, ladder stops here")
            break
        recs.append(rec)
        all_rows[label] = rows

    # ---- deltas ------------------------------------------------------------
    # anything below the solver's own absolute tolerance is not a real cost
    # The solver's gap is relative to the OBJECTIVE (cost + ALPHA * weighted
    # loss), which is an order of magnitude larger than cost alone, so the
    # absolute dollar tolerance has to be derived from the objective.
    max_obj = max(r["total_cost"] + ALPHA_BIG * r["weighted_loss"] for r in recs)
    abs_tol = 2 * max(MIP_GAP * max_obj, 1.0)
    prev = None
    for r in recs:
        if prev is None:
            r["lives_saved_vs_FIFO"] = 0.0
            r["added_cost_vs_prev"] = None
            r["cost_per_life_this_rung"] = None
            r["comparable"] = True
        else:
            comparable = (r["status"] in ("optimal", "constructed")
                          and prev["status"] in ("optimal", "constructed"))
            r["comparable"] = comparable
            r["lives_saved_vs_FIFO"] = recs[0]["expected_lost"] - r["expected_lost"]
            d_cost = r["total_cost"] - prev["total_cost"]
            d_life = prev["expected_lost"] - r["expected_lost"]
            r["added_cost_vs_prev"] = d_cost
            r["cost_below_tolerance"] = abs(d_cost) < abs_tol
            r["cost_per_life_this_rung"] = (
                None if d_life <= 1e-9 or abs(d_cost) < abs_tol
                else d_cost / d_life)
        prev = r

    ish.write_csv(os.path.join(OUT, f"ladder_N{n}.csv"), recs)
    _print(recs, abs_tol)
    _plot(recs, n, cap, abs_tol, os.path.join(OUT, f"ladder_N{n}.png"))
    return 0


def _money(v):
    return "-" if v is None else f"${v:,.0f}"


def _print(recs, abs_tol):
    print("\n" + "=" * 123)
    hdr = (f"{'rung':<22}{'E[lost]':>9}{'lives saved':>13}{'total cost':>15}"
           f"{'added cost':>21}{'cost/life':>16}{'status':>13}{'gap':>10}")
    print(hdr)
    print("-" * 123)
    for r in recs:
        added = ("-" if r["added_cost_vs_prev"] is None else
                 ("no measurable cost" if r.get("cost_below_tolerance")
                  else _money(r["added_cost_vs_prev"])))
        cpl = ("-" if r["cost_per_life_this_rung"] is None
               else _money(r["cost_per_life_this_rung"]))
        gap = "-" if r["mip_gap"] is None else f"{r['mip_gap']:.1e}"
        print(f"{r['rung']:<22}{r['expected_lost']:>9.4f}"
              f"{r['lives_saved_vs_FIFO']:>13.4f}{_money(r['total_cost']):>15}"
              f"{added:>21}{cpl:>16}{r['status']:>13}{gap:>10}")
    print("=" * 123)
    print(f"Cost differences below ${abs_tol:,.0f} (twice the solver's absolute "
          f"optimality tolerance) are reported as 'no measurable cost'.")


def _plot(recs, n, cap, abs_tol, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    xs = [r["expected_lost"] for r in recs]
    ys = [r["total_cost"] / 1e6 for r in recs]

    # staircase: cost is held while lives are bought, then stepped up
    for a, b in zip(recs, recs[1:]):
        ax.plot([a["expected_lost"], b["expected_lost"]],
                [a["total_cost"] / 1e6] * 2, "-", color="#9ca3af", lw=1.4,
                zorder=1)
        ax.plot([b["expected_lost"]] * 2,
                [a["total_cost"] / 1e6, b["total_cost"] / 1e6], "-",
                color="#9ca3af", lw=1.4, zorder=1)
        mid_x = (a["expected_lost"] + b["expected_lost"]) / 2
        txt = ("no measurable cost" if b.get("cost_below_tolerance")
               else (f"${b['cost_per_life_this_rung']:,.0f}/life"
                     if b["cost_per_life_this_rung"] else ""))
        if txt:
            ax.annotate(txt, (mid_x, a["total_cost"] / 1e6),
                        textcoords="offset points", xytext=(0, -17),
                        ha="center", va="top", fontsize=8.5, color="#374151",
                        fontweight="bold")

    for i, r in enumerate(recs):
        ax.plot(xs[i], ys[i], "o", ms=11, color=LADDER_C[i % len(LADDER_C)],
                zorder=3, markeredgecolor="white", markeredgewidth=1.4)
        # every point label sits ABOVE its marker; the per-step cost labels
        # sit BELOW the horizontal run, so the two families cannot collide
        ax.annotate(f"{r['rung']}\nE[lost] {r['expected_lost']:.2f}",
                    (xs[i], ys[i]), textcoords="offset points",
                    xytext=(0, 13), ha="center", va="bottom", fontsize=9,
                    color=LADDER_C[i % len(LADDER_C)], fontweight="bold")

    ax.set_xlabel("Expected patients lost   $\\Sigma_p\\,(1 - S[p])$")
    ax.set_ylabel("Total supply-chain cost  [$M]")
    ax.set_title(f"Ladder of interventions - N = {n}", fontweight="bold",
                 loc="left")
    ax.grid(alpha=.3)
    ax.margins(x=.22, y=.26)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    raise SystemExit(main())
