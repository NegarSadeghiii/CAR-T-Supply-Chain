"""
Property tests for the queue + survival extension.

Run:  python test_extension.py
"""
from __future__ import annotations

import sys
from collections import defaultdict

from pyomo.environ import value

import cart_data as cd
import ishipment_survival as ish

DAT = "Data200_profileA.dat"
FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def small_instance(n=30, mult=1):
    inst = cd.build_instance(DAT, mult=mult)
    inst.patients = inst.patients[:n]
    inst.n = len(inst.patients)
    return inst


def test_sigma_lookup():
    print("\n[sigma table]")
    sig = cd.sigma_table()
    check("sigma covers d = 17..42", set(sig) == set(range(17, 43)))
    check("S_H(42) == 1 - w_H", abs(sig[42]["H"] - 0.85) < 1e-12)
    check("S_M(42) == 1 - w_M", abs(sig[42]["M"] - 0.95) < 1e-12)
    check("S_L(42) == 1 - w_L", abs(sig[42]["L"] - 0.98) < 1e-12)
    check("survival is strictly decreasing in d for every tier",
          all(sig[d][u] > sig[d + 1][u] for d in range(17, 42) for u in "HML"))
    check("high risk always has the lowest survival",
          all(sig[d]["H"] < sig[d]["M"] < sig[d]["L"] for d in range(17, 43)))


def test_instance_generation():
    print("\n[instance generation]")
    for mult, n in ((2, 100), (4, 200), (10, 500)):
        inst = cd.build_instance(DAT, mult=mult)
        tiers = defaultdict(int)
        sites = defaultdict(int)
        for p in inst.patients:
            tiers[p.tier] += 1
            sites[p.c] += 1
        check(f"N={n}: exactly {n} patients", inst.n == n)
        check(f"N={n}: tier mix is 30/40/30",
              (tiers["H"], tiers["M"], tiers["L"]) == (int(.30 * n), int(.40 * n), int(.30 * n)),
              str(dict(tiers)))
        base_sites = defaultdict(int)
        for _, c, _ in cd.base_cohort(DAT):
            base_sites[c] += 1
        check(f"N={n}: LS-site distribution preserved",
              all(sites[c] == base_sites[c] * mult for c in base_sites))
        check(f"N={n}: arrivals inside the admissible window",
              all(cd.ARRIVAL_WINDOW[0] <= p.t0 <= cd.ARRIVAL_WINDOW[1]
                  for p in inst.patients))
    a = [p.t0 for p in cd.build_instance(DAT, mult=4, seed=0).patients]
    b = [p.t0 for p in cd.build_instance(DAT, mult=4, seed=0).patients]
    check("generation is reproducible for a fixed seed", a == b)
    d = [cd.build_instance(DAT, mult=m).density for m in (1, 2, 4, 10)]
    check("arrival density rises with N", all(x < y for x, y in zip(d, d[1:])),
          " -> ".join(f"{x:.2f}" for x in d))


def _solved_extension(alpha, n=30):
    net = cd.load_network(DAT)
    inst = small_instance(n)
    mdl = ish.build_extension(net, inst, alpha=alpha)
    out = ish.solve(mdl, time_limit=300)
    assert out.status in ("optimal", "feasible"), out.as_dict()
    rows, summary = ish.extract(mdl, inst, net)
    return net, inst, mdl, out, rows, summary


def test_extension_invariants():
    print("\n[extension invariants  (alpha = 1e5, N = 30)]")
    net, inst, mdl, out, rows, summary = _solved_extension(1e5)
    sig = cd.sigma_table()
    pat = {p.pid: p for p in inst.patients}

    check("every patient is started exactly once",
          all(abs(sum(value(mdl.INM[r["pid"], m, t]) for m in net.m
                      for t in mdl.win[r["pid"]]) - 1) < 1e-6 for r in rows))
    check("(32) no start before the sample arrives: HOLD >= 0",
          all(r["hold"] >= 0 for r in rows), f"min={min(r['hold'] for r in rows)}")
    check("start == MS arrival + HOLD",
          all(r["start"] == r["ms_arrival"] + r["hold"] for r in rows))
    check("TRT == CTT - STT",
          all(r["TRT"] == r["delivery"] - r["t0"] for r in rows))
    check("TRT within [17, ND=42] for every patient",
          all(17 <= r["TRT"] <= ish.ND_EXT for r in rows),
          f"range={min(r['TRT'] for r in rows)}..{max(r['TRT'] for r in rows)}")
    check("S[p] == sigma[TRT[p], tier(p)] (exact integer-day lookup)",
          all(abs(r["survival"] - sig[r["TRT"]][pat[r["pid"]].tier]) < 1e-9
              for r in rows))
    check("delta[p,d] picks exactly the realised TRT",
          all(abs(value(mdl.delta[r["pid"], r["TRT"]]) - 1) < 1e-6 for r in rows))
    check("model S variable agrees with the recomputed survival",
          all(abs(value(mdl.S[r["pid"]]) - r["survival"]) < 1e-6 for r in rows))

    # (34) capacity: occupancy of facility m on day t is the number of jobs
    # started in [t-TMFE, t-1]
    occ = defaultdict(int)
    for r in rows:
        for t in range(r["start"] + 1, r["start"] + net.TMFE + 1):
            occ[(r["facility"], t)] += 1
    viol = [(k, v) for k, v in occ.items() if v > net.FCAP[k[0]]]
    check("(34) concurrent capacity respected at every facility/day",
          not viol, str(viol[:3]))

    check("only opened facilities are used",
          set(r["facility"] for r in rows) <= set(summary["opened"]))
    check("CON1: at most 2 facilities opened", summary["n_opened"] <= 2,
          "+".join(summary["opened"]))

    cost = summary["total_cost"]
    loss = summary["weighted_loss"]
    check("objective == cost + ALPHA * sum rho*(1-S)",
          abs(out.obj - (cost + 1e5 * loss)) < 1.0,
          f"obj={out.obj:,.1f} vs {cost + 1e5*loss:,.1f}")


def test_cumulative_reduction_is_exact():
    """(32) is emitted only on the days where the cumulative RHS is still a
    strict subset of the modes.  Rebuild with ALL cumulative rows and check the
    optimum is unchanged."""
    print("\n[(32) row reduction is exact]")
    net = cd.load_network(DAT)
    inst = small_instance(24)

    reduced = ish.build_extension(net, inst, alpha=1e5)
    r1 = ish.solve(reduced, time_limit=300)

    # full cumulative form, built here explicitly
    full = ish.build_extension(net, inst, alpha=1e5)
    TLS = int(net.TLS)
    for p in inst.patients:
        arr = {j: p.t0 + TLS + net.TT1[j] for j in net.j}
        for m in net.m:
            for t in full.win[p.pid]:
                full.con.add(
                    sum(full.INM[p.pid, m, tau] for tau in full.win[p.pid] if tau <= t)
                    <= sum(full.y1[p.pid, m, j] for j in net.j if arr[j] <= t))
    r2 = ish.solve(full, time_limit=300)
    check("reduced (32) and full cumulative (32) give the same optimum",
          r1.obj is not None and r2.obj is not None and abs(r1.obj - r2.obj) < 1.0,
          f"{r1.obj} vs {r2.obj}")


def test_survival_design_prioritises_high_risk():
    print("\n[emergent priority]")
    net = cd.load_network(DAT)
    inst = cd.build_instance(DAT, mult=2)          # N = 100, real contention
    mdl = ish.build_extension(net, inst, alpha=1e5)
    out = ish.solve(mdl, time_limit=900)
    assert out.status in ("optimal", "feasible"), out.as_dict()
    _, s = ish.extract(mdl, inst, net)
    check("survival design: mean HOLD  H <= M <= L",
          s["mean_HOLD_H"] <= s["mean_HOLD_M"] + 1e-9 <= s["mean_HOLD_L"] + 1e-9,
          f"H={s['mean_HOLD_H']:.2f} M={s['mean_HOLD_M']:.2f} L={s['mean_HOLD_L']:.2f}")
    check("survival design: mean TRT  H <= M <= L",
          s["mean_TRT_H"] <= s["mean_TRT_M"] + 1e-9 <= s["mean_TRT_L"] + 1e-9,
          f"H={s['mean_TRT_H']:.2f} M={s['mean_TRT_M']:.2f} L={s['mean_TRT_L']:.2f}")


def main():
    test_sigma_lookup()
    test_instance_generation()
    test_extension_invariants()
    test_cumulative_reduction_is_exact()
    test_survival_design_prioritises_high_risk()
    print("\n%d failure(s)%s" % (len(FAILS), ": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
