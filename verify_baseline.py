"""
Equivalence check: the index-reduced baseline in ``ishipment_survival.py`` vs.
the FULL-index i-SHIPMENT MILP transcribed verbatim from the (unmodified)
notebook ``i-SHIPMENT_Pyomo.ipynb``.

Both models are built on the same patients, drawn from ``Data200_profileA.dat``,
and solved with the same solver.  They must agree on the optimal objective, the
set of facilities opened and the TRT distribution.

The full-index model is declared over the raw index sets, so it explodes: at the
full 50-patient cohort Y1 alone is 50*4*6*2*130 = 312,000 binaries and Pyomo
needs >8 GB merely to construct the ~2.5M constraints -- which is precisely why
the study itself uses the index-reduced form.  The equivalence is therefore
checked on a small sub-cohort (12 patients by default, ``--n-patients``), where
both models are solvable to proven optimality.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

from pyomo.environ import (AbstractModel, Set, RangeSet, Param, Var, Binary,
                           NonNegativeReals, NonNegativeIntegers, Objective,
                           Constraint, minimize, value)

import cart_data as cd
import ishipment_survival as ish


def subset_dat(src_path, out_path, keep):
    """Write a .dat identical to `src_path` but restricted to the patients in
    `keep` (same sets, params, costs, arrivals).  Used so the full-index model
    stays buildable: at 50 patients it declares ~2.5M constraints and needs
    >8 GB just to construct, which is exactly why the study uses the
    index-reduced form."""
    text = open(src_path).read()
    keep = list(keep)
    text = re.sub(r"set p :=[^;]*;",
                  "set p := " + " ".join(keep) + " ;", text, flags=re.S)
    m = re.search(r"param INC :=(.*?);", text, flags=re.S)
    lines = [ln for ln in m.group(1).strip().splitlines()
             if ln.split() and ln.split()[0] in keep]
    text = text[:m.start()] + "param INC := \n" + "\n".join(lines) + "\n;" + text[m.end():]
    open(out_path, "w").write(text)
    return out_path


def build_full_index_baseline(data_file="Data200_profileA.dat"):
    """Transcribed from i-SHIPMENT_Pyomo.ipynb (cell 0) without changes."""
    model = AbstractModel()
    model.c = Set()
    model.h = Set()
    model.j = Set()
    model.m = Set()
    model.p = Set()
    model.t = RangeSet(130)
    model.tt = Set(initialize=model.t)

    model.CIM = Param(model.m)
    model.FCAP = Param(model.m)
    model.TT1 = Param(model.j)
    model.TT3 = Param(model.j)
    model.U1 = Param(model.c, model.m, model.j)
    model.U3 = Param(model.m, model.h, model.j)
    model.INC = Param(model.p, model.c, model.t, initialize=0)
    model.CVM = Param(model.m)

    model.FMAX = Param()
    model.FMIN = Param()
    model.TAD = Param(within=NonNegativeReals)
    model.TLS = Param(within=NonNegativeReals)
    model.TMFE = Param(default=7)
    model.TQC = Param(default=7)
    model.C_material = Param(default=10476)
    model.CQC = Param(default=9312)
    model.ND = Param(default=18)

    model.E1 = Var(model.m, within=Binary)
    model.X1 = Var(model.c, model.m, within=Binary)
    model.X2 = Var(model.m, model.h, within=Binary)
    model.Y1 = Var(model.p, model.c, model.m, model.j, model.t, within=Binary)
    model.Y2 = Var(model.p, model.m, model.h, model.j, model.t, within=Binary)
    model.INH = Var(model.p, model.h, model.t, within=NonNegativeIntegers)

    model.CTM = Var(model.p, within=NonNegativeReals)
    model.FTD = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
    model.TTC = Var(model.p, within=NonNegativeReals)
    model.LSA = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
    model.LSR = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
    model.MSO = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
    model.OUTC = Var(model.p, model.c, model.t, within=NonNegativeReals)
    model.OUTM = Var(model.p, model.m, model.t, within=NonNegativeReals)
    model.INM = Var(model.p, model.m, model.t, within=NonNegativeReals)
    model.DURV = Var(model.p, model.m, model.t, within=NonNegativeReals)
    model.RATIO = Var(model.m, model.t, within=NonNegativeReals)

    model.TOTCOST = Var()
    model.CAP = Var(model.m, model.t)
    model.TRT = Var(model.p)
    model.ATRT = Var()
    model.STT = Var(model.p)
    model.CTT = Var(model.p)

    def obj_rule(m_):
        return (sum(m_.CTM[p] for p in m_.p) + sum(m_.TTC[p] for p in m_.p)
                + (m_.C_material + m_.CQC) * len(m_.p))
    model.OBJ = Objective(rule=obj_rule, sense=minimize)

    def C1_rule(m_, p):
        return m_.CTM[p] == sum((m_.E1[m] * (m_.CIM[m] + m_.CVM[m]))
                                * len(m_.t) / len(m_.p) for m in m_.m)
    model.C1 = Constraint(model.p, rule=C1_rule)

    def C2_rule(m_, p):
        return m_.TTC[p] == (
            sum(m_.Y1[p, c, m, j, t] * m_.U1[c, m, j]
                for c in m_.c for m in m_.m for j in m_.j for t in m_.t)
            + sum(m_.Y2[p, m, h, j, t] * m_.U3[m, h, j]
                  for m in m_.m for h in m_.h for j in m_.j for t in m_.t))
    model.C2 = Constraint(model.p, rule=C2_rule)

    def RATIOEQ_rule(m_, m, t):
        return m_.RATIO[m, t] == sum(m_.DURV[p, m, t] / m_.FCAP[m] for p in m_.p)
    model.RATIOEQ = Constraint(model.m, model.t, rule=RATIOEQ_rule)

    def MSBnew_rule(m_, p, m, t):
        return m_.DURV[p, m, t] == sum(m_.INM[p, m, tt - 1] - m_.OUTM[p, m, tt]
                                       for tt in m_.tt if 1 < tt <= t) + m_.OUTM[p, m, t]
    model.MSBnew = Constraint(model.p, model.m, model.t, rule=MSBnew_rule)

    def MSB1_rule(m_, p, c, t, tt):
        return m_.INC[p, c, t] == m_.OUTC[p, c, tt] if tt == t + m_.TLS else Constraint.Skip
    model.MSB1 = Constraint(model.p, model.c, model.t, model.tt, rule=MSB1_rule)

    def MSB3_rule(m_, p, c, m, j, t, tt):
        return (m_.LSR[p, c, m, j, t] == m_.LSA[p, c, m, j, tt]
                if tt == t + m_.TT1[j] else Constraint.Skip)
    model.MSB3 = Constraint(model.p, model.c, model.m, model.j, model.t, model.tt,
                            rule=MSB3_rule)

    def MSB7_rule(m_, p, c, t):
        return m_.OUTC[p, c, t] == sum(m_.LSR[p, c, m, j, t] for m in m_.m for j in m_.j)
    model.MSB7 = Constraint(model.p, model.c, model.t, rule=MSB7_rule)

    def MSB5_rule(m_, p, m, t):
        return m_.INM[p, m, t] == sum(m_.LSA[p, c, m, j, t] for c in m_.c for j in m_.j)
    model.MSB5 = Constraint(model.p, model.m, model.t, rule=MSB5_rule)

    def MSB2_rule(m_, p, m, t, tt):
        return (m_.INM[p, m, t] == m_.OUTM[p, m, tt]
                if tt == t + m_.TMFE else Constraint.Skip)
    model.MSB2 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB2_rule)

    def MSB8_rule(m_, p, m, t, tt):
        return (m_.OUTM[p, m, t] == sum(m_.MSO[p, m, h, j, tt] for h in m_.h for j in m_.j)
                if tt == t + m_.TQC else Constraint.Skip)
    model.MSB8 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB8_rule)

    def MSB4_rule(m_, p, m, h, j, t, tt):
        return (m_.MSO[p, m, h, j, t] == m_.FTD[p, m, h, j, tt]
                if tt == t + m_.TT3[j] else Constraint.Skip)
    model.MSB4 = Constraint(model.p, model.m, model.h, model.j, model.t, model.tt,
                            rule=MSB4_rule)

    def MSB6_rule(m_, p, h, t):
        return m_.INH[p, h, t] == sum(m_.FTD[p, m, h, j, t] for m in m_.m for j in m_.j)
    model.MSB6 = Constraint(model.p, model.h, model.t, rule=MSB6_rule)

    def CAP1_rule(m_, m, t):
        return m_.CAP[m, t] == m_.FCAP[m] - sum(
            m_.INM[p, m, tt] for p in m_.p for tt in m_.tt if t - m_.TMFE <= tt < t)
    model.CAP1 = Constraint(model.m, model.t, rule=CAP1_rule)

    def CAPCON1_rule(m_, m, t):
        return (sum(m_.INM[p, m, t] for p in m_.p)
                - sum(m_.OUTM[p, m, t] for p in m_.p) <= m_.CAP[m, t])
    model.CAPCON1 = Constraint(model.m, model.t, rule=CAPCON1_rule)

    def CON1_rule(m_):
        return sum(m_.E1[m] for m in m_.m) <= 2
    model.CON1 = Constraint(rule=CON1_rule)

    model.CON2 = Constraint(model.c, model.m, rule=lambda m_, c, m: m_.X1[c, m] <= m_.E1[m])
    model.CON3 = Constraint(model.m, model.h, rule=lambda m_, m, h: m_.X2[m, h] <= m_.E1[m])
    model.CON4 = Constraint(model.p, model.c, model.m, model.j, model.t,
                            rule=lambda m_, p, c, m, j, t: m_.Y1[p, c, m, j, t] <= m_.X1[c, m])
    model.CON5 = Constraint(model.p, model.m, model.h, model.j, model.t,
                            rule=lambda m_, p, m, h, j, t: m_.Y2[p, m, h, j, t] <= m_.X2[m, h])
    model.CON6 = Constraint(model.p, rule=lambda m_, p: sum(
        m_.Y1[p, c, m, j, t] for c in m_.c for m in m_.m for j in m_.j for t in m_.t) == 1)
    model.CON7 = Constraint(model.p, rule=lambda m_, p: sum(
        m_.Y2[p, m, h, j, t] for m in m_.m for h in m_.h for j in m_.j for t in m_.t) == 1)
    model.DEM = Constraint(rule=lambda m_: sum(
        m_.INH[p, h, t] for p in m_.p for h in m_.h for t in m_.t) <= len(m_.p))
    model.CON8 = Constraint(model.p, model.c, model.m, model.j, model.t,
                            rule=lambda m_, p, c, m, j, t:
                            m_.LSR[p, c, m, j, t] >= m_.Y1[p, c, m, j, t] * m_.FMIN)
    model.CON9 = Constraint(model.p, model.c, model.m, model.j, model.t,
                            rule=lambda m_, p, c, m, j, t:
                            m_.LSR[p, c, m, j, t] <= m_.Y1[p, c, m, j, t] * m_.FMAX)
    model.CON10 = Constraint(model.p, model.m, model.h, model.j, model.t,
                             rule=lambda m_, p, m, h, j, t:
                             m_.MSO[p, m, h, j, t] >= m_.Y2[p, m, h, j, t] * m_.FMIN)
    model.CON11 = Constraint(model.p, model.m, model.h, model.j, model.t,
                             rule=lambda m_, p, m, h, j, t:
                             m_.MSO[p, m, h, j, t] <= m_.Y2[p, m, h, j, t] * m_.FMAX)

    def _con_hosp(m_, p, hh, cc):
        return (sum(m_.Y2[p, m, hh, j, t] for m in m_.m for j in m_.j for t in m_.t)
                == sum(m_.INC[p, cc, t] for t in m_.t))
    model.CON12 = Constraint(model.p, rule=lambda m_, p: _con_hosp(m_, p, 'h1', 'c1'))
    model.CON13 = Constraint(model.p, rule=lambda m_, p: _con_hosp(m_, p, 'h2', 'c2'))
    model.CON14 = Constraint(model.p, rule=lambda m_, p: _con_hosp(m_, p, 'h3', 'c3'))
    model.CON15 = Constraint(model.p, rule=lambda m_, p: _con_hosp(m_, p, 'h4', 'c4'))

    model.TCON = Constraint(model.p, rule=lambda m_, p: m_.TRT[p] <= m_.ND)
    model.START = Constraint(model.p, rule=lambda m_, p: m_.STT[p] == sum(
        m_.INC[p, c, t] * t for c in m_.c for t in m_.t))
    model.END = Constraint(model.p, rule=lambda m_, p: m_.CTT[p] == sum(
        m_.INH[p, h, t] * t for h in m_.h for t in m_.t))
    model.TSEQ = Constraint(model.p, rule=lambda m_, p: m_.STT[p] <= m_.CTT[p])
    model.TIME = Constraint(model.p, rule=lambda m_, p: m_.TRT[p] == m_.CTT[p] - m_.STT[p])
    model.ATIME = Constraint(rule=lambda m_: m_.ATRT == sum(
        m_.TRT[p] for p in m_.p) / len(m_.p))
    return model.create_instance(data_file)


def main():
    import argparse, os, tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=12,
                    help="size of the equivalence instance (the full-index "
                         "model is only buildable at small N)")
    ap.add_argument("--time-limit", type=int, default=1800)
    args = ap.parse_args()

    dat = "Data200_profileA.dat"
    base = cd.base_cohort(dat)
    keep = [p for p, _, _ in base][:args.n_patients]
    sub = os.path.join(tempfile.gettempdir(), f"subset_{args.n_patients}.dat")
    subset_dat(dat, sub, keep)

    net = cd.load_network(sub)
    inst = cd.build_instance(sub, mult=1)
    base_days = {p: t for p, c, t in cd.base_cohort(sub)}
    for pat in inst.patients:                      # use the ORIGINAL arrival days
        pat.t0 = base_days[pat.pid]

    print(f"Equivalence instance: {len(keep)} patients from the base cohort\n")

    print("[1/2] index-reduced baseline ...")
    red = ish.build_baseline(net, inst)
    r1 = ish.solve(red, time_limit=args.time_limit)
    rows_r, sum_r = ish.extract(red, inst, net)
    trt_red = dict(sorted(Counter(r["TRT"] for r in rows_r).items()))
    print("     status=%s obj=%.2f opened=%s TRT=%s"
          % (r1.status, r1.obj, sum_r["opened"], trt_red))

    print("[2/2] full-index baseline transcribed from the notebook ...")
    full = build_full_index_baseline(sub)
    nv = sum(len(v) for v in full.component_objects(Var))
    nc = sum(len(c) for c in full.component_objects(Constraint))
    print(f"     full-index size: {nv:,} variables / {nc:,} constraints")
    r2 = ish.solve(full, time_limit=args.time_limit)
    trt_full = dict(sorted(Counter(int(round(value(full.TRT[p])))
                                   for p in full.p).items()))
    opened_full = [m for m in full.m if value(full.E1[m]) > 0.5]
    print("     status=%s obj=%.2f opened=%s TRT=%s"
          % (r2.status, r2.obj, opened_full, trt_full))

    ok = (abs(r1.obj - r2.obj) < 1.0
          and sorted(sum_r["opened"]) == sorted(opened_full)
          and trt_red == trt_full)
    print("\nEQUIVALENCE:", "PASS" if ok else "MISMATCH")

    red_size = {"variables": sum(len(v) for v in red.component_objects(Var)),
                "constraints": sum(len(c) for c in red.component_objects(Constraint))}
    ish.write_json(f"{ish.RESULTS}/verification_baseline_equivalence.json", {
        "n_patients": len(keep),
        "reduced": {"status": r1.status, "objective": r1.obj,
                    "opened": sum_r["opened"], "TRT_distribution": trt_red,
                    "wall_s": r1.wall, "solver": r1.solver, **red_size},
        "full_index_from_notebook": {"status": r2.status, "objective": r2.obj,
                                     "opened": opened_full,
                                     "TRT_distribution": trt_full,
                                     "wall_s": r2.wall, "solver": r2.solver,
                                     "variables": nv, "constraints": nc},
        "equivalent": bool(ok),
    })
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
