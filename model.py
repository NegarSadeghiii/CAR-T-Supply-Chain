"""Compact CAR-T scheduling model, epsilon-constraint version (as in CART_epsilon.py, branch epsilon-constraint).

Variable X[p, k] = 1 if patient p uses plan k = (m, j1, j2, t): facility m, outbound mode j1, return mode j2,
manufacturing start day t. Precomputed for every admissible plan (t >= arrival, TRT <= ND, delivery <= NT):
    arrival = a_p + T_LS + TT1,  wait = t - arrival,  delivery = t + T_MFE + T_QC + TT2
    cost = CVM + TT1*U1 + TT2*U2 + HC*wait,  riskD / riskU = pi[g_p, delivery - a_p, D / U]
Constraints: one plan per patient (C9); time-indexed capacity (C10); FIFO (C14-C16, Baseline-FIFO only).
Every model minimises total cost (cost only, no weights); E_D = sum riskD*X is either free, used as a tie-break,
or limited from above (E_D <= epsilon). E_U = sum riskU*X is reported, never optimised.
"""
import itertools
import math
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo

EPS_TOL = 1e-7          # numerical slack on E_D <= epsilon (the endpoints equal E_D values of existing plans)


def enumerate_plans(d):
    plans = {}
    for p in d["P"]:
        c, h, a, g = d["SITE"][p], d["HOSP"][p], d["A"][p], d["G"][p]
        for m, j1, j2 in itertools.product(d["M"], d["J"], d["J"]):
            tt1, tt2 = d["TT1"][c, m, j1], d["TT2"][m, h, j2]
            arrival = a + d["T_LS"] + tt1
            for t in range(arrival, d["NT"] + 1):
                delivery = t + d["T_MFE"] + d["T_QC"] + tt2
                if delivery - a > d["ND"] or delivery > d["NT"]:
                    break
                wait = t - arrival
                plans[p, m, j1, j2, t] = dict(
                    arrival=arrival, wait=wait, delivery=delivery, TRT=delivery - a,
                    cost=d["CVM"][m] + tt1 * d["U1"][c, m, j1] + tt2 * d["U2"][m, h, j2] + d["HC"] * wait,
                    riskD=d["PI"][g, delivery - a, "D"], riskU=d["PI"][g, delivery - a, "U"])
    return plans


def build_model(d, plans, fifo):
    """fifo=True: Baseline-FIFO. fifo=False: the model shared by Baseline-MinRisk, the ED_cap diagnostic and
    PatientAware; its side constraints (cost cap, E_D limit) are switched on per solve."""
    mdl = pyo.ConcreteModel()
    mdl.K = pyo.Set(initialize=list(plans), dimen=5, ordered=True)
    mdl.X = pyo.Var(mdl.K, domain=pyo.Binary)
    by_p, by_mt = {}, {}
    for k in plans:
        by_p.setdefault(k[0], []).append(k)
        by_mt.setdefault((k[1], k[4]), []).append(k)
    mdl.R1 = pyo.Constraint(d["P"], rule=lambda mdl, p: sum(mdl.X[k] for k in by_p[p]) == 1)          # C9
    def r2(mdl, m, t):                                                                                    # C10
        ks = [k for tt in range(max(1, t - d["T_MFE"] + 1), t + 1) for k in by_mt.get((m, tt), [])]
        return sum(mdl.X[k] for k in ks) <= d["FCAP"][m] if ks else pyo.Constraint.Skip
    mdl.R2 = pyo.Constraint(d["M"], range(1, d["NT"] + 1), rule=r2)
    mdl.FIFO = pyo.ConstraintList()                                                                       # C14-C16
    if fifo:
        z = {(p, m): sum(mdl.X[k] for k in by_p[p] if k[1] == m) for p in d["P"] for m in d["M"]}
        S = {p: sum(k[4] * mdl.X[k] for k in by_p[p]) for p in d["P"]}
        order = d["ORDER"]
        for m in d["M"]:
            for i, p in enumerate(order):
                for q in order[i + 1:]:
                    mdl.FIFO.add(S[p] <= S[q] + d["NT"] * (2 - z[p, m] - z[q, m]))
    mdl.cost = pyo.Expression(expr=sum(plans[k]["cost"] * mdl.X[k] for k in plans))
    mdl.ED = pyo.Expression(expr=sum(plans[k]["riskD"] * mdl.X[k] for k in plans))
    mdl.EU = pyo.Expression(expr=sum(plans[k]["riskU"] * mdl.X[k] for k in plans))
    mdl.obj_cost = pyo.Objective(expr=mdl.cost, sense=pyo.minimize)
    mdl.obj_ED = pyo.Objective(expr=mdl.ED, sense=pyo.minimize)
    mdl.obj_ED.deactivate()
    mdl.cost_cap_rhs = pyo.Param(mutable=True, initialize=0.0)
    mdl.cost_cap = pyo.Constraint(expr=mdl.cost <= mdl.cost_cap_rhs)
    mdl.cost_cap.deactivate()
    mdl.eps = pyo.Param(mutable=True, initialize=0.0)
    mdl.ED_cap = pyo.Constraint(expr=mdl.ED <= mdl.eps + EPS_TOL)
    mdl.ED_cap.deactivate()
    return mdl


# ----------------------------------------------------------------------------------------------------------------
# Solvers (Pyomo appsi interface; one solver object per model, reused across that model's solves)
# ----------------------------------------------------------------------------------------------------------------
def new_solver(name, time_limit, threads=1):
    if name == "highs":
        from pyomo.contrib.appsi.solvers import Highs
        opt = Highs()
        opt.highs_options = {"mip_abs_gap": 1e-9, "random_seed": 0}
        if threads:
            opt.highs_options["threads"] = threads
    elif name == "gurobi":
        from pyomo.contrib.appsi.solvers import Gurobi
        opt = Gurobi()
        avail = opt.available()
        if not avail:
            raise RuntimeError(f"Gurobi not usable ({avail.name}): install gurobipy and a Gurobi license")
        opt.gurobi_options = {"MIPGapAbs": 1e-9, "Seed": 0}
        if threads:
            opt.gurobi_options["Threads"] = threads
    else:
        raise ValueError(f"unknown solver {name!r}")
    opt.config.time_limit = time_limit
    opt.config.mip_gap = 0.0
    opt.config.load_solution = False
    return opt


def solve(opt, mdl, objective="cost", cost_cap=None, eps=None):
    """objective 'cost' or 'ED'; cost_cap / eps switch the side constraints on (None = off)."""
    from pyomo.contrib.appsi.base import TerminationCondition as TC
    (mdl.obj_cost.activate, mdl.obj_cost.deactivate)[objective != "cost"]()
    (mdl.obj_ED.activate, mdl.obj_ED.deactivate)[objective != "ED"]()
    for con, par, val in ((mdl.cost_cap, mdl.cost_cap_rhs, cost_cap), (mdl.ED_cap, mdl.eps, eps)):
        if val is None:
            con.deactivate()
        else:
            par.set_value(val)
            con.activate()
    t0 = time.time()
    res = opt.solve(mdl)
    secs = time.time() - t0
    tc = res.termination_condition
    obj, bound = res.best_feasible_objective, res.best_objective_bound
    has_sol = obj is not None and tc in (TC.optimal, TC.maxTimeLimit, TC.interrupted, TC.maxIterations)
    if has_sol:
        res.solution_loader.load_vars()
    if tc == TC.optimal and has_sol:
        status = "optimal"
    elif tc in (TC.infeasible, TC.infeasibleOrUnbounded):
        status = "infeasible"
    elif tc == TC.maxTimeLimit:
        status = "time limit, feasible" if has_sol else "time limit, no solution"
    else:
        status = f"{tc.name}{', feasible' if has_sol else ''}"
    gap = (abs(obj - bound) / max(abs(obj), 1e-10)) if has_sol and bound is not None and math.isfinite(bound) else np.nan
    return dict(status=status, has_solution=has_sol, objective=obj if has_sol else np.nan,
                best_bound=bound if bound is not None else np.nan, mip_gap=gap, solve_s=secs)


# ----------------------------------------------------------------------------------------------------------------
# Solution table and checks
# ----------------------------------------------------------------------------------------------------------------
def plan_table(d, plans, mdl):
    rows = []
    for k in plans:
        v = mdl.X[k].value
        if v is not None and v > 0.5:
            p, m, j1, j2, t = k
            a = plans[k]
            rows.append(dict(p=p, cls=d["G"][p], site=d["SITE"][p], hospital=d["HOSP"][p], facility=m,
                             nearest=d["NEAREST"][d["SITE"][p]], j1=j1, j2=j2, collection=d["A"][p],
                             arrival=a["arrival"], start=t, delivery=a["delivery"], TRT=a["TRT"], W=a["wait"],
                             cost=a["cost"], riskD=a["riskD"], riskU=a["riskU"]))
    return pd.DataFrame(rows)


def check_plan(d, plans, tab, mdl, fifo, eps=None):
    out = {}
    out["one plan per patient"] = len(tab) == len(d["P"]) and set(tab.p) == set(d["P"])
    keys = [(r.p, r.facility, r.j1, r.j2, int(r.start)) for r in tab.itertuples()]
    tt1 = np.array([d["TT1"][r.site, r.facility, r.j1] for r in tab.itertuples()])
    tt2 = np.array([d["TT2"][r.facility, r.hospital, r.j2] for r in tab.itertuples()])
    a = tab.collection.values
    arrival = a + d["T_LS"] + tt1
    delivery = tab.start.values + d["T_MFE"] + d["T_QC"] + tt2
    out["every plan admissible"] = bool(all(k in plans for k in keys) and (tab.start.values >= arrival).all()
                                        and (delivery <= d["NT"]).all())
    out["capacity respected"] = all(
        int(((tab.facility == m) & (tab.start >= t - d["T_MFE"] + 1) & (tab.start <= t)).sum()) <= d["FCAP"][m]
        for m in d["M"] for t in range(1, d["NT"] + 1))
    out["TRT <= ND"] = bool((delivery - a <= d["ND"]).all())
    ed = sum(d["PI"][g, int(dl - aa), "D"] for g, dl, aa in zip(tab.cls, delivery, a))
    out["E_D recomputed = solver value (1e-6)"] = abs(ed - pyo.value(mdl.ED)) <= 1e-6
    if eps is not None:
        out["E_D <= epsilon"] = ed <= eps + 1e-6
    if fifo:
        rank = {p: i for i, p in enumerate(d["ORDER"])}
        ok = True
        for m in d["M"]:
            s = tab[tab.facility == m].assign(o=lambda x: x.p.map(rank)).sort_values("o").start.values
            ok &= bool((np.diff(s) >= 0).all())
        out["FIFO holds"] = ok
    return out


def plan_metrics(tab):
    u, r = tab[tab.cls == "U"], tab[tab.cls == "R"]
    return {"cost": tab.cost.sum(), "E_D": tab.riskD.sum(), "E_U": tab.riskU.sum(),
            "mean wait urgent": u.W.mean() if len(u) else np.nan, "max wait urgent": u.W.max() if len(u) else np.nan,
            "mean wait regular": r.W.mean() if len(r) else np.nan, "max wait regular": r.W.max() if len(r) else np.nan,
            "non-nearest assignments": int((tab.facility != tab.nearest).sum()), "max TRT": int(tab.TRT.max())}
