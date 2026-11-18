"""Compact CAR-T scheduling model (same formulation as CART_doe_main.ipynb).

Variable X[p, k] = 1 if patient p uses plan k = (m, j1, j2, t): facility m, outbound mode j1, return mode j2,
manufacturing start day t. For every admissible plan:
    arrival  = a_p + T_LS + TT1[c(p), m, j1]
    wait     = t - arrival
    delivery = t + T_MFE + T_QC + TT2[m, h(p), j2]
    cost     = CVM[m] + TT1*U1 + TT2*U2 + HC*wait
    riskD    = pi[g(p), delivery - a_p, D],  riskU = pi[g(p), delivery - a_p, U]
A plan is admissible when t >= arrival, delivery - a_p <= ND and delivery <= NT.

Constraints
    (R1) one plan per patient
    (R2) capacity: for every facility m and day t, plans at m started in [t - T_MFE + 1, t] <= FCAP[m]
Models
    Baseline      : min total cost, with FIFO (C14-C16) at every facility
    Patient-aware : min total cost + lambda_D * E_D (lambda_U = 0), no FIFO

FIFO order: p before q if a_p < a_q, or a_p == a_q and p has the smaller index.
z[p,m] = sum of X[p,k] over plans k at m;  S[p] = sum of t(k) * X[p,k].

Two equivalent FIFO forms (select with fifo_form):
  "bigM"        S_p <= S_q + NT * (2 - z_pm - z_qm)                     for every m and pair p before q
  "timeindexed" sum_{k of q at m, t<=tau} X_qk <= sum_{k of p at m, t<=tau} X_pk + (1 - z_pm)
                for every m, day tau, pair p before q. Because z_pm = sum of all X_pk at m, this is written as
                sum_{k of q at m, t<=tau} X_qk + sum_{k of p at m, t>tau} X_pk <= 1
                ("q may not have started by tau while p, at the same facility, starts after tau").

Only constraints that can bind are generated (dropping the others does not change the feasible set):
  - a pair (p, q, m) is kept only if both patients have admissible plans at m and the start windows overlap,
    i.e. p's latest start at m > q's earliest start at m (otherwise S_p <= S_q holds for every choice);
  - timeindexed: tau runs over q's admissible start days at m with tau < p's latest start at m. For other tau the
    constraint is either empty (q cannot have started) or implied by the constraint at q's own start day.
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyomo.environ as pyo

FIFO_FORMS = ("bigM", "timeindexed")


# ----------------------------------------------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------------------------------------------
def enumerate_plans(d):
    plans = {}
    for p in d["P"]:
        c, h, a, g = d["SITE"][p], d["HOSP"][p], d["A"][p], d["G"][p]
        for m in d["M"]:
            for j1 in d["J"]:
                for j2 in d["J"]:
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


def index_plans(plans):
    by_p, by_mt, by_pm = defaultdict(list), defaultdict(list), defaultdict(list)
    for k in plans:
        by_p[k[0]].append(k)
        by_mt[k[1], k[4]].append(k)
        by_pm[k[0], k[1]].append(k)
    for key in by_pm:
        by_pm[key].sort(key=lambda k: k[4])
    window = {key: (ks[0][4], ks[-1][4]) for key, ks in by_pm.items()}            # (earliest, latest) start
    starts = {key: sorted({k[4] for k in ks}) for key, ks in by_pm.items()}
    return dict(by_p=by_p, by_mt=by_mt, by_pm=by_pm, window=window, starts=starts)


# ----------------------------------------------------------------------------------------------------------------
# FIFO pairs and size
# ----------------------------------------------------------------------------------------------------------------
def fifo_pairs(d, idx):
    """(m, p, q) with p before q, both admissible at m, and start windows that overlap."""
    order = d["ORDER"]
    for m in d["M"]:
        at_m = [p for p in order if (p, m) in idx["window"]]
        for i, p in enumerate(at_m):
            p_latest = idx["window"][p, m][1]
            for q in at_m[i + 1:]:
                if p_latest > idx["window"][q, m][0]:
                    yield m, p, q


def fifo_size(d, plans, form, idx=None):
    """Number of FIFO constraints and their nonzeros, without building the Pyomo model."""
    idx = idx or index_plans(plans)
    n_con = nnz = 0
    for m, p, q in fifo_pairs(d, idx):
        if form == "bigM":
            n_con += 1
            nnz += len(idx["by_p"][p]) + len(idx["by_p"][q])
        else:
            p_latest = idx["window"][p, m][1]
            q_t = np.array([k[4] for k in idx["by_pm"][q, m]])
            p_t = np.array([k[4] for k in idx["by_pm"][p, m]])
            for tau in idx["starts"][q, m]:
                if tau >= p_latest:
                    break
                n_con += 1
                nnz += int((q_t <= tau).sum() + (p_t > tau).sum())
    return n_con, nnz


# ----------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------
def build_model(d, plans, kind, fifo_form="timeindexed", lambda_D=150_000.0, lambda_U=0.0):
    """kind = "Baseline" (min cost + FIFO) or "Patient-aware" (min cost + lambda_D*E_D, no FIFO)."""
    assert kind in ("Baseline", "Patient-aware")
    assert fifo_form in FIFO_FORMS
    t0 = time.time()
    idx = index_plans(plans)
    mdl = pyo.ConcreteModel(name=f"CART_{kind}")
    mdl.K = pyo.Set(initialize=list(plans), dimen=5, ordered=True)
    mdl.X = pyo.Var(mdl.K, domain=pyo.Binary)
    X = mdl.X
    # (R1) one plan per patient
    mdl.R1 = pyo.Constraint(d["P"], rule=lambda mdl, p: pyo.quicksum(X[k] for k in idx["by_p"][p]) == 1)
    # (R2) capacity
    def r2(mdl, m, t):
        ks = [k for tt in range(max(1, t - d["T_MFE"] + 1), t + 1) for k in idx["by_mt"].get((m, tt), [])]
        return pyo.quicksum(X[k] for k in ks) <= d["FCAP"][m] if ks else pyo.Constraint.Skip
    mdl.R2 = pyo.Constraint(d["M"], range(1, d["NT"] + 1), rule=r2)
    # FIFO (Baseline only)
    mdl.FIFO = pyo.ConstraintList()
    if kind == "Baseline":
        if fifo_form == "bigM":
            S = {p: pyo.quicksum(k[4] * X[k] for k in idx["by_p"][p]) for p in d["P"]}
            z = {key: pyo.quicksum(X[k] for k in ks) for key, ks in idx["by_pm"].items()}
            for m, p, q in fifo_pairs(d, idx):
                mdl.FIFO.add(S[p] <= S[q] + d["NT"] * (2 - z[p, m] - z[q, m]))
        else:
            for m, p, q in fifo_pairs(d, idx):
                p_latest = idx["window"][p, m][1]
                q_plans, p_plans = idx["by_pm"][q, m], idx["by_pm"][p, m]
                for tau in idx["starts"][q, m]:
                    if tau >= p_latest:
                        break
                    mdl.FIFO.add(pyo.quicksum(X[k] for k in q_plans if k[4] <= tau)
                                 + pyo.quicksum(X[k] for k in p_plans if k[4] > tau) <= 1)
    mdl.cost = pyo.Expression(expr=pyo.quicksum(plans[k]["cost"] * X[k] for k in plans))
    mdl.ED = pyo.Expression(expr=pyo.quicksum(plans[k]["riskD"] * X[k] for k in plans))
    mdl.EU = pyo.Expression(expr=pyo.quicksum(plans[k]["riskU"] * X[k] for k in plans))
    if kind == "Baseline":
        mdl.obj = pyo.Objective(expr=mdl.cost, sense=pyo.minimize)
    else:
        mdl.obj = pyo.Objective(expr=mdl.cost + lambda_D * mdl.ED + lambda_U * mdl.EU, sense=pyo.minimize)
    info = dict(n_variables=len(plans), n_constraints=len(mdl.R1) + len(mdl.R2) + len(mdl.FIFO),
                n_fifo_constraints=len(mdl.FIFO), fifo_form=fifo_form if kind == "Baseline" else "",
                build_s=time.time() - t0)
    return mdl, info


# ----------------------------------------------------------------------------------------------------------------
# Solution extraction and checks
# ----------------------------------------------------------------------------------------------------------------
def chosen_plans(d, plans, mdl):
    rows = []
    for k in plans:
        v = mdl.X[k].value
        if v is not None and v > 0.5:
            p, m, j1, j2, t = k
            a = plans[k]
            rows.append({"p": p, "class": d["G"][p], "site": d["SITE"][p], "hospital": d["HOSP"][p], "facility": m,
                         "out mode": d["MODE"][j1], "return mode": d["MODE"][j2], "collection day": d["A"][p],
                         "arrival": a["arrival"], "start": t, "delivery": a["delivery"], "TRT": a["TRT"], "W_p": a["wait"],
                         "plan cost": a["cost"], "pi_D at delivery": a["riskD"], "pi_U at delivery": a["riskU"],
                         "j1": j1, "j2": j2})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("p", key=lambda s: s.str[1:].astype(int)).reset_index(drop=True)
    return df


def check_solution(d, plans, tab, ED_solver, fifo):
    """Checks recomputed from the raw data. Returns {check name: bool}."""
    out = {}
    out["every patient has exactly one plan"] = len(tab) == len(d["P"]) and set(tab.p) == set(d["P"])
    if not len(tab):
        return out
    keys = [(r.p, r.facility, r.j1, r.j2, int(r.start)) for r in tab.itertuples()]
    tt1 = np.array([d["TT1"][r.site, r.facility, r.j1] for r in tab.itertuples()])
    tt2 = np.array([d["TT2"][r.facility, r.hospital, r.j2] for r in tab.itertuples()])
    a = tab["collection day"].values
    arrival = a + d["T_LS"] + tt1
    delivery = tab.start.values + d["T_MFE"] + d["T_QC"] + tt2
    out["every plan admissible (start >= arrival, delivery <= NT, paired hospital)"] = bool(
        all(k in plans for k in keys) and (tab.start.values >= arrival).all() and (delivery <= d["NT"]).all()
        and (tab.hospital.values == tab.site.map(d["PAIRED_H"]).values).all())
    out["capacity respected every day"] = all(
        int(((tab.facility == m) & (tab.start >= t - d["T_MFE"] + 1) & (tab.start <= t)).sum()) <= d["FCAP"][m]
        for m in d["M"] for t in range(1, d["NT"] + 1))
    out[f"TRT <= ND ({d['ND']})"] = bool((delivery - a <= d["ND"]).all())
    ed = sum(d["PI"][g, int(dl - aa), "D"] for g, dl, aa in zip(tab["class"], delivery, a))
    out["E_D recomputed from the plans = solver value (1e-6)"] = abs(ed - ED_solver) <= 1e-6
    if fifo:
        rank = {p: i for i, p in enumerate(d["ORDER"])}
        ok = True
        for m in d["M"]:
            s = tab[tab.facility == m].assign(_o=lambda x: x.p.map(rank)).sort_values("_o").start.values
            ok &= bool((np.diff(s) >= 0).all())
        out["FIFO holds at every facility"] = ok
    return out


def waiting_by_day(d, tab):
    """Number of patients waiting at each facility on each day (arrival <= day < start)."""
    days = np.arange(1, d["NT"] + 1)
    out = pd.DataFrame({"day": days})
    for m in d["M"]:
        sub = tab[tab.facility == m]
        out[m] = [int(((sub.arrival <= t) & (sub.start > t)).sum()) for t in days]
    return out
