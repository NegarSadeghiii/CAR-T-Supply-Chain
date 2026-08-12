"""
ishipment_survival.py
=====================

Survival-aware SCHEDULING extension of i-SHIPMENT (Pyomo + Gurobi).

The BASELINE model is `i-SHIPMENT_Pyomo.ipynb` and is **never modified**.
This file contains

  * ``build_baseline``  -- a faithful, index-reduced re-statement of the
    baseline i-SHIPMENT MILP (no survival, no queue, INM = arrival, ND = 18);
  * ``build_extension`` -- the queue + survival extension (eqs. 32-35 plus the
    exact integer-day survival lookup and the relaxed ND = 42);
  * the four experimental phases and all reporting.

Index reduction (exactness note)
--------------------------------
The baseline declares Y1 over (p, c, m, j, t) and Y2 over (p, m, h, j, t).
In every feasible baseline solution those index sets collapse:

  * MSB1/MSB7 + CON6 force the single Y1 of patient p onto its own
    leukapheresis site ``c_p`` and onto the single day ``t = t0_p + TLS``;
  * CON12-CON15 + CON7 force the single Y2 onto the hospital ``h_p``
    co-located with ``c_p``.

So Y1 -> y1[p, m, j] and Y2 -> y2[p, m, j] without changing the feasible set
or the objective.  Likewise CAP1 + CAPCON1 + MSB2 reduce algebraically to the
7-day rolling start-window capacity

    sum_p sum_{tau = t-TMFE+1}^{t} INM[p, m, tau]  <=  FCAP[m],

and MSBnew makes DURV[p,m,t] = 1 exactly on t in [start+1, start+TMFE], i.e.
eq. (34) is the same rolling window shifted by one day.  All of this is
verified numerically at the 50-patient scale by ``--verify``.

Usage
-----
    python ishipment_survival.py --phase all
    python ishipment_survival.py --phase 1 --scales 100 200 500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

from pyomo.environ import (ConcreteModel, Var, Binary, NonNegativeReals, Reals,
                           Objective, Constraint, ConstraintList, minimize,
                           value, SolverFactory, Set, RangeSet)

import cart_data as cd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------------------
# Study-level constants
# ---------------------------------------------------------------------------
ND_BASELINE = 18          # baseline maximum turnaround time
ND_EXT = 42               # eq. (31): relaxed turnaround cap for the extension
MAX_FACILITIES = 2        # CON1, centralised network (the default)

# i-SHIPMENT does not hold CON1 fixed across demand: it runs the centralised
# 2-facility configuration at moderate demand and relaxes to 3 facilities at
# high demand.  This study matches that configuration, so the baseline is given
# the same budget the paper would give it before being compared with the
# extension.  Overridable with --max-facilities.
MAX_FACILITIES_BY_SCALE = {100: 2, 200: 2, 500: 3}


def max_facilities_for(n, override=None):
    """CON1 for demand scale n -- matches i-SHIPMENT's centralised vs
    high-demand configuration (2 facilities at N = 100/200, 3 at N = 500)."""
    if override:
        return int(override)
    return MAX_FACILITIES_BY_SCALE.get(n, MAX_FACILITIES)


def caps_for(n, override=None):
    """Facility caps to report at scale n: the configured one first, plus the
    strict centralised cap when they differ (so both are on the record)."""
    primary = max_facilities_for(n, override)
    return [primary] if primary == MAX_FACILITIES else [primary, MAX_FACILITIES]

# Cost-equivalence weight used for the "survival design" of Phase 3.
# The objective is  Z = cost[$] + ALPHA * sum_p rho_u(p) * (1 - S[p]).
# ALPHA therefore has units of $ per unit of rho-weighted expected loss.
ALPHA_SURVIVAL = 1.0e5

# Phase-4 sweep: the values requested in the brief, plus a log-extension that
# actually reaches the region where the NETWORK DESIGN flips (see README).
ALPHA_SWEEP_REQUESTED = [0, 0.5, 1, 2, 5, 10, 50]
ALPHA_SWEEP_EXTENDED = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

DEFAULT_TIME_LIMIT = 600
DEFAULT_MIP_GAP = 1e-4


# ---------------------------------------------------------------------------
# Solver plumbing (Gurobi first, documented fallback)
# ---------------------------------------------------------------------------
class SolveOutcome:
    def __init__(self, status, obj=None, bound=None, gap=None, wall=None,
                 solver=None, note=""):
        self.status = status          # 'optimal' | 'feasible' | 'infeasible' | 'error'
        self.obj = obj
        self.bound = bound
        self.gap = gap
        self.wall = wall
        self.solver = solver
        self.note = note

    def as_dict(self):
        return {"status": self.status, "objective": self.obj, "bound": self.bound,
                "gap": self.gap, "wall_s": self.wall, "solver": self.solver,
                "note": self.note}


GUROBI_FALLBACK_NOTE = (
    "Gurobi is requested first on every solve, but refuses the study-scale "
    "models: \"Model too large for size-limited license\". This container "
    "carries only the pip size-limited Gurobi licence (2000 variables / 2000 "
    "constraints) and every model here is far larger, so those solves fall back "
    "to HiGHS (open-source MILP, driven through the same Pyomo interface). "
    "Re-run with a full Gurobi licence to use Gurobi throughout.")

_SOLVER_CACHE = {"name": None, "note": "", "used": set()}


def _try_gurobi(model, time_limit, mip_gap):
    opt = SolverFactory("gurobi_direct")
    if not opt.available(exception_flag=False):
        raise RuntimeError("gurobi_direct not available")
    opt.options["TimeLimit"] = time_limit
    opt.options["MIPGap"] = mip_gap
    opt.options["Threads"] = 0
    opt.options["Seed"] = 0
    return opt.solve(model, tee=False, load_solutions=True)


def solve(model, time_limit=DEFAULT_TIME_LIMIT, mip_gap=DEFAULT_MIP_GAP,
          solver_pref="gurobi", tee=False):
    """Solve with Gurobi; fall back to HiGHS if the Gurobi licence is too small."""
    t0 = time.time()

    if solver_pref == "gurobi":
        try:
            res = _try_gurobi(model, time_limit, mip_gap)
            _SOLVER_CACHE["used"].add("gurobi")
            _SOLVER_CACHE["name"] = _SOLVER_CACHE["name"] or "gurobi"
            return _interpret_pyomo(res, model, time.time() - t0, "gurobi")
        except Exception as exc:                       # noqa: BLE001
            # Gurobi is always asked first.  The only expected failure here is
            # the pip size-limited licence (2000 vars / 2000 constraints), which
            # every study-scale model exceeds; fall back to HiGHS and say so.
            msg = str(exc).strip().splitlines()[0][:160]
            if not _SOLVER_CACHE["note"]:
                _SOLVER_CACHE["note"] = GUROBI_FALLBACK_NOTE
                print("[solver] %s (first refusal: %s)"
                      % (GUROBI_FALLBACK_NOTE, msg), file=sys.stderr)

    from pyomo.contrib.appsi.solvers import Highs
    opt = Highs()
    opt.config.time_limit = time_limit
    opt.config.mip_gap = mip_gap
    opt.config.load_solution = True
    opt.config.stream_solver = tee
    opt.highs_options["random_seed"] = 0
    try:
        res = opt.solve(model)
    except Exception as exc:                           # noqa: BLE001
        msg = str(exc)
        if "load_solution" in msg or "no feasible" in msg.lower():
            opt.config.load_solution = False
            try:
                res = opt.solve(model)
                return _interpret_appsi(res, model, time.time() - t0, "highs")
            except Exception as exc2:                  # noqa: BLE001
                return SolveOutcome("error", wall=time.time() - t0,
                                    solver="highs", note=str(exc2)[:300])
        return SolveOutcome("error", wall=time.time() - t0, solver="highs",
                            note=msg[:300])
    _SOLVER_CACHE["used"].add("highs")
    return _interpret_appsi(res, model, time.time() - t0, "highs")


def _interpret_pyomo(res, model, wall, solver):
    from pyomo.opt import TerminationCondition as TC
    tc = res.solver.termination_condition
    if tc in (TC.infeasible, TC.infeasibleOrUnbounded):
        return SolveOutcome("infeasible", wall=wall, solver=solver)
    if tc in (TC.optimal, TC.locallyOptimal, TC.globallyOptimal, TC.feasible,
              TC.maxTimeLimit):
        try:
            obj = value(model.OBJ)
        except Exception:                              # noqa: BLE001
            return SolveOutcome("infeasible", wall=wall, solver=solver)
        status = "optimal" if tc == TC.optimal else "feasible"
        bound = None
        try:
            bound = float(res.problem.lower_bound)
        except Exception:                              # noqa: BLE001
            pass
        gap = None if bound is None or obj is None or abs(obj) < 1e-9 \
            else abs(obj - bound) / max(1e-9, abs(obj))
        return SolveOutcome(status, obj, bound, gap, wall, solver)
    return SolveOutcome("error", wall=wall, solver=solver, note=str(tc))


def _interpret_appsi(res, model, wall, solver):
    from pyomo.contrib.appsi.base import TerminationCondition as ATC
    tc = res.termination_condition
    if tc in (ATC.infeasible, ATC.infeasibleOrUnbounded):
        return SolveOutcome("infeasible", wall=wall, solver=solver)
    if tc in (ATC.optimal, ATC.maxTimeLimit, ATC.maxIterations,
              ATC.interrupted, ATC.objectiveLimit):
        try:
            obj = float(res.best_feasible_objective)
        except Exception:                              # noqa: BLE001
            obj = None
        if obj is None:
            return SolveOutcome("infeasible" if tc != ATC.optimal else "error",
                                wall=wall, solver=solver, note=str(tc))
        bound = None
        try:
            bound = float(res.best_objective_bound)
        except Exception:                              # noqa: BLE001
            pass
        gap = None if bound is None or abs(obj) < 1e-9 \
            else abs(obj - bound) / max(1e-9, abs(obj))
        status = "optimal" if tc == ATC.optimal else "feasible"
        return SolveOutcome(status, obj, bound, gap, wall, solver)
    return SolveOutcome("error", wall=wall, solver=solver, note=str(tc))


# ---------------------------------------------------------------------------
# Shared model scaffolding
# ---------------------------------------------------------------------------
def _facility_cost(net):
    """Baseline CTM aggregation: sum_p CTM[p] = |t| * sum_m E1[m]*(CIM+CVM)."""
    return {m: net.horizon * (net.CIM[m] + net.CVM[m]) for m in net.m}


def _network_core(mdl, net, inst, max_facilities, dominance_cuts=False):
    """E1 / X1 / X2 / y1 / y2 and CON1-CON7, CON12-CON15 (index-reduced)."""
    P = [p.pid for p in inst.patients]
    pat = {p.pid: p for p in inst.patients}
    mdl.P, mdl.pat, mdl.net = P, pat, net

    mdl.E1 = Var(net.m, within=Binary)
    mdl.X1 = Var(net.c, net.m, within=Binary)
    mdl.X2 = Var(net.m, net.h, within=Binary)
    mdl.y1 = Var(P, net.m, net.j, within=Binary)      # LS -> MS  (c, t implied)
    mdl.y2 = Var(P, net.m, net.j, within=Binary)      # MS -> hospital (h implied)

    mdl.con = ConstraintList()
    # CON1 - number of facilities
    mdl.con.add(sum(mdl.E1[m] for m in net.m) <= max_facilities)
    # CON2 / CON3 - no match with a non-existent facility
    for c in net.c:
        for m in net.m:
            mdl.con.add(mdl.X1[c, m] <= mdl.E1[m])
    for m in net.m:
        for h in net.h:
            mdl.con.add(mdl.X2[m, h] <= mdl.E1[m])
    # CON6 / CON7 - exactly one journey per leg; CON4 / CON5 - link to matches
    for pid in P:
        p = pat[pid]
        mdl.con.add(sum(mdl.y1[pid, m, j] for m in net.m for j in net.j) == 1)
        mdl.con.add(sum(mdl.y2[pid, m, j] for m in net.m for j in net.j) == 1)
        for m in net.m:
            # material leaves the facility it entered (MSB2/MSB8 flow balance)
            mdl.con.add(sum(mdl.y1[pid, m, j] for j in net.j)
                        == sum(mdl.y2[pid, m, j] for j in net.j))
            for j in net.j:
                mdl.con.add(mdl.y1[pid, m, j] <= mdl.X1[p.c, m])       # CON4
                mdl.con.add(mdl.y2[pid, m, j] <= mdl.X2[m, p.h])       # CON5
    # Objective-preserving dominance cuts (extension only).  m4/m5/m6 have the
    # same CIM, CVM and FCAP as m1/m2/m3 but weakly higher U1 and U3 from every
    # site, so any solution using m4 without m1 can be relabelled onto m1 at no
    # extra cost and with identical timing.
    if dominance_cuts:
        for lo, hi in (("m1", "m4"), ("m2", "m5"), ("m3", "m6")):
            if lo in net.m and hi in net.m:
                mdl.con.add(mdl.E1[hi] <= mdl.E1[lo])


def _transport_cost_expr(mdl):
    net, pat = mdl.net, mdl.pat
    return sum(net.U1[(pat[pid].c, m, j)] * mdl.y1[pid, m, j]
               + net.U3[(m, pat[pid].h, j)] * mdl.y2[pid, m, j]
               for pid in mdl.P for m in net.m for j in net.j)


def _facility_cost_expr(mdl):
    fc = _facility_cost(mdl.net)
    return sum(fc[m] * mdl.E1[m] for m in mdl.net.m)


# ---------------------------------------------------------------------------
# PHASE 1 model - baseline i-SHIPMENT (no survival, no queue, ND = 18)
# ---------------------------------------------------------------------------
def build_baseline(net, inst, nd=ND_BASELINE, max_facilities=MAX_FACILITIES):
    mdl = ConcreteModel(name="i-SHIPMENT baseline")
    _network_core(mdl, net, inst, max_facilities)
    P, pat = mdl.P, mdl.pat
    TLS = int(net.TLS)
    lead = net.TMFE + net.TQC

    # INM is NOT a decision here: manufacturing starts on the day the sample
    # arrives at the facility (old MSB5).
    def start_day(p, j):
        return p.t0 + TLS + net.TT1[j]

    mdl.TRT = Var(P, within=NonNegativeReals)
    for pid in P:
        p = pat[pid]
        mdl.con.add(mdl.TRT[pid] ==
                    sum((TLS + net.TT1[j]) * mdl.y1[pid, m, j]
                        for m in net.m for j in net.j)
                    + lead
                    + sum(net.TT3[j] * mdl.y2[pid, m, j]
                          for m in net.m for j in net.j))
        mdl.con.add(mdl.TRT[pid] <= nd)                       # TCON
        mdl.con.add(mdl.TRT[pid] <= net.horizon - p.t0)       # stay in horizon

    # CAP1 + CAPCON1 (+ MSB2) == 7-day rolling window of manufacturing starts
    starts = defaultdict(list)                 # (m, day) -> list of y1 terms
    for pid in P:
        p = pat[pid]
        for m in net.m:
            for j in net.j:
                starts[(m, start_day(p, j))].append(mdl.y1[pid, m, j])
    for m in net.m:
        for t in range(1, net.horizon + 1):
            terms = []
            for tau in range(t - net.TMFE + 1, t + 1):
                terms.extend(starts.get((m, tau), []))
            if terms:
                mdl.con.add(sum(terms) <= net.FCAP[m])

    mdl.OBJ = Objective(expr=_facility_cost_expr(mdl) + _transport_cost_expr(mdl)
                        + (net.C_material + net.CQC) * len(P), sense=minimize)
    mdl.kind = "baseline"
    return mdl


# ---------------------------------------------------------------------------
# PHASE 2+ model - queue + survival extension (eqs. 32-35, 24-25, 31, 1)
# ---------------------------------------------------------------------------
def build_extension(net, inst, alpha, nd=ND_EXT, max_facilities=MAX_FACILITIES,
                    sigma=None, dominance_cuts=True, symmetry_cuts=True):
    sigma = sigma or cd.sigma_table()
    mdl = ConcreteModel(name="i-SHIPMENT + queue + survival")
    _network_core(mdl, net, inst, max_facilities, dominance_cuts=dominance_cuts)
    P, pat = mdl.P, mdl.pat
    TLS = int(net.TLS)
    lead = net.TMFE + net.TQC
    tt1_min, tt3_min = min(net.TT1.values()), min(net.TT3.values())

    # ---- start-time windows -------------------------------------------------
    # earliest start = arrival at the MS by the fastest mode
    # latest start   = whatever still satisfies TRT <= ND and the 130-day horizon
    win = {}
    for pid in P:
        p = pat[pid]
        lo = p.t0 + TLS + tt1_min
        hi = min(p.t0 + nd - lead - tt3_min, net.horizon - lead - tt3_min)
        win[pid] = list(range(lo, hi + 1))
        if not win[pid]:
            raise ValueError(f"patient {pid} has an empty start window")
    mdl.win = win

    idx = [(pid, m, t) for pid in P for m in net.m for t in win[pid]]
    mdl.INM = Var(idx, within=Binary)          # START DECISION (was MSB5)
    mdl.TRT = Var(P, within=NonNegativeReals)
    mdl.HOLD = Var(P, within=NonNegativeReals)
    mdl.STT = Var(P, within=NonNegativeReals)  # eq. 24
    mdl.CTT = Var(P, within=NonNegativeReals)  # eq. 25
    mdl.S = Var(P, bounds=(0, 1))
    dl_idx = [(pid, d) for pid in P for d in cd.D_RANGE]
    mdl.delta = Var(dl_idx, within=Binary)

    for pid in P:
        p = pat[pid]
        arrivals = {j: p.t0 + TLS + net.TT1[j] for j in net.j}

        # (33) every patient is started exactly once, at the facility it was
        #      shipped to
        mdl.con.add(sum(mdl.INM[pid, m, t] for m in net.m for t in win[pid]) == 1)
        for m in net.m:
            mdl.con.add(sum(mdl.INM[pid, m, t] for t in win[pid])
                        == sum(mdl.y1[pid, m, j] for j in net.j))

            # (32) no start before the sample has arrived:
            #        sum_{tau<=t} INM <= sum_{tau<=t} A,   A = arrivals at MS.
            #      Only the days on which the cumulative RHS is still a strict
            #      subset of the modes carry information; the remaining rows are
            #      implied by the equality above, so they are not emitted.
            for t in win[pid]:
                avail = [j for j in net.j if arrivals[j] <= t]
                if len(avail) == len(net.j):
                    break
                mdl.con.add(sum(mdl.INM[pid, m, tau] for tau in win[pid] if tau <= t)
                            <= sum(mdl.y1[pid, m, j] for j in avail))

        start_expr = sum(t * mdl.INM[pid, m, t] for m in net.m for t in win[pid])
        # HOLD = start - arrival at the manufacturing site  (>= 0 by (32))
        mdl.con.add(mdl.HOLD[pid] == start_expr
                    - sum(arrivals[j] * mdl.y1[pid, m, j]
                          for m in net.m for j in net.j))
        # eqs. (24)-(25): start / completion of treatment
        mdl.con.add(mdl.STT[pid] == p.t0)
        mdl.con.add(mdl.CTT[pid] == start_expr + lead
                    + sum(net.TT3[j] * mdl.y2[pid, m, j]
                          for m in net.m for j in net.j))
        mdl.con.add(mdl.CTT[pid] <= net.horizon)
        mdl.con.add(mdl.TRT[pid] == mdl.CTT[pid] - mdl.STT[pid])

        # exact integer-day survival lookup
        mdl.con.add(sum(mdl.delta[pid, d] for d in cd.D_RANGE) == 1)
        mdl.con.add(mdl.TRT[pid] == sum(d * mdl.delta[pid, d] for d in cd.D_RANGE))
        mdl.con.add(mdl.S[pid] == sum(sigma[d][p.tier] * mdl.delta[pid, d]
                                      for d in cd.D_RANGE))

    # (34) concurrent-capacity: sum_p DURV[p,m,t] <= FCAP[m], where MSBnew makes
    #      DURV[p,m,t] = 1 exactly on t in [start+1, start+TMFE].
    by_day = defaultdict(list)                     # (m, start-day) -> INM vars
    for pid in P:
        for m in net.m:
            for t in win[pid]:
                by_day[(m, t)].append(mdl.INM[pid, m, t])
    for m in net.m:
        for t in range(1, net.horizon + 1):
            terms = []
            for tau in range(t - net.TMFE, t):
                terms.extend(by_day.get((m, tau), []))
            if terms:
                mdl.con.add(sum(terms) <= net.FCAP[m])

    # Aggregate throughput cuts.  These are IMPLIED by (34) -- they do not
    # change the feasible set or the optimum, they only tighten the root LP
    # bound, which is what makes N = 500 provable.
    #
    # Derivation: (34) allows at most FCAP[m] starts at facility m in any 7
    # consecutive days.  Chop the union of the per-patient start windows into
    # K disjoint 7-day blocks; every start falls in exactly one block, so
    #        (number of starts)  <=  K * sum_m FCAP[m] * E1[m].
    # Applying the same argument to every prefix [t_lo, T] gives one cut per T,
    # counting only the patients whose start window ENDS at or before T (those
    # cannot be pushed past T).
    t_lo = min(win[pid][0] for pid in P)
    t_hi = max(win[pid][-1] for pid in P)
    cap_expr = sum(net.FCAP[m] * mdl.E1[m] for m in net.m)
    for T in list(range(t_lo + net.TMFE - 1, t_hi, net.TMFE)) + [t_hi]:
        forced = sum(1 for pid in P if win[pid][-1] <= T)
        if forced:
            k = math.ceil((T - t_lo + 1) / net.TMFE)
            mdl.con.add(k * cap_expr >= forced)

    # Symmetry breaking among fully interchangeable patients.  Two patients with
    # the same leukapheresis site, the same risk tier and the same arrival day
    # have identical costs, identical arrival dynamics and identical survival
    # curves, so any solution can be permuted among them; requiring their start
    # days to be non-decreasing removes that symmetry without cutting off any
    # objective value.
    if symmetry_cuts:
        groups = defaultdict(list)
        for pid in P:
            p = pat[pid]
            groups[(p.c, p.tier, p.t0)].append(pid)
        start_of = {pid: sum(t * mdl.INM[pid, m, t]
                             for m in net.m for t in win[pid]) for pid in P}
        for g in groups.values():
            for a, b in zip(g, g[1:]):
                mdl.con.add(start_of[a] <= start_of[b])

    # (35)/(1) objective: original cost + ALPHA * clinical loss
    cost = (_facility_cost_expr(mdl) + _transport_cost_expr(mdl)
            + (net.C_material + net.CQC) * len(P))
    loss = sum(cd.RHO[pat[pid].tier] * (1 - mdl.S[pid]) for pid in P)
    mdl.COSTEXPR, mdl.LOSSEXPR = cost, loss
    mdl.OBJ = Objective(expr=cost + alpha * loss, sense=minimize)
    mdl.kind = "extension"
    mdl.alpha = alpha
    return mdl


# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------
def extract(mdl, inst, net):
    """Per-patient records + aggregate summary for a solved model."""
    P, pat = mdl.P, mdl.pat
    TLS = int(net.TLS)
    lead = net.TMFE + net.TQC
    sigma = cd.sigma_table()

    opened = [m for m in net.m if value(mdl.E1[m]) > 0.5]
    rows = []
    for pid in P:
        p = pat[pid]
        m_in = j_in = None
        for m in net.m:
            for j in net.j:
                if value(mdl.y1[pid, m, j]) > 0.5:
                    m_in, j_in = m, j
        j_out = None
        for m in net.m:
            for j in net.j:
                if value(mdl.y2[pid, m, j]) > 0.5:
                    j_out = j
        ms_arrival = p.t0 + TLS + net.TT1[j_in]
        if mdl.kind == "extension":
            start = round(sum(t * value(mdl.INM[pid, m, t])
                              for m in net.m for t in mdl.win[pid]))
        else:
            start = ms_arrival
        hold = start - ms_arrival
        trt = int(round(start + lead + net.TT3[j_out] - p.t0))
        s = sigma[trt][p.tier]
        rows.append({
            "pid": pid, "tier": p.tier, "c": p.c, "h": p.h, "t0": p.t0,
            "facility": m_in, "mode_in": j_in, "mode_out": j_out,
            "ms_arrival": ms_arrival, "start": start, "hold": hold,
            "delivery": start + lead + net.TT3[j_out], "TRT": trt,
            "survival": s, "expected_loss": 1 - s,
            "weighted_loss": cd.RHO[p.tier] * (1 - s),
        })

    fc = _facility_cost(net)
    fac_cost = sum(fc[m] for m in opened)
    tr_cost = sum(net.U1[(pat[r['pid']].c, r["facility"], r["mode_in"])]
                  + net.U3[(r["facility"], pat[r['pid']].h, r["mode_out"])]
                  for r in rows)
    mat_cost = (net.C_material + net.CQC) * len(P)
    summary = {
        "n": len(P),
        "opened": opened,
        "n_opened": len(opened),
        "capacity_opened": sum(net.FCAP[m] for m in opened),
        "facility_cost": fac_cost,
        "transport_cost": tr_cost,
        "material_qc_cost": mat_cost,
        "total_cost": fac_cost + tr_cost + mat_cost,
        "mean_TRT": statistics.fmean(r["TRT"] for r in rows),
        "max_TRT": max(r["TRT"] for r in rows),
        "mean_HOLD": statistics.fmean(r["hold"] for r in rows),
        "max_HOLD": max(r["hold"] for r in rows),
        "mean_survival": statistics.fmean(r["survival"] for r in rows),
        "expected_lost": sum(r["expected_loss"] for r in rows),
        "weighted_loss": sum(r["weighted_loss"] for r in rows),
    }
    for u in cd.TIER_ORDER:
        sub = [r for r in rows if r["tier"] == u]
        summary[f"n_{u}"] = len(sub)
        summary[f"mean_TRT_{u}"] = statistics.fmean(r["TRT"] for r in sub)
        summary[f"mean_HOLD_{u}"] = statistics.fmean(r["hold"] for r in sub)
        summary[f"mean_survival_{u}"] = statistics.fmean(r["survival"] for r in sub)
        summary[f"expected_lost_{u}"] = sum(r["expected_loss"] for r in sub)
    return rows, summary


def tier_table(rows, label):
    """Per-tier aggregate rows for the Phase-3 comparison tables."""
    out = []
    for u in cd.TIER_ORDER:
        sub = [r for r in rows if r["tier"] == u]
        if not sub:
            continue
        out.append({
            "design": label, "tier": u, "n": len(sub), "rho": cd.RHO[u],
            "w_risk": cd.W_RISK[u],
            "mean_TRT": statistics.fmean(r["TRT"] for r in sub),
            "mean_HOLD": statistics.fmean(r["hold"] for r in sub),
            "median_HOLD": statistics.median(r["hold"] for r in sub),
            "max_HOLD": max(r["hold"] for r in sub),
            "mean_survival": statistics.fmean(r["survival"] for r in sub),
            "expected_lost": sum(r["expected_loss"] for r in sub),
            "weighted_loss": sum(r["weighted_loss"] for r in sub),
        })
    out.append({
        "design": label, "tier": "ALL", "n": len(rows), "rho": "",
        "w_risk": "",
        "mean_TRT": statistics.fmean(r["TRT"] for r in rows),
        "mean_HOLD": statistics.fmean(r["hold"] for r in rows),
        "median_HOLD": statistics.median(r["hold"] for r in rows),
        "max_HOLD": max(r["hold"] for r in rows),
        "mean_survival": statistics.fmean(r["survival"] for r in rows),
        "expected_lost": sum(r["expected_loss"] for r in rows),
        "weighted_loss": sum(r["weighted_loss"] for r in rows),
    })
    return out


# ---------------------------------------------------------------------------
# tiny CSV writer (no pandas dependency in the model path)
# ---------------------------------------------------------------------------
def write_csv(path, rows, fieldnames=None):
    import csv
    if not rows:
        open(path, "w").write("")
        return
    if fieldnames is None:
        # union of every row's keys, first-seen order -- taking only rows[0]'s
        # keys silently drops columns that appear on later rows
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


# ---------------------------------------------------------------------------
# Solution cache -- phases share solves; re-runs are cheap
# ---------------------------------------------------------------------------
CACHE = os.path.join(RESULTS, "cache")


def _cache_key(kind, n, alpha, max_fac, nd, tag=""):
    base = f"{kind}_n{n}_alpha{alpha:g}_fac{max_fac}_nd{nd}"
    return base + (f"_{tag}" if tag else "")


def run_design(net, inst, kind, alpha=0.0, max_fac=MAX_FACILITIES, nd=None,
               time_limit=DEFAULT_TIME_LIMIT, mip_gap=DEFAULT_MIP_GAP,
               use_cache=True, sigma=None, solver_pref="gurobi",
               tag=""):
    """Build + solve one design, with an on-disk cache of the outcome."""
    nd = nd or (ND_BASELINE if kind == "baseline" else ND_EXT)
    key = _cache_key(kind, inst.n, alpha, max_fac, nd, tag)
    path = os.path.join(CACHE, key + ".json")
    if use_cache and os.path.exists(path):
        blob = json.load(open(path))
        _SOLVER_CACHE["used"].add(blob.get("solve", {}).get("solver", "?"))
        print(f"  [cache] {key}")
        return blob

    print(f"  [solve] {key} (limit {time_limit}s) ...", flush=True)
    if kind == "baseline":
        mdl = build_baseline(net, inst, nd=nd, max_facilities=max_fac)
    else:
        mdl = build_extension(net, inst, alpha=alpha, nd=nd,
                              max_facilities=max_fac, sigma=sigma)
    from pyomo.environ import Var as _V, Constraint as _C
    size = {"variables": sum(len(v) for v in mdl.component_objects(_V)),
            "constraints": sum(len(c) for c in mdl.component_objects(_C))}
    out = solve(mdl, time_limit=time_limit, mip_gap=mip_gap,
                solver_pref=solver_pref)
    blob = {"key": key, "kind": kind, "n": inst.n, "alpha": alpha,
            "max_facilities": max_fac, "ND": nd, "size": size,
            "tag": tag, "solve": out.as_dict()}
    if out.status in ("optimal", "feasible"):
        rows, summary = extract(mdl, inst, net)
        blob["rows"], blob["summary"] = rows, summary
        if kind == "extension":
            blob["summary"]["cost_component"] = float(value(mdl.COSTEXPR))
            blob["summary"]["loss_component"] = float(value(mdl.LOSSEXPR))
    print("    -> %s obj=%s wall=%.1fs" %
          (out.status, None if out.obj is None else round(out.obj, 2), out.wall or 0))
    write_json(path, blob)
    return blob


# ---------------------------------------------------------------------------
# PHASES
# ---------------------------------------------------------------------------
def phase0_setup(net, dat, scales):
    """Freeze the instances, the tier assignment and the sigma lookup."""
    sig = cd.sigma_table()
    write_csv(os.path.join(RESULTS, "sigma_lookup.csv"),
              [{"d_days": d, **{f"sigma_{u}": sig[d][u] for u in cd.TIER_ORDER}}
               for d in cd.D_RANGE])
    write_json(os.path.join(RESULTS, "calibration.json"), {
        "gamma": cd.GAMMA, "eta": cd.ETA, "kappa": cd.KAPPA,
        "tier_mix": cd.TIER_MIX, "w_risk": cd.W_RISK, "rho": cd.RHO,
        "survival_formula": "S_u(t) = (1 - w_u) ** (t / 42)",
        "TRT_support_days": [cd.D_MIN, cd.D_MAX],
        "ND_baseline": ND_BASELINE, "ND_extension": ND_EXT,
        "seed": cd.SEED, "arrival_window": list(cd.ARRIVAL_WINDOW),
        "horizon": cd.HORIZON, "max_facilities_default": MAX_FACILITIES,
        "max_facilities_by_scale": MAX_FACILITIES_BY_SCALE,
        "max_facilities_note": (
            "matches i-SHIPMENT's centralized vs high-demand configuration: "
            "2 facilities at N=100/200, relaxed to 3 at N=500"),
        "alpha_survival_design": ALPHA_SURVIVAL,
    })
    insts = {}
    for n in scales:
        mult = n // 50
        inst = cd.build_instance(dat, mult=mult)
        assert inst.n == n, (inst.n, n)
        insts[n] = inst
        cd.save_instance(inst, os.path.join(RESULTS, f"instance_N{n}.json"))
        write_csv(os.path.join(RESULTS, f"tiers_N{n}.csv"),
                  [{"pid": p.pid, "tier": p.tier, "c": p.c, "h": p.h,
                    "t0": p.t0, "rho": cd.RHO[p.tier], "w_risk": cd.W_RISK[p.tier]}
                   for p in inst.patients])
    # merge with any scales already on record, so running a subset of scales
    # never silently truncates this study-level file
    ov_path = os.path.join(RESULTS, "instances_overview.csv")
    prior = {}
    if os.path.exists(ov_path):
        import csv as _csv
        with open(ov_path) as f:
            for r in _csv.DictReader(f):
                prior[str(r.get("n"))] = r
    fresh = [{"n": n, "mult": n // 50, "seed": cd.SEED,
                "arrival_window": f"{cd.ARRIVAL_WINDOW[0]}-{cd.ARRIVAL_WINDOW[1]}",
                "arrivals_per_day": round(insts[n].density, 3),
                "n_H": sum(1 for p in insts[n].patients if p.tier == "H"),
                "n_M": sum(1 for p in insts[n].patients if p.tier == "M"),
                "n_L": sum(1 for p in insts[n].patients if p.tier == "L"),
                **{f"n_{c}": sum(1 for p in insts[n].patients if p.c == c)
                   for c in net.c}}
             for n in scales]
    merged = {**prior, **{str(r["n"]): r for r in fresh}}
    write_csv(ov_path, [merged[k] for k in sorted(merged, key=int)])
    return insts


def phase1(net, insts, time_limit, args):
    """Baseline i-SHIPMENT at every scale (the motivation for the queue)."""
    print("\n=== PHASE 1: baseline i-SHIPMENT (no survival, no queue, ND=18) ===")
    table, tiers = [], []
    for n, inst in sorted(insts.items()):
        cap = max_facilities_for(n, args.max_facilities)
        blob = run_design(net, inst, "baseline", max_fac=cap,
                          time_limit=time_limit, use_cache=not args.no_cache)
        row = {"n": n, "arrivals_per_day": round(inst.density, 3),
               "model": "baseline", "ND": ND_BASELINE,
               "max_facilities": cap,
               "status": blob["solve"]["status"], "solver": blob["solve"]["solver"],
               "wall_s": round(blob["solve"]["wall_s"] or 0, 1),
               "variables": blob["size"]["variables"],
               "constraints": blob["size"]["constraints"]}
        if "summary" in blob:
            s = blob["summary"]
            row.update({
                "opened": "+".join(s["opened"]), "n_opened": s["n_opened"],
                "capacity_opened": s["capacity_opened"],
                "facility_cost": round(s["facility_cost"], 2),
                "transport_cost": round(s["transport_cost"], 2),
                "total_cost": round(s["total_cost"], 2),
                "mean_TRT": round(s["mean_TRT"], 3), "max_TRT": s["max_TRT"],
                "mean_survival": round(s["mean_survival"], 5),
                "expected_lost": round(s["expected_lost"], 3),
                "weighted_loss": round(s["weighted_loss"], 3),
                "TRT_distribution": json.dumps(
                    dict(sorted(_counter(r["TRT"] for r in blob["rows"]).items()))),
            })
            write_csv(os.path.join(RESULTS, f"phase1_patients_N{n}.csv"), blob["rows"])
            for tr in tier_table(blob["rows"], f"baseline_N{n}"):
                tr["n_patients_total"] = n
                tiers.append(tr)
        else:
            # The configured cap is itself infeasible: show what it would take.
            diag = run_design(net, inst, "baseline", max_fac=len(net.m),
                              time_limit=time_limit, use_cache=not args.no_cache)
            row["diagnostic_CON1_relaxed"] = diag["solve"]["status"]
            if "summary" in diag:
                d = diag["summary"]
                row["diagnostic_opened"] = "+".join(d["opened"])
                row["diagnostic_capacity"] = d["capacity_opened"]
                row["diagnostic_total_cost"] = round(d["total_cost"], 2)

        # Where the scale runs a relaxed CON1, also record what the strict
        # centralised cap does -- documented as a diagnostic, not the headline.
        if cap != MAX_FACILITIES:
            strict = run_design(net, inst, "baseline", max_fac=MAX_FACILITIES,
                                time_limit=time_limit,
                                use_cache=not args.no_cache)
            row[f"strict_CON1_{MAX_FACILITIES}_status"] = strict["solve"]["status"]
            if "summary" in strict:
                d = strict["summary"]
                row[f"strict_CON1_{MAX_FACILITIES}_opened"] = "+".join(d["opened"])
                row[f"strict_CON1_{MAX_FACILITIES}_total_cost"] = round(
                    d["total_cost"], 2)
        table.append(row)
    write_csv(os.path.join(RESULTS, "phase1_baseline_by_scale.csv"), table)
    if tiers:
        write_csv(os.path.join(RESULTS, "phase1_baseline_by_tier.csv"), tiers)
    _print_table(table, ["n", "max_facilities", "status", "opened",
                         "capacity_opened", "total_cost", "mean_TRT",
                         "expected_lost", "strict_CON1_2_status",
                         "diagnostic_CON1_relaxed", "diagnostic_opened"])
    return table


def phase2(net, insts, time_limits, args):
    """Queue + survival extension: confirm feasibility at every scale."""
    print("\n=== PHASE 2: queue + survival extension (eqs 32-35, ND=42) ===")
    table = []
    for n, inst in sorted(insts.items()):
      for cap in caps_for(n, args.max_facilities):
        blob = _extension_with_fallback(net, inst, ALPHA_SURVIVAL,
                                        time_limits.get(n, DEFAULT_TIME_LIMIT),
                                        args, max_fac=cap)
        row = {"n": n, "arrivals_per_day": round(inst.density, 3),
               "model": "extension", "ND": ND_EXT, "alpha": ALPHA_SURVIVAL,
               "max_facilities": blob["max_facilities"],
               "is_configured_cap": cap == max_facilities_for(n, args.max_facilities),
               "status": blob["solve"]["status"], "solver": blob["solve"]["solver"],
               "mip_gap": (None if blob["solve"]["gap"] is None
                           else round(blob["solve"]["gap"], 6)),
               "wall_s": round(blob["solve"]["wall_s"] or 0, 1),
               "variables": blob["size"]["variables"],
               "constraints": blob["size"]["constraints"]}
        if "summary" in blob:
            s = blob["summary"]
            row.update({"opened": "+".join(s["opened"]),
                        "capacity_opened": s["capacity_opened"],
                        "total_cost": round(s["total_cost"], 2),
                        "mean_TRT": round(s["mean_TRT"], 3),
                        "mean_HOLD": round(s["mean_HOLD"], 3),
                        "max_HOLD": s["max_HOLD"],
                        "mean_survival": round(s["mean_survival"], 5),
                        "expected_lost": round(s["expected_lost"], 3)})
            suffix = "" if len(caps_for(n, args.max_facilities)) == 1 else f"_fac{cap}"
            write_csv(os.path.join(RESULTS, f"phase2_patients_N{n}{suffix}.csv"),
                      blob["rows"])
        table.append(row)
    write_csv(os.path.join(RESULTS, "phase2_extension_by_scale.csv"), table)
    _print_table(table, ["n", "max_facilities", "status", "opened",
                         "capacity_opened", "total_cost", "mean_TRT",
                         "mean_HOLD", "expected_lost"])
    return table


def _extension_with_fallback(net, inst, alpha, time_limit, args, max_fac=None):
    """Solve the extension at the given CON1; if that is infeasible, relax and
    report the relaxation."""
    max_fac = max_fac or max_facilities_for(inst.n, args.max_facilities)
    blob = run_design(net, inst, "extension", alpha=alpha, max_fac=max_fac,
                      time_limit=time_limit, mip_gap=args.mip_gap,
                      use_cache=not args.no_cache)
    if blob["solve"]["status"] in ("infeasible",):
        print("    CON1<=%d infeasible at N=%d -> relaxing CON1 to <=%d"
              % (max_fac, inst.n, len(net.m)))
        blob = run_design(net, inst, "extension", alpha=alpha,
                          max_fac=len(net.m), time_limit=time_limit,
                          mip_gap=args.mip_gap, use_cache=not args.no_cache)
        blob["CON1_relaxed"] = True
    return blob


def phase3(net, insts, time_limits, args):
    """Cost design (ALPHA=0) vs survival design (ALPHA>0), by tier, per scale."""
    print("\n=== PHASE 3: cost vs survival design, by tier ===")
    summary_rows, tier_rows, hold_rows = [], [], []
    for n, inst in sorted(insts.items()):
        tl = time_limits.get(n, DEFAULT_TIME_LIMIT)
        designs = [("cost", 0.0), ("survival", ALPHA_SURVIVAL)]
        caps = caps_for(n, args.max_facilities)
        for cap in caps:
          tagc = "" if len(caps) == 1 else f"_fac{cap}"
          for label, alpha in designs:
            blob = _extension_with_fallback(net, inst, alpha, tl, args,
                                            max_fac=cap)
            if "summary" not in blob:
                summary_rows.append({"n": n, "design": label, "alpha": alpha,
                                     "max_facilities": cap,
                                     "status": blob["solve"]["status"]})
                continue
            s = blob["summary"]
            summary_rows.append({
                "n": n, "design": label, "alpha": alpha,
                "max_facilities": cap,
                "is_configured_cap": cap == max_facilities_for(
                    n, args.max_facilities),
                "status": blob["solve"]["status"],
                "mip_gap": (None if blob["solve"]["gap"] is None
                            else round(blob["solve"]["gap"], 6)),
                "opened": "+".join(s["opened"]),
                "capacity_opened": s["capacity_opened"],
                "facility_cost": round(s["facility_cost"], 2),
                "transport_cost": round(s["transport_cost"], 2),
                "total_cost": round(s["total_cost"], 2),
                "mean_TRT": round(s["mean_TRT"], 3),
                "mean_HOLD": round(s["mean_HOLD"], 3),
                "max_HOLD": s["max_HOLD"],
                "mean_survival": round(s["mean_survival"], 5),
                "expected_lost": round(s["expected_lost"], 3),
                "weighted_loss": round(s["weighted_loss"], 3),
                **{f"mean_HOLD_{u}": round(s[f"mean_HOLD_{u}"], 3)
                   for u in cd.TIER_ORDER},
                **{f"mean_TRT_{u}": round(s[f"mean_TRT_{u}"], 3)
                   for u in cd.TIER_ORDER},
                **{f"expected_lost_{u}": round(s[f"expected_lost_{u}"], 3)
                   for u in cd.TIER_ORDER},
            })
            for tr in tier_table(blob["rows"], label):
                tr["n_patients_total"] = n
                tr["max_facilities"] = cap
                tier_rows.append(tr)
            write_csv(os.path.join(RESULTS,
                                   f"phase3_patients_N{n}_{label}{tagc}.csv"),
                      blob["rows"])
            # hold distribution by tier
            for u in cd.TIER_ORDER:
                holds = [r["hold"] for r in blob["rows"] if r["tier"] == u]
                dist = _counter(holds)
                hold_rows.append({"n": n, "design": label,
                                  "max_facilities": cap, "tier": u,
                                  "n_patients": len(holds),
                                  "mean_hold": round(statistics.fmean(holds), 3),
                                  "p50": statistics.median(holds),
                                  "p90": _quantile(holds, 0.90),
                                  "max": max(holds),
                                  "share_hold_0": round(
                                      sum(1 for h in holds if h == 0) / len(holds), 4),
                                  "distribution": json.dumps(
                                      dict(sorted(dist.items())))})
    write_csv(os.path.join(RESULTS, "phase3_design_comparison.csv"), summary_rows)
    write_csv(os.path.join(RESULTS, "phase3_by_tier.csv"), tier_rows)
    write_csv(os.path.join(RESULTS, "phase3_hold_distribution.csv"), hold_rows)
    _print_table(summary_rows, ["n", "max_facilities", "design", "opened",
                                "total_cost", "mean_TRT", "mean_HOLD",
                                "mean_HOLD_H", "mean_HOLD_M", "mean_HOLD_L",
                                "expected_lost"])
    return summary_rows, tier_rows, hold_rows


def phase4(net, insts, scale, time_limit, args):
    """Cost-lives frontier: sweep ALPHA at one representative scale."""
    print(f"\n=== PHASE 4: cost-lives frontier at N={scale} ===")
    inst = insts[scale]
    alphas = sorted(set(list(ALPHA_SWEEP_REQUESTED)
                        + (list(ALPHA_SWEEP_EXTENDED) if args.extended_sweep else [])))
    rows = []
    cap4 = max_facilities_for(scale, args.max_facilities)
    for a in alphas:
        blob = _extension_with_fallback(net, inst, float(a), time_limit, args,
                                        max_fac=cap4)
        if "summary" not in blob:
            rows.append({"alpha": a, "status": blob["solve"]["status"]})
            continue
        s = blob["summary"]
        rows.append({
            "alpha": a, "status": blob["solve"]["status"],
            "requested_sweep": a in ALPHA_SWEEP_REQUESTED,
            "mip_gap": (None if blob["solve"]["gap"] is None
                        else round(blob["solve"]["gap"], 6)),
            "opened": "+".join(s["opened"]),
            "capacity_opened": s["capacity_opened"],
            "total_cost": round(s["total_cost"], 2),
            "facility_cost": round(s["facility_cost"], 2),
            "transport_cost": round(s["transport_cost"], 2),
            "expected_lost": round(s["expected_lost"], 4),
            "expected_lost_H": round(s["expected_lost_H"], 4),
            "expected_lost_M": round(s["expected_lost_M"], 4),
            "expected_lost_L": round(s["expected_lost_L"], 4),
            "weighted_loss": round(s["weighted_loss"], 4),
            "mean_TRT": round(s["mean_TRT"], 3),
            "mean_HOLD": round(s["mean_HOLD"], 3),
            "mean_HOLD_H": round(s["mean_HOLD_H"], 3),
            "mean_HOLD_L": round(s["mean_HOLD_L"], 3),
            "wall_s": round(blob["solve"]["wall_s"] or 0, 1),
        })
    write_csv(os.path.join(RESULTS, f"phase4_frontier_N{scale}.csv"), rows)
    _plot_frontier(rows, scale)
    _print_table(rows, ["alpha", "opened", "total_cost", "expected_lost",
                        "expected_lost_H", "mean_HOLD_H", "mean_HOLD_L"])
    return rows


def _plot_frontier(rows, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r.get("total_cost") is not None]
    if not pts:
        return
    pts.sort(key=lambda r: r["alpha"])
    cost = [r["total_cost"] / 1e6 for r in pts]
    lost = [r["expected_lost"] for r in pts]
    lostH = [r["expected_lost_H"] for r in pts]
    alph = [r["alpha"] for r in pts]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- left panel: the frontier itself --------------------------------
    order = sorted(range(len(pts)), key=lambda i: (lost[i], cost[i]))
    ax[0].plot([lost[i] for i in order], [cost[i] for i in order],
               "-o", color="#1f77b4", zorder=2, label="all patients")
    ax[0].plot(lostH, cost, "s", color="#d62728", zorder=3, ms=5,
               label="high-risk tier only")

    # annotate only points that are visually separated, plus the endpoints,
    # and summarise the degenerate cluster with a single label
    last = None
    cluster = []
    for i in sorted(range(len(pts)), key=lambda i: lost[i]):
        far = last is None or abs(lost[i] - lost[last]) > 0.18 * (
            max(lost) - min(lost) + 1e-9)
        if far or i in (0, len(pts) - 1):
            ax[0].annotate(f"$\\alpha$={alph[i]:g}", (lost[i], cost[i]),
                           textcoords="offset points", xytext=(7, 6), fontsize=8)
            last = i
        else:
            cluster.append(i)
    if cluster:
        i = cluster[len(cluster) // 2]
        lo, hi = min(alph[k] for k in cluster), max(alph[k] for k in cluster)
        ax[0].annotate(f"$\\alpha$ = {lo:g} ... {hi:g}",
                       (lost[i], cost[i]), textcoords="offset points",
                       xytext=(10, -26), fontsize=8, color="#555555",
                       arrowprops=dict(arrowstyle="-", color="#999999", lw=.7))

    ax[0].set_xlabel("Expected patients lost   $\\Sigma_p\\,(1 - S[p])$")
    ax[0].set_ylabel("Total supply-chain cost  [$M]")
    ax[0].set_title(f"Cost-lives frontier, N = {scale}")
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="center right")
    ax[0].margins(x=.16, y=.12)

    # ---- right panel: response to ALPHA ---------------------------------
    a_pos = [max(a, 1e-2) for a in alph]
    ax[1].semilogx(a_pos, lost, "-o", label="E[lost], all patients",
                   color="#1f77b4")
    ax[1].semilogx(a_pos, lostH, "-s", label="E[lost], high-risk tier",
                   color="#d62728")
    ax2 = ax[1].twinx()
    ax2.semilogx(a_pos, cost, "--^", color="#2ca02c", label="total cost [$M]")
    ax2.set_ylabel("Total cost [$M]", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    flips = [i for i in range(1, len(pts))
             if pts[i]["opened"] != pts[i - 1]["opened"]]
    for i in flips:
        ax[1].axvline(a_pos[i], color="#888888", ls=":", lw=1)
        ax[1].annotate(f"network flips\nto {pts[i]['opened']}", (a_pos[i], max(lost)),
                       textcoords="offset points", xytext=(-78, -14), fontsize=8,
                       color="#555555", ha="left")
    ax[1].set_xlabel("ALPHA   ($\\alpha$ = 0 plotted at $10^{-2}$)")
    ax[1].set_ylabel("Expected patients lost")
    ax[1].set_title("Effect of the clinical-loss weight")
    ax[1].grid(alpha=.3)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

    fig.tight_layout()
    out = os.path.join(RESULTS, f"phase4_frontier_N{scale}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [plot] {out}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _counter(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return dict(d)


def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return float(xs[lo]) if lo == hi else round(
        xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def _print_table(rows, cols):
    cols = [c for c in cols if any(c in r for r in rows)]
    if not rows or not cols:
        return
    widths = {c: max(len(c), max(len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + " | ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + " | ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in cols))


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 1e4 else f"{v:,.0f}"
    return str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all",
                    choices=["0", "1", "2", "3", "4", "all"])
    ap.add_argument("--scales", nargs="+", type=int, default=[100, 200, 500])
    ap.add_argument("--frontier-scale", type=int, default=200)
    ap.add_argument("--dat", default="Data200_profileA.dat")
    ap.add_argument("--baseline-time-limit", type=int, default=900)
    ap.add_argument("--ext-time-limit", type=int, default=0,
                    help="0 = scale-dependent defaults")
    ap.add_argument("--frontier-time-limit", type=int, default=900)
    ap.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    ap.add_argument("--max-facilities", type=int, default=0,
                    help="override CON1 for every scale; 0 (default) uses the "
                         "demand-dependent i-SHIPMENT configuration "
                         "(2 facilities at N=100/200, 3 at N=500)")
    ap.add_argument("--extended-sweep", action="store_true", default=True)
    ap.add_argument("--no-extended-sweep", dest="extended_sweep",
                    action="store_false")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    dat = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dat)
    net = cd.load_network(dat)

    scales = sorted(set(args.scales) | ({args.frontier_scale}
                                        if args.phase in ("4", "all") else set()))
    insts = phase0_setup(net, dat, scales)

    if args.ext_time_limit:
        tls = {n: args.ext_time_limit for n in scales}
    else:
        tls = {n: {100: 900, 200: 2400, 500: 5400}.get(n, 1800) for n in scales}

    out = {"solver_note": _SOLVER_CACHE["note"]}
    if args.phase in ("1", "all"):
        out["phase1"] = phase1(net, insts, args.baseline_time_limit, args)
    if args.phase in ("2", "all"):
        out["phase2"] = phase2(net, insts, tls, args)
    if args.phase in ("3", "all"):
        s3, t3, h3 = phase3(net, insts, tls, args)
        out["phase3"] = {"summary": s3, "by_tier": t3, "holds": h3}
    if args.phase in ("4", "all"):
        out["phase4"] = phase4(net, insts, args.frontier_scale,
                               args.frontier_time_limit, args)

    used = sorted(x for x in _SOLVER_CACHE["used"] if x and x != "?")
    if "highs" in used and not _SOLVER_CACHE["note"]:
        _SOLVER_CACHE["note"] = GUROBI_FALLBACK_NOTE
    out["solver_note"] = _SOLVER_CACHE["note"] or "Gurobi throughout."
    out["solver_used"] = used or ["gurobi"]
    write_json(os.path.join(RESULTS, "study_summary.json"), out)
    print("\nWrote results to", RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
