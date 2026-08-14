"""
policies.py
===========

The five operating policies compared in the study, exactly as the finalized
policy table of ``Survival_Aware_iSHIPMENT_Formulation.docx`` defines them.

  ``fifo``             serve in arrival order -- no objective
  ``survival_index``   greedy (P8): serve the largest weighted one-day
                       survival loss
  ``static_survival``  the survival schedule optimised ONCE up front (the
                       strategic INM start order), dispatched greedily and
                       never re-solved
  ``adaptive_mpc``     the survival schedule re-optimised at EVERY decision
                       epoch on the observed state, via (P1)-(P6)
  ``best_achievable``  the loss if all manufacturing outcomes were known in
                       advance -- a perfect-information solve per replication,
                       the lower-bound reference

Every policy sees the same frozen network, the same patients and (through
common random numbers) the same batch failures; they differ only in who they
choose to start.

Non-idling.  The four online policies never leave a free slot idle while a
ready patient waits: ``select`` always returns the full ``n_start``.
``best_achievable`` is the exception -- it is a BOUND, not an executable rule,
and deliberately holding a slot open for a sicker patient arriving tomorrow is
part of what perfect information buys.  Forcing it to be non-idling would make
it something other than a lower bound, so its schedule is dispatched exactly as
the perfect-information MILP planned it.
"""

from __future__ import annotations

import cart_data as cd
import per_epoch as pe

POLICY_NAMES = ("fifo", "survival_index", "static_survival", "adaptive_mpc",
                "best_achievable")


# ---------------------------------------------------------------------------
# The four online policies
# ---------------------------------------------------------------------------
class Fifo:
    """Serve in arrival order: earliest first collection, ties by patient id."""
    name = "fifo"

    def select(self, sim, m, t, n_start, cands):
        order = sorted(cands, key=lambda c: (c.t0, c.pid))
        return [c.pid for c in order[:n_start]]


class SurvivalIndex:
    """(P8): start whoever's rho-weighted survival falls most from one more day."""
    name = "survival_index"

    def select(self, sim, m, t, n_start, cands):
        return pe.index_choice(cands, t, n_start, sim.cfg.tmfe, sim.cfg.tqc)


class StaticSurvival:
    """The up-front survival schedule, dispatched greedily and never re-solved.

    The order comes from the strategic solve's manufacturing-start days at
    alpha = $500K.  Because failures are invisible to a plan made before the
    horizon starts, a remade patient simply goes to the back of that order.
    """
    name = "static_survival"

    def prepare(self, sim):
        self.rank = {pid: i for i, pid in enumerate(sim.plan.static_order)}

    def select(self, sim, m, t, n_start, cands):
        # a patient on its second attempt has been pushed to the back
        def key(c):
            r = sim.rec[c.pid]
            return (1, self.rank[c.pid]) if r.attempts else (0, self.rank[c.pid])
        return [c.pid for c in sorted(cands, key=key)[:n_start]]


class AdaptiveMPC:
    """(P1)-(P6) re-solved at every decision epoch on the observed state."""
    name = "adaptive_mpc"

    def select(self, sim, m, t, n_start, cands):
        return pe.solve_epoch(cands, t, sim.busy[m], sim.plan.fcap[m],
                              sim.cfg.tmfe, sim.cfg.tqc,
                              horizon_H=sim.cfg.lookahead, n_start=n_start,
                              solver=sim.cfg.epoch_solver)


# ---------------------------------------------------------------------------
# The perfect-information bound
# ---------------------------------------------------------------------------
class BestAchievable:
    """Dispatches a schedule computed with every batch outcome known up front.

    The bound is the optimal value of ``solve_perfect_information`` on this
    replication's realised failures; the simulation then executes that schedule
    so that costs and metrics come out of exactly the same accounting code as
    every other policy.
    """
    name = "best_achievable"

    def __init__(self, time_limit=600, mip_gap=1e-4, proxy_on_failure=True):
        self.time_limit, self.mip_gap = time_limit, mip_gap
        self.proxy_on_failure = proxy_on_failure
        self.status = "not-run"
        self.bound = None

    def prepare(self, sim):
        sched, info = solve_perfect_information(
            sim.plan, sim.seed, sim.cfg, time_limit=self.time_limit,
            mip_gap=self.mip_gap)
        self.status = info["status"]
        self.bound = info.get("objective")
        self.info = info
        if sched is None:
            if not self.proxy_on_failure:
                raise RuntimeError(f"perfect-information solve failed: {info}")
            # Labelled proxy: the same MPC, but with the failures revealed.
            # This is NOT a bound and is reported as a proxy wherever used.
            self.status = "proxy_failures_revealed_mpc"
            self.schedule = None
            self._proxy = AdaptiveMPC()
            return
        self.schedule = sched          # (pid, attempt) -> start day
        self._proxy = None

    def select(self, sim, m, t, n_start, cands):
        if self._proxy is not None:
            return self._proxy.select(sim, m, t, n_start, cands)
        due = [c.pid for c in cands
               if self.schedule.get((c.pid, sim.rec[c.pid].attempts + 1)) == t]
        return due[:n_start]           # may idle a slot -- see the module docstring


def build(name, **kw):
    """Instantiate a policy by its identifier."""
    table = {"fifo": Fifo, "survival_index": SurvivalIndex,
             "static_survival": StaticSurvival, "adaptive_mpc": AdaptiveMPC,
             "best_achievable": BestAchievable}
    if name not in table:
        raise KeyError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")
    return table[name](**kw) if name == "best_achievable" else table[name]()


# ---------------------------------------------------------------------------
# Perfect-information (clairvoyant) scheduling MILP
# ---------------------------------------------------------------------------
def failure_realisation(plan, seed, cfg):
    """Which attempt of each patient fails, under this replication's CRN.

    Identical draws to the ones the simulation will make, so the clairvoyant
    schedule is clairvoyant about the SAME replication.
    """
    from simulation import yield_draw
    fails = {}
    for pid, p in plan.patients.items():
        p_pass = (1.0 - cfg.fail_rate) if cfg.fail_rate is not None else p.p_pass
        fails[pid] = [yield_draw(seed, pid, k) > p_pass
                      for k in range(1, cfg.k_remake + 1)]
    return fails


def _futility_deadline(tier, s_min):
    """Latest elapsed day at which delivery still clears the S_min gate."""
    import math
    return int(math.floor(cd.ETA * math.log(s_min) / math.log(1 - cd.W_RISK[tier])))


def solve_perfect_information(plan, seed, cfg, time_limit=600, mip_gap=1e-4):
    """Minimise realised clinical loss with every batch outcome known up front.

    Decisions are the start day of each manufacturing attempt.  Attempt k+1 of
    a patient may only be run if attempt k was run and failed, and cannot start
    before that attempt's release plus a fresh leukapheresis (TLS + TT1).  The
    S_min gate applies to the realised delivery of each attempt.  Capacity is
    the same rolling T_MFE window the simulation enforces.

    Returns ``(schedule, info)`` where ``schedule`` maps (pid, attempt) to a
    start day, or ``(None, info)`` if the model could not be solved.
    """
    fails = failure_realisation(plan, seed, cfg)
    lead = cfg.tmfe + cfg.tqc
    horizon = cfg.drain_cap

    # Enumerate the attempts each patient could possibly need, with the start
    # window of each: attempt 1 no earlier than its staging day, attempt k+1 no
    # earlier than attempt k's release plus a re-collection.
    attempts = []                       # (pid, k, earliest, latest)
    for pid, p in plan.patients.items():
        deadline = _futility_deadline(p.tier, cfg.s_min)       # elapsed-at-delivery
        earliest = p.t0 + cfg.tls + p.tt1
        for k in range(1, cfg.k_remake + 1):
            latest = min(p.t0 + deadline - lead - p.tt3, horizon - lead - p.tt3)
            if earliest > latest:
                break
            attempts.append((pid, k, earliest, latest))
            if not fails[pid][k - 1]:                # k succeeds: no attempt k+1
                break
            earliest = earliest + lead + cfg.tls + p.tt1

    # Built in Pyomo and solved through the study's own solver plumbing, which
    # asks Gurobi first and falls back to HiGHS when the size-limited licence
    # refuses a model this large.
    from pyomo.environ import (ConcreteModel, Var, Binary, Objective,
                               ConstraintList, maximize, value)
    import ishipment_survival as ish

    days, by_patient = {}, {}
    for (pid, k, lo, hi) in attempts:
        days[(pid, k)] = list(range(lo, hi + 1))
        by_patient.setdefault(pid, []).append(k)

    mdl = ConcreteModel(name="perfect-information schedule")
    mdl.x = Var([(pid, k, tau) for (pid, k, _, _) in attempts
                 for tau in days[(pid, k)]], within=Binary)
    mdl.con = ConstraintList()

    def run(pid, k):
        return sum(mdl.x[pid, k, tau] for tau in days[(pid, k)])

    def start(pid, k):
        return sum(tau * mdl.x[pid, k, tau] for tau in days[(pid, k)])

    for pid, ks in by_patient.items():
        p = plan.patients[pid]
        for k in ks:
            mdl.con.add(run(pid, k) <= 1)          # each attempt runs at most once
            if k == 1:
                continue
            # attempt k only if attempt k-1 ran (and, by construction of
            # ``attempts``, failed) ...
            mdl.con.add(run(pid, k) <= run(pid, k - 1))
            # ... and no earlier than that attempt's release plus a re-collection
            gap = lead + cfg.tls + p.tt1
            mdl.con.add(start(pid, k) >= start(pid, k - 1) + gap
                        - (horizon + gap) * (1 - run(pid, k)))

    # rolling T_MFE-window capacity, per facility -- identical to the simulation
    rows = {}
    for (pid, k, _, _) in attempts:
        m = plan.patients[pid].facility
        for tau in days[(pid, k)]:
            for occ in range(tau, tau + cfg.tmfe):
                rows.setdefault((m, occ), []).append(mdl.x[pid, k, tau])
    for (m, occ), terms in rows.items():
        mdl.con.add(sum(terms) <= plan.fcap[m])

    # realised clinical loss: only a successful attempt delivers any survival
    const, obj = 0.0, []
    for pid, ks in by_patient.items():
        p = plan.patients[pid]
        const += cd.RHO[p.tier]
        for k in ks:
            if fails[pid][k - 1]:
                continue                           # a failed batch delivers nothing
            for tau in days[(pid, k)]:
                trt = tau + lead + p.tt3 - p.t0
                obj.append(cd.RHO[p.tier] * cd.survival(p.tier, trt)
                           * mdl.x[pid, k, tau])
    mdl.OBJ = Objective(expr=sum(obj), sense=maximize)

    out = ish.solve(mdl, time_limit=time_limit, mip_gap=mip_gap)
    if out.status not in ("optimal", "feasible"):
        return None, {"status": out.status, "note": out.note,
                      "attempts": len(attempts)}
    sched = {(pid, k): tau for (pid, k, _, _) in attempts
             for tau in days[(pid, k)] if value(mdl.x[pid, k, tau]) > 0.5}
    return sched, {"status": out.status,
                   "objective": const - out.obj,   # weighted clinical loss
                   "gap": out.gap, "wall_s": out.wall, "solver": out.solver,
                   "attempts": len(attempts), "variables": len(mdl.x)}
