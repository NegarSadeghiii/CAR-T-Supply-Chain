"""
simulation.py
=============

The daily discrete-event simulation that EXECUTES an operating policy on the
frozen strategic network -- the "MPC execution loop" of
``Survival_Aware_iSHIPMENT_Formulation.docx``.

The strategic MILP (1)-(35) is solved once (``strategic.py``) and fixes the
network.  This module advances a daily clock over the same 130-day horizon,
realises the one stochastic element the deterministic model omits --
manufacturing yield, Bernoulli(p_m) at release testing -- evolves each
patient's survival deterministically with the wait, and at every decision epoch
asks the policy who starts next.

Daily event sequence (the doc's order, one day at a time)
--------------------------------------------------------
1. **Arrivals.**  Patients collected on day t enter as waiting; their material
   is staged at the MS after TLS + TT1 days, at which point they join the ready
   set W_t.  Collection itself is subject to the S_min futility gate.
2. **Completions & yield.**  A job started on day s holds its slot for T_MFE
   days and frees it on day s + T_MFE; T_QC days later it is release-tested --
   pass (prob p_m) ships and is delivered TT3 days later, fail returns the
   patient to the waiting set with phi = 1 and a larger accrued wait.
3. **Health update.**  Every waiting patient's accrued wait grows by a day;
   because the clock runs continuously from the FIRST collection, this is
   implicit in ``t - t0`` rather than carried as separate state.
4. **Decision.**  For each facility with a free slot and ready patients, the
   policy chooses who starts; the slot is occupied for T_MFE days.
5. **Losses.**  A patient is lost (S = 0) only when a required (re-)collection
   fails the projected-survival gate or when K_remake is exhausted.  There is
   NO calendar-based removal at all -- no raw-wait cutoff and no backstop.
   Anyone still queueing keeps queueing and is credited with whatever survival
   their eventual delivery earns, even past the 130-day reporting horizon.
6. Advance to t+1.

Failure recourse (supersedes assumption 3).  A remake needs a FRESH
leukapheresis: feasible iff the PROJECTED survival at the remake's delivery is
still >= S_min, where the wait for a slot is read from the live occupancy; if
feasible it adds TLS + TT1 days and rho_leuk dollars, if not the patient is
cancelled and carries the full clinical loss.

Uncertainty and common random numbers
-------------------------------------
Only yield is random.  The draw for attempt k of patient i is a pure function
of ``(seed, pid, k)``, so the SAME batches fail under every policy, at every
offered load, and (nested) across the failure-rate sweep: differences between
policies are differences in scheduling, never in luck.

Expected loss is the probability sum ``sum_i (1 - S_i)`` on the realised
timeline (assumption 8), with S = 0 for a lost patient -- randomness lives only
in which batches fail, and survival given the timeline is exact.

Nothing in this module re-tunes the survival kernel: ``cart_data.survival`` is
the single source of S_u(t).
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

import cart_data as cd
import per_epoch as pe
import strategic as st

TIER_ORDER = cd.TIER_ORDER


@dataclass
class SimConfig:
    """Durations and policy parameters; every value is the confirmed one."""
    horizon: int = cd.HORIZON            # 130 d -- the reporting horizon
    tls: int = 1                         # leukapheresis duration
    tmfe: int = 7                        # manufacturing, slot-occupying
    tqc: int = 7                         # QC -- a delay, NOT a capacity resource
    lookahead: int = pe.LOOKAHEAD_H      # MPC window H
    s_min: float = pe.S_MIN              # futility gate
    k_remake: int = pe.K_REMAKE          # max REMAKES, then cancel (attempts = k+1)
    rho_leuk: float = pe.RHO_LEUK        # $ per re-collection
    backstop_wait: int = pe.BACKSTOP_WAIT    # None = no calendar-based removal
    fail_rate: float = None              # Exp E: common (1 - p) at every facility
    epoch_solver: str = "gurobi"
    drain_cap: int = 3 * cd.HORIZON      # run past the horizon until the queue drains

    @property
    def max_attempts(self) -> int:
        """Manufacturing attempts allowed per patient: the first plus k_remake."""
        return self.k_remake + 1


def yield_draw(seed, pid, attempt):
    """CRN uniform for attempt ``attempt`` of patient ``pid`` under ``seed``.

    Keyed on the patient and the attempt index rather than on call order, so
    the realisation is identical no matter which policy is running or when the
    attempt happens.
    """
    return random.Random(f"{seed}|{pid}|{attempt}").random()


@dataclass
class PatientRecord:
    pid: str
    tier: str
    t0: int
    facility: str
    status: str = "unborn"          # waiting|ready|manufacturing|qc|transit|treated|lost
    attempts: int = 0
    starts: list = field(default_factory=list)     # (attempt, start_day, outcome)
    ready_days: list = field(default_factory=list)
    failures: int = 0
    recollections: int = 0
    delivery: int = None
    trt: int = None
    survival: float = 0.0
    lost_reason: str = ""
    hold_total: int = 0
    hold_first: int = None
    spilled: bool = False           # delivered/resolved after the 130-day horizon

    @property
    def loss(self) -> float:
        return 1.0 - self.survival

    @property
    def weighted_loss(self) -> float:
        return cd.RHO[self.tier] * self.loss


@dataclass
class SimResult:
    policy: str
    seed: int
    n: int
    records: dict
    costs: dict
    metrics: dict
    events: list                     # one dict per manufacturing attempt started
    idle_slot_days: int = 0


class Simulator:
    """One replication of one policy on one frozen plan."""

    def __init__(self, plan, cfg: SimConfig, seed: int):
        self.plan, self.cfg, self.seed = plan, cfg, seed
        span = cfg.drain_cap + cfg.tmfe + cfg.tqc + 8
        self.span = span
        self.busy = {m: [0] * span for m in plan.opened}
        self.rec = {pid: PatientRecord(pid=pid, tier=p.tier, t0=p.t0,
                                       facility=p.facility)
                    for pid, p in plan.patients.items()}
        self.arrivals_on = {}
        for pid, p in plan.patients.items():
            self.arrivals_on.setdefault(p.t0, []).append(pid)
        self.release_on, self.deliver_on = {}, {}
        self.ready_at = {}                       # pid -> day it (re)joins W
        self.events = []
        self.costs = {"facility": plan.facility_cost, "material_qc": 0.0,
                      "transport_in": 0.0, "transport_out": 0.0, "releuk": 0.0}
        self.idle_slot_days = 0
        self.n_gate_initial = self.n_gate_recollect = self.n_backstop = 0

    # -- helpers ----------------------------------------------------------
    def p_pass(self, m) -> float:
        """Yield at facility m, or the common override Exp E sweeps."""
        if self.cfg.fail_rate is None:
            return st.YIELD_P[m]
        return 1.0 - self.cfg.fail_rate

    def occupy(self, m, start):
        """A job started on ``start`` holds its slot for T_MFE days."""
        for tau in range(start, min(start + self.cfg.tmfe, self.span)):
            self.busy[m][tau] += 1

    def candidates(self, m, t):
        """Ready waiting patients at facility m -- the doc's W_t, restricted to m."""
        out = []
        for pid, day in self.ready_at.items():
            if day > t:
                continue
            p = self.plan.patients[pid]
            if p.facility != m:
                continue
            out.append(pe.Candidate(pid=pid, tier=p.tier, t0=p.t0, tt3=p.tt3))
        return out

    def gate(self, pid, t):
        """S_min gate on a (re-)collection decided on day t; wait read from busy."""
        p = self.plan.patients[pid]
        cfg = self.cfg
        pre = cfg.tls + p.tt1
        wait = pe.wait_for_slot(self.busy[p.facility], self.plan.fcap[p.facility],
                                t + pre)
        return pe.futility_ok(p.tier, p.t0, t, pre, wait, cfg.tmfe, cfg.tqc,
                              p.tt3, cfg.s_min)

    def lose(self, pid, reason, t):
        r = self.rec[pid]
        r.status, r.lost_reason, r.survival = "lost", reason, 0.0
        r.spilled = t > self.cfg.horizon
        self.ready_at.pop(pid, None)

    # -- the daily loop ---------------------------------------------------
    def run(self, policy) -> SimResult:
        cfg = self.cfg
        self.policy = policy
        if hasattr(policy, "prepare"):
            policy.prepare(self)

        for t in range(1, cfg.drain_cap + 1):
            self._arrivals(t)
            self._completions(t)
            self._backstop(t)
            self._decisions(policy, t)
            if self._drained(t):
                break
        return self._result(policy)

    def _arrivals(self, t):
        """Step 1 -- collection, subject to the futility gate."""
        for pid in self.arrivals_on.get(t, []):
            r, p = self.rec[pid], self.plan.patients[pid]
            if not self.gate(pid, t):
                self.n_gate_initial += 1
                self.lose(pid, "gate_initial", t)
                continue
            if self._policy_cancels(pid):
                self.lose(pid, "policy_cancel", t)
                continue
            r.status = "waiting"
            ready = t + self.cfg.tls + p.tt1
            self.ready_at[pid] = ready
            r.ready_days.append(ready)
            self.costs["transport_in"] += p.u1       # material shipped LS -> MS

    def _completions(self, t):
        """Step 2 -- deliveries, release testing, and the failure recourse."""
        for pid in self.deliver_on.pop(t, []):
            r, p = self.rec[pid], self.plan.patients[pid]
            r.status, r.delivery = "treated", t
            r.trt = t - p.t0                          # one clock, first collection
            r.survival = cd.survival(p.tier, r.trt)
            r.spilled = t > self.cfg.horizon
            self.costs["transport_out"] += p.u3

        for pid in self.release_on.pop(t, []):
            r, p = self.rec[pid], self.plan.patients[pid]
            passed = yield_draw(self.seed, pid, r.attempts) <= self.p_pass(p.facility)
            outcome = "pass" if passed else "fail"
            r.starts[-1]["outcome"] = outcome          # at most one job in flight
            self.events[r.starts[-1]["event"]]["outcome"] = outcome
            if passed:
                r.status = "transit"
                self.deliver_on.setdefault(t + p.tt3, []).append(pid)
                continue
            r.failures += 1
            if r.attempts >= self.cfg.max_attempts:
                self.lose(pid, "k_remake", t)
                continue
            if not self.gate(pid, t):
                self.n_gate_recollect += 1
                self.lose(pid, "gate_recollection", t)
                continue
            if self._policy_cancels(pid):
                self.lose(pid, "policy_cancel", t)
                continue
            r.status = "waiting"                       # phi_i = 1, larger a_i
            r.recollections += 1
            ready = t + self.cfg.tls + p.tt1           # fresh leukapheresis
            self.ready_at[pid] = ready
            r.ready_days.append(ready)
            self.costs["releuk"] += self.cfg.rho_leuk
            self.costs["transport_in"] += p.u1

    def _policy_cancels(self, pid) -> bool:
        """Whether the policy declines this patient's next attempt outright.

        Only ``best_achievable`` uses it: its perfect-information solve may
        judge an attempt not worth the capacity, which is a decision it is
        entitled to make.  Every online policy starts everyone eventually, so
        the hook is absent and this is False.
        """
        policy = getattr(self, "policy", None)
        return bool(policy is not None and hasattr(policy, "will_never_start")
                    and policy.will_never_start(self, pid))

    def _backstop(self, t):
        """Step 3/5 -- optional calendar backstop, DISABLED by default.

        With ``backstop_wait = None`` there is no calendar-based removal at
        all: a patient leaves the system only by failing the S_min gate on a
        required (re-)collection, by exhausting K_remake, or by being treated.
        Everyone else keeps waiting and is credited with the survival their
        realised delivery earns them.
        """
        if self.cfg.backstop_wait is None:
            return
        for pid in [pid for pid in self.ready_at
                    if t - self.plan.patients[pid].t0 > self.cfg.backstop_wait]:
            self.n_backstop += 1
            self.lose(pid, "backstop", t)

    def _decisions(self, policy, t):
        """Step 4 -- one decision epoch per facility with a free slot."""
        for m in self.plan.opened:
            free = self.plan.fcap[m] - self.busy[m][t]
            if free <= 0:
                continue
            cands = self.candidates(m, t)
            if not cands:
                continue
            n_start = min(free, len(cands))
            chosen = policy.select(self, m, t, n_start, cands)
            self.idle_slot_days += n_start - len(chosen)
            for pid in chosen:
                self._start(pid, m, t)

    def _start(self, pid, m, t):
        r, p = self.rec[pid], self.plan.patients[pid]
        r.attempts += 1
        hold = t - self.ready_at.pop(pid)
        r.hold_total += hold
        if r.hold_first is None:
            r.hold_first = hold
        r.status = "manufacturing"
        r.starts.append({"event": len(self.events), "attempt": r.attempts,
                         "day": t, "facility": m, "hold": hold,
                         "outcome": "started"})
        self.events.append({"pid": pid, "tier": r.tier, "facility": m,
                            "attempt": r.attempts, "start": t, "hold": hold,
                            "outcome": "started"})
        self.occupy(m, t)
        self.costs["material_qc"] += self.plan.c_material + self.plan.c_qc
        self.release_on.setdefault(t + self.cfg.tmfe + self.cfg.tqc,
                                   []).append(pid)

    def _drained(self, t):
        return (not self.ready_at and not self.release_on and not self.deliver_on
                and all(d <= t for d in self.arrivals_on))

    # -- reporting --------------------------------------------------------
    def _result(self, policy):
        recs = self.rec
        unresolved = [r.pid for r in recs.values()
                      if r.status not in ("treated", "lost")]
        if unresolved:
            raise RuntimeError(f"{len(unresolved)} patients unresolved at drain "
                               f"cap (e.g. {unresolved[:3]})")
        costs = dict(self.costs)
        costs["total"] = sum(costs.values())
        treated = [r for r in recs.values() if r.status == "treated"]
        costs["per_therapy"] = costs["total"] / len(treated) if treated else None
        return SimResult(policy=getattr(policy, "name", str(policy)),
                         seed=self.seed, n=len(recs), records=recs, costs=costs,
                         metrics=self.metrics(costs, treated), events=self.events,
                         idle_slot_days=self.idle_slot_days)

    def metrics(self, costs, treated):
        recs = list(self.rec.values())
        out = {
            "policy_n": len(recs),
            "treated": len(treated),
            "lost": sum(1 for r in recs if r.status == "lost"),
            "expected_lost": sum(r.loss for r in recs),
            "weighted_loss": sum(r.weighted_loss for r in recs),
            "failures": sum(r.failures for r in recs),
            "recollections": sum(r.recollections for r in recs),
            "attempts": sum(r.attempts for r in recs),
            "lost_gate_initial": self.n_gate_initial,
            "lost_gate_recollection": self.n_gate_recollect,
            "lost_k_remake": sum(1 for r in recs if r.lost_reason == "k_remake"),
            "lost_backstop": self.n_backstop,
            "lost_policy_cancel": sum(1 for r in recs
                                      if r.lost_reason == "policy_cancel"),
            "spillover": sum(1 for r in recs if r.spilled),
            "idle_slot_days": self.idle_slot_days,
            "total_cost": costs["total"],
            "cost_per_therapy": costs["per_therapy"],
            "mean_TRT": statistics.fmean([r.trt for r in treated]) if treated else None,
        }
        out["spillover_share"] = out["spillover"] / len(recs)
        for u in TIER_ORDER:
            sub = [r for r in recs if r.tier == u]
            tre = [r for r in sub if r.status == "treated"]
            out[f"n_{u}"] = len(sub)
            out[f"expected_lost_{u}"] = sum(r.loss for r in sub)
            out[f"lost_{u}"] = sum(1 for r in sub if r.status == "lost")
            out[f"mean_hold_{u}"] = statistics.fmean([r.hold_total for r in sub])
            first = [r.hold_first for r in sub if r.hold_first is not None]
            out[f"mean_hold_first_{u}"] = statistics.fmean(first) if first else None
            out[f"mean_TRT_{u}"] = (statistics.fmean([r.trt for r in tre])
                                    if tre else None)
        out["mean_hold"] = statistics.fmean([r.hold_total for r in recs])
        return out


def simulate(plan, policy, seed=0, cfg: SimConfig = None) -> SimResult:
    """Run one replication of ``policy`` on ``plan`` under yield seed ``seed``."""
    return Simulator(plan, cfg or SimConfig(), seed).run(policy)
