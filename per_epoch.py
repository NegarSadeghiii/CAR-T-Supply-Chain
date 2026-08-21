"""
per_epoch.py
============

The per-epoch operational optimisation of the survival-aware i-SHIPMENT model
-- equations (P1)-(P8) of ``Survival_Aware_iSHIPMENT_Formulation.docx``.

The strategic MILP (1)-(35) is solved ONCE (see ``strategic.py``) and freezes
the network: open facilities, the link structure, FCAP, the assignment m(i) and
the transport modes.  Nothing here re-decides any of that.  The ONLY decision
is ``x[i, tau] in {0,1}`` -- start patient i's manufacturing on day tau at its
frozen facility m(i) -- and only the immediate start (tau = t) is implemented
before the simulation rolls forward to the next epoch.

Contents
--------
``solve_epoch``   (P1)-(P6): the look-ahead MILP over [t, t+H], used by the
                  ``adaptive_mpc`` policy.
``index_choice``  (P8): the closed-form survival index -- serve whoever's
                  life-value-weighted survival falls most from one more day of
                  waiting -- used by the ``survival_index`` policy.
``futility_ok``   the S_min gate on (re-)collection: feasible iff the PROJECTED
                  survival at delivery is still >= S_min, where the wait for a
                  slot is read from the live occupancy the capacity constraint
                  (P5) uses.

Survival clock
--------------
(P1) reads ``elapsed_i(tau) = a_i + (tau - t) + T_MFE + T_QC + TT3_m(i)`` with
``a_i`` the accrued wait observed at the epoch.  With one continuous clock from
the FIRST collection, ``a_i = t - t0_i``, so

    elapsed_i(tau) = tau - t0_i + T_MFE + T_QC + TT3_i

which is what the code evaluates: the same quantity, without carrying ``a_i``
as separate state.  (P2) is then ``S_i(tau) = (1 - w_u)^(elapsed/42)``, i.e.
``cart_data.survival`` -- the frozen kernel, never re-tuned here.

The problem decomposes exactly by facility: m(i) is frozen, so patients
partition across facilities and (P5) never couples two of them.  Every routine
below therefore takes the candidate set of ONE facility.

Gurobi is imported lazily inside the solver so that importing this module never
requires a licence; if it is unavailable the same MILP is solved through the
Pyomo/HiGHS plumbing already used by ``ishipment_survival``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cart_data as cd

# ---------------------------------------------------------------------------
# Confirmed operational parameters (doc: "Parameters (all values)" + the
# resolved re-collection recourse).  None of these are re-fitted anywhere.
# ---------------------------------------------------------------------------
LOOKAHEAD_H = 7          # MPC look-ahead window [t, t+H], days
S_MIN = 0.75             # futility gate on PROJECTED survival at delivery
K_REMAKE = 2             # max REMAKES per patient (3 attempts), then cancel
C_RELEUK = 5000.0        # $ per re-leukapheresis (re-collection) attempt
BACKSTOP_WAIT = None     # no calendar-based removal; the S_min gate is the
                         # only futility rule (T_elig was dropped)

# Gurobi's size-limited licence caps a model at 2000 columns; above that the
# epoch model is sent to HiGHS instead.
_GUROBI_VAR_BUDGET = 1800

_ENV = {"gurobi": None, "checked": False}


# ---------------------------------------------------------------------------
# Epoch state
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """One ready, waiting patient as the per-epoch model sees it.

    ``t0`` is the FIRST collection day -- the origin of the survival clock, so
    a remade patient carries the deterioration of every earlier attempt.
    """
    pid: str
    tier: str
    t0: int
    tt3: int                 # MS -> hospital transport time of its frozen mode
    rank: tuple = ()         # ordering key used by the rule-based policies

    def elapsed_at_delivery(self, start_day: int, tmfe: int, tqc: int) -> int:
        """(P1) elapsed at delivery if manufacturing starts on ``start_day``."""
        return start_day + tmfe + tqc + self.tt3 - self.t0

    def survival_if_started(self, start_day: int, tmfe: int, tqc: int) -> float:
        """(P2) S_i(tau), evaluated from the observed state."""
        return cd.survival(self.tier, self.elapsed_at_delivery(start_day, tmfe, tqc))


# ---------------------------------------------------------------------------
# Slot look-ahead and the S_min futility gate
# ---------------------------------------------------------------------------
def first_free_day(busy, fcap: int, earliest: int) -> int:
    """First day >= ``earliest`` on which facility m still has a free slot.

    ``busy[tau]`` is the occupancy the capacity constraint (P5) sees: slots
    taken by jobs already in progress.  Beyond the last in-progress job the
    facility is idle, so the scan always terminates.
    """
    tau = max(earliest, 0)
    while tau < len(busy) and busy[tau] >= fcap:
        tau += 1
    return tau


def wait_for_slot(busy, fcap: int, earliest: int) -> int:
    """Days between ``earliest`` and the first free slot (0 if free now)."""
    return first_free_day(busy, fcap, earliest) - max(earliest, 0)


def projected_delivery_survival(tier, t0, now, pre_days, slot_wait, tmfe, tqc, tt3):
    """Projected survival at delivery for a (re-)collection decided at ``now``.

        elapsed = a_i + (TLS + TT1) + wait_for_slot + T_MFE + T_QC + TT3
        a_i     = now - t0            (one clock, from the FIRST collection)

    ``pre_days`` is TLS + TT1 -- the (re-)leukapheresis and its transport --
    which is still ahead of the patient at both initial collection and remake.
    """
    elapsed = (now - t0) + pre_days + slot_wait + tmfe + tqc + tt3
    return cd.survival(tier, elapsed)


def futility_ok(tier, t0, now, pre_days, slot_wait, tmfe, tqc, tt3,
                s_min: float = S_MIN) -> bool:
    """The single gate on collection and re-collection: projected S >= S_min.

    Sicker and later patients cross the floor first, so they are precisely the
    ones who cannot (re-)collect.  There is no raw-wait eligibility cutoff.
    """
    return projected_delivery_survival(tier, t0, now, pre_days, slot_wait,
                                       tmfe, tqc, tt3) >= s_min


# ---------------------------------------------------------------------------
# (P8) the interpretable survival index
# ---------------------------------------------------------------------------
def index_score(c: Candidate, t: int, tmfe: int, tqc: int) -> float:
    """alpha_u(i) * [ S_i(t) - S_i(t+1) ] -- the one-day loss of life value.

    Weights are the reference-tier-normalised life values ALPHA_W, so the score
    is in reference-tier-equivalent lives per day; the argmax is unchanged by
    that normalisation because it is a single positive scale factor.
    """
    return cd.ALPHA_W[c.tier] * (c.survival_if_started(t, tmfe, tqc)
                                 - c.survival_if_started(t + 1, tmfe, tqc))


def index_choice(cands, t: int, n_start: int, tmfe: int, tqc: int):
    """(P8) start the ``n_start`` patients whose weighted survival falls most.

    Ties break on the earliest first collection, then the patient id, so the
    rule is deterministic and reproducible.
    """
    ordered = sorted(cands,
                     key=lambda c: (-index_score(c, t, tmfe, tqc), c.t0, c.pid))
    return [c.pid for c in ordered[:n_start]]


# ---------------------------------------------------------------------------
# (P1)-(P6) the look-ahead MILP
# ---------------------------------------------------------------------------
def _epoch_data(cands, t, busy, fcap, tmfe, tqc, horizon_H):
    """Coefficients shared by both solver backends.

    Returns the start-day window, the per-candidate survival gain of starting
    on tau instead of being deferred past the window, and the free capacity
    profile.  (P3)'s objective

        min  sum_i alpha_i (1 - S_hat_i),
        S_hat_i = sum_tau S_i(tau) x_i,tau + d_i (1 - sum_tau x_i,tau),
        d_i     = S_i(t + H)                       (deferral charged at the
                                                    survival it would lose)

    is, dropping the constant sum_i alpha_i (1 - d_i), equivalent to

        max  sum_i,tau alpha_i [S_i(tau) - d_i] x_i,tau.

    alpha_i is carried in reference-tier-equivalent lives (ALPHA_W); the epoch
    problem has no cost term, so the common factor alpha_ref scales the whole
    objective and leaves the argmax untouched.

    The frozen operational cost c^op_i is inert on the frozen network -- it is
    a per-patient constant and every started patient is started exactly once --
    so it drops out of the argmin and is not monetised into (P3).
    """
    taus = list(range(t, t + horizon_H + 1))
    gain = {}
    for c in cands:
        d_i = c.survival_if_started(taus[-1], tmfe, tqc)      # continuation
        a_i = cd.ALPHA_W[c.tier]
        gain[c.pid] = {tau: a_i * (c.survival_if_started(tau, tmfe, tqc) - d_i)
                       for tau in taus}
    free = {tau: fcap - (busy[tau] if tau < len(busy) else 0) for tau in taus}
    return taus, gain, free


def solve_epoch(cands, t, busy, fcap, tmfe, tqc, horizon_H=LOOKAHEAD_H,
                n_start=None, solver="gurobi"):
    """(P1)-(P6) at one facility: which ready patients start TODAY.

    ``cands``    ready waiting patients assigned to this facility
    ``busy``     per-day occupancy from jobs already in progress
    ``fcap``     FCAP_m, the facility's concurrent-slot capacity
    ``n_start``  starts to implement today; defaults to the non-idling value
                 min(free slots today, |cands|)

    Only the immediate starts are returned -- the rest of the look-ahead plan
    is discarded, as the MPC loop re-solves at the next epoch on the updated
    state.
    """
    if not cands:
        return []
    taus, gain, free = _epoch_data(cands, t, busy, fcap, tmfe, tqc, horizon_H)
    if n_start is None:
        n_start = min(max(free[t], 0), len(cands))
    if n_start <= 0:
        return []
    if n_start >= len(cands):                     # every candidate starts today
        return [c.pid for c in cands]

    backend = solver
    if backend == "gurobi" and len(cands) * len(taus) > _GUROBI_VAR_BUDGET:
        backend = "highs"                          # size-limited licence
    if backend == "gurobi":
        chosen = _solve_epoch_gurobi(cands, taus, gain, free, tmfe, n_start)
        if chosen is not None:
            return chosen
    return _solve_epoch_pyomo(cands, taus, gain, free, tmfe, n_start)


def _capacity_rows(taus, tmfe):
    """(P5) rows: for each tau, the starts that still occupy a slot on tau."""
    for tau in taus:
        window = [tp for tp in taus if tau - tmfe < tp <= tau]
        if window:
            yield tau, window


def _solve_epoch_gurobi(cands, taus, gain, free, tmfe, n_start):
    """(P1)-(P6) through gurobipy.  Returns None if Gurobi is unusable."""
    try:
        import gurobipy as gp                      # lazy: no licence needed to import
        from gurobipy import GRB
    except Exception:                              # noqa: BLE001
        return None
    if not _ENV["checked"]:
        _ENV["checked"] = True
        try:
            env = gp.Env(params={"OutputFlag": 0, "Seed": 0, "Threads": 1})
            _ENV["gurobi"] = env
        except Exception:                          # noqa: BLE001
            _ENV["gurobi"] = None
    if _ENV["gurobi"] is None:
        return None

    try:
        mdl = gp.Model(env=_ENV["gurobi"])
        x = mdl.addVars([(c.pid, tau) for c in cands for tau in taus],
                        vtype=GRB.BINARY, name="x")
        # (P4) started at most once; otherwise deferred to a later epoch
        for c in cands:
            mdl.addConstr(gp.quicksum(x[c.pid, tau] for tau in taus) <= 1)
        # (P5) rolling-window capacity net of the jobs already in progress
        for tau, window in _capacity_rows(taus, tmfe):
            mdl.addConstr(
                gp.quicksum(x[c.pid, tp] for c in cands for tp in window)
                <= max(free[tau], 0))
        # non-idling: a free slot with a ready patient is always used
        mdl.addConstr(gp.quicksum(x[c.pid, taus[0]] for c in cands) == n_start)
        # (P3) -- (P6) is enforced by construction: tau >= t in ``taus``
        mdl.setObjective(gp.quicksum(gain[c.pid][tau] * x[c.pid, tau]
                                     for c in cands for tau in taus),
                         GRB.MAXIMIZE)
        mdl.optimize()
        if mdl.SolCount == 0:
            return None
        return [c.pid for c in cands if x[c.pid, taus[0]].X > 0.5]
    except Exception:                              # noqa: BLE001
        return None


def _solve_epoch_pyomo(cands, taus, gain, free, tmfe, n_start):
    """Same MILP through Pyomo, solved by the HiGHS fallback of the study."""
    from pyomo.environ import (ConcreteModel, Var, Binary, Objective,
                               ConstraintList, maximize, value)
    import ishipment_survival as ish

    mdl = ConcreteModel(name="per-epoch (P1)-(P6)")
    idx = [(c.pid, tau) for c in cands for tau in taus]
    mdl.x = Var(idx, within=Binary)
    mdl.con = ConstraintList()
    for c in cands:
        mdl.con.add(sum(mdl.x[c.pid, tau] for tau in taus) <= 1)
    for tau, window in _capacity_rows(taus, tmfe):
        mdl.con.add(sum(mdl.x[c.pid, tp] for c in cands for tp in window)
                    <= max(free[tau], 0))
    mdl.con.add(sum(mdl.x[c.pid, taus[0]] for c in cands) == n_start)
    mdl.OBJ = Objective(expr=sum(gain[c.pid][tau] * mdl.x[c.pid, tau]
                                 for c in cands for tau in taus),
                        sense=maximize)
    out = ish.solve(mdl, time_limit=60, mip_gap=1e-6, solver_pref="highs")
    if out.status not in ("optimal", "feasible"):
        # Substituting a different rule here would silently turn adaptive_mpc
        # into another policy, so the epoch fails loudly instead.
        raise RuntimeError(f"per-epoch MILP not solved: {out.status} {out.note}")
    return [c.pid for c in cands if value(mdl.x[c.pid, taus[0]]) > 0.5]
