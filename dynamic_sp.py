"""
dynamic_sp.py — dynamic two-stage CAR-T SP with endogenous, delay-driven
clinical deterioration.

This EXTENDS yield_sp_v1 (which it does NOT import-modify) by replacing the
time-independent re-collection eligibility B_i(omega) ~ Bernoulli(beta_u) with
StillEligible_i(omega), an OUTCOME of the proportional-hazards decline kernel
(declineprob.py) evaluated at the patient's accrued WaitTime, where the wait
includes any congestion-driven re-manufacture delay (clearing_function.py).

Design (kept inside the scenario-based SAA/MILP framework)
---------------------------------------------------------
For each scenario omega the eligibility used by the optimizer is precomputed as
a binary parameter, so the MILP stays linear (mirrors yield_sp_v1.build_sp_model
exactly, constraint-for-constraint, with StillEligible replacing B):

    m*(i)            = nominal (decision-independent) facility for patient i
    F_i(omega)       = 1 - Y[omega, i, m*(i)]            (primary failure at nominal)
    rho_m(omega)     = (# nominal failures routed to m) / s_max[m]   (congestion)
    RemakeDelay(omega)= clearing_function(rho_{m*(i)}(omega))         (days)
    WaitTime_i(omega)= F_i(omega) * RemakeDelay_{m*(i)}(omega)        (extra wait)
    ExtraSurvive_i(omega) ~ Bernoulli( exp(-kappa * HR_u * (WaitTime/lambda)^gamma) )
    StillEligible_i(omega) = B_i(omega) AND ExtraSurvive_i(omega)

kappa = 0  =>  ExtraSurvive == 1  =>  StillEligible == B  =>  EXACTLY v1.
This is the nesting knob validated in test_nesting.py.

Endogeneity note. The optimizer's congestion proxy is DECISION-INDEPENDENT (uses
the nominal assignment and installed s_max), which keeps the model linear; the
fully decision-dependent congestion (using the chosen C and actual recourse
routing) lives in the out-of-sample simulator in value_of_endogeneity.py. The
value of endogeneity comes from the optimizer using the correct delay-degraded
eligibility rather than the optimistic flat beta_u.

Backend. HiGHS (as in disruption_experiment.py / case_study.py), because the
full 50-patient instance exceeds the size-limited Gurobi license. Stage-2
recourse vars are continuous [0,1]; the fixed-first-stage matrix is network-flow
/ totally-unimodular so the LP relaxation is integral at the optimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import highspy as hs

from yield_sp_v1 import Instance, sample_scenarios
from declineprob import calibrate_lambda, extra_survival, HR_TIER
from clearing_function import ClearingFunction, DEFAULT_CLEARING


# ---------------------------------------------------------------------------
# Endogenous eligibility (optimization-side, decision-independent congestion)
# ---------------------------------------------------------------------------

def _nominal_assignment(inst: Instance) -> np.ndarray:
    """
    Decision-independent reference facility per patient: nearest shelf-life-
    feasible facility (min transport time); if no transport data, highest-yield
    facility. Used only to precompute the scenario congestion/eligibility
    parameters — it does not constrain the optimizer's actual assignment x.
    """
    n_p, n_f = inst.n_patients, inst.n_facilities
    if inst.t_trans is not None:
        t = np.asarray(inst.t_trans, float).copy()
        if inst.shelf_life is not None:
            t = np.where(t <= inst.shelf_life, t, np.inf)
        # patients with no feasible facility fall back to global best-yield
        assign = []
        for i in range(n_p):
            row = t[i]
            assign.append(int(np.argmin(row)) if np.isfinite(row).any()
                          else int(np.argmax(inst.p)))
        return np.asarray(assign)
    return np.full(n_p, int(np.argmax(inst.p)))


def compute_still_eligible(
    inst: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    tiers,
    *,
    kappa: float,
    gamma: float = 1.0,
    clearing: ClearingFunction = DEFAULT_CLEARING,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Precompute the binary StillEligible[omega, i] used by the optimizer.

    StillEligible = B AND Bernoulli(extra_survival(WaitTime)).  With kappa=0 the
    extra survival is identically 1, so StillEligible == B (exact v1 nesting) for
    ANY input B — including the hand-crafted toy scenarios.

    Returns (StillEligible[N,I] float 0/1, diagnostics dict).
    """
    N, n_p, n_f = Y.shape
    lam = calibrate_lambda(gamma)
    assign = _nominal_assignment(inst)                     # (n_p,)
    slots = np.asarray(inst.s_max, float)

    # Primary failure at the nominal facility, per scenario.
    Ynom = Y[:, np.arange(n_p), assign]                    # (N, n_p)
    F = 1.0 - Ynom                                          # 1 if nominal batch failed

    # Facility congestion from nominal failures; RemakeDelay at each patient's
    # nominal facility.
    wait = np.zeros((N, n_p))
    rho_diag = np.zeros((N, n_f))
    for o in range(N):
        demand = np.zeros(n_f)
        for i in range(n_p):
            if F[o, i] > 0.5:
                demand[assign[i]] += 1.0
        rho = demand / np.maximum(slots, 1e-9)
        rho_diag[o] = rho
        delay_m = clearing.remake_delay(rho)               # (n_f,)
        wait[o] = F[o] * delay_m[assign]                   # extra wait if failed

    # Extra-survival probability per (scenario, patient) via the PH kernel.
    hr = np.array([HR_TIER[t] for t in tiers])             # (n_p,)
    p_extra = np.empty((N, n_p))
    for i in range(n_p):
        p_extra[:, i] = extra_survival(wait[:, i], hr[i], lam, gamma, kappa)

    # Independent Bernoulli draw on a stream kept separate from (Y, B).
    rng = np.random.default_rng(seed + 20_260_713)
    extra = (rng.random((N, n_p)) < p_extra).astype(float)
    still = B * extra

    diag = {
        "lambda": lam, "gamma": gamma, "kappa": kappa,
        "mean_wait_days": float(wait[F > 0.5].mean()) if np.any(F > 0.5) else 0.0,
        "mean_rho": float(rho_diag.mean()),
        "mean_still_elig": float(still.mean()),
        "mean_B": float(B.mean()),
    }
    return still, diag


# ---------------------------------------------------------------------------
# HiGHS SAA model (mirrors yield_sp_v1.build_sp_model; StillEligible replaces B)
# ---------------------------------------------------------------------------

def solve_dynamic_sp(
    inst: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    *,
    tiers=None,
    kappa: float = 0.0,
    gamma: float = 1.0,
    clearing: ClearingFunction = DEFAULT_CLEARING,
    still_eligible: np.ndarray | None = None,
    fix_first_stage: dict | None = None,
    seed: int = 0,
    mip_gap: float = 1e-4,
    time_limit: float = 600.0,
) -> dict:
    """
    Build and solve the dynamic SP with HiGHS and return the v1-style result dict
    (total_cost, stage1_cost, expected_stage2_cost, z, C, x, r_remfg, r_sub,
    r_cancel, solve_time, gap, plus 'still_eligible' and 'elig_diag').

    If `still_eligible` is provided it is used directly; otherwise it is computed
    from (Y, B, tiers, kappa, gamma, clearing). `tiers` is a length-n_patients
    list of "H"/"M"/"L"; defaults to all "M" (kappa=0 makes tiers irrelevant).
    `fix_first_stage` pins {'z':{m:v},'C':{m:v},'x':{(i,m):v}} exactly as
    yield_sp_v1.build_sp_model does (used by the nesting test).
    """
    N, n_p, n_f = Y.shape
    if tiers is None:
        tiers = ["M"] * n_p

    if still_eligible is None:
        still_eligible, elig_diag = compute_still_eligible(
            inst, Y, B, tiers, kappa=kappa, gamma=gamma, clearing=clearing, seed=seed)
    else:
        elig_diag = {"provided": True}
    SE = np.asarray(still_eligible, float)

    n_sub_pp = n_f * (n_f - 1)
    n_per_s = n_p * n_f + n_p * n_sub_pp + n_p
    n_1st = 2 * n_f + n_p * n_f
    n_total = n_1st + N * n_per_s

    def _soff(m, mp):
        return m * (n_f - 1) + (mp if mp < m else mp - 1)

    def vz(m):            return m
    def vC(m):            return n_f + m
    def vx(i, m):         return 2 * n_f + i * n_f + m
    def vr(o, i, m):      return n_1st + o * n_per_s + i * n_f + m
    def vs(o, i, m, mp):  return n_1st + o * n_per_s + n_p * n_f + i * n_sub_pp + _soff(m, mp)
    def vc(o, i):         return n_1st + o * n_per_s + n_p * n_f + n_p * n_sub_pp + i

    h = hs.Highs()
    h.silent()
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("mip_rel_gap", mip_gap)

    lbs = np.zeros(n_total)
    ubs = np.ones(n_total)
    for m in range(n_f):
        ubs[vC(m)] = float(inst.s_max[m])

    costs = np.zeros(n_total)
    for m in range(n_f):
        costs[vz(m)] = inst.f[m]
        costs[vC(m)] = inst.pi[m]
    for i in range(n_p):
        for m in range(n_f):
            costs[vx(i, m)] = inst.c[m]
    prob = 1.0 / N
    for o in range(N):
        for i in range(n_p):
            for m in range(n_f):
                costs[vr(o, i, m)] = prob * (inst.rho_leuk + inst.rho_remfg[m])
                for mp in range(n_f):
                    if mp != m:
                        costs[vs(o, i, m, mp)] = prob * (inst.rho_leuk + inst.rho_sub[m, mp])
            costs[vc(o, i)] = prob * inst.rho_cancel[i]

    h.addCols(n_total, costs, lbs, ubs, 0,
              np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64))

    int_t = hs.HighsVarType.kInteger
    fs_idx = np.arange(n_1st, dtype=np.int32)
    h.changeColsIntegrality(n_1st, fs_idx, np.array([int_t] * n_1st))

    # Shelf-life prefilter (constraint 13), matching case_study.py.
    if inst.t_trans is not None and inst.shelf_life is not None:
        for i in range(n_p):
            for m in range(n_f):
                if inst.t_trans[i, m] > inst.shelf_life:
                    h.changeColBounds(vx(i, m), 0.0, 0.0)

    # Pin the first stage exactly (nesting / evaluation).
    if fix_first_stage is not None:
        for m in range(n_f):
            h.changeColBounds(vz(m), float(fix_first_stage["z"][m]), float(fix_first_stage["z"][m]))
            h.changeColBounds(vC(m), float(fix_first_stage["C"][m]), float(fix_first_stage["C"][m]))
        for i in range(n_p):
            for m in range(n_f):
                v = float(fix_first_stage["x"][i, m])
                h.changeColBounds(vx(i, m), v, v)

    INF = 1e30

    def _row(lb, ub, inds, vals):
        ia = np.asarray(inds, dtype=np.int32)
        va = np.asarray(vals, dtype=np.float64)
        h.addRow(lb, ub, len(ia), ia, va)

    # (1) sum_m x[i,m] = 1
    for i in range(n_p):
        _row(1.0, 1.0, [vx(i, m) for m in range(n_f)], [1.0] * n_f)
    # (2) x[i,m] <= z[m]
    for i in range(n_p):
        for m in range(n_f):
            _row(-INF, 0.0, [vx(i, m), vz(m)], [1.0, -1.0])
    # (3) sum_i x[i,m] <= C[m]
    for m in range(n_f):
        _row(-INF, 0.0, [vx(i, m) for i in range(n_p)] + [vC(m)], [1.0] * n_p + [-1.0])
    # (4) C[m] <= s_max[m] * z[m]
    for m in range(n_f):
        _row(-INF, 0.0, [vC(m), vz(m)], [1.0, -float(inst.s_max[m])])
    # (15) MNF
    if inst.mnf is not None:
        _row(-INF, float(inst.mnf), [vz(m) for m in range(n_f)], [1.0] * n_f)

    for o in range(N):
        for i in range(n_p):
            remfg_i = [vr(o, i, m) for m in range(n_f)]
            sub_i = [vs(o, i, m, mp) for m in range(n_f) for mp in range(n_f) if mp != m]
            x_i = [vx(i, m) for m in range(n_f)]
            # (8) completion: sum recourse = sum_m (1 - Y[i,m]) x[i,m]
            _row(0.0, 0.0,
                 remfg_i + sub_i + [vc(o, i)] + x_i,
                 [1.0] * n_f + [1.0] * n_sub_pp + [1.0]
                 + [-(1.0 - Y[o, i, m]) for m in range(n_f)])
            # (10) eligibility: re-collection limited by StillEligible (was B)
            _row(-INF, float(SE[o, i]), remfg_i + sub_i, [1.0] * (n_f + n_sub_pp))
            # (10b) source tied to primary assignment
            for m in range(n_f):
                sub_from_m = [vs(o, i, m, mp) for mp in range(n_f) if mp != m]
                _row(-INF, 0.0, [vr(o, i, m)] + sub_from_m + [vx(i, m)],
                     [1.0] * (n_f) + [-1.0])
        for m in range(n_f):
            # (9) recourse capacity: remfg + incoming sub <= C[m] - primary successes
            remfg_m = [vr(o, i, m) for i in range(n_p)]
            sub_to_m = [vs(o, i, mp, m) for i in range(n_p) for mp in range(n_f) if mp != m]
            x_m = [vx(i, m) for i in range(n_p)]
            _row(-INF, 0.0,
                 remfg_m + sub_to_m + x_m + [vC(m)],
                 [1.0] * n_p + [1.0] * (n_p * (n_f - 1))
                 + [float(Y[o, i, m]) for i in range(n_p)] + [-1.0])

    t0 = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t0

    status = h.getModelStatus()
    ok = {hs.HighsModelStatus.kOptimal, hs.HighsModelStatus.kObjectiveBound,
          hs.HighsModelStatus.kSolutionLimit, hs.HighsModelStatus.kTimeLimit}
    if status not in ok:
        raise RuntimeError(f"HiGHS dynamic SP solve failed: {status}")
    gap = h.getInfoValue("mip_gap")[1] if status != hs.HighsModelStatus.kOptimal else 0.0
    col = h.getSolution().col_value

    z_arr = np.array([round(col[vz(m)]) for m in range(n_f)])
    C_arr = np.array([round(col[vC(m)]) for m in range(n_f)])
    x_arr = np.array([[round(col[vx(i, m)]) for m in range(n_f)] for i in range(n_p)])

    r_remfg = np.zeros((N, n_p, n_f))
    r_sub = np.zeros((N, n_p, n_f, n_f))
    r_cancel = np.zeros((N, n_p))
    for o in range(N):
        for i in range(n_p):
            for m in range(n_f):
                r_remfg[o, i, m] = round(col[vr(o, i, m)])
                for mp in range(n_f):
                    if mp != m:
                        r_sub[o, i, m, mp] = round(col[vs(o, i, m, mp)])
            r_cancel[o, i] = round(col[vc(o, i)])

    stage1 = (sum(inst.f[m] * z_arr[m] for m in range(n_f))
              + sum(inst.pi[m] * C_arr[m] for m in range(n_f))
              + sum(inst.c[m] * x_arr[i, m] for i in range(n_p) for m in range(n_f)))
    s2 = np.zeros(N)
    for o in range(N):
        for i in range(n_p):
            for m in range(n_f):
                s2[o] += (inst.rho_leuk + inst.rho_remfg[m]) * r_remfg[o, i, m]
                for mp in range(n_f):
                    if mp != m:
                        s2[o] += (inst.rho_leuk + inst.rho_sub[m, mp]) * r_sub[o, i, m, mp]
            s2[o] += inst.rho_cancel[i] * r_cancel[o, i]

    return {
        "total_cost": stage1 + s2.mean(),
        "stage1_cost": stage1,
        "expected_stage2_cost": s2.mean(),
        "z": z_arr, "C": C_arr, "x": x_arr,
        "r_remfg": r_remfg, "r_sub": r_sub, "r_cancel": r_cancel,
        "solve_time": solve_time, "gap": gap,
        "still_eligible": SE, "elig_diag": elig_diag,
    }
