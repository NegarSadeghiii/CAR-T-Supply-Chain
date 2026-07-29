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

Endogeneity note. Each individual MILP solve treats StillEligible as a fixed
scenario parameter (linear). The DECISION-DEPENDENT congestion — a design's
eligibility depends on the congestion that design creates through its capacity C
and routing x — is captured by a FIXED-POINT iteration
(solve_dynamic_sp_endogenous): re-estimate congestion from the current design,
re-solve, repeat to self-consistency. This is the same congestion model the
out-of-sample simulator (value_of_endogeneity.py) uses, so the endogenous design
is optimized against the eligibility it will actually face. The nominal-proxy
path (compute_still_eligible) is retained only for the kappa=0 nesting check,
where it returns B exactly.

Backend. HiGHS (as in case_study.py), because the
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
    Decision-independent reference facility per patient: the highest-yield
    facility. Used only to precompute the scenario congestion/eligibility
    parameters — it does not constrain the optimizer's actual assignment x.
    """
    n_p = inst.n_patients
    return np.full(n_p, int(np.argmax(inst.p)))


def _still_from_assignment(inst, Y, B, assign, slots, tiers, *,
                           kappa, gamma, clearing, seed):
    """
    Core eligibility computation shared by the nominal-proxy and the
    design-consistent variants. `assign[i]` is the facility patient i is routed
    to; `slots[m]` is the capacity used as the congestion denominator.

    StillEligible = B AND Bernoulli(extra_survival(WaitTime)); WaitTime is the
    congestion-driven re-manufacture delay incurred by a failed patient. With
    kappa=0 the survival factor is identically 1, so StillEligible == B.
    """
    N, n_p, n_f = Y.shape
    lam = calibrate_lambda(gamma)
    assign = np.asarray(assign)
    slots = np.asarray(slots, float)

    Yass = Y[:, np.arange(n_p), assign]                    # (N, n_p)
    F = 1.0 - Yass                                          # 1 if routed batch failed

    # Base primary utilization a_m / slots_m (a_m = patients routed to m). A
    # cost-minimizing design keeps capacity tight (a_m ~ C_m -> rho ~ 1), so a
    # re-manufacture queues behind a near-full line; buying spare capacity or
    # spreading load lowers rho and hence delay. Per scenario the failure surge
    # (batches needing a re-make at m) adds to the consumed slots.
    a_m = np.zeros(n_f)
    for i in range(n_p):
        a_m[assign[i]] += 1.0

    wait = np.zeros((N, n_p))
    rho_diag = np.zeros((N, n_f))
    for o in range(N):
        surge = np.zeros(n_f)
        for i in range(n_p):
            if F[o, i] > 0.5:
                surge[assign[i]] += 1.0
        rho = (a_m + surge) / np.maximum(slots, 1e-9)
        rho_diag[o] = rho
        delay_m = clearing.remake_delay(rho)               # (n_f,)
        wait[o] = F[o] * delay_m[assign]

    hr = np.array([HR_TIER[t] for t in tiers])
    p_extra = np.empty((N, n_p))
    for i in range(n_p):
        p_extra[:, i] = extra_survival(wait[:, i], hr[i], lam, gamma, kappa)

    rng = np.random.default_rng(seed + 20_260_713)
    extra = (rng.random((N, n_p)) < p_extra).astype(float)
    still = B * extra

    diag = {
        "lambda": lam, "gamma": gamma, "kappa": kappa,
        "mean_wait_days": float(wait[F > 0.5].mean()) if np.any(F > 0.5) else 0.0,
        "mean_rho": float(rho_diag.mean()),
        "mean_still_elig": float(still.mean()), "mean_B": float(B.mean()),
    }
    return still, diag


def compute_still_eligible(inst, Y, B, tiers, *, kappa, gamma=1.0,
                           clearing=DEFAULT_CLEARING, seed=0):
    """
    StillEligible from the DECISION-INDEPENDENT nominal congestion proxy (nearest
    facility, s_max denominator). Used for the nesting path; at kappa=0 it returns
    B exactly for any input B (including the hand-crafted toy scenarios).
    """
    return _still_from_assignment(
        inst, Y, B, _nominal_assignment(inst), np.asarray(inst.s_max, float),
        tiers, kappa=kappa, gamma=gamma, clearing=clearing, seed=seed)


def still_eligible_for_design(inst, Y, B, x, C, tiers, *, kappa, gamma=1.0,
                              clearing=DEFAULT_CLEARING, seed=0):
    """
    StillEligible under the congestion of a SPECIFIC design: patients routed by
    the design's assignment x, congestion measured against the design's
    contracted capacity C (rho_m = failures routed to m / C_m). This is the same
    congestion model the out-of-sample simulator uses, so it is the quantity the
    fixed-point design solve drives to self-consistency.
    """
    assign = np.asarray(x, float).argmax(axis=1)
    return _still_from_assignment(
        inst, Y, B, assign, np.asarray(C, float),
        tiers, kappa=kappa, gamma=gamma, clearing=clearing, seed=seed)


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
    delay_surrogate: dict | None = None,
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
    n_base = n_1st + N * n_per_s
    # Optional convex delay-mortality surrogate: one aux var t_m per facility
    # (>= the piecewise-linear expected delay-driven cancellation cost as a
    # function of C_m), added to the objective. Lets the endogenous design buy
    # spare capacity where it is cost-justified. None => exactly v1 objective.
    n_aux = n_f if delay_surrogate else 0
    n_total = n_base + n_aux

    def vt(m):           return n_base + m

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
    if delay_surrogate:
        for m in range(n_f):
            costs[vt(m)] = 1.0
            ubs[vt(m)] = 1e30

    h.addCols(n_total, costs, lbs, ubs, 0,
              np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64))

    int_t = hs.HighsVarType.kInteger
    fs_idx = np.arange(n_1st, dtype=np.int32)
    h.changeColsIntegrality(n_1st, fs_idx, np.array([int_t] * n_1st))

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
    # (15) MNF: sum_m z[m] <= mnf (non-binding at this scale)
    if inst.mnf is not None:
        _row(-INF, float(inst.mnf), [vz(m) for m in range(n_f)], [1.0] * n_f)

    # Delay-mortality surrogate: t_m >= convex PWL of C_m via tangent lower
    # bounds t_m >= g(c_k) + slope_k (C_m - c_k)  <=>  t_m - slope_k C_m >= g_k - slope_k c_k.
    if delay_surrogate:
        for m in range(n_f):
            curve = delay_surrogate.get(m)
            if not curve:
                continue
            Cs, gs = curve
            for k in range(len(Cs)):
                c_k, g_k = float(Cs[k]), float(gs[k])
                if k + 1 < len(Cs):
                    slope = (gs[k + 1] - gs[k]) / (Cs[k + 1] - Cs[k])
                else:
                    slope = (gs[k] - gs[k - 1]) / (Cs[k] - Cs[k - 1]) if len(Cs) > 1 else 0.0
                _row(g_k - slope * c_k, INF, [vt(m), vC(m)], [1.0, -slope])

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


# ---------------------------------------------------------------------------
# Fixed-point endogenous design (decision-dependent congestion)
# ---------------------------------------------------------------------------

def _design_fix(res):
    n_p, n_f = res["x"].shape
    return {"z": {m: int(res["z"][m]) for m in range(n_f)},
            "C": {m: int(res["C"][m]) for m in range(n_f)},
            "x": {(i, m): int(res["x"][i, m]) for i in range(n_p) for m in range(n_f)}}


def _lower_convex_hull(xs, ys):
    """Lower convex hull of (x, y) points with x strictly increasing."""
    hull = []
    for x, y in zip(xs, ys):
        while len(hull) >= 2:
            (x0, y0), (x1, y1) = hull[-2], hull[-1]
            # keep only if the new point makes a convex (upward) turn
            if (y1 - y0) * (x - x1) >= (y - y1) * (x1 - x0):
                hull.pop()
            else:
                break
        hull.append((x, y))
    return [p[0] for p in hull], [p[1] for p in hull]


def _delay_surrogate_curves(inst, assign, tiers, kappa, gamma, clearing):
    """
    Convex piecewise-linear delay-mortality cost g_m(C) per facility: the
    EXPECTED delay-induced EXTRA cancellation cost (beyond the v1 baseline) for
    the patients routed to m, as a function of the capacity C the design
    contracts there. More capacity -> lower utilization a_m/C -> shorter
    re-manufacture delay -> higher post-delay eligibility -> fewer forced
    cancellations, so g_m is decreasing (and convexified for valid tangents).

    g_m(C) = sum_{i routed to m} (1 - p_m) * beta_i * (1 - extra_survival(delay(a_m/C)))
                                * rho_cancel_i

    Using baseline eligibility B in the SAA and this surrogate for the delay
    channel avoids double-counting; at kappa=0 extra_survival=1 so g_m = 0.
    """
    lam = calibrate_lambda(gamma)
    n_p, n_f = inst.n_patients, inst.n_facilities
    assign = np.asarray(assign)
    beta = np.asarray(inst.beta, float)
    rho_cancel = np.asarray(inst.rho_cancel, float)
    curves = {}
    for m in range(n_f):
        pat = [i for i in range(n_p) if assign[i] == m]
        a_m = len(pat)
        if a_m == 0:
            continue
        Cs = list(range(a_m, int(inst.s_max[m]) + 1))
        if len(Cs) < 2:
            continue
        gs = []
        for C in Cs:
            delay = float(clearing.remake_delay(a_m / C))
            g = 0.0
            for i in pat:
                es = float(extra_survival(delay, HR_TIER[tiers[i]], lam, gamma, kappa))
                g += (1.0 - inst.p[m]) * beta[i] * (1.0 - es) * rho_cancel[i]
            gs.append(g)
        curves[m] = _lower_convex_hull(Cs, gs)
    return curves


def _real_cost(inst, Y, B, tiers, cand, kappa, gamma, clearing, seed, mip_gap, time_limit):
    """In-sample cost of a fixed design under its OWN delay-degraded eligibility."""
    still, _ = still_eligible_for_design(inst, Y, B, cand["x"], cand["C"], tiers,
                                         kappa=kappa, gamma=gamma, clearing=clearing, seed=seed)
    ev = solve_dynamic_sp(inst, Y, B, tiers=tiers, still_eligible=still,
                          fix_first_stage=_design_fix(cand), mip_gap=mip_gap,
                          time_limit=time_limit)
    return ev["total_cost"]


def solve_dynamic_sp_endogenous(
    inst: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    tiers,
    *,
    kappa: float,
    gamma: float = 1.0,
    clearing: ClearingFunction = DEFAULT_CLEARING,
    seed: int = 0,
    mip_gap: float = 1e-3,
    time_limit: float = 150.0,
    max_iter: int = 5,
) -> dict:
    """
    Endogenous design D_endo. Because a design's congestion (and hence the delay
    that degrades eligibility) depends on the capacity it contracts, we let the
    optimizer internalize that link through a convex delay-mortality surrogate on
    C_m (see _delay_surrogate_curves): the SAA uses the baseline eligibility B
    (v1 recourse) plus an added expected delay-induced cancellation cost that
    DECREASES with capacity, so the design buys spare capacity / spreads load
    exactly where it is cost-justified.

    The surrogate coefficients depend on the assignment (which patients sit at
    each facility), so we iterate: build surrogate from the current assignment,
    re-solve, repeat until the assignment stabilizes. Among all designs visited
    (including the exogenous optimum) we return the one with the lowest in-sample
    cost under its OWN true delay-degraded eligibility — so D_endo is, in sample,
    never worse than D_exo. kappa=0 makes the surrogate identically zero, so
    D_endo == D_exo == v1.
    """
    exo = solve_dynamic_sp(inst, Y, B, tiers=tiers, kappa=0.0, gamma=gamma,
                           clearing=clearing, seed=seed, mip_gap=mip_gap,
                           time_limit=time_limit)
    if kappa == 0.0:
        exo["endo_iters"] = 0
        exo["self_consistent_cost"] = exo["total_cost"]
        return exo

    assign = np.asarray(exo["x"], float).argmax(axis=1)
    visited = [exo]
    iters = 0
    for _ in range(max_iter):
        iters += 1
        curves = _delay_surrogate_curves(inst, assign, tiers, kappa, gamma, clearing)
        res = solve_dynamic_sp(inst, Y, B, tiers=tiers, still_eligible=B,
                               delay_surrogate=curves, mip_gap=mip_gap, time_limit=time_limit)
        visited.append(res)
        new_assign = np.asarray(res["x"], float).argmax(axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign

    # Return the design with the lowest real (true-eligibility) in-sample cost.
    best, best_cost = None, float("inf")
    for cand in visited:
        rc = _real_cost(inst, Y, B, tiers, cand, kappa, gamma, clearing, seed, mip_gap, time_limit)
        if rc < best_cost - 1e-9:
            best_cost, best = rc, cand
    best = dict(best)
    best["endo_iters"] = iters
    best["self_consistent_cost"] = best_cost
    return best
