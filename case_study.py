"""
case_study.py — CAR-T supply-chain case study: US geographic network.

Computes VSS (value of stochastic solution), capacity gap, recourse mix,
and tier-stratified cancellation rates by comparing:
  EV  — deterministic mean-yield (no-failure) plan
  EEV — EV plan evaluated under stochastic yield scenarios
  RP  — full recourse (stochastic) plan

Run: python case_study.py

Solver: HiGHS (open-source MILP, no license required, no size limit).

Case study calibration: facility labels correspond to four US biopharmaceutical
  hubs — Newark NJ, Boston MA, Raleigh-Durham NC, San Francisco CA — selected
  from the candidate-city network of Wan et al. (2026) (adapted from Avramescu
  et al., 2021b). Cost parameters (f_m facility opening cost, c_m per-batch
  primary manufacturing cost, π_m per-slot capacity contracting cost,
  s_m^max maximum capacity) are NOT taken from Wan's f_jk / c_jkt / Maxcap_j
  tables; they retain the Avramescu-/Bernardi-aligned calibration documented in
  `calibration_table.md`. Preliminary experiments substituting Wan's
  facility-specific cost values produced degenerate optimization with
  VSS-vs-ECD ≈ 0%, indicating that Wan's compressed cost spread does not
  satisfy the cost-yield trade-off conditions required for two-stage stochastic
  programming to add value at this scale; this finding is consistent with the
  narrow-spread regime characterized in the p_m sensitivity heatmap (Figure 16)
  and is reported in the sensitivity-analysis discussion of the manuscript.
  Patient yield p_m and tier-dependent eligibility β_u retain their clinical
  literature calibration (UK National CAR-T Panel, CARTITUDE-4, Bachy 2022,
  Locke 2022). Single planning period preserved; multi-period extension and
  57-hospital × 1,000-facility scale-up deferred to future work.

Uncertainty and capacity:
  Cryopreservation shelf life (24 h) exceeds all realized transport times
  (<=13 h) at US air-freight distances, so the shipping-feasibility constraint
  never binds and is omitted from the model. A max-open-facilities cap
  (MAX_FACILITIES_OPEN = 3) is retained but is non-binding — the optimizer opens
  2 of 4 on cost grounds (opening cost f_m + per-slot capacity cost pi_m). Per-
  facility capacity s_max = 75 (Wan et al. 2026 Maxcap); the tight, two-facility
  regime is reached by scaling demand (scaled_instance.mult), not by shrinking
  capacity.

Cost and clinical calibration
  f_m: annual CDMO access / slot-reservation fee (Avramescu 2023 supp.).
       Mode gradient consistent with Wan's k1 < k2 < k3 relative structure.
  c_m: per-batch primary mfg cost ~$150–200K (Avramescu 2023, Bernardi 2022).
  p_m: UK National CAR-T Panel (Dulobdas et al. 2025) + per-product OOS
       rates: Locke 2022 (axi-cel ~7%), Schuster 2019 (tisa-cel ~11%),
       Abramson 2020 (liso-cel ~6%), mapped to production mode.
  β_u: CARTITUDE-4 (San-Miguel et al. 2023), Bachy et al. (2022).
  ρ^cancel_u: cost-of-life-year × remaining life-expectancy under refractory
       hematologic malignancy; range $1–8M (health-economics literature).
"""

from __future__ import annotations

import json
import time

import numpy as np
import highspy as hs

from yield_sp_v1 import Instance, sample_scenarios


# ---------------------------------------------------------------------------
# Geographic network (Wan et al. 2026): 4 US facility hubs — Newark NJ (j67),
# Boston MA (j24), Raleigh-Durham NC (j43), San Francisco CA (j14) — serving 15
# highest-demand hospitals. Cryopreservation shelf life (24 h) exceeds all
# realized transport times (<=13 h) at these air-freight distances, so the
# shipping-feasibility constraint never binds and is omitted from the model
# (no t_trans / shelf-life data are carried by the instance).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FACILITIES_OPEN: int = 3     # max facilities open (MNF); non-binding at this scale

TIERS = ["H"] * 10 + ["M"] * 25 + ["L"] * 15  # 50 patients: 20% H / 50% M / 30% L

BETA_MAP      = {"H": 0.55, "M": 0.78, "L": 0.92}
# Cancellation penalty: cost-of-life-year × remaining life-expectancy under
# refractory hematologic malignancy.  Tier-H patients (aggressive disease,
# poor prognosis without CAR-T) carry the highest value-of-life estimate.
# Range $1–8M per patient is consistent with health-economics literature for
# curative oncology interventions; tier-H assigned $6M reflecting high
# counterfactual mortality without CAR-T.
RHOCANCEL_MAP = {"H": 6.00, "M": 2.00, "L": 0.75}

TIER_IDX = {
    "H": list(range(0, 10)),
    "M": list(range(10, 35)),
    "L": list(range(35, 50)),
}


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

def build_case_study_instance() -> Instance:
    """
    US geographic CAR-T network: 50 patients, 4 facilities.

    Facility / production mode / US location (Wan j-index):
      m0  Newark NJ           k1 manual          (Wan j67)
      m1  Boston MA           k2 semi-automated  (Wan j24)
      m2  Raleigh-Durham NC   k3 fully automated (Wan j43)
      m3  San Francisco CA    k2 semi-automated  (Wan j14)

    Geographic structure from Wan et al. (2026); cost parameters from
    clinical literature — see module docstring for full rationale.
    s_max = 75 slots from Wan's Maxcap_j sheet.
    """
    n_p = len(TIERS)
    n_f = 4

    beta       = np.array([BETA_MAP[t]      for t in TIERS])
    rho_cancel = np.array([RHOCANCEL_MAP[t] for t in TIERS])

    # Per-batch cost: lower with more automation (Avramescu 2023, Bernardi 2022)
    c = np.array([0.20, 0.18, 0.15, 0.18])

    # Subcontracting = partner facility cost + 15% premium
    rho_sub = np.zeros((n_f, n_f))
    for m in range(n_f):
        for mp in range(n_f):
            if m != mp:
                rho_sub[m, mp] = c[mp] * 1.15

    return Instance(
        n_patients=n_p,
        n_facilities=n_f,
        # Opening cost: annual CDMO slot-reservation fee (Avramescu 2023 supp.),
        # mode gradient consistent with Wan's k1 < k2 < k3 relative structure.
        f=np.array([0.5, 2.0, 3.0, 2.0]),
        pi=np.array([0.04, 0.06, 0.09, 0.06]),
        c=c,
        # Per-facility capacity s_max = 75 (Wan et al. 2026 Maxcap). The tight,
        # two-facility regime is reached by scaling demand (scaled_instance.mult),
        # not by shrinking capacity.
        s_max=np.array([75, 75, 75, 75]),
        # p_m: clinical calibration (UK National CAR-T Panel + per-product OOS)
        p=np.array([0.85, 0.92, 0.95, 0.92]),
        beta=beta,
        rho_leuk=0.005,
        rho_remfg=c.copy(),
        rho_sub=rho_sub,
        rho_cancel=rho_cancel,
        mnf=MAX_FACILITIES_OPEN,
    )


# ---------------------------------------------------------------------------
# HiGHS SP solver
# ---------------------------------------------------------------------------

def _highs_solve_sp(
    inst: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    *,
    fix_first_stage: dict | None = None,
    relax_first_stage_constraints: bool = False,
    mip_gap: float = 1e-3,
    time_limit: float = 1800.0,
) -> dict:
    """
    Build and solve the two-stage SP using HiGHS.

    Stage-2 recourse variables are continuous [0, 1].  The constraint matrix
    for fixed first-stage is totally unimodular in practice (all-integer RHS,
    network-flow structure), so the LP relaxation is integral at optimum.
    First-stage variables (z, C, x) remain binary/integer.
    """
    n_p, n_f = inst.n_patients, inst.n_facilities
    N        = Y.shape[0]

    # We build the deterministic-equivalent LP matrix directly for the HiGHS
    # backend, which addresses variables by integer column position rather than
    # by name. The counters below lay out the column order
    #   [ first-stage design | scenario-0 block | scenario-1 block | ... ]
    # and the vX() helpers map each named decision variable to its column:
    #   vz(m)=open z_m, vC(m)=capacity C_m, vx(i,m)=assign x_im,
    #   vr(o,i,m)=remake, vs(o,i,m,mp)=subcontract m->mp, vc(o,i)=cancel.
    # n_sub_pp = n_f*(n_f-1): off-diagonal (source!=dest) subcontract pairs.
    n_sub_pp = n_f * (n_f - 1)            # off-diagonal (src, dst) pairs per patient
    n_per_s  = n_p * n_f + n_p * n_sub_pp + n_p   # vars per scenario
    n_1st    = 2 * n_f + n_p * n_f        # first-stage var count
    n_total  = n_1st + N * n_per_s

    # Map (m, mp) with m≠mp → linear index within a patient's sub block
    def _soff(m, mp):
        return m * (n_f - 1) + (mp if mp < m else mp - 1)

    def vz(m):           return m
    def vC(m):           return n_f + m
    def vx(i, m):        return 2 * n_f + i * n_f + m
    def vr(o, i, m):     return n_1st + o * n_per_s + i * n_f + m
    def vs(o, i, m, mp): return n_1st + o * n_per_s + n_p * n_f + i * n_sub_pp + _soff(m, mp)
    def vc(o, i):        return n_1st + o * n_per_s + n_p * n_f + n_p * n_sub_pp + i

    # --- Build model ---
    h = hs.Highs()
    h.silent()
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("mip_rel_gap", mip_gap)

    # Variable bounds
    lbs = np.zeros(n_total)
    ubs = np.ones(n_total)
    for m in range(n_f):
        ubs[vC(m)] = float(inst.s_max[m])
    # Recourse stays [0,1] continuous — already correct

    # Objective
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

    # Add variables (columns) in batch
    h.addCols(
        n_total, costs, lbs, ubs,
        0, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64),
    )

    # Integer types for first-stage only
    int_t  = hs.HighsVarType.kInteger
    cont_t = hs.HighsVarType.kContinuous
    fs_idx = np.arange(n_1st, dtype=np.int32)
    fs_typ = np.array([int_t] * n_1st)
    h.changeColsIntegrality(n_1st, fs_idx, fs_typ)

    # Fix first-stage bounds when pinning EV solution
    if fix_first_stage:
        for m, v in fix_first_stage.get("z", {}).items():
            h.changeColBounds(vz(m), float(v), float(v))
        for m, v in fix_first_stage.get("C", {}).items():
            h.changeColBounds(vC(m), float(v), float(v))
        for (i, m), v in fix_first_stage.get("x", {}).items():
            h.changeColBounds(vx(i, m), float(v), float(v))

    # --- Constraints ---
    INF = 1e30

    def _row(lb, ub, inds, vals):
        ia = np.asarray(inds, dtype=np.int32)
        va = np.asarray(vals, dtype=np.float64)
        h.addRow(lb, ub, len(ia), ia, va)

    if not relax_first_stage_constraints:
        # (1) Σ_m x[i,m] = 1
        for i in range(n_p):
            _row(1.0, 1.0, [vx(i, m) for m in range(n_f)], [1.0] * n_f)
        # (2) x[i,m] ≤ z[m]
        for i in range(n_p):
            for m in range(n_f):
                _row(-INF, 0.0, [vx(i, m), vz(m)], [1.0, -1.0])
        # (3) Σ_i x[i,m] ≤ C[m]
        for m in range(n_f):
            _row(-INF, 0.0,
                 [vx(i, m) for i in range(n_p)] + [vC(m)],
                 [1.0] * n_p + [-1.0])
        # (4) C[m] ≤ s_max[m] · z[m]
        for m in range(n_f):
            _row(-INF, 0.0, [vC(m), vz(m)], [1.0, -float(inst.s_max[m])])
        # (15) MNF: Σ_m z[m] ≤ mnf (non-binding at this scale)
        if inst.mnf is not None:
            _row(-INF, float(inst.mnf), [vz(m) for m in range(n_f)], [1.0] * n_f)

    for o in range(N):
        for i in range(n_p):
            remfg_i = [vr(o, i, m) for m in range(n_f)]
            sub_i   = [vs(o, i, m, mp)
                       for m in range(n_f) for mp in range(n_f) if mp != m]
            x_i     = [vx(i, m) for m in range(n_f)]

            # (8) recourse completion:
            #   Σ r_remfg + Σ r_sub + r_cancel = Σ_m (1−Y[o,i,m])·x[i,m]
            _row(0.0, 0.0,
                 remfg_i + sub_i + [vc(o, i)] + x_i,
                 [1.0] * n_f + [1.0] * n_sub_pp + [1.0]
                 + [-(1.0 - Y[o, i, m]) for m in range(n_f)])

            # (10) re-collection eligibility
            _row(-INF, float(B[o, i]),
                 remfg_i + sub_i,
                 [1.0] * (n_f + n_sub_pp))

            # (10b) source tied to primary assignment
            for m in range(n_f):
                sub_from_m = [vs(o, i, m, mp) for mp in range(n_f) if mp != m]
                _row(-INF, 0.0,
                     [vr(o, i, m)] + sub_from_m + [vx(i, m)],
                     [1.0] * (1 + n_f - 1) + [-1.0])

        for m in range(n_f):
            # (9) capacity slack:
            #   Σ_i r_remfg[o,i,m] + Σ_{i,mp≠m} r_sub[o,i,mp,m]
            #   + Σ_i Y[o,i,m]·x[i,m] ≤ C[m]
            remfg_m = [vr(o, i, m) for i in range(n_p)]
            sub_to_m = [vs(o, i, mp, m)
                        for i in range(n_p) for mp in range(n_f) if mp != m]
            x_m     = [vx(i, m) for i in range(n_p)]
            _row(-INF, 0.0,
                 remfg_m + sub_to_m + x_m + [vC(m)],
                 [1.0] * n_p + [1.0] * (n_p * (n_f - 1))
                 + [float(Y[o, i, m]) for i in range(n_p)] + [-1.0])

    # --- Solve ---
    t0 = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t0

    status = h.getModelStatus()
    ok_statuses = {
        hs.HighsModelStatus.kOptimal,
        hs.HighsModelStatus.kObjectiveBound,
        hs.HighsModelStatus.kSolutionLimit,
        hs.HighsModelStatus.kTimeLimit,
    }
    if status not in ok_statuses:
        raise RuntimeError(f"HiGHS solve failed: {status}")
    if status == hs.HighsModelStatus.kTimeLimit:
        print(f"  WARNING: time limit hit after {solve_time:.1f}s")

    gap = h.getInfoValue("mip_gap")[1] if status != hs.HighsModelStatus.kOptimal else 0.0

    col = h.getSolution().col_value

    # Extract first-stage
    z_arr = np.array([round(col[vz(m)]) for m in range(n_f)])
    C_arr = np.array([round(col[vC(m)]) for m in range(n_f)])
    x_arr = np.array([[round(col[vx(i, m)]) for m in range(n_f)] for i in range(n_p)])

    # Extract recourse (round continuous to nearest integer)
    r_remfg  = np.zeros((N, n_p, n_f))
    r_sub    = np.zeros((N, n_p, n_f, n_f))
    r_cancel = np.zeros((N, n_p))
    for o in range(N):
        for i in range(n_p):
            for m in range(n_f):
                r_remfg[o, i, m] = round(col[vr(o, i, m)])
                for mp in range(n_f):
                    if mp != m:
                        r_sub[o, i, m, mp] = round(col[vs(o, i, m, mp)])
            r_cancel[o, i] = round(col[vc(o, i)])

    # Costs
    stage1 = (sum(inst.f[m]  * z_arr[m] for m in range(n_f))
              + sum(inst.pi[m] * C_arr[m] for m in range(n_f))
              + sum(inst.c[m]  * x_arr[i, m]
                    for i in range(n_p) for m in range(n_f)))

    s2 = np.zeros(N)
    for o in range(N):
        for i in range(n_p):
            for m in range(n_f):
                s2[o] += (inst.rho_leuk + inst.rho_remfg[m]) * r_remfg[o, i, m]
                for mp in range(n_f):
                    if mp != m:
                        s2[o] += (inst.rho_leuk + inst.rho_sub[m, mp]) * r_sub[o, i, m, mp]
            s2[o] += inst.rho_cancel[i] * r_cancel[o, i]

    exp_s2 = s2.mean()

    return {
        "total_cost":            stage1 + exp_s2,
        "stage1_cost":           stage1,
        "expected_stage2_cost":  exp_s2,
        "z": z_arr, "C": C_arr, "x": x_arr,
        "r_remfg": r_remfg, "r_sub": r_sub, "r_cancel": r_cancel,
        "solve_time": solve_time,
        "gap": gap,
    }


# ---------------------------------------------------------------------------
# EV (deterministic mean-yield)
# ---------------------------------------------------------------------------

def _solve_ev(inst: Instance) -> dict:
    """
    Solve the deterministic EV problem: no recourse, minimise Stage-1 cost
    only.  Equivalent to assuming every batch succeeds at its mean yield
    (i.e. the baseline every existing paper implicitly uses).
    """
    n_p, n_f = inst.n_patients, inst.n_facilities

    h = hs.Highs()
    h.silent()
    h.setOptionValue("mip_rel_gap", 1e-4)

    n_vars = 2 * n_f + n_p * n_f
    vz = lambda m: m
    vC = lambda m: n_f + m
    vx = lambda i, m: 2 * n_f + i * n_f + m

    costs = np.zeros(n_vars)
    lbs   = np.zeros(n_vars)
    ubs   = np.ones(n_vars)
    for m in range(n_f):
        costs[vz(m)] = inst.f[m]
        costs[vC(m)] = inst.pi[m]
        ubs[vC(m)]   = float(inst.s_max[m])
    for i in range(n_p):
        for m in range(n_f):
            costs[vx(i, m)] = inst.c[m]

    h.addCols(n_vars, costs, lbs, ubs,
              0, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64))

    int_t = hs.HighsVarType.kInteger
    h.changeColsIntegrality(n_vars,
                            np.arange(n_vars, dtype=np.int32),
                            np.array([int_t] * n_vars))

    INF = 1e30
    def _row(lb, ub, inds, vals):
        ia = np.asarray(inds, dtype=np.int32)
        va = np.asarray(vals, dtype=np.float64)
        h.addRow(lb, ub, len(ia), ia, va)

    for i in range(n_p):
        _row(1.0, 1.0, [vx(i, m) for m in range(n_f)], [1.0] * n_f)
    for i in range(n_p):
        for m in range(n_f):
            _row(-INF, 0.0, [vx(i, m), vz(m)], [1.0, -1.0])
    for m in range(n_f):
        _row(-INF, 0.0,
             [vx(i, m) for i in range(n_p)] + [vC(m)],
             [1.0] * n_p + [-1.0])
    for m in range(n_f):
        _row(-INF, 0.0, [vC(m), vz(m)], [1.0, -float(inst.s_max[m])])
    # (15) MNF: Σ_m z[m] ≤ mnf (non-binding at this scale)
    if inst.mnf is not None:
        _row(-INF, float(inst.mnf), [vz(m) for m in range(n_f)], [1.0] * n_f)

    t0 = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t0

    col = h.getSolution().col_value
    z_arr = np.array([round(col[vz(m)]) for m in range(n_f)])
    C_arr = np.array([round(col[vC(m)]) for m in range(n_f)])
    x_arr = np.array([[round(col[vx(i, m)]) for m in range(n_f)] for i in range(n_p)])

    ev_cost = h.getInfoValue("objective_function_value")[1]
    return {"ev_cost": ev_cost, "z": z_arr, "C": C_arr, "x": x_arr,
            "solve_time": solve_time}


# ---------------------------------------------------------------------------
# EV + EEV
# ---------------------------------------------------------------------------

def solve_ev_and_evaluate(inst, n_scenarios=200, seed=0):
    ev = _solve_ev(inst)

    fix = {
        "z": {m: int(ev["z"][m]) for m in inst.M},
        "C": {m: int(ev["C"][m]) for m in inst.M},
        "x": {(i, m): int(ev["x"][i, m]) for i in inst.I for m in inst.M},
    }

    Y, B = sample_scenarios(inst, n_scenarios, seed=seed)
    t0 = time.perf_counter()
    eev_r = _highs_solve_sp(inst, Y, B,
                             fix_first_stage=fix,
                             relax_first_stage_constraints=True,
                             mip_gap=1e-3, time_limit=1800.0)
    eev_time = time.perf_counter() - t0

    rc = eev_r["r_cancel"]
    rem = eev_r["r_remfg"].sum()
    sub = eev_r["r_sub"].sum()
    can = rc.sum()
    tot = rem + sub + can
    mix = {
        "re-manufacture": 100.0 * rem / tot if tot > 0 else 0.0,
        "subcontract":    100.0 * sub / tot if tot > 0 else 0.0,
        "cancel":         100.0 * can / tot if tot > 0 else 0.0,
    }

    return {
        "ev_cost":        ev["ev_cost"],
        "ev_z":           ev["z"],
        "ev_C":           ev["C"],
        "ev_x":           ev["x"],
        "ev_solve_time":  ev["solve_time"],
        "eev_cost":       ev["ev_cost"] + eev_r["expected_stage2_cost"],
        "eev_stage2":     eev_r["expected_stage2_cost"],
        "eev_time":       eev_time,
        "eev_gap":        eev_r["gap"],
        "recourse_mix":   mix,
        "tier_h_cancel_pct": rc[:, TIER_IDX["H"]].mean() * 100.0,
        "r_cancel":       rc,
    }


# ---------------------------------------------------------------------------
# RP
# ---------------------------------------------------------------------------

def solve_rp(inst, n_scenarios=200, seed=0):
    Y, B = sample_scenarios(inst, n_scenarios, seed=seed)
    t0 = time.perf_counter()
    r = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)
    rp_time = time.perf_counter() - t0

    if r["gap"] and r["gap"] > 0.05:
        print(f"  WARNING: RP gap {r['gap']*100:.1f}% — solution may be suboptimal")

    rc = r["r_cancel"]
    rem = r["r_remfg"].sum()
    sub = r["r_sub"].sum()
    can = rc.sum()
    tot = rem + sub + can
    mix = {
        "re-manufacture": 100.0 * rem / tot if tot > 0 else 0.0,
        "subcontract":    100.0 * sub / tot if tot > 0 else 0.0,
        "cancel":         100.0 * can / tot if tot > 0 else 0.0,
    }

    return {
        "rp_cost":          r["total_cost"],
        "rp_stage1":        r["stage1_cost"],
        "rp_stage2":        r["expected_stage2_cost"],
        "rp_z":             r["z"],
        "rp_C":             r["C"],
        "rp_x":             r["x"],
        "rp_time":          rp_time,
        "rp_gap":           r["gap"],
        "recourse_mix":     mix,
        "tier_cancel_pct":  {t: rc[:, TIER_IDX[t]].mean() * 100.0 for t in ("H", "M", "L")},
        "r_cancel":         rc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_SCENARIOS = 200
    SEED        = 0

    print("Building calibrated instance …", flush=True)
    inst = build_case_study_instance()

    print("Solving EV (deterministic) …", flush=True)
    ev_result = solve_ev_and_evaluate(inst, n_scenarios=N_SCENARIOS, seed=SEED)

    print("Solving RP (stochastic SP, free Stage-1) …", flush=True)
    rp_result = solve_rp(inst, n_scenarios=N_SCENARIOS, seed=SEED)

    # Headline numbers
    vss         = ev_result["eev_cost"] - rp_result["rp_cost"]
    vss_pct     = 100.0 * vss / rp_result["rp_cost"]
    ev_cap      = int(ev_result["ev_C"].sum())
    rp_cap      = int(rp_result["rp_C"].sum())
    cap_gap     = rp_cap - ev_cap
    cap_gap_pct = 100.0 * cap_gap / ev_cap if ev_cap > 0 else float("nan")
    h_ev        = ev_result["tier_h_cancel_pct"]
    h_rp        = rp_result["tier_cancel_pct"]["H"]

    ev_open = [f"m{m}" for m in inst.M if ev_result["ev_z"][m] > 0.5]
    rp_open = [f"m{m}" for m in inst.M if rp_result["rp_z"][m] > 0.5]

    def _asgn(x_arr):
        rows = []
        for m in inst.M:
            pts = [i for i in inst.I if x_arr[i, m] > 0.5]
            if pts:
                rows.append(f"m{m}: {len(pts)} patients")
        return ", ".join(rows)

    print()
    print("=== Case study results ===")
    print()
    print(f"Instance: {inst.n_patients} patients "
          f"({len(TIER_IDX['H'])} H / {len(TIER_IDX['M'])} M / "
          f"{len(TIER_IDX['L'])} L), "
          f"{inst.n_facilities} facilities, {N_SCENARIOS} scenarios")
    print()
    print("--- EV (mean-yield deterministic) solution ---")
    print(f"Stage-1 cost:           {ev_result['ev_cost']:.4f} M USD")
    print(f"Facilities opened:      {ev_open}")
    print(f"Total capacity:         {ev_cap} slots")
    print(f"Patients assigned:      {_asgn(ev_result['ev_x'])}")
    print()
    print("--- EEV (EV plan evaluated stochastically) ---")
    print(f"EEV cost:               {ev_result['eev_cost']:.4f} M USD")
    print(f"  (Stage-1: {ev_result['ev_cost']:.4f}  +  "
          f"Expected Stage-2: {ev_result['eev_stage2']:.4f})")
    print("Realized recourse mix:")
    for action, pct in ev_result["recourse_mix"].items():
        print(f"  {action + ':':20s} {pct:5.1f} %")
    print(f"Tier-H cancellation:    {ev_result['tier_h_cancel_pct']:.1f} %")
    print()
    print("--- RP (stochastic SP, free Stage-1) ---")
    print(f"RP cost:                {rp_result['rp_cost']:.4f} M USD")
    print(f"  (Stage-1: {rp_result['rp_stage1']:.4f}  +  "
          f"Expected Stage-2: {rp_result['rp_stage2']:.4f})")
    print(f"Facilities opened:      {rp_open}")
    print(f"Total capacity:         {rp_cap} slots  "
          f"({'+'if cap_gap>=0 else ''}{cap_gap} vs EV)")
    print(f"Patients assigned:      {_asgn(rp_result['rp_x'])}")
    print("Realized recourse mix:")
    for action, pct in rp_result["recourse_mix"].items():
        print(f"  {action + ':':20s} {pct:5.1f} %")
    for tier in ("H", "M", "L"):
        print(f"Tier-{tier} cancellation:    {rp_result['tier_cancel_pct'][tier]:.1f} %")
    print()
    print("--- Headline numbers ---")
    print(f"VSS (EEV − RP):         {vss:.4f} M USD  ({vss_pct:.1f} % of RP cost)")
    print(f"Capacity gap (SP − EV): {cap_gap} slots ({cap_gap_pct:.1f} %)")
    print(f"Tier-H violation reduction (EV → SP): {h_ev - h_rp:.1f} pp")
    if vss_pct < 1.0:
        print("  [FLAG] VSS < 1% — story may be weak")
    elif vss_pct > 15.0:
        print("  [FLAG] VSS > 15% — calibration may be too aggressive")
    print()
    print("--- Solve times ---")
    print(f"EV solve:               {ev_result['ev_solve_time']:.2f} s")
    print(f"EEV evaluation:         {ev_result['eev_time']:.2f} s"
          f"  (gap {ev_result['eev_gap']*100:.2f} %)")
    print(f"RP solve:               {rp_result['rp_time']:.2f} s"
          f"  (gap {rp_result['rp_gap']*100:.2f} %)")

    # Save JSON
    out = {
        "solver": "HiGHS",
        "instance": {
            "n_patients": inst.n_patients,
            "n_facilities": inst.n_facilities,
            "n_scenarios": N_SCENARIOS,
            "seed": SEED,
            "tiers": TIERS,
            "p":          inst.p.tolist(),
            "beta":       inst.beta.tolist(),
            "f":          inst.f.tolist(),
            "pi":         inst.pi.tolist(),
            "c":          inst.c.tolist(),
            "s_max":      inst.s_max.tolist(),
            "rho_leuk":   inst.rho_leuk,
            "rho_remfg":  inst.rho_remfg.tolist(),
            "rho_sub":    inst.rho_sub.tolist(),
            "rho_cancel": inst.rho_cancel.tolist(),
        },
        "ev": {
            "ev_cost":    ev_result["ev_cost"],
            "z":          ev_result["ev_z"].tolist(),
            "C":          ev_result["ev_C"].tolist(),
            "x":          ev_result["ev_x"].tolist(),
            "solve_time": ev_result["ev_solve_time"],
        },
        "eev": {
            "eev_cost":          ev_result["eev_cost"],
            "stage2_cost":       ev_result["eev_stage2"],
            "recourse_mix":      ev_result["recourse_mix"],
            "tier_h_cancel_pct": ev_result["tier_h_cancel_pct"],
            "solve_time":        ev_result["eev_time"],
            "gap":               ev_result["eev_gap"],
        },
        "rp": {
            "rp_cost":         rp_result["rp_cost"],
            "stage1_cost":     rp_result["rp_stage1"],
            "stage2_cost":     rp_result["rp_stage2"],
            "z":               rp_result["rp_z"].tolist(),
            "C":               rp_result["rp_C"].tolist(),
            "x":               rp_result["rp_x"].tolist(),
            "recourse_mix":    rp_result["recourse_mix"],
            "tier_cancel_pct": rp_result["tier_cancel_pct"],
            "solve_time":      rp_result["rp_time"],
            "gap":             rp_result["rp_gap"],
        },
        "headline": {
            "vss":                         vss,
            "vss_pct_of_rp":              vss_pct,
            "capacity_gap_slots":          cap_gap,
            "capacity_gap_pct":            cap_gap_pct,
            "tier_h_violation_reduction_pp": h_ev - h_rp,
        },
    }
    with open("case_study_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Results saved to case_study_results.json")


if __name__ == "__main__":
    main()
