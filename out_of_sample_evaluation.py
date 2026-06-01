"""
out_of_sample_evaluation.py
============================
Out-of-sample Monte Carlo evaluation of three CAR-T supply-chain planning
strategies on 2,000 held-out scenarios (seed 4242, independent of the
seed-0 optimization set used in Phases 1-2).

Three frozen Stage-1 plans are evaluated:
  1. Naive deterministic  — EV plan, no yield uncertainty modelled
  2. Expected-cost det.   — EV plan with per-batch cost adjusted for expected
                            recourse (from expected_cost_baseline.py)
  3. Stochastic plan (SP) — full two-stage recourse plan (RP solve)

For each plan × scenario the Stage-2 recourse LP is solved independently:
  min  Σ (ρ_leuk + ρ_remfg[m]) r_remfg[i,m]
       + Σ (ρ_leuk + ρ_sub[m,m']) r_sub[i,m,m']
       + Σ ρ_cancel[i] r_cancel[i]
  s.t. completion, eligibility (β), source-tie (10b), capacity constraints

Service level definition:
  service_level[ω] = 1 - total_cancel_count[ω] / |I|
Rationale: a patient is "served" if their CAR-T product is delivered
(re-manufacture and subcontract are assumed to succeed after the delay);
cancellation is the binary failure outcome.

Run: python out_of_sample_evaluation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))

from case_study import (
    build_case_study_instance,
    TIER_IDX,
    _solve_ev,
)
from expected_cost_baseline import _compute_effective_costs
from yield_sp_v1 import Instance, sample_scenarios

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_OOS   = 2000
SEED_OOS = 4242   # seed 42 used in test_tier2.py → switch to 4242
SEED_BOOT = 100
N_BOOT  = 500

_TIER_H_THRESHOLD  = 5.0    # %
_SERVICE_THRESHOLD = 95.0   # %

_REPO_RESULTS = _REPO / "results"


# ---------------------------------------------------------------------------
# Stage-1 plan loader
# ---------------------------------------------------------------------------

def _load_plans(inst: Instance) -> dict[str, dict]:
    """
    Return the three frozen Stage-1 plans, each with keys:
      name, z, C, x_mat (n_p × n_f ndarray), stage1_cost
    """
    n_p, n_f = inst.n_patients, inst.n_facilities

    # --- 1. Naive deterministic ---
    with open(_REPO / "case_study_results.json") as fh:
        cs = json.load(fh)

    z_naive = np.array(cs["ev"]["z"])
    C_naive = np.array(cs["ev"]["C"])
    x_naive = np.array(cs["ev"]["x"])   # (n_p, n_f)

    s1_naive = float(
        sum(inst.f[m]  * z_naive[m] for m in range(n_f))
        + sum(inst.pi[m] * C_naive[m] for m in range(n_f))
        + sum(inst.c[m] * x_naive[:, m].sum() for m in range(n_f))
    )

    # --- 2. Expected-cost deterministic (re-run EV with effective costs) ---
    c_eff, _ = _compute_effective_costs(inst)
    inst_smart = Instance(
        n_patients=inst.n_patients,
        n_facilities=inst.n_facilities,
        f=inst.f.copy(), pi=inst.pi.copy(),
        c=c_eff,
        s_max=inst.s_max.copy(), p=inst.p.copy(),
        beta=inst.beta.copy(), rho_leuk=inst.rho_leuk,
        rho_remfg=inst.rho_remfg.copy(),
        rho_sub=inst.rho_sub.copy(),
        rho_cancel=inst.rho_cancel.copy(),
    )
    ev_smart = _solve_ev(inst_smart)
    z_ecd = ev_smart["z"]
    C_ecd = ev_smart["C"]
    x_ecd = ev_smart["x"]   # (n_p, n_f)

    # Stage-1 cost uses ORIGINAL per-batch cost c (not c_eff)
    s1_ecd = float(
        sum(inst.f[m]  * z_ecd[m] for m in range(n_f))
        + sum(inst.pi[m] * C_ecd[m] for m in range(n_f))
        + sum(inst.c[m] * x_ecd[:, m].sum() for m in range(n_f))
    )

    # --- 3. Stochastic plan ---
    z_sp = np.array(cs["rp"]["z"])
    C_sp = np.array(cs["rp"]["C"])
    x_sp = np.array(cs["rp"]["x"])   # (n_p, n_f)

    s1_sp = float(
        sum(inst.f[m]  * z_sp[m] for m in range(n_f))
        + sum(inst.pi[m] * C_sp[m] for m in range(n_f))
        + sum(inst.c[m] * x_sp[:, m].sum() for m in range(n_f))
    )

    return {
        "naive_deterministic": dict(
            label="Naive det.", z=z_naive, C=C_naive, x_mat=x_naive,
            stage1_cost=s1_naive),
        "expected_cost_deterministic": dict(
            label="Expected-cost det.", z=z_ecd, C=C_ecd, x_mat=x_ecd,
            stage1_cost=s1_ecd),
        "stochastic_plan": dict(
            label="Stochastic plan", z=z_sp, C=C_sp, x_mat=x_sp,
            stage1_cost=s1_sp),
    }


# ---------------------------------------------------------------------------
# Per-scenario Stage-2 LP infrastructure
# ---------------------------------------------------------------------------

def _build_stage2_matrices(inst: Instance, x_mat: np.ndarray) -> dict:
    """
    Precompute the structural parts of the per-scenario Stage-2 LP.

    Variable layout per patient i (n_vars_pp = n_f + n_sub_pp + 1 vars):
      vr(i, m)       = i*n_vars_pp + m                    re-manufacture at m
      vs(i, m, mp)   = i*n_vars_pp + n_f + _soff(m, mp)  subcontract m → mp
      vc(i)          = i*n_vars_pp + n_f + n_sub_pp       cancel

    The A_eq (completion) and A_ub (eligibility, source-tie, capacity)
    structural coefficient matrices are independent of Y/B once x is fixed.
    Only the RHS vectors b_eq and b_ub change per scenario.
    """
    n_p = inst.n_patients
    n_f = inst.n_facilities
    n_sub_pp  = n_f * (n_f - 1)
    n_vars_pp = n_f + n_sub_pp + 1
    n_vars    = n_p * n_vars_pp

    def _soff(m: int, mp: int) -> int:
        return m * (n_f - 1) + (mp if mp < m else mp - 1)

    def vr(i, m):      return i * n_vars_pp + m
    def vs(i, m, mp):  return i * n_vars_pp + n_f + _soff(m, mp)
    def vc(i):         return i * n_vars_pp + n_f + n_sub_pp

    # --- Objective (fixed per plan) ---
    c_obj = np.zeros(n_vars)
    for i in range(n_p):
        for m in range(n_f):
            c_obj[vr(i, m)] = inst.rho_leuk + inst.rho_remfg[m]
            for mp in range(n_f):
                if mp != m:
                    c_obj[vs(i, m, mp)] = inst.rho_leuk + inst.rho_sub[m, mp]
        c_obj[vc(i)] = inst.rho_cancel[i]

    # --- A_eq: completion  Σ vars = b_eq[i] ---
    req, ceq, veq = [], [], []
    for i in range(n_p):
        for m in range(n_f):
            req.append(i); ceq.append(vr(i, m)); veq.append(1.)
            for mp in range(n_f):
                if mp != m:
                    req.append(i); ceq.append(vs(i, m, mp)); veq.append(1.)
        req.append(i); ceq.append(vc(i)); veq.append(1.)
    A_eq = sp.csr_matrix((veq, (req, ceq)), shape=(n_p, n_vars))

    # --- A_ub: eligibility + source-tie (10b) + capacity ---
    rub, cub, vub = [], [], []
    row = 0

    # Eligibility: Σ_m r_remfg[i,m] + Σ_{m,m'} r_sub[i,m,m'] ≤ B[i]
    for i in range(n_p):
        for m in range(n_f):
            rub.append(row); cub.append(vr(i, m)); vub.append(1.)
            for mp in range(n_f):
                if mp != m:
                    rub.append(row); cub.append(vs(i, m, mp)); vub.append(1.)
        row += 1
    row_elig_end = row   # = n_p

    # Source-tie (10b): r_remfg[i,m] + Σ_{m'≠m} r_sub[i,m,m'] ≤ x[i,m]
    for i in range(n_p):
        for m in range(n_f):
            rub.append(row); cub.append(vr(i, m)); vub.append(1.)
            for mp in range(n_f):
                if mp != m:
                    rub.append(row); cub.append(vs(i, m, mp)); vub.append(1.)
            row += 1
    row_srctie_end = row  # = n_p + n_p*n_f

    # Capacity (9): Σ_i r_remfg[i,m] + Σ_{i,m'≠m} r_sub[i,m',m] ≤ C[m] - fixed_load
    row_cap_start = row
    for m in range(n_f):
        for i in range(n_p):
            rub.append(row); cub.append(vr(i, m)); vub.append(1.)
            for mp in range(n_f):
                if mp != m:
                    rub.append(row); cub.append(vs(i, mp, m)); vub.append(1.)
        row += 1

    n_ub = row
    A_ub = sp.csr_matrix((vub, (rub, cub)), shape=(n_ub, n_vars))

    # Fixed part of b_ub: source-tie RHS = x[i,m]
    b_ub_fixed = np.zeros(n_ub)
    for i in range(n_p):
        for m in range(n_f):
            b_ub_fixed[row_elig_end + i * n_f + m] = float(x_mat[i, m])

    return {
        "c_obj": c_obj,
        "A_eq": A_eq, "A_ub": A_ub,
        "b_ub_fixed": b_ub_fixed,
        "n_p": n_p, "n_f": n_f,
        "n_vars": n_vars, "n_vars_pp": n_vars_pp,
        "n_ub": n_ub, "row_cap_start": row_cap_start,
        "vc": vc,
    }


def _solve_one_scenario(
    mats: dict,
    x_mat: np.ndarray,
    C_arr: np.ndarray,
    Y_ω: np.ndarray,   # (n_p, n_f)
    B_ω: np.ndarray,   # (n_p,)
) -> tuple[float, np.ndarray]:
    """
    Solve Stage-2 LP for one OOS scenario with fixed Stage-1 plan.
    Returns (stage2_cost, r_cancel_vec[n_p]).
    """
    n_p, n_f = mats["n_p"], mats["n_f"]
    m_assign  = x_mat.argmax(axis=1)   # assigned facility per patient (n_p,)

    # b_eq[i] = 1 − Y[i, m_i]  (failure indicator at primary assignment)
    b_eq = 1.0 - Y_ω[np.arange(n_p), m_assign]

    # Build per-scenario b_ub
    b_ub = mats["b_ub_fixed"].copy()
    # Eligibility rows 0..n_p−1
    b_ub[:n_p] = B_ω.astype(float)
    # Capacity rows: C[m] − (primary successes at m)
    rs = mats["row_cap_start"]
    for m in range(n_f):
        mask = (m_assign == m)
        b_ub[rs + m] = float(C_arr[m]) - float(Y_ω[mask, m].sum())

    bounds = [(0.0, 1.0)] * mats["n_vars"]
    res = linprog(
        mats["c_obj"],
        A_ub=mats["A_ub"], b_ub=b_ub,
        A_eq=mats["A_eq"], b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
    )

    if res.status != 0:
        raise RuntimeError(
            f"Stage-2 LP failed (status {res.status}): {res.message}"
        )

    vc       = mats["vc"]
    r_cancel = np.array([round(res.x[vc(i)]) for i in range(n_p)], dtype=float)
    return float(res.fun), r_cancel


# ---------------------------------------------------------------------------
# Plan evaluation over all OOS scenarios
# ---------------------------------------------------------------------------

def _evaluate_plan(
    inst: Instance,
    plan: dict,
    Y_oos: np.ndarray,   # (N, n_p, n_f)
    B_oos: np.ndarray,   # (N, n_p)
    verbose: bool = True,
) -> dict:
    N   = Y_oos.shape[0]
    n_p = inst.n_patients
    x_mat = plan["x_mat"]
    C_arr = plan["C"]
    s1    = plan["stage1_cost"]

    mats = _build_stage2_matrices(inst, x_mat)

    per_s2_cost  = np.zeros(N)
    per_r_cancel = np.zeros((N, n_p))

    t0 = time.perf_counter()
    for ω in range(N):
        s2, r_cancel = _solve_one_scenario(mats, x_mat, C_arr, Y_oos[ω], B_oos[ω])
        per_s2_cost[ω]   = s2
        per_r_cancel[ω]  = r_cancel
        if verbose and (ω + 1) % 400 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {ω+1}/{N} scenarios  "
                  f"({elapsed:.1f}s, {elapsed/(ω+1)*1000:.1f}ms/scen)")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    Done: {elapsed:.1f}s total ({elapsed/N*1000:.1f}ms/scen)")

    per_total_cost = s1 + per_s2_cost

    # Tier cancel rates
    th_rates = per_r_cancel[:, TIER_IDX["H"]].sum(axis=1) / len(TIER_IDX["H"]) * 100
    tm_rates = per_r_cancel[:, TIER_IDX["M"]].sum(axis=1) / len(TIER_IDX["M"]) * 100
    tl_rates = per_r_cancel[:, TIER_IDX["L"]].sum(axis=1) / len(TIER_IDX["L"]) * 100
    sv_rates = (1.0 - per_r_cancel.sum(axis=1) / n_p) * 100

    return {
        "stage1_cost":                       round(float(s1), 6),
        "per_scenario_total_cost":           per_total_cost.tolist(),
        "per_scenario_tier_H_cancel_rate":   th_rates.tolist(),
        "per_scenario_tier_M_cancel_rate":   tm_rates.tolist(),
        "per_scenario_tier_L_cancel_rate":   tl_rates.tolist(),
        "per_scenario_service_level":        sv_rates.tolist(),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    arr: np.ndarray,
    target_fn,
    n_boot: int = N_BOOT,
    seed: int = SEED_BOOT,
) -> tuple[float, float, float]:
    """Returns (estimate, ci_low, ci_high) with 90% CI via percentile bootstrap."""
    rng    = np.random.default_rng(seed)
    N      = len(arr)
    point  = float(target_fn(arr))
    boots  = [float(target_fn(rng.choice(arr, size=N, replace=True)))
              for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [5, 95])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall0 = time.perf_counter()
    print("=== Out-of-sample Monte Carlo evaluation ===\n")

    # ------------------------------------------------------------------
    # Instance
    # ------------------------------------------------------------------
    inst = build_case_study_instance()
    n_p  = inst.n_patients
    n_f  = inst.n_facilities
    print(f"Instance: {n_p} patients, {n_f} facilities")

    # ------------------------------------------------------------------
    # Plans
    # ------------------------------------------------------------------
    print("Loading / reconstructing Stage-1 plans …")
    plans = _load_plans(inst)
    for name, p in plans.items():
        opens = [m for m in range(n_f) if p["z"][m] > 0.5]
        print(f"  {name}: opens={opens}  C={p['C'].tolist()}  "
              f"Stage-1={p['stage1_cost']:.4f} M")

    # ------------------------------------------------------------------
    # OOS scenarios
    # ------------------------------------------------------------------
    print(f"\nGenerating {N_OOS} OOS scenarios (seed {SEED_OOS}) …")
    Y_oos, B_oos = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
    print(f"  Y.shape={Y_oos.shape}  B.shape={B_oos.shape}")

    # ------------------------------------------------------------------
    # Evaluate each plan
    # ------------------------------------------------------------------
    plan_results = {}
    for key, plan in plans.items():
        print(f"\nEvaluating: {plan['label']} …")
        plan_results[key] = _evaluate_plan(inst, plan, Y_oos, B_oos, verbose=True)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    print("\n=== Summary stats ===")
    summary = {}
    for key in plans:
        pr = plan_results[key]
        tc = np.array(pr["per_scenario_total_cost"])
        th = np.array(pr["per_scenario_tier_H_cancel_rate"])
        sv = np.array(pr["per_scenario_service_level"])
        row = {
            "mean_total_cost":    round(float(tc.mean()), 6),
            "p05_total_cost":     round(float(np.percentile(tc, 5)), 6),
            "p95_total_cost":     round(float(np.percentile(tc, 95)), 6),
            "mean_tier_H_rate":   round(float(th.mean()), 4),
            "mean_service_level": round(float(sv.mean()), 4),
        }
        summary[key] = row
        label = plans[key]["label"]
        print(f"  {label}:")
        print(f"    Cost  mean={row['mean_total_cost']:.4f}  "
              f"P05={row['p05_total_cost']:.4f}  P95={row['p95_total_cost']:.4f}")
        print(f"    Tier-H mean={row['mean_tier_H_rate']:.4f}%  "
              f"SvcLv mean={row['mean_service_level']:.4f}%")

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    print("\n=== Sanity checks ===")
    naive_mean = summary["naive_deterministic"]["mean_total_cost"]
    ecd_mean   = summary["expected_cost_deterministic"]["mean_total_cost"]
    sp_mean    = summary["stochastic_plan"]["mean_total_cost"]
    th_naive   = summary["naive_deterministic"]["mean_tier_H_rate"]
    th_ecd     = summary["expected_cost_deterministic"]["mean_tier_H_rate"]
    th_sp      = summary["stochastic_plan"]["mean_tier_H_rate"]
    sv_naive   = summary["naive_deterministic"]["mean_service_level"]
    sv_ecd     = summary["expected_cost_deterministic"]["mean_service_level"]
    sv_sp      = summary["stochastic_plan"]["mean_service_level"]

    ok_cost = naive_mean >= ecd_mean >= sp_mean
    ok_th   = th_sp <= th_naive  # SP must have lowest tier-H
    ok_sv   = sv_sp >= sv_naive  # SP must have highest service level

    print(f"  Cost ordering (naive≥ecd≥sp):  "
          f"{naive_mean:.4f} ≥ {ecd_mean:.4f} ≥ {sp_mean:.4f}  {'OK' if ok_cost else 'FAIL'}")
    print(f"  Tier-H ordering (sp≤naive):    "
          f"SP={th_sp:.4f}% ≤ Naive={th_naive:.4f}%  {'OK' if ok_th else 'FAIL'}")
    print(f"  Service level (sp≥naive):      "
          f"SP={sv_sp:.4f}% ≥ Naive={sv_naive:.4f}%  {'OK' if ok_sv else 'FAIL'}")

    if not (ok_cost and ok_th and ok_sv):
        print("\n[STOP] Sanity check(s) failed. "
              "Do not commit — investigate OOS overfitting or model mismatch.")
        return

    print("  All sanity checks passed. OOS confirms in-sample finding.")

    # ------------------------------------------------------------------
    # Budget threshold
    # ------------------------------------------------------------------
    budget = round((naive_mean + sp_mean) / 2, 2)
    print(f"\nBudget threshold: {budget:.2f} M USD "
          f"(midpoint of naive {naive_mean:.4f} and SP {sp_mean:.4f})")

    # ------------------------------------------------------------------
    # Probability-of-target with bootstrap CIs
    # ------------------------------------------------------------------
    print("\n=== Probability-of-target (90% bootstrap CIs) ===")
    pot = {}
    for key in plans:
        pr     = plan_results[key]
        tc_arr = np.array(pr["per_scenario_total_cost"])
        th_arr = np.array(pr["per_scenario_tier_H_cancel_rate"])
        sv_arr = np.array(pr["per_scenario_service_level"])

        p_cost, ci_lo_c, ci_hi_c = _bootstrap_ci(
            tc_arr, lambda a: (a <= budget).mean())
        p_th,   ci_lo_t, ci_hi_t = _bootstrap_ci(
            th_arr, lambda a: (a <= _TIER_H_THRESHOLD).mean())
        p_sv,   ci_lo_s, ci_hi_s = _bootstrap_ci(
            sv_arr, lambda a: (a >= _SERVICE_THRESHOLD).mean())

        pot[key] = {
            "P_cost_below_budget":   {"value": round(p_cost, 4),
                                      "ci_low": round(ci_lo_c, 4),
                                      "ci_high": round(ci_hi_c, 4)},
            "P_tier_H_below_5pct":   {"value": round(p_th, 4),
                                      "ci_low": round(ci_lo_t, 4),
                                      "ci_high": round(ci_hi_t, 4)},
            "P_service_above_95pct": {"value": round(p_sv, 4),
                                      "ci_low": round(ci_lo_s, 4),
                                      "ci_high": round(ci_hi_s, 4)},
        }
        label = plans[key]["label"]
        print(f"  {label}:")
        print(f"    P(cost≤{budget})   = {p_cost:.4f}  "
              f"90%CI [{ci_lo_c:.4f}, {ci_hi_c:.4f}]")
        print(f"    P(TH≤5%)        = {p_th:.4f}  "
              f"90%CI [{ci_lo_t:.4f}, {ci_hi_t:.4f}]")
        print(f"    P(SvcLv≥95%)    = {p_sv:.4f}  "
              f"90%CI [{ci_lo_s:.4f}, {ci_hi_s:.4f}]")

    # ------------------------------------------------------------------
    # Assemble and save JSON
    # ------------------------------------------------------------------
    tier_sizes = {t: len(TIER_IDX[t]) for t in ("H", "M", "L")}
    out = {
        "calibration": {
            "n_patients":    n_p,
            "scenarios":     N_OOS,
            "seed_oos":      SEED_OOS,
            "seed_bootstrap": SEED_BOOT,
            "n_bootstrap":   N_BOOT,
            "tier_sizes":    tier_sizes,
        },
        "plans": plan_results,
        "summary_stats": summary,
        "probability_of_target": pot,
        "budget_threshold_M_USD":   budget,
        "tier_H_threshold_pct":     _TIER_H_THRESHOLD,
        "service_level_threshold_pct": _SERVICE_THRESHOLD,
        "elapsed_s": round(time.perf_counter() - t_wall0, 1),
    }

    out_path = _REPO_RESULTS / "out_of_sample_results.json"
    _REPO_RESULTS.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total elapsed: {time.perf_counter() - t_wall0:.1f}s")


if __name__ == "__main__":
    main()
