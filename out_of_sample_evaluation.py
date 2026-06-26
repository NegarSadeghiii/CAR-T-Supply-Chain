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
        t_trans=inst.t_trans,
        shelf_life=inst.shelf_life,
        mnf=inst.mnf,
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
        "n_sub_pp": n_sub_pp,
        "n_ub": n_ub, "row_cap_start": row_cap_start,
        "vr": vr, "vs": vs, "vc": vc,
    }


def _solve_one_scenario(
    mats: dict,
    x_mat: np.ndarray,
    C_arr: np.ndarray,
    Y_w: np.ndarray,   # (n_p, n_f)
    B_w: np.ndarray,   # (n_p,)
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Stage-2 LP for one OOS scenario with fixed Stage-1 plan.
    Returns (stage2_cost, r_remfg[n_p,n_f], r_sub[n_p,n_f,n_f], r_cancel[n_p]).
    """
    n_p, n_f = mats["n_p"], mats["n_f"]
    m_assign  = x_mat.argmax(axis=1)   # assigned facility per patient (n_p,)

    # b_eq[i] = 1 − Y[i, m_i]  (failure indicator at primary assignment)
    b_eq = 1.0 - Y_w[np.arange(n_p), m_assign]

    # Build per-scenario b_ub
    b_ub = mats["b_ub_fixed"].copy()
    # Eligibility rows 0..n_p−1
    b_ub[:n_p] = B_w.astype(float)
    # Capacity rows: C[m] − (primary successes at m)
    rs = mats["row_cap_start"]
    for m in range(n_f):
        mask = (m_assign == m)
        b_ub[rs + m] = float(C_arr[m]) - float(Y_w[mask, m].sum())

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

    vr, vs, vc = mats["vr"], mats["vs"], mats["vc"]
    x = res.x
    r_cancel = np.array([round(x[vc(i)]) for i in range(n_p)], dtype=float)
    r_remfg  = np.array([[round(x[vr(i, m)]) for m in range(n_f)]
                          for i in range(n_p)], dtype=float)
    r_sub    = np.zeros((n_p, n_f, n_f), dtype=float)
    for i in range(n_p):
        for m in range(n_f):
            for mp in range(n_f):
                if mp != m:
                    r_sub[i, m, mp] = round(x[vs(i, m, mp)])
    return float(res.fun), r_remfg, r_sub, r_cancel


# ---------------------------------------------------------------------------
# Plan evaluation over all OOS scenarios
# ---------------------------------------------------------------------------

def _evaluate_plan(
    inst: Instance,
    plan: dict,
    Y_oos: np.ndarray,   # (N, n_p, n_f)
    B_oos: np.ndarray,   # (N, n_p)
    verbose: bool = True,
) -> tuple[dict, dict]:
    N   = Y_oos.shape[0]
    n_p = inst.n_patients
    n_f = inst.n_facilities
    x_mat = plan["x_mat"]
    C_arr = plan["C"]
    s1    = plan["stage1_cost"]

    mats = _build_stage2_matrices(inst, x_mat)

    per_s2_cost  = np.zeros(N)
    per_r_remfg  = np.zeros((N, n_p, n_f))
    per_r_sub    = np.zeros((N, n_p, n_f, n_f))
    per_r_cancel = np.zeros((N, n_p))

    t0 = time.perf_counter()
    for w in range(N):
        s2, r_remfg, r_sub, r_cancel = _solve_one_scenario(
            mats, x_mat, C_arr, Y_oos[w], B_oos[w])
        per_s2_cost[w]   = s2
        per_r_remfg[w]   = r_remfg
        per_r_sub[w]     = r_sub
        per_r_cancel[w]  = r_cancel
        if verbose and (w + 1) % 400 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {w+1}/{N} scenarios  "
                  f"({elapsed:.1f}s, {elapsed/(w+1)*1000:.1f}ms/scen)")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    Done: {elapsed:.1f}s total ({elapsed/N*1000:.1f}ms/scen)")

    # Sanity check: reconstruct stage-2 cost from recourse arrays
    rho_r = inst.rho_leuk + inst.rho_remfg            # (n_f,)
    rho_s = inst.rho_leuk + inst.rho_sub               # (n_f, n_f)
    s2_remfg   = (per_r_remfg  * rho_r[np.newaxis, np.newaxis, :]).sum(axis=(1, 2))
    s2_sub     = (per_r_sub    * rho_s[np.newaxis, np.newaxis, :, :]).sum(axis=(1, 2, 3))
    s2_cancel  = (per_r_cancel * inst.rho_cancel[np.newaxis, :]).sum(axis=1)
    s2_check   = s2_remfg + s2_sub + s2_cancel
    max_err    = float(np.max(np.abs(s2_check - per_s2_cost)))
    if max_err > 1e-6:
        raise RuntimeError(
            f"Recourse-array sanity check failed: max |err| = {max_err:.2e}"
        )
    if verbose:
        print(f"    Sanity check OK: max |reconstructed - LP| = {max_err:.2e}")

    per_total_cost = s1 + per_s2_cost

    # Tier cancel rates
    th_rates = per_r_cancel[:, list(TIER_IDX["H"])].sum(axis=1) / len(TIER_IDX["H"]) * 100
    tm_rates = per_r_cancel[:, list(TIER_IDX["M"])].sum(axis=1) / len(TIER_IDX["M"]) * 100
    tl_rates = per_r_cancel[:, list(TIER_IDX["L"])].sum(axis=1) / len(TIER_IDX["L"]) * 100
    sv_rates = (1.0 - per_r_cancel.sum(axis=1) / n_p) * 100

    summary = {
        "stage1_cost":                       round(float(s1), 6),
        "per_scenario_total_cost":           per_total_cost.tolist(),
        "per_scenario_tier_H_cancel_rate":   th_rates.tolist(),
        "per_scenario_tier_M_cancel_rate":   tm_rates.tolist(),
        "per_scenario_tier_L_cancel_rate":   tl_rates.tolist(),
        "per_scenario_service_level":        sv_rates.tolist(),
        # Per-scenario recourse arrays stored in results/out_of_sample_recourse.npz
        "_recourse_npz_key": None,   # filled in by caller after saving
    }
    arrays = {
        "r_remfg":  per_r_remfg,
        "r_sub":    per_r_sub,
        "r_cancel": per_r_cancel,
    }
    return summary, arrays


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
    recourse_arrays = {}
    for key, plan in plans.items():
        print(f"\nEvaluating: {plan['label']} …")
        summary, arrays = _evaluate_plan(inst, plan, Y_oos, B_oos, verbose=True)
        plan_results[key] = summary
        recourse_arrays[key] = arrays

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
        mean_c, ci_lo_mc, ci_hi_mc = _bootstrap_ci(tc, lambda a: a.mean())
        row = {
            "mean_total_cost":    round(float(tc.mean()), 6),
            "mean_cost_ci":       {"value":   round(float(mean_c), 6),
                                   "ci_low":  round(float(ci_lo_mc), 6),
                                   "ci_high": round(float(ci_hi_mc), 6)},
            "p05_total_cost":     round(float(np.percentile(tc, 5)), 6),
            "p95_total_cost":     round(float(np.percentile(tc, 95)), 6),
            "mean_tier_H_rate":   round(float(th.mean()), 4),
            "mean_service_level": round(float(sv.mean()), 4),
        }
        summary[key] = row
        label = plans[key]["label"]
        print(f"  {label}:")
        print(f"    Cost  mean={row['mean_total_cost']:.4f}  "
              f"90%CI [{ci_lo_mc:.4f}, {ci_hi_mc:.4f}]  "
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
    # Mean cost CI overlap check
    # ------------------------------------------------------------------
    print("\n=== Mean cost 90% CI overlap check ===")
    ci_sp    = summary["stochastic_plan"]["mean_cost_ci"]
    ci_ecd   = summary["expected_cost_deterministic"]["mean_cost_ci"]
    ci_naive = summary["naive_deterministic"]["mean_cost_ci"]
    overlaps_sp_ecd   = ci_sp["ci_high"]  >= ci_ecd["ci_low"]
    overlaps_sp_naive = ci_sp["ci_high"]  >= ci_naive["ci_low"]
    overlaps_ecd_naive = ci_ecd["ci_high"] >= ci_naive["ci_low"]
    for lbl, lo, hi in [
        ("SP   ", ci_sp["ci_low"],    ci_sp["ci_high"]),
        ("ECD  ", ci_ecd["ci_low"],   ci_ecd["ci_high"]),
        ("Naive", ci_naive["ci_low"], ci_naive["ci_high"]),
    ]:
        print(f"  {lbl}  90%CI [{lo:.4f}, {hi:.4f}]")
    print(f"  SP vs ECD overlap:   {'YES — CIs touch' if overlaps_sp_ecd  else 'NO  — clearly separated'}")
    print(f"  SP vs Naive overlap: {'YES — CIs touch' if overlaps_sp_naive else 'NO  — clearly separated'}")
    print(f"  ECD vs Naive overlap:{'YES — CIs touch' if overlaps_ecd_naive else 'NO  — clearly separated'}")

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
    # Attach first-stage decisions (needed for cost decomposition figure)
    # ------------------------------------------------------------------
    for key in plans:
        plan_results[key]["first_stage"] = {
            "z": plans[key]["z"].tolist(),
            "C": plans[key]["C"].tolist(),
            "x": plans[key]["x_mat"].tolist(),
        }

    # ------------------------------------------------------------------
    # Save per-scenario recourse arrays to compressed numpy file
    # (kept out of the JSON to stay under GitHub's 100 MB limit)
    # ------------------------------------------------------------------
    npz_path = _REPO_RESULTS / "out_of_sample_recourse.npz"
    npz_data = {"Y_oos": Y_oos}
    for key in plans:
        pfx = key  # e.g. "naive_deterministic"
        npz_data[f"{pfx}__r_remfg"]  = recourse_arrays[key]["r_remfg"]
        npz_data[f"{pfx}__r_sub"]    = recourse_arrays[key]["r_sub"]
        npz_data[f"{pfx}__r_cancel"] = recourse_arrays[key]["r_cancel"]
        plan_results[key]["_recourse_npz_key"] = pfx
    np.savez_compressed(str(npz_path), **npz_data)
    print(f"\nRecourse arrays saved: {npz_path}  "
          f"({npz_path.stat().st_size // (1024*1024)} MB compressed)")

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
