"""
expected_cost_baseline.py
==========================
Implements the expected-cost-deterministic (ECD) baseline: a smarter
deterministic plan that penalises each facility's per-batch cost by the
expected recourse cost of a failure, before solving the Stage-1 problem.

The key modification
---------------------
For each facility m:

    expected_recourse_per_batch = (
        beta_avg * cheapest_recollection_cost
        + (1 - beta_avg) * rho_cancel_avg
    )
    c_m_effective = c_m + (1 - p_m) * expected_recourse_per_batch

where
  beta_avg               — patient-population-weighted mean re-collection
                           eligibility across all 50 patients and 3 tiers
  cheapest_recollection  — rho_leuk + min(mean(rho_remfg), mean(rho_sub_offdiag))
                           (the optimizer would pick the cheaper of in-house
                           re-manufacture vs subcontracting)
  rho_cancel_avg         — patient-population-weighted mean cancellation
                           penalty across all 50 patients and 3 tiers

Everything else (opening costs f, capacity costs pi, s_max, beta, rho_cancel,
rho_sub, scenarios) is kept identical to the main case study.

Run: python expected_cost_baseline.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from case_study import (
    TIER_IDX,
    build_case_study_instance,
    _highs_solve_sp,
    _solve_ev,
)
from yield_sp_v1 import Instance, sample_scenarios

N_SCENARIOS = 200
SEED        = 0
_REPO       = Path(__file__).parent


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _compute_effective_costs(inst: Instance) -> tuple[np.ndarray, dict]:
    """
    Return (c_effective, calibration_dict).

    c_effective[m] = c[m] + (1 - p[m]) * expected_recourse_per_batch

    The calibration dict is logged verbatim to the result JSON for
    reproducibility.
    """
    n    = inst.n_patients
    n_f  = inst.n_facilities

    # Population-weighted averages (over all 50 patients, uniform scenario weight)
    beta_avg        = float(inst.beta.mean())
    rho_cancel_avg  = float(inst.rho_cancel.mean())
    rho_remfg_avg   = float(inst.rho_remfg.mean())

    # Off-diagonal subcontracting costs (src m → dst mp, m ≠ mp)
    sub_off = [inst.rho_sub[m, mp]
               for m in range(n_f) for mp in range(n_f) if mp != m]
    rho_sub_avg = float(np.mean(sub_off))

    # Cheapest recollection action: in-house or subcontract
    cheapest_recollection = float(inst.rho_leuk + min(rho_remfg_avg, rho_sub_avg))

    # Expected recourse cost triggered by a single batch failure
    expected_recourse = float(
        beta_avg * cheapest_recollection
        + (1.0 - beta_avg) * rho_cancel_avg
    )

    c_eff = inst.c + (1.0 - inst.p) * expected_recourse

    calib = {
        "beta_avg":                    round(beta_avg, 6),
        "rho_cancel_avg":              round(rho_cancel_avg, 6),
        "rho_remfg_avg":               round(rho_remfg_avg, 6),
        "rho_sub_offdiag_avg":         round(rho_sub_avg, 6),
        "cheapest_recollection_cost":  round(cheapest_recollection, 6),
        "expected_recourse_per_batch": round(expected_recourse, 6),
        "c_m_effective":               [round(float(v), 6) for v in c_eff],
        "c_m_original":                inst.c.tolist(),
        "p_m":                         inst.p.tolist(),
    }
    return c_eff, calib


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall0 = time.perf_counter()

    print("=== Expected-cost-deterministic baseline ===")
    print()

    # Build central instance and sample scenarios (same as case study)
    inst_orig = build_case_study_instance()
    print(f"Instance: {inst_orig.n_patients} patients, "
          f"{inst_orig.n_facilities} facilities")

    Y, B = sample_scenarios(inst_orig, N_SCENARIOS, seed=SEED)
    print(f"Scenarios: N={N_SCENARIOS}, seed={SEED}  "
          f"(Y.shape={Y.shape})")

    # ------------------------------------------------------------------
    # Compute effective per-batch costs
    # ------------------------------------------------------------------
    c_eff, calib = _compute_effective_costs(inst_orig)
    print()
    print("Cost adjustment:")
    print(f"  expected_recourse_per_batch = {calib['expected_recourse_per_batch']:.4f} M")
    print(f"  c_original = {calib['c_m_original']}")
    print(f"  c_effective = {calib['c_m_effective']}")

    # ------------------------------------------------------------------
    # Build a modified instance that substitutes c_eff as the per-batch cost.
    # The EV solver minimises Σ_i Σ_m c[m]*x[i,m]; no other change.
    # ------------------------------------------------------------------
    inst_smart = Instance(
        n_patients=inst_orig.n_patients,
        n_facilities=inst_orig.n_facilities,
        f=inst_orig.f.copy(),
        pi=inst_orig.pi.copy(),
        c=c_eff,                    # <-- only change vs original instance
        s_max=inst_orig.s_max.copy(),
        p=inst_orig.p.copy(),
        beta=inst_orig.beta.copy(),
        rho_leuk=inst_orig.rho_leuk,
        rho_remfg=inst_orig.rho_remfg.copy(),
        rho_sub=inst_orig.rho_sub.copy(),
        rho_cancel=inst_orig.rho_cancel.copy(),
    )

    # ------------------------------------------------------------------
    # Solve smart EV (Stage-1 only, deterministic)
    # ------------------------------------------------------------------
    print()
    print("Solving smart EV …", flush=True)
    ev_smart = _solve_ev(inst_smart)
    print(f"  Smart EV cost:      {ev_smart['ev_cost']:.4f} M")
    print(f"  Facilities opened:  "
          f"{[m for m in range(inst_smart.n_facilities) if ev_smart['z'][m] > 0.5]}")
    print(f"  Capacity:           {ev_smart['C'].tolist()}")

    # Compare allocation with naive EV
    with open(_REPO / "case_study_results.json") as fh:
        cs = json.load(fh)
    naive_z  = np.array(cs["ev"]["z"])
    naive_C  = np.array(cs["ev"]["C"])
    naive_x  = np.array(cs["ev"]["x"])
    smart_z  = ev_smart["z"]
    smart_C  = ev_smart["C"]
    smart_x  = ev_smart["x"]

    alloc_differs = bool(
        not np.array_equal(smart_z, naive_z) or
        not np.array_equal(smart_x, naive_x)
    )
    print(f"  Naive opens: {[m for m in range(4) if naive_z[m] > 0.5]}, "
          f"C={naive_C.tolist()}")
    print(f"  Smart opens: {[m for m in range(4) if smart_z[m] > 0.5]}, "
          f"C={smart_C.tolist()}")
    print(f"  First-stage allocation differs from naive: {alloc_differs}")

    # Patients per tier per facility under smart plan
    x_by_tier_fac = {}
    for tier, idx in TIER_IDX.items():
        x_by_tier_fac[tier] = smart_x[idx].sum(axis=0).tolist()
    print(f"  Smart allocation by tier x facility:")
    for t, cnts in x_by_tier_fac.items():
        print(f"    Tier {t}: {cnts}")

    # ------------------------------------------------------------------
    # Fix smart Stage-1 decisions and evaluate stochastically (EEV_smart)
    # ------------------------------------------------------------------
    ev_fix = {
        "z": {m: int(ev_smart["z"][m]) for m in inst_orig.M},
        "C": {m: int(ev_smart["C"][m]) for m in inst_orig.M},
        "x": {(i, m): int(ev_smart["x"][i, m])
               for i in inst_orig.I for m in inst_orig.M},
    }

    print()
    print("Evaluating smart plan stochastically (EEV_smart) …", flush=True)
    # Use ORIGINAL instance for evaluation (actual costs, not effective costs)
    eev_r = _highs_solve_sp(
        inst_orig, Y, B,
        fix_first_stage=ev_fix,
        relax_first_stage_constraints=True,
        mip_gap=1e-3,
        time_limit=1800.0,
    )

    # Stage-1 cost uses original c (not c_eff) — what the plan will actually cost
    stage1_cost = float(
        sum(inst_orig.f[m]  * int(ev_smart["z"][m]) for m in inst_orig.M)
        + sum(inst_orig.pi[m] * int(ev_smart["C"][m]) for m in inst_orig.M)
        + sum(inst_orig.c[m]  * int(ev_smart["x"][i, m])
              for i in inst_orig.I for m in inst_orig.M)
    )
    exp_s2_cost   = float(eev_r["expected_stage2_cost"])
    total_eev_smart = stage1_cost + exp_s2_cost

    print(f"  Stage-1 cost (actual):    {stage1_cost:.4f} M")
    print(f"  Expected Stage-2 cost:    {exp_s2_cost:.4f} M")
    print(f"  Total EEV_smart:          {total_eev_smart:.4f} M")

    # ------------------------------------------------------------------
    # Recourse mix
    # ------------------------------------------------------------------
    rem = float(eev_r["r_remfg"].sum())
    sub = float(eev_r["r_sub"].sum())
    can = float(eev_r["r_cancel"].sum())
    tot = rem + sub + can
    mix = {
        "remfg_pct":       round(100.0 * rem / tot, 4) if tot > 0 else 0.0,
        "subcontract_pct": round(100.0 * sub / tot, 4) if tot > 0 else 0.0,
        "cancel_pct":      round(100.0 * can / tot, 4) if tot > 0 else 0.0,
    }

    # ------------------------------------------------------------------
    # Per-tier cancellation rates (simulated)
    # ------------------------------------------------------------------
    rc = eev_r["r_cancel"]  # (N, n_patients)
    tier_cancel = {
        t: round(float(rc[:, TIER_IDX[t]].sum()) /
                 (N_SCENARIOS * len(TIER_IDX[t])) * 100.0, 4)
        for t in ("H", "M", "L")
    }

    # ------------------------------------------------------------------
    # VSS comparisons
    # ------------------------------------------------------------------
    naive_eev_cost = float(cs["eev"]["eev_cost"])
    rp_cost        = float(cs["rp"]["rp_cost"])

    vss_naive_pct = round(100.0 * (naive_eev_cost - rp_cost) / rp_cost, 4)
    vss_smart_pct = round(100.0 * (total_eev_smart - rp_cost) / rp_cost, 4)

    # ------------------------------------------------------------------
    # Sanity checks (print before deciding to proceed)
    # ------------------------------------------------------------------
    print()
    print("=== Sanity checks ===")
    ordering_ok = naive_eev_cost >= total_eev_smart >= rp_cost
    print(f"  naive_eev ({naive_eev_cost:.4f}) ≥ "
          f"smart_eev ({total_eev_smart:.4f}) ≥ "
          f"rp ({rp_cost:.4f}): {ordering_ok}")
    print(f"  VSS naive vs RP:  {vss_naive_pct:.2f}%  (expected ~14.6%)")
    print(f"  VSS smart vs RP:  {vss_smart_pct:.2f}%")
    if not ordering_ok:
        print("  [STOP] Cost ordering violated — do not commit.")
        return
    if vss_smart_pct < 4.0:
        print(f"  [FLAG] VSS_smart = {vss_smart_pct:.2f}% < 4% threshold — "
              "structural recourse contribution may be too small; "
              "review contribution framing before committing.")

    # ------------------------------------------------------------------
    # Print tier cancellation
    # ------------------------------------------------------------------
    print()
    print("Per-tier cancellation under smart ECD plan:")
    for tier, rate in tier_cancel.items():
        print(f"  Tier {tier}: {rate:.4f}%")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out = {
        "calibration_used": calib,
        "stage1_decisions": {
            "z": smart_z.tolist(),
            "C": smart_C.tolist(),
            "x_by_tier_and_facility": x_by_tier_fac,
        },
        "stage1_cost":                 round(stage1_cost, 6),
        "expected_stage2_cost":        round(exp_s2_cost, 6),
        "total_eev_smart":             round(total_eev_smart, 6),
        "facility_allocation_differs_from_naive": alloc_differs,
        "tier_cancel_rates_simulated": tier_cancel,
        "recourse_mix":                mix,
        "vss_vs_naive_pct":            vss_naive_pct,
        "vss_smart_vs_rp_pct":         vss_smart_pct,
        "naive_eev_cost":              round(naive_eev_cost, 6),
        "rp_cost":                     round(rp_cost, 6),
        "n_scenarios": N_SCENARIOS,
        "seed":        SEED,
    }

    out_path = _REPO / "results" / "expected_cost_baseline_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print()
    print(f"Results saved to {out_path}")
    print(f"Elapsed: {time.perf_counter() - t_wall0:.1f} s")


if __name__ == "__main__":
    main()
