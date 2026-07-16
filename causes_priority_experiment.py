"""
causes_priority_experiment.py — STEP 1 (why high-urgency patients are lost) and
STEP 2 (does treating the sickest first protect them when factories are full).

Everything is reported in plain healthcare terms. Uses the patient-level
simulator (patient_simulator.py) that tags each loss as (a) lost during the
normal wait on a successful batch, or (b) lost after a failure, and supports the
"sickest first" prioritization lever. Seeds fixed; HiGHS backend.

Run: python causes_priority_experiment.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from yield_sp_v1 import sample_scenarios
from scaled_instance import scaled_case_study_instance
from declineprob import calibrate_lambda
from dynamic_sp import solve_dynamic_sp, solve_dynamic_sp_endogenous
from patient_simulator import simulate_patients, BASE_WAIT_DEATH_6WK

_RES = Path(__file__).resolve().parent / "results"
_RES.mkdir(exist_ok=True)

CALIB_KAPPA = 1.0          # decline speed matched to real survival data (Dulobdas HR 1.64)
S_MAX_REAL = 75
SEED_TRAIN = 0
SEED_OOS = 100
N_TRAIN = 80
N_OOS = 500
MIP_GAP = 1e-2
TIME_LIMIT = 120.0
KAPPA_GRID = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]


def _draws(inst):
    Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED_TRAIN)
    Yte, Bte = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
    rng = np.random.default_rng(SEED_OOS + 777)
    U = rng.random((N_OOS, inst.n_patients))
    return Ytr, Btr, Yte, Bte, U


def _plan_row(inst, design, tiers, tidx, Yte, Bte, U, kappa, priority):
    s = simulate_patients(inst, design, tiers, tidx, Yte, Bte, U,
                          kappa=kappa, gamma=1.0, priority=priority)
    return {"high_urgency_lost": s["high_urgency_lost"],
            "cause_a_by_tier": s["cause_a_by_tier"], "cause_b_by_tier": s["cause_b_by_tier"],
            "lost_by_tier": s["lost_by_tier"], "treated_share_by_tier": s["treated_share_by_tier"],
            "total_cost": s["total_cost"], "cost_per_treated": s["cost_per_treated"],
            "spare_capacity_built": s["spare_capacity_built"],
            "facilities_open": s["facilities_open"], "open_facilities": s["open_facilities"]}


def sweep_setting(mult, s_max, label, kappas):
    """
    On-time plan (decline speed 0 design) simulated across decline speeds, WITHOUT
    and WITH sickest-first prioritization. Gives the cause split (STEP 1) and the
    prioritization effect (STEP 2) in one pass. At a full network the decline-aware
    design equals the on-time design, so plan 2 = plan 1 here.
    """
    inst, tiers, tidx = scaled_case_study_instance(mult=mult, s_max=s_max)
    Ytr, Btr, Yte, Bte, U = _draws(inst)
    D = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0, mip_gap=MIP_GAP, time_limit=TIME_LIMIT)
    rows = []
    for k in kappas:
        rows.append({"kappa": k,
                     "on_time": _plan_row(inst, D, tiers, tidx, Yte, Bte, U, k, False),
                     "sickest_first": _plan_row(inst, D, tiers, tidx, Yte, Bte, U, k, True)})
        r = rows[-1]
        print(f"    [{label}] speed={k:<4}: on-time H-lost={r['on_time']['high_urgency_lost']:.2f} "
              f"(a={r['on_time']['cause_a_by_tier']['H']:.2f} b={r['on_time']['cause_b_by_tier']['H']:.2f})"
              f" | sickest-first H-lost={r['sickest_first']['high_urgency_lost']:.2f}")
    return {"label": label, "n_patients": inst.n_patients, "s_max": s_max,
            "D_on_time": {"z": D["z"].tolist(), "C": D["C"].tolist()}, "rows": rows}


def three_plans_full_network(kappa):
    """
    The three-plan comparison in the busy 150-patient network at the real decline
    rate: on-time; decline-aware (spare capacity only); decline-aware + sickest-first.
    """
    inst, tiers, tidx = scaled_case_study_instance(mult=3, s_max=S_MAX_REAL)
    Ytr, Btr, Yte, Bte, U = _draws(inst)
    D_exo = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0, mip_gap=MIP_GAP, time_limit=TIME_LIMIT)
    D_endo = solve_dynamic_sp_endogenous(inst, Ytr, Btr, tiers, kappa=kappa, gamma=1.0,
                                         mip_gap=MIP_GAP, time_limit=TIME_LIMIT, max_iter=3)
    plans = {
        "on_time": _plan_row(inst, D_exo, tiers, tidx, Yte, Bte, U, kappa, False),
        "decline_aware": _plan_row(inst, D_endo, tiers, tidx, Yte, Bte, U, kappa, False),
        "sickest_first": _plan_row(inst, D_endo, tiers, tidx, Yte, Bte, U, kappa, True),
    }
    return {"kappa": kappa, "n_patients": inst.n_patients,
            "D_exo": {"z": D_exo["z"].tolist(), "C": D_exo["C"].tolist()},
            "D_endo": {"z": D_endo["z"].tolist(), "C": D_endo["C"].tolist()},
            "plans": plans}


def cap_sweep(kappa):
    """Capacity cap {40,55,75} at the real rate (100-patient network), on-time plan."""
    rows = []
    for smax in [40, 55, 75]:
        inst, tiers, tidx = scaled_case_study_instance(mult=2, s_max=smax)
        Ytr, Btr, Yte, Bte, U = _draws(inst)
        D = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0, mip_gap=MIP_GAP, time_limit=TIME_LIMIT)
        rows.append({"s_max": smax, "n_patients": inst.n_patients,
                     "on_time": _plan_row(inst, D, tiers, tidx, Yte, Bte, U, kappa, False),
                     "sickest_first": _plan_row(inst, D, tiers, tidx, Yte, Bte, U, kappa, True)})
        print(f"    s_max={smax}: on-time H-lost={rows[-1]['on_time']['high_urgency_lost']:.2f} "
              f"open={rows[-1]['on_time']['open_facilities']}")
    return {"kappa": kappa, "rows": rows}


def run(quick=False):
    t0 = time.perf_counter()
    kappas = [0.0, 0.25, 1.0, 2.0] if quick else KAPPA_GRID
    payload = {"calib_kappa": CALIB_KAPPA, "base_wait_death_6wk": BASE_WAIT_DEATH_6WK,
               "lambda_g1": calibrate_lambda(1.0), "n_train": N_TRAIN, "n_oos": N_OOS,
               "seed_train": SEED_TRAIN, "seed_oos": SEED_OOS}
    print("[busy 150-patient network]")
    payload["busy"] = sweep_setting(3, S_MAX_REAL, "busy-150", kappas)
    print("[low-demand 50-patient network]")
    payload["low"] = sweep_setting(1, S_MAX_REAL, "low-50", kappas)
    print("[three plans, full network, real decline rate]")
    payload["three_plans"] = three_plans_full_network(CALIB_KAPPA)
    print("[capacity cap sweep, real decline rate]")
    payload["cap_sweep"] = cap_sweep(CALIB_KAPPA)
    payload["compute_seconds"] = time.perf_counter() - t0
    out = _RES / "causes_priority_results.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out} ({payload['compute_seconds']:.0f}s)")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    payload = run(quick=args.quick)
    if not args.no_figures and not args.quick:
        from causes_priority_figures import make_all
        make_all(payload)
    print("Done.")
