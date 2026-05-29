"""
results/compute_per_tier_cancel.py
===================================
Re-simulate the deterministic-plan evaluation to extract per-tier
cancellation rates for all three urgency tiers (H, M, L).

The main case study already saves tier-H EEV and all-tier RP rates, but
not tier-M/L EEV rates, because the full r_cancel array was not serialised.
This script fills that gap by re-running only the second-stage solve with
the deterministic first-stage fixed, using the identical scenarios (seed=0,
N=200) as the original case study.

Outputs
-------
results/per_tier_cancel_rates.json
  {
    "deterministic_plan": {"H": ..., "M": ..., "L": ...},
    "stochastic_plan":    {"H": ..., "M": ..., "L": ...},
    "methodology": "Simulated from r_cancel arrays across 200 scenarios, seed=0"
  }

Run: python results/compute_per_tier_cancel.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from case_study import (
    TIER_IDX,
    build_case_study_instance,
    _highs_solve_sp,
    _solve_ev,
)
from yield_sp_v1 import sample_scenarios


N_SCENARIOS = 200
SEED        = 0


def main() -> None:
    t0 = time.perf_counter()

    print("Building calibrated instance …", flush=True)
    inst = build_case_study_instance()

    print("Sampling 200 scenarios (seed=0) …", flush=True)
    Y, B = sample_scenarios(inst, N_SCENARIOS, seed=SEED)

    # ------------------------------------------------------------------
    # Deterministic plan: solve EV, then fix first stage and re-run SP
    # ------------------------------------------------------------------
    print("Solving EV (deterministic) …", flush=True)
    ev = _solve_ev(inst)
    ev_fix = {
        "z": {m: int(ev["z"][m]) for m in inst.M},
        "C": {m: int(ev["C"][m]) for m in inst.M},
        "x": {(i, m): int(ev["x"][i, m]) for i in inst.I for m in inst.M},
    }

    print("Evaluating deterministic plan stochastically (EEV) …", flush=True)
    eev_r = _highs_solve_sp(
        inst, Y, B,
        fix_first_stage=ev_fix,
        relax_first_stage_constraints=True,
        mip_gap=1e-3,
        time_limit=1800.0,
    )
    # r_cancel shape: (N, n_patients)
    rc_det = eev_r["r_cancel"]

    # ------------------------------------------------------------------
    # Stochastic plan (RP)
    # ------------------------------------------------------------------
    print("Solving stochastic plan (RP) …", flush=True)
    rp_r = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)
    rc_sp = rp_r["r_cancel"]

    # ------------------------------------------------------------------
    # Per-tier rates
    # ------------------------------------------------------------------
    def _tier_rates(rc: np.ndarray) -> dict[str, float]:
        """
        cancel_rate(u) = Σ_ω Σ_{i in tier_u} r_cancel[ω,i]
                         ──────────────────────────────────────
                         N_scenarios × |tier_u|
        """
        return {
            t: float(rc[:, TIER_IDX[t]].sum() / (N_SCENARIOS * len(TIER_IDX[t])))
            for t in ("H", "M", "L")
        }

    det_rates = _tier_rates(rc_det)
    sp_rates  = _tier_rates(rc_sp)

    elapsed = time.perf_counter() - t0

    print()
    print("=== Per-tier cancellation rates ===")
    print(f"{'Tier':>6} | {'Det plan (%)':>14} | {'SP plan (%)':>12}")
    print("-" * 40)
    for t in ("H", "M", "L"):
        print(f"{'Tier ' + t:>6} | "
              f"{det_rates[t]*100:>14.4f} | "
              f"{sp_rates[t]*100:>12.4f}")

    # Sanity check: tier-H must match the previously published values
    det_h_pct = det_rates["H"] * 100.0
    sp_h_pct  = sp_rates["H"]  * 100.0
    if abs(det_h_pct - 5.40) > 0.20:
        print(f"\nWARNING: Tier-H DET rate changed: {det_h_pct:.4f}% (expected 5.40%)")
    if abs(sp_h_pct - 2.15) > 0.20:
        print(f"\nWARNING: Tier-H SP rate changed: {sp_h_pct:.4f}% (expected 2.15%)")

    out = {
        "deterministic_plan": {t: round(det_rates[t] * 100.0, 4) for t in ("H", "M", "L")},
        "stochastic_plan":    {t: round(sp_rates[t]  * 100.0, 4) for t in ("H", "M", "L")},
        "methodology": (
            "Simulated from r_cancel arrays across 200 scenarios with seed 0. "
            "Deterministic plan: EV first-stage fixed, second-stage optimised. "
            "Stochastic plan: full two-stage SP."
        ),
        "elapsed_s": round(elapsed, 1),
    }

    out_path = Path(__file__).parent / "per_tier_cancel_rates.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
