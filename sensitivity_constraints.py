"""
sensitivity_constraints.py — structural-constraint sensitivity for Yield_upgrade.

Tests whether tightening either of the two structural constraints added in the
Yield_upgrade branch becomes binding at a tighter, biologically/operationally
realistic value:

  - SHELF_LIFE_HOURS:  cell-viability shelf life (24h cryopreserved → 12h fresh)
  - MAX_FACILITIES_OPEN (MNF): maximum open-facility network footprint (3 → 2)

A 2×2 grid: {24h, 12h} × {MNF=3, MNF=2}. For each cell it reports:
  - number of infeasible (i, m) patient-facility pairs under the shelf life
  - SP facility selection
  - VSS-vs-naive deterministic
  - VSS-vs-expected-parameter deterministic (ECD)
  - Tier-H cancellation rate under the SP plan

This is an ADDITIVE sensitivity. It does not modify any production constant
permanently: each cell builds a fresh case-study instance and overrides only
the `shelf_life` / `mnf` Instance fields. All cost / yield / β parameters stay
at the case-study calibration. Solver: HiGHS, mip_rel_gap = 1e-3 (matching
case_study.py).

Run: python sensitivity_constraints.py
Output: results/sensitivity_constraints_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from case_study import (
    build_case_study_instance,
    _highs_solve_sp, _solve_ev,
)
from sensitivity_beta import (
    _ecd_effective_costs, _fix_dict, _tier_h_rate, _opens,
    N_SCENARIOS, SEED,
)
from yield_sp_v1 import Instance, sample_scenarios

_REPO = Path(__file__).parent
_OUT  = _REPO / "results" / "sensitivity_constraints_results.json"

# ── 2×2 sweep grid ──────────────────────────────────────────────────────────
SWEEP = [
    {"label": "baseline (24h, MNF=3)",   "shelf_life": 24.0, "mnf": 3},
    {"label": "fresh-cell (12h, MNF=3)", "shelf_life": 12.0, "mnf": 3},
    {"label": "tight-MNF (24h, MNF=2)",  "shelf_life": 24.0, "mnf": 2},
    {"label": "joint (12h, MNF=2)",      "shelf_life": 12.0, "mnf": 2},
]


def _count_infeasible(inst: Instance, shelf_life: float) -> tuple[int, int, int]:
    """Return (n_pairs_infeasible, n_pairs_total, n_patients_with_no_facility)."""
    n_p, n_f = inst.n_patients, inst.n_facilities
    n_pairs_total = n_p * n_f
    n_pairs_infeasible = sum(
        1 for i in range(n_p) for m in range(n_f)
        if inst.t_trans[i, m] > shelf_life
    )
    n_patients_infeasible = sum(
        1 for i in range(n_p)
        if all(inst.t_trans[i, m] > shelf_life for m in range(n_f))
    )
    return n_pairs_infeasible, n_pairs_total, n_patients_infeasible


def run_cell(point: dict) -> dict:
    sl, mnf = point["shelf_life"], point["mnf"]
    print(f"\n{'='*60}\n{point['label']}\n{'='*60}")

    # Build a fresh case-study instance and override only the two constraints.
    inst = build_case_study_instance()
    inst.shelf_life = sl
    inst.mnf = mnf

    n_pairs_inf, n_pairs_total, n_pat_inf = _count_infeasible(inst, sl)
    print(f"  Pairs infeasible: {n_pairs_inf}/{n_pairs_total}  "
          f"(patients with no feasible facility: {n_pat_inf})")

    if n_pat_inf > 0:
        print(f"  → INFEASIBLE: {n_pat_inf} patients have no feasible facility "
              f"under {sl}h shelf life")
        return {
            "label": point["label"],
            "shelf_life_hours": sl,
            "max_facilities_open": mnf,
            "status": "INFEASIBLE",
            "n_pair_infeasibilities": n_pairs_inf,
            "n_pair_total": n_pairs_total,
            "n_patients_with_no_feasible_facility": n_pat_inf,
        }

    Y, B = sample_scenarios(inst, N_SCENARIOS, seed=SEED)

    # (1) Naive EV → EEV
    print("  [1/3] Naive EV …", flush=True)
    ev_naive = _solve_ev(inst)
    fix_naive = _fix_dict(inst, ev_naive["z"], ev_naive["C"], ev_naive["x"])
    eev_naive_r = _highs_solve_sp(inst, Y, B, fix_first_stage=fix_naive,
                                  relax_first_stage_constraints=True,
                                  mip_gap=1e-3, time_limit=1800.0)
    eev_naive_cost = eev_naive_r["total_cost"]

    # (2) ECD EV → EEV
    print("  [2/3] ECD EV …", flush=True)
    c_eff, exp_recourse, c_eff_list = _ecd_effective_costs(inst)
    inst_ecd = Instance(
        n_patients=inst.n_patients, n_facilities=inst.n_facilities,
        f=inst.f.copy(), pi=inst.pi.copy(), c=c_eff,
        s_max=inst.s_max.copy(), p=inst.p.copy(),
        beta=inst.beta.copy(), rho_leuk=inst.rho_leuk,
        rho_remfg=inst.rho_remfg.copy(), rho_sub=inst.rho_sub.copy(),
        rho_cancel=inst.rho_cancel.copy(),
        t_trans=inst.t_trans, shelf_life=inst.shelf_life, mnf=inst.mnf,
    )
    ev_ecd = _solve_ev(inst_ecd)
    fix_ecd = _fix_dict(inst, ev_ecd["z"], ev_ecd["C"], ev_ecd["x"])
    eev_ecd_r = _highs_solve_sp(inst, Y, B, fix_first_stage=fix_ecd,
                                relax_first_stage_constraints=True,
                                mip_gap=1e-3, time_limit=1800.0)
    eev_ecd_cost = eev_ecd_r["total_cost"]

    # (3) SP (RP)
    print("  [3/3] SP …", flush=True)
    t0 = time.perf_counter()
    sp_r = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)
    elapsed = time.perf_counter() - t0
    sp_cost   = sp_r["total_cost"]
    tier_h_sp = _tier_h_rate(sp_r["r_cancel"])
    sp_opens  = _opens(sp_r["z"], sp_r["C"])

    vss_naive = 100.0 * (eev_naive_cost - sp_cost) / sp_cost
    vss_ecd   = 100.0 * (eev_ecd_cost   - sp_cost) / sp_cost

    print(f"  Pairs infeasible: {n_pairs_inf}/{n_pairs_total}")
    print(f"  SP opens:    {sp_opens['facilities']}")
    print(f"  VSS-vs-naive: {vss_naive:.2f}%")
    print(f"  VSS-vs-ECD:   {vss_ecd:.2f}%")
    print(f"  Tier-H SP:    {tier_h_sp:.2f}%")
    print(f"  Runtime:      {elapsed:.1f}s")

    return {
        "label": point["label"],
        "shelf_life_hours": sl,
        "max_facilities_open": mnf,
        "status": "OK",
        "n_pair_infeasibilities": n_pairs_inf,
        "n_pair_total": n_pairs_total,
        "n_patients_with_no_feasible_facility": 0,
        "ev_naive_cost": round(float(ev_naive["ev_cost"]), 6),
        "eev_naive_cost": round(float(eev_naive_cost), 6),
        "ev_ecd_cost": round(float(ev_ecd["ev_cost"]), 6),
        "eev_ecd_cost": round(float(eev_ecd_cost), 6),
        "rp_cost": round(float(sp_cost), 6),
        "vss_vs_naive_pct": round(float(vss_naive), 4),
        "vss_vs_ecd_pct": round(float(vss_ecd), 4),
        "sp_facility_set": sp_opens["facilities"],
        "sp_capacities": sp_opens["capacities"],
        "tier_h_sp_pct": round(float(tier_h_sp), 4),
        "elapsed_s": round(float(elapsed), 2),
    }


def main() -> None:
    t_wall = time.perf_counter()
    print("=== Structural-constraint sensitivity (2×2: SHELF × MNF) ===")
    print(f"  N = {N_SCENARIOS} scenarios, seed = {SEED}")

    results = [run_cell(point) for point in SWEEP]

    out = {
        "calibration": {
            "n_patients": 50,
            "n_facilities": 4,
            "n_scenarios": N_SCENARIOS,
            "seed": SEED,
            "solver": "HiGHS",
            "mip_rel_gap": 1e-3,
        },
        "sweep": results,
        "interpretation": (
            "Tests whether tightening the structural constraints (shelf life from "
            "24h cryopreserved to 12h fresh-cell, and MNF from 3 to 2) changes the "
            "SP's optimal solution or the magnitude of the contribution claim. A "
            "binding constraint produces either a different SP facility set or a "
            "meaningful shift in VSS values; a non-binding constraint leaves the SP "
            "solution and VSS unchanged."
        ),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    elapsed = time.perf_counter() - t_wall
    print(f"\nResults written to {_OUT}  (total {elapsed:.0f}s)")
    print("\nSummary of 4 runs:")
    for r in results:
        if r["status"] == "OK":
            print(f"  {r['label']:24s}  "
                  f"inf={r['n_pair_infeasibilities']:>2d}/{r['n_pair_total']}  "
                  f"SP={'+'.join(r['sp_facility_set'])}  "
                  f"VSS/Naive={r['vss_vs_naive_pct']:.2f}%  "
                  f"VSS/ECD={r['vss_vs_ecd_pct']:.2f}%  "
                  f"Tier-H={r['tier_h_sp_pct']:.2f}%")
        else:
            print(f"  {r['label']:24s}  STATUS: {r['status']}  "
                  f"({r['n_patients_with_no_feasible_facility']} patients "
                  f"with no feasible facility)")


if __name__ == "__main__":
    main()
