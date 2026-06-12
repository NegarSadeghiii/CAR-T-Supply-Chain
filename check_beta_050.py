"""
check_beta_050.py — single sweep point at β_H = 0.50.

Solves naive EV→EEV, ECD EV→EEV, and SP at β_H = 0.50.
Compares against β_H = 0.55 from results/sensitivity_beta_results.json.

Run: python check_beta_050.py
Output: results/sensitivity_beta_check_050.json (not committed until decision made)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from sensitivity_beta import (
    build_sweep_instance, _ecd_effective_costs,
    _fix_dict, _tier_h_rate, _opens,
    N_SCENARIOS, SEED,
)
from case_study import _highs_solve_sp, _solve_ev
from yield_sp_v1 import Instance, sample_scenarios

_REPO = Path(__file__).parent
_OUT  = _REPO / "results" / "sensitivity_beta_check_050.json"
_PREV = _REPO / "results" / "sensitivity_beta_results.json"

BETA_H_NEW = 0.50


def run_single(beta_H: float) -> dict:
    inst = build_sweep_instance(beta_H)
    Y, B = sample_scenarios(inst, N_SCENARIOS, seed=SEED)

    # (1) Naive EV → EEV
    ev_naive = _solve_ev(inst)
    fix_n    = _fix_dict(inst, ev_naive["z"], ev_naive["C"], ev_naive["x"])
    eev_n    = _highs_solve_sp(inst, Y, B, fix_first_stage=fix_n,
                                relax_first_stage_constraints=True,
                                mip_gap=1e-3, time_limit=1800.0)

    # (2) ECD EV → EEV
    c_eff, exp_rec, c_eff_list = _ecd_effective_costs(inst)
    inst_ecd = Instance(
        n_patients=inst.n_patients, n_facilities=inst.n_facilities,
        f=inst.f.copy(), pi=inst.pi.copy(), c=c_eff,
        s_max=inst.s_max.copy(), p=inst.p.copy(),
        beta=inst.beta.copy(), rho_leuk=inst.rho_leuk,
        rho_remfg=inst.rho_remfg.copy(), rho_sub=inst.rho_sub.copy(),
        rho_cancel=inst.rho_cancel.copy(),
    )
    ev_ecd = _solve_ev(inst_ecd)
    fix_e  = _fix_dict(inst, ev_ecd["z"], ev_ecd["C"], ev_ecd["x"])
    eev_e  = _highs_solve_sp(inst, Y, B, fix_first_stage=fix_e,
                              relax_first_stage_constraints=True,
                              mip_gap=1e-3, time_limit=1800.0)

    # (3) SP
    sp = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)

    eev_naive_cost = eev_n["total_cost"]
    eev_ecd_cost   = eev_e["total_cost"]
    sp_cost        = sp["total_cost"]

    vss_naive = 100.0 * (eev_naive_cost - sp_cost) / sp_cost
    vss_ecd   = 100.0 * (eev_ecd_cost   - sp_cost) / sp_cost

    return {
        "beta_H": beta_H,
        "naive": {
            "ev_cost":    round(float(ev_naive["ev_cost"]), 6),
            "eev_cost":   round(float(eev_naive_cost), 6),
            "tier_h_pct": round(float(_tier_h_rate(eev_n["r_cancel"])), 4),
            **_opens(ev_naive["z"], ev_naive["C"]),
        },
        "expected_cost_det": {
            "expected_recourse_per_batch": round(float(exp_rec), 6),
            "c_m_eff": c_eff_list,
            "ev_cost":    round(float(ev_ecd["ev_cost"]), 6),
            "eev_cost":   round(float(eev_ecd_cost), 6),
            "tier_h_pct": round(float(_tier_h_rate(eev_e["r_cancel"])), 4),
            **_opens(ev_ecd["z"], ev_ecd["C"]),
        },
        "stochastic": {
            "rp_cost":     round(float(sp_cost), 6),
            "stage1_cost": round(float(sp["stage1_cost"]), 6),
            "stage2_cost": round(float(sp["expected_stage2_cost"]), 6),
            "tier_h_pct":  round(float(_tier_h_rate(sp["r_cancel"])), 4),
            "gap_pct":     round(float(sp["gap"] * 100), 4),
            **_opens(sp["z"], sp["C"]),
        },
        "vss_vs_naive_pct": round(float(vss_naive), 4),
        "vss_vs_ecd_pct":   round(float(vss_ecd), 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    print(f"Running β_H = {BETA_H_NEW} …", flush=True)
    r050 = run_single(BETA_H_NEW)
    elapsed = time.perf_counter() - t0

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as fh:
        json.dump(r050, fh, indent=2)
    print(f"Saved: {_OUT}  ({elapsed:.0f}s)\n")

    # Load β_H = 0.55 from existing sweep
    with open(_PREV) as fh:
        prev = json.load(fh)
    r055 = next(p for p in prev["sweep"] if p["beta_H"] == 0.55)

    rows = [r055, r050]
    print(f"{'β_H':>5}  {'VSS/Naive':>10}  {'VSS/ECD':>9}  "
          f"{'H/Naive':>8}  {'H/ECD':>7}  {'H/SP':>6}  SP opens")
    print("─" * 72)
    for r in rows:
        opens_str = "+".join(r["stochastic"]["facilities"])
        print(f"{r['beta_H']:>5.2f}  "
              f"{r['vss_vs_naive_pct']:>9.2f}%  "
              f"{r['vss_vs_ecd_pct']:>8.2f}%  "
              f"{r['naive']['tier_h_pct']:>7.2f}%  "
              f"{r['expected_cost_det']['tier_h_pct']:>6.2f}%  "
              f"{r['stochastic']['tier_h_pct']:>5.2f}%  "
              f"{opens_str}")

    # |Δ| row
    def d(key, sub): return abs(r050[sub][key] - r055[sub][key])
    delta_vss_naive = abs(r050["vss_vs_naive_pct"] - r055["vss_vs_naive_pct"])
    delta_vss_ecd   = abs(r050["vss_vs_ecd_pct"]   - r055["vss_vs_ecd_pct"])
    delta_h_naive   = d("tier_h_pct", "naive")
    delta_h_ecd     = d("tier_h_pct", "expected_cost_det")
    delta_h_sp      = d("tier_h_pct", "stochastic")
    same_fac = (set(r050["stochastic"]["facilities"]) ==
                set(r055["stochastic"]["facilities"]))
    print("─" * 72)
    print(f"{'|Δ|':>5}  "
          f"{delta_vss_naive:>9.2f}pp  "
          f"{delta_vss_ecd:>8.2f}pp  "
          f"{delta_h_naive:>7.2f}pp  "
          f"{delta_h_ecd:>6.2f}pp  "
          f"{delta_h_sp:>5.2f}pp  "
          f"{'same' if same_fac else 'DIFFERENT'}")

    # Decision
    print()
    if delta_vss_ecd < 1.5 and same_fac:
        cls = "SAFE TO RE-ANCHOR"
        rec = ("Re-anchoring is cosmetic. Re-run all case-study experiments at "
               "β_H = 0.50 and regenerate all figures.")
    elif delta_vss_ecd <= 3.0:
        cls = "MARGINAL"
        rec = ("Keep β_H = 0.55. Add disclosure: 'We adopt β_H = 0.55 as the "
               "central calibration, sitting at the optimistic end of the "
               "empirically observed range β_H ∈ [0.40, 0.55]. Sensitivity in "
               "Figure 15 confirms the contribution claim holds at β_H = 0.40 "
               "(VSS-vs-ECD = 6.76%).'")
    else:
        cls = "MATERIAL"
        rec = ("Full re-anchor required. β_H = 0.50 produces meaningfully "
               "different headline numbers.")

    print(f"Classification: {cls}")
    print(f"Δ VSS vs ECD = {delta_vss_ecd:.2f} pp  |  "
          f"Facility set: {'unchanged' if same_fac else 'CHANGED'}")
    print(f"Recommendation: {rec}")

    if cls == "MATERIAL":
        print("\nDownstream files requiring re-run:")
        for f, t in [
            ("case_study.py",              "~30s"),
            ("expected_cost_baseline.py",  "~60s"),
            ("multi_instance_case_study.py","~10–15min"),
            ("out_of_sample_evaluation.py","~5–10min"),
            ("figures/generate_phase1.py", "~5s"),
            ("figures/generate_phase2.py", "~5s (if exists)"),
            ("figures/generate_phase4.py", "~5s (if exists)"),
            ("figures/generate_phase5.py", "~5s (if exists)"),
        ]:
            print(f"  {f:<45} {t}")


if __name__ == "__main__":
    main()
