"""
sensitivity_p_m.py — manufacturing success-probability sensitivity sweep.

Sweeps a 5×3 grid:
  p_shift  ∈ {-0.10, -0.05, 0.00, +0.05, +0.10}  (additive shift to all p_m)
  p_spread ∈ {0.5, 1.0, 1.5}                       (multiplicative spread of
                                                      deviations from the mean)

The base success probabilities are p = [0.85, 0.92, 0.95, 0.92] (m0..m3).
For each grid point, the perturbed vector is:
    p_m(shift, spread) = clip(p_base_mean + spread*(p_base - p_base_mean) + shift, 0.01, 0.99)

For each grid point three plans are solved:
  (1) Naive deterministic EV → EEV
  (2) ECD expected-cost deterministic → EEV
  (3) Full two-stage stochastic plan (SP / RP)

Sanity check at (shift=0, spread=1.0): VSS/naive ≈ 14.6%, VSS/ECD ≈ 6.9%
(Yield_upgrade: ECD now finds m0+m3 instead of m0+m1; same stage-1 cost, lower
EEV due to more patients at higher-yield m3), SP opens m0+m2.

Run: python sensitivity_p_m.py
Output: results/sensitivity_p_m_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sensitivity_beta import (
    _ecd_effective_costs, _fix_dict, _tier_h_rate, _opens,
    N_SCENARIOS, SEED,
)
from case_study import (
    TIERS, TIER_IDX, RHOCANCEL_MAP, _highs_solve_sp, _solve_ev,
    T_TRANS, SHELF_LIFE_HOURS, MAX_FACILITIES_OPEN,
)
from yield_sp_v1 import Instance, sample_scenarios

_REPO = Path(__file__).parent
_OUT  = _REPO / "results" / "sensitivity_p_m_results.json"

# ── sweep grid ────────────────────────────────────────────────────────────────

P_SHIFT_VALS  = [-0.10, -0.05, 0.00, +0.05, +0.10]
P_SPREAD_VALS = [0.5, 1.0, 1.5]

P_BASE = np.array([0.85, 0.92, 0.95, 0.92])   # case-study calibration


# ── instance builder ──────────────────────────────────────────────────────────

def build_pm_instance(p_shift: float, p_spread: float) -> Instance:
    """Build case-study instance with perturbed p vector."""
    p_mean = float(P_BASE.mean())
    p_new  = np.clip(p_mean + p_spread * (P_BASE - p_mean) + p_shift,
                     0.01, 0.99)

    n_p = len(TIERS)
    n_f = 4

    beta_map   = {"H": 0.55, "M": 0.78, "L": 0.92}
    beta       = np.array([beta_map[t]      for t in TIERS])
    rho_cancel = np.array([RHOCANCEL_MAP[t] for t in TIERS])

    c = np.array([0.20, 0.18, 0.15, 0.18])
    rho_sub = np.zeros((n_f, n_f))
    for m in range(n_f):
        for mp in range(n_f):
            if m != mp:
                rho_sub[m, mp] = c[mp] * 1.15

    return Instance(
        n_patients=n_p,
        n_facilities=n_f,
        f=np.array([0.5, 2.0, 3.0, 2.0]),
        pi=np.array([0.04, 0.06, 0.09, 0.06]),
        c=c,
        s_max=np.array([40, 40, 40, 40], dtype=float),
        p=p_new,
        beta=beta,
        rho_leuk=0.005,
        rho_remfg=c.copy(),
        rho_sub=rho_sub,
        rho_cancel=rho_cancel,
        t_trans=T_TRANS,
        shelf_life=SHELF_LIFE_HOURS,
        mnf=MAX_FACILITIES_OPEN,
    )


# ── single grid-point solver ──────────────────────────────────────────────────

def run_point(p_shift: float, p_spread: float) -> dict:
    inst = build_pm_instance(p_shift, p_spread)
    Y, B = sample_scenarios(inst, N_SCENARIOS, seed=SEED)

    # (1) Naive EV → EEV
    ev_naive  = _solve_ev(inst)
    fix_n     = _fix_dict(inst, ev_naive["z"], ev_naive["C"], ev_naive["x"])
    eev_n     = _highs_solve_sp(inst, Y, B, fix_first_stage=fix_n,
                                 relax_first_stage_constraints=True,
                                 mip_gap=1e-3, time_limit=1800.0)

    # (2) ECD EV → EEV
    c_eff, exp_rec, c_eff_list = _ecd_effective_costs(inst)
    inst_ecd = Instance(
        n_patients=inst.n_patients, n_facilities=inst.n_facilities,
        f=inst.f.copy(), pi=inst.pi.copy(), c=c_eff,
        s_max=inst.s_max.copy(), p=inst.p.copy(),
        beta=inst.beta.copy(), rho_leuk=inst.rho_leuk,
        rho_remfg=inst.rho_remfg.copy(),
        rho_sub=inst.rho_sub.copy(),
        rho_cancel=inst.rho_cancel.copy(),
        t_trans=inst.t_trans,
        shelf_life=inst.shelf_life,
        mnf=inst.mnf,
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
        "p_shift":  round(float(p_shift), 4),
        "p_spread": round(float(p_spread), 4),
        "p_vec":    [round(float(v), 4) for v in inst.p],
        "naive": {
            "ev_cost":    round(float(ev_naive["ev_cost"]), 6),
            "eev_cost":   round(float(eev_naive_cost), 6),
            "tier_h_pct": round(float(_tier_h_rate(eev_n["r_cancel"])), 4),
            **_opens(ev_naive["z"], ev_naive["C"]),
        },
        "expected_cost_det": {
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_wall = time.perf_counter()
    total = len(P_SHIFT_VALS) * len(P_SPREAD_VALS)
    print(f"=== p_m sensitivity sweep ({total} grid points) ===")
    print(f"  p_shift  ∈ {P_SHIFT_VALS}")
    print(f"  p_spread ∈ {P_SPREAD_VALS}")
    print(f"  N = {N_SCENARIOS} scenarios, seed = {SEED}")
    print()

    grid = []
    done = 0
    for p_spread in P_SPREAD_VALS:
        for p_shift in P_SHIFT_VALS:
            done += 1
            t0 = time.perf_counter()
            print(f"  [{done:2d}/{total}] shift={p_shift:+.2f}  spread={p_spread:.1f} …",
                  flush=True)
            pt = run_point(p_shift, p_spread)
            elapsed_pt = time.perf_counter() - t0
            print(f"         VSS/naive={pt['vss_vs_naive_pct']:.2f}%  "
                  f"VSS/ECD={pt['vss_vs_ecd_pct']:.2f}%  "
                  f"SP={pt['stochastic']['facilities']}  "
                  f"({elapsed_pt:.0f}s)")
            grid.append(pt)

    # Sanity check at calibrated point (shift=0, spread=1.0)
    calib = next(p for p in grid
                 if abs(p["p_shift"]) < 1e-9 and abs(p["p_spread"] - 1.0) < 1e-9)
    print("\n=== Sanity check at (shift=0, spread=1.0) ===")
    print(f"  VSS/naive = {calib['vss_vs_naive_pct']:.2f}%  "
          f"(expect ≈ 14.6%)")
    print(f"  VSS/ECD   = {calib['vss_vs_ecd_pct']:.2f}%  "
          f"(expect ≈ 6.9%, Yield_upgrade: ECD finds m0+m3)")
    print(f"  SP opens  = {calib['stochastic']['facilities']}  "
          f"(expect ['m0', 'm2'])")

    ok_vss_n = abs(calib["vss_vs_naive_pct"] - 14.6) < 2.0
    ok_vss_e = abs(calib["vss_vs_ecd_pct"]   - 6.9) < 2.0
    ok_fac   = set(calib["stochastic"]["facilities"]) == {"m0", "m2"}
    status = "PASS" if (ok_vss_n and ok_vss_e and ok_fac) else "WARN"
    print(f"  Status: {status}")

    out = {
        "calibration": {
            "p_base": P_BASE.tolist(),
            "n_scenarios": N_SCENARIOS,
            "seed": SEED,
        },
        "sweep": grid,
        "calibrated_point_sanity": {
            "p_shift": 0.0, "p_spread": 1.0,
            "vss_vs_naive_pct": calib["vss_vs_naive_pct"],
            "vss_vs_ecd_pct":   calib["vss_vs_ecd_pct"],
            "sp_facilities":    calib["stochastic"]["facilities"],
            "status":           status,
        },
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    elapsed = time.perf_counter() - t_wall
    print(f"\nSaved: {_OUT}  (total {elapsed:.0f}s / {elapsed/60:.1f}min)")

    # Summary table
    print()
    print(f"{'shift':>7} {'spread':>7}  {'VSS/naive':>10}  {'VSS/ECD':>9}  SP opens")
    print("─" * 60)
    for pt in grid:
        print(f"  {pt['p_shift']:>5.2f}  {pt['p_spread']:>6.1f}  "
              f"{pt['vss_vs_naive_pct']:>9.2f}%  "
              f"{pt['vss_vs_ecd_pct']:>8.2f}%  "
              f"{'+'.join(pt['stochastic']['facilities'])}")


if __name__ == "__main__":
    main()
