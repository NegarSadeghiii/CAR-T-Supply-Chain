"""
sensitivity_beta.py — β_H sensitivity sweep for the CAR-T supply-chain SP.

Sweeps tier-H re-collection eligibility β_H ∈ {0.40, 0.55, 0.70, 0.85},
holding β_M = 0.78, β_L = 0.92 fixed at the case-study calibration.
For each sweep point, solves three plans:
  (1) Naive deterministic EV → EEV (first-stage fixed, Stage-2 stochastic)
  (2) Expected-cost-deterministic (ECD): adjusts per-batch costs for expected
      failure recourse, solves EV, evaluates EEV with original costs
  (3) Full two-stage stochastic plan (SP / RP)

All solves: HiGHS, mip_rel_gap = 1e-3 (matching case_study.py).
Instance: N = 50 patients, 4 facilities, 200 planning scenarios, seed = 0.

Central calibration: β_H = 0.55 (Locke 2022, Bachy 2022) — probability
that a tier-H patient remains clinically eligible for re-collection after
the 14–21-day manufacturing delay caused by a batch failure. Tier-H patients
are heavily pre-treated and face the highest clinical deterioration risk
during manufacturing delays.

Run: python sensitivity_beta.py
Output: results/sensitivity_beta_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from case_study import (
    TIERS, TIER_IDX, RHOCANCEL_MAP,
    _highs_solve_sp, _solve_ev,
)
from yield_sp_v1 import Instance, sample_scenarios

_REPO   = Path(__file__).parent
_OUT    = _REPO / "results" / "sensitivity_beta_results.json"

# ── sweep parameters ──────────────────────────────────────────────────────────
BETA_H_SWEEP = [0.40, 0.55, 0.70, 0.85]
BETA_M_FIXED = 0.78
BETA_L_FIXED = 0.92

N_SCENARIOS  = 200
SEED         = 0


# ── instance builder ──────────────────────────────────────────────────────────

def build_sweep_instance(beta_H: float) -> Instance:
    """Build the case-study instance with β_H substituted."""
    n_p = len(TIERS)
    n_f = 4

    beta_map = {"H": beta_H, "M": BETA_M_FIXED, "L": BETA_L_FIXED}
    beta      = np.array([beta_map[t]       for t in TIERS])
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
        s_max=np.array([40, 40, 40, 40]),
        p=np.array([0.85, 0.92, 0.95, 0.92]),
        beta=beta,
        rho_leuk=0.005,
        rho_remfg=c.copy(),
        rho_sub=rho_sub,
        rho_cancel=rho_cancel,
    )


# ── ECD effective cost ────────────────────────────────────────────────────────

def _ecd_effective_costs(inst: Instance) -> tuple[np.ndarray, float, list[float]]:
    """
    Compute ECD effective per-batch costs for the given instance.
    Returns (c_eff, expected_recourse_per_batch, c_eff_list).
    """
    n_f = inst.n_facilities
    beta_avg       = float(inst.beta.mean())
    rho_cancel_avg = float(inst.rho_cancel.mean())
    rho_remfg_avg  = float(inst.rho_remfg.mean())
    sub_off = [inst.rho_sub[m, mp]
               for m in range(n_f) for mp in range(n_f) if mp != m]
    rho_sub_avg = float(np.mean(sub_off))
    cheapest    = float(inst.rho_leuk + min(rho_remfg_avg, rho_sub_avg))
    exp_recourse = float(beta_avg * cheapest + (1.0 - beta_avg) * rho_cancel_avg)
    c_eff = inst.c + (1.0 - inst.p) * exp_recourse
    return c_eff, exp_recourse, [round(float(v), 6) for v in c_eff]


# ── helper: build fix_first_stage dict ───────────────────────────────────────

def _fix_dict(inst: Instance, z, C, x) -> dict:
    return {
        "z": {m: int(z[m]) for m in range(inst.n_facilities)},
        "C": {m: int(C[m]) for m in range(inst.n_facilities)},
        "x": {(i, m): int(x[i, m])
              for i in range(inst.n_patients)
              for m in range(inst.n_facilities)},
    }


# ── helper: tier-H cancel rate ────────────────────────────────────────────────

def _tier_h_rate(r_cancel) -> float:
    return float(r_cancel[:, TIER_IDX["H"]].mean() * 100.0)


# ── helper: open facilities ───────────────────────────────────────────────────

def _opens(z, C) -> dict:
    opens = [m for m in range(len(z)) if z[m] > 0.5]
    caps  = {f"m{m}": int(C[m]) for m in opens}
    return {"facilities": [f"m{m}" for m in opens], "capacities": caps}


# ── main sweep ────────────────────────────────────────────────────────────────

def main() -> None:
    t_wall_start = time.perf_counter()
    print("=== β_H sensitivity sweep ===")
    print(f"  Sweep: β_H ∈ {BETA_H_SWEEP}")
    print(f"  Fixed: β_M = {BETA_M_FIXED}, β_L = {BETA_L_FIXED}")
    print(f"  N = {N_SCENARIOS} scenarios, seed = {SEED}")
    print()

    sweep_results = []
    sanity_ok = True

    for beta_H in BETA_H_SWEEP:
        t_pt = time.perf_counter()
        print(f"─── β_H = {beta_H} ───────────────────────────────────────────")

        # Build instance for this sweep point
        inst = build_sweep_instance(beta_H)

        # Sample 200 scenarios (Y fixed by seed; B varies with beta_H)
        Y, B = sample_scenarios(inst, N_SCENARIOS, seed=SEED)

        # ── (1) Naive EV → EEV ────────────────────────────────────────────
        print(f"  [1/3] Naive EV …", flush=True)
        t0 = time.perf_counter()
        ev_naive = _solve_ev(inst)
        t_ev = time.perf_counter() - t0
        print(f"        EV cost={ev_naive['ev_cost']:.4f}  "
              f"opens={[m for m in range(4) if ev_naive['z'][m]>0.5]}  "
              f"t={t_ev:.1f}s")

        fix_naive = _fix_dict(inst, ev_naive["z"], ev_naive["C"], ev_naive["x"])
        print(f"        EEV eval …", flush=True)
        t0 = time.perf_counter()
        eev_naive_r = _highs_solve_sp(inst, Y, B,
                                       fix_first_stage=fix_naive,
                                       relax_first_stage_constraints=True,
                                       mip_gap=1e-3, time_limit=1800.0)
        t_eev = time.perf_counter() - t0
        eev_naive_cost = eev_naive_r["total_cost"]
        tier_h_naive   = _tier_h_rate(eev_naive_r["r_cancel"])
        print(f"        EEV cost={eev_naive_cost:.4f}  "
              f"Tier-H cancel={tier_h_naive:.2f}%  t={t_eev:.1f}s")

        # ── (2) ECD EV → EEV ──────────────────────────────────────────────
        print(f"  [2/3] ECD EV …", flush=True)
        c_eff, exp_recourse, c_eff_list = _ecd_effective_costs(inst)

        inst_ecd = Instance(
            n_patients=inst.n_patients,
            n_facilities=inst.n_facilities,
            f=inst.f.copy(), pi=inst.pi.copy(),
            c=c_eff,
            s_max=inst.s_max.copy(), p=inst.p.copy(),
            beta=inst.beta.copy(), rho_leuk=inst.rho_leuk,
            rho_remfg=inst.rho_remfg.copy(), rho_sub=inst.rho_sub.copy(),
            rho_cancel=inst.rho_cancel.copy(),
        )
        t0 = time.perf_counter()
        ev_ecd = _solve_ev(inst_ecd)
        t_ev_ecd = time.perf_counter() - t0
        print(f"        ECD EV cost={ev_ecd['ev_cost']:.4f}  "
              f"opens={[m for m in range(4) if ev_ecd['z'][m]>0.5]}  "
              f"t={t_ev_ecd:.1f}s")

        fix_ecd = _fix_dict(inst, ev_ecd["z"], ev_ecd["C"], ev_ecd["x"])
        print(f"        ECD EEV eval …", flush=True)
        t0 = time.perf_counter()
        # Evaluate with original inst (true Stage-1 costs, not c_eff)
        eev_ecd_r = _highs_solve_sp(inst, Y, B,
                                     fix_first_stage=fix_ecd,
                                     relax_first_stage_constraints=True,
                                     mip_gap=1e-3, time_limit=1800.0)
        t_eev_ecd = time.perf_counter() - t0
        eev_ecd_cost = eev_ecd_r["total_cost"]
        tier_h_ecd   = _tier_h_rate(eev_ecd_r["r_cancel"])
        print(f"        ECD EEV cost={eev_ecd_cost:.4f}  "
              f"Tier-H cancel={tier_h_ecd:.2f}%  t={t_eev_ecd:.1f}s")

        # ── (3) SP (RP) ────────────────────────────────────────────────────
        print(f"  [3/3] SP …", flush=True)
        t0 = time.perf_counter()
        sp_r = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)
        t_sp = time.perf_counter() - t0
        sp_cost     = sp_r["total_cost"]
        tier_h_sp   = _tier_h_rate(sp_r["r_cancel"])
        sp_opens    = _opens(sp_r["z"], sp_r["C"])
        print(f"        SP cost={sp_cost:.4f}  "
              f"Tier-H cancel={tier_h_sp:.2f}%  "
              f"opens={sp_opens['facilities']}  "
              f"gap={sp_r['gap']*100:.3f}%  t={t_sp:.1f}s")

        # ── VSS ────────────────────────────────────────────────────────────
        vss_naive = 100.0 * (eev_naive_cost - sp_cost) / sp_cost
        vss_ecd   = 100.0 * (eev_ecd_cost   - sp_cost) / sp_cost

        t_pt_total = time.perf_counter() - t_pt
        print(f"  → VSS vs Naive: {vss_naive:.2f}%  "
              f"VSS vs ECD: {vss_ecd:.2f}%  "
              f"(total {t_pt_total:.0f}s)")
        print()

        # ── Sanity checks ──────────────────────────────────────────────────
        if eev_naive_cost < eev_ecd_cost - 0.01:
            print(f"  [SANITY FAIL] β_H={beta_H}: EEV_naive < EEV_ECD "
                  f"({eev_naive_cost:.4f} < {eev_ecd_cost:.4f})")
            sanity_ok = False
        if eev_ecd_cost < sp_cost - 0.01:
            print(f"  [SANITY FAIL] β_H={beta_H}: EEV_ECD < RP "
                  f"({eev_ecd_cost:.4f} < {sp_cost:.4f})")
            sanity_ok = False
        if tier_h_sp > min(tier_h_naive, tier_h_ecd) + 0.5:
            print(f"  [SANITY WARN] β_H={beta_H}: SP Tier-H rate "
                  f"({tier_h_sp:.2f}%) not lower than both baselines")
        if vss_ecd < 4.0:
            print(f"  [CONTRIBUTION WARNING] β_H={beta_H}: "
                  f"VSS vs ECD = {vss_ecd:.2f}% < 4% threshold. "
                  f"The structural-value claim is weak at this β_H value.")

        # Record result
        sweep_results.append({
            "beta_H":   beta_H,
            "naive": {
                "ev_cost":     round(float(ev_naive["ev_cost"]), 6),
                "eev_cost":    round(float(eev_naive_cost), 6),
                "tier_h_pct":  round(float(tier_h_naive), 4),
                **_opens(ev_naive["z"], ev_naive["C"]),
            },
            "expected_cost_det": {
                "expected_recourse_per_batch": round(float(exp_recourse), 6),
                "c_m_eff": c_eff_list,
                "ev_cost":    round(float(ev_ecd["ev_cost"]), 6),
                "eev_cost":   round(float(eev_ecd_cost), 6),
                "tier_h_pct": round(float(tier_h_ecd), 4),
                **_opens(ev_ecd["z"], ev_ecd["C"]),
            },
            "stochastic": {
                "rp_cost":    round(float(sp_cost), 6),
                "stage1_cost": round(float(sp_r["stage1_cost"]), 6),
                "stage2_cost": round(float(sp_r["expected_stage2_cost"]), 6),
                "tier_h_pct": round(float(tier_h_sp), 4),
                "gap_pct":    round(float(sp_r["gap"] * 100), 4),
                "solve_time": round(float(t_sp), 2),
                **_opens(sp_r["z"], sp_r["C"]),
            },
            "vss_vs_naive_pct": round(float(vss_naive), 4),
            "vss_vs_ecd_pct":   round(float(vss_ecd), 4),
        })

    total_wall = time.perf_counter() - t_wall_start
    print(f"=== Sweep complete — {total_wall:.0f}s total ===")
    print()

    # Summary table
    print(f"{'β_H':>5}  {'VSS/Naive':>10}  {'VSS/ECD':>9}  "
          f"{'H/Naive':>8}  {'H/ECD':>7}  {'H/SP':>6}  {'SP opens'}")
    print("─" * 72)
    for r in sweep_results:
        opens_str = "+".join(r["stochastic"]["facilities"])
        print(f"{r['beta_H']:>5.2f}  "
              f"{r['vss_vs_naive_pct']:>9.2f}%  "
              f"{r['vss_vs_ecd_pct']:>8.2f}%  "
              f"{r['naive']['tier_h_pct']:>7.2f}%  "
              f"{r['expected_cost_det']['tier_h_pct']:>6.2f}%  "
              f"{r['stochastic']['tier_h_pct']:>5.2f}%  "
              f"{opens_str}")

    if not sanity_ok:
        print("\n[ERROR] One or more sanity checks failed — review above.")
    else:
        print("\n[OK] All sanity checks passed.")

    # Save JSON
    out_data = {
        "calibration": {
            "N_patients": 50,
            "n_scenarios": N_SCENARIOS,
            "seed": SEED,
            "solver": "HiGHS",
            "mip_rel_gap": 1e-3,
            "beta_M_fixed": BETA_M_FIXED,
            "beta_L_fixed": BETA_L_FIXED,
            "beta_H_central": 0.55,
            "beta_H_sweep": BETA_H_SWEEP,
            "total_wall_time_s": round(total_wall, 1),
        },
        "sweep": sweep_results,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"\nSaved: {_OUT}")


if __name__ == "__main__":
    main()
