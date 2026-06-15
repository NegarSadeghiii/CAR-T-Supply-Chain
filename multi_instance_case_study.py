"""
multi_instance_case_study.py
============================
Runs the full three-baseline pipeline (naive EV, expected-cost deterministic,
stochastic SP) at patient cohort sizes |I| ∈ {50, 100, 200, 500}.

All four instances share the same parameter assumptions as the 50-patient
case study; only |I| and s_max (capacity per facility) are scaled.

Run:
    python multi_instance_case_study.py 2>&1 | tee results/multi_instance_log.txt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from case_study import TIER_IDX, _highs_solve_sp, _solve_ev
from expected_cost_baseline import _compute_effective_costs
from yield_sp_v1 import Instance, sample_scenarios

_REPO    = Path(__file__).parent
_RESULTS = _REPO / "results"
_RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SIZES = [50, 100, 200, 500]

SCENARIO_COUNT = {50: 200, 100: 200, 200: 200, 500: 200}

# Tier mix proportions (20 % H, 50 % M, 30 % L) — held constant
TIER_PROPS = {"H": 0.20, "M": 0.50, "L": 0.30}

# Calibrated parameter held constant across sizes
P_BASE      = np.array([0.85, 0.92, 0.95, 0.92])
F_BASE      = np.array([0.5,  2.0,  3.0,  2.0 ])
PI_BASE     = np.array([0.04, 0.06, 0.09, 0.06])
C_BASE      = np.array([0.20, 0.18, 0.15, 0.18])
BETA_MAP    = {"H": 0.55, "M": 0.78, "L": 0.92}
RHO_CANCEL  = {"H": 6.00, "M": 2.00, "L": 0.75}
RHO_LEUK    = 0.005
SEED        = 0


def _tier_mix(n: int) -> list[str]:
    """Return a TIERS list for n patients at fixed H/M/L proportions."""
    n_H = round(n * TIER_PROPS["H"])
    n_L = round(n * TIER_PROPS["L"])
    n_M = n - n_H - n_L
    return ["H"] * n_H + ["M"] * n_M + ["L"] * n_L


def _build_instance(n: int) -> Instance:
    """Build a calibrated Instance scaled to n patients."""
    tiers      = _tier_mix(n)
    n_f        = 4
    s_max_val  = max(40, math.ceil(n / 2))

    beta       = np.array([BETA_MAP[t]   for t in tiers])
    rho_cancel = np.array([RHO_CANCEL[t] for t in tiers])

    rho_sub = np.zeros((n_f, n_f))
    for m in range(n_f):
        for mp in range(n_f):
            if m != mp:
                rho_sub[m, mp] = C_BASE[mp] * 1.15

    return Instance(
        n_patients=n,
        n_facilities=n_f,
        f=F_BASE.copy(),
        pi=PI_BASE.copy(),
        c=C_BASE.copy(),
        s_max=np.full(n_f, s_max_val, dtype=float),
        p=P_BASE.copy(),
        beta=beta,
        rho_leuk=RHO_LEUK,
        rho_remfg=C_BASE.copy(),
        rho_sub=rho_sub,
        rho_cancel=rho_cancel,
    )


def _tier_counts(tiers: list[str]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(tiers))


def _tier_idx(tiers: list[str]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {"H": [], "M": [], "L": []}
    for i, t in enumerate(tiers):
        idx[t].append(i)
    return idx


def _recourse_mix(r_remfg, r_sub, r_cancel) -> dict[str, float]:
    rem = float(r_remfg.sum())
    sub = float(r_sub.sum())
    can = float(r_cancel.sum())
    tot = rem + sub + can
    if tot == 0:
        return {"remfg": 0.0, "sub": 0.0, "cancel": 0.0}
    return {
        "remfg":  round(100.0 * rem / tot, 4),
        "sub":    round(100.0 * sub / tot, 4),
        "cancel": round(100.0 * can / tot, 4),
    }


def _tier_cancel_rates(r_cancel, t_idx, N) -> dict[str, float]:
    return {
        t: round(float(r_cancel[:, t_idx[t]].sum()) /
                 (N * max(len(t_idx[t]), 1)) * 100.0, 4)
        for t in ("H", "M", "L")
    }


def _facility_alloc(x_mat, t_idx, n_f) -> dict[str, dict[str, int]]:
    out = {}
    for m in range(n_f):
        out[f"m_{m}"] = {
            t: int(x_mat[t_idx[t], m].sum()) for t in ("H", "M", "L")
        }
    return out


def _stage1_actual_cost(inst: Instance, z, C, x) -> float:
    """Stage-1 cost using the *original* c (not effective) for comparability."""
    return float(
        sum(inst.f[m]  * int(z[m]) for m in range(inst.n_facilities))
        + sum(inst.pi[m] * int(C[m]) for m in range(inst.n_facilities))
        + sum(inst.c[m]  * int(x[i, m])
              for i in range(inst.n_patients)
              for m in range(inst.n_facilities))
    )


# ---------------------------------------------------------------------------
# Per-instance runner
# ---------------------------------------------------------------------------

def run_instance(n: int) -> dict:
    t_wall = time.perf_counter()
    N      = SCENARIO_COUNT[n]

    print(f"\n{'='*60}")
    print(f"  Instance: {n} patients | {N} scenarios | seed={SEED}")
    print(f"{'='*60}")

    inst        = _build_instance(n)
    tiers       = _tier_mix(n)
    t_idx       = _tier_idx(tiers)
    s_max_val   = int(inst.s_max[0])

    print(f"  Tier mix: H={len(t_idx['H'])} M={len(t_idx['M'])} L={len(t_idx['L'])}")
    print(f"  s_max = {s_max_val} per facility")

    print(f"  Sampling {N} scenarios ...", flush=True)
    Y, B = sample_scenarios(inst, N, seed=SEED)

    # -----------------------------------------------------------
    # 1. Naive EV
    # -----------------------------------------------------------
    print(f"  [1/3] Naive EV ...", flush=True)
    t0       = time.perf_counter()
    ev_naive = _solve_ev(inst)
    ev_fix_naive = {
        "z": {m: int(ev_naive["z"][m]) for m in inst.M},
        "C": {m: int(ev_naive["C"][m]) for m in inst.M},
        "x": {(i, m): int(ev_naive["x"][i, m]) for i in inst.I for m in inst.M},
    }
    eev_naive_r = _highs_solve_sp(
        inst, Y, B,
        fix_first_stage=ev_fix_naive,
        relax_first_stage_constraints=True,
        mip_gap=1e-3, time_limit=1800.0,
    )
    naive_s1   = _stage1_actual_cost(inst, ev_naive["z"], ev_naive["C"], ev_naive["x"])
    naive_s2   = float(eev_naive_r["expected_stage2_cost"])
    naive_time = time.perf_counter() - t0
    naive_mix  = _recourse_mix(eev_naive_r["r_remfg"], eev_naive_r["r_sub"], eev_naive_r["r_cancel"])
    naive_tc   = _tier_cancel_rates(eev_naive_r["r_cancel"], t_idx, N)
    print(f"       Stage-1={naive_s1:.3f}  Stage-2={naive_s2:.3f}  "
          f"Total={naive_s1+naive_s2:.3f}  ({naive_time:.1f}s)")

    # -----------------------------------------------------------
    # 2. Smart ECD baseline
    # -----------------------------------------------------------
    print(f"  [2/3] Smart ECD ...", flush=True)
    t0        = time.perf_counter()
    c_eff, calib = _compute_effective_costs(inst)
    inst_smart = Instance(
        n_patients=inst.n_patients, n_facilities=inst.n_facilities,
        f=inst.f.copy(), pi=inst.pi.copy(), c=c_eff,
        s_max=inst.s_max.copy(), p=inst.p.copy(), beta=inst.beta.copy(),
        rho_leuk=inst.rho_leuk, rho_remfg=inst.rho_remfg.copy(),
        rho_sub=inst.rho_sub.copy(), rho_cancel=inst.rho_cancel.copy(),
    )
    ev_smart = _solve_ev(inst_smart)
    ev_fix_smart = {
        "z": {m: int(ev_smart["z"][m]) for m in inst.M},
        "C": {m: int(ev_smart["C"][m]) for m in inst.M},
        "x": {(i, m): int(ev_smart["x"][i, m]) for i in inst.I for m in inst.M},
    }
    eev_smart_r = _highs_solve_sp(
        inst, Y, B,
        fix_first_stage=ev_fix_smart,
        relax_first_stage_constraints=True,
        mip_gap=1e-3, time_limit=1800.0,
    )
    smart_s1   = _stage1_actual_cost(inst, ev_smart["z"], ev_smart["C"], ev_smart["x"])
    smart_s2   = float(eev_smart_r["expected_stage2_cost"])
    smart_time = time.perf_counter() - t0
    smart_mix  = _recourse_mix(eev_smart_r["r_remfg"], eev_smart_r["r_sub"], eev_smart_r["r_cancel"])
    smart_tc   = _tier_cancel_rates(eev_smart_r["r_cancel"], t_idx, N)
    alloc_differs = bool(
        not np.array_equal(ev_smart["z"], ev_naive["z"]) or
        not np.array_equal(ev_smart["x"], ev_naive["x"])
    )
    print(f"       Stage-1={smart_s1:.3f}  Stage-2={smart_s2:.3f}  "
          f"Total={smart_s1+smart_s2:.3f}  ({smart_time:.1f}s)")

    # -----------------------------------------------------------
    # 3. Stochastic SP (RP)
    # -----------------------------------------------------------
    print(f"  [3/3] Stochastic SP (RP) ...", flush=True)
    t0    = time.perf_counter()
    rp_r  = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)
    rp_time = time.perf_counter() - t0
    rp_s1  = float(rp_r["stage1_cost"])
    rp_s2  = float(rp_r["expected_stage2_cost"])
    rp_mix = _recourse_mix(rp_r["r_remfg"], rp_r["r_sub"], rp_r["r_cancel"])
    rp_tc  = _tier_cancel_rates(rp_r["r_cancel"], t_idx, N)
    rp_gap = float(rp_r["gap"]) if rp_r["gap"] else 0.0
    if rp_gap > 0.05:
        print(f"       WARNING: RP gap = {rp_gap*100:.1f}%")
    print(f"       Stage-1={rp_s1:.3f}  Stage-2={rp_s2:.3f}  "
          f"Total={rp_s1+rp_s2:.3f}  ({rp_time:.1f}s, gap={rp_gap*100:.2f}%)")

    # -----------------------------------------------------------
    # VSS
    # -----------------------------------------------------------
    naive_total = naive_s1 + naive_s2
    smart_total = smart_s1 + smart_s2
    rp_total    = rp_s1   + rp_s2
    vss_naive   = round(100.0 * (naive_total - rp_total) / rp_total, 4)
    vss_smart   = round(100.0 * (smart_total - rp_total) / rp_total, 4)

    print(f"\n  VSS naive: {vss_naive:.2f}%   VSS smart: {vss_smart:.2f}%")
    if not (naive_total >= smart_total >= rp_total):
        print(f"  [WARNING] Cost ordering violated: "
              f"naive={naive_total:.4f} smart={smart_total:.4f} rp={rp_total:.4f}")

    # -----------------------------------------------------------
    # Facility allocation
    # -----------------------------------------------------------
    naive_alloc = _facility_alloc(ev_naive["x"], t_idx, inst.n_facilities)
    smart_alloc = _facility_alloc(ev_smart["x"], t_idx, inst.n_facilities)
    rp_alloc    = _facility_alloc(rp_r["x"],     t_idx, inst.n_facilities)

    sp_opens  = sorted(m for m in range(inst.n_facilities) if rp_r["z"][m] > 0.5)
    ev_opens  = sorted(m for m in range(inst.n_facilities) if ev_naive["z"][m] > 0.5)
    sm_opens  = sorted(m for m in range(inst.n_facilities) if ev_smart["z"][m] > 0.5)
    print(f"  Naive opens: m{ev_opens}   Smart opens: m{sm_opens}   SP opens: m{sp_opens}")

    elapsed = round(time.perf_counter() - t_wall, 1)

    result = {
        "instance_size": n,
        "n_facilities":  4,
        "n_scenarios":   N,
        "seed":          SEED,
        "tier_mix":      {t: len(t_idx[t]) for t in ("H", "M", "L")},
        "s_max":         s_max_val,
        "calibration":   {"expected_recourse_per_batch": calib["expected_recourse_per_batch"]},
        "naive_eev": {
            "stage1":    round(naive_s1, 6), "stage2": round(naive_s2, 6),
            "total":     round(naive_total, 6),
            "solve_time": round(naive_time, 2),
            "opens":     ev_opens,
        },
        "smart_eev": {
            "stage1":    round(smart_s1, 6), "stage2": round(smart_s2, 6),
            "total":     round(smart_total, 6),
            "solve_time": round(smart_time, 2),
            "facility_allocation_differs_from_naive": alloc_differs,
            "opens":     sm_opens,
        },
        "rp": {
            "stage1":    round(rp_s1, 6), "stage2": round(rp_s2, 6),
            "total":     round(rp_total, 6),
            "solve_time": round(rp_time, 2),
            "gap":       round(rp_gap, 6),
            "opens":     sp_opens,
        },
        "vss_vs_naive_pct":  vss_naive,
        "vss_smart_vs_rp_pct": vss_smart,
        "tier_cancel_rates": {
            "naive": naive_tc, "smart": smart_tc, "sp": rp_tc,
        },
        "recourse_mix": {
            "naive": naive_mix, "smart": smart_mix, "sp": rp_mix,
        },
        "facility_allocation": {
            "naive": naive_alloc, "smart": smart_alloc, "sp": rp_alloc,
        },
        "elapsed_s": elapsed,
    }

    path = _RESULTS / f"case_study_results_N{n}.json"
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Saved: {path}  (elapsed {elapsed}s)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()
    print("=== Multi-instance pipeline ===")
    print(f"Sizes: {SIZES}")
    print(f"Scenarios: {SCENARIO_COUNT}")

    summary = []
    for n in SIZES:
        r = run_instance(n)
        summary.append({
            "size":           n,
            "n_scenarios":    r["n_scenarios"],
            "naive_total":    r["naive_eev"]["total"],
            "smart_total":    r["smart_eev"]["total"],
            "rp_total":       r["rp"]["total"],
            "vss_naive_pct":  r["vss_vs_naive_pct"],
            "vss_smart_pct":  r["vss_smart_vs_rp_pct"],
            "tier_h_naive":   r["tier_cancel_rates"]["naive"]["H"],
            "tier_h_smart":   r["tier_cancel_rates"]["smart"]["H"],
            "tier_h_sp":      r["tier_cancel_rates"]["sp"]["H"],
            "sp_solve_time":  r["rp"]["solve_time"],
            "sp_gap_pct":     round(r["rp"]["gap"] * 100, 3),
            "naive_opens":    r["naive_eev"]["opens"],
            "smart_opens":    r["smart_eev"]["opens"],
            "sp_opens":       r["rp"]["opens"],
            "sp_remfg_pct":   r["recourse_mix"]["sp"]["remfg"],
            "sp_sub_pct":     r["recourse_mix"]["sp"]["sub"],
            "sp_cancel_pct":  r["recourse_mix"]["sp"]["cancel"],
            "naive_time":     r["naive_eev"]["solve_time"],
            "smart_time":     r["smart_eev"]["solve_time"],
        })

    multi_path = _RESULTS / "multi_instance_results.json"
    with open(multi_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nAggregated summary saved: {multi_path}")

    # Print summary table
    print()
    print(f"{'|I|':>5} | {'N_scen':>6} | {'Naive':>8} | {'Smart':>8} | {'RP':>8} | "
          f"{'VSS_n%':>7} | {'VSS_s%':>7} | {'TH_n':>6} | {'TH_s':>6} | {'TH_sp':>6} | "
          f"{'SP_t(s)':>8} | {'gap%':>6}")
    print("-" * 115)
    for r in summary:
        print(f"{r['size']:>5} | {r['n_scenarios']:>6} | "
              f"{r['naive_total']:>8.2f} | {r['smart_total']:>8.2f} | "
              f"{r['rp_total']:>8.2f} | {r['vss_naive_pct']:>7.2f} | "
              f"{r['vss_smart_pct']:>7.2f} | {r['tier_h_naive']:>6.2f} | "
              f"{r['tier_h_smart']:>6.2f} | {r['tier_h_sp']:>6.2f} | "
              f"{r['sp_solve_time']:>8.1f} | {r['sp_gap_pct']:>6.3f}")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall-clock time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
