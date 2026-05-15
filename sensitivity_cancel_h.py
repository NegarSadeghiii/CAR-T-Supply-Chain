"""
sensitivity_cancel_h.py — ρ^cancel_H sensitivity sweep.

Sweeps ρ^cancel_H ∈ {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0} M USD, holding all
other parameters constant (50 patients, 4 facilities, N=200 scenarios, seed=0,
ρ^cancel_M=1.50, ρ^cancel_L=0.75).  The same 200 stochastic scenarios (same
seed) are reused at every sweep point so that only the cancellation cost
changes, not the realised uncertainty.

Run: python sensitivity_cancel_h.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from yield_sp_v1 import Instance, sample_scenarios
from case_study import TIERS, TIER_IDX, BETA_MAP, _highs_solve_sp, _solve_ev


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

RHO_CANCEL_H_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

# Held constant across sweep (ρ_M spec-defined; differs from case_study.py's
# 2.00 M — that value was chosen to achieve VSS > 0 at 30 patients; here we
# hold M/L at original spec values and vary H only).
RHO_CANCEL_M = 1.50
RHO_CANCEL_L = 0.75

N_SCENARIOS = 200
SEED        = 0


# ---------------------------------------------------------------------------
# Instance factory
# ---------------------------------------------------------------------------

def _build_instance(rho_cancel_h: float) -> Instance:
    """50-patient, 4-facility instance with swept ρ^cancel_H."""
    n_p = len(TIERS)   # 50
    n_f = 4

    rho_map    = {"H": rho_cancel_h, "M": RHO_CANCEL_M, "L": RHO_CANCEL_L}
    beta       = np.array([BETA_MAP[t] for t in TIERS])
    rho_cancel = np.array([rho_map[t]  for t in TIERS])

    c = np.array([0.20, 0.18, 0.15, 0.18])

    rho_sub = np.zeros((n_f, n_f))
    for m in range(n_f):
        for mp in range(n_f):
            if m != mp:
                rho_sub[m, mp] = c[mp] * 1.15   # 15% sub premium

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


# ---------------------------------------------------------------------------
# Single sweep point
# ---------------------------------------------------------------------------

def _run_point(rho_cancel_h: float, Y: np.ndarray, B: np.ndarray) -> dict:
    inst = _build_instance(rho_cancel_h)

    # EV (deterministic, no scenarios)
    ev = _solve_ev(inst)
    ev_fix = {
        "z": {m: int(ev["z"][m]) for m in inst.M},
        "C": {m: int(ev["C"][m]) for m in inst.M},
        "x": {(i, m): int(ev["x"][i, m]) for i in inst.I for m in inst.M},
    }

    # EEV
    eev_r = _highs_solve_sp(inst, Y, B,
                             fix_first_stage=ev_fix,
                             relax_first_stage_constraints=True,
                             mip_gap=1e-3, time_limit=1800.0)
    eev_cost = ev["ev_cost"] + eev_r["expected_stage2_cost"]

    # RP
    t_rp0 = time.perf_counter()
    rp_r = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)
    rp_time = time.perf_counter() - t_rp0

    vss     = eev_cost - rp_r["total_cost"]
    vss_pct = 100.0 * vss / rp_r["total_cost"] if rp_r["total_cost"] > 0 else 0.0

    # Recourse mix under RP
    rem = rp_r["r_remfg"].sum()
    sub = rp_r["r_sub"].sum()
    can = rp_r["r_cancel"].sum()
    tot = rem + sub + can
    mix = {
        "re-manufacture": 100.0 * rem / tot if tot > 0 else 0.0,
        "subcontract":    100.0 * sub / tot if tot > 0 else 0.0,
        "cancel":         100.0 * can / tot if tot > 0 else 0.0,
    }

    rc_ev = eev_r["r_cancel"]
    rc_rp = rp_r["r_cancel"]

    return {
        "rho_cancel_h":  rho_cancel_h,
        "ev_stage1":     ev["ev_cost"],
        "ev_cap":        int(ev["C"].sum()),
        "ev_open":       sorted(int(m) for m in inst.M if ev["z"][m] > 0.5),
        "eev_cost":      eev_cost,
        "rp_cost":       rp_r["total_cost"],
        "rp_stage1":     rp_r["stage1_cost"],
        "rp_stage2":     rp_r["expected_stage2_cost"],
        "rp_cap":        int(rp_r["C"].sum()),
        "rp_open":       sorted(int(m) for m in inst.M if rp_r["z"][m] > 0.5),
        "vss":           vss,
        "vss_pct":       vss_pct,
        "h_cancel_ev":   rc_ev[:, TIER_IDX["H"]].mean() * 100.0,
        "h_cancel_rp":   rc_rp[:, TIER_IDX["H"]].mean() * 100.0,
        "recourse_mix":  mix,
        "rp_time":       rp_time,
        "rp_gap":        rp_r["gap"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_wall0 = time.perf_counter()

    print("=== ρ^cancel_H sensitivity sweep ===")
    print(f"  Sweep:     {RHO_CANCEL_H_VALUES}")
    print(f"  N={N_SCENARIOS} scenarios, seed={SEED}")
    print(f"  ρ_cancel_M={RHO_CANCEL_M}, ρ_cancel_L={RHO_CANCEL_L}")
    print()

    # Sample scenarios once — same uncertainty for all sweep points
    inst0 = _build_instance(RHO_CANCEL_H_VALUES[0])
    Y, B  = sample_scenarios(inst0, N_SCENARIOS, seed=SEED)
    print(f"Scenarios sampled (Y shape={Y.shape}, B shape={B.shape}).")
    print()

    results = []
    for idx, rho_h in enumerate(RHO_CANCEL_H_VALUES, 1):
        print(f"[{idx}/{len(RHO_CANCEL_H_VALUES)}] ρ^cancel_H = {rho_h:.1f} M …", flush=True)
        try:
            pt = _run_point(rho_h, Y, B)
            results.append(pt)
            ev_lbl = "[" + ",".join(f"m{m}" for m in pt["ev_open"]) + "]"
            rp_lbl = "[" + ",".join(f"m{m}" for m in pt["rp_open"]) + "]"
            print(f"    VSS = {pt['vss']:.4f} M ({pt['vss_pct']:.1f}%)  "
                  f"EV {ev_lbl} → RP {rp_lbl}  "
                  f"(RP {pt['rp_time']:.1f}s)", flush=True)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append({"rho_cancel_h": rho_h, "error": str(exc)})

    t_wall = time.perf_counter() - t_wall0

    # ----- Table -----
    good = [r for r in results if "error" not in r]
    print()
    hdr = (
        f"{'ρ^cancel_H':>10} | {'VSS (M)':>8} | {'VSS (%)':>8} | "
        f"{'EV opens':<12} | {'RP opens':<12} | "
        f"{'Tier-H EV→RP':<16} | {'Sub%':>5} | {'Remfg%':>6} | {'Cancel%':>7}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"  {r['rho_cancel_h']:.1f}{'':>7} | ERROR: {r['error'][:80]}")
            continue
        mix    = r["recourse_mix"]
        ev_lbl = "[" + ",".join(f"m{m}" for m in r["ev_open"]) + "]"
        rp_lbl = "[" + ",".join(f"m{m}" for m in r["rp_open"]) + "]"
        tier_h = f"{r['h_cancel_ev']:.1f}% → {r['h_cancel_rp']:.1f}%"
        print(
            f"{r['rho_cancel_h']:>10.1f} | "
            f"{r['vss']:>8.4f} | "
            f"{r['vss_pct']:>8.1f} | "
            f"{ev_lbl:<12} | "
            f"{rp_lbl:<12} | "
            f"{tier_h:<16} | "
            f"{mix['subcontract']:>5.1f} | "
            f"{mix['re-manufacture']:>6.1f} | "
            f"{mix['cancel']:>7.1f}"
        )

    # ----- Analysis -----
    vss_vals = [r["vss_pct"] for r in good]
    all_pos  = all(v > 0 for v in vss_vals)
    ordering = all(r["eev_cost"] > r["rp_cost"] for r in good)

    # Central value: VSS in 5-7%, closest to 6%
    in_range = [r for r in good if 5.0 <= r["vss_pct"] <= 7.0]
    central  = (min(in_range, key=lambda r: abs(r["vss_pct"] - 6.0))
                if in_range else
                min(good, key=lambda r: abs(r["vss_pct"] - 6.0)))

    # Structural threshold: first rho_h where RP opens a different facility set
    base_open = tuple(good[0]["rp_open"])
    threshold = next(
        (r["rho_cancel_h"] for r in good[1:] if tuple(r["rp_open"]) != base_open),
        None,
    )

    print()
    print(f"Identified central calibration:")
    print(f"  ρ^cancel_H = {central['rho_cancel_h']:.1f} M USD → VSS = {central['vss_pct']:.1f}%"
          f"{'  ✓ (in 5–7% range)' if 5.0 <= central['vss_pct'] <= 7.0 else '  (nearest to 6%)'}")
    print()
    print(f"Range across ρ^cancel_H ∈ [{RHO_CANCEL_H_VALUES[0]}, {RHO_CANCEL_H_VALUES[-1]}] M:")
    print(f"  VSS spans {min(vss_vals):.1f}% to {max(vss_vals):.1f}%")
    print(f"  All points show VSS > 0 (yield-aware planning always beats deterministic): {all_pos}")
    print(f"  EEV > RP at all points (ordering preserved): {ordering}")
    if threshold is not None:
        pre  = [r for r in good if r["rho_cancel_h"] < threshold][-1]["rp_open"]
        post = next(r for r in good if r["rho_cancel_h"] >= threshold)["rp_open"]
        print(f"  RP facility choice changes at ρ^cancel_H ≥ {threshold:.1f} M "
              f"(structural threshold): "
              f"[{','.join(f'm{m}' for m in pre)}]"
              f" → [{','.join(f'm{m}' for m in post)}]")
    else:
        print(f"  RP facility choice stable: [{','.join(f'm{m}' for m in base_open)}]")
    print()
    print(f"Total wall-clock time: {t_wall:.1f} s")

    # ----- JSON output -----
    out = {
        "sweep_param": "rho_cancel_h",
        "values": RHO_CANCEL_H_VALUES,
        "held_constant": {
            "n_patients": len(TIERS),
            "n_facilities": 4,
            "n_scenarios": N_SCENARIOS,
            "seed": SEED,
            "rho_cancel_m": RHO_CANCEL_M,
            "rho_cancel_l": RHO_CANCEL_L,
            "sub_premium": 0.15,
            "f": [0.5, 2.0, 3.0, 2.0],
            "p": [0.85, 0.92, 0.95, 0.92],
        },
        "results": results,
        "analysis": {
            "central_rho_cancel_h":  central["rho_cancel_h"],
            "central_vss_pct":       central["vss_pct"],
            "vss_min_pct":           min(vss_vals),
            "vss_max_pct":           max(vss_vals),
            "all_positive":          all_pos,
            "ordering_preserved":    ordering,
            "structural_threshold":  threshold,
        },
    }
    with open("sensitivity_cancel_h_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("Results saved to sensitivity_cancel_h_results.json")


if __name__ == "__main__":
    main()
