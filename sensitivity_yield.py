"""
sensitivity_yield.py — additive yield-rate sensitivity sweep.

Sweeps p_shift ∈ {-0.10, -0.05, 0.00, +0.03}, applying the same additive
shift to all four facility yield rates (clipped to [0, 1]).  Calibrated
central values: p = [0.85, 0.92, 0.95, 0.92].

Scenarios are re-sampled at each shift point (same seed=0) because the yield
distribution Y[ω,i,m] ~ Bernoulli(p[m]) changes with p.  All other parameters
are held at the locked central calibration (ρ^cancel_H=6.0, 50 patients,
4 facilities, N=200 scenarios, 15% subcontract premium).

Run: python sensitivity_yield.py
"""

from __future__ import annotations

import json
import time

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from yield_sp_v1 import Instance, sample_scenarios
from case_study import TIERS, TIER_IDX, BETA_MAP, RHOCANCEL_MAP, _highs_solve_sp, _solve_ev


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

P_SHIFTS    = [-0.10, -0.05, 0.00, +0.03]
P_BASE      = np.array([0.85, 0.92, 0.95, 0.92])   # central calibration

N_SCENARIOS = 200
SEED        = 0


# ---------------------------------------------------------------------------
# Instance factory
# ---------------------------------------------------------------------------

def _build_instance(p_shift: float) -> Instance:
    """50-patient, 4-facility instance with shifted yield rates."""
    n_p = len(TIERS)
    n_f = 4

    p_new = np.clip(P_BASE + p_shift, 0.0, 1.0)

    beta       = np.array([BETA_MAP[t]      for t in TIERS])
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
        p=p_new,
        beta=beta,
        rho_leuk=0.005,
        rho_remfg=c.copy(),
        rho_sub=rho_sub,
        rho_cancel=rho_cancel,
    )


# ---------------------------------------------------------------------------
# Single sweep point
# ---------------------------------------------------------------------------

def _run_point(p_shift: float) -> dict:
    inst = _build_instance(p_shift)
    p_vec = [round(float(v), 4) for v in inst.p]

    # Sample scenarios — must re-sample because p changed
    Y, B = sample_scenarios(inst, N_SCENARIOS, seed=SEED)

    # EV (deterministic, no scenarios)
    ev = _solve_ev(inst)
    ev_fix = {
        "z": {m: int(ev["z"][m]) for m in inst.M},
        "C": {m: int(ev["C"][m]) for m in inst.M},
        "x": {(i, m): int(ev["x"][i, m]) for i in inst.I for m in inst.M},
    }

    # EEV — fix EV first-stage, evaluate stochastically
    eev_r = _highs_solve_sp(inst, Y, B,
                             fix_first_stage=ev_fix,
                             relax_first_stage_constraints=True,
                             mip_gap=1e-3, time_limit=1800.0)
    eev_cost = ev["ev_cost"] + eev_r["expected_stage2_cost"]

    # RP — free first-stage
    t_rp0 = time.perf_counter()
    rp_r  = _highs_solve_sp(inst, Y, B, mip_gap=1e-3, time_limit=1800.0)
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
        "p_shift":      p_shift,
        "p_vec":        p_vec,
        "ev_stage1":    ev["ev_cost"],
        "ev_cap":       int(ev["C"].sum()),
        "ev_open":      sorted(int(m) for m in inst.M if ev["z"][m] > 0.5),
        "eev_cost":     eev_cost,
        "rp_cost":      rp_r["total_cost"],
        "rp_stage1":    rp_r["stage1_cost"],
        "rp_stage2":    rp_r["expected_stage2_cost"],
        "rp_cap":       int(rp_r["C"].sum()),
        "rp_open":      sorted(int(m) for m in inst.M if rp_r["z"][m] > 0.5),
        "vss":          vss,
        "vss_pct":      vss_pct,
        "h_cancel_ev":  rc_ev[:, TIER_IDX["H"]].mean() * 100.0,
        "h_cancel_rp":  rc_rp[:, TIER_IDX["H"]].mean() * 100.0,
        "recourse_mix": mix,
        "rp_time":      rp_time,
        "rp_gap":       rp_r["gap"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_wall0 = time.perf_counter()

    print("=== Yield-rate sensitivity sweep ===")
    print(f"  Sweep:  p_shift ∈ {P_SHIFTS}")
    print(f"  Base p: {P_BASE.tolist()}")
    print(f"  N={N_SCENARIOS} scenarios, seed={SEED}")
    print(f"  ρ^cancel_H=6.0, ρ^cancel_M=1.50, ρ^cancel_L=0.75 (locked)")
    print()

    results = []
    for idx, ps in enumerate(P_SHIFTS, 1):
        p_new = np.clip(P_BASE + ps, 0.0, 1.0)
        print(f"[{idx}/{len(P_SHIFTS)}] p_shift={ps:+.2f} → p={p_new.tolist()} …",
              flush=True)
        try:
            pt = _run_point(ps)
            results.append(pt)
            ev_lbl = "[" + ",".join(f"m{m}" for m in pt["ev_open"]) + "]"
            rp_lbl = "[" + ",".join(f"m{m}" for m in pt["rp_open"]) + "]"
            print(f"    VSS = {pt['vss']:.4f} M ({pt['vss_pct']:.1f}%)  "
                  f"EV {ev_lbl} → RP {rp_lbl}  "
                  f"(RP {pt['rp_time']:.1f}s)", flush=True)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append({"p_shift": ps, "error": str(exc)})

    t_wall = time.perf_counter() - t_wall0

    # ----- Table -----
    good = [r for r in results if "error" not in r]
    print()
    hdr = (
        f"{'p_shift':>7} | {'p (4-facility)':^29} | {'VSS (M)':>7} | {'VSS (%)':>7} | "
        f"{'EV opens':<10} | {'RP opens':<10} | "
        f"{'Tier-H EV→RP':<16} | {'Sub%':>5} | {'Remfg%':>6} | {'Cancel%':>7}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"  {r['p_shift']:+.2f}{'':>5} | ERROR: {r['error'][:80]}")
            continue
        mix    = r["recourse_mix"]
        ev_lbl = "[" + ",".join(f"m{m}" for m in r["ev_open"]) + "]"
        rp_lbl = "[" + ",".join(f"m{m}" for m in r["rp_open"]) + "]"
        tier_h = f"{r['h_cancel_ev']:.1f}%→{r['h_cancel_rp']:.1f}%"
        p_str  = str(r["p_vec"])
        print(
            f"{r['p_shift']:>+7.2f} | "
            f"{p_str:^29} | "
            f"{r['vss']:>7.4f} | "
            f"{r['vss_pct']:>7.1f} | "
            f"{ev_lbl:<10} | "
            f"{rp_lbl:<10} | "
            f"{tier_h:<16} | "
            f"{mix['subcontract']:>5.1f} | "
            f"{mix['re-manufacture']:>6.1f} | "
            f"{mix['cancel']:>7.1f}"
        )

    # ----- Analysis -----
    vss_vals = [r["vss_pct"] for r in good]
    all_pos  = all(v > 0 for v in vss_vals)
    ordering = all(r["eev_cost"] > r["rp_cost"] for r in good)

    # Monotonicity check: VSS should increase as p_shift decreases
    vss_by_shift = [(r["p_shift"], r["vss_pct"]) for r in good]
    vss_by_shift.sort(key=lambda x: x[0])
    monotone = all(
        vss_by_shift[i][1] >= vss_by_shift[i + 1][1]
        for i in range(len(vss_by_shift) - 1)
    )

    # Structural transition in RP facility choice
    base_open = tuple(good[0]["rp_open"])
    threshold = next(
        (r["p_shift"] for r in good[1:] if tuple(r["rp_open"]) != base_open),
        None,
    )

    # Subcontracting presence
    sub_active = {r["p_shift"]: r["recourse_mix"]["subcontract"] > 0.0 for r in good}

    print()
    print(f"Range across p_shift ∈ [{P_SHIFTS[0]}, {P_SHIFTS[-1]}]:")
    print(f"  VSS spans {min(vss_vals):.1f}% to {max(vss_vals):.1f}%")
    print(f"  All points show VSS > 0: {all_pos}")
    print(f"  EEV > RP at all points (ordering preserved): {ordering}")
    print(f"  VSS increases monotonically as yields degrade: {monotone}")
    if not monotone:
        print("  [FLAG] Non-monotonic VSS — inspect individual points")
    if threshold is not None:
        pre  = [r for r in good if r["p_shift"] < threshold][-1]["rp_open"]
        post = next(r for r in good if r["p_shift"] >= threshold)["rp_open"]
        print(f"  RP facility choice changes at p_shift = {threshold:+.2f}: "
              f"[{','.join(f'm{m}' for m in pre)}]"
              f" → [{','.join(f'm{m}' for m in post)}]")
    else:
        print(f"  RP facility choice stable: [{','.join(f'm{m}' for m in base_open)}]")
    sub_all   = all(sub_active.values())
    sub_none  = not any(sub_active.values())
    sub_label = ("all yield levels" if sub_all
                 else "no yield levels" if sub_none
                 else "some yield levels only")
    print(f"  Subcontracting activates at: {sub_label}")
    sub_detail = ", ".join(
        f"p_shift={ps:+.2f}:{'yes' if act else 'no'}"
        for ps, act in sorted(sub_active.items())
    )
    print(f"    ({sub_detail})")
    print()
    print(f"Total wall-clock time: {t_wall:.1f} s")

    # ----- JSON output -----
    out = {
        "sweep_param": "p_shift",
        "values": P_SHIFTS,
        "p_base": P_BASE.tolist(),
        "held_constant": {
            "n_patients":   len(TIERS),
            "n_facilities": 4,
            "n_scenarios":  N_SCENARIOS,
            "seed":         SEED,
            "rho_cancel_h": 6.00,
            "rho_cancel_m": 2.00,
            "rho_cancel_l": 0.75,
            "sub_premium":  0.15,
            "f":  [0.5, 2.0, 3.0, 2.0],
        },
        "results": results,
        "analysis": {
            "vss_min_pct":          float(min(vss_vals)),
            "vss_max_pct":          float(max(vss_vals)),
            "all_positive":         bool(all_pos),
            "ordering_preserved":   bool(ordering),
            "monotone_vss":         bool(monotone),
            "structural_threshold": float(threshold) if threshold is not None else None,
            "subcontract_active":   {f"{ps:+.2f}": bool(act)
                                     for ps, act in sorted(sub_active.items())},
        },
    }
    with open("sensitivity_yield_results.json", "w") as fh:
        json.dump(out, fh, indent=2, cls=_NumpyEncoder)
    print("Results saved to sensitivity_yield_results.json")


if __name__ == "__main__":
    main()
