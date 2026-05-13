"""
test_tier1.py — Adversarial Tier 1 tests for yield_sp_v1.py.
Exercises constraints (9), (10), (10b), asymmetric subcontracting, and
tier-dependent cancellation pricing with focused hand-crafted instances.
Run: python test_tier1.py
"""

import numpy as np

from yield_sp_v1 import Instance, build_and_solve_sp


# ---------------------------------------------------------------------------
# Test 1 — Eligibility filter binding (constraint 10)
# ---------------------------------------------------------------------------

def test_eligibility_filter():
    """
    All 3 batches fail; patients 0 & 1 are eligible (B=1) but patient 2 is not (B=0).
    Constraint (10) must force patient 2 to cancel; the other two re-manufacture.
    """
    n_p, n_f = 3, 1
    inst = Instance(
        n_patients=n_p, n_facilities=n_f,
        f=np.array([1.0]), pi=np.array([0.05]), c=np.array([0.20]),
        s_max=np.array([5]), p=np.array([0.90]), beta=np.full(n_p, 0.80),
        rho_leuk=0.005, rho_remfg=np.array([0.20]),
        rho_sub=np.zeros((1, 1)), rho_cancel=np.full(n_p, 1.00),
    )
    Y = np.zeros((1, n_p, n_f))        # all batches fail
    B = np.array([[1.0, 1.0, 0.0]])    # patient 2 ineligible

    fix = {"z": {0: 1}, "C": {0: 3}, "x": {(i, 0): 1 for i in range(n_p)}}
    r = build_and_solve_sp(inst, Y, B, fix_first_stage=fix)

    assert r["r_cancel"][0, 2] == 1,    "Patient 2 (ineligible) must cancel"
    assert r["r_cancel"][0, 0] == 0,    "Patient 0 (eligible) must not cancel"
    assert r["r_cancel"][0, 1] == 0,    "Patient 1 (eligible) must not cancel"
    assert r["r_remfg"][0].sum() == 2,  "Expected 2 re-manufactures"

    # Recourse: 2×(0.005+0.20) + 1×1.00 = 1.41
    expected_q = 2 * (0.005 + 0.20) + 1.00
    assert abs(r["expected_stage2_cost"] - expected_q) < 1e-3, (
        f"Recourse {r['expected_stage2_cost']:.5f} != {expected_q:.5f}"
    )

    return True, (
        f"r_cancel={r['r_cancel'][0].tolist()}, r_remfg_total={int(r['r_remfg'][0].sum())}, "
        f"recourse={r['expected_stage2_cost']:.4f} (t={r['solve_time']:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Test 2 — Capacity slack tight binding (constraint 9)
# ---------------------------------------------------------------------------

def test_capacity_slack_binding():
    """
    Three patients all fail at facility 0; C[0]=1 gives slack=1, so constraint (9)
    permits exactly 1 re-manufacture at m0. Constraint (10b) blocks re-mfg at m1
    (patients not assigned there), so the remaining 2 subcontract m0→m1.
    Uses symmetric ρ_remfg=[0.20,0.20].

    Stage-1 is pinned with C[0]=1 but 3 assignments — violates constraint (3),
    hence relax_first_stage_constraints=True.
    """
    n_p, n_f = 3, 2
    rho_sub = np.zeros((2, 2))
    rho_sub[0, 1] = rho_sub[1, 0] = 0.25
    inst = Instance(
        n_patients=n_p, n_facilities=n_f,
        f=np.array([1.0, 0.8]), pi=np.array([0.05, 0.06]), c=np.array([0.20, 0.20]),
        s_max=np.array([5, 5]), p=np.array([0.90, 0.85]), beta=np.full(n_p, 0.80),
        rho_leuk=0.005, rho_remfg=np.array([0.20, 0.20]),
        rho_sub=rho_sub, rho_cancel=np.full(n_p, 1.00),
    )
    Y = np.zeros((1, n_p, n_f))   # all fail everywhere
    B = np.ones((1, n_p))         # all eligible

    fix = {
        "z": {0: 1, 1: 1},
        "C": {0: 1, 1: 5},
        "x": {(i, 0): 1 for i in range(n_p)} | {(i, 1): 0 for i in range(n_p)},
    }
    r = build_and_solve_sp(inst, Y, B, fix_first_stage=fix,
                           relax_first_stage_constraints=True)

    # Cost ranking (with 10b): remfg@m0=0.205 < sub@m0→m1=0.255 < cancel=1.00
    # With slack=1: 1 remfg@m0 + 2 sub@m0→m1 = 0.715 is optimal.
    assert r["r_remfg"][0].sum() == 1, (
        f"Capacity=1 at m0 → exactly 1 re-mfg; got {r['r_remfg'][0].sum()}"
    )
    assert r["r_sub"][0].sum() == 2, (
        f"2 failures must subcontract; got {r['r_sub'][0].sum()}"
    )
    assert r["r_cancel"][0].sum() == 0, (
        f"No cancellations expected; got {r['r_cancel'][0].sum()}"
    )

    # Recourse: 1×(0.005+0.20) + 2×(0.005+0.25) = 0.715
    expected_q = 1 * (0.005 + 0.20) + 2 * (0.005 + 0.25)
    assert abs(r["expected_stage2_cost"] - expected_q) < 1e-3, (
        f"Recourse {r['expected_stage2_cost']:.5f} != {expected_q:.5f}"
    )

    return True, (
        f"remfg={int(r['r_remfg'][0].sum())}, sub={int(r['r_sub'][0].sum())}, "
        f"cancel={int(r['r_cancel'][0].sum())}, "
        f"recourse={r['expected_stage2_cost']:.4f} (t={r['solve_time']:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Test 3 — Subcontract directional asymmetry
# ---------------------------------------------------------------------------

def test_subcontract_asymmetry():
    """
    1 patient at m0, batch fails, B=1. C[0]=1 (slack=1), C[1]=5.
    ρ_remfg=[0.20,0.20] → re-mfg cost 0.205.

    Phase A: ρ_sub[0,1]=0.10 (cheap), ρ_sub[1,0]=0.90.
      r_sub[ω=0, i=0, src=0, dst=1] must fire (cost 0.115 < 0.205).

    Phase B: swap — ρ_sub[0,1]=0.90, ρ_sub[1,0]=0.10.
      Constraint (10b) forbids r_sub[src=1] since x[0,1]=0.
      Only options: sub 0→1 (0.905) or re-mfg at m0 (0.205). Re-mfg wins.
    """
    n_p, n_f = 1, 2
    Y = np.array([[[0.0, 1.0]]])   # patient 0 fails at m0
    B = np.array([[1.0]])
    fix = {"z": {0: 1, 1: 1}, "C": {0: 1, 1: 5}, "x": {(0, 0): 1, (0, 1): 0}}

    def _inst(sub_0_to_1, sub_1_to_0):
        rho_sub = np.zeros((2, 2))
        rho_sub[0, 1] = sub_0_to_1
        rho_sub[1, 0] = sub_1_to_0
        return Instance(
            n_patients=n_p, n_facilities=n_f,
            f=np.array([1.0, 0.8]), pi=np.array([0.05, 0.06]), c=np.array([0.20, 0.20]),
            s_max=np.array([5, 5]), p=np.array([0.90, 0.85]), beta=np.full(n_p, 0.80),
            rho_leuk=0.005, rho_remfg=np.array([0.20, 0.20]),
            rho_sub=rho_sub, rho_cancel=np.full(n_p, 1.00),
        )

    # Phase A: cheap sub 0→1
    ra = build_and_solve_sp(_inst(0.10, 0.90), Y, B, fix_first_stage=fix,
                            relax_first_stage_constraints=True)
    assert ra["r_sub"][0, 0, 0, 1] == 1, f"Phase A: r_sub[ω0,i0,src0,dst1] must fire"
    assert ra["r_sub"][0, 0, 1, 0] == 0, f"Phase A: reverse direction must not fire"
    assert ra["r_remfg"][0, 0, 0] == 0,  f"Phase A: r_remfg[ω0,i0,m0] must not fire"
    assert abs(ra["expected_stage2_cost"] - (0.005 + 0.10)) < 1e-3

    # Phase B: cheap ρ_sub[1,0] forbidden by (10b); re-mfg wins
    rb = build_and_solve_sp(_inst(0.90, 0.10), Y, B, fix_first_stage=fix,
                            relax_first_stage_constraints=True)
    assert rb["r_remfg"][0, 0, 0] == 1, f"Phase B: r_remfg[ω0,i0,m0] must fire"
    assert rb["r_sub"][0].sum() == 0,   f"Phase B: no subcontract should fire"
    assert abs(rb["expected_stage2_cost"] - (0.005 + 0.20)) < 1e-3

    return True, (
        f"A: sub[0→1] at {ra['expected_stage2_cost']:.4f} (t={ra['solve_time']:.3f}s); "
        f"B: remfg[m0] at {rb['expected_stage2_cost']:.4f} (t={rb['solve_time']:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Test 4 — Tier-aware cancellation pricing
# ---------------------------------------------------------------------------

def test_tier_cancellation():
    """
    All batches fail and all patients are ineligible (B=0), forcing cancellation.
    Run A: ρ_cancel=[1.50, 1.00, 0.50] (tiers H, M, L) → recourse=3.00.
    Run B: ρ_cancel=[0.50, 0.50, 0.50] (all tier L) → recourse=1.50.
    Catches bugs where ρ_cancel is read with a fixed tier rather than per-patient.
    """
    n_p, n_f = 3, 1
    Y = np.zeros((1, n_p, n_f))
    B = np.zeros((1, n_p))
    fix = {"z": {0: 1}, "C": {0: 3}, "x": {(i, 0): 1 for i in range(n_p)}}

    def _inst(rho_cancel_vec):
        return Instance(
            n_patients=n_p, n_facilities=n_f,
            f=np.array([1.0]), pi=np.array([0.05]), c=np.array([0.20]),
            s_max=np.array([5]), p=np.array([0.90]), beta=np.full(n_p, 0.80),
            rho_leuk=0.005, rho_remfg=np.array([0.20]),
            rho_sub=np.zeros((1, 1)), rho_cancel=np.array(rho_cancel_vec),
        )

    # Run A: H+M+L tiers
    ra = build_and_solve_sp(_inst([1.50, 1.00, 0.50]), Y, B, fix_first_stage=fix)
    assert ra["r_cancel"][0].sum() == 3, f"Run A: all 3 must cancel"
    assert abs(ra["expected_stage2_cost"] - 3.00) < 1e-3, (
        f"Run A recourse {ra['expected_stage2_cost']:.5f} != 3.00"
    )

    # Run B: all tier L
    rb = build_and_solve_sp(_inst([0.50, 0.50, 0.50]), Y, B, fix_first_stage=fix)
    assert rb["r_cancel"][0].sum() == 3, f"Run B: all 3 must cancel"
    assert abs(rb["expected_stage2_cost"] - 1.50) < 1e-3, (
        f"Run B recourse {rb['expected_stage2_cost']:.5f} != 1.50"
    )

    assert ra["expected_stage2_cost"] > rb["expected_stage2_cost"], (
        "Mixed tiers must cost more than all-L"
    )

    return True, (
        f"A(H+M+L): recourse={ra['expected_stage2_cost']:.4f} (t={ra['solve_time']:.3f}s); "
        f"B(all-L): recourse={rb['expected_stage2_cost']:.4f} (t={rb['solve_time']:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Test 1 — Eligibility filter binding (constraint 10)", test_eligibility_filter),
    ("Test 2 — Capacity slack tight binding (constraint 9)", test_capacity_slack_binding),
    ("Test 3 — Subcontract directional asymmetry",          test_subcontract_asymmetry),
    ("Test 4 — Tier-aware cancellation pricing",            test_tier_cancellation),
]


def main():
    print("=== Tier 1 adversarial tests ===")
    passed = 0
    for name, fn in TESTS:
        try:
            ok, detail = fn()
            print(f"[PASS] {name}")
            print(f"       {detail}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {name}")
            print(f"       {exc}")
        except Exception as exc:
            print(f"[FAIL] {name}")
            print(f"       Unexpected error: {exc}")
    print(f"{passed} / {len(TESTS)} passed.")
    if passed < len(TESTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
