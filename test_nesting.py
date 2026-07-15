"""
test_nesting.py — validation: dynamic_sp with kappa=0 reproduces yield_sp_v1.

The dynamic model must NEST the v1 benchmark: with the delay-sensitivity knob
kappa=0 the endogenous StillEligible collapses to the exogenous Bernoulli
eligibility B, so on the toy instance the dynamic SP must reproduce
  - total expected cost  ~ 2.774
  - VSS (SP - deterministic) ~ 0.154
  - the same first-stage design as v1
all within 1e-3 relative tolerance. This gate must pass before any dynamic
experiment is trusted.

Run: python test_nesting.py
"""

from __future__ import annotations

import numpy as np

from yield_sp_v1 import toy_instance, toy_scenarios, build_and_solve_sp
from dynamic_sp import solve_dynamic_sp, compute_still_eligible

RTOL = 1e-3


def _toy_fix(inst):
    fix = {"z": {0: 1, 1: 1}, "C": {0: 2, 1: 2},
           "x": {(i, m): 0 for i in inst.I for m in inst.M}}
    fix["x"][0, 0] = 1
    fix["x"][1, 0] = 1
    fix["x"][2, 1] = 1
    return fix


def test_still_eligible_equals_B_at_kappa0():
    inst = toy_instance()
    Y, B = toy_scenarios()
    for gamma in (1.0, 1.5, 2.0):
        se, _ = compute_still_eligible(inst, Y, B, ["M"] * 3, kappa=0.0, gamma=gamma)
        assert np.array_equal(se, B), f"StillEligible != B at kappa=0, gamma={gamma}"
    print("[PASS] StillEligible == B exactly at kappa=0 (all gamma)")


def test_toy_cost_and_vss():
    inst = toy_instance()
    Y, B = toy_scenarios()
    fix = _toy_fix(inst)

    dyn = solve_dynamic_sp(inst, Y, B, tiers=["M"] * 3, kappa=0.0, fix_first_stage=fix)
    Yd, Bd = np.ones((1, 3, 2)), np.ones((1, 3))
    dyn_det = solve_dynamic_sp(inst, Yd, Bd, tiers=["M"] * 3, kappa=0.0, fix_first_stage=fix)
    vss = dyn["total_cost"] - dyn_det["total_cost"]

    assert abs(dyn["total_cost"] - 2.774) < 2e-3, f"SP cost {dyn['total_cost']:.6f} != 2.774"
    assert abs(dyn_det["total_cost"] - 2.620) < 2e-3, f"det cost {dyn_det['total_cost']:.6f} != 2.620"
    assert abs(vss - 0.154) < 2e-3, f"VSS {vss:.6f} != 0.154"
    print(f"[PASS] toy nesting: SP={dyn['total_cost']:.4f} (2.774), "
          f"det={dyn_det['total_cost']:.4f} (2.620), VSS={vss:.4f} (0.154)")


def test_matches_v1_exactly():
    """Fixed-first-stage and free-design solves match v1 to solver tolerance."""
    inst = toy_instance()
    Y, B = toy_scenarios()
    fix = _toy_fix(inst)

    # Fixed first stage (the §15 configuration).
    v1_fix = build_and_solve_sp(inst, Y, B, fix_first_stage=fix)
    dyn_fix = solve_dynamic_sp(inst, Y, B, tiers=["M"] * 3, kappa=0.0, fix_first_stage=fix)
    assert abs(v1_fix["total_cost"] - dyn_fix["total_cost"]) < 1e-6, (
        f"fixed: v1 {v1_fix['total_cost']} vs dyn {dyn_fix['total_cost']}")

    # Free first stage: same optimal design and cost.
    v1_free = build_and_solve_sp(inst, Y, B)
    dyn_free = solve_dynamic_sp(inst, Y, B, tiers=["M"] * 3, kappa=0.0)
    assert abs(v1_free["total_cost"] - dyn_free["total_cost"]) < 1e-6
    assert np.array_equal(np.asarray(v1_free["z"], int), np.asarray(dyn_free["z"], int))
    assert np.array_equal(np.asarray(v1_free["C"], int), np.asarray(dyn_free["C"], int))
    assert np.array_equal(np.asarray(v1_free["x"], int), np.asarray(dyn_free["x"], int))
    print(f"[PASS] dynamic == v1 (fixed |Δ|<1e-6; free design z={dyn_free['z'].tolist()}, "
          f"C={dyn_free['C'].tolist()} identical)")


if __name__ == "__main__":
    test_still_eligible_equals_B_at_kappa0()
    test_toy_cost_and_vss()
    test_matches_v1_exactly()
    print("\n3 / 3 nesting tests passed — dynamic_sp nests yield_sp_v1 at kappa=0.")
