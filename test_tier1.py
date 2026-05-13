"""
test_tier1.py — Adversarial Tier 1 tests for yield_sp_v1.py.
Exercises constraints (9), (10), asymmetric subcontracting, and tier-dependent
cancellation pricing with focused hand-crafted instances.
Run: python test_tier1.py
"""

import time
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from yield_sp_v1 import Instance


# ---------------------------------------------------------------------------
# Internal solver that returns variable values in addition to costs.
# When fix_first_stage is provided, first-stage constraints (1)–(4) are
# omitted: Stage-1 is treated as an external parameter, not optimised, so
# the capacity-assignment constraint (3) need not hold for the given fix.
# ---------------------------------------------------------------------------

def _solve_detail(
    instance: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    fix_first_stage: Optional[dict] = None,
):
    """
    Build and solve the two-stage SP; return costs and solved variable values.

    Returns
    -------
    result : dict  — obj, stage1, stage2, solve_time
    rv_cancel : dict  (i, o) -> 0|1
    rv_remfg  : dict  (i, m, o) -> 0|1
    rv_sub    : dict  (i, m, mp, o) -> 0|1   (m != mp only)
    """
    N = Y.shape[0]
    I_set = list(instance.I)
    M_set = list(instance.M)
    O_set = list(range(N))
    sub_pairs = [(m, mp) for m in M_set for mp in M_set if m != mp]

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-4
    model.Params.TimeLimit = 300

    z = model.addVars(M_set, vtype=GRB.BINARY, name="z")
    C = model.addVars(M_set, vtype=GRB.INTEGER, lb=0, name="C")
    x = model.addVars(I_set, M_set, vtype=GRB.BINARY, name="x")

    if fix_first_stage is not None:
        # Pin Stage-1 via variable bounds; skip constraints (1)-(4) so that
        # adversarial configurations (e.g. C[0]=1 with 3 patients at m0) remain
        # feasible for second-stage testing.
        for m in M_set:
            z[m].lb = z[m].ub = fix_first_stage["z"][m]
            C[m].lb = C[m].ub = fix_first_stage["C"][m]
        for i in I_set:
            for m in M_set:
                x[i, m].lb = x[i, m].ub = fix_first_stage["x"][i, m]
    else:
        for i in I_set:
            model.addConstr(x.sum(i, "*") == 1)            # (1)
            for m in M_set:
                model.addConstr(x[i, m] <= z[m])           # (2)
        for m in M_set:
            model.addConstr(x.sum("*", m) <= C[m])         # (3)
            model.addConstr(C[m] <= instance.s_max[m] * z[m])  # (4)

    r_remfg = model.addVars(I_set, M_set, O_set, vtype=GRB.BINARY, name="r_remfg")
    r_sub = model.addVars(
        [(i, m, mp, o) for i in I_set for m, mp in sub_pairs for o in O_set],
        vtype=GRB.BINARY,
        name="r_sub",
    )
    r_cancel = model.addVars(I_set, O_set, vtype=GRB.BINARY, name="r_cancel")

    for o in O_set:
        for i in I_set:
            F_i = gp.quicksum(x[i, m] * (1.0 - Y[o, i, m]) for m in M_set)
            remfg_i = r_remfg.sum(i, "*", o)
            sub_i = gp.quicksum(r_sub[i, m, mp, o] for m, mp in sub_pairs)
            model.addConstr(remfg_i + sub_i + r_cancel[i, o] == F_i)   # (8)
            model.addConstr(remfg_i + sub_i <= B[o, i])                 # (10)
            for m in M_set:                                              # (10b)
                sub_from_m = gp.quicksum(r_sub[i, m, mp, o] for mp in M_set if mp != m)
                model.addConstr(r_remfg[i, m, o] + sub_from_m <= x[i, m])
        for m in M_set:
            slack = C[m] - gp.quicksum(x[i, m] * Y[o, i, m] for i in I_set)
            incoming = gp.quicksum(
                r_sub[i, mp, m, o] for i in I_set for mp in M_set if mp != m
            )
            model.addConstr(r_remfg.sum("*", m, o) + incoming <= slack)  # (9)

    s1_expr = (
        gp.quicksum(instance.f[m] * z[m] for m in M_set)
        + gp.quicksum(instance.pi[m] * C[m] for m in M_set)
        + gp.quicksum(instance.c[m] * x[i, m] for i in I_set for m in M_set)
    )
    s2_expr = gp.LinExpr()
    for o in O_set:
        for i in I_set:
            s2_expr += instance.rho_cancel[i] * r_cancel[i, o]
            for m in M_set:
                s2_expr += (instance.rho_leuk + instance.rho_remfg[m]) * r_remfg[i, m, o]
                for mp in M_set:
                    if mp != m:
                        s2_expr += (
                            instance.rho_leuk + instance.rho_sub[m, mp]
                        ) * r_sub[i, m, mp, o]

    model.setObjective(s1_expr + (1.0 / N) * s2_expr, GRB.MINIMIZE)

    t0 = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t0

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi status {model.Status} — infeasible or error")

    obj = model.ObjVal
    s1_val = (
        sum(instance.f[m] * z[m].X for m in M_set)
        + sum(instance.pi[m] * C[m].X for m in M_set)
        + sum(instance.c[m] * x[i, m].X for i in I_set for m in M_set)
    )

    rv_cancel = {(i, o): round(r_cancel[i, o].X) for i in I_set for o in O_set}
    rv_remfg = {
        (i, m, o): round(r_remfg[i, m, o].X)
        for i in I_set for m in M_set for o in O_set
    }
    rv_sub = {
        (i, m, mp, o): round(r_sub[i, m, mp, o].X)
        for i in I_set for m, mp in sub_pairs for o in O_set
    }

    result = {"obj": obj, "stage1": s1_val, "stage2": obj - s1_val, "solve_time": solve_time}
    return result, rv_cancel, rv_remfg, rv_sub


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
    Y = np.zeros((1, n_p, n_f))            # all batches fail
    B = np.array([[1.0, 1.0, 0.0]])        # patient 2 ineligible

    fix = {
        "z": {0: 1},
        "C": {0: 3},
        "x": {(i, 0): 1 for i in range(n_p)},
    }
    result, rv_cancel, rv_remfg, rv_sub = _solve_detail(inst, Y, B, fix_first_stage=fix)

    n_remfg = sum(rv_remfg.values())
    n_cancel = sum(rv_cancel.values())

    assert rv_cancel[(2, 0)] == 1, (
        f"Patient 2 (ineligible) must cancel; got r_cancel={(rv_cancel)}"
    )
    assert rv_cancel[(0, 0)] == 0, (
        f"Patient 0 (eligible) should not cancel; got {rv_cancel[(0, 0)]}"
    )
    assert rv_cancel[(1, 0)] == 0, (
        f"Patient 1 (eligible) should not cancel; got {rv_cancel[(1, 0)]}"
    )
    assert n_remfg == 2, f"Expected 2 re-manufactures, got {n_remfg}"

    # Recourse: 2×(0.005+0.20) + 1×1.00 = 1.41
    expected_q = 2 * (0.005 + 0.20) + 1.00
    assert abs(result["stage2"] - expected_q) < 1e-3, (
        f"Recourse cost {result['stage2']:.5f} != expected {expected_q:.5f}"
    )

    return True, (
        f"r_cancel={[rv_cancel[(i,0)] for i in range(n_p)]}, "
        f"r_remfg_total={n_remfg}, recourse={result['stage2']:.4f} "
        f"(t={result['solve_time']:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Test 2 — Capacity slack tight binding (constraint 9)
# ---------------------------------------------------------------------------

def test_capacity_slack_binding():
    """
    Three patients all fail at facility 0; C[0]=1 gives slack=1, so constraint (9)
    permits exactly 1 re-manufacture at m0. Constraint (10b) prevents re-manufacturing
    at m1 for patients assigned to m0, so the remaining 2 failures must subcontract
    m0→m1 (cheaper than cancellation). Uses symmetric ρ_remfg=[0.20,0.20].

    Stage-1 is pinned with C[0]=1 but 3 assignments — intentionally violates
    constraint (3), which is why _solve_detail drops (3) when Stage-1 is fixed.
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

    # C[0]=1: only 1 capacity slot at m0; C[1]=5: plenty of room for subs
    fix = {
        "z": {0: 1, 1: 1},
        "C": {0: 1, 1: 5},
        "x": {(i, 0): 1 for i in range(n_p)} | {(i, 1): 0 for i in range(n_p)},
    }
    result, rv_cancel, rv_remfg, rv_sub = _solve_detail(inst, Y, B, fix_first_stage=fix)

    n_remfg = sum(rv_remfg.values())
    n_sub = sum(rv_sub.values())
    n_cancel = sum(rv_cancel.values())

    # Cost ranking: remfg@m0=0.205 < sub@m0→m1=0.255 < cancel=1.00
    # re-mfg@m1 forbidden by (10b); sub@m1→m0 forbidden by (10b).
    # With slack=1 at m0: 1 remfg@m0 + 2 sub@m0→m1 = 0.205+2×0.255=0.715 is optimal.
    assert n_remfg == 1, (
        f"Capacity=1 at m0 + constraint (10b) → exactly 1 re-mfg at m0, "
        f"got total remfg={n_remfg}\n  rv_remfg={rv_remfg}"
    )
    assert n_sub == 2, (
        f"Remaining 2 failures must subcontract m0→m1 (sub < cancel), got {n_sub}\n"
        f"  rv_sub={rv_sub}"
    )
    assert n_cancel == 0, (
        f"No cancellations expected, got {n_cancel}; rv_cancel={rv_cancel}"
    )

    # Recourse: 1×(0.005+0.20) + 2×(0.005+0.25) = 0.205 + 0.510 = 0.715
    expected_q = 1 * (0.005 + 0.20) + 2 * (0.005 + 0.25)
    assert abs(result["stage2"] - expected_q) < 1e-3, (
        f"Recourse cost {result['stage2']:.5f} != expected {expected_q:.5f}"
    )

    return True, (
        f"remfg={n_remfg}, sub={n_sub}, cancel={n_cancel}, "
        f"recourse={result['stage2']:.4f} (t={result['solve_time']:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Test 3 — Subcontract directional asymmetry
# ---------------------------------------------------------------------------

def test_subcontract_asymmetry():
    """
    1 patient at m0, batch fails, B=1. C[0]=1 (slack=1), C[1]=5.
    ρ_remfg=[0.20,0.20] → re-mfg cost 0.205.

    Phase A: ρ_sub[0,1]=0.10 (cheap), ρ_sub[1,0]=0.90 (expensive).
      Cheapest allowed recourse: sub 0→1 at 0.115. Assert r_sub[0,0,1] fires.

    Phase B: swap costs — ρ_sub[0,1]=0.90, ρ_sub[1,0]=0.10.
      Constraint (10b) forbids r_sub[0,1,0] (x[0,1]=0 → source m1 not allowed).
      Only options: sub 0→1 at 0.905, re-mfg at m0 at 0.205, cancel at 1.00.
      Re-mfg wins. Assert r_remfg[0,0] fires and no subcontract fires.
    """
    n_p, n_f = 1, 2
    Y = np.array([[[0.0, 1.0]]])   # patient 0 fails at m0
    B = np.array([[1.0]])

    fix = {
        "z": {0: 1, 1: 1},
        "C": {0: 1, 1: 5},
        "x": {(0, 0): 1, (0, 1): 0},
    }

    def _make_inst(sub_0_to_1, sub_1_to_0):
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

    # Phase A: cheap sub 0→1 wins (0.115 < re-mfg 0.205)
    inst_a = _make_inst(sub_0_to_1=0.10, sub_1_to_0=0.90)
    res_a, rc_a, rr_a, rs_a = _solve_detail(inst_a, Y, B, fix_first_stage=fix)

    assert rs_a[(0, 0, 1, 0)] == 1, (
        f"Phase A: r_sub[0,0,1] (ρ_sub[0,1]=0.10) must fire; got {rs_a}"
    )
    assert rs_a[(0, 1, 0, 0)] == 0, (
        f"Phase A: r_sub[0,1,0] must NOT fire (ρ_sub[1,0]=0.90 expensive); got {rs_a}"
    )
    assert rr_a[(0, 0, 0)] == 0, f"Phase A: r_remfg[0,0] must not fire; got {rr_a}"
    expected_q_a = 0.005 + 0.10
    assert abs(res_a["stage2"] - expected_q_a) < 1e-3, (
        f"Phase A recourse {res_a['stage2']:.5f} != {expected_q_a:.5f}"
    )

    # Phase B: ρ_sub[0,1]=0.90 expensive; ρ_sub[1,0]=0.10 cheap but forbidden by (10b)
    # since x[0,1]=0. Re-mfg at m0 (0.205) is the cheapest allowed action.
    inst_b = _make_inst(sub_0_to_1=0.90, sub_1_to_0=0.10)
    res_b, rc_b, rr_b, rs_b = _solve_detail(inst_b, Y, B, fix_first_stage=fix)

    assert rr_b[(0, 0, 0)] == 1, (
        f"Phase B: r_remfg[0,0] must fire (sub 0→1=0.905 > remfg=0.205; "
        f"sub 1→0 forbidden by 10b); got {rr_b}"
    )
    assert sum(rs_b.values()) == 0, (
        f"Phase B: no subcontract should fire; got {rs_b}"
    )
    expected_q_b = 0.005 + 0.20
    assert abs(res_b["stage2"] - expected_q_b) < 1e-3, (
        f"Phase B recourse {res_b['stage2']:.5f} != {expected_q_b:.5f}"
    )

    return True, (
        f"A: sub[0→1] fires at {res_a['stage2']:.4f} (t={res_a['solve_time']:.3f}s); "
        f"B: remfg[m0] fires at {res_b['stage2']:.4f} (t={res_b['solve_time']:.3f}s)"
    )


# ---------------------------------------------------------------------------
# Test 4 — Tier-aware cancellation pricing
# ---------------------------------------------------------------------------

def test_tier_cancellation():
    """
    All batches fail and all patients are ineligible (B=0), forcing cancellation.
    Run A: tiers H+M+L with ρ_cancel=[1.50, 1.00, 0.50] → recourse=3.00.
    Run B: all tier L with ρ_cancel=[0.50, 0.50, 0.50] → recourse=1.50.
    Catches bugs where ρ_cancel_u(i) is read with a fixed tier rather than per-patient.
    """
    n_p, n_f = 3, 1
    Y = np.zeros((1, n_p, n_f))   # all fail
    B = np.zeros((1, n_p))        # all ineligible → cancellation mandatory

    fix = {
        "z": {0: 1},
        "C": {0: 3},
        "x": {(i, 0): 1 for i in range(n_p)},
    }

    def _make_inst(rho_cancel_vec):
        return Instance(
            n_patients=n_p, n_facilities=n_f,
            f=np.array([1.0]), pi=np.array([0.05]), c=np.array([0.20]),
            s_max=np.array([5]), p=np.array([0.90]), beta=np.full(n_p, 0.80),
            rho_leuk=0.005, rho_remfg=np.array([0.20]),
            rho_sub=np.zeros((1, 1)), rho_cancel=np.array(rho_cancel_vec),
        )

    # Run A: mixed tiers (H=1.50, M=1.00, L=0.50)
    inst_a = _make_inst([1.50, 1.00, 0.50])
    res_a, rc_a, rr_a, rs_a = _solve_detail(inst_a, Y, B, fix_first_stage=fix)

    n_cancel_a = sum(rc_a.values())
    assert n_cancel_a == 3, f"Run A: all 3 must cancel, got {n_cancel_a}"
    expected_q_a = 1.50 + 1.00 + 0.50
    assert abs(res_a["stage2"] - expected_q_a) < 1e-3, (
        f"Run A recourse {res_a['stage2']:.5f} != {expected_q_a:.5f}"
    )

    # Run B: all tier L
    inst_b = _make_inst([0.50, 0.50, 0.50])
    res_b, rc_b, rr_b, rs_b = _solve_detail(inst_b, Y, B, fix_first_stage=fix)

    n_cancel_b = sum(rc_b.values())
    assert n_cancel_b == 3, f"Run B: all 3 must cancel, got {n_cancel_b}"
    expected_q_b = 3 * 0.50
    assert abs(res_b["stage2"] - expected_q_b) < 1e-3, (
        f"Run B recourse {res_b['stage2']:.5f} != {expected_q_b:.5f}"
    )

    # Sanity: mixed tiers must cost strictly more than all-L
    assert res_a["stage2"] > res_b["stage2"], (
        f"Mixed tiers ({res_a['stage2']:.4f}) should exceed all-L ({res_b['stage2']:.4f})"
    )

    return True, (
        f"A(H+M+L): recourse={res_a['stage2']:.4f} (t={res_a['solve_time']:.3f}s); "
        f"B(all-L): recourse={res_b['stage2']:.4f} (t={res_b['solve_time']:.3f}s)"
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
