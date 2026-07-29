"""
yield_sp_v1.py — v1 two-stage stochastic program for CAR-T yield uncertainty.

The worked example is self-contained here: `toy_instance()` / `toy_scenarios()`
plus the verification block at the bottom. ("§15" tags below refer to the
retired Yield_Model_Formulation.md note that this example originally came from;
the numbers it asserts are reproduced in-file and are also the nesting gate in
`test_nesting.py`.) The model as the paper states it lives in
`dynamic_sp.py`; every published number is traceable in
`paper/results/numbers.md`.

Run: python yield_sp_v1.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class Instance:
    """Parameters for the CAR-T yield-uncertainty SP (§4.2–§4.4)."""

    n_patients: int
    n_facilities: int
    f: np.ndarray         # [m] facility opening cost
    pi: np.ndarray        # [m] per-slot capacity cost
    c: np.ndarray         # [m] per-batch primary manufacturing cost
    s_max: np.ndarray     # [m] maximum capacity slots
    p: np.ndarray         # [m] per-batch yield probability
    beta: np.ndarray      # [i] re-collection feasibility probability (tier-dependent)
    rho_leuk: float       # re-leukapheresis cost
    rho_remfg: np.ndarray # [m] re-manufacturing cost
    rho_sub: np.ndarray   # [m, m'] subcontracting cost
    rho_cancel: np.ndarray  # [i] cancellation clinical loss (tier-dependent)
    mnf: Optional[int] = None  # max number of facilities open (MNF); None = no cap

    @property
    def I(self):
        return range(self.n_patients)

    @property
    def M(self):
        return range(self.n_facilities)


def toy_instance() -> Instance:
    """§15 toy instance: 3 patients (all tier M), 2 facilities."""
    n_p, n_f = 3, 2
    rho_sub = np.zeros((n_f, n_f))
    rho_sub[0, 1] = rho_sub[1, 0] = 0.25
    return Instance(
        n_patients=n_p,
        n_facilities=n_f,
        f=np.array([1.0, 0.8]),
        pi=np.array([0.05, 0.06]),
        c=np.array([0.20, 0.20]),
        s_max=np.array([5, 5]),
        p=np.array([0.90, 0.85]),
        beta=np.full(n_p, 0.80),
        rho_leuk=0.005,
        rho_remfg=np.array([0.20, 0.20]),
        rho_sub=rho_sub,
        rho_cancel=np.full(n_p, 1.00),
    )


def toy_scenarios():
    """Four hand-crafted scenarios from §15 table. Returns Y[ω,i,m], B[ω,i]."""
    Y = np.array(
        [
            [[1, 1], [1, 1], [1, 1]],  # ω1: all batches succeed
            [[1, 0], [1, 1], [0, 1]],  # ω2: failures not at assigned facilities
            [[0, 1], [1, 0], [1, 1]],  # ω3: patient 1 fails at m1
            [[0, 1], [0, 1], [0, 1]],  # ω4: patients 1 & 2 fail at m1
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    return Y, B


def sample_scenarios(instance: Instance, n: int, seed: int):
    """Draw n independent Bernoulli scenarios for Y[ω,i,m] and B[ω,i]."""
    rng = np.random.default_rng(seed)
    Y = rng.binomial(
        1,
        instance.p[np.newaxis, np.newaxis, :],
        size=(n, instance.n_patients, instance.n_facilities),
    ).astype(float)
    B = rng.binomial(
        1,
        instance.beta[np.newaxis, :],
        size=(n, instance.n_patients),
    ).astype(float)
    return Y, B


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_sp_model(
    instance: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    *,
    fix_first_stage: Optional[dict] = None,
    relax_first_stage_constraints: bool = False,
) -> tuple[gp.Model, dict]:
    """
    Construct the Gurobi model for the two-stage SP and return it with variable handles.
    Does NOT call optimize().

    Parameters
    ----------
    instance : Instance
    Y : (N, I, M) yield indicators per scenario
    B : (N, I) re-collection eligibility indicators per scenario
    fix_first_stage : {'z': {m: val}, 'C': {m: val}, 'x': {(i,m): val}}
        Pin z, C, x to specified values via variable bounds.
    relax_first_stage_constraints : bool
        When True, omit constraints (1)–(4). Used by tests that pin Stage-1 to
        a configuration that would otherwise violate constraint (3), e.g. Test 2
        of the Tier 1 suite (C[0]=1 with 3 patients assigned to m0).

    Returns
    -------
    model : gp.Model  — all constraints and objective set, not yet solved
    variables : dict with keys 'z', 'C', 'x', 'r_remfg', 'r_sub', 'r_cancel'
                each a gurobipy tupledict keyed by the natural index tuples.
                r_remfg keyed by (i, m, o); r_sub by (i, m, mp, o) with m≠mp;
                r_cancel by (i, o).
    """
    N = Y.shape[0]
    I_set = list(instance.I)
    M_set = list(instance.M)
    O_set = list(range(N))
    sub_pairs = [(m, mp) for m in M_set for mp in M_set if m != mp]

    model = gp.Model("yield_sp")
    model.Params.OutputFlag = 0

    # --- First-stage variables ---
    z = model.addVars(M_set, vtype=GRB.BINARY, name="z")
    C = model.addVars(M_set, vtype=GRB.INTEGER, lb=0, name="C")
    x = model.addVars(I_set, M_set, vtype=GRB.BINARY, name="x")

    if fix_first_stage is not None:
        for m in M_set:
            z[m].lb = z[m].ub = fix_first_stage["z"][m]
            C[m].lb = C[m].ub = fix_first_stage["C"][m]
        for i in I_set:
            for m in M_set:
                x[i, m].lb = x[i, m].ub = fix_first_stage["x"][i, m]

    # --- Second-stage variables ---
    r_remfg = model.addVars(I_set, M_set, O_set, vtype=GRB.BINARY, name="r_remfg")
    r_sub = model.addVars(
        [(i, m, mp, o) for i in I_set for m, mp in sub_pairs for o in O_set],
        vtype=GRB.BINARY,
        name="r_sub",
    )
    r_cancel = model.addVars(I_set, O_set, vtype=GRB.BINARY, name="r_cancel")

    # --- First-stage constraints (omitted when relax_first_stage_constraints=True) ---
    if not relax_first_stage_constraints:
        for i in I_set:
            model.addConstr(x.sum(i, "*") == 1, name=f"assign_{i}")           # (1)
            for m in M_set:
                model.addConstr(x[i, m] <= z[m], name=f"open_{i}_{m}")        # (2)
        for m in M_set:
            model.addConstr(x.sum("*", m) <= C[m], name=f"cap_{m}")           # (3)
            model.addConstr(C[m] <= instance.s_max[m] * z[m], name=f"smax_{m}")  # (4)

    # --- Second-stage constraints per scenario ---
    for o in O_set:
        for i in I_set:
            # Failure indicator F_i(ω) inlined as LinExpr (eq. 18)
            F_i = gp.quicksum(x[i, m] * (1.0 - Y[o, i, m]) for m in M_set)
            remfg_i = r_remfg.sum(i, "*", o)
            sub_i = gp.quicksum(r_sub[i, m, mp, o] for m, mp in sub_pairs)

            # (8) Every failed batch triggers exactly one recourse action
            model.addConstr(
                remfg_i + sub_i + r_cancel[i, o] == F_i,
                name=f"recourse_{i}_{o}",
            )
            # (10) Re-collection requires clinical eligibility
            model.addConstr(
                remfg_i + sub_i <= B[o, i],
                name=f"elig_{i}_{o}",
            )
            # (10b) Recourse source facility tied to primary assignment
            for m in M_set:
                sub_from_m = gp.quicksum(
                    r_sub[i, m, mp, o] for mp in M_set if mp != m
                )
                model.addConstr(
                    r_remfg[i, m, o] + sub_from_m <= x[i, m],
                    name=f"src_{i}_{m}_{o}",
                )

        for m in M_set:
            # Recourse-capacity slack: unused contracted capacity at m after the
            # successful primary batches have consumed their slots (eq. 19):
            #   slack = C_m − Σ_i x_im·Y_im
            slack = (
                C[m] - gp.quicksum(x[i, m] * Y[o, i, m] for i in I_set)
            )
            # Incoming subcontracts to m from any other facility mp
            incoming = gp.quicksum(
                r_sub[i, mp, m, o] for i in I_set for mp in M_set if mp != m
            )
            # (9) Re-manufactures + incoming subcontracts fit in unused capacity
            model.addConstr(
                r_remfg.sum("*", m, o) + incoming <= slack,
                name=f"capslack_{m}_{o}",
            )

    # --- Objective (P): Stage-1 cost + (1/N) * expected Stage-2 cost ---
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

    variables = {
        "z": z, "C": C, "x": x,
        "r_remfg": r_remfg, "r_sub": r_sub, "r_cancel": r_cancel,
    }
    return model, variables


def solve_and_extract(
    model: gp.Model,
    variables: dict,
    instance: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    *,
    time_limit: float = 600.0,
    mip_gap: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """
    Set Gurobi parameters, solve the model, and extract results into a dict.

    Recourse-variable arrays follow the (scenario, patient, ...) axis convention
    matching the input Y[ω,i,m] and B[ω,i] shapes:
      r_remfg  : (N, I, M)    — r_remfg[ω, i, m]
      r_sub    : (N, I, M, M) — r_sub[ω, i, src_m, dst_m]; diagonal is always 0
      r_cancel : (N, I)       — r_cancel[ω, i]

    Returns dict with keys: total_cost, stage1_cost, expected_stage2_cost,
    z, C, x, r_remfg, r_sub, r_cancel, solve_time, gap.
    """
    model.Params.MIPGap = mip_gap
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = int(verbose)

    t0 = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t0

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Solver status {model.Status} — no feasible solution found.")

    N = Y.shape[0]
    I_list = list(instance.I)
    M_list = list(instance.M)
    sub_pairs = [(m, mp) for m in M_list for mp in M_list if m != mp]

    z_v, C_v, x_v = variables["z"], variables["C"], variables["x"]
    r_remfg_v, r_sub_v, r_cancel_v = (
        variables["r_remfg"], variables["r_sub"], variables["r_cancel"]
    )

    # Stage-1 cost from solved variable values
    s1_val = (
        sum(instance.f[m] * z_v[m].X for m in M_list)
        + sum(instance.pi[m] * C_v[m].X for m in M_list)
        + sum(instance.c[m] * x_v[i, m].X for i in I_list for m in M_list)
    )

    # Extract first-stage arrays
    z_arr = np.array([z_v[m].X for m in M_list])
    C_arr = np.array([round(C_v[m].X) for m in M_list])
    x_arr = np.array([[round(x_v[i, m].X) for m in M_list] for i in I_list])

    # Extract second-stage arrays — axes: (scenario, patient, ...)
    I, M = instance.n_patients, instance.n_facilities
    r_remfg_arr = np.zeros((N, I, M))
    r_sub_arr = np.zeros((N, I, M, M))
    r_cancel_arr = np.zeros((N, I))

    for o in range(N):
        for i in I_list:
            r_cancel_arr[o, i] = round(r_cancel_v[i, o].X)
            for m in M_list:
                r_remfg_arr[o, i, m] = round(r_remfg_v[i, m, o].X)
        for i in I_list:
            for m, mp in sub_pairs:
                r_sub_arr[o, i, m, mp] = round(r_sub_v[i, m, mp, o].X)

    obj = model.ObjVal
    achieved_gap = abs(obj - model.ObjBound) / max(1e-10, abs(obj))

    return {
        "total_cost": obj,
        "stage1_cost": s1_val,
        "expected_stage2_cost": obj - s1_val,
        "z": z_arr,
        "C": C_arr,
        "x": x_arr,
        "r_remfg": r_remfg_arr,
        "r_sub": r_sub_arr,
        "r_cancel": r_cancel_arr,
        "solve_time": solve_time,
        "gap": achieved_gap,
    }


def build_and_solve_sp(
    instance: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    *,
    fix_first_stage: Optional[dict] = None,
    relax_first_stage_constraints: bool = False,
    time_limit: float = 600.0,
    mip_gap: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """Thin wrapper: build → solve → extract. Single entry point for callers."""
    model, variables = build_sp_model(
        instance, Y, B,
        fix_first_stage=fix_first_stage,
        relax_first_stage_constraints=relax_first_stage_constraints,
    )
    return solve_and_extract(
        model, variables, instance, Y, B,
        time_limit=time_limit, mip_gap=mip_gap, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# §15 verification
# ---------------------------------------------------------------------------

def verify_toy_example():
    """Reproduce §15 worked example and assert cost targets."""
    inst = toy_instance()
    Y, B = toy_scenarios()

    # §15 stipulated Stage-1: z=(1,1), C=(2,2), patients 1&2 → m1, patient 3 → m2
    fix = {
        "z": {0: 1, 1: 1},
        "C": {0: 2, 1: 2},
        "x": {(i, m): 0 for i in inst.I for m in inst.M},
    }
    fix["x"][0, 0] = 1  # patient 1 → m1
    fix["x"][1, 0] = 1  # patient 2 → m1
    fix["x"][2, 1] = 1  # patient 3 → m2

    sp = build_and_solve_sp(inst, Y, B, fix_first_stage=fix)

    # Deterministic baseline: single all-success scenario (Y=1, B=1 everywhere)
    Y_det = np.ones((1, inst.n_patients, inst.n_facilities))
    B_det = np.ones((1, inst.n_patients))
    det = build_and_solve_sp(inst, Y_det, B_det, fix_first_stage=fix)

    vss_gap = sp["total_cost"] - det["total_cost"]
    print("=== Toy instance verification (§15 worked example) ===")
    print(f"Stochastic SP cost:            {sp['total_cost']:.4f}")
    print(f"  Stage-1 cost:                {sp['stage1_cost']:.4f}")
    print(f"  Expected Stage-2 cost:       {sp['expected_stage2_cost']:.4f}")
    print(f"Deterministic baseline cost:   {det['total_cost']:.4f}")
    print(f"Cost of ignoring uncertainty:  {vss_gap:.4f}  "
          f"({vss_gap / sp['total_cost'] * 100:.1f}% of total)")
    print(f"Solve time (SP):               {sp['solve_time']:.2f} s")
    print(f"Solve time (baseline):         {det['solve_time']:.2f} s")
    print()

    assert abs(sp["total_cost"] - 2.774) < 1e-3, (
        f"FAIL: SP cost {sp['total_cost']:.6f} deviates from 2.774 by "
        f"{abs(sp['total_cost'] - 2.774):.6f} > 1e-3"
    )
    assert abs(det["total_cost"] - 2.620) < 1e-3, (
        f"FAIL: Baseline cost {det['total_cost']:.6f} deviates from 2.620 by "
        f"{abs(det['total_cost'] - 2.620):.6f} > 1e-3"
    )
    assert abs(vss_gap - 0.154) < 1e-2, (
        f"FAIL: VSS gap {vss_gap:.6f} deviates from 0.154 by "
        f"{abs(vss_gap - 0.154):.6f} > 1e-2"
    )

    print("PASS — reproduces §15.")


if __name__ == "__main__":
    verify_toy_example()
