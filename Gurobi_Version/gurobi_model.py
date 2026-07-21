"""
Gurobi_Version/gurobi_model.py — clean Gurobi backend for the CAR-T resilient
two-stage stochastic program.

This module is a thin, self-contained Gurobi entry point for the SAME model the
rest of the repo solves — yield uncertainty + three-action recourse
(re-manufacture / subcontract / cancel) + the capacity-disruption channel with
the corrected non-negative capacity floor. It deliberately does NOT reimplement
the model: the canonical Gurobi builder already lives in
``yield_sp_v1.build_sp_model`` (with the eq.-19 floor, so a fully disrupted
facility contributes ``(1 − ξ)·C_m = 0`` recourse capacity). We import that,
plus the repo's ``Instance`` and scenario samplers, and expose a solve function
whose return dict matches the HiGHS backend in ``disruption_experiment.py`` so
the two are drop-in comparable.

No Gurobi license material (keys, WLS credentials, gurobi.lic) is embedded or
read here — gurobipy picks up whatever license is installed on the machine.

The full 50-patient case-study instance needs an unrestricted Gurobi license;
the size-limited pip license only builds small subsets (see README).
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

# Make the repository root importable when run from inside Gurobi_Version/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yield_sp_v1 import (          # noqa: E402  (import after sys.path tweak)
    Instance,
    build_and_solve_sp,
    sample_scenarios,
    sample_disruptions,
)


# ---------------------------------------------------------------------------
# Case-study subsetting (for license-limited demonstration / parity)
# ---------------------------------------------------------------------------

def subset_instance(inst: Instance, k: int) -> Instance:
    """
    Return a copy of `inst` restricted to its first `k` patients.

    Only per-patient fields are sliced (beta, rho_cancel, t_trans); all
    per-facility structure and costs are preserved. Used so the size-limited
    Gurobi license can solve a genuine case-study sub-instance for parity and
    timing, without touching the model itself.
    """
    if k > inst.n_patients:
        raise ValueError(f"k={k} exceeds instance size {inst.n_patients}")
    s = copy.deepcopy(inst)
    s.n_patients = k
    s.beta = np.asarray(inst.beta)[:k].copy()
    s.rho_cancel = np.asarray(inst.rho_cancel)[:k].copy()
    if inst.t_trans is not None:
        s.t_trans = np.asarray(inst.t_trans)[:k].copy()
    return s


# ---------------------------------------------------------------------------
# Gurobi solve — same return contract as disruption_experiment.solve_sp_disrupted
# ---------------------------------------------------------------------------

def solve_sp_gurobi(
    inst: Instance,
    Y: np.ndarray,
    B: np.ndarray,
    Xi: np.ndarray | None = None,
    *,
    mip_gap: float = 1e-9,
    time_limit: float = 1800.0,
    verbose: bool = False,
) -> dict:
    """
    Solve the two-stage SP with Gurobi via the canonical yield_sp_v1 builder.

    Parameters mirror disruption_experiment.solve_sp_disrupted; the returned
    dict carries the same keys (total_cost, stage1_cost, expected_stage2_cost,
    z, C, x, r_remfg, r_sub, r_cancel, solve_time, gap) so callers can swap
    backends freely.

    mip_gap defaults to 1e-9 (effectively exact) so Gurobi and HiGHS agree to
    solver tolerance on these MILPs — Gurobi is for speed and reporting, not a
    different answer.
    """
    res = build_and_solve_sp(
        inst, Y, B,
        Xi=Xi,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )
    # build_and_solve_sp already returns the canonical dict; pass it through.
    return res


def recourse_counts(res: dict) -> dict:
    """Expected per-scenario counts of each recourse action (as HiGHS runner)."""
    N = res["r_remfg"].shape[0]
    return {
        "remfg": float(res["r_remfg"].sum() / N),
        "sub": float(res["r_sub"].sum() / N),
        "cancel": float(res["r_cancel"].sum() / N),
    }


__all__ = [
    "Instance",
    "subset_instance",
    "solve_sp_gurobi",
    "recourse_counts",
    "sample_scenarios",
    "sample_disruptions",
]
