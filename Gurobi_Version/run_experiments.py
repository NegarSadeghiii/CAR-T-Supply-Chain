"""
Gurobi_Version/run_experiments.py — solve the case study and the
disruption/correlation experiments with Gurobi, writing to
Gurobi_Version/results/.

This is the Gurobi counterpart to case_study.py / disruption_experiment.py /
disruption_correlation_sweep.py. It reuses the repo's Instance, scenario
sampler, and disruption sampler, and solves with the clean Gurobi backend in
gurobi_model.py.

Licensing. The full 50-patient case-study instance needs an UNRESTRICTED Gurobi
license. Under the size-limited pip license the full build raises a GurobiError;
this runner detects that and transparently falls back to the largest
case-study *subset* that fits, recording which instance size was actually
solved so the output is never silently wrong. On a machine with a full license,
set FULL_INSTANCE=1 (or pass --full) to force the real 50-patient run.

Run:  python Gurobi_Version/run_experiments.py
      FULL_INSTANCE=1 python Gurobi_Version/run_experiments.py   # full license
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import gurobipy as gp
    _GRB_ERROR = gp.GurobiError
except Exception:                       # pragma: no cover
    _GRB_ERROR = Exception

from case_study import build_case_study_instance                 # noqa: E402
from Gurobi_Version.gurobi_model import (                        # noqa: E402
    solve_sp_gurobi, subset_instance, recourse_counts,
    sample_scenarios, sample_disruptions,
)

_RESULTS = _HERE / "results"
_RESULTS.mkdir(exist_ok=True)

# Experiment grid — mirrors disruption_correlation_sweep.py so Gurobi and HiGHS
# results are directly comparable when run on the same instance.
RHO_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
Q_LEVEL = 0.10
SEED = 0

# Full-license settings vs. size-limited fallback.
FULL_N_SCENARIOS = 200
FALLBACK_K = 6           # patients that fit the size-limited license
FALLBACK_N = 10          # scenarios that fit alongside


def _select_instance(force_full: bool):
    """
    Return (inst, meta) for the largest instance this license can solve.

    Tries the full 50-patient / FULL_N_SCENARIOS build first; on a size-limit
    GurobiError falls back to a (FALLBACK_K, FALLBACK_N) subset. `meta` records
    exactly what was solved.
    """
    base = build_case_study_instance()

    def _probe(inst, n):
        Y, B = sample_scenarios(inst, n, seed=SEED)
        solve_sp_gurobi(inst, Y, B, Xi=None)   # raises if too large

    if force_full:
        return base, {"instance": "full", "n_patients": base.n_patients,
                      "n_scenarios": FULL_N_SCENARIOS}

    # Auto-detect: attempt full, fall back on size limit.
    try:
        _probe(base, FULL_N_SCENARIOS)
        return base, {"instance": "full", "n_patients": base.n_patients,
                      "n_scenarios": FULL_N_SCENARIOS}
    except _GRB_ERROR as exc:
        if "size-limited" not in str(exc).lower():
            raise
        sub = subset_instance(base, FALLBACK_K)
        print(f"[license] full 50-patient build exceeds the size-limited "
              f"license; falling back to a k={FALLBACK_K}, N={FALLBACK_N} "
              f"case-study subset.\n           (Run with a full license and "
              f"--full / FULL_INSTANCE=1 for the real 50-patient results.)\n")
        return sub, {"instance": "subset", "n_patients": FALLBACK_K,
                     "n_scenarios": FALLBACK_N}


def run_case_study(inst, meta) -> dict:
    n = meta["n_scenarios"]
    Y, B = sample_scenarios(inst, n, seed=SEED)
    res = solve_sp_gurobi(inst, Y, B, Xi=None)
    counts = recourse_counts(res)
    payload = {
        **meta,
        "seed": SEED,
        "total_cost": res["total_cost"],
        "stage1_cost": res["stage1_cost"],
        "expected_stage2_cost": res["expected_stage2_cost"],
        "z": res["z"].tolist(),
        "C": res["C"].tolist(),
        "exp_remfg": counts["remfg"],
        "exp_sub": counts["sub"],
        "exp_cancel": counts["cancel"],
        "solve_time": res["solve_time"],
        "gap": res["gap"],
        "solver": "gurobi",
    }
    with open(_RESULTS / "case_study_gurobi.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def run_correlation_sweep(inst, meta) -> dict:
    n = meta["n_scenarios"]
    n_f = inst.n_facilities
    inst = subset_instance(inst, inst.n_patients)   # fresh copy
    inst.q = np.full(n_f, Q_LEVEL)
    Y, B = sample_scenarios(inst, n, seed=SEED)

    rows = []
    for rho in RHO_SWEEP:
        Xi = sample_disruptions(inst, n, seed=SEED, mode="partial", rho=rho)
        res = solve_sp_gurobi(inst, Y, B, Xi=Xi)
        counts = recourse_counts(res)
        rows.append({
            "rho": rho,
            "total_cost": res["total_cost"],
            "exp_remfg": counts["remfg"],
            "exp_sub": counts["sub"],
            "exp_cancel": counts["cancel"],
            "solve_time": res["solve_time"],
            "gap": res["gap"],
        })
    payload = {**meta, "q": Q_LEVEL, "seed": SEED,
               "rho_sweep": RHO_SWEEP, "solver": "gurobi", "results": rows}
    with open(_RESULTS / "correlation_sweep_gurobi.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Force the full 50-patient instance (needs full license).")
    args = ap.parse_args()
    force_full = args.full or os.environ.get("FULL_INSTANCE") == "1"

    print("=== Gurobi runner: case study + disruption correlation ===\n")
    inst, meta = _select_instance(force_full)
    print(f"Solving: {meta['instance']} instance — "
          f"{meta['n_patients']} patients, {meta['n_scenarios']} scenarios, "
          f"{inst.n_facilities} facilities.\n")

    cs = run_case_study(inst, meta)
    print(f"[case study] total_cost={cs['total_cost']:.4f}  "
          f"z={cs['z']}  C={cs['C']}  "
          f"E[remfg/sub/cancel]={cs['exp_remfg']:.3f}/"
          f"{cs['exp_sub']:.3f}/{cs['exp_cancel']:.3f}  "
          f"({cs['solve_time']:.2f}s)")

    sweep = run_correlation_sweep(inst, meta)
    print("\n[correlation sweep] q=%.2f" % sweep["q"])
    print(f"{'rho':>6}{'tot_cost':>12}{'E[sub]':>9}{'E[cancel]':>11}{'solve_s':>9}")
    print("-" * 47)
    for r in sweep["results"]:
        print(f"{r['rho']:>6.2f}{r['total_cost']:>12.4f}{r['exp_sub']:>9.3f}"
              f"{r['exp_cancel']:>11.3f}{r['solve_time']:>9.2f}")

    print(f"\nResults written to {_RESULTS.relative_to(_REPO_ROOT)}/"
          f" (case_study_gurobi.json, correlation_sweep_gurobi.json)")


if __name__ == "__main__":
    main()
