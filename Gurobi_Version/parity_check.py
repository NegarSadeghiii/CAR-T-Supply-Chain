"""
Gurobi_Version/parity_check.py — Gurobi vs HiGHS solver parity + timing.

Both HiGHS (the repo's default backend, disruption_experiment.solve_sp_disrupted)
and Gurobi (Gurobi_Version.gurobi_model.solve_sp_gurobi) are exact MILP solvers,
so on the SAME model / scenarios / tolerance they must return the SAME optimal
objective. This script confirms that agreement to within 1e-6 and prints a
like-for-like solve-time comparison, on cases that fit the size-limited Gurobi
license:

  * the toy instance (3 patients, 4 scenarios), with and without a disruption,
  * a small case-study subset (few patients, few scenarios), undisrupted and
    with a partial-correlation disruption pattern.

Gurobi is for speed and reporting, not a different answer — any mismatch above
tolerance is a bug and raises AssertionError.

Run: python Gurobi_Version/parity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yield_sp_v1 import toy_instance, toy_scenarios              # noqa: E402
from case_study import build_case_study_instance                 # noqa: E402
from disruption_experiment import solve_sp_disrupted             # noqa: E402  (HiGHS)
from Gurobi_Version.gurobi_model import (                        # noqa: E402
    solve_sp_gurobi, subset_instance, sample_scenarios, sample_disruptions,
)

TOL = 1e-6
_MIP_GAP = 1e-9   # tight, so both solvers hit the exact optimum


def _one_case(label, inst, Y, B, Xi, rows):
    """Solve one case on both backends, assert parity, record timing."""
    g = solve_sp_gurobi(inst, Y, B, Xi=Xi, mip_gap=_MIP_GAP)
    h = solve_sp_disrupted(inst, Y, B, Xi=Xi, mip_gap=_MIP_GAP)
    diff = abs(g["total_cost"] - h["total_cost"])
    ok = diff <= TOL
    rows.append({
        "case": label,
        "gurobi_cost": g["total_cost"],
        "highs_cost": h["total_cost"],
        "abs_diff": diff,
        "gurobi_s": g["solve_time"],
        "highs_s": h["solve_time"],
        "match": ok,
    })
    if not ok:
        raise AssertionError(
            f"PARITY FAIL [{label}]: Gurobi={g['total_cost']:.9f} "
            f"HiGHS={h['total_cost']:.9f} |Δ|={diff:.2e} > {TOL:.0e}"
        )
    return g, h


def run_parity() -> list[dict]:
    rows: list[dict] = []

    # --- Toy instance: with and without a disruption ---
    ti = toy_instance()
    Yt, Bt = toy_scenarios()
    N = Yt.shape[0]
    _one_case("toy / no-disruption", ti, Yt, Bt, None, rows)

    Xi_toy = np.zeros((N, ti.n_facilities))
    Xi_toy[2, 0] = 1.0   # disrupt m0 in scenario 3
    _one_case("toy / m0-disrupted-w3", ti, Yt, Bt, Xi_toy, rows)

    # --- Small case-study subset (fits size-limited license) ---
    base = build_case_study_instance()
    k, Nsub = 6, 10
    sub = subset_instance(base, k)
    sub.q = np.full(sub.n_facilities, 0.10)
    Ys, Bs = sample_scenarios(sub, Nsub, seed=0)
    _one_case(f"case-study subset k={k} N={Nsub} / no-disruption",
              sub, Ys, Bs, None, rows)

    Xi_sub = sample_disruptions(sub, Nsub, seed=0, mode="partial", rho=0.5)
    _one_case(f"case-study subset k={k} N={Nsub} / partial rho=0.5",
              sub, Ys, Bs, Xi_sub, rows)

    return rows


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'case':<42}{'Gurobi cost':>14}{'HiGHS cost':>14}"
           f"{'|Δ|':>11}{'Gurobi s':>10}{'HiGHS s':>10}{'match':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['case']:<42}{r['gurobi_cost']:>14.6f}{r['highs_cost']:>14.6f}"
              f"{r['abs_diff']:>11.2e}{r['gurobi_s']:>10.3f}{r['highs_s']:>10.3f}"
              f"{'  ✓' if r['match'] else '  ✗':>7}")


def main() -> bool:
    print("=== Gurobi vs HiGHS parity + timing (exact MILP, tol=1e-6) ===\n")
    rows = run_parity()
    _print_table(rows)
    all_ok = all(r["match"] for r in rows)
    max_diff = max(r["abs_diff"] for r in rows)
    print(f"\nAll cases match within {TOL:.0e}: {all_ok}  (max |Δ| = {max_diff:.2e})")
    print("Timing is like-for-like (same model, scenarios, tolerance). Gurobi and "
          "HiGHS are both exact\nMILP solvers, so identical objectives are expected; "
          "Gurobi is used for speed and reporting.")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
