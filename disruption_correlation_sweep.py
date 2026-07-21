"""
disruption_correlation_sweep.py — Phase 1b resilience experiment.

Phase 1 (disruption_experiment.py) compared two extremes of facility
disruption: fully "independent" failures — where surviving facilities absorb
failed batches through subcontracting — and a total "common"-mode outage, where
every facility drops at once and the network can only cancel. The interesting
regime is in between: *partially correlated* disruptions.

This experiment fixes a moderate disruption level (q on every facility) and
sweeps the correlation weight rho ∈ [0, 1] of the common-factor model added to
yield_sp_v1.sample_disruptions (mode="partial"):

  s_o        ~ Bernoulli(rho)                 # per-scenario common shock
  u_{o,m}    ~ Bernoulli(q_m)                 # idiosyncratic disruption
  Xi[o, m]   = s_o OR ((1 - s_o)·u_{o,m})     # eq. 17 disruption indicator

rho=0 reproduces "independent"; rho=1 drives the common shock certain (every
scenario a total outage). As rho climbs, disruptions become correlated: when a
scenario's shock fires, all facilities go down together, so no surviving
facility is left to subcontract to. The figure this supports — subcontracting
value vs. correlation — expects E[subcontract] to fall monotonically toward
zero as rho → 1, with E[cancel] rising to compensate.

The SP itself is unchanged from Phase 1; only the disruption sampler differs.
As in case_study.py / disruption_experiment.py, the 50-patient case-study
instance is solved with HiGHS (the full model exceeds the size-limited Gurobi
license), reusing disruption_experiment.solve_sp_disrupted so the two Phase-1
experiments share an identical backend and scenario count.

Run: python disruption_correlation_sweep.py
"""

from __future__ import annotations

import json

import numpy as np

from yield_sp_v1 import sample_scenarios, sample_disruptions
from case_study import build_case_study_instance
from disruption_experiment import solve_sp_disrupted


# Scenario count and seed match Phase 1 (disruption_experiment.py) so the two
# experiments are directly comparable.
RHO_SWEEP = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
Q_LEVELS = [0.05, 0.10, 0.15]   # 0.10 is the headline level; 0.05/0.15 = severity axis
N_SCENARIOS = 200
SEED = 0


# ---------------------------------------------------------------------------
# Capacity-floor guard (methodology eq. 19)
# ---------------------------------------------------------------------------

def check_capacity_floor() -> str:
    """
    Guard test for the eq.-19 capacity floor in yield_sp_v1.build_sp_model.

    Builds the canonical Gurobi model with a *partial* capacity loss that
    exceeds the contracted capacity at one facility (delta_cap_m > C_m, with
    delta_cap_m < s_max_m so it takes the partial branch, not full shutdown)
    and a disruption in one scenario. The floored form C_m − min(Δ_m, C_m)·ξ_m
    keeps that row's available capacity non-negative and the model feasible;
    the raw form C_m − Δ_m·ξ_m would drive it negative and infeasible.

    Returns a short status string; raises AssertionError on failure. Skipped
    silently (returns a note) if Gurobi is unavailable.
    """
    try:
        import gurobipy  # noqa: F401
        from yield_sp_v1 import toy_instance, toy_scenarios, build_and_solve_sp
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"skipped ({type(exc).__name__})"

    inst = toy_instance()               # 3 patients, 2 facilities, s_max = 5
    Y, B = toy_scenarios()              # 4 scenarios
    N = Y.shape[0]

    # Pin a small contracted capacity at m0 so the partial loss exceeds it.
    C0 = 2
    delta0 = 3                          # C0 (=2) < delta0 (=3) < s_max (=5)
    inst.delta_cap = np.array([delta0, 0.0])

    fix = {
        "z": {0: 1, 1: 1},
        "C": {0: C0, 1: 2},
        "x": {(i, m): 0 for i in inst.I for m in inst.M},
    }
    fix["x"][0, 0] = fix["x"][1, 0] = 1   # patients 1,2 → m0
    fix["x"][2, 1] = 1                    # patient 3 → m1

    # Disrupt m0 in exactly one scenario.
    Xi = np.zeros((N, inst.n_facilities))
    Xi[2, 0] = 1.0

    # Floored available recourse capacity at the disrupted facility.
    floored = C0 - min(delta0, C0)        # = 0
    raw = C0 - delta0                     # = -1 (what the unfloored form gives)
    assert floored >= 0, f"floored capacity RHS {floored} is negative"

    res = build_and_solve_sp(inst, Y, B, fix_first_stage=fix, Xi=Xi)
    # A feasible optimum was found (solve_and_extract raises otherwise).
    assert np.isfinite(res["total_cost"]), "model did not solve to a finite optimum"

    return (f"passed (delta_cap={delta0} > C={C0}; floored RHS={floored} ≥ 0, "
            f"raw would be {raw} < 0; model feasible, cost={res['total_cost']:.4f})")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _recourse_counts(res: dict) -> dict:
    """Expected per-scenario counts of each recourse action."""
    N = res["r_remfg"].shape[0]
    return {
        "remfg": float(res["r_remfg"].sum() / N),
        "sub": float(res["r_sub"].sum() / N),
        "cancel": float(res["r_cancel"].sum() / N),
    }


def run_experiment() -> dict:
    inst = build_case_study_instance()
    n_f = inst.n_facilities

    # Yield/eligibility scenarios are shared across the whole sweep so that
    # differences are attributable to disruption correlation alone — the same
    # convention as disruption_experiment.py.
    Y, B = sample_scenarios(inst, N_SCENARIOS, seed=SEED)

    results = []
    for q in Q_LEVELS:
        for rho in RHO_SWEEP:
            inst_q = build_case_study_instance()
            inst_q.q = np.full(n_f, q)
            # delta_cap left None ⇒ full removal of recourse capacity (s_max),
            # matching Phase 1: a disrupted facility offers zero recourse.
            Xi = sample_disruptions(inst_q, N_SCENARIOS, seed=SEED,
                                    mode="partial", rho=rho)
            res = solve_sp_disrupted(inst_q, Y, B, Xi=Xi)

            counts = _recourse_counts(res)
            frac_all_down = float((Xi.sum(axis=1) == n_f).mean())
            results.append({
                "q": q,
                "rho": rho,
                "total_cost": res["total_cost"],
                "stage1_cost": res["stage1_cost"],
                "expected_stage2_cost": res["expected_stage2_cost"],
                "exp_remfg": counts["remfg"],
                "exp_sub": counts["sub"],
                "exp_cancel": counts["cancel"],
                "frac_scenarios_all_down": frac_all_down,
                "gap": res["gap"],
                "solve_time": res["solve_time"],
            })
    return {
        "instance": {"n_patients": inst.n_patients,
                     "n_facilities": inst.n_facilities,
                     "n_scenarios": N_SCENARIOS,
                     "seed": SEED},
        "rho_sweep": RHO_SWEEP,
        "q_levels": Q_LEVELS,
        "headline_q": 0.10,
        "results": results,
    }


def _print_table(payload: dict) -> None:
    for q in payload["q_levels"]:
        rows = [r for r in payload["results"] if r["q"] == q]
        tag = " (headline)" if q == payload["headline_q"] else ""
        print(f"\nq = {q:.2f}{tag}")
        hdr = (f"{'rho':>6}{'tot_cost':>11}{'E[remfg]':>10}{'E[sub]':>9}"
               f"{'E[cancel]':>11}{'all_down':>10}{'solve_s':>9}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['rho']:>6.2f}{r['total_cost']:>11.4f}{r['exp_remfg']:>10.3f}"
                  f"{r['exp_sub']:>9.3f}{r['exp_cancel']:>11.3f}"
                  f"{r['frac_scenarios_all_down']:>10.3f}{r['solve_time']:>9.1f}")


def _trend_summary(payload: dict) -> None:
    print("\n--- Subcontracting value vs. correlation ---")
    for q in payload["q_levels"]:
        rows = sorted((r for r in payload["results"] if r["q"] == q),
                      key=lambda r: r["rho"])
        sub0, sub1 = rows[0]["exp_sub"], rows[-1]["exp_sub"]
        can0, can1 = rows[0]["exp_cancel"], rows[-1]["exp_cancel"]
        subs = [r["exp_sub"] for r in rows]
        cans = [r["exp_cancel"] for r in rows]
        sub_mono = all(subs[i] >= subs[i + 1] - 1e-6 for i in range(len(subs) - 1))
        can_mono = all(cans[i] <= cans[i + 1] + 1e-6 for i in range(len(cans) - 1))
        print(f"q={q:.2f}: E[sub] {sub0:.3f} → {sub1:.3f} "
              f"({'monotone ↓' if sub_mono else 'non-monotone'}); "
              f"E[cancel] {can0:.3f} → {can1:.3f} "
              f"({'monotone ↑' if can_mono else 'non-monotone'})")


def main() -> None:
    print("=== Phase-1b: subcontracting value vs. disruption correlation ===\n")
    print(f"Capacity-floor guard (eq. 19, build_sp_model): {check_capacity_floor()}\n")

    payload = run_experiment()
    _print_table(payload)

    with open("disruption_correlation_sweep_results.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nResults saved to disruption_correlation_sweep_results.json")

    _trend_summary(payload)


if __name__ == "__main__":
    main()
