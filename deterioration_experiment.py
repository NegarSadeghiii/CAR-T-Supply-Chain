"""
deterioration_experiment.py — capacity-tight rerun of the delay-deterioration
study, in plain healthcare-operations terms.

Setting. We remove the artificial capacity cap (s_max 40 -> 75, Wan 2026's real
Maxcap) and scale demand up so facilities run near full capacity, because
crowding-driven manufacturing delay only bites near capacity. We then ask, in
patient terms: how many high-urgency patients die (are never treated in time),
what share of patients are treated within their clinical window, and what it
costs — comparing a plan that ASSUMES manufacturing always succeeds on time with
a plan that ACCOUNTS for patients getting sicker while they wait.

Two output buckets (written by make_main_figures / make_technical_figures and by
RESULTS.md vs TECHNICAL_APPENDIX.md):
  * MAIN (figures/, RESULTS.md): plain healthcare language only.
  * TECHNICAL (figures/technical/, TECHNICAL_APPENDIX.md): validation checks and
    the internal model-parameter curves; technical names live only here / in code.

Reproducible: seeds fixed (design seed 0, patient-simulation seed 100). HiGHS
backend. Run via run_all.py or `python deterioration_experiment.py [--quick]`.

--- technical <-> plain glossary (kept in code only) --------------------------
  kappa (delay-sensitivity)     -> "how fast patients decline while waiting"
  gamma (Weibull shape)         -> internal decline-curve shape (technical only)
  D_exo (kappa=0 optimum)       -> "plan assuming manufacturing succeeds on time"
  D_endo (endogenous optimum)   -> "plan accounting for patient decline"
  tier-H / M / L cancellation   -> high- / medium- / low-urgency patients not
                                   treated in time (high-urgency = deaths)
  VoE (cost_exo - cost_endo)    -> "cost saved by planning for decline"
  calibrated kappa = 1.0        -> decline speed matching real survival data
                                   (Dulobdas 2025 delayed-vs-control PFS HR 1.64)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from yield_sp_v1 import sample_scenarios
from scaled_instance import scaled_case_study_instance
from declineprob import calibrate_lambda
from clearing_function import DEFAULT_CLEARING
from dynamic_sp import (solve_dynamic_sp, solve_dynamic_sp_endogenous,
                        still_eligible_for_design, _delay_surrogate_curves)
from value_of_endogeneity import simulate_endogenous

_HERE = Path(__file__).resolve().parent
_FIG = _HERE / "figures"
_FIGT = _FIG / "technical"
_RES = _HERE / "results"
_FIGT.mkdir(parents=True, exist_ok=True)
_RES.mkdir(exist_ok=True)

# --- settings ---------------------------------------------------------------
S_MAX_REAL = 75            # Wan 2026 Maxcap (real); base case_study.py uses 40
CALIB_KAPPA = 1.0          # decline speed matching Dulobdas HR 1.64 (see declineprob)
SEED_TRAIN = 0
SEED_OOS = 100
N_TRAIN = 80
N_OOS = 500
MIP_GAP = 1e-2
TIME_LIMIT = 120.0
ENDO_MAX_ITER = 3          # fixed-point iterations for the decline-aware plan

# Finer detail on decline speed between 0 and 0.5, plus the higher stress points.
KAPPA_FINE = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
KAPPA_COARSE = [0.0, 0.25, 0.5, 1.0, 2.0]
GAMMA_TECH = [1.5, 2.0]    # extra internal-shape curves for the technical appendix


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _healthcare_metrics(inst, design, tiers, tier_idx, ev):
    """Translate a simulator result into plain patient/cost outcomes."""
    n_p = inst.n_patients
    sizes = {t: len(tier_idx[t]) for t in ("H", "M", "L")}
    treated = n_p - ev["exp_cancel"]
    C = np.asarray(design["C"], float)
    assigned = np.asarray(design["x"], float).sum(axis=0)
    return {
        "high_urgency_deaths": ev["tier_cancel"]["H"],
        "med_not_treated": ev["tier_cancel"]["M"],
        "low_not_treated": ev["tier_cancel"]["L"],
        "treated_share": treated / n_p,
        "treated_share_by_tier": {t: 1.0 - ev["tier_cancel"][t] / sizes[t] for t in ("H", "M", "L")},
        "total_cost": ev["total_cost"],
        "cost_per_treated": ev["total_cost"] / max(treated, 1e-9),
        "facilities_open": int((np.asarray(design["z"]) > 0.5).sum()),
        "open_facilities": [m for m in range(inst.n_facilities) if design["z"][m] > 0.5],
        "spare_capacity_built": float((C - assigned).clip(0).sum()),
        "capacity": C.tolist(),
    }


def _sim(inst, design, tiers, tier_idx, Yte, Bte, U, kappa, gamma):
    return simulate_endogenous(inst, design, tiers, Yte, Bte, U,
                               kappa=kappa, gamma=gamma, tier_idx=tier_idx)


# ---------------------------------------------------------------------------
# Core sweep: for one instance, sweep decline speed; both plans under true dynamics
# ---------------------------------------------------------------------------

def sweep_decline_speed(inst, tiers, tier_idx, kappas, gamma, *, label):
    Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED_TRAIN)
    Yte, Bte = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
    rng = np.random.default_rng(SEED_OOS + 777)
    U = rng.random((N_OOS, inst.n_patients))

    D_exo = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0,
                             mip_gap=MIP_GAP, time_limit=TIME_LIMIT)
    # What the on-time planner THINKS will happen (no decline).
    naive = _sim(inst, D_exo, tiers, tier_idx, Yte, Bte, U, 0.0, gamma)
    rows = []
    for kappa in kappas:
        if kappa == 0.0:
            D_endo = D_exo
        else:
            D_endo = solve_dynamic_sp_endogenous(inst, Ytr, Btr, tiers, kappa=kappa,
                                                 gamma=gamma, mip_gap=MIP_GAP,
                                                 time_limit=TIME_LIMIT, max_iter=ENDO_MAX_ITER)
        ev_exo = _sim(inst, D_exo, tiers, tier_idx, Yte, Bte, U, kappa, gamma)
        ev_endo = _sim(inst, D_endo, tiers, tier_idx, Yte, Bte, U, kappa, gamma)
        m_exo = _healthcare_metrics(inst, D_exo, tiers, tier_idx, ev_exo)
        m_endo = _healthcare_metrics(inst, D_endo, tiers, tier_idx, ev_endo)
        rows.append({
            "kappa": kappa, "gamma": gamma,
            "on_time_plan": m_exo, "decline_aware_plan": m_endo,
            "voe": ev_exo["total_cost"] - ev_endo["total_cost"],  # technical
            "high_urgency_deaths_saved": m_exo["high_urgency_deaths"] - m_endo["high_urgency_deaths"],
        })
        print(f"    [{label}] decline-speed={kappa:<4}: on-time H-deaths="
              f"{m_exo['high_urgency_deaths']:.2f} cost/pt={m_exo['cost_per_treated']:.3f} "
              f"| decline-aware H-deaths={m_endo['high_urgency_deaths']:.2f} "
              f"spare={m_endo['spare_capacity_built']:.0f}")
    return {"label": label, "n_patients": inst.n_patients, "s_max": int(inst.s_max[0]),
            "gamma": gamma, "kappas": kappas,
            "naive_predicted": _healthcare_metrics(inst, D_exo, tiers, tier_idx, naive),
            "D_exo": {"z": D_exo["z"].tolist(), "C": D_exo["C"].tolist()},
            "rows": rows}


# ---------------------------------------------------------------------------
# Technical validation: planned vs simulated crowding deaths as spare grows
# ---------------------------------------------------------------------------

def planned_vs_simulated_deaths(inst, tiers, tier_idx, kappa, gamma):
    """
    Take the on-time design, then progressively add spare capacity at its busiest
    facility, and compare the PLANNING model's internal estimate of delay-caused
    deaths (the surrogate) with the DETAILED patient simulation.
    """
    Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED_TRAIN)
    Yte, Bte = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
    rng = np.random.default_rng(SEED_OOS + 777)
    U = rng.random((N_OOS, inst.n_patients))
    D = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0, mip_gap=MIP_GAP, time_limit=TIME_LIMIT)
    x = np.asarray(D["x"], float)
    assign = x.argmax(axis=1)
    busiest = int(np.bincount(assign, minlength=inst.n_facilities).argmax())
    a_busy = int((assign == busiest).sum())
    rho_cancel = np.asarray(inst.rho_cancel, float)
    from declineprob import extra_survival, HR_TIER
    lam = calibrate_lambda(gamma)

    out = []
    for extra in range(0, int(inst.s_max[busiest]) - a_busy + 1, max(1, (int(inst.s_max[busiest]) - a_busy) // 6 or 1)):
        C = np.asarray(D["C"], float).copy()
        C[busiest] = a_busy + extra
        cand = {"z": D["z"], "C": C, "x": x}
        # planned delay-deaths (surrogate) for the busiest facility at this C
        delay = float(DEFAULT_CLEARING.remake_delay(a_busy / C[busiest]))
        planned = 0.0
        for i in np.where(assign == busiest)[0]:
            es = float(extra_survival(delay, HR_TIER[tiers[i]], lam, gamma, kappa))
            planned += (1.0 - inst.p[busiest]) * inst.beta[i] * (1.0 - es) * rho_cancel[i]
        # simulated total delay-death COST (extra cancellation cost vs no delay)
        ev = _sim(inst, cand, tiers, tier_idx, Yte, Bte, U, kappa, gamma)
        ev0 = _sim(inst, cand, tiers, tier_idx, Yte, Bte, U, 0.0, gamma)
        sim_delay_cost = ev["total_cost"] - ev0["total_cost"]
        out.append({"spare_at_busiest": extra, "planned_delay_death_cost": planned,
                    "simulated_delay_cost": sim_delay_cost})
    return {"busiest_facility": busiest, "base_load": a_busy, "kappa": kappa,
            "gamma": gamma, "points": out}


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(quick=False):
    t0 = time.perf_counter()
    tight_mult = 3          # 150 patients -> ~100% of two-facility capacity at s_max=75
    low_mult = 1            # 50 patients  -> low-demand contrast
    smax_mult = 2           # 100 patients -> feasible for s_max in {40,55,75}
    kfine = [0.0, 0.25, 1.0, 2.0] if quick else KAPPA_FINE
    kcoarse = [0.0, 1.0] if quick else KAPPA_COARSE
    gtech = [2.0] if quick else GAMMA_TECH

    payload = {"calib_kappa": CALIB_KAPPA, "s_max_real": S_MAX_REAL,
               "n_train": N_TRAIN, "n_oos": N_OOS,
               "seed_train": SEED_TRAIN, "seed_oos": SEED_OOS,
               "lambda_by_gamma": {str(g): calibrate_lambda(g) for g in [1.0] + gtech}}

    # 1. MAIN capacity-tight setting (real capacity, growing demand), gamma=1.
    print("[MAIN tight] 150 patients, s_max=75 ...")
    inst, tiers, tidx = scaled_case_study_instance(mult=tight_mult, s_max=S_MAX_REAL)
    payload["main_tight"] = sweep_decline_speed(inst, tiers, tidx, kfine, 1.0, label="tight")

    # 2. Low-demand contrast.
    print("[contrast low-demand] 50 patients, s_max=75 ...")
    inst, tiers, tidx = scaled_case_study_instance(mult=low_mult, s_max=S_MAX_REAL)
    payload["low_demand"] = sweep_decline_speed(inst, tiers, tidx, kfine, 1.0, label="low")

    # 3. Capacity-cap sweep {40, 55, 75} at the calibrated decline speed.
    print("[capacity-cap sweep] 100 patients, s_max in {40,55,75} ...")
    cap_rows = []
    for smax in ([40, 75] if quick else [40, 55, 75]):
        inst, tiers, tidx = scaled_case_study_instance(mult=smax_mult, s_max=smax)
        Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED_TRAIN)
        Yte, Bte = sample_scenarios(inst, N_OOS, seed=SEED_OOS)
        rng = np.random.default_rng(SEED_OOS + 777); U = rng.random((N_OOS, inst.n_patients))
        D_exo = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0, mip_gap=MIP_GAP, time_limit=TIME_LIMIT)
        D_endo = solve_dynamic_sp_endogenous(inst, Ytr, Btr, tiers, kappa=CALIB_KAPPA, gamma=1.0,
                                             mip_gap=MIP_GAP, time_limit=TIME_LIMIT, max_iter=ENDO_MAX_ITER)
        ev_exo = _sim(inst, D_exo, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, 1.0)
        ev_endo = _sim(inst, D_endo, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, 1.0)
        cap_rows.append({"s_max": smax, "n_patients": inst.n_patients,
                         "on_time_plan": _healthcare_metrics(inst, D_exo, tiers, tidx, ev_exo),
                         "decline_aware_plan": _healthcare_metrics(inst, D_endo, tiers, tidx, ev_endo)})
        print(f"    s_max={smax}: on-time H-deaths={ev_exo['tier_cancel']['H']:.2f} "
              f"open={[m for m in range(4) if D_exo['z'][m]>0.5]} | "
              f"decline-aware H-deaths={ev_endo['tier_cancel']['H']:.2f}")
    payload["cap_sweep"] = {"kappa": CALIB_KAPPA, "rows": cap_rows}

    # 4. TECHNICAL: internal-shape (gamma) curves for the inverted-U, tight setting.
    print("[technical] internal-shape sweeps ...")
    inst, tiers, tidx = scaled_case_study_instance(mult=tight_mult, s_max=S_MAX_REAL)
    payload["tech_gamma"] = [sweep_decline_speed(inst, tiers, tidx, kcoarse, g, label=f"g{g}")
                             for g in gtech]

    # 5. TECHNICAL: planned vs simulated crowding deaths as spare grows.
    print("[technical] planned vs simulated crowding deaths ...")
    inst, tiers, tidx = scaled_case_study_instance(mult=tight_mult, s_max=S_MAX_REAL)
    payload["planned_vs_sim"] = planned_vs_simulated_deaths(inst, tiers, tidx, CALIB_KAPPA, 1.0)

    payload["compute_seconds"] = time.perf_counter() - t0
    out = _RES / ("deterioration_results_quick.json" if quick else "deterioration_results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out} ({payload['compute_seconds']:.0f}s)")
    return payload, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    payload, out = run_experiment(quick=args.quick)
    if not args.no_figures and not args.quick:
        from deterioration_figures import make_all_figures
        make_all_figures(payload)
    print("Done.")
