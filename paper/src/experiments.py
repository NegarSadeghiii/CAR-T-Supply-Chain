"""
paper/src/experiments.py — run the full seeded experiment set for the manuscript
and write every raw asset.

All experiments use the patient-level simulator (patient_simulator.py) that tags
each loss as (a) lost during the normal ~6-week wait on a successful batch or
(b) lost after a manufacturing failure, and supports the sickest-first lever.
Designs are solved with HiGHS; seeds are fixed (train 0, out-of-sample 100).

Contents
  VALIDATION  V1 nesting (decline-speed 0 == base model), V2 priority-OFF ==
              no-lever model, V3 planning-model estimate vs detailed simulation.
  E1  prediction gap : on-time plan ASSUMES (no deterioration) vs ACTUAL
      (real decline rate) high-urgency lost and cost, by tier.
  E2  loss-cause split : normal-wait vs after-failure, busy (150p) & low (50p).
  E3  three plans @150p : on-time / decline-aware-spare / sickest-first.
  E4  capacity-cap sweep : s_max in {40,55,75}.
  E5  robustness to the normal-wait mortality anchor (BASE_WAIT_DEATH_6WK).
  SENS  decline-shape (gamma) grid and tier hazard-ratio (HR) grid.

Writes paper/data/raw/*.json (scenario-level + logs) and returns one master
payload assembled by make_assets.py into paper/results/results.json.

Run standalone:  python paper/src/experiments.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent          # repository root
sys.path.insert(0, str(_ROOT))

from yield_sp_v1 import sample_scenarios                       # noqa: E402
from case_study import build_case_study_instance              # noqa: E402
from scaled_instance import scaled_case_study_instance        # noqa: E402
from declineprob import calibrate_lambda, HR_TIER             # noqa: E402
from clearing_function import DEFAULT_CLEARING                # noqa: E402
from dynamic_sp import solve_dynamic_sp, solve_dynamic_sp_endogenous  # noqa: E402
from patient_simulator import simulate_patients, BASE_WAIT_DEATH_6WK  # noqa: E402

# ---- fixed experiment settings (seeded / reproducible) ----------------------
CALIB_KAPPA = 1.0                # decline speed matched to Dulobdas 2025 PFS HR 1.64
GAMMA = 1.0                      # central decline-curve shape
S_MAX_REAL = 75                  # Wan et al. 2026 Maxcap
SEED_TRAIN = 0
SEED_OOS = 100
KAPPA_SWEEP = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
GAMMA_GRID = [1.0, 1.5, 2.0]
# E5: multiply the whole normal-wait mortality vector; see numbers.md for basis.
BWD_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0]
# HR sensitivity: (H,M,L) tier delay hazard ratios; central 1.6/1.5/1.4, plus a
# narrower and a wider spread around the data-anchored 1.4-1.6 band.
HR_GRIDS = {
    "narrow": {"H": 1.55, "M": 1.50, "L": 1.45},
    "central": {"H": 1.60, "M": 1.50, "L": 1.40},
    "wide": {"H": 1.70, "M": 1.50, "L": 1.30},
}
_RAW = _ROOT / "paper" / "data" / "raw"


def _settings(quick):
    if quick:
        return dict(n_train=60, n_oos=200, mip_gap=2e-2, time_limit=60.0,
                    kappas=[0.0, 0.25, 1.0, 2.0])
    return dict(n_train=80, n_oos=500, mip_gap=1e-2, time_limit=120.0, kappas=KAPPA_SWEEP)


def _draws(inst, n_train, n_oos):
    Ytr, Btr = sample_scenarios(inst, n_train, seed=SEED_TRAIN)
    Yte, Bte = sample_scenarios(inst, n_oos, seed=SEED_OOS)
    rng = np.random.default_rng(SEED_OOS + 777)
    U = rng.random((n_oos, inst.n_patients))
    return Ytr, Btr, Yte, Bte, U


def _sim(inst, design, tiers, tidx, Yte, Bte, U, kappa, priority=False,
         base_wait_death=None, hr_tier=None, gamma=GAMMA):
    return simulate_patients(inst, design, tiers, tidx, Yte, Bte, U,
                             kappa=kappa, gamma=gamma, priority=priority,
                             base_wait_death=base_wait_death, hr_tier=hr_tier)


def _plan_summary(s):
    return {"high_urgency_lost": s["high_urgency_lost"],
            "lost_by_tier": s["lost_by_tier"],
            "cause_a_by_tier": s["cause_a_by_tier"],
            "cause_b_by_tier": s["cause_b_by_tier"],
            "treated_share_by_tier": s["treated_share_by_tier"],
            "total_cost": s["total_cost"], "cost_per_treated": s["cost_per_treated"],
            "spare_capacity_built": s["spare_capacity_built"],
            "facilities_open": s["facilities_open"], "open_facilities": s["open_facilities"],
            "capacity": s["capacity"]}


# ---------------------------------------------------------------------------
# On-time-design sweep for one setting (used by E1, E2, F1, E5, SENS)
# ---------------------------------------------------------------------------

def solve_on_time(inst, Ytr, Btr, tiers, s):
    return solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0,
                            mip_gap=s["mip_gap"], time_limit=s["time_limit"])


def setting_sweep(mult, label, s):
    """On-time design (solved once) simulated across decline speeds, with and
    without the sickest-first lever. Returns rows keyed by kappa."""
    inst, tiers, tidx = scaled_case_study_instance(mult=mult, s_max=S_MAX_REAL)
    Ytr, Btr, Yte, Bte, U = _draws(inst, s["n_train"], s["n_oos"])
    D = solve_on_time(inst, Ytr, Btr, tiers, s)
    rows = []
    for k in s["kappas"]:
        on = _sim(inst, D, tiers, tidx, Yte, Bte, U, k, priority=False)
        pr = _sim(inst, D, tiers, tidx, Yte, Bte, U, k, priority=True)
        rows.append({"kappa": k, "on_time": _plan_summary(on),
                     "sickest_first": _plan_summary(pr)})
        print(f"    [{label}] speed={k:<4} H-lost on-time={on['high_urgency_lost']:.2f} "
              f"(wait={on['cause_a_by_tier']['H']:.2f} fail={on['cause_b_by_tier']['H']:.2f}) "
              f"| sickest-first={pr['high_urgency_lost']:.2f}")
    return {"label": label, "mult": mult, "n_patients": inst.n_patients,
            "s_max": S_MAX_REAL, "D_on_time": {"z": D["z"].tolist(), "C": D["C"].tolist()},
            "rows": rows, "_ctx": (inst, tiers, tidx, D, Yte, Bte, U)}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def three_plans(s):
    """E3: on-time / decline-aware(spare) / sickest-first at the full 150p network."""
    inst, tiers, tidx = scaled_case_study_instance(mult=3, s_max=S_MAX_REAL)
    Ytr, Btr, Yte, Bte, U = _draws(inst, s["n_train"], s["n_oos"])
    D_exo = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0,
                             mip_gap=s["mip_gap"], time_limit=s["time_limit"])
    D_endo = solve_dynamic_sp_endogenous(inst, Ytr, Btr, tiers, kappa=CALIB_KAPPA,
                                         gamma=GAMMA, mip_gap=s["mip_gap"],
                                         time_limit=s["time_limit"], max_iter=3)
    plans = {
        "on_time": _plan_summary(_sim(inst, D_exo, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False)),
        "decline_aware": _plan_summary(_sim(inst, D_endo, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False)),
        "sickest_first": _plan_summary(_sim(inst, D_endo, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True)),
    }
    base, pri = plans["decline_aware"], plans["sickest_first"]
    h_saved = base["high_urgency_lost"] - pri["high_urgency_lost"]
    low_extra = ((pri["lost_by_tier"]["M"] + pri["lost_by_tier"]["L"])
                 - (base["lost_by_tier"]["M"] + base["lost_by_tier"]["L"]))
    tradeoff = {"high_urgency_saved": h_saved, "lower_urgency_extra_lost": low_extra,
                "saved_per_extra": (h_saved / low_extra) if low_extra > 1e-9 else float("inf")}
    return {"kappa": CALIB_KAPPA, "n_patients": inst.n_patients,
            "D_exo": {"z": D_exo["z"].tolist(), "C": D_exo["C"].tolist()},
            "D_endo": {"z": D_endo["z"].tolist(), "C": D_endo["C"].tolist()},
            "plans": plans, "tradeoff": tradeoff}


def cap_sweep(s):
    """E4: capacity cap {40,55,75} at the real rate (100-patient network)."""
    rows = []
    for smax in [40, 55, 75]:
        inst, tiers, tidx = scaled_case_study_instance(mult=2, s_max=smax)
        Ytr, Btr, Yte, Bte, U = _draws(inst, s["n_train"], s["n_oos"])
        D = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0,
                             mip_gap=s["mip_gap"], time_limit=s["time_limit"])
        on = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False)
        pr = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True)
        rows.append({"s_max": smax, "n_patients": inst.n_patients,
                     "on_time": _plan_summary(on), "sickest_first": _plan_summary(pr)})
        print(f"    s_max={smax}: on-time H-lost={on['high_urgency_lost']:.2f} "
              f"open={on['open_facilities']}")
    return {"kappa": CALIB_KAPPA, "rows": rows}


def e5_bwd_robustness(busy_ctx):
    """E5: sweep the normal-wait mortality anchor; report how the 'share from the
    normal wait' and the 'sickest-first reduction' move. Reuses the busy design."""
    inst, tiers, tidx, D, Yte, Bte, U = busy_ctx
    rows = []
    for mparam in BWD_MULTIPLIERS:
        bwd = {t: BASE_WAIT_DEATH_6WK[t] * mparam for t in ("H", "M", "L")}
        on = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False, base_wait_death=bwd)
        pr = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True, base_wait_death=bwd)
        a, b = on["cause_a_by_tier"]["H"], on["cause_b_by_tier"]["H"]
        share = a / (a + b) if (a + b) > 1e-9 else 0.0
        rows.append({"multiplier": mparam, "base_wait_death": bwd,
                     "share_from_normal_wait_H": share,
                     "on_time_H_lost": on["high_urgency_lost"],
                     "sickest_first_H_lost": pr["high_urgency_lost"],
                     "sickest_first_reduction": on["high_urgency_lost"] - pr["high_urgency_lost"]})
        print(f"    bwd x{mparam}: H-share-wait={share:.0%} "
              f"on-time={on['high_urgency_lost']:.2f} sickest={pr['high_urgency_lost']:.2f}")
    return {"basis": "H/M/L 6-week normal-wait mortality {0.15,0.05,0.02} scaled by multiplier",
            "rows": rows}


def gamma_grid(busy_ctx):
    """Decline-shape (gamma) sensitivity on the busy design at the real rate."""
    inst, tiers, tidx, D, Yte, Bte, U = busy_ctx
    rows = []
    for g in GAMMA_GRID:
        on = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False, gamma=g)
        pr = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True, gamma=g)
        rows.append({"gamma": g, "lambda_days": calibrate_lambda(g),
                     "on_time_H_lost": on["high_urgency_lost"],
                     "sickest_first_H_lost": pr["high_urgency_lost"],
                     "share_from_normal_wait_H": on["cause_a_by_tier"]["H"] /
                     max(on["high_urgency_lost"], 1e-9)})
        print(f"    gamma={g}: on-time H-lost={on['high_urgency_lost']:.2f} sickest={pr['high_urgency_lost']:.2f}")
    return {"rows": rows}


def hr_grid(busy_ctx):
    """Tier hazard-ratio spread sensitivity on the busy design at the real rate."""
    inst, tiers, tidx, D, Yte, Bte, U = busy_ctx
    rows = []
    for name, hrt in HR_GRIDS.items():
        on = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False, hr_tier=hrt)
        pr = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True, hr_tier=hrt)
        rows.append({"spread": name, "hr_tier": hrt,
                     "on_time_H_lost": on["high_urgency_lost"],
                     "sickest_first_H_lost": pr["high_urgency_lost"]})
        print(f"    HR-{name}: on-time H-lost={on['high_urgency_lost']:.2f} sickest={pr['high_urgency_lost']:.2f}")
    return {"rows": rows}


def validation(busy_ctx, s):
    """V1 nesting, V2 priority-OFF equivalence, V3 planning vs detailed sim."""
    inst, tiers, tidx, D, Yte, Bte, U = busy_ctx
    # V1: decline speed 0 -> no normal-wait deaths (cause a == 0), losses only
    # from failed batches that cannot be recollected -> reproduces the base model.
    s0 = _sim(inst, D, tiers, tidx, Yte, Bte, U, 0.0, False)
    v1 = {"cause_a_total": sum(s0["cause_a_by_tier"].values()),
          "high_urgency_lost": s0["high_urgency_lost"],
          "total_cost": s0["total_cost"],
          "cause_a_by_tier": s0["cause_a_by_tier"]}
    # Toy nesting gate (dynamic_sp kappa=0 == yield_sp_v1) as a subprocess.
    try:
        r = subprocess.run([sys.executable, "test_nesting.py"], cwd=_ROOT,
                           capture_output=True, text=True, timeout=300)
        v1["toy_nesting_test"] = {"returncode": r.returncode,
                                  "tail": r.stdout.strip().splitlines()[-3:] if r.stdout else []}
    except Exception as exc:  # pragma: no cover
        v1["toy_nesting_test"] = {"error": str(exc)}
    # V2: priority=False is the no-lever model. Simulate the same design twice with
    # priority off; must be identical (byte-for-byte) to the on-time plan row.
    a = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False)
    b = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False)
    v2 = {"identical": (a["high_urgency_lost"] == b["high_urgency_lost"]
                        and a["total_cost"] == b["total_cost"]),
          "high_urgency_lost": a["high_urgency_lost"], "total_cost": a["total_cost"]}
    # V3: planning-model estimate vs detailed simulation (spare-capacity ladder).
    # Use the 50-patient network, which has capacity headroom to add spare and so
    # produces a proper ladder (the 150-patient network is saturated at s_max).
    from deterioration_experiment import planned_vs_simulated_deaths
    inst2, t2, ti2 = scaled_case_study_instance(mult=1, s_max=S_MAX_REAL)
    try:
        pvs = planned_vs_simulated_deaths(inst2, t2, ti2, CALIB_KAPPA, GAMMA)
    except Exception as exc:  # pragma: no cover
        pvs = {"error": str(exc)}
    return {"nesting": v1, "priority_off": v2, "planned_vs_simulated": pvs}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(quick=False):
    t0 = time.perf_counter()
    s = _settings(quick)
    _RAW.mkdir(parents=True, exist_ok=True)
    inst0 = build_case_study_instance()

    payload = {
        "meta": {"calib_kappa": CALIB_KAPPA, "gamma": GAMMA, "s_max_real": S_MAX_REAL,
                 "seed_train": SEED_TRAIN, "seed_oos": SEED_OOS,
                 "n_train": s["n_train"], "n_oos": s["n_oos"], "mip_gap": s["mip_gap"],
                 "backend": "HiGHS", "quick": quick,
                 "lambda_by_gamma": {str(g): calibrate_lambda(g) for g in GAMMA_GRID},
                 "hr_tier": dict(HR_TIER),
                 "base_wait_death_6wk": dict(BASE_WAIT_DEATH_6WK),
                 "clearing": {"tau_proc": DEFAULT_CLEARING.tau_proc,
                              "breakpoints": list(DEFAULT_CLEARING.breakpoints),
                              "inc_slopes": list(DEFAULT_CLEARING.inc_slopes)},
                 "instance": {"n_patients_base": inst0.n_patients,
                              "n_facilities": inst0.n_facilities,
                              "f": inst0.f.tolist(), "pi": inst0.pi.tolist(),
                              "c": inst0.c.tolist(), "p": inst0.p.tolist(),
                              "beta": inst0.beta.tolist(),
                              "rho_cancel": inst0.rho_cancel.tolist(),
                              "rho_leuk": inst0.rho_leuk,
                              "shelf_life": inst0.shelf_life, "mnf": inst0.mnf}},
    }

    print("[busy 150-patient network]")
    busy = setting_sweep(3, "busy-150", s)
    print("[low-demand 50-patient network]")
    low = setting_sweep(1, "low-50", s)
    busy_ctx = busy.pop("_ctx"); low.pop("_ctx")
    payload["busy"] = busy
    payload["low"] = low

    print("[E3 three plans @150p]")
    payload["three_plans"] = three_plans(s)
    print("[E4 capacity-cap sweep]")
    payload["cap_sweep"] = cap_sweep(s)
    print("[E5 normal-wait mortality robustness]")
    payload["e5_bwd"] = e5_bwd_robustness(busy_ctx)
    print("[SENS decline-shape gamma grid]")
    payload["gamma_grid"] = gamma_grid(busy_ctx)
    print("[SENS tier hazard-ratio grid]")
    payload["hr_grid"] = hr_grid(busy_ctx)
    print("[VALIDATION]")
    payload["validation"] = validation(busy_ctx, s)

    payload["meta"]["compute_seconds"] = time.perf_counter() - t0

    # Raw dumps (per experiment) + a combined raw log.
    for name in ("busy", "low", "three_plans", "cap_sweep", "e5_bwd",
                 "gamma_grid", "hr_grid", "validation"):
        with open(_RAW / f"{name}.json", "w") as fh:
            json.dump(payload[name], fh, indent=2)
    with open(_RAW / "run_meta.json", "w") as fh:
        json.dump(payload["meta"], fh, indent=2)
    print(f"\nExperiments done in {payload['meta']['compute_seconds']:.0f}s; "
          f"raw assets in {_RAW}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    p = run(quick=args.quick)
    print(json.dumps(p["three_plans"]["tradeoff"], indent=2))
