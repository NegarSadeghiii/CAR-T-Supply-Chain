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
from patient_simulator import (simulate_patients, WAIT_DEATH_REF,  # noqa: E402
                               ELIGIBILITY_CUTOFF, T_V2V_DEFAULT, DELTA_REMAKE_DEFAULT)

# ---- fixed experiment settings (seeded / reproducible) ----------------------
CALIB_KAPPA = 1.0                # decline speed matched to Dulobdas 2025 PFS HR 1.64
GAMMA = 1.0                      # central decline-curve shape
S_MAX_REAL = 75                  # Wan et al. 2026 Maxcap
T_V2V = T_V2V_DEFAULT            # base normal vein-to-vein wait (28 d)
DELTA_REMAKE = DELTA_REMAKE_DEFAULT  # base re-make increment (12 d, observed)
T_V2V_GRID = [21, 28, 35, 42]    # normal-wait sweep (3-6 weeks)
FAILURE_GRID = [0.05, 0.10, 0.20, 0.28]  # target mean failure rate (Issue 6)
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
# E6 priority-rule family (Tseng et al. 2024). FIFO baseline + Threshold-X.
THRESHOLD_GRID = [75, 80, 85, 90, 95, 100]
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
         base_wait_death=None, hr_tier=None, gamma=GAMMA,
         t_v2v=T_V2V, delta_remake=DELTA_REMAKE):
    return simulate_patients(inst, design, tiers, tidx, Yte, Bte, U,
                             kappa=kappa, gamma=gamma, priority=priority,
                             t_v2v=t_v2v, delta_remake=delta_remake,
                             base_wait_death=base_wait_death, hr_tier=hr_tier)


def _scaled_failure_instance(mult, s_max, target_fail):
    """Copy the scaled instance and rescale per-facility failure so the mean
    failure rate is `target_fail`, preserving the k1<k2<k3 ordering (Issue 6)."""
    import copy
    inst, tiers, tidx = scaled_case_study_instance(mult=mult, s_max=s_max)
    if target_fail is not None:
        inst = copy.deepcopy(inst)
        u = 1.0 - np.asarray(inst.p, float)          # base failure per facility
        scale = target_fail / u.mean()
        inst.p = np.clip(1.0 - u * scale, 0.01, 0.999)
    return inst, tiers, tidx


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
        bwd = {t: WAIT_DEATH_REF[t] * mparam for t in ("H", "M", "L")}
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


def _tseng_metrics(surv_rule, surv_fifo, cutoff=ELIGIBILITY_CUTOFF, eps=1e-6):
    """Tseng-style survival-rate metrics for one rule vs FIFO, over all patients
    (each patient's survival rate at therapy = expected survival at realized wait)."""
    sr = np.asarray(surv_rule, float)
    sf = np.asarray(surv_fifo, float)
    inc = sr > sf + eps
    dec = sr < sf - eps
    become_elig = (sf < cutoff) & (sr >= cutoff)
    become_inelig = (sf >= cutoff) & (sr < cutoff)
    return {
        "n_eligible": int((sr > cutoff).sum()),
        "avg_survival": float(sr.mean()), "sd_survival": float(sr.std()),
        "max_survival": float(sr.max()), "min_survival": float(sr.min()),
        "n_increased": int(inc.sum()),
        "avg_increase": float((sr[inc] - sf[inc]).mean()) if inc.any() else 0.0,
        "n_become_eligible": int(become_elig.sum()),
        "n_decreased": int(dec.sum()),
        "avg_decrease": float((sf[dec] - sr[dec]).mean()) if dec.any() else 0.0,
        "n_become_ineligible": int(become_inelig.sum()),
    }


def e6_priority_rules(s):
    """E6: FIFO vs Threshold-X priority-rule comparison in the busy 150p network at
    the real decline rate. Tseng survival metrics + our tier / cost / cause columns.
    The FIFO survival rates are the ordering signal AND the comparison baseline."""
    inst, tiers, tidx = scaled_case_study_instance(mult=3, s_max=S_MAX_REAL)
    Ytr, Btr, Yte, Bte, U = _draws(inst, s["n_train"], s["n_oos"])
    D = solve_on_time(inst, Ytr, Btr, tiers, s)
    fifo = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, priority=False)
    signal = fifo["survival_rate_by_patient"]           # FIFO survival = ordering signal
    surv_fifo = fifo["survival_rate_by_patient"]

    def our_cols(sim):
        return {"high_urgency_lost": sim["high_urgency_lost"],
                "lost_by_tier": sim["lost_by_tier"],
                "cause_a_by_tier": sim["cause_a_by_tier"],
                "cause_b_by_tier": sim["cause_b_by_tier"],
                "treated_share_by_tier": sim["treated_share_by_tier"],
                "cost_per_treated": sim["cost_per_treated"],
                "total_cost": sim["total_cost"]}

    rules = []
    m = _tseng_metrics(surv_fifo, surv_fifo)
    rules.append({"rule": "FIFO", "threshold": None, **m, **our_cols(fifo)})
    print(f"    FIFO: elig={m['n_eligible']} avg-surv={m['avg_survival']:.3f} H-lost={fifo['high_urgency_lost']:.2f}")
    for X in THRESHOLD_GRID:
        sim = simulate_patients(inst, D, tiers, tidx, Yte, Bte, U, kappa=CALIB_KAPPA,
                                gamma=GAMMA, rule="THRESHOLD", threshold=X / 100.0,
                                signal=signal)
        mt = _tseng_metrics(sim["survival_rate_by_patient"], surv_fifo)
        rules.append({"rule": f"Threshold-{X}", "threshold": X, **mt, **our_cols(sim)})
        print(f"    Th-{X}: elig={mt['n_eligible']} avg-surv={mt['avg_survival']:.3f} "
              f"inc={mt['n_increased']} dec={mt['n_decreased']} H-lost={sim['high_urgency_lost']:.2f}")
    return {"n_patients": inst.n_patients, "kappa": CALIB_KAPPA,
            "eligibility_cutoff": ELIGIBILITY_CUTOFF, "thresholds": THRESHOLD_GRID,
            "D_on_time": {"z": D["z"].tolist(), "C": D["C"].tolist()},
            "fifo_survival_by_patient": surv_fifo, "tiers": tiers, "rules": rules}


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


def tv2v_sweep(s):
    """Issue 1: normal vein-to-vein wait sweep {21,28,35,42} d (busy 150p). On-time
    design solved once; simulate FIFO and best-threshold at each T_V2V."""
    inst, tiers, tidx = scaled_case_study_instance(mult=3, s_max=S_MAX_REAL)
    Ytr, Btr, Yte, Bte, U = _draws(inst, s["n_train"], s["n_oos"])
    D = solve_on_time(inst, Ytr, Btr, tiers, s)
    rows = []
    for tv in T_V2V_GRID:
        on = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False, t_v2v=tv)
        pr = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True, t_v2v=tv)
        a, b = on["cause_a_by_tier"]["H"], on["cause_b_by_tier"]["H"]
        rows.append({"t_v2v": tv, "on_time_H_lost": on["high_urgency_lost"],
                     "sickest_first_H_lost": pr["high_urgency_lost"],
                     "share_from_normal_wait_H": a / (a + b) if (a + b) > 1e-9 else 0.0,
                     "cost_per_treated": on["cost_per_treated"]})
        print(f"    T_V2V={tv}d: on-time H-lost={on['high_urgency_lost']:.2f} "
              f"wait-share={100*rows[-1]['share_from_normal_wait_H']:.0f}% sickest={pr['high_urgency_lost']:.2f}")
    return {"grid": T_V2V_GRID, "rows": rows}


def remake_delay_compare(s):
    """Issue 4: DELTA_REMAKE = 12 d (observed increment) vs = T_V2V (full extra
    cycle), busy 150p at the real rate. Report high-urgency lost & wait share."""
    inst, tiers, tidx = scaled_case_study_instance(mult=3, s_max=S_MAX_REAL)
    Ytr, Btr, Yte, Bte, U = _draws(inst, s["n_train"], s["n_oos"])
    D = solve_on_time(inst, Ytr, Btr, tiers, s)
    rows = []
    for name, dr in (("observed_12d", DELTA_REMAKE), ("full_cycle_T_V2V", T_V2V)):
        on = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False, delta_remake=dr)
        pr = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True, delta_remake=dr)
        a, b = on["cause_a_by_tier"]["H"], on["cause_b_by_tier"]["H"]
        rows.append({"setting": name, "delta_remake": dr,
                     "on_time_H_lost": on["high_urgency_lost"],
                     "sickest_first_H_lost": pr["high_urgency_lost"],
                     "H_normal_wait": a, "H_after_failure": b,
                     "share_from_normal_wait_H": a / (a + b) if (a + b) > 1e-9 else 0.0,
                     "cost_per_treated": on["cost_per_treated"]})
        print(f"    remake={name} ({dr:.0f}d): on-time H-lost={on['high_urgency_lost']:.2f} "
              f"(wait {a:.2f} / fail {b:.2f}) sickest={pr['high_urgency_lost']:.2f}")
    return {"rows": rows}


def failure_tv2v_heatmap(s):
    """Issue 6: two-way sweep of manufacturing failure rate x normal wait T_V2V.
    For each cell: share of high-urgency losses from the normal wait, high-urgency
    lost FIFO vs best threshold rule, cost per patient. One design solved per
    failure rate (T_V2V does not enter the design)."""
    cells = []
    for f in FAILURE_GRID:
        inst, tiers, tidx = _scaled_failure_instance(3, S_MAX_REAL, f)
        Ytr, Btr, Yte, Bte, U = _draws(inst, s["n_train"], s["n_oos"])
        D = solve_on_time(inst, Ytr, Btr, tiers, s)
        for tv in T_V2V_GRID:
            fifo = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, False, t_v2v=tv)
            # best threshold rule = pure lowest-survival-first (Threshold-100%).
            best = _sim(inst, D, tiers, tidx, Yte, Bte, U, CALIB_KAPPA, True, t_v2v=tv)
            a, b = fifo["cause_a_by_tier"]["H"], fifo["cause_b_by_tier"]["H"]
            cells.append({"failure_rate": f, "t_v2v": tv,
                          "share_from_normal_wait_H": a / (a + b) if (a + b) > 1e-9 else 0.0,
                          "H_lost_fifo": fifo["high_urgency_lost"],
                          "H_lost_best_rule": best["high_urgency_lost"],
                          "cost_per_treated": fifo["cost_per_treated"]})
        print(f"    failure={100*f:.0f}%: wait-share@T_V2V "
              f"{[round(100*c['share_from_normal_wait_H']) for c in cells if c['failure_rate']==f]}%")
    return {"failure_grid": FAILURE_GRID, "t_v2v_grid": T_V2V_GRID, "cells": cells}


def implemented_table():
    """Issue 5: explicit implemented / not-implemented statement."""
    return {"items": [
        {"feature": "Per-class clinical deadline tau_u", "implemented": False,
         "note": "No explicit per-tier deadline; a loss is any patient not treated. "
                 "'Treated within deadline' means simply treated (not lost)."},
        {"feature": "Transport time t_im in the deterioration wait", "implemented": False,
         "note": "Transport time enters only the shelf-life feasibility filter "
                 "(case_study.py); it does not add to the deterioration wait."},
        {"feature": "Subcontracting to m' != m", "implemented": True,
         "note": "Recourse LP includes subcontracting at rho_sub[m,m'] = 1.15 c_m' "
                 "(off-diagonal), solved for failed batches."},
    ]}


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
                 "wait_death_ref": dict(WAIT_DEATH_REF), "t_ref": 42.0,
                 "t_v2v": T_V2V, "delta_remake": DELTA_REMAKE,
                 "t_v2v_grid": T_V2V_GRID, "failure_grid": FAILURE_GRID,
                 "clearing": {"tau_proc": DEFAULT_CLEARING.tau_proc,
                              "breakpoints": list(DEFAULT_CLEARING.breakpoints),
                              "inc_slopes": list(DEFAULT_CLEARING.inc_slopes)},
                 "instance": {"n_patients_base": inst0.n_patients,
                              "n_facilities": inst0.n_facilities,
                              "f": inst0.f.tolist(), "pi": inst0.pi.tolist(),
                              "c": inst0.c.tolist(), "p": inst0.p.tolist(),
                              "beta": inst0.beta.tolist(),
                              "rho_cancel": inst0.rho_cancel.tolist(),
                              "rho_leuk": inst0.rho_leuk, "mnf": inst0.mnf}},
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
    print("[E6 priority-rule comparison]")
    payload["e6"] = e6_priority_rules(s)
    print("[Issue 1: T_V2V sweep]")
    payload["tv2v_sweep"] = tv2v_sweep(s)
    print("[Issue 4: re-make delay comparison]")
    payload["remake_compare"] = remake_delay_compare(s)
    print("[Issue 6: failure x T_V2V heatmap]")
    payload["failure_tv2v"] = failure_tv2v_heatmap(s)
    payload["implemented"] = implemented_table()
    print("[VALIDATION]")
    payload["validation"] = validation(busy_ctx, s)

    payload["meta"]["compute_seconds"] = time.perf_counter() - t0

    # Raw dumps (per experiment) + a combined raw log.
    for name in ("busy", "low", "three_plans", "cap_sweep", "e5_bwd",
                 "gamma_grid", "hr_grid", "e6", "tv2v_sweep", "remake_compare",
                 "failure_tv2v", "implemented", "validation"):
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
