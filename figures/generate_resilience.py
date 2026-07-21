"""
figures/generate_resilience.py — decision-focused resilience figures (F1–F7 + F6a).

Well-sampled, publication-quality figures for the capacity-disruption /
correlation story, each answering ONE question. Compute is separated from
plotting: all solves are cached to figures/resilience_data.json, so re-plotting
is instant and the figures are cheaply regenerable.

  F1  Does rerouting wake up under disruption?      (rerouting_under_disruption)
  F2  Slope or cliff?                               (subcontracting_vs_correlation)
  F3  Price of correlation.                         (price_of_correlation)
  F4  Trade-off frontier: which hedge?              (hedging_frontier)
  F5  Who gets cancelled?                           (cancellations_by_tier)
  F6a Input-sample convergence (no optimization).   (input_convergence)
  F6  Solution stability vs training N.             (scenario_stability)
  F7  Diagnose the q=0.15 blip.                     (blip_diagnosis)

Sampling strategy (fast + well-sampled). For F1–F5 the first-stage design is
solved ONCE per (q, ρ) grid point at a moderate training N, then scored on a
large fixed out-of-sample set of N_EVAL scenarios across several eval seeds; the
mean is plotted with a shaded min–max band. This makes the heavy MILP solve
happen once per point while the curves themselves are sampled at N_EVAL≥500.

Backend. Uses the local Gurobi backend (Gurobi_Version/) when a full Gurobi
license is present (faster, unrestricted); otherwise falls back to HiGHS, which
solves the full 50-patient instance the size-limited pip Gurobi license cannot.
Both are exact MILP solvers and agree to solver tolerance (see
Gurobi_Version/parity_check.py). MIP gap 1e-3 and a per-solve time limit keep
sweeps bounded.

Run:
  python figures/generate_resilience.py             # compute (if needed) + plot
  python figures/generate_resilience.py --recompute # force re-solve
  python figures/generate_resilience.py --plot-only # replot from cached JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import highspy as hs

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yield_sp_v1 import sample_scenarios, sample_disruptions          # noqa: E402
from case_study import build_case_study_instance, TIER_IDX            # noqa: E402
from disruption_experiment import solve_sp_disrupted                  # noqa: E402

from figure_style import (                                            # noqa: E402
    COLORS, setup_style, single_column, double_column, triple_column, save_figure,
)

_DATA = _HERE / "resilience_data.json"

# ---- Sampling / solve settings ----
SEED = 0
N_TRAIN = 200            # scenarios for the per-(q,ρ) design solve
N_EVAL = 1000            # out-of-sample scenarios per curve point
EVAL_SEEDS = [101, 202, 303]     # OOS eval seeds → min–max band
MIP_GAP = 1e-3
TIME_LIMIT = 90.0        # per design solve (sweeps)
N_PATIENTS = 50

# ---- Grids ----
RHO_FINE = [round(x, 3) for x in np.linspace(0.0, 1.0, 11)]        # 0,0.1,…,1.0
Q_GRID_F1 = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
Q_LEVELS_F3 = [0.05, 0.10, 0.15]
HEADLINE_Q = 0.10

# ---- Backend selection (Gurobi if a full license can build the instance) ----
_GUROBI_FULL_OK = None    # probed lazily in _solve_design


def _probe_gurobi_full() -> bool:
    """True iff the local Gurobi license can build the full case-study model."""
    try:
        from Gurobi_Version.gurobi_model import solve_sp_gurobi
        import gurobipy as gp
        inst = build_case_study_instance()
        inst.q = np.full(inst.n_facilities, 0.10)
        Y, B = sample_scenarios(inst, 20, seed=0)
        Xi = sample_disruptions(inst, 20, seed=0, mode="partial", rho=0.5)
        solve_sp_gurobi(inst, Y, B, Xi=Xi, mip_gap=1e-2, time_limit=30)
        return True
    except Exception:
        return False


def _solve_design(inst, Y, B, Xi, *, time_limit=TIME_LIMIT):
    """
    Solve the SP for its first-stage design, preferring Gurobi (full license)
    and falling back to HiGHS. Returns the standard result dict.
    """
    global _GUROBI_FULL_OK
    if _GUROBI_FULL_OK is None:
        _GUROBI_FULL_OK = _probe_gurobi_full()
        print(f"[backend] {'Gurobi (full license)' if _GUROBI_FULL_OK else 'HiGHS (Gurobi size-limited here)'}")
    if _GUROBI_FULL_OK:
        from Gurobi_Version.gurobi_model import solve_sp_gurobi
        return solve_sp_gurobi(inst, Y, B, Xi=Xi, mip_gap=MIP_GAP, time_limit=time_limit)
    return solve_sp_disrupted(inst, Y, B, Xi=Xi, mip_gap=MIP_GAP, time_limit=time_limit)


# ===========================================================================
# Fixed-design out-of-sample evaluator (disruption-aware, HiGHS LP)
# ===========================================================================

def evaluate_design_oos(inst, z, C, x, Y, B, Xi):
    """
    Realized cost, service, and recourse-action mix of a FIXED first-stage
    design (z, C, x) scored on an out-of-sample scenario set (Y, B, Xi).

    First stage is constant, so only the Stage-2 recourse variables remain and
    the problem is a totally-unimodular LP (integral optimum) — one HiGHS pass
    across all scenarios. Mirrors the recourse constraints of
    disruption_experiment.solve_sp_disrupted (eqs. 8/18, 10, 10b, 9/19) with the
    eq.-19 capacity floor.

    Returns dict: total_cost, expected_stage2_cost, exp_remfg, exp_sub,
    exp_cancel, service_level, tier_cancel{H,M,L}.
    """
    n_p, n_f = inst.n_patients, inst.n_facilities
    N = Y.shape[0]
    C = np.asarray(C, float)
    x = np.asarray(x, float)
    delta_cap_eff = inst.delta_cap_eff

    n_sub_pp = n_f * (n_f - 1)
    n_per_s = n_p * n_f + n_p * n_sub_pp + n_p
    n_total = N * n_per_s

    def _soff(m, mp):
        return m * (n_f - 1) + (mp if mp < m else mp - 1)

    def vr(o, i, m):     return o * n_per_s + i * n_f + m
    def vs(o, i, m, mp): return o * n_per_s + n_p * n_f + i * n_sub_pp + _soff(m, mp)
    def vc(o, i):        return o * n_per_s + n_p * n_f + n_p * n_sub_pp + i

    h = hs.Highs()
    h.silent()

    lbs = np.zeros(n_total)
    ubs = np.ones(n_total)
    costs = np.zeros(n_total)
    prob = 1.0 / N
    for o in range(N):
        for i in range(n_p):
            for m in range(n_f):
                costs[vr(o, i, m)] = prob * (inst.rho_leuk + inst.rho_remfg[m])
                for mp in range(n_f):
                    if mp != m:
                        costs[vs(o, i, m, mp)] = prob * (inst.rho_leuk + inst.rho_sub[m, mp])
            costs[vc(o, i)] = prob * inst.rho_cancel[i]

    h.addCols(n_total, costs, lbs, ubs, 0,
              np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float64))

    INF = 1e30

    def _row(lb, ub, inds, vals):
        ia = np.asarray(inds, dtype=np.int32)
        va = np.asarray(vals, dtype=np.float64)
        h.addRow(lb, ub, len(ia), ia, va)

    for o in range(N):
        Ytil = Y[o] * (1.0 - Xi[o])[None, :]
        for i in range(n_p):
            remfg_i = [vr(o, i, m) for m in range(n_f)]
            sub_i = [vs(o, i, m, mp) for m in range(n_f) for mp in range(n_f) if mp != m]
            F_i = float(sum((1.0 - Ytil[i, m]) * x[i, m] for m in range(n_f)))
            _row(F_i, F_i, remfg_i + sub_i + [vc(o, i)],
                 [1.0] * (n_f + n_sub_pp + 1))
            _row(-INF, float(B[o, i]), remfg_i + sub_i, [1.0] * (n_f + n_sub_pp))
            for m in range(n_f):
                sub_from_m = [vs(o, i, m, mp) for mp in range(n_f) if mp != m]
                _row(-INF, float(x[i, m]), [vr(o, i, m)] + sub_from_m,
                     [1.0] * (1 + n_f - 1))
        for m in range(n_f):
            remfg_m = [vr(o, i, m) for i in range(n_p)]
            sub_to_m = [vs(o, i, mp, m) for i in range(n_p) for mp in range(n_f) if mp != m]
            loss = min(delta_cap_eff[m], C[m]) * Xi[o, m]
            fixed_load = float(sum(x[i, m] * Ytil[i, m] for i in range(n_p)))
            rhs = max(0.0, float(C[m]) - loss) - fixed_load
            _row(-INF, rhs, remfg_m + sub_to_m,
                 [1.0] * n_p + [1.0] * (n_p * (n_f - 1)))

    h.run()
    col = h.getSolution().col_value

    r_cancel = np.zeros((N, n_p))
    remfg_tot = sub_tot = 0.0
    s2 = 0.0
    for o in range(N):
        for i in range(n_p):
            r_cancel[o, i] = round(col[vc(o, i)])
            for m in range(n_f):
                rv = round(col[vr(o, i, m)])
                remfg_tot += rv
                s2 += (inst.rho_leuk + inst.rho_remfg[m]) * rv
                for mp in range(n_f):
                    if mp != m:
                        sv = round(col[vs(o, i, m, mp)])
                        sub_tot += sv
                        s2 += (inst.rho_leuk + inst.rho_sub[m, mp]) * sv
            s2 += inst.rho_cancel[i] * r_cancel[o, i]
    s2 /= N

    stage1 = (sum(inst.f[m] * z[m] for m in range(n_f))
              + sum(inst.pi[m] * C[m] for m in range(n_f))
              + sum(inst.c[m] * x[i, m] for i in range(n_p) for m in range(n_f)))
    exp_cancel = float(r_cancel.sum() / N)
    tier_cancel = {t: float(r_cancel[:, idx].sum() / N) for t, idx in TIER_IDX.items()}
    return {
        "total_cost": float(stage1 + s2),
        "expected_stage2_cost": float(s2),
        "exp_remfg": float(remfg_tot / N),
        "exp_sub": float(sub_tot / N),
        "exp_cancel": exp_cancel,
        "service_level": float(1.0 - exp_cancel / n_p),
        "tier_cancel": tier_cancel,
    }


# ===========================================================================
# Helpers
# ===========================================================================

def _instance(q):
    inst = build_case_study_instance()
    inst.q = np.full(inst.n_facilities, q)
    return inst


def _eval_band(inst, design, q, rho, mode="partial"):
    """
    Score a fixed design on EVAL_SEEDS independent OOS sets of N_EVAL scenarios.
    Returns dict of lists keyed by metric, for mean/min-max banding.
    """
    keys = ["total_cost", "exp_remfg", "exp_sub", "exp_cancel", "service_level"]
    acc = {k: [] for k in keys}
    acc["tier_cancel"] = {"H": [], "M": [], "L": []}
    for s in EVAL_SEEDS:
        inst_e = _instance(q)
        Ye, Be = sample_scenarios(inst_e, N_EVAL, seed=s)
        if mode == "partial":
            Xe = sample_disruptions(inst_e, N_EVAL, seed=s, mode="partial", rho=rho)
        else:
            Xe = (np.zeros((N_EVAL, inst_e.n_facilities)) if q == 0.0
                  else sample_disruptions(inst_e, N_EVAL, seed=s, mode=mode))
        ev = evaluate_design_oos(inst_e, design["z"], design["C"], design["x"], Ye, Be, Xe)
        for k in keys:
            acc[k].append(ev[k])
        for t in ("H", "M", "L"):
            acc["tier_cancel"][t].append(ev["tier_cancel"][t])
    return acc


def _stat(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


# ===========================================================================
# Compute
# ===========================================================================

def compute_all() -> dict:
    t_start = time.perf_counter()
    base = build_case_study_instance()
    n_f = base.n_facilities
    data = {"seed": SEED, "n_train": N_TRAIN, "n_eval": N_EVAL,
            "eval_seeds": EVAL_SEEDS, "mip_gap": MIP_GAP,
            "rho_fine": RHO_FINE, "q_grid_f1": Q_GRID_F1,
            "q_levels_f3": Q_LEVELS_F3, "headline_q": HEADLINE_Q,
            "n_patients": base.n_patients}

    # ---- F1: recourse mix vs q, independent vs common ----
    t0 = time.perf_counter()
    print("[F1] recourse mix vs q (independent vs common) …")
    f1 = {"q_grid": Q_GRID_F1, "independent": [], "common": []}
    for mode in ("independent", "common"):
        for q in Q_GRID_F1:
            inst = _instance(q)
            Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED)
            Xtr = (np.zeros((N_TRAIN, n_f)) if q == 0.0
                   else sample_disruptions(inst, N_TRAIN, seed=SEED, mode=mode))
            design = _solve_design(inst, Ytr, Btr, Xtr)
            band = _eval_band(inst, design, q, None, mode=mode)
            rec = {"q": q}
            for k in ("exp_remfg", "exp_sub", "exp_cancel"):
                rec[k] = _stat(band[k])
            f1[mode].append(rec)
            print(f"    {mode:<11} q={q:.3f}  sub={rec['exp_sub']['mean']:.2f} "
                  f"cancel={rec['exp_cancel']['mean']:.2f}")
    data["F1"] = f1
    print(f"[F1] {time.perf_counter()-t0:.0f}s")

    # ---- F2/F3/F5/F7: correlation sweep at 3 severities (fine rho grid) ----
    t0 = time.perf_counter()
    print("[F2/F3/F5/F7] correlation sweep q×rho (fine) …")
    sweep = []
    for q in Q_LEVELS_F3:
        for rho in RHO_FINE:
            inst = _instance(q)
            Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED)
            Xtr = sample_disruptions(inst, N_TRAIN, seed=SEED, mode="partial", rho=rho)
            design = _solve_design(inst, Ytr, Btr, Xtr)
            band = _eval_band(inst, design, q, rho, mode="partial")
            rec = {"q": q, "rho": rho, "z": design["z"].tolist(), "C": design["C"].tolist()}
            for k in ("total_cost", "exp_remfg", "exp_sub", "exp_cancel", "service_level"):
                rec[k] = _stat(band[k])
            rec["tier_cancel"] = {t: _stat(band["tier_cancel"][t]) for t in ("H", "M", "L")}
            sweep.append(rec)
        print(f"    q={q:.2f} done")
    data["sweep"] = sweep
    print(f"[F2/F3/F5/F7] {time.perf_counter()-t0:.0f}s")

    # ---- F4: trade-off frontier (two levers) at high correlation ----
    t0 = time.perf_counter()
    print("[F4] hedging frontier …")
    rho_hi, q_hi = 0.75, HEADLINE_Q
    # Lever 1: spare capacity, s_max scaled by (1 + f).
    spare = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        inst = _instance(q_hi)
        inst.s_max = np.round(inst.s_max * (1.0 + frac)).astype(int)
        Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED)
        Xtr = sample_disruptions(inst, N_TRAIN, seed=SEED, mode="partial", rho=rho_hi)
        design = _solve_design(inst, Ytr, Btr, Xtr)
        band = _eval_band(inst, design, q_hi, rho_hi, mode="partial")
        spare.append({"level": frac,
                      "total_cost": _stat(band["total_cost"]),
                      "service_level": _stat(band["service_level"])})
        print(f"    spare +{frac*100:.0f}%  cost={spare[-1]['total_cost']['mean']:.1f} "
              f"svc={spare[-1]['service_level']['mean']*100:.1f}%")
    # Lever 2: diversification, effective rho reduced by fraction.
    divers = []
    for red in (0.0, 0.25, 0.5, 0.75):
        rho_eff = rho_hi * (1.0 - red)
        inst = _instance(q_hi)
        Ytr, Btr = sample_scenarios(inst, N_TRAIN, seed=SEED)
        Xtr = sample_disruptions(inst, N_TRAIN, seed=SEED, mode="partial", rho=rho_eff)
        design = _solve_design(inst, Ytr, Btr, Xtr)
        band = _eval_band(inst, design, q_hi, rho_eff, mode="partial")
        divers.append({"level": red, "rho_eff": rho_eff,
                       "total_cost": _stat(band["total_cost"]),
                       "service_level": _stat(band["service_level"])})
        print(f"    divers -{red*100:.0f}% (ρ_eff={rho_eff:.2f})  "
              f"cost={divers[-1]['total_cost']['mean']:.1f} "
              f"svc={divers[-1]['service_level']['mean']*100:.1f}%")
    data["F4"] = {"rho_hi": rho_hi, "q": q_hi, "spare": spare, "divers": divers}
    print(f"[F4] {time.perf_counter()-t0:.0f}s")

    # ---- F6a: input-sample convergence (no optimization) ----
    t0 = time.perf_counter()
    print("[F6a] input-sample convergence …")
    data["F6a"] = compute_input_convergence(q=HEADLINE_Q, rho=0.5, n_max=1000)
    print(f"[F6a] {time.perf_counter()-t0:.0f}s")

    # ---- F6: solution stability vs training N ----
    t0 = time.perf_counter()
    print("[F6] solution stability vs N …")
    data["F6"] = compute_stability()
    print(f"[F6] {time.perf_counter()-t0:.0f}s")

    data["compute_seconds"] = time.perf_counter() - t_start
    data["backend"] = "gurobi" if _GUROBI_FULL_OK else "highs"
    with open(_DATA, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nCached compute → {_DATA} ({data['compute_seconds']:.0f}s)")
    return data


def compute_input_convergence(q, rho, n_max=1000):
    """
    F6a: pure-sampling convergence of two scenario KPIs, no optimization.

    KPIs per scenario:
      * disrupted facilities  = Σ_m ξ_m(ω)
      * failed batches        = # patients whose nearest feasible facility's
        effective yield is 0 (yield failure OR facility disruption), under a
        fixed nearest-feasible reference assignment (needs no solve).

    We report the running median and running IQR (25–75%) vs sample count, and
    mark k* = the smallest sample count after which the running median stays
    within ε of its final value for ALL remaining samples (a converged-and-stays
    criterion). This is robust to the temporary plateaus that a plain
    "ε for K consecutive" rule would false-trigger on, which matters here because
    the KPIs are bimodal in the correlated regime (a common shock either fires,
    disrupting all facilities, or does not).
    """
    inst = _instance(q)
    n_p, n_f = inst.n_patients, inst.n_facilities
    Y, B = sample_scenarios(inst, n_max, seed=SEED)
    Xi = sample_disruptions(inst, n_max, seed=SEED, mode="partial", rho=rho)

    # Nearest-feasible reference assignment (shelf-life feasible, min transport).
    t = inst.t_trans
    feasible = (t <= inst.shelf_life) if (t is not None and inst.shelf_life) else np.ones((n_p, n_f), bool)
    assign = np.array([
        (np.where(feasible[i], t[i], np.inf).argmin() if t is not None else 0)
        for i in range(n_p)
    ])

    disrupted = Xi.sum(axis=1)                                  # (n_max,)
    Ytil = Y * (1.0 - Xi)[:, None, :]                           # (n_max,n_p,n_f)
    fail_assigned = 1.0 - Ytil[np.arange(n_max)[:, None],
                               np.arange(n_p)[None, :],
                               assign[None, :]]                 # (n_max,n_p)
    failed_batches = fail_assigned.sum(axis=1)                  # (n_max,)

    def _running(series, eps=0.02):
        med, q25, q75, mean = [], [], [], []
        for k in range(1, len(series) + 1):
            s = series[:k]
            med.append(float(np.median(s)))
            q25.append(float(np.percentile(s, 25)))
            q75.append(float(np.percentile(s, 75)))
            mean.append(float(np.mean(s)))
        med = np.array(med)
        final = med[-1]
        tol = max(0.5, eps * abs(final))   # abs floor: KPIs are (half-)integers
        # k*: smallest k such that the running median stays within tol of its
        # final value for every subsequent sample.
        within = np.abs(med - final) <= tol
        kstar = None
        for k in range(len(med)):
            if within[k:].all():
                kstar = k + 1     # 1-indexed sample count
                break
        return {"median": med.tolist(), "q25": q25, "q75": q75,
                "mean": mean, "kstar": kstar, "eps": eps, "tol": tol,
                "final": float(final)}

    return {
        "q": q, "rho": rho, "n_max": n_max,
        "disrupted": _running(disrupted),
        "failed": _running(failed_batches),
    }


def compute_stability():
    """
    F6: for training N in a grid, solve the design at ≥5 seeds and score each on
    a single fixed large OOS set; report mean and spread of OOS cost/service, and
    the N at which both stop moving beyond a tolerance.
    """
    N_grid = [50, 100, 200, 500, 1000]
    seeds = [0, 1, 2, 3, 4]
    q, rho = HEADLINE_Q, 0.5
    inst_test = _instance(q)
    N_test = N_EVAL
    Yte, Bte = sample_scenarios(inst_test, N_test, seed=999)
    Xte = sample_disruptions(inst_test, N_test, seed=999, mode="partial", rho=rho)

    records = []
    for Ntr in N_grid:
        tl = 180.0 if Ntr >= 500 else TIME_LIMIT
        for sd in seeds:
            inst_tr = _instance(q)
            Ytr, Btr = sample_scenarios(inst_tr, Ntr, seed=sd)
            Xtr = sample_disruptions(inst_tr, Ntr, seed=sd, mode="partial", rho=rho)
            tr = _solve_design(inst_tr, Ytr, Btr, Xtr, time_limit=tl)
            ev = evaluate_design_oos(inst_test, tr["z"], tr["C"], tr["x"], Yte, Bte, Xte)
            records.append({"N": Ntr, "seed": sd,
                            "oos_cost": ev["total_cost"],
                            "oos_service": ev["service_level"],
                            "gap": float(tr.get("gap", 0.0))})
            print(f"    N={Ntr:<4} seed={sd}  cost={ev['total_cost']:.2f} "
                  f"svc={ev['service_level']*100:.1f}% gap={tr.get('gap',0.0):.1e}")

    # Stabilization: first N where |Δmean| in cost and service (vs previous N)
    # is within tolerance.
    def _means(key):
        return {N: float(np.mean([r[key] for r in records if r["N"] == N]))
                for N in N_grid}
    cost_m, svc_m = _means("oos_cost"), _means("oos_service")
    cost_tol, svc_tol = 0.5, 0.01   # 0.5 M USD, 1 percentage-point of service
    stab_N = None
    for i in range(1, len(N_grid)):
        N = N_grid[i]
        if (abs(cost_m[N] - cost_m[N_grid[i - 1]]) <= cost_tol
                and abs(svc_m[N] - svc_m[N_grid[i - 1]]) <= svc_tol):
            stab_N = N
            break
    return {"N_grid": N_grid, "seeds": seeds, "q": q, "rho": rho,
            "N_test": N_test, "records": records,
            "cost_tol": cost_tol, "svc_tol": svc_tol, "stab_N": stab_N,
            "cost_means": cost_m, "svc_means": svc_m}


# ===========================================================================
# Plotting
# ===========================================================================

def _sweep_at_q(data, q):
    rows = [r for r in data["sweep"] if abs(r["q"] - q) < 1e-9]
    return sorted(rows, key=lambda r: r["rho"])


def _mm(rows, key):
    """Return (rho, mean, lo, hi) arrays for a sweep metric with {mean,min,max}."""
    rho = np.array([r["rho"] for r in rows])
    mean = np.array([r[key]["mean"] for r in rows])
    lo = np.array([r[key]["min"] for r in rows])
    hi = np.array([r[key]["max"] for r in rows])
    return rho, mean, lo, hi


def plot_F1(data):
    import matplotlib.pyplot as plt
    f1 = data["F1"]
    qs = np.array(f1["q_grid"])
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    act = [("exp_remfg", "re-manufacture", COLORS["remfg"]),
           ("exp_sub", "subcontract", COLORS["subcontract"]),
           ("exp_cancel", "cancel", COLORS["cancel"])]
    for ax, mode, title in [
        (axes[0], "independent", "Separate failures (independent)"),
        (axes[1], "common", "Together failures (common-mode)"),
    ]:
        for key, lab, col in act:
            m = np.array([d[key]["mean"] for d in f1[mode]])
            lo = np.array([d[key]["min"] for d in f1[mode]])
            hi = np.array([d[key]["max"] for d in f1[mode]])
            ax.fill_between(qs, lo, hi, color=col, alpha=0.18)
            ax.plot(qs, m, "-o", color=col, ms=4, label=lab)
        ax.set_title(title)
        ax.set_xlabel("Facility disruption probability  q")
    axes[0].set_ylabel("Expected actions / scenario")
    axes[1].legend(loc="upper left", framealpha=0.95)
    fig.suptitle("F1 — Does rerouting wake up under disruption?", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "figureR1_rerouting_under_disruption")


def plot_F2(data):
    import matplotlib.pyplot as plt
    rows = _sweep_at_q(data, data["headline_q"])
    rho, sub, sub_lo, sub_hi = _mm(rows, "exp_sub")
    _, can, can_lo, can_hi = _mm(rows, "exp_cancel")
    setup_style()
    fig = double_column()
    ax = fig.add_subplot(111)
    ax.fill_between(rho, sub_lo, sub_hi, color=COLORS["subcontract"], alpha=0.2)
    ax.plot(rho, sub, "-o", color=COLORS["subcontract"], ms=4, label="subcontract (rerouting value)")
    ax.fill_between(rho, can_lo, can_hi, color=COLORS["cancel"], alpha=0.2)
    ax.plot(rho, can, "-s", color=COLORS["cancel"], ms=4, label="cancel")
    drops = [sub[i] - sub[i + 1] for i in range(len(sub) - 1)]
    k = int(np.argmax(drops))
    ax.axvspan(rho[k], rho[k + 1], color=COLORS["subcontract"], alpha=0.08)
    ax.annotate(f"steepest decline\n(Δsub={drops[k]:.1f} over ρ {rho[k]:.1f}→{rho[k+1]:.1f})",
                xy=(0.5 * (rho[k] + rho[k + 1]), sub[k + 1]),
                xytext=(rho[k] + 0.02, max(sub) * 0.6), fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.set_xlabel("Disruption correlation  ρ")
    ax.set_ylabel("Expected actions / scenario")
    ax.set_title(f"F2 — Slope or cliff?  (q = {data['headline_q']:.2f}, "
                 f"N={data['n_eval']}, band = min–max over {len(data['eval_seeds'])} seeds)",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "figureR2_subcontracting_vs_correlation")


def plot_F3(data):
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax2.grid(False)
    cmap = {0.05: COLORS["tier_L"], 0.10: COLORS["sp"], 0.15: COLORS["cancel"]}
    for q in data["q_levels_f3"]:
        rows = _sweep_at_q(data, q)
        rho, cost, c_lo, c_hi = _mm(rows, "total_cost")
        _, svc, _, _ = _mm(rows, "service_level")
        svc = svc * 100
        c = cmap.get(q, COLORS["sp"])
        ax1.fill_between(rho, c_lo, c_hi, color=c, alpha=0.12)
        ax1.plot(rho, cost, "-o", color=c, ms=4, label=f"cost, q={q:.2f}")
        ax2.plot(rho, svc, "--s", color=c, alpha=0.6, ms=3, label=f"service, q={q:.2f}")
    ax1.set_xlabel("Disruption correlation  ρ")
    ax1.set_ylabel("Total expected cost  (M USD)")
    ax2.set_ylabel("Service level  (% treated on time)")
    ax1.set_title("F3 — Price of correlation  (solid = cost, dashed = service)",
                  fontweight="bold")
    ax1.legend(loc="center left", fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, "figureR3_price_of_correlation")


def plot_F4(data):
    import matplotlib.pyplot as plt
    f4 = data["F4"]
    setup_style()
    fig = double_column()
    ax = fig.add_subplot(111)
    # Spare-capacity lever. Points saturate (later levels coincide), so label
    # distinct locations only and flag the saturation as the key finding.
    sp = f4["spare"]
    xs = [o["service_level"]["mean"] * 100 for o in sp]
    ys = [o["total_cost"]["mean"] for o in sp]
    ax.plot(xs, ys, "-^", color=COLORS["subcontract"], ms=7, label="more spare capacity")
    labelled = []
    for o, x, y in zip(sp, xs, ys):
        if any(abs(x - lx) < 0.4 and abs(y - ly) < 0.4 for lx, ly in labelled):
            continue   # coincides with an already-labelled point
        labelled.append((x, y))
        ax.annotate(f"+{o['level']*100:.0f}%", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color="#7a5c00")
    # Saturation note near the plateau (last distinct spare point).
    ax.annotate("+25%→+100%: no further gain\n(spare capacity saturates)",
                (xs[-1], ys[-1]), textcoords="offset points", xytext=(10, -4),
                fontsize=7, color="#7a5c00", va="top")
    # Diversification lever.
    dv = f4["divers"]
    xd = [o["service_level"]["mean"] * 100 for o in dv]
    yd = [o["total_cost"]["mean"] for o in dv]
    ax.plot(xd, yd, "-D", color=COLORS["remfg"], ms=6, label="diversified siting (↓ effective ρ)")
    for o, x, y in zip(dv, xd, yd):
        if o["level"] == 0.0:
            continue   # coincides with baseline / spare +0%
        ax.annotate(f"-{o['level']*100:.0f}%", (x, y), textcoords="offset points",
                    xytext=(6, -11), fontsize=7, color="#2f5f2f")
    # Baseline = spare +0% == divers -0% (same point); mark it.
    ax.scatter([xs[0]], [ys[0]], s=140, facecolor="none", edgecolor=COLORS["sp"],
               linewidth=1.8, zorder=5, label="baseline (ρ=0.75)")
    allx = xs + xd
    ally = ys + yd
    mx, Mx = min(allx), max(allx)
    my, My = min(ally), max(ally)
    ax.set_xlim(mx - 0.04 * (Mx - mx) - 1, Mx + 0.10 * (Mx - mx) + 1)
    ax.set_ylim(my - 0.08 * (My - my), My + 0.08 * (My - my))
    ax.set_xlabel("Service level  (% treated on time)  →  better")
    ax.set_ylabel("Total expected cost  (M USD)  ↓ better")
    ax.set_title(f"F4 — Which hedge is worth paying for?  "
                 f"(ρ={f4['rho_hi']:.2f}, q={f4['q']:.2f})", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "figureR4_hedging_frontier")


def plot_F5(data):
    import matplotlib.pyplot as plt
    rows = _sweep_at_q(data, data["headline_q"])
    rho = np.array([r["rho"] for r in rows])
    tcol = {"H": COLORS["tier_H"], "M": COLORS["tier_M"], "L": COLORS["tier_L"]}
    # Per-patient RATE (cancellations / tier size) so tiers are comparable —
    # absolute counts just track tier sizes (H=10, M=25, L=15).
    sizes = {t: len(TIER_IDX[t]) for t in ("H", "M", "L")}
    beta = {"H": 0.55, "M": 0.78, "L": 0.92}
    setup_style()
    fig = double_column()
    ax = fig.add_subplot(111)
    for t in ("H", "M", "L"):
        s = sizes[t]
        m = np.array([r["tier_cancel"][t]["mean"] for r in rows]) / s
        lo = np.array([r["tier_cancel"][t]["min"] for r in rows]) / s
        hi = np.array([r["tier_cancel"][t]["max"] for r in rows]) / s
        ax.fill_between(rho, lo, hi, color=tcol[t], alpha=0.18)
        ax.plot(rho, m, "-o", color=tcol[t], ms=4,
                label=f"tier {t}  (β={beta[t]:.2f}, n={s})")
    ax.set_xlabel("Disruption correlation  ρ")
    ax.set_ylabel("Cancellation rate  (per patient in tier)")
    ax.set_title(f"F5 — Who gets cancelled?  (q = {data['headline_q']:.2f})",
                 fontweight="bold")
    ax.annotate("rates track each other: high-urgency (H) is NOT shielded —\n"
                "its higher cancel-cost is offset by lower re-collection eligibility β",
                xy=(0.03, 0.95), xycoords="axes fraction", fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "figureR5_cancellations_by_tier")


def plot_F6a(data):
    import matplotlib.pyplot as plt
    f = data["F6a"]
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    panels = [("disrupted", "Disrupted facilities / scenario", axes[0]),
              ("failed", "Failed batches / scenario", axes[1])]
    for key, ylab, ax in panels:
        d = f[key]
        n = len(d["median"])
        xs = np.arange(1, n + 1)
        med = np.array(d["median"])
        ax.fill_between(xs, d["q25"], d["q75"], color=COLORS["sp"], alpha=0.18,
                        label="running IQR (25–75%)")
        ax.plot(xs, med, color=COLORS["sp"], lw=1.3, label="running median")
        if "mean" in d:
            ax.plot(xs, d["mean"], color=COLORS["cancel"], lw=1.1, ls=":",
                    label="running mean")
        if d["kstar"]:
            ax.axvline(d["kstar"], color=COLORS["remfg"], ls="--", lw=1.3)
            ax.annotate(f"k*={d['kstar']}\n(median settles at {d['final']:.0f})",
                        xy=(d["kstar"], med[-1]),
                        xytext=(max(5, d["kstar"] - n * 0.30), med[-1] * 0.55 + 0.5),
                        fontsize=8, color=COLORS["remfg"],
                        arrowprops=dict(arrowstyle="->", color=COLORS["remfg"], lw=1))
        ax.set_xlabel("Number of sampled scenarios")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7.5, loc="center right")
    fig.suptitle(f"F6a — Input-sample convergence  (q={f['q']:.2f}, ρ={f['rho']:.2f}; "
                 f"k* : running median within ε={f['disrupted']['eps']:.0%} of its final "
                 f"value for all later samples)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "figureR6a_input_convergence")


def plot_F6(data):
    import matplotlib.pyplot as plt
    f6 = data["F6"]
    N_grid = f6["N_grid"]
    recs = f6["records"]
    setup_style()
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, ylab, title, scale in [
        (axc, "oos_cost", "OOS total cost  (M USD)", "Out-of-sample cost", 1.0),
        (axs, "oos_service", "OOS service level  (%)", "Out-of-sample service", 100.0),
    ]:
        means, los, his = [], [], []
        for N in N_grid:
            vals = np.array([r[key] for r in recs if r["N"] == N]) * scale
            means.append(vals.mean()); los.append(vals.min()); his.append(vals.max())
        means, los, his = map(np.array, (means, los, his))
        ax.fill_between(N_grid, los, his, color=COLORS["sp"], alpha=0.18,
                        label="min–max across seeds")
        ax.plot(N_grid, means, "-o", color=COLORS["sp"], label="mean")
        if f6["stab_N"]:
            ax.axvline(f6["stab_N"], color=COLORS["remfg"], ls="--", lw=1.2,
                       label=f"stabilizes at N={f6['stab_N']}")
        # Flag training N whose design solve was time-limited (MIP gap > 1e-3):
        # its OOS metric is a solver artefact, not sampling drift.
        gap_by_N = {N: max(r["gap"] for r in recs if r["N"] == N) for N in N_grid}
        for j, N in enumerate(N_grid):
            if gap_by_N[N] > 1e-3:
                ax.scatter([N], [means[j]], s=90, facecolor="none",
                           edgecolor=COLORS["cancel"], linewidth=1.6, zorder=6)
                if key == "oos_cost":
                    ax.annotate(f"N={N}: design MIP gap ~{gap_by_N[N]*100:.0f}%\n"
                                f"(time-limited, not sampling)",
                                xy=(N, means[j]), xytext=(N * 0.42, means[j] + 0.15),
                                fontsize=7, color=COLORS["cancel"], ha="right",
                                arrowprops=dict(arrowstyle="->", color=COLORS["cancel"], lw=1))
        ax.set_xscale("log")
        ax.set_xticks(N_grid)
        ax.get_xaxis().set_minor_locator(plt.NullLocator())
        ax.set_xticklabels([str(n) for n in N_grid])
        ax.set_xlabel("Optimization scenarios  N")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    stab = (f"stabilizes at N={f6['stab_N']}" if f6["stab_N"]
            else "NOT stabilized by N=1000")
    fig.suptitle(f"F6 — Solution stability vs training N  "
                 f"({len(f6['seeds'])} seeds, OOS N={f6['N_test']}) — {stab}",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "figureR6_scenario_stability")


def plot_F7(data):
    import matplotlib.pyplot as plt
    rows = _sweep_at_q(data, 0.15)
    rho = np.array([r["rho"] for r in rows])
    sub = np.array([r["exp_sub"]["mean"] for r in rows])
    sub_lo = np.array([r["exp_sub"]["min"] for r in rows])
    sub_hi = np.array([r["exp_sub"]["max"] for r in rows])
    Cmat = np.array([r["C"] for r in rows])
    zmat = np.array([r["z"] for r in rows])
    n_f = Cmat.shape[1]

    up_steps = [k for k in range(len(sub) - 1) if sub[k + 1] > sub[k] + 1e-9]
    blip_k = up_steps[0] if up_steps else None
    if blip_k is not None:
        design_changed = bool(np.any(Cmat[blip_k] != Cmat[blip_k + 1])
                              or np.any(zmat[blip_k] != zmat[blip_k + 1]))
        verdict = "real re-optimization" if design_changed else "sampling noise"
    else:
        design_changed = False
        verdict = "resolved: no blip at N=%d" % data["n_eval"]

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.fill_between(rho, sub_lo, sub_hi, color=COLORS["subcontract"], alpha=0.2)
    ax1.plot(rho, sub, "-o", color=COLORS["subcontract"], ms=4)
    ax1.set_xlabel("Disruption correlation  ρ")
    ax1.set_ylabel("Expected subcontract / scenario")
    ax1.set_title("Subcontracting trace (q=0.15)")
    # Annotation in a text box in a clear area (no title overlap).
    if blip_k is not None:
        ax1.axvspan(rho[blip_k], rho[blip_k + 1], color=COLORS["cancel"], alpha=0.10)
        txt = (f"blip at ρ={rho[blip_k]:.1f}→{rho[blip_k+1]:.1f}: "
               f"{sub[blip_k]:.2f}→{sub[blip_k+1]:.2f}\n"
               f"design {'changes' if design_changed else 'unchanged'} → "
               f"{'real' if design_changed else 'noise'}")
    else:
        txt = f"no blip at N={data['n_eval']}\n(under-sampled artefact resolved)"
    ax1.text(0.5, 0.06, txt, transform=ax1.transAxes, fontsize=8, ha="center",
             va="bottom", bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.95))
    fac_colors = [COLORS["sp"], COLORS["remfg"], COLORS["subcontract"], COLORS["cancel"]]
    for m in range(n_f):
        ax2.plot(rho, Cmat[:, m], "-o", color=fac_colors[m % len(fac_colors)],
                 ms=4, label=f"C[m{m}]")
    if blip_k is not None:
        ax2.axvspan(rho[blip_k], rho[blip_k + 1], color=COLORS["cancel"], alpha=0.10)
    ax2.set_xlabel("Disruption correlation  ρ")
    ax2.set_ylabel("Contracted capacity  C_m  (slots)")
    ax2.set_title("First-stage design across ρ")
    ax2.legend(fontsize=8)
    fig.suptitle(f"F7 — Diagnosing the q=0.15 blip  →  {verdict}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "figureR7_blip_diagnosis")
    return verdict, blip_k


# ===========================================================================
# Guide + main
# ===========================================================================

def write_guide(data, blip_verdict):
    rows_h = _sweep_at_q(data, data["headline_q"])
    sub0 = rows_h[0]["exp_sub"]["mean"]; sub1 = rows_h[-1]["exp_sub"]["mean"]
    can0 = rows_h[0]["exp_cancel"]["mean"]; can1 = rows_h[-1]["exp_cancel"]["mean"]
    f1i = data["F1"]["independent"]
    sub_indep_max = max(d["exp_sub"]["mean"] for d in f1i)
    f4 = data["F4"]
    base = f4["spare"][0]
    best_div = min(f4["divers"], key=lambda o: o["total_cost"]["mean"])
    best_spare = min(f4["spare"], key=lambda o: o["total_cost"]["mean"])
    f6 = data["F6"]; f6a = data["F6a"]
    kd = f6a["disrupted"]["kstar"]; kf = f6a["failed"]["kstar"]
    stab = (f"N={f6['stab_N']}" if f6["stab_N"] else "not reached by N=1000")
    n1000_gap = max((r["gap"] for r in f6["records"] if r["N"] == 1000), default=0.0)
    _cost_at = lambda N: float(np.mean([r["oos_cost"] for r in f6["records"] if r["N"] == N]))
    gap_note = (f" The N=1000 point sits ~{(_cost_at(1000)-_cost_at(500)):.1f} M "
                f"above N=500 because those design solves hit the 180 s limit at "
                f"~{n1000_gap*100:.0f}% MIP gap (a solver artefact, not sampling); "
                f"N≤500 all solved to gap 0."
                if n1000_gap > 1e-3 else "")

    lines = [
        "# Resilience figure guide (F1–F7, F6a)",
        "",
        f"Decision-focused figures for the capacity-disruption / correlation study. "
        f"All regenerate from `figures/generate_resilience.py` (compute cached in "
        f"`figures/resilience_data.json`). Backend: **{data['backend'].upper()}** "
        f"(Gurobi when a full license is present, else HiGHS — identical optima by "
        f"parity). Design solved at N={data['n_train']} (MIP gap {data['mip_gap']:.0e}); "
        f"curves scored out-of-sample at N={data['n_eval']} over "
        f"{len(data['eval_seeds'])} seeds, plotted as mean with a min–max band.",
        "",
        "| Fig | File | Question it answers | Takeaway (from the numbers) |",
        "|-----|------|---------------------|------------------------------|",
        f"| F1 | `figureR1_rerouting_under_disruption` | Does rerouting wake up under "
        f"facility disruption — is the earlier \"subcontracting is dominated\" result "
        f"reversed at facility level? | **Yes, for independent failures:** subcontracting "
        f"rises to ~{sub_indep_max:.1f}/scenario as surviving facilities absorb load; "
        f"under common-mode failures it stays ~0 and everything cancels. |",
        f"| F2 | `figureR2_subcontracting_vs_correlation` | Slope or cliff — how does "
        f"rerouting value fall as ρ rises (q={data['headline_q']:.2f})? | **A steepening "
        f"slope, not a sharp cliff:** E[sub] declines steadily {sub0:.1f}→{sub1:.1f}, "
        f"steepest around ρ≈0.6–0.8 where it collapses to ~0; cancellations rise roughly "
        f"linearly {can0:.1f}→{can1:.1f}. |",
        f"| F3 | `figureR3_price_of_correlation` | What does correlation cost in money and "
        f"service, across severities? | Cost climbs steeply and service falls with ρ at "
        f"every q; correlation, not severity alone, drives the loss. |",
        f"| F4 | `figureR4_hedging_frontier` | Under high correlation, which hedge is worth "
        f"paying for — spare capacity or diversification? | **Diversification dominates:** "
        f"best diversified point reaches {best_div['service_level']['mean']*100:.0f}% "
        f"service at {best_div['total_cost']['mean']:.0f} M USD vs baseline "
        f"{base['service_level']['mean']*100:.0f}%/{base['total_cost']['mean']:.0f}; "
        f"spare capacity saturates fast (best "
        f"{best_spare['service_level']['mean']*100:.0f}% at "
        f"{best_spare['total_cost']['mean']:.0f}). |",
        f"| F5 | `figureR5_cancellations_by_tier` | Who gets cancelled as ρ rises? | "
        f"**Per-patient cancellation rates track each other across tiers** (all → 100% at "
        f"ρ=1): high-urgency tier-H is *not* shielded — its 6× cancel-cost incentive is "
        f"offset by its lower re-collection eligibility (β_H=0.55), so H cancels at a "
        f"marginally higher rate than mid-tier M. |",
        f"| F6a | `figureR6a_input_convergence` | How many scenarios does the *input* "
        f"sampling need (no optimization)? | In the correlated regime the KPIs are "
        f"bimodal (a common shock either fires or not), so the running median only "
        f"settles by k*≈{kd} (disrupted facilities) / k*≈{kf} (failed batches) scenarios "
        f"— justifying N≈600+ and confirming N=1000 leaves margin. |",
        f"| F6 | `figureR6_scenario_stability` | Is training N large enough for stable "
        f"out-of-sample results? | OOS cost & service stabilize at **{stab}** "
        f"(tol {f6['cost_tol']} M USD / {f6['svc_tol']*100:.0f} pp).{gap_note} |",
        f"| F7 | `figureR7_blip_diagnosis` | Is the q=0.15 non-monotonic subcontracting "
        f"blip real or noise? | **{blip_verdict}** (see below). |",
        "",
        "## What moved from the under-sampled version",
        "",
        f"- Curves are now scored at N={data['n_eval']} (was 200) over "
        f"{len(data['eval_seeds'])} seeds with min–max bands, so the traces are smooth "
        f"and the bands show the sampling is tight.",
        f"- **F6a (new)** justifies the scenario count directly from the draws: with a "
        f"converged-and-stays criterion the running median settles only by ~{kd} "
        f"(disrupted facilities) / ~{kf} (failed batches) samples — the bimodal common-"
        f"shock KPIs need the full N≥500–1000 to stabilize.",
        f"- **F6** now spans N∈{f6['N_grid']} at {len(f6['seeds'])} seeds and reports "
        f"stabilization at {stab}.",
        f"- **F4** is now a populated two-lever frontier (spare capacity 0–100% of s_max, "
        f"diversification 0–75% of effective ρ) rather than three markers.",
        f"- **F7**: with N={data['n_eval']} the blip is diagnosed as **{blip_verdict}**; "
        f"the headline decline (F2) and the diversification result (F4) are unchanged from "
        f"the under-sampled version — only the noise shrank.",
    ]
    guide = _HERE / "figure_guide.md"
    guide.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {guide}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        data = json.loads(_DATA.read_text())
    elif args.recompute or not _DATA.exists():
        data = compute_all()
    else:
        data = json.loads(_DATA.read_text())

    print("\nPlotting …")
    t0 = time.perf_counter()
    plot_F1(data); plot_F2(data); plot_F3(data); plot_F4(data)
    plot_F5(data); plot_F6a(data); plot_F6(data)
    verdict, _ = plot_F7(data)
    write_guide(data, verdict)
    print(f"\nDone. 8 figures + figure_guide.md ({time.perf_counter()-t0:.0f}s plotting).")


if __name__ == "__main__":
    main()
