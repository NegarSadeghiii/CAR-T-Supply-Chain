"""
value_of_endogeneity.py — Value of Endogeneity (VoE) experiment.

Question: what is the cost (and tier-H mortality) of IGNORING delay-driven
clinical deterioration when designing the network?

  D_exo  = optimal design from the exogenous model (dynamic_sp at kappa=0 == v1).
  D_endo = optimal design from the endogenous model (dynamic_sp at kappa>0).

Both are evaluated under a COMMON endogenous Monte-Carlo simulator that captures
the true dynamics: primary yields -> facility congestion (from the design's
actual capacity and routing) -> re-manufacture delay (clearing function) ->
delay-driven decline (PH kernel) -> forced cancellation of deteriorated patients.
A cancelled patient is untreated (mortality proxy); we decompose by urgency tier.

  VoE(kappa, gamma) = Cost(D_exo | true dynamics) - Cost(D_endo | true dynamics)

We sweep the delay-sensitivity knob kappa upward and the (un-fitted) Weibull
shape gamma over {1.0, 1.5, 2.0}, and report a table, figures (VoE vs kappa;
tier-H mortality vs kappa), and JSON.

Seeds: in-sample design solve seed 0; out-of-sample simulator seed 100
(calibration_table.md). Backend HiGHS.

Run: python value_of_endogeneity.py            (full 50-patient, ~minutes)
     python value_of_endogeneity.py --quick    (smaller, for smoke testing)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import highspy as hs

from yield_sp_v1 import Instance, sample_scenarios
from case_study import build_case_study_instance, TIERS, TIER_IDX
from declineprob import calibrate_lambda, extra_survival, HR_TIER
from clearing_function import ClearingFunction, DEFAULT_CLEARING
from dynamic_sp import solve_dynamic_sp, solve_dynamic_sp_endogenous

_HERE = Path(__file__).resolve().parent
_FIG = _HERE / "figures"
_RES = _HERE / "results"
_RES.mkdir(exist_ok=True)

# Experiment settings (calibration_table.md).
N_TRAIN = 200
SEED_TRAIN = 0
N_OOS = 2000
SEED_OOS = 100
KAPPA_SWEEP = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
GAMMA_SWEEP = [1.0, 1.5, 2.0]
MIP_GAP = 1e-3
TIME_LIMIT = 150.0


# ---------------------------------------------------------------------------
# Common endogenous simulator (true dynamics), shared by both designs
# ---------------------------------------------------------------------------

def _tier_hr(tiers):
    return np.array([HR_TIER[t] for t in tiers])


def simulate_endogenous(inst, design, tiers, Y, B, U, *,
                        kappa, gamma, clearing=DEFAULT_CLEARING):
    """
    Evaluate a FIXED design under the true endogenous dynamics on shared OOS
    draws (Y yields, B baseline eligibility, U uniforms for the decline event).

    Congestion is DECISION-DEPENDENT: rho_m = (primary failures routed to m) /
    C_m, using the design's own capacity and assignment. Delay -> decline ->
    StillEligible; the recourse LP is then solved with that eligibility.

    Returns dict: total_cost, expected_stage2_cost, exp_cancel, tier_cancel{H,M,L},
    mean_wait_days, mean_rho.
    """
    z, C, x = np.asarray(design["z"], float), np.asarray(design["C"], float), np.asarray(design["x"], float)
    N, n_p, n_f = Y.shape
    lam = calibrate_lambda(gamma)
    hr = _tier_hr(tiers)
    m_assign = x.argmax(axis=1)                              # (n_p,) chosen facility

    # Primary failure at assigned facility.
    Yass = Y[:, np.arange(n_p), m_assign]                    # (N, n_p)
    F = 1.0 - Yass

    # Decision-dependent congestion -> delay -> StillEligible. rho_m combines
    # the design's primary utilization (a_m / C_m) with the per-scenario failure
    # surge (batches needing a re-make at m), exactly as the optimizer's
    # still_eligible_for_design does — so the design is scored on the congestion
    # it actually creates.
    a_m = np.zeros(n_f)
    for i in range(n_p):
        a_m[m_assign[i]] += 1.0
    still = np.zeros((N, n_p))
    wait = np.zeros((N, n_p))
    rho_acc = np.zeros((N, n_f))
    Csafe = np.maximum(C, 1e-9)
    for o in range(N):
        surge = np.zeros(n_f)
        for i in range(n_p):
            if F[o, i] > 0.5:
                surge[m_assign[i]] += 1.0
        rho = (a_m + surge) / Csafe
        rho_acc[o] = rho
        delay_m = clearing.remake_delay(rho)                 # (n_f,)
        w = F[o] * delay_m[m_assign]                         # (n_p,)
        wait[o] = w
        p_extra = np.array([extra_survival(w[i], hr[i], lam, gamma, kappa) for i in range(n_p)])
        still[o] = B[o] * (U[o] < p_extra).astype(float)

    r_cancel, s2 = _solve_recourse(inst, z, C, x, Y, still)

    stage1 = (sum(inst.f[m] * z[m] for m in range(n_f))
              + sum(inst.pi[m] * C[m] for m in range(n_f))
              + sum(inst.c[m] * x[i, m] for i in range(n_p) for m in range(n_f)))
    tier_cancel = {t: float(r_cancel[:, TIER_IDX[t]].sum() / N) for t in ("H", "M", "L")}
    return {
        "total_cost": float(stage1 + s2),
        "expected_stage2_cost": float(s2),
        "exp_cancel": float(r_cancel.sum() / N),
        "tier_cancel": tier_cancel,
        "mean_wait_days": float(wait[F > 0.5].mean()) if np.any(F > 0.5) else 0.0,
        "mean_rho": float(rho_acc.mean()),
    }


def _solve_recourse(inst, z, C, x, Y, still):
    """
    Optimal fixed-design recourse across all scenarios given eligibility `still`
    (StillEligible). Continuous [0,1] recourse over a TU network -> integral.
    Returns (r_cancel[N,n_p], expected_stage2_cost).
    """
    N, n_p, n_f = Y.shape
    n_sub_pp = n_f * (n_f - 1)
    n_per_s = n_p * n_f + n_p * n_sub_pp + n_p
    n_total = N * n_per_s

    def _soff(m, mp): return m * (n_f - 1) + (mp if mp < m else mp - 1)
    def vr(o, i, m): return o * n_per_s + i * n_f + m
    def vs(o, i, m, mp): return o * n_per_s + n_p * n_f + i * n_sub_pp + _soff(m, mp)
    def vc(o, i): return o * n_per_s + n_p * n_f + n_p * n_sub_pp + i

    h = hs.Highs(); h.silent()
    lbs = np.zeros(n_total); ubs = np.ones(n_total); costs = np.zeros(n_total)
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
        h.addRow(lb, ub, len(inds), np.asarray(inds, np.int32), np.asarray(vals, np.float64))

    for o in range(N):
        for i in range(n_p):
            remfg_i = [vr(o, i, m) for m in range(n_f)]
            sub_i = [vs(o, i, m, mp) for m in range(n_f) for mp in range(n_f) if mp != m]
            F_i = float(sum((1.0 - Y[o, i, m]) * x[i, m] for m in range(n_f)))
            _row(F_i, F_i, remfg_i + sub_i + [vc(o, i)], [1.0] * (n_f + n_sub_pp + 1))
            _row(-INF, float(still[o, i]), remfg_i + sub_i, [1.0] * (n_f + n_sub_pp))
            for m in range(n_f):
                sub_from_m = [vs(o, i, m, mp) for mp in range(n_f) if mp != m]
                _row(-INF, float(x[i, m]), [vr(o, i, m)] + sub_from_m, [1.0] * n_f)
        for m in range(n_f):
            remfg_m = [vr(o, i, m) for i in range(n_p)]
            sub_to_m = [vs(o, i, mp, m) for i in range(n_p) for mp in range(n_f) if mp != m]
            fixed_load = float(sum(x[i, m] * Y[o, i, m] for i in range(n_p)))
            rhs = max(0.0, float(C[m]) - fixed_load)
            _row(-INF, rhs, remfg_m + sub_to_m, [1.0] * n_p + [1.0] * (n_p * (n_f - 1)))

    h.run()
    col = h.getSolution().col_value
    r_cancel = np.zeros((N, n_p))
    s2 = 0.0
    for o in range(N):
        for i in range(n_p):
            r_cancel[o, i] = round(col[vc(o, i)])
            for m in range(n_f):
                s2 += (inst.rho_leuk + inst.rho_remfg[m]) * round(col[vr(o, i, m)])
                for mp in range(n_f):
                    if mp != m:
                        s2 += (inst.rho_leuk + inst.rho_sub[m, mp]) * round(col[vs(o, i, m, mp)])
            s2 += inst.rho_cancel[i] * r_cancel[o, i]
    return r_cancel, s2 / N


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(quick=False):
    inst = build_case_study_instance()
    tiers = list(TIERS)
    n_train = 60 if quick else N_TRAIN
    n_oos = 300 if quick else N_OOS
    kappas = [0.0, 1.0, 2.0] if quick else KAPPA_SWEEP
    gammas = [1.0, 2.0] if quick else GAMMA_SWEEP
    tl = 60.0 if quick else TIME_LIMIT

    # Shared in-sample training scenarios (design solve) and OOS draws (evaluation).
    Ytr, Btr = sample_scenarios(inst, n_train, seed=SEED_TRAIN)
    Yte, Bte = sample_scenarios(inst, n_oos, seed=SEED_OOS)
    rng = np.random.default_rng(SEED_OOS + 777)
    U = rng.random((n_oos, inst.n_patients))     # shared decline-event uniforms

    # D_exo: exogenous design (kappa=0 == v1), computed once.
    t0 = time.perf_counter()
    D_exo = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0,
                             seed=SEED_TRAIN, mip_gap=MIP_GAP, time_limit=tl)
    print(f"[D_exo] z={D_exo['z'].tolist()} C={D_exo['C'].tolist()} "
          f"({time.perf_counter()-t0:.0f}s)")

    rows = []
    designs = {}
    for gamma in gammas:
        for kappa in kappas:
            key = (round(kappa, 3), gamma)
            # D_endo: endogenous design at (kappa, gamma).
            if kappa == 0.0:
                D_endo = D_exo                          # identical by construction
            else:
                # Fixed-point endogenous design: congestion consistent with the
                # design's own capacity/routing (matches the true simulator).
                D_endo = solve_dynamic_sp_endogenous(
                    inst, Ytr, Btr, tiers, kappa=kappa, gamma=gamma,
                    seed=SEED_TRAIN, mip_gap=MIP_GAP, time_limit=tl)
            designs[key] = {"z": D_endo["z"].tolist(), "C": D_endo["C"].tolist()}

            # Evaluate BOTH designs under the common true dynamics at (kappa, gamma).
            ev_exo = simulate_endogenous(inst, D_exo, tiers, Yte, Bte, U, kappa=kappa, gamma=gamma)
            ev_endo = simulate_endogenous(inst, D_endo, tiers, Yte, Bte, U, kappa=kappa, gamma=gamma)
            voe = ev_exo["total_cost"] - ev_endo["total_cost"]
            dH = ev_exo["tier_cancel"]["H"] - ev_endo["tier_cancel"]["H"]
            rows.append({
                "kappa": kappa, "gamma": gamma,
                "cost_exo": ev_exo["total_cost"], "cost_endo": ev_endo["total_cost"],
                "voe": voe,
                "tierH_mort_exo": ev_exo["tier_cancel"]["H"],
                "tierH_mort_endo": ev_endo["tier_cancel"]["H"],
                "tierH_mort_reduction": dH,
                "cancel_exo": ev_exo["exp_cancel"], "cancel_endo": ev_endo["exp_cancel"],
                "tier_cancel_exo": ev_exo["tier_cancel"], "tier_cancel_endo": ev_endo["tier_cancel"],
                "mean_wait_endo": ev_endo["mean_wait_days"], "mean_rho_endo": ev_endo["mean_rho"],
                "C_exo": D_exo["C"].tolist(), "C_endo": D_endo["C"].tolist(),
                "z_endo": D_endo["z"].tolist(),
            })
            print(f"  gamma={gamma} kappa={kappa:<4}: VoE={voe:+.3f}  "
                  f"tierH_mort {ev_exo['tier_cancel']['H']:.2f}->{ev_endo['tier_cancel']['H']:.2f} "
                  f"(Δ{dH:+.2f})  C_endo={D_endo['C'].tolist()}")

    payload = {
        "n_train": n_train, "n_oos": n_oos, "seed_train": SEED_TRAIN, "seed_oos": SEED_OOS,
        "kappa_sweep": kappas, "gamma_sweep": gammas,
        "clearing": {"tau_proc": DEFAULT_CLEARING.tau_proc,
                     "breakpoints": list(DEFAULT_CLEARING.breakpoints),
                     "inc_slopes": list(DEFAULT_CLEARING.inc_slopes)},
        "lambda_by_gamma": {str(g): calibrate_lambda(g) for g in gammas},
        "hr_tier": HR_TIER,
        "D_exo": {"z": D_exo["z"].tolist(), "C": D_exo["C"].tolist()},
        "results": rows,
    }
    out = _RES / ("value_of_endogeneity_quick.json" if quick else "value_of_endogeneity_results.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out}")
    return payload


def _print_table(payload):
    print("\n=== Value of Endogeneity ===")
    hdr = f"{'gamma':>6}{'kappa':>7}{'cost_exo':>10}{'cost_endo':>11}{'VoE':>9}{'H_mort_exo':>12}{'H_mort_endo':>13}"
    print(hdr); print("-" * len(hdr))
    for r in payload["results"]:
        print(f"{r['gamma']:>6.1f}{r['kappa']:>7.2f}{r['cost_exo']:>10.3f}{r['cost_endo']:>11.3f}"
              f"{r['voe']:>9.3f}{r['tierH_mort_exo']:>12.3f}{r['tierH_mort_endo']:>13.3f}")


def make_figures(payload):
    sys.path.insert(0, str(_FIG))
    from figure_style import COLORS, setup_style, double_column, save_figure
    import matplotlib.pyplot as plt

    gammas = payload["gamma_sweep"]
    gcolors = {1.0: COLORS["sp"], 1.5: COLORS["subcontract"], 2.0: COLORS["cancel"]}

    # Figure 1: VoE vs kappa (one line per gamma).
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    for g in gammas:
        rs = sorted([r for r in payload["results"] if r["gamma"] == g], key=lambda r: r["kappa"])
        ax.plot([r["kappa"] for r in rs], [r["voe"] for r in rs], "-o",
                color=gcolors.get(g, COLORS["sp"]), ms=4, label=f"gamma={g}")
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("Delay-sensitivity  kappa")
    ax.set_ylabel("Value of Endogeneity  (M USD)")
    ax.set_title("Value of Endogeneity vs delay-sensitivity", fontweight="bold")
    ax.legend()
    fig.tight_layout(); save_figure(fig, "figureD1_voe_vs_kappa")

    # Figure 2: TRUE tier-H mortality vs kappa (endo design under true dynamics),
    # against the flat mortality the exogenous model assumes (its kappa=0 level).
    # The endogenous and exogenous designs incur essentially the same tier-H
    # mortality here (the high-yield facility serving tier-H is pinned at s_max),
    # so the message is the large gap between TRUE mortality and what an
    # exogenous (no-delay) model would predict.
    setup_style(); fig = double_column(); ax = fig.add_subplot(111)
    base = next(r["tierH_mort_endo"] for r in payload["results"]
                if r["kappa"] == 0.0 and r["gamma"] == gammas[0])
    for g in gammas:
        rs = sorted([r for r in payload["results"] if r["gamma"] == g], key=lambda r: r["kappa"])
        k = [r["kappa"] for r in rs]
        ax.plot(k, [r["tierH_mort_endo"] for r in rs], "-o", color=gcolors.get(g, COLORS["sp"]),
                ms=4, label=f"true tier-H mortality, gamma={g}")
    ax.axhline(base, color="#555", ls="--", lw=1.2,
               label="exogenous model's assumption (no delay)")
    ax.annotate("mortality the exogenous\nmodel is blind to",
                xy=(1.5, 0.47), xytext=(0.9, 0.30), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlabel("Delay-sensitivity  kappa")
    ax.set_ylabel("Tier-H cancellations / scenario  (mortality proxy)")
    ax.set_title("True tier-H mortality vs the exogenous assumption", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); save_figure(fig, "figureD2_tierH_mortality_vs_kappa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    payload = run_experiment(quick=args.quick)
    _print_table(payload)
    if not args.no_figures:
        make_figures(payload)
    print("\nDone.")


if __name__ == "__main__":
    main()
