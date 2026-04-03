"""
CAR-T Supply Chain — CCP Monte Carlo Simulation Validator
==========================================================
Validates the CCP chance-constraint guarantee empirically by simulating
T_MF ~ LogNormal across N_SIM runs and measuring realized service levels.

Also provides a Gamma-robust plan generator (Bertsimas-Sim box uncertainty)
for comparison against CCP and the deterministic baseline.

Monte Carlo validation
----------------------
  Given a solved CCP schedule (patient assignments), draw
      T_MF^(s) ~ LogNormal(mu_L, sigma_L^2)   for each patient in each run s
  and compute realized TRT:
      TRT_realized = (trt_effective - TMFE_eff) + T_MF^(s)
                   = (TLS + TT1 + TQC + TT3)   + T_MF^(s)

  Empirical patient service level = fraction of (patient, run) pairs with
      TRT_realized <= deadline

  The CCP guarantee is P(TRT_p <= D_p) >= 1-alpha per patient, so the
  empirical per-patient service level should approach 1-alpha as N_SIM -> inf.

Gamma-robust (Bertsimas-Sim)
-----------------------------
  Uncertainty set: T_MF in [TMFE - delta, TMFE + delta], delta = sigma_L * TMFE
  Budget:          at most Gamma deviations can hit their worst case simultaneously
  For a single-patient constraint the worst case is T_MF = TMFE + delta  (Gamma >= 1)
  => TMFE_eff_robust = TMFE + Gamma * delta = TMFE * (1 + Gamma * sigma_L)

  For Gamma < 1 this interpolates between nominal and worst-case.
"""

import math
import os
import random
import statistics
import time as _time

from cart_colgen import parse_dat, solve_master, DEFAULT_URGENCY
from cart_colgen_ccp import (
    DEFAULT_PROCESS_CCP,
    DEFAULT_PROCESS_DET,
    compute_tmfe_quantile,
    generate_plans_ccp,
    run_experiment_ccp,
)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_schedule(patients, tmfe_eff, tmfe_nominal, sigma_L,
                      n_sim=1000, seed=42):
    """Simulate realized TRT for a fixed CCP schedule under LogNormal T_MF.

    Parameters
    ----------
    patients     : list of patient dicts from solve_master result
                   each has keys: id, trt (effective), deadline
    tmfe_eff     : float — effective TMFE used in the schedule (quantile value)
    tmfe_nominal : float — nominal mean of T_MF distribution (= E[T_MF])
    sigma_L      : float — log-scale std-dev of T_MF distribution
    n_sim        : int   — number of Monte Carlo runs
    seed         : int   — RNG seed

    Returns
    -------
    dict with empirical service levels, violation distributions, per-patient stats
    """
    if sigma_L <= 0.0:
        # Deterministic: T_MF is always tmfe_nominal, no variability
        return {
            'n_sim': n_sim,
            'n_patients': len(patients),
            'empirical_patient_svc': 1.0,
            'empirical_all_on_time': 1.0,
            'avg_violations_per_run': 0.0,
            'max_violations_per_run': 0,
            'violations_histogram': {0: n_sim},
            'patient_stats': {p['id']: {'violation_rate': 0.0,
                                        'avg_lateness': 0.0,
                                        'p95_lateness': 0.0} for p in patients},
            'sigma_L': sigma_L,
        }

    mu_L = math.log(tmfe_nominal) - 0.5 * sigma_L ** 2
    rng  = random.Random(seed)

    n_patients = len(patients)
    violations_per_run = []
    # per-patient tracking
    pat_violations = {p['id']: 0    for p in patients}
    pat_lateness   = {p['id']: []   for p in patients}

    # Non-manufacturing TRT components are fixed by the schedule
    # non_mfg = TLS + TT1[j_out] + TQC + TT3[j_ret]  = trt_effective - TMFE_eff
    non_mfg = {p['id']: p['trt'] - tmfe_eff for p in patients}

    for _ in range(n_sim):
        run_violations = 0
        for p in patients:
            pid  = p['id']
            dl   = p['deadline']
            T_MF_sim    = rng.lognormvariate(mu_L, sigma_L)
            realized_trt = non_mfg[pid] + T_MF_sim
            late = realized_trt - dl
            if late > 0:
                run_violations    += 1
                pat_violations[pid] += 1
                pat_lateness[pid].append(late)
        violations_per_run.append(run_violations)

    # Aggregate
    total_patient_runs = n_sim * n_patients
    total_violations   = sum(violations_per_run)

    empirical_patient_svc  = 1.0 - total_violations / total_patient_runs
    empirical_all_on_time  = sum(1 for v in violations_per_run if v == 0) / n_sim

    hist = {}
    for v in violations_per_run:
        hist[v] = hist.get(v, 0) + 1

    patient_stats = {}
    for p in patients:
        pid = p['id']
        vr  = pat_violations[pid] / n_sim
        lats = pat_lateness[pid]
        avg_l = sum(lats) / len(lats) if lats else 0.0
        p95_l = sorted(lats)[int(0.95 * len(lats))] if len(lats) >= 20 else (max(lats) if lats else 0.0)
        patient_stats[pid] = {
            'violation_rate': round(vr, 4),
            'avg_lateness':   round(avg_l, 3),
            'p95_lateness':   round(p95_l, 3),
            'group':          p.get('group', '?'),
            'deadline':       p['deadline'],
        }

    return {
        'n_sim':                   n_sim,
        'n_patients':              n_patients,
        'empirical_patient_svc':   round(empirical_patient_svc, 4),
        'empirical_all_on_time':   round(empirical_all_on_time, 4),
        'avg_violations_per_run':  round(total_violations / n_sim, 3),
        'max_violations_per_run':  max(violations_per_run),
        'violations_histogram':    hist,
        'patient_stats':           patient_stats,
        'sigma_L':                 sigma_L,
        'tmfe_eff':                tmfe_eff,
        'tmfe_nominal':            tmfe_nominal,
    }


# ---------------------------------------------------------------------------
# Gamma-robust plan generator
# ---------------------------------------------------------------------------

def compute_tmfe_robust(process_config, gamma):
    """Bertsimas-Sim Gamma-robust effective T_MF.

    Uncertainty set: T_MF in [TMFE - delta, TMFE + delta]
                     delta = sigma_L * TMFE   (proportional box)
    Robust TMFE_eff = TMFE + Gamma * delta = TMFE * (1 + Gamma * sigma_L)

    Gamma=0   -> nominal (deterministic)
    Gamma=0.5 -> half-way to worst case
    Gamma=1   -> full worst case per patient (most conservative)
    """
    tmfe_nom = float(process_config.get('tmfe', 7))
    sigma_L  = float(process_config.get('sigma_L', 0.20))
    delta    = sigma_L * tmfe_nom
    return tmfe_nom + gamma * delta


def generate_plans_robust(data, gamma, tau=0.0,
                          urgency_config=None, process_config=None,
                          random_seed=42):
    """Generate plans using Gamma-robust TMFE_eff (Bertsimas-Sim).

    Structurally identical to generate_plans_ccp; only the TMFE_eff
    formula changes (box uncertainty instead of quantile).
    """
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = dict(DEFAULT_PROCESS_CCP)

    TMFE_eff     = compute_tmfe_robust(process_config, gamma)
    TMFE_eff_int = math.ceil(TMFE_eff)

    # Delegate to generate_plans_ccp by temporarily patching the config:
    # we set alpha such that the CCP quantile equals the robust TMFE_eff.
    # Equivalently, patch the process_config to use a very small sigma_L
    # and the corresponding alpha — but that's fragile.
    # Instead, we duplicate the core logic here for clarity.

    import random as _rand
    _rand.seed(random_seed)

    P  = data['p']
    C  = data['c']
    H  = data['h']
    J  = data['j']
    M  = data['m']

    TLS  = int(process_config.get('tls', 1))
    TMFE = int(process_config.get('tmfe', 7))
    TQC  = int(process_config.get('tqc', 7))
    MAX_FAC = int(process_config.get('max_facilities', 2))
    sigma_L = float(process_config.get('sigma_L', 0.20))

    TT1  = {j: int(data['TT1'][j]) for j in J}
    TT3  = {j: int(data['TT3'][j]) for j in J}
    FCAP = {m: int(data['FCAP'][m]) for m in M}
    CIM  = {m: data['CIM'][m] for m in M}
    CVM  = {m: data.get('CVM', {}).get(m, 0) for m in M}

    patient_arrival = {}
    patient_center  = {}
    c_to_h = {C[i]: H[i] for i in range(len(C))}
    for (p, c, t), v in data['INC'].items():
        if v == 1:
            patient_arrival[p] = t
            patient_center[p]  = c

    n        = len(P)
    num_high = int(round(n * urgency_config['high']['fraction']))
    num_med  = int(round(n * urgency_config['medium']['fraction']))
    shuffled = sorted(P, key=str)
    _rand.shuffle(shuffled)

    group_map, deadlines = {}, {}
    for idx, p in enumerate(shuffled):
        g   = 'high' if idx < num_high else ('medium' if idx < num_high + num_med else 'low')
        cfg = urgency_config[g]
        group_map[p]  = g
        deadlines[p]  = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)

    T_max          = 130
    fac_cost_total = {m: (CIM[m] + CVM[m]) * T_max for m in M}

    plans            = []
    plans_by_patient = {p: [] for p in P}
    plan_id          = 0

    for p in P:
        arr = patient_arrival[p]
        c   = patient_center[p]
        h   = c_to_h[c]
        dl  = deadlines[p]

        for mi in M:
            for j_out in J:
                for j_ret in J:
                    base_trt_robust = TLS + TT1[j_out] + TMFE_eff + TQC + TT3[j_ret]
                    if base_trt_robust > dl:
                        continue

                    base_trt_nom = TLS + TT1[j_out] + TMFE + TQC + TT3[j_ret]
                    cost_out = data['U1'].get((c, mi, j_out), 0)
                    cost_ret = data['U3'].get((mi, h, j_ret), 0)

                    mfg_start = arr + TLS + TT1[j_out]
                    mfg_end   = mfg_start + TMFE_eff_int

                    plan = {
                        'id':             plan_id,
                        'patient':        p,
                        'facility':       mi,
                        'j_out':          j_out,
                        'j_ret':          j_ret,
                        'wait':           0,
                        'trt':            base_trt_robust,
                        'trt_nominal':    base_trt_nom,
                        'mfg_start':      mfg_start,
                        'mfg_end':        mfg_end,
                        'transport_cost': cost_out + cost_ret,
                        'cost':           cost_out + cost_ret,
                        'tmfe_eff':       round(TMFE_eff, 4),
                        'gamma':          gamma,
                        'sigma_L':        sigma_L,
                    }
                    plans.append(plan)
                    plans_by_patient[p].append(plan_id)
                    plan_id += 1

    return {
        'plans':            plans,
        'plans_by_patient': plans_by_patient,
        'P': P, 'M': M, 'J': J,
        'FCAP':             FCAP,
        'CIM':              CIM,
        'CVM':              CVM,
        'fac_cost_total':   fac_cost_total,
        'T_max':            T_max,
        'TMFE':             TMFE_eff_int,
        'MAX_FAC':          MAX_FAC,
        'deadlines':        deadlines,
        'group_map':        group_map,
        'patient_arrival':  patient_arrival,
        'patient_center':   patient_center,
        'tmfe_nominal':     TMFE,
        'tmfe_eff':         round(TMFE_eff, 4),
        'gamma':            gamma,
        'sigma_L':          sigma_L,
    }


def run_experiment_robust(gamma, data_file=None, tau=0.0,
                          urgency_config=None, process_config=None,
                          time_limit=300, random_seed=42,
                          solver_name='appsi_highs'):
    """Run Gamma-robust experiment and return result dict."""
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data_N15.dat')
    if process_config is None:
        process_config = dict(DEFAULT_PROCESS_CCP)

    t0   = _time.time()
    data = parse_dat(data_file)
    plan_data = generate_plans_robust(
        data, gamma=gamma, tau=tau,
        urgency_config=urgency_config,
        process_config=process_config,
        random_seed=random_seed,
    )
    gen_time = _time.time() - t0

    n_plans    = len(plan_data['plans'])
    n_patients = len(plan_data['P'])
    tmfe_eff   = plan_data['tmfe_eff']
    sigma_L    = plan_data['sigma_L']

    infeasible = [p for p in plan_data['P']
                  if not plan_data['plans_by_patient'][p]]
    if infeasible:
        return {
            'method':              f'Γ-robust(Γ={gamma})',
            'gamma':               gamma,
            'solved':              False,
            'reason':              'infeasible_patients',
            'infeasible_patients': infeasible,
            'num_patients':        n_patients,
            'num_plans':           0,
            'tmfe_eff':            tmfe_eff,
            'sigma_L':             sigma_L,
            'gen_time':            round(gen_time, 3),
        }

    t0 = _time.time()
    _, res = solve_master(plan_data, time_limit=time_limit, solver_name=solver_name)
    solve_time = _time.time() - t0

    res['method']      = f'Γ-robust(Γ={gamma})'
    res['gamma']       = gamma
    res['num_patients'] = n_patients
    res['num_plans']   = n_plans
    res['tmfe_eff']    = tmfe_eff
    res['tmfe_nominal'] = plan_data['tmfe_nominal']
    res['sigma_L']     = sigma_L
    res['gen_time']    = round(gen_time, 3)
    res['solve_time']  = round(solve_time, 3)
    return res


# ---------------------------------------------------------------------------
# Full comparison runner
# ---------------------------------------------------------------------------

def run_full_comparison(data_file=None, sigma_L=0.20, tau=0.0,
                        alphas=None, gammas=None,
                        n_sim=1000, sim_seed=42,
                        time_limit=120, solve_seed=42,
                        solver_name='appsi_highs'):
    """Run all methods and Monte Carlo validation, return list of result dicts.

    Methods
    -------
    1. Deterministic   (sigma_L=0)
    2. CCP(alpha)      for each alpha in alphas
    3. Gamma-robust    for each gamma in gammas

    Each result dict contains the solve result PLUS a 'simulation' sub-dict
    from simulate_schedule().
    """
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data_N15.dat')
    if alphas is None:
        alphas = [0.20, 0.10, 0.05]
    if gammas is None:
        gammas = [0.5, 1.0, 1.5]

    results = []

    # --- 1. Deterministic baseline ---
    print('Running: Deterministic baseline...', flush=True)
    det_cfg = dict(DEFAULT_PROCESS_DET)
    det_cfg['sigma_L'] = 0.0
    det_cfg['alpha']   = 0.50
    r = run_experiment_ccp(tau=tau, data_file=data_file,
                           process_config=det_cfg,
                           time_limit=time_limit, random_seed=solve_seed,
                           solver_name=solver_name)
    r['method'] = 'Deterministic'
    if r.get('solved'):
        sim = simulate_schedule(
            patients=r['patients'],
            tmfe_eff=r['tmfe_eff'],
            tmfe_nominal=r['tmfe_nominal'],
            sigma_L=sigma_L,          # expose to sigma_L uncertainty
            n_sim=n_sim, seed=sim_seed,
        )
        r['simulation'] = sim
    results.append(r)
    _print_result(r)

    # --- 2. CCP variants ---
    for alpha in alphas:
        label = f'CCP(α={alpha})'
        print(f'Running: {label}...', flush=True)
        cfg = dict(DEFAULT_PROCESS_CCP)
        cfg['alpha']   = alpha
        cfg['sigma_L'] = sigma_L
        r = run_experiment_ccp(tau=tau, data_file=data_file,
                               process_config=cfg,
                               time_limit=time_limit, random_seed=solve_seed,
                               solver_name=solver_name)
        r['method'] = label
        if r.get('solved'):
            sim = simulate_schedule(
                patients=r['patients'],
                tmfe_eff=r['tmfe_eff'],
                tmfe_nominal=r['tmfe_nominal'],
                sigma_L=sigma_L,
                n_sim=n_sim, seed=sim_seed,
            )
            r['simulation'] = sim
        results.append(r)
        _print_result(r)

    # --- 3. Gamma-robust variants ---
    for gamma in gammas:
        label = f'Γ-robust(Γ={gamma})'
        print(f'Running: {label}...', flush=True)
        cfg = dict(DEFAULT_PROCESS_CCP)
        cfg['sigma_L'] = sigma_L
        r = run_experiment_robust(gamma=gamma, data_file=data_file, tau=tau,
                                  process_config=cfg,
                                  time_limit=time_limit, random_seed=solve_seed,
                                  solver_name=solver_name)
        r['method'] = label
        if r.get('solved'):
            sim = simulate_schedule(
                patients=r['patients'],
                tmfe_eff=r['tmfe_eff'],
                tmfe_nominal=r['tmfe_nominal'],
                sigma_L=sigma_L,
                n_sim=n_sim, seed=sim_seed,
            )
            r['simulation'] = sim
        results.append(r)
        _print_result(r)

    return results


def _print_result(r):
    solved = r.get('solved', False)
    sim    = r.get('simulation', {})
    cost   = f"${r['total_cost']:,.0f}" if solved else 'INFEASIBLE'
    esvc   = f"{sim.get('empirical_patient_svc', 0)*100:.1f}%" if sim else '—'
    tsvc   = f"{r.get('service_level', 1-r.get('alpha',0))*100:.0f}%" if solved else '—'
    print(f"  {r.get('method','?'):<22} | TMFE_eff={r.get('tmfe_eff',7):.2f}d "
          f"| Cost={cost:>12} | Theory={tsvc} Empirical={esvc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser(
        description='CCP Monte Carlo simulation validator + Gamma-robust comparison')
    parser.add_argument('--data',      default=None)
    parser.add_argument('--sigma-L',   type=float, default=0.20, dest='sigma_L')
    parser.add_argument('--n-sim',     type=int,   default=1000, dest='n_sim')
    parser.add_argument('--time-limit', type=int,  default=120)
    parser.add_argument('--json',      action='store_true')
    args = parser.parse_args()

    data_file = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Data_N15.dat')

    print(f'\nCAR-T CCP Validation | data={os.path.basename(data_file)} '
          f'sigma_L={args.sigma_L} n_sim={args.n_sim}\n')

    results = run_full_comparison(
        data_file=data_file,
        sigma_L=args.sigma_L,
        n_sim=args.n_sim,
        time_limit=args.time_limit,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print('\n' + '='*80)
        print(f"{'Method':<24} {'TMFE_eff':>8} {'Cost':>12} "
              f"{'Theory SL':>10} {'Empirical SL':>13} {'Avg Viol':>9}")
        print('-'*80)
        for r in results:
            sim  = r.get('simulation', {})
            esvc = sim.get('empirical_patient_svc', None)
            tsvc = r.get('service_level', None)
            print(
                f"{r.get('method','?'):<24} "
                f"{r.get('tmfe_eff',7):>8.2f} "
                f"{'$'+'{:,.0f}'.format(r['total_cost']) if r.get('solved') else 'INFEASIBLE':>12} "
                f"{f'{tsvc*100:.0f}%' if tsvc is not None else '—':>10} "
                f"{f'{esvc*100:.1f}%' if esvc is not None else '—':>13} "
                f"{sim.get('avg_violations_per_run',0):>9.2f}"
            )
