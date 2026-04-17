"""
CAR-T Supply Chain — Stochastic CCP Column Generation
======================================================
Extends cart_colgen.py with Chance-Constrained Programming (CCP)
to account for uncertain manufacturing time T_MF.

Uncertainty model
-----------------
  T_MF ~ LogNormal(mu_L, sigma_L^2)  with  E[T_MF] = tmfe_nominal (default 7 days)
  mu_L = ln(tmfe_nominal) - sigma_L^2 / 2

CCP deadline guarantee (per patient)
-------------------------------------
  P(TRT_p <= D_p) >= 1 - alpha
  <=>  F^{-1}(1-alpha) <= D_p - T_LS - TT1[j_out] - T_QC - TT3[j_ret]
  <=>  TRT^alpha_i  <= D_p

where  T^alpha_MF = F^{-1}(1-alpha) = exp(mu_L + z_alpha * sigma_L)
       z_alpha    = Phi^{-1}(1-alpha)  (standard-normal quantile)

This is a single scalar substitution: TMFE_eff replaces TMFE=7 in
  (a) the deadline feasibility filter at plan generation, and
  (b) mfg_end = mfg_start + TMFE_eff  for capacity occupancy.

Everything else — objective function, master IP, facility constraints,
routing constraints, capacity constraint structure — is UNCHANGED from
the deterministic cart_colgen.py.  No penalty terms.  No second stage.
No scenarios.

New parameters in process_config
---------------------------------
  alpha   : float in (0,1)  — allowed miss probability per patient (default 0.10)
  sigma_L : float > 0       — log-scale std-dev of T_MF distribution (default 0.20)
  dist    : str             — distribution family; only 'lognormal' supported (default)

Usage
-----
  python cart_colgen_ccp.py --data Data_N15.dat --alpha 0.10 --sigma-L 0.2
  python cart_colgen_ccp.py --sensitivity --data Data_N15.dat
"""

import math
import os
import random
import time as _time

# Re-use parse_dat and solve_master verbatim from the deterministic solver.
# The master IP is structurally identical; only plan generation changes.
from cart_colgen import parse_dat, solve_master, DEFAULT_URGENCY

# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

DEFAULT_PROCESS_DET = {'tls': 1, 'tmfe': 7, 'tqc': 7, 'max_facilities': 6}

DEFAULT_PROCESS_CCP = {
    'tls': 1,
    'tmfe': 7,          # nominal  E[T_MF]  — calibrates the distribution mean
    'tqc': 7,
    'max_facilities': 6,
    'alpha':   0.10,    # deadline-miss probability budget per patient
    'sigma_L': 0.20,    # LogNormal log-scale std-dev; 0 => deterministic
    'dist':    'lognormal',
}


# ---------------------------------------------------------------------------
# Quantile helper
# ---------------------------------------------------------------------------

def _norm_ppf(p):
    """Standard-normal quantile Phi^{-1}(p) via Python stdlib (3.8+)."""
    import statistics
    return statistics.NormalDist().inv_cdf(p)


def compute_tmfe_quantile(process_config):
    """Return T^alpha_MF = F^{-1}(1-alpha) for T_MF ~ LogNormal.

    Parameters read from process_config
    ------------------------------------
    tmfe    : nominal mean E[T_MF]  (default 7)
    alpha   : miss-probability budget in (0, 1)  (default 0.10)
    sigma_L : log-scale std-dev > 0              (default 0.20)

    Returns
    -------
    float — the effective manufacturing time to use in plan generation.
            Equals tmfe_nominal when sigma_L=0 or alpha=0.50.
    """
    tmfe_nom = float(process_config.get('tmfe', 7))
    alpha    = float(process_config.get('alpha', 0.10))
    sigma_L  = float(process_config.get('sigma_L', 0.20))

    if sigma_L <= 0.0:
        return tmfe_nom                   # degenerate: no uncertainty

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    z_alpha  = _norm_ppf(1.0 - alpha)    # e.g. alpha=0.10 -> z=1.2816
    mu_L     = math.log(tmfe_nom) - 0.5 * sigma_L ** 2
    return math.exp(mu_L + z_alpha * sigma_L)


# ---------------------------------------------------------------------------
# Plan generation — CCP variant
# ---------------------------------------------------------------------------

def generate_plans_ccp(data, tau=0.0, urgency_config=None,
                       process_config=None, random_seed=42):
    """Generate feasible patient routes under the CCP service-level guarantee.

    Identical to cart_colgen.generate_plans() except:
      - TMFE_eff = compute_tmfe_quantile(process_config)  replaces TMFE=7
      - base_trt filter uses TMFE_eff   (deadline feasibility at alpha level)
      - mfg_end uses TMFE_eff           (capacity occupancy duration)
      - plan dict records tmfe_eff, alpha, sigma_L, trt_nominal

    The master problem receives the same data structure as the deterministic
    solver and is called without any modification.
    """
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS_CCP

    P = data['p']
    C = data['c']
    H = data['h']
    J = data['j']
    M = data['m']

    TLS     = int(process_config.get('tls', 1))
    TMFE    = int(process_config.get('tmfe', 7))   # nominal (kept for reference)
    TQC     = int(process_config.get('tqc', 7))
    MAX_FAC = int(process_config.get('max_facilities', 6))
    alpha   = float(process_config.get('alpha', 0.10))
    sigma_L = float(process_config.get('sigma_L', 0.20))

    # --- CCP substitution: single scalar replaces TMFE=7 everywhere ---
    TMFE_eff     = compute_tmfe_quantile(process_config)
    # Ceiling integer used wherever range() / integer arithmetic is needed
    # (capacity window in solve_master).  The float is kept for exact reporting.
    TMFE_eff_int = math.ceil(TMFE_eff)

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

    # Urgency groups and deadlines (identical logic to deterministic)
    n        = len(P)
    num_high = int(round(n * urgency_config['high']['fraction']))
    num_med  = int(round(n * urgency_config['medium']['fraction']))
    random.seed(random_seed)
    shuffled = sorted(P, key=str)
    random.shuffle(shuffled)
    group_map, deadlines = {}, {}
    for idx, p in enumerate(shuffled):
        g   = 'high' if idx < num_high else ('medium' if idx < num_high + num_med else 'low')
        cfg = urgency_config[g]
        group_map[p]  = g
        deadlines[p]  = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)

    T_max         = 130
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
                    # CCP filter: use TMFE_eff (the (1-alpha)-quantile)
                    base_trt_ccp = TLS + TT1[j_out] + TMFE_eff + TQC + TT3[j_ret]
                    if base_trt_ccp > dl:
                        continue   # route cannot guarantee P(TRT<=D) >= 1-alpha

                    # Nominal TRT (for reporting only)
                    base_trt_nom = TLS + TT1[j_out] + TMFE + TQC + TT3[j_ret]

                    cost_out = data['U1'].get((c, mi, j_out), 0)
                    cost_ret = data['U3'].get((mi, h, j_ret), 0)

                    mfg_start = arr + TLS + TT1[j_out]
                    # Capacity occupancy uses ceiling of TMFE_eff (conservative)
                    mfg_end   = mfg_start + TMFE_eff_int

                    plan = {
                        'id':            plan_id,
                        'patient':       p,
                        'facility':      mi,
                        'j_out':         j_out,
                        'j_ret':         j_ret,
                        'wait':          0,
                        'trt':           base_trt_ccp,    # effective TRT at alpha
                        'trt_nominal':   base_trt_nom,    # what TRT would be at TMFE=7
                        'mfg_start':     mfg_start,
                        'mfg_end':       mfg_end,
                        'transport_cost': cost_out + cost_ret,
                        'cost':          cost_out + cost_ret,
                        'tmfe_eff':      round(TMFE_eff, 4),
                        'alpha':         alpha,
                        'sigma_L':       sigma_L,
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
        'TMFE':             TMFE_eff_int,  # master uses this int for capacity window
        'MAX_FAC':          MAX_FAC,
        'deadlines':        deadlines,
        'group_map':        group_map,
        'patient_arrival':  patient_arrival,
        'patient_center':   patient_center,
        # CCP metadata
        'tmfe_nominal':     TMFE,
        'tmfe_eff':         round(TMFE_eff, 4),
        'alpha':            alpha,
        'sigma_L':          sigma_L,
        'service_level':    1.0 - alpha,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment_ccp(tau=0.0, data_file=None, urgency_config=None,
                       process_config=None, time_limit=300, random_seed=42,
                       solver_name='appsi_highs'):
    """Run full CCP ColGen experiment and return result dict."""
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data_N50.dat')
    if process_config is None:
        process_config = DEFAULT_PROCESS_CCP

    t0   = _time.time()
    data = parse_dat(data_file)
    plan_data = generate_plans_ccp(
        data, tau=tau,
        urgency_config=urgency_config,
        process_config=process_config,
        random_seed=random_seed,
    )
    gen_time = _time.time() - t0

    n_plans    = len(plan_data['plans'])
    n_patients = len(plan_data['P'])
    tmfe_eff   = plan_data['tmfe_eff']
    alpha      = plan_data['alpha']
    sigma_L    = plan_data['sigma_L']

    infeasible_patients = [p for p in plan_data['P']
                           if not plan_data['plans_by_patient'][p]]
    if infeasible_patients:
        return {
            'solver':               'CCP-ColGen',
            'solved':               False,
            'reason':               'infeasible_patients',
            'infeasible_patients':  infeasible_patients,
            'num_patients':         n_patients,
            'num_plans':            0,
            'tmfe_eff':             tmfe_eff,
            'alpha':                alpha,
            'sigma_L':              sigma_L,
            'service_level':        1.0 - alpha,
            'gen_time':             round(gen_time, 3),
        }

    t0 = _time.time()
    _, res = solve_master(plan_data, time_limit=time_limit, solver_name=solver_name)
    solve_time = _time.time() - t0

    res['solver']        = 'CCP-ColGen'
    res['tau']           = tau
    res['num_patients']  = n_patients
    res['num_plans']     = n_plans
    res['tmfe_eff']      = tmfe_eff
    res['tmfe_nominal']  = plan_data['tmfe_nominal']
    res['alpha']         = alpha
    res['sigma_L']       = sigma_L
    res['service_level'] = 1.0 - alpha
    res['gen_time']      = round(gen_time, 3)
    res['solve_time']    = round(solve_time, 3)
    res['build_time']    = round(gen_time, 3)
    return res


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------

def run_sensitivity_ccp(data_file=None, tau=0.0,
                        alphas=None, sigma_Ls=None,
                        time_limit=120, random_seed=42,
                        solver_name='appsi_highs',
                        include_deterministic=True):
    """Sweep over (alpha, sigma_L) pairs and return list of result dicts.

    Parameters
    ----------
    alphas   : list of miss-probability values, default [0.20, 0.10, 0.05, 0.01]
    sigma_Ls : list of log-scale std-dev values, default [0.1, 0.2, 0.3, 0.5]
    include_deterministic : also run the baseline (alpha=0.50, sigma_L=0) first
    """
    if alphas is None:
        alphas = [0.20, 0.10, 0.05, 0.01]
    if sigma_Ls is None:
        sigma_Ls = [0.1, 0.2, 0.3, 0.5]
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data_N15.dat')

    results = []

    if include_deterministic:
        # Baseline: sigma_L=0 => TMFE_eff = 7 exactly (deterministic)
        det_cfg = dict(DEFAULT_PROCESS_DET)
        det_cfg['alpha']   = 0.50
        det_cfg['sigma_L'] = 0.0
        r = run_experiment_ccp(tau=tau, data_file=data_file,
                               process_config=det_cfg,
                               time_limit=time_limit, random_seed=random_seed,
                               solver_name=solver_name)
        r['label'] = 'Deterministic (baseline)'
        results.append(r)

    for sigma_L in sigma_Ls:
        for alpha in alphas:
            cfg = dict(DEFAULT_PROCESS_CCP)
            cfg['alpha']   = alpha
            cfg['sigma_L'] = sigma_L
            r = run_experiment_ccp(tau=tau, data_file=data_file,
                                   process_config=cfg,
                                   time_limit=time_limit, random_seed=random_seed,
                                   solver_name=solver_name)
            r['label'] = f'sigma_L={sigma_L}, alpha={alpha}'
            results.append(r)

    return results


def print_sensitivity_table(results):
    """Print a formatted ASCII table of sensitivity results."""
    hdr = (f"{'Label':<35} {'TMFE_eff':>8} {'Plans':>6} "
           f"{'Cost':>12} {'Avg TRT':>8} {'SvcLvl':>7} {'Status':<10}")
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        label    = r.get('label', '')
        teff     = f"{r.get('tmfe_eff', 7.0):.2f}d"
        plans    = str(r.get('num_plans', 0))
        solved   = r.get('solved', False)
        cost     = f"${r['total_cost']:,.0f}" if solved else 'INFEASIBLE'
        avg_trt  = f"{r['avg_trt']:.1f}d"   if solved else '—'
        svc      = f"{r.get('service_level', 1.0)*100:.0f}%"
        status   = 'OK' if solved else 'FAIL'
        print(f"{label:<35} {teff:>8} {plans:>6} "
              f"{cost:>12} {avg_trt:>8} {svc:>7} {status:<10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser(
        description='CCP Stochastic Column Generation for CAR-T supply chain')
    parser.add_argument('--data',      default=None,  help='.dat file path')
    parser.add_argument('--tau',       type=float, default=0.0,
                        help='Deadline tightness (0=loose, 1=tight)')
    parser.add_argument('--alpha',     type=float, default=0.10,
                        help='Allowed miss-probability per patient (default 0.10)')
    parser.add_argument('--sigma-L',   type=float, default=0.20, dest='sigma_L',
                        help='LogNormal log-std-dev of T_MF (default 0.20)')
    parser.add_argument('--time-limit', type=int,  default=300)
    parser.add_argument('--sensitivity', action='store_true',
                        help='Sweep alpha x sigma_L grid')
    parser.add_argument('--json',      action='store_true',
                        help='Output results as JSON')
    args = parser.parse_args()

    data_file = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Data_N15.dat')

    if args.sensitivity:
        results = run_sensitivity_ccp(
            data_file=data_file, tau=args.tau,
            time_limit=args.time_limit)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print('\nCCP Sensitivity Analysis — CAR-T ColGen\n')
            print_sensitivity_table(results)
    else:
        cfg = dict(DEFAULT_PROCESS_CCP)
        cfg['alpha']   = args.alpha
        cfg['sigma_L'] = args.sigma_L
        r = run_experiment_ccp(
            tau=args.tau, data_file=data_file,
            process_config=cfg, time_limit=args.time_limit)
        if args.json:
            print(json.dumps(r, indent=2))
        elif r['solved']:
            print(
                f"CCP-ColGen | alpha={args.alpha} sigma_L={args.sigma_L} "
                f"TMFE_eff={r['tmfe_eff']:.2f}d | "
                f"Cost: ${r['total_cost']:,.0f} | "
                f"TRT: {r['avg_trt']:.1f}d | "
                f"Plans: {r['num_plans']} | "
                f"SvcLvl: {r['service_level']*100:.0f}% | "
                f"{'OPTIMAL' if r.get('optimal') else 'FEASIBLE'}"
            )
        else:
            print(f"CCP-ColGen | FAILED: {r.get('reason', r.get('termination', '?'))}")
            if 'infeasible_patients' in r:
                print(f"  Infeasible patients: {r['infeasible_patients']}")
