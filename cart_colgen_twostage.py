"""
CAR-T Supply Chain — Two-Stage Stochastic ColGen with Expediting Recourse
=========================================================================
Extends cart_colgen.py with a two-stage stochastic formulation where the
only recourse action is expediting the return logistics after T_MF is revealed.

Model
-----
  Stage 1 (here-and-now):
      Select route x_i (facility m, outbound hub j_out, standard return hub j_ret).
      These decisions are fixed before T_MF is known.

  Stage 2 (wait-and-see):
      Observe actual T_MF. If the patient would be late with standard return,
      expedite: pay premium pi_p, return time reduced by delta days.
      No penalty terms — deadlines are enforced via the plan filter.

Modified column cost
--------------------
  c_tilde_i = c_transport_i  +  pi_p * Pr(T_MF > K_i)

  where  K_i    = D_p - T_LS - TT1[j_out] - T_QC - TT3[j_ret]   (slack)
         Pr(T_MF > K_i) = 1 - Phi((ln(K_i) - mu_L) / sigma_L)    (LogNormal survival)
         pi_p   = expediting premium ($/shipment), urgency-dependent

Plan feasibility filter
-----------------------
  K_i + delta >= F^{-1}(1 - epsilon)

  i.e. even with expedited return the patient can meet the deadline at
  probability (1 - epsilon). Routes that fail this are removed.
  This replaces the deterministic filter  K_i >= T_MF_nominal.

New parameters (process_config keys)
-------------------------------------
  sigma_L  : float  — LogNormal log-scale std-dev of T_MF     (default 0.20)
  delta    : float  — days saved by expedited return            (default 2.0)
  epsilon  : float  — filter: allowed infeasibility probability (default 0.05)
  pi_high  : float  — expediting premium, high-urgency ($)     (default 8000)
  pi_medium: float  — expediting premium, medium-urgency ($)   (default 4000)
  pi_low   : float  — expediting premium, low-urgency ($)      (default 1500)

Usage
-----
  python cart_colgen_twostage.py --data Data_N15.dat
  python cart_colgen_twostage.py --data Data_N15.dat --delta 2 --sigma-L 0.20
  python cart_colgen_twostage.py --data Data_N15.dat --sensitivity
"""

import math
import os
import random
import time as _time
import statistics

from cart_colgen import parse_dat, solve_master, DEFAULT_URGENCY


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_PROCESS_2S = {
    'tls':      1,
    'tmfe':     7,       # nominal mean E[T_MF]
    'tqc':      7,
    'max_facilities': 2,
    'sigma_L':  0.20,    # LogNormal log-scale std-dev
    'delta':    2.0,     # days saved by expedited return
    'epsilon':  0.05,    # filter: allowed infeasibility probability
    'pi_high':  8000,    # expediting premium — high-urgency patient
    'pi_medium':4000,    # expediting premium — medium-urgency patient
    'pi_low':   1500,    # expediting premium — low-urgency patient
}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    """Standard normal CDF via Python stdlib."""
    return statistics.NormalDist().cdf(x)

def _norm_ppf(p):
    """Standard normal quantile (inverse CDF)."""
    return statistics.NormalDist().inv_cdf(p)

def _lognormal_survival(K, tmfe_nominal, sigma_L):
    """Pr(T_MF > K) for T_MF ~ LogNormal(mu_L, sigma_L^2).

    Returns 1.0 when K <= 0 (T_MF always exceeds non-positive slack).
    Returns 0.0 when sigma_L = 0 (deterministic: T_MF = tmfe_nominal exactly).
    """
    if sigma_L <= 0.0:
        return 1.0 if tmfe_nominal > K else 0.0
    if K <= 0.0:
        return 1.0
    mu_L = math.log(tmfe_nominal) - 0.5 * sigma_L ** 2
    return 1.0 - _norm_cdf((math.log(K) - mu_L) / sigma_L)

def _lognormal_quantile(p, tmfe_nominal, sigma_L):
    """F^{-1}(p) for T_MF ~ LogNormal(mu_L, sigma_L^2)."""
    if sigma_L <= 0.0:
        return tmfe_nominal
    mu_L = math.log(tmfe_nominal) - 0.5 * sigma_L ** 2
    return math.exp(mu_L + _norm_ppf(p) * sigma_L)


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def generate_plans_twostage(data, tau=0.0, urgency_config=None,
                             process_config=None, random_seed=42):
    """Generate feasible plans with two-stage expediting recourse.

    Key differences from deterministic generate_plans():
    1. Plan filter: K_i + delta >= F^{-1}(1-epsilon)   [not K_i >= TMFE]
    2. Column cost: transport_cost + pi_p * Pr(T_MF > K_i)
    3. Capacity window: uses ceil(F^{-1}(1-epsilon)) — the (1-epsilon) quantile
       of T_MF — so slot reservation is consistent with the route filter's
       risk threshold (not the mean TMFE). This prevents slot overflow when
       actual manufacturing exceeds the nominal 7-day mean.
    """
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS_2S

    P  = data['p'];  C = data['c'];  H = data['h']
    J  = data['j'];  M = data['m']

    TLS     = int(process_config.get('tls',   1))
    TMFE    = int(process_config.get('tmfe',  7))
    TQC     = int(process_config.get('tqc',   7))
    MAX_FAC = int(process_config.get('max_facilities', 2))
    sigma_L = float(process_config.get('sigma_L',  0.20))
    delta   = float(process_config.get('delta',    2.0))
    epsilon = float(process_config.get('epsilon',  0.05))

    pi_map = {
        'high':   float(process_config.get('pi_high',   8000)),
        'medium': float(process_config.get('pi_medium', 4000)),
        'low':    float(process_config.get('pi_low',    1500)),
    }

    # Filter threshold: K_i + delta >= F^{-1}(1-epsilon)
    # => K_i >= filter_threshold
    tmfe_quantile = _lognormal_quantile(1.0 - epsilon, TMFE, sigma_L)
    filter_threshold = tmfe_quantile - delta   # minimum slack required

    # Capacity slot quantile: F^{-1}(0.90) — 90th percentile of T_MF.
    # Matches CCP's alpha=0.10 so both models plan capacity at the same
    # risk level. Using 1-epsilon (95th pct) over-constrains scheduling
    # at large N, forcing unnecessarily expensive facility choices.
    cap_alpha   = float(process_config.get('cap_alpha', 0.10))
    tmfe_cap_q  = _lognormal_quantile(1.0 - cap_alpha, TMFE, sigma_L)

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

    # Urgency groups and deadlines
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
        g   = group_map[p]
        pi  = pi_map[g]

        for mi in M:
            for j_out in J:
                for j_ret in J:
                    # Slack: manufacturing time available before patient is late
                    # with STANDARD return
                    K_i = dl - (TLS + TT1[j_out] + TQC + TT3[j_ret])

                    # Filter: even with expediting, deadline must be achievable
                    # at probability (1 - epsilon)
                    if K_i < filter_threshold:
                        continue

                    # Probability that expediting will be needed
                    prob_expedite = _lognormal_survival(K_i, TMFE, sigma_L)

                    # Expected expediting cost
                    exp_expedite_cost = pi * prob_expedite

                    # Transport cost (Stage 1)
                    cost_out = data['U1'].get((c, mi, j_out), 0)
                    cost_ret = data['U3'].get((mi, h, j_ret), 0)
                    transport_cost = cost_out + cost_ret

                    # Modified column cost = transport + expected expediting
                    total_cost = transport_cost + exp_expedite_cost

                    mfg_start = arr + TLS + TT1[j_out]
                    # Capacity slot uses ceil(F^{-1}(1-cap_alpha)) — 90th pct
                    # of T_MF (matching CCP's alpha=0.10). Prevents slot
                    # overflow without over-constraining scheduling at large N.
                    TMFE_cap  = math.ceil(tmfe_cap_q)
                    mfg_end   = mfg_start + TMFE_cap

                    # Nominal TRT (with standard return)
                    trt_nominal = TLS + TT1[j_out] + TMFE + TQC + TT3[j_ret]
                    # Best-case TRT (with expedited return)
                    trt_expedited = TLS + TT1[j_out] + TMFE + TQC + (TT3[j_ret] - delta)

                    plan = {
                        'id':                 plan_id,
                        'patient':            p,
                        'facility':           mi,
                        'j_out':              j_out,
                        'j_ret':              j_ret,
                        'wait':               0,
                        'trt':                trt_nominal,
                        'trt_expedited':      trt_expedited,
                        'mfg_start':          mfg_start,
                        'mfg_end':            mfg_end,
                        'slack':              round(K_i, 4),
                        'prob_expedite':      round(prob_expedite, 4),
                        'transport_cost':     transport_cost,
                        'exp_expedite_cost':  round(exp_expedite_cost, 2),
                        'cost':               total_cost,   # solve_master reads this
                        'sigma_L':            sigma_L,
                        'delta':              delta,
                        'pi':                 pi,
                        'group':              g,
                    }
                    plans.append(plan)
                    plans_by_patient[p].append(plan_id)
                    plan_id += 1

    return {
        'plans':             plans,
        'plans_by_patient':  plans_by_patient,
        'P': P, 'M': M, 'J': J,
        'FCAP':              FCAP,
        'CIM':               CIM,
        'CVM':               CVM,
        'fac_cost_total':    fac_cost_total,
        'T_max':             T_max,
        'TMFE':              math.ceil(tmfe_cap_q),  # capacity window = slot duration (90th pct)
        'TMFE_nominal':      TMFE,                   # original mean, kept for reference
        'MAX_FAC':           MAX_FAC,
        'deadlines':         deadlines,
        'group_map':         group_map,
        'patient_arrival':   patient_arrival,
        'patient_center':    patient_center,
        # Two-stage metadata
        'sigma_L':           sigma_L,
        'delta':             delta,
        'epsilon':           epsilon,
        'filter_threshold':  round(filter_threshold, 4),
        'tmfe_quantile':     round(tmfe_quantile, 4),
        'pi_map':            pi_map,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment_twostage(tau=0.0, data_file=None, urgency_config=None,
                             process_config=None, time_limit=300,
                             random_seed=42, solver_name='appsi_highs'):
    """Run full two-stage expediting experiment and return result dict."""
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data_N15.dat')
    if process_config is None:
        process_config = DEFAULT_PROCESS_2S

    t0        = _time.time()
    data      = parse_dat(data_file)
    plan_data = generate_plans_twostage(
        data, tau=tau, urgency_config=urgency_config,
        process_config=process_config, random_seed=random_seed)
    gen_time  = _time.time() - t0

    n_plans    = len(plan_data['plans'])
    n_patients = len(plan_data['P'])

    infeasible = [p for p in plan_data['P']
                  if not plan_data['plans_by_patient'][p]]
    if infeasible:
        return {
            'solver':              '2S-Expediting',
            'solved':              False,
            'reason':              'infeasible_patients',
            'infeasible_patients': infeasible,
            'num_patients':        n_patients,
            'num_plans':           0,
            'sigma_L':             plan_data['sigma_L'],
            'delta':               plan_data['delta'],
            'epsilon':             plan_data['epsilon'],
            'gen_time':            round(gen_time, 3),
        }

    t0         = _time.time()
    _, res     = solve_master(plan_data, time_limit=time_limit,
                              solver_name=solver_name)
    solve_time = _time.time() - t0

    res['solver']           = '2S-Expediting'
    res['tau']              = tau
    res['num_patients']     = n_patients
    res['num_plans']        = n_plans
    res['sigma_L']          = plan_data['sigma_L']
    res['delta']            = plan_data['delta']
    res['epsilon']          = plan_data['epsilon']
    res['filter_threshold'] = plan_data['filter_threshold']
    res['tmfe_quantile']    = plan_data['tmfe_quantile']
    res['pi_map']           = plan_data['pi_map']
    res['gen_time']         = round(gen_time, 3)
    res['solve_time']       = round(solve_time, 3)
    res['build_time']       = round(gen_time, 3)

    # Annotate patients with expediting probability
    if res.get('patients'):
        plan_lookup = {pl['id']: pl for pl in plan_data['plans']}
        for pat in res['patients']:
            # find which plan was selected for this patient
            pid = pat['id']
            selected_plan = next(
                (plan_lookup[i] for i in plan_data['plans_by_patient'][pid]
                 if plan_lookup[i]['facility'] == pat['facility']
                 and plan_lookup[i]['j_out'] == pat['j_out']
                 and plan_lookup[i]['j_ret'] == pat['j_ret']), None)
            if selected_plan:
                pat['prob_expedite']     = selected_plan['prob_expedite']
                pat['exp_expedite_cost'] = selected_plan['exp_expedite_cost']
                pat['slack']             = selected_plan['slack']

    # Summary expediting stats
    if res.get('patients'):
        probs = [p.get('prob_expedite', 0) for p in res['patients']]
        res['avg_prob_expedite'] = round(sum(probs) / len(probs), 4)
        res['total_exp_expedite_cost'] = round(
            sum(p.get('exp_expedite_cost', 0) for p in res['patients']), 2)

    return res


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------

def run_sensitivity_twostage(data_file=None, tau=0.0,
                              deltas=None, sigma_Ls=None, epsilons=None,
                              time_limit=120, random_seed=42,
                              solver_name='appsi_highs'):
    """Sweep over (delta, sigma_L) pairs and return list of result dicts."""
    if deltas   is None: deltas   = [1.0, 2.0, 3.0]
    if sigma_Ls is None: sigma_Ls = [0.10, 0.20, 0.30]
    if epsilons is None: epsilons = [0.05]
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data_N15.dat')

    results = []
    # Deterministic baseline
    from cart_colgen import run_experiment as det_run
    r = det_run(tau=tau, data_file=data_file, time_limit=time_limit)
    r['label'] = 'Deterministic (baseline)'
    r['delta'] = 0; r['sigma_L'] = 0; r['epsilon'] = 0
    results.append(r)

    for sigma_L in sigma_Ls:
        for delta in deltas:
            cfg = dict(DEFAULT_PROCESS_2S)
            cfg['sigma_L'] = sigma_L
            cfg['delta']   = delta
            r = run_experiment_twostage(
                tau=tau, data_file=data_file, process_config=cfg,
                time_limit=time_limit, random_seed=random_seed,
                solver_name=solver_name)
            r['label'] = f'sigma_L={sigma_L}, delta={delta}d'
            results.append(r)

    return results


def print_sensitivity_table(results):
    hdr = (f"{'Label':<35} {'Filter':>8} {'Plans':>6} "
           f"{'Total Cost':>14} {'Avg TRT':>8} {'AvgP(exp)':>10} {'Status'}")
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        solved = r.get('solved', False)
        cost_s  = f"${r['total_cost']:,.0f}" if solved else 'INFEASIBLE'
        trt_s   = f"{r['avg_trt']:.1f}d"    if solved else '—'
        pexp_s  = f"{r.get('avg_prob_expedite',0)*100:.1f}%" if solved else '—'
        print(
            f"{r.get('label',''):<35} "
            f"{r.get('filter_threshold', 0):>8.2f} "
            f"{r.get('num_plans', 0):>6} "
            f"{cost_s:>14} {trt_s:>8} {pexp_s:>10} "
            f"{'OK' if solved else 'FAIL'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser(
        description='Two-Stage Stochastic ColGen with Expediting Recourse')
    parser.add_argument('--data',       default=None)
    parser.add_argument('--tau',        type=float, default=0.0)
    parser.add_argument('--sigma-L',    type=float, default=0.20, dest='sigma_L')
    parser.add_argument('--delta',      type=float, default=2.0,
                        help='Days saved by expedited return (default 2.0)')
    parser.add_argument('--epsilon',    type=float, default=0.05,
                        help='Filter: allowed infeasibility probability (default 0.05)')
    parser.add_argument('--pi-high',    type=float, default=8000,  dest='pi_high')
    parser.add_argument('--pi-medium',  type=float, default=4000,  dest='pi_medium')
    parser.add_argument('--pi-low',     type=float, default=1500,  dest='pi_low')
    parser.add_argument('--time-limit', type=int,   default=300)
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--json',       action='store_true')
    args = parser.parse_args()

    data_file = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Data_N15.dat')

    if args.sensitivity:
        results = run_sensitivity_twostage(
            data_file=data_file, tau=args.tau, time_limit=args.time_limit)
        if args.json:
            print(json.dumps([{k:v for k,v in r.items() if k!='patients'} for r in results], indent=2))
        else:
            print('\nTwo-Stage Expediting Sensitivity — CAR-T ColGen\n')
            print_sensitivity_table(results)
    else:
        cfg = dict(DEFAULT_PROCESS_2S)
        cfg['sigma_L']   = args.sigma_L
        cfg['delta']     = args.delta
        cfg['epsilon']   = args.epsilon
        cfg['pi_high']   = args.pi_high
        cfg['pi_medium'] = args.pi_medium
        cfg['pi_low']    = args.pi_low

        r = run_experiment_twostage(
            tau=args.tau, data_file=data_file, process_config=cfg,
            time_limit=args.time_limit)

        if args.json:
            print(json.dumps({k:v for k,v in r.items() if k!='patients'}, indent=2))
        elif r['solved']:
            print(
                f"2S-Expediting | sigma_L={args.sigma_L} delta={args.delta}d "
                f"epsilon={args.epsilon} | "
                f"Cost: ${r['total_cost']:,.0f} | TRT: {r['avg_trt']:.1f}d | "
                f"Plans: {r['num_plans']} | "
                f"AvgP(expedite): {r.get('avg_prob_expedite',0)*100:.1f}% | "
                f"E[expedite cost]: ${r.get('total_exp_expedite_cost',0):,.0f} | "
                f"{'OPTIMAL' if r.get('optimal') else 'FEASIBLE'}"
            )
        else:
            print(f"2S-Expediting | FAILED: {r.get('reason', r.get('termination', '?'))}")
            if 'infeasible_patients' in r:
                print(f"  Infeasible: {r['infeasible_patients']}")
