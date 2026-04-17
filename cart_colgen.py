"""
CAR-T Supply Chain — Column Generation (Hard Deadline)
======================================================
Generates feasible patient plans (columns), then solves a
compact set-partitioning master problem.

A "plan" for patient p = (facility m, outbound mode j, return mode k, wait w)
  - TRT = TLS + TT1[j] + TMF + TQC + TT3[k] + w
  - Hard constraint: TRT <= D_p
  - mfg_start = arrival_p + TLS + TT1[j] + w
  - mfg_end   = mfg_start + TMF

Master problem (IP):
  min  sum_{plan} cost_{plan} * x_{plan}
  s.t. sum_{plan in Plans_p} x_{plan} = 1          for each patient p
       sum_{plan using m at time t} x_{plan} <= FCAP_m   for each m, t
       sum_m z_m <= MAX_FAC
       x_{plan} <= z_{facility(plan)}
       x_{plan} in {0,1}, z_m in {0,1}
"""

import re, random, os, time as _time, math
from itertools import combinations


def parse_dat(filepath):
    with open(filepath) as f:
        txt = f.read()
    data = {}
    for s in ['c', 'h', 'j', 'm', 'p']:
        m = re.search(rf'set {s}\s*:=(.*?);', txt, re.DOTALL)
        data[s] = m.group(1).split() if m else []
    for param in ['CIM', 'CVM', 'FCAP', 'TT1', 'TT3']:
        m = re.search(rf'param {param}\s*:=(.*?);', txt, re.DOTALL)
        if m:
            pairs = re.findall(r'(\S+)\s+(\S+)', m.group(1))
            data[param] = {k: float(v) for k, v in pairs}
    m = re.search(r'param U1\s*:=(.*?);', txt, re.DOTALL)
    data['U1'] = {}
    if m:
        for c, mi, j, v in re.findall(r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', m.group(1)):
            data['U1'][(c, mi, j)] = float(v)
    m = re.search(r'param U3\s*:=(.*?);', txt, re.DOTALL)
    data['U3'] = {}
    if m:
        for mi, h, j, v in re.findall(r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', m.group(1)):
            data['U3'][(mi, h, j)] = float(v)
    m = re.search(r'param INC\s*:=(.*?);', txt, re.DOTALL)
    data['INC'] = {}
    if m:
        for p, c, t, v in re.findall(r'(\S+)\s+(\S+)\s+(\d+)\s+(\d+)', m.group(1)):
            if int(v) == 1:
                data['INC'][(p, c, int(t))] = 1
    m = re.search(r'param TLS\s*:=\s*(\S+)\s*;', txt)
    data['TLS'] = float(m.group(1)) if m else 1
    return data


DEFAULT_URGENCY = {
    'high':   {'fraction': 0.20, 'base_due': 18, 'tolerance': 1},
    'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
    'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
}
DEFAULT_PROCESS = {'tls': 1, 'tmfe': 7, 'tqc': 7, 'max_facilities': 6}


def generate_plans(data, tau=0.0, urgency_config=None, process_config=None, random_seed=42):
    """Generate all feasible patient plans."""
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS

    P = data['p']
    C = data['c']
    H = data['h']
    J = data['j']
    M = data['m']

    TLS  = int(process_config.get('tls', 1))
    TMFE = int(process_config.get('tmfe', 7))
    TQC  = int(process_config.get('tqc', 7))
    MAX_FAC = int(process_config.get('max_facilities', 6))
    TT1  = {j: int(data['TT1'][j]) for j in J}
    TT3  = {j: int(data['TT3'][j]) for j in J}
    FCAP = {m: int(data['FCAP'][m]) for m in M}
    CIM  = {m: data['CIM'][m] for m in M}
    CVM  = {m: data.get('CVM', {}).get(m, 0) for m in M}

    # Patient data
    patient_arrival = {}
    patient_center = {}
    c_to_h = {C[i]: H[i] for i in range(len(C))}
    for (p, c, t), v in data['INC'].items():
        if v == 1:
            patient_arrival[p] = t
            patient_center[p] = c

    # Assign urgency groups and deadlines
    n = len(P)
    num_high = int(round(n * urgency_config['high']['fraction']))
    num_med  = int(round(n * urgency_config['medium']['fraction']))
    random.seed(random_seed)
    shuffled = sorted(P, key=str)  # must match MIP's sorted(instance.p)
    random.shuffle(shuffled)
    group_map, deadlines = {}, {}
    for idx, p in enumerate(shuffled):
        g = 'high' if idx < num_high else ('medium' if idx < num_high + num_med else 'low')
        cfg = urgency_config[g]
        group_map[p] = g
        deadlines[p] = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)

    # Compute time horizon (must match MIP: hardcoded 130 in build_model)
    max_arr = max(patient_arrival.values())
    T_max = 130  # match hard_deadline_model.py RangeSet(130)

    # Facility cost (amortized — matches MIP formula exactly)
    # MIP: CTM_p = sum_m E_m * (CIM_m + CVM_m) * |T| / |P|
    # Total over all patients: sum_m E_m * (CIM_m + CVM_m) * |T|
    fac_cost_total = {m: (CIM[m] + CVM[m]) * T_max for m in M}

    # Generate plans
    plans = []  # list of plan dicts
    plans_by_patient = {p: [] for p in P}
    plan_id = 0

    for p in P:
        arr = patient_arrival[p]
        c = patient_center[p]
        h = c_to_h[c]
        dl = deadlines[p]

        for mi in M:
            for j_out in J:
                for j_ret in J:
                    base_trt = TLS + TT1[j_out] + TMFE + TQC + TT3[j_ret]
                    if base_trt > dl:
                        continue  # infeasible even with wait=0

                    # MIP MSB1 constraint forces patient to leave center exactly
                    # TLS days after arrival — NO waiting at center allowed.
                    # wait is always 0 to match MIP formulation exactly.
                    max_wait = 0

                    # Transport cost
                    cost_out = data['U1'].get((c, mi, j_out), 0)
                    cost_ret = data['U3'].get((mi, h, j_ret), 0)
                    transport_cost = cost_out + cost_ret

                    # Generate plans (wait=0 only to match MIP)
                    for w in range(max_wait + 1):
                        trt = base_trt + w
                        mfg_start = arr + TLS + TT1[j_out] + w
                        mfg_end = mfg_start + TMFE

                        plan = {
                            'id': plan_id,
                            'patient': p,
                            'facility': mi,
                            'j_out': j_out,
                            'j_ret': j_ret,
                            'wait': w,
                            'trt': trt,
                            'mfg_start': mfg_start,
                            'mfg_end': mfg_end,
                            'transport_cost': transport_cost,
                            'cost': transport_cost,  # facility cost added in master
                        }
                        plans.append(plan)
                        plans_by_patient[p].append(plan_id)
                        plan_id += 1

    return {
        'plans': plans,
        'plans_by_patient': plans_by_patient,
        'P': P, 'M': M, 'J': J,
        'FCAP': FCAP, 'CIM': CIM, 'CVM': CVM,
        'fac_cost_total': fac_cost_total,
        'T_max': T_max, 'TMFE': TMFE,
        'MAX_FAC': MAX_FAC,
        'deadlines': deadlines,
        'group_map': group_map,
        'patient_arrival': patient_arrival,
        'patient_center': patient_center,
    }


def solve_master(plan_data, time_limit=300, solver_name='appsi_highs'):
    """Solve the master set-partitioning problem."""
    from pyomo.environ import (ConcreteModel, Var, Binary, Objective, minimize,
                                Constraint, SolverFactory, SolverStatus,
                                TerminationCondition, value, RangeSet)

    plans = plan_data['plans']
    plans_by_patient = plan_data['plans_by_patient']
    P = plan_data['P']
    M = plan_data['M']
    FCAP = plan_data['FCAP']
    fac_cost = plan_data['fac_cost_total']
    T_max = plan_data['T_max']
    TMFE = plan_data['TMFE']
    MAX_FAC = plan_data['MAX_FAC']
    N = len(P)

    if not plans:
        return None, {'solved': False, 'reason': 'no feasible plans'}

    model = ConcreteModel()

    # Plan selection variables
    plan_ids = range(len(plans))
    model.x = Var(plan_ids, within=Binary)

    # Facility open variables
    model.z = Var(M, within=Binary)

    # Objective: transport cost + facility cost (amortized)
    def obj_rule(model):
        transport = sum(plans[i]['cost'] * model.x[i] for i in plan_ids)
        facility = sum(fac_cost[m] * model.z[m] / N for m in M) * N
        return transport + facility
    model.obj = Objective(rule=obj_rule)

    # Each patient selects exactly one plan
    def assign_rule(model, p_idx):
        p = P[p_idx]
        return sum(model.x[i] for i in plans_by_patient[p]) == 1
    model.assign = Constraint(range(len(P)), rule=assign_rule)

    # Facility selection limit
    def fac_limit_rule(model):
        return sum(model.z[m] for m in M) <= MAX_FAC
    model.fac_limit = Constraint(rule=fac_limit_rule)

    # Plan can only use open facility
    def fac_link_rule(model, i):
        return model.x[i] <= model.z[plans[i]['facility']]
    model.fac_link = Constraint(plan_ids, rule=fac_link_rule)

    # Capacity: replicate MIP's exact CAP1 + CAPCON1 formulation
    # CAP1: CAP[m,t] = FCAP[m] - sum(INM[p,m,tt] for tt in [t-TMFE, t-1])
    # CAPCON1: sum_p INM[p,m,t] - sum_p OUTM[p,m,t] <= CAP[m,t]
    #
    # In plan terms:
    # INM at (m, t) = plans with facility=m and mfg_start=t
    # OUTM at (m, t) = plans with facility=m and mfg_start=t-TMFE (i.e., mfg_end=t)
    #
    # Combined: entering(t) - leaving(t) <= FCAP - sum(entering(tt) for tt in [t-TMFE, t-1])

    fac_entering = {}  # (facility, time) -> list of plan ids entering mfg at that time
    fac_leaving = {}   # (facility, time) -> list of plan ids leaving mfg at that time
    all_fac_times = set()

    for i, plan in enumerate(plans):
        mi = plan['facility']
        ms = plan['mfg_start']
        me = plan['mfg_end']  # = mfg_start + TMFE
        fac_entering.setdefault((mi, ms), []).append(i)
        fac_leaving.setdefault((mi, me), []).append(i)
        # Track all relevant times for this facility
        for t in range(max(1, ms - TMFE), me + TMFE + 1):
            all_fac_times.add((mi, t))

    cap_keys = []
    cap_data = []  # (mi, t, entering_plans, leaving_plans, recent_plans)
    for mi in M:
        times_for_fac = sorted(set(t for (m, t) in all_fac_times if m == mi))
        for t in times_for_fac:
            entering = fac_entering.get((mi, t), [])
            leaving = fac_leaving.get((mi, t), [])
            # Recent entries: plans entering at [t-TMFE, t-1]
            recent = []
            for tt in range(max(1, t - TMFE), t):
                recent.extend(fac_entering.get((mi, tt), []))
            if entering or leaving or recent:
                cap_keys.append(len(cap_data))
                cap_data.append((mi, t, entering, leaving, recent))

    def cap_rule(model, key_idx):
        mi, t, entering, leaving, recent = cap_data[key_idx]
        lhs = (sum(model.x[i] for i in entering) -
               sum(model.x[i] for i in leaving))
        rhs = FCAP[mi] - sum(model.x[i] for i in recent)
        return lhs <= rhs
    model.capacity = Constraint(range(len(cap_data)), rule=cap_rule)

    # Solve — try appsi_highs / highs, fall back to glpk, then cbc
    def _try_solve(sname):
        try:
            s = SolverFactory(sname)
            try:
                avail = s.available(exception_flag=False)
            except TypeError:
                avail = s.available()
            except Exception:
                avail = False
            if not avail:
                return None, None
            if sname in ('appsi_highs', 'highs'):
                try: s.options['time_limit'] = time_limit
                except Exception: pass
            elif sname == 'glpk':
                try: s.options['tmlim'] = time_limit
                except Exception: pass
            elif sname == 'cbc':
                try: s.options['sec'] = time_limit
                except Exception: pass
            r = s.solve(model, tee=False)
            return s, r
        except Exception:
            return None, None

    solvers_to_try = ['appsi_highs', 'highs', 'glpk', 'cbc']
    if solver_name not in ('appsi_highs', 'highs'):
        solvers_to_try = [solver_name] + solvers_to_try
    results = None
    used_solver = None
    for sn in solvers_to_try:
        used_solver, results = _try_solve(sn)
        if results is not None:
            break

    if results is None:
        import warnings
        warnings.warn(
            "No MIP solver found (tried appsi_highs, highs, glpk, cbc). "
            "Install one with:  pip install highspy  OR  sudo apt-get install glpk-utils",
            RuntimeWarning, stacklevel=2,
        )
        return model, {'solved': False, 'termination': 'no_solver_available'}

    ok = (results.solver.status == SolverStatus.ok and
          results.solver.termination_condition in
          (TerminationCondition.optimal, TerminationCondition.feasible))

    if not ok:
        return model, {
            'solved': False,
            'termination': str(results.solver.termination_condition),
        }

    # Extract results
    selected = [i for i in plan_ids if value(model.x[i]) > 0.5]
    facs_open = [m for m in M if value(model.z[m]) > 0.5]

    total_transport = sum(plans[i]['cost'] for i in selected)
    total_fac = sum(fac_cost[m] for m in facs_open)
    mat_cost = (10476 + 9312) * N
    total_cost = total_transport + total_fac + mat_cost

    patient_results = []
    group_summary = {g: {'count': 0, 'trts': []} for g in ('high', 'medium', 'low')}
    for i in selected:
        plan = plans[i]
        p = plan['patient']
        g = plan_data['group_map'][p]
        patient_results.append({
            'id': p, 'group': g,
            'facility': plan['facility'],
            'j_out': plan['j_out'], 'j_ret': plan['j_ret'],
            'wait': plan['wait'], 'trt': plan['trt'],
            'mfg_start': plan['mfg_start'],
            'deadline': plan_data['deadlines'][p],
            'on_time': plan['trt'] <= plan_data['deadlines'][p],
        })
        if g in group_summary:
            group_summary[g]['count'] += 1
            group_summary[g]['trts'].append(plan['trt'])

    summary = {}
    for g, info in group_summary.items():
        if info['count'] > 0:
            summary[g] = {
                'count': info['count'],
                'avg_trt': round(sum(info['trts']) / len(info['trts']), 2),
                'min_trt': min(info['trts']),
                'max_trt': max(info['trts']),
            }

    optimal = (results.solver.termination_condition == TerminationCondition.optimal)

    # Get bounds for gap calculation
    lower_bound = None
    upper_bound = None
    try:
        if hasattr(results.problem, '__len__') and len(results.problem) > 0:
            prob = results.problem[0]
            lower_bound = getattr(prob, 'lower_bound', None)
            upper_bound = getattr(prob, 'upper_bound', None)
    except:
        pass
    if lower_bound is None:
        lower_bound = value(model.obj) if optimal else None

    return model, {
        'solved': True,
        'optimal': optimal,
        'total_cost': round(total_cost, 2),
        'transport_cost': round(total_transport, 2),
        'facility_cost': round(total_fac, 2),
        'avg_trt': round(sum(p['trt'] for p in patient_results) / len(patient_results), 2),
        'all_on_time': all(p['on_time'] for p in patient_results),
        'facilities_open': facs_open,
        'group_summary': summary,
        'patients': patient_results,
        'num_plans': len(plans),
        'num_cap_constraints': len(cap_keys),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
    }


def evaluate_with_simulation(data, plan_data, result,
                             n_replications=100, output_csv=None):
    """Evaluate the deterministic plan under stochastic process variability.

    Calls simulation.stochastic_simulation.run() using the facility and
    transport assignments fixed by solve_master().  Manufacturing and QC
    durations are sampled stochastically; the facility assignments are kept
    exactly as in the deterministic plan.

    Parameters
    ----------
    data           : dict  raw data from parse_dat()
    plan_data      : dict  output of generate_plans()
    result         : dict  output of solve_master() — must have 'patients' key
    n_replications : int   number of simulation replications (default 100)
    output_csv     : str   optional path to write per-replication CSV

    Returns
    -------
    dict  simulation summary (or None if the deterministic solve failed)
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from simulation import stochastic_simulation

    if not result.get('solved') or not result.get('patients'):
        print('[simulation] Cannot run: no valid deterministic solution found.')
        return None

    # Augment plan_data with the transport-time maps and lab-setup days that
    # generate_plans() uses internally but does not expose in its return dict.
    sim_plan_data = dict(plan_data)
    sim_plan_data['TT1'] = data['TT1']   # {mode: outbound days}
    sim_plan_data['TT3'] = data['TT3']   # {mode: return days}
    sim_plan_data['TLS'] = data['TLS']   # lab-setup days at treatment centre

    plan_assignments = result['patients']   # list of per-patient assignment dicts

    print(f'\n[simulation] Running {n_replications} stochastic replications '
          f'on {len(plan_assignments)} patients...')

    summary = stochastic_simulation.run(
        plan_assignments,
        sim_plan_data,
        n_replications=n_replications,
        output_csv=output_csv,
        verbose=True,
    )
    return summary


def run_experiment(tau=0.0, data_file=None, urgency_config=None,
                   process_config=None, time_limit=300, random_seed=42,
                   solver_name='appsi_highs'):
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N50.dat')

    t0 = _time.time()
    data = parse_dat(data_file)
    plan_data = generate_plans(data, tau=tau, urgency_config=urgency_config,
                                process_config=process_config, random_seed=random_seed)
    gen_time = _time.time() - t0

    n_plans = len(plan_data['plans'])
    n_patients = len(plan_data['P'])

    # Check if any patient has no feasible plans
    infeasible_patients = [p for p in plan_data['P']
                           if not plan_data['plans_by_patient'][p]]
    if infeasible_patients:
        return {
            'solver': 'ColGen', 'solved': False,
            'reason': 'infeasible_patients',
            'infeasible_patients': infeasible_patients,
            'num_patients': n_patients,
            'num_plans': 0,
            'gen_time': round(gen_time, 2),
        }

    t0 = _time.time()
    model, res = solve_master(plan_data, time_limit=time_limit, solver_name=solver_name)
    solve_time = _time.time() - t0

    res['solver'] = 'ColGen'
    res['tau'] = tau
    res['num_patients'] = n_patients
    res['num_plans'] = n_plans
    res['gen_time'] = round(gen_time, 2)
    res['solve_time'] = round(solve_time, 2)
    res['build_time'] = round(gen_time, 2)
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='CAR-T Column Generation Solver with optional stochastic simulation'
    )
    parser.add_argument('--data', default=None,
                        help='Path to .dat instance file (default: Data_N50.dat)')
    parser.add_argument('--tau', type=float, default=0.0,
                        help='Deadline tightness factor 0-1 (default 0.0)')
    parser.add_argument('--time-limit', type=int, default=300,
                        help='MIP solver time limit in seconds (default 300)')
    # Simulation options
    parser.add_argument('--simulate', action='store_true',
                        help='After solving, run stochastic SimPy simulation')
    parser.add_argument('--replications', type=int, default=100,
                        help='Number of simulation replications (default 100)')
    parser.add_argument('--sim-output', default=None,
                        help='CSV file path to save per-replication simulation results')
    args = parser.parse_args()

    data_file = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Data_N50.dat'
    )

    if args.simulate:
        # When running simulation we need intermediate objects (data, plan_data)
        # that run_experiment() does not expose, so we call the steps separately.
        raw_data   = parse_dat(data_file)
        t0         = _time.time()
        pdata      = generate_plans(raw_data, tau=args.tau)
        gen_time   = _time.time() - t0

        infeasible = [p for p in pdata['P'] if not pdata['plans_by_patient'][p]]
        if infeasible:
            print('ColGen | FAILED: infeasible_patients %s' % infeasible)
        else:
            t0         = _time.time()
            _, res     = solve_master(pdata, time_limit=args.time_limit)
            solve_time = _time.time() - t0

            if res.get('solved'):
                print('ColGen | Cost: $%s | TRT: %.1f | Plans: %d | '
                      'Gen: %.1fs | Solve: %.1fs | %s' % (
                    '{:,.0f}'.format(res['total_cost']), res['avg_trt'],
                    len(pdata['plans']), gen_time, solve_time,
                    'OPTIMAL' if res.get('optimal') else 'FEASIBLE'))
                # Run stochastic simulation on the deterministic plan
                evaluate_with_simulation(
                    raw_data, pdata, res,
                    n_replications=args.replications,
                    output_csv=args.sim_output,
                )
            else:
                print('ColGen | FAILED: %s' % res.get('reason', res.get('termination', '?')))
    else:
        # Original behaviour — unchanged
        r = run_experiment(tau=args.tau, data_file=data_file, time_limit=args.time_limit)
        if r['solved']:
            print('ColGen | Cost: $%s | TRT: %.1f | Plans: %d | Gen: %.1fs | Solve: %.1fs | %s' % (
                '{:,.0f}'.format(r['total_cost']), r['avg_trt'], r['num_plans'],
                r['gen_time'], r['solve_time'],
                'OPTIMAL' if r.get('optimal') else 'FEASIBLE'))
        else:
            print('ColGen | FAILED: %s' % r.get('reason', r.get('termination', '?')))
