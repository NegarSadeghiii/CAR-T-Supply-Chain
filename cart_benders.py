"""
CAR-T Supply Chain — Benders Decomposition (Hard Deadline)
==========================================================
Master: facility selection + patient-facility-transport assignment
Subproblem: timing/capacity feasibility check per facility

Decomposition:
  Master (IP): choose facilities E_m, assign patient routes r_{pmjk}
  Sub (per facility): given assigned patients, check if capacity allows
    scheduling all within their deadlines. If not, generate feasibility cut.
"""

import re, random, os, time as _time, math
from pyomo.environ import (ConcreteModel, Var, Binary, NonNegativeIntegers,
                            Objective, minimize, Constraint, SolverFactory,
                            SolverStatus, TerminationCondition, value, Set)


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
DEFAULT_PROCESS = {'tls': 1, 'tmfe': 7, 'tqc': 7, 'max_facilities': 2}


def run_experiment(tau=0.0, data_file=None, urgency_config=None,
                   process_config=None, time_limit=300, random_seed=42,
                   solver_name='appsi_highs', max_iterations=50):
    """Benders decomposition for hard-deadline CAR-T model."""
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N50.dat')
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS

    data = parse_dat(data_file)
    P = data['p']
    C = data['c']
    H = data['h']
    J = data['j']
    M = data['m']
    N = len(P)

    TLS  = int(process_config.get('tls', 1))
    TMFE = int(process_config.get('tmfe', 7))
    TQC  = int(process_config.get('tqc', 7))
    MAX_FAC = int(process_config.get('max_facilities', 2))
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

    # Deadlines
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

    # Time horizon
    max_arr = max(patient_arrival.values())
    # Must match MIP time horizon: hardcoded 130 in build_model
    T_max = 130

    # Precompute feasible routes per patient
    feasible_routes = {}  # (p, m, j_out, j_ret) -> {trt, cost, min_mfg_start, max_wait}
    for p in P:
        arr = patient_arrival[p]
        c = patient_center[p]
        h = c_to_h[c]
        dl = deadlines[p]
        for mi in M:
            for j_out in J:
                for j_ret in J:
                    trt = TLS + TT1[j_out] + TMFE + TQC + TT3[j_ret]
                    if trt > dl:
                        continue
                    cost_out = data['U1'].get((c, mi, j_out), 0)
                    cost_ret = data['U3'].get((mi, h, j_ret), 0)
                    # MIP MSB1 forces no waiting at center (max_wait=0)
                    max_wait = 0
                    mfg_start_min = arr + TLS + TT1[j_out]
                    feasible_routes[(p, mi, j_out, j_ret)] = {
                        'trt': trt, 'cost': cost_out + cost_ret,
                        'mfg_start_min': mfg_start_min,
                        'max_wait': max_wait,
                    }

    # Check if any patient has no feasible routes
    for p in P:
        has_route = any(k[0] == p for k in feasible_routes)
        if not has_route:
            return {
                'solver': 'Benders', 'solved': False,
                'reason': 'patient %s has no feasible route (deadline %.1f too tight)' % (p, deadlines[p]),
                'num_patients': N, 'tau': tau,
            }

    t_start = _time.time()

    # ================================================================
    # MASTER PROBLEM: facility selection + route assignment
    # ================================================================
    master = ConcreteModel()

    # Facility open
    master.z = Var(M, within=Binary)
    master.fac_limit = Constraint(expr=sum(master.z[m] for m in M) <= MAX_FAC)

    # Route assignment: r[p, m, j_out, j_ret] in {0,1}
    route_keys = list(feasible_routes.keys())
    master.r = Var(route_keys, within=Binary)

    # Each patient exactly one route
    def assign_rule(master, p_idx):
        p = P[p_idx]
        return sum(master.r[k] for k in route_keys if k[0] == p) == 1
    master.assign = Constraint(range(N), rule=assign_rule)

    # Route only if facility is open
    for k in route_keys:
        p, mi, jo, jr = k
        master.add_component('link_%s' % str(k).replace(' ', ''),
                             Constraint(expr=master.r[k] <= master.z[mi]))

    # Objective: transport + facility cost
    fac_cost_total = {m: (CIM[m] + CVM[m]) * T_max for m in M}
    mat_cost = (10476 + 9312) * N

    def obj_rule(master):
        transport = sum(feasible_routes[k]['cost'] * master.r[k] for k in route_keys)
        facility = sum(fac_cost_total[m] * master.z[m] for m in M)
        return transport + facility + mat_cost
    master.obj = Objective(rule=obj_rule)

    # Benders cuts will be added dynamically
    master.cut_set = Set(dimen=1)
    master.cuts = Constraint(master.cut_set)
    cut_count = 0

    solver = SolverFactory(solver_name)
    if solver_name == 'appsi_highs':
        solver.options['time_limit'] = max(30, time_limit // max_iterations)

    # ================================================================
    # BENDERS ITERATION
    # ================================================================
    best_obj = float('inf')
    best_solution = None
    lower_bound = 0
    iterations = 0

    for iteration in range(max_iterations):
        if _time.time() - t_start > time_limit:
            break
        iterations += 1

        # Solve master
        results = solver.solve(master, tee=False)
        ok = (results.solver.status == SolverStatus.ok and
              results.solver.termination_condition in
              (TerminationCondition.optimal, TerminationCondition.feasible))

        if not ok:
            return {
                'solver': 'Benders', 'solved': False,
                'termination': str(results.solver.termination_condition),
                'iterations': iterations, 'num_patients': N, 'tau': tau,
                'solve_time': round(_time.time() - t_start, 2),
            }

        master_obj = value(master.obj)
        lower_bound = master_obj

        # Extract master solution
        open_facs = [m for m in M if value(master.z[m]) > 0.5]
        selected_routes = {p: None for p in P}
        for k in route_keys:
            if value(master.r[k]) > 0.5:
                p, mi, jo, jr = k
                info = feasible_routes[k]
                selected_routes[p] = {
                    'facility': mi, 'j_out': jo, 'j_ret': jr,
                    'trt': info['trt'], 'cost': info['cost'],
                    'mfg_start_min': info['mfg_start_min'],
                    'max_wait': info['max_wait'],
                }

        # ============================================================
        # SUBPROBLEM: check capacity feasibility per facility
        # ============================================================
        # For each open facility, schedule the assigned patients
        # Each patient can start manufacturing in [mfg_start_min, mfg_start_min + max_wait]
        # At most FCAP[m] patients in manufacturing at any time

        all_feasible = True
        facility_schedules = {}  # mi -> {p: mfg_start}

        for mi in open_facs:
            assigned = [(p, selected_routes[p]) for p in P
                        if selected_routes[p] and selected_routes[p]['facility'] == mi]
            if not assigned:
                continue

            cap = FCAP[mi]

            # Use CP-SAT for optimal scheduling (fast for small problems)
            from ortools.sat.python import cp_model as cpm
            sub = cpm.CpModel()

            starts = {}
            intervals = []
            for p, route in assigned:
                earliest = route['mfg_start_min']
                latest = earliest + route['max_wait']
                s = sub.new_int_var(earliest, latest, 'S_%s' % p)
                starts[p] = s
                iv = sub.new_fixed_size_interval_var(s, TMFE, 'I_%s' % p)
                intervals.append(iv)

            # Cumulative capacity
            sub.add_cumulative(intervals, [1] * len(intervals), cap)

            sub_solver = cpm.CpSolver()
            sub_solver.parameters.max_time_in_seconds = 5
            status = sub_solver.solve(sub)

            if status in (cpm.OPTIMAL, cpm.FEASIBLE):
                facility_schedules[mi] = {p: sub_solver.value(starts[p])
                                          for p, _ in assigned}
            else:
                all_feasible = False
                # Targeted cut: find the conflicting time window
                # Identify patients whose earliest windows overlap most
                patient_ids = [p for p, _ in assigned]

                # Find the most congested time window
                time_counts = {}
                for p, route in assigned:
                    for t in range(route['mfg_start_min'],
                                   route['mfg_start_min'] + TMFE):
                        time_counts[t] = time_counts.get(t, 0) + 1

                if time_counts:
                    peak_time = max(time_counts, key=time_counts.get)
                    peak_count = time_counts[peak_time]
                    # Find patients overlapping at peak time
                    conflicting = []
                    for p, route in assigned:
                        if (route['mfg_start_min'] <= peak_time <
                                route['mfg_start_min'] + TMFE + route['max_wait']):
                            conflicting.append(p)

                    # Cut: at most FCAP of these patients can use this facility
                    if len(conflicting) > cap:
                        cut_expr = sum(master.r[k] for k in route_keys
                                       if k[0] in conflicting and k[1] == mi)
                        cut_count += 1
                        master.cut_set.add(cut_count)
                        master.cuts[cut_count] = cut_expr <= cap
                else:
                    # Fallback: generic cut
                    cut_expr = sum(master.r[k] for k in route_keys
                                  if k[0] in patient_ids and k[1] == mi)
                    cut_count += 1
                    master.cut_set.add(cut_count)
                    master.cuts[cut_count] = cut_expr <= len(patient_ids) - 1

        if all_feasible:
            # Found feasible solution — compute wait from schedule
            for mi, sched in facility_schedules.items():
                for p, mfg_start in sched.items():
                    if selected_routes[p]:
                        route = selected_routes[p]
                        route['mfg_start'] = mfg_start
                        route['wait'] = mfg_start - route['mfg_start_min']
                        route['trt'] = route['trt'] + route['wait']  # add wait to TRT

            if master_obj < best_obj:
                best_obj = master_obj
                best_solution = {
                    'routes': dict(selected_routes),
                    'facilities': open_facs,
                    'schedules': dict(facility_schedules),
                    'obj': master_obj,
                }
            break  # Optimal when master = sub (no cuts added)

    total_time = _time.time() - t_start

    if best_solution is None:
        return {
            'solver': 'Benders', 'solved': False,
            'reason': 'no feasible solution found in %d iterations' % iterations,
            'iterations': iterations, 'num_patients': N, 'tau': tau,
            'solve_time': round(total_time, 2),
        }

    # Extract results
    patient_results = []
    group_summary = {g: {'count': 0, 'trts': []} for g in ('high', 'medium', 'low')}
    for p in P:
        route = best_solution['routes'][p]
        g = group_map[p]
        patient_results.append({
            'id': p, 'group': g,
            'facility': route['facility'],
            'j_out': route['j_out'], 'j_ret': route['j_ret'],
            'wait': route.get('wait', 0),
            'trt': route['trt'],
            'mfg_start': route.get('mfg_start'),
            'deadline': deadlines[p],
        })
        if g in group_summary:
            group_summary[g]['count'] += 1
            group_summary[g]['trts'].append(route['trt'])

    summary = {}
    for g, info in group_summary.items():
        if info['count'] > 0:
            summary[g] = {
                'count': info['count'],
                'avg_trt': round(sum(info['trts']) / len(info['trts']), 2),
                'min_trt': min(info['trts']),
                'max_trt': max(info['trts']),
            }

    gap = abs(best_obj - lower_bound) / max(abs(best_obj), 1e-10) * 100

    return {
        'solver': 'Benders', 'solved': True,
        'optimal': gap < 0.01,
        'tau': tau, 'num_patients': N,
        'total_cost': round(best_obj, 2),
        'avg_trt': round(sum(r['trt'] for r in patient_results) / N, 2),
        'all_on_time': True,  # hard deadline — if solved, all on time
        'facilities_open': best_solution['facilities'],
        'group_summary': summary,
        'iterations': iterations,
        'num_cuts': cut_count,
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(best_obj, 2),
        'gap_pct': round(gap, 4),
        'solve_time': round(total_time, 2),
        'build_time': 0,
        'num_routes': len(route_keys),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--time-limit', type=int, default=300)
    args = parser.parse_args()

    data_file = args.data or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N50.dat')
    r = run_experiment(tau=args.tau, data_file=data_file, time_limit=args.time_limit)
    if r['solved']:
        print('Benders | Cost: $%s | TRT: %.1f | Iter: %d | Cuts: %d | Solve: %.1fs | %s' % (
            '{:,.0f}'.format(r['total_cost']), r['avg_trt'], r['iterations'],
            r['num_cuts'], r['solve_time'],
            'OPTIMAL' if r.get('optimal') else 'gap=%.2f%%' % r.get('gap_pct', 0)))
    else:
        print('Benders | FAILED: %s' % r.get('reason', r.get('termination', '?')))
