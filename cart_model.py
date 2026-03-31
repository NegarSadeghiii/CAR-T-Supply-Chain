"""
CAR-T Supply Chain Model — Hard Deadlines via Big-M
====================================================
Urgency-based patient deadlines with tightness factor τ.
Big-M lateness penalty ensures the model is always feasible:
  - If LATE[p]=0 for all p → all deadlines met (effectively hard)
  - If any LATE[p]>0 → that patient would violate the hard deadline

Objective: min( total_cost + M × Σ LATE[p] )
    where M is large enough that deadlines are prioritized over cost.

τ ∈ [0, 1]: tightness factor
    effective_deadline[p] = base_due[group] + tolerance[group] × (1 - τ)
"""

from pyomo.environ import *
import random, json, sys, os, time

# ── Urgency configuration ──────────────────────────────────────────

DEFAULT_URGENCY = {
    'high':   {'fraction': 0.20, 'base_due': 16, 'tolerance': 1},
    'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
    'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
}

BIG_M = 1_000_000  # penalty weight — much larger than any realistic cost


def build_model(time_horizon=130):
    model = AbstractModel()

    # Sets
    model.c  = Set()
    model.h  = Set()
    model.j  = Set()
    model.m  = Set()
    model.p  = Set()
    model.t  = RangeSet(time_horizon)
    model.tt = Set(initialize=model.t)

    # Parameters
    model.CIM        = Param(model.m)
    model.FCAP       = Param(model.m)
    model.TT1        = Param(model.j)
    model.TT3        = Param(model.j)
    model.U1         = Param(model.c, model.m, model.j)
    model.U3         = Param(model.m, model.h, model.j)
    model.INC        = Param(model.p, model.c, model.t, initialize=0)
    model.CVM        = Param(model.m,
                             default={'m1': 20920, 'm2': 156900, 'm3': 52300,
                                      'm4': 20920, 'm5': 156900, 'm6': 52300})
    model.FMAX       = Param()
    model.FMIN       = Param()
    model.TAD        = Param(within=NonNegativeReals)
    model.TLS        = Param(within=NonNegativeReals)
    model.TMFE       = Param(default=7)
    model.TQC        = Param(default=7)
    model.C_material = Param(default=10476)
    model.CQC        = Param(default=9312)
    model.ND         = Param(default=18)

    # Deadline parameter (set per patient after instantiation)
    model.DEADLINE = Param(model.p, initialize=130, mutable=True, within=NonNegativeReals)

    # Decision variables
    model.E1   = Var(model.m,                                         within=Binary)
    model.X1   = Var(model.c, model.m,                                within=Binary)
    model.X2   = Var(model.m, model.h,                                within=Binary)
    model.Y1   = Var(model.p, model.c, model.m, model.j, model.t,    within=Binary)
    model.Y2   = Var(model.p, model.m, model.h, model.j, model.t,    within=Binary)
    model.INH  = Var(model.p, model.h, model.t,                       within=NonNegativeIntegers)
    model.CTM  = Var(model.p,                                          within=NonNegativeReals)
    model.FTD  = Var(model.p, model.m, model.h, model.j, model.t,    within=NonNegativeReals)
    model.TTC  = Var(model.p,                                          within=NonNegativeReals)
    model.LSA  = Var(model.p, model.c, model.m, model.j, model.t,    within=NonNegativeReals)
    model.LSR  = Var(model.p, model.c, model.m, model.j, model.t,    within=NonNegativeReals)
    model.MSO  = Var(model.p, model.m, model.h, model.j, model.t,    within=NonNegativeReals)
    model.OUTC = Var(model.p, model.c, model.t,                        within=NonNegativeReals)
    model.OUTM = Var(model.p, model.m, model.t,                        within=NonNegativeReals)
    model.INM  = Var(model.p, model.m, model.t,                        within=NonNegativeReals)
    model.DURV = Var(model.p, model.m, model.t,                        within=NonNegativeReals)
    model.RATIO= Var(model.m, model.t,                                 within=NonNegativeReals)
    model.CAP  = Var(model.m, model.t)
    model.TRT  = Var(model.p)
    model.ATRT = Var()
    model.STT  = Var(model.p)
    model.CTT  = Var(model.p)
    model.LATE = Var(model.p, within=NonNegativeReals)  # lateness (0 = deadline met)

    # ── Objective: cost + Big-M × total lateness ──
    def obj_rule(model):
        return (sum(model.CTM[p] for p in model.p)
                + sum(model.TTC[p] for p in model.p)
                + (model.C_material + model.CQC) * len(model.p)
                + BIG_M * sum(model.LATE[p] for p in model.p))
    model.obj = Objective(rule=obj_rule)

    # ── Supply chain constraints (unchanged from original) ──
    def C1_rule(model, p):
        return model.CTM[p] == sum((model.E1[m] * (model.CIM[m] + model.CVM[m]))
                                   * len(model.t) / len(model.p) for m in model.m)
    model.C1 = Constraint(model.p, rule=C1_rule)

    def C2_rule(model, p):
        return model.TTC[p] == (
            sum(model.Y1[p, c, m, j, t] * model.U1[c, m, j]
                for c in model.c for m in model.m for j in model.j for t in model.t)
            + sum(model.Y2[p, m, h, j, t] * model.U3[m, h, j]
                  for m in model.m for h in model.h for j in model.j for t in model.t))
    model.C2 = Constraint(model.p, rule=C2_rule)

    def RATIOEQ_rule(model, m, t):
        return model.RATIO[m, t] == sum(model.DURV[p, m, t] / model.FCAP[m] for p in model.p)
    model.RATIOEQ = Constraint(model.m, model.t, rule=RATIOEQ_rule)

    def MSBnew_rule(model, p, m, t):
        return model.DURV[p, m, t] == (
            sum(model.INM[p, m, tt - 1] - model.OUTM[p, m, tt]
                for tt in model.tt if tt <= t and tt > 1)
            + model.OUTM[p, m, t])
    model.MSBnew = Constraint(model.p, model.m, model.t, rule=MSBnew_rule)

    def MSB1_rule(model, p, c, t, tt):
        if tt == t + model.TLS: return model.INC[p, c, t] == model.OUTC[p, c, tt]
        return Constraint.Skip
    model.MSB1 = Constraint(model.p, model.c, model.t, model.tt, rule=MSB1_rule)

    def MSB3_rule(model, p, c, m, j, t, tt):
        if tt == t + model.TT1[j]: return model.LSR[p, c, m, j, t] == model.LSA[p, c, m, j, tt]
        return Constraint.Skip
    model.MSB3 = Constraint(model.p, model.c, model.m, model.j, model.t, model.tt, rule=MSB3_rule)

    def MSB7_rule(model, p, c, t):
        return model.OUTC[p, c, t] == sum(model.LSR[p, c, m, j, t] for m in model.m for j in model.j)
    model.MSB7 = Constraint(model.p, model.c, model.t, rule=MSB7_rule)

    def MSB5_rule(model, p, m, t):
        return model.INM[p, m, t] == sum(model.LSA[p, c, m, j, t] for c in model.c for j in model.j)
    model.MSB5 = Constraint(model.p, model.m, model.t, rule=MSB5_rule)

    def MSB2_rule(model, p, m, t, tt):
        if tt == t + model.TMFE: return model.INM[p, m, t] == model.OUTM[p, m, tt]
        return Constraint.Skip
    model.MSB2 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB2_rule)

    def MSB8_rule(model, p, m, t, tt):
        if tt == t + model.TQC:
            return model.OUTM[p, m, t] == sum(model.MSO[p, m, h, j, tt] for h in model.h for j in model.j)
        return Constraint.Skip
    model.MSB8 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB8_rule)

    def MSB4_rule(model, p, m, h, j, t, tt):
        if tt == t + model.TT3[j]: return model.MSO[p, m, h, j, t] == model.FTD[p, m, h, j, tt]
        return Constraint.Skip
    model.MSB4 = Constraint(model.p, model.m, model.h, model.j, model.t, model.tt, rule=MSB4_rule)

    def MSB6_rule(model, p, h, t):
        return model.INH[p, h, t] == sum(model.FTD[p, m, h, j, t] for m in model.m for j in model.j)
    model.MSB6 = Constraint(model.p, model.h, model.t, rule=MSB6_rule)

    def CAP1_rule(model, m, t):
        return model.CAP[m, t] == model.FCAP[m] - sum(
            model.INM[p, m, tt] for p in model.p for tt in model.tt
            if tt < t and tt >= t - model.TMFE)
    model.CAP1 = Constraint(model.m, model.t, rule=CAP1_rule)

    def CAPCON1_rule(model, m, t):
        return (sum(model.INM[p, m, t] for p in model.p)
                - sum(model.OUTM[p, m, t] for p in model.p) <= model.CAP[m, t])
    model.CAPCON1 = Constraint(model.m, model.t, rule=CAPCON1_rule)

    def CON1_rule(model): return sum(model.E1[m] for m in model.m) <= 2
    model.CON1 = Constraint(rule=CON1_rule)

    def CON2_rule(model, c, m):      return model.X1[c, m] <= model.E1[m]
    model.CON2 = Constraint(model.c, model.m, rule=CON2_rule)

    def CON3_rule(model, m, h):      return model.X2[m, h] <= model.E1[m]
    model.CON3 = Constraint(model.m, model.h, rule=CON3_rule)

    def CON4_rule(model, p, c, m, j, t): return model.Y1[p, c, m, j, t] <= model.X1[c, m]
    model.CON4 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON4_rule)

    def CON5_rule(model, p, m, h, j, t): return model.Y2[p, m, h, j, t] <= model.X2[m, h]
    model.CON5 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON5_rule)

    def CON6_rule(model, p):
        return sum(model.Y1[p, c, m, j, t]
                   for c in model.c for m in model.m for j in model.j for t in model.t) == 1
    model.CON6 = Constraint(model.p, rule=CON6_rule)

    def CON7_rule(model, p):
        return sum(model.Y2[p, m, h, j, t]
                   for m in model.m for h in model.h for j in model.j for t in model.t) == 1
    model.CON7 = Constraint(model.p, rule=CON7_rule)

    def DEM_rule(model):
        return sum(model.INH[p, h, t] for p in model.p for h in model.h for t in model.t) <= len(model.p)
    model.DEM = Constraint(rule=DEM_rule)

    def CON8_rule(model, p, c, m, j, t):  return model.LSR[p, c, m, j, t] >= model.Y1[p, c, m, j, t] * model.FMIN
    model.CON8  = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON8_rule)

    def CON9_rule(model, p, c, m, j, t):  return model.LSR[p, c, m, j, t] <= model.Y1[p, c, m, j, t] * model.FMAX
    model.CON9  = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON9_rule)

    def CON10_rule(model, p, m, h, j, t): return model.MSO[p, m, h, j, t] >= model.Y2[p, m, h, j, t] * model.FMIN
    model.CON10 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON10_rule)

    def CON11_rule(model, p, m, h, j, t): return model.MSO[p, m, h, j, t] <= model.Y2[p, m, h, j, t] * model.FMAX
    model.CON11 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON11_rule)

    def CON12_rule(model, p):
        return (sum(model.Y2[p, m, 'h1', j, t] for m in model.m for j in model.j for t in model.t)
                == sum(model.INC[p, 'c1', t] for t in model.t))
    model.CON12 = Constraint(model.p, rule=CON12_rule)

    def CON13_rule(model, p):
        return (sum(model.Y2[p, m, 'h2', j, t] for m in model.m for j in model.j for t in model.t)
                == sum(model.INC[p, 'c2', t] for t in model.t))
    model.CON13 = Constraint(model.p, rule=CON13_rule)

    def CON14_rule(model, p):
        return (sum(model.Y2[p, m, 'h3', j, t] for m in model.m for j in model.j for t in model.t)
                == sum(model.INC[p, 'c3', t] for t in model.t))
    model.CON14 = Constraint(model.p, rule=CON14_rule)

    def CON15_rule(model, p):
        return (sum(model.Y2[p, m, 'h4', j, t] for m in model.m for j in model.j for t in model.t)
                == sum(model.INC[p, 'c4', t] for t in model.t))
    model.CON15 = Constraint(model.p, rule=CON15_rule)

    def START_rule(model, p):
        return model.STT[p] == sum(model.INC[p, c, t] * t for c in model.c for t in model.t)
    model.START = Constraint(model.p, rule=START_rule)

    def END_rule(model, p):
        return model.CTT[p] == sum(model.INH[p, h, t] * t for h in model.h for t in model.t)
    model.END = Constraint(model.p, rule=END_rule)

    def TSEQ_rule(model, p):  return model.STT[p] <= model.CTT[p]
    model.TSEQ  = Constraint(model.p, rule=TSEQ_rule)

    def TIME_rule(model, p):  return model.TRT[p] == model.CTT[p] - model.STT[p]
    model.TIME  = Constraint(model.p, rule=TIME_rule)

    def ATIME_rule(model):
        return model.ATRT == sum(model.TRT[p] for p in model.p) / len(model.p)
    model.ATIME = Constraint(rule=ATIME_rule)

    # ── Lateness constraint: LATE[p] >= TRT[p] - DEADLINE[p] ──
    def LATE_rule(model, p):
        return model.LATE[p] >= model.TRT[p] - model.DEADLINE[p]
    model.LATECON = Constraint(model.p, rule=LATE_rule)

    return model


def assign_deadlines(instance, tau=0.0, urgency_config=None, random_seed=42):
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY

    patients = sorted(instance.p, key=str)
    n = len(patients)

    num_high = int(round(n * urgency_config['high']['fraction']))
    num_med  = int(round(n * urgency_config['medium']['fraction']))

    random.seed(random_seed)
    shuffled = patients[:]
    random.shuffle(shuffled)

    group_map = {}
    for idx, p in enumerate(shuffled):
        if idx < num_high:
            g = 'high'
        elif idx < num_high + num_med:
            g = 'medium'
        else:
            g = 'low'

        cfg = urgency_config[g]
        effective = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)
        group_map[str(p)] = g
        instance.DEADLINE[p] = effective

    return group_map


def solve(instance, solver_name='appsi_highs', time_limit=300):
    solver = SolverFactory(solver_name)
    if solver_name == 'appsi_highs':
        solver.options['time_limit'] = time_limit
    elif solver_name == 'gurobi':
        solver.options['TimeLimit'] = time_limit

    results = solver.solve(instance, tee=False)
    ok = (results.solver.status == SolverStatus.ok and
          results.solver.termination_condition in
          (TerminationCondition.optimal, TerminationCondition.feasible))
    return results, ok


def get_results(instance, group_map, tau):
    patients = []
    group_summary = {g: {'count': 0, 'trts': [], 'lates': [], 'deadline': 0}
                     for g in ('high', 'medium', 'low')}

    for p in sorted(instance.p, key=str):
        pid = str(p)
        trt  = value(instance.TRT[p])
        dl   = value(instance.DEADLINE[p])
        late = value(instance.LATE[p])
        grp  = group_map.get(pid, 'unknown')
        stt  = value(instance.STT[p])
        ctt  = value(instance.CTT[p])

        patients.append({
            'id': pid, 'group': grp,
            'arrival_day': round(stt, 1),
            'completion_day': round(ctt, 1),
            'turnaround': round(trt, 2),
            'deadline': round(dl, 2),
            'lateness': round(late, 2),
            'on_time': late < 0.01,
        })

        if grp in group_summary:
            group_summary[grp]['count'] += 1
            group_summary[grp]['trts'].append(trt)
            group_summary[grp]['lates'].append(late)
            group_summary[grp]['deadline'] = round(dl, 2)

    # Cost WITHOUT the Big-M penalty (pure supply chain cost)
    real_cost = value(
        sum(instance.CTM[p] for p in instance.p) +
        sum(instance.TTC[p] for p in instance.p) +
        (instance.C_material + instance.CQC) * len(instance.p)
    )
    total_lateness = sum(value(instance.LATE[p]) for p in instance.p)
    num_late = sum(1 for p in instance.p if value(instance.LATE[p]) > 0.01)

    summary = {}
    for g, info in group_summary.items():
        if info['count'] > 0:
            summary[g] = {
                'count': info['count'],
                'deadline': info['deadline'],
                'avg_trt': round(sum(info['trts']) / len(info['trts']), 2),
                'min_trt': round(min(info['trts']), 2),
                'max_trt': round(max(info['trts']), 2),
                'avg_lateness': round(sum(info['lates']) / len(info['lates']), 2),
                'max_lateness': round(max(info['lates']), 2),
                'num_late': sum(1 for l in info['lates'] if l > 0.01),
            }

    return {
        'tau': tau,
        'num_patients': len(patients),
        'real_cost': round(real_cost, 2),
        'avg_trt': round(value(instance.ATRT), 2),
        'total_lateness': round(total_lateness, 2),
        'num_late': num_late,
        'all_on_time': num_late == 0,
        'facilities_open': [str(m) for m in instance.m if value(instance.E1[m]) > 0.5],
        'group_summary': summary,
        'patients': patients,
    }


def run_experiment(tau=0.0, data_file=None, urgency_config=None,
                   solver_name='appsi_highs', random_seed=42,
                   time_horizon=None, time_limit=300):
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N15.dat')

    if time_horizon is None:
        # Auto-detect from data file
        import re
        with open(data_file) as f:
            txt = f.read()
        inc_matches = re.findall(r'p\d+\s+c\d+\s+(\d+)\s+1', txt)
        max_day = max(int(d) for d in inc_matches) if inc_matches else 90
        time_horizon = max_day + 30

    t0 = time.time()
    model = build_model(time_horizon=time_horizon)
    instance = model.create_instance(data_file)
    build_time = time.time() - t0

    group_map = assign_deadlines(instance, tau=tau, urgency_config=urgency_config,
                                  random_seed=random_seed)

    t0 = time.time()
    results, ok = solve(instance, solver_name=solver_name, time_limit=time_limit)
    solve_time = time.time() - t0

    if not ok:
        return {
            'tau': tau, 'solved': False,
            'termination': str(results.solver.termination_condition),
            'build_time': round(build_time, 1),
            'solve_time': round(solve_time, 1),
        }

    res = get_results(instance, group_map, tau)
    res['solved'] = True
    res['build_time'] = round(build_time, 1)
    res['solve_time'] = round(solve_time, 1)
    return res


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CAR-T Hard Deadline Model (Big-M)')
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--data', default=None, help='Data file path')
    parser.add_argument('--solver', default='appsi_highs')
    parser.add_argument('--sweep', action='store_true', help='Sweep tau 0→1')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        print(f"{'tau':>5}  {'OnTime':>6}  {'Late#':>5}  {'TotLate':>8}  "
              f"{'Cost':>12}  {'AvgTRT':>7}  {'H_DL':>5}  {'M_DL':>5}  {'L_DL':>5}  {'Time':>6}")
        print('-' * 80)
        for tau_val in [i / 10.0 for i in range(11)]:
            r = run_experiment(tau=tau_val, data_file=args.data, solver_name=args.solver)
            cfg = DEFAULT_URGENCY
            h_dl = cfg['high']['base_due'] + cfg['high']['tolerance'] * (1 - tau_val)
            m_dl = cfg['medium']['base_due'] + cfg['medium']['tolerance'] * (1 - tau_val)
            l_dl = cfg['low']['base_due'] + cfg['low']['tolerance'] * (1 - tau_val)
            if r['solved']:
                ot = 'YES' if r['all_on_time'] else 'NO'
                print(f"{tau_val:>5.1f}  {ot:>6}  {r['num_late']:>5}  {r['total_lateness']:>8.1f}  "
                      f"{r['real_cost']:>12,.0f}  {r['avg_trt']:>7.2f}  {h_dl:>5.1f}  {m_dl:>5.1f}  {l_dl:>5.1f}  "
                      f"{r['solve_time']:>5.1f}s")
            else:
                print(f"{tau_val:>5.1f}  {'FAIL':>6}  {'—':>5}  {'—':>8}  "
                      f"{'—':>12}  {'—':>7}  {h_dl:>5.1f}  {m_dl:>5.1f}  {l_dl:>5.1f}  "
                      f"{r['solve_time']:>5.1f}s")
    else:
        r = run_experiment(tau=args.tau, data_file=args.data, solver_name=args.solver)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            if r['solved']:
                status = 'ALL ON TIME' if r['all_on_time'] else f'{r["num_late"]} LATE'
                print(f"tau={args.tau} | {status} | Cost: {r['real_cost']:,.0f} | Avg TRT: {r['avg_trt']:.2f}")
                print(f"Build: {r['build_time']}s | Solve: {r['solve_time']}s")
                print(f"Facilities: {r['facilities_open']}")
                for g in ('high', 'medium', 'low'):
                    if g in r['group_summary']:
                        s = r['group_summary'][g]
                        late_str = f", LATE: {s['num_late']}/{s['count']}" if s['num_late'] > 0 else ""
                        print(f"  {g:>8}: n={s['count']}, DL={s['deadline']}, "
                              f"TRT=[{s['min_trt']},{s['avg_trt']},{s['max_trt']}]{late_str}")
            else:
                print(f"tau={args.tau} | SOLVER FAILED: {r['termination']}")
