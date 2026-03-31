"""
Hard-Deadline CAR-T Supply Chain Model with Tightness Factor (τ)
================================================================
Replaces the soft due-date penalty approach with hard constraints:
    TRT[p] <= base_due[group] + tolerance[group] * (1 - τ)

τ = 0  → most relaxed (full tolerance)
τ = 1  → tightest (zero tolerance, bare due dates only)

No lateness penalty in the objective — pure cost minimisation.
"""

from pyomo.environ import *
from pyomo.common.timing import TicTocTimer
import random, json, sys, os

# ── Default urgency configuration ──────────────────────────────────

DEFAULT_URGENCY = {
    'high':   {'fraction': 0.20, 'base_due': 16, 'tolerance': 1},
    'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
    'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
}


def build_model(num_patients=None):
    """
    Construct the Pyomo AbstractModel.

    Compared to the soft-due-date version:
    - Removed: PEN, LATE variables and penalty term in objective
    - Added:   DEADLINE[p] mutable param, HARD_DUE constraint (TRT[p] <= DEADLINE[p])
    """
    model = AbstractModel()

    # Sets
    model.c  = Set()
    model.h  = Set()
    model.j  = Set()
    model.m  = Set()
    model.p  = Set()
    model.t  = RangeSet(130)
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

    # Hard deadline parameter (set per patient after instantiation)
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

    # ── Objective: PURE COST (no penalty) ──
    def obj_rule(model):
        return (sum(model.CTM[p] for p in model.p)
                + sum(model.TTC[p] for p in model.p)
                + (model.C_material + model.CQC) * len(model.p))
    model.obj = Objective(rule=obj_rule)

    # ── Constraints (same as original) ──
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

    # ── NEW: Hard deadline constraint ──
    def HARD_DUE_rule(model, p):
        return model.TRT[p] <= model.DEADLINE[p]
    model.HARD_DUE = Constraint(model.p, rule=HARD_DUE_rule)

    return model


def assign_deadlines(instance, tau=0.0, urgency_config=None, random_seed=42):
    """
    Assign hard deadlines based on urgency group and tightness factor τ.

    effective_deadline = base_due + tolerance * (1 - τ)

    Returns group_map for reporting.
    """
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY

    patients = sorted(instance.p, key=str)
    n = len(patients)

    fractions = {g: urgency_config[g]['fraction'] for g in urgency_config}
    total_frac = sum(fractions.values())
    num_high = int(round(n * fractions['high'] / total_frac))
    num_med  = int(round(n * fractions['medium'] / total_frac))

    random.seed(random_seed)
    shuffled = patients[:]
    random.shuffle(shuffled)

    group_map = {}
    deadlines = {}

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
        deadlines[str(p)] = effective
        instance.DEADLINE[p] = effective

    return group_map, deadlines


def solve(instance, solver_name='appsi_highs', time_limit=300):
    """Solve and return (results, status_ok)."""
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


def get_results(instance, group_map):
    """Extract per-patient and summary results."""
    patients = []
    group_summary = {g: {'count': 0, 'trts': [], 'deadlines': []}
                     for g in ('high', 'medium', 'low')}

    for p in sorted(instance.p, key=str):
        pid = str(p)
        trt = value(instance.TRT[p])
        dl  = value(instance.DEADLINE[p])
        grp = group_map.get(pid, 'unknown')
        stt = value(instance.STT[p])
        ctt = value(instance.CTT[p])

        patients.append({
            'id': pid, 'group': grp,
            'arrival_day': stt, 'completion_day': ctt,
            'turnaround': round(trt, 2), 'deadline': round(dl, 2),
            'slack': round(dl - trt, 2)
        })

        if grp in group_summary:
            group_summary[grp]['count'] += 1
            group_summary[grp]['trts'].append(trt)
            group_summary[grp]['deadlines'].append(dl)

    total_cost = value(
        sum(instance.CTM[p] for p in instance.p) +
        sum(instance.TTC[p] for p in instance.p) +
        (instance.C_material + instance.CQC) * len(instance.p)
    )

    summary = {}
    for g, info in group_summary.items():
        if info['count'] > 0:
            summary[g] = {
                'count': info['count'],
                'avg_trt': round(sum(info['trts']) / len(info['trts']), 2),
                'min_trt': round(min(info['trts']), 2),
                'max_trt': round(max(info['trts']), 2),
                'deadline': round(info['deadlines'][0], 2),
            }

    return {
        'total_cost': round(total_cost, 2),
        'avg_trt': round(value(instance.ATRT), 2),
        'num_patients': len(patients),
        'group_summary': summary,
        'patients': patients,
        'facilities_open': [str(m) for m in instance.m if value(instance.E1[m]) > 0.5],
    }


def run_experiment(tau=0.0, num_patients=None, urgency_config=None,
                   solver_name='appsi_highs', random_seed=42, data_file=None,
                   time_limit=600):
    """
    Full pipeline: build → assign deadlines → solve → return results dict.
    """
    if data_file is None:
        data_file = os.path.join(os.path.dirname(__file__), 'Data200_profileA.dat')

    model = build_model(num_patients=num_patients)
    instance = model.create_instance(data_file)

    # Optionally reduce patient count
    if num_patients is not None and num_patients < len(instance.p):
        all_patients = sorted(instance.p, key=str)
        random.seed(random_seed)
        keep = set(random.sample(all_patients, num_patients))
        remove = [p for p in all_patients if p not in keep]
        for p in remove:
            instance.p.remove(p)

    group_map, deadlines = assign_deadlines(
        instance, tau=tau, urgency_config=urgency_config, random_seed=random_seed
    )

    import time as _time
    t0 = _time.time()
    try:
        results, ok = solve(instance, solver_name=solver_name, time_limit=time_limit)
    except RuntimeError as e:
        solve_time = _time.time() - t0
        return {
            'feasible': False, 'tau': tau,
            'num_patients': len(list(instance.p)),
            'termination': 'infeasible',
            'solve_time': round(solve_time, 1),
        }
    solve_time = _time.time() - t0

    if not ok:
        return {
            'feasible': False,
            'tau': tau,
            'num_patients': len(list(instance.p)),
            'deadlines_by_group': {
                g: round(urgency_config[g]['base_due'] + urgency_config[g]['tolerance'] * (1 - tau), 2)
                if urgency_config else round(DEFAULT_URGENCY[g]['base_due'] + DEFAULT_URGENCY[g]['tolerance'] * (1 - tau), 2)
                for g in ('high', 'medium', 'low')
            },
            'termination': str(results.solver.termination_condition),
            'solve_time': round(solve_time, 1),
        }

    res = get_results(instance, group_map)
    res['feasible'] = True
    res['tau'] = tau
    res['solve_time'] = round(solve_time, 1)
    return res


# ── CLI entry point ──────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hard-deadline CAR-T model')
    parser.add_argument('--tau', type=float, default=0.0, help='Tightness factor (0=relaxed, 1=tight)')
    parser.add_argument('--patients', type=int, default=None, help='Number of patients (default: all 50)')
    parser.add_argument('--solver', default='appsi_highs', help='Solver name')
    parser.add_argument('--sweep', action='store_true', help='Sweep tau from 0 to 1 in steps of 0.1')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    if args.sweep:
        print(f"{'tau':>5}  {'Feasible':>8}  {'Cost':>12}  {'Avg TRT':>8}  {'High DL':>8}  {'Med DL':>8}  {'Low DL':>8}")
        print('-' * 70)
        for tau_val in [i / 10.0 for i in range(11)]:
            r = run_experiment(tau=tau_val, num_patients=args.patients, solver_name=args.solver)
            cfg = DEFAULT_URGENCY
            h_dl = cfg['high']['base_due'] + cfg['high']['tolerance'] * (1 - tau_val)
            m_dl = cfg['medium']['base_due'] + cfg['medium']['tolerance'] * (1 - tau_val)
            l_dl = cfg['low']['base_due'] + cfg['low']['tolerance'] * (1 - tau_val)
            if r['feasible']:
                print(f"{tau_val:>5.1f}  {'YES':>8}  {r['total_cost']:>12.0f}  {r['avg_trt']:>8.2f}  {h_dl:>8.1f}  {m_dl:>8.1f}  {l_dl:>8.1f}")
            else:
                print(f"{tau_val:>5.1f}  {'NO':>8}  {'—':>12}  {'—':>8}  {h_dl:>8.1f}  {m_dl:>8.1f}  {l_dl:>8.1f}")
    else:
        r = run_experiment(tau=args.tau, num_patients=args.patients, solver_name=args.solver)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            if r['feasible']:
                print(f"τ = {args.tau} | Feasible: YES")
                print(f"Total cost: {r['total_cost']:,.0f}")
                print(f"Avg turnaround: {r['avg_trt']:.2f} days")
                print(f"Facilities open: {r['facilities_open']}")
                print(f"\nGroup summary:")
                for g in ('high', 'medium', 'low'):
                    if g in r['group_summary']:
                        s = r['group_summary'][g]
                        print(f"  {g:>8}: n={s['count']}, deadline={s['deadline']}, "
                              f"avg_trt={s['avg_trt']}, range=[{s['min_trt']}, {s['max_trt']}]")
            else:
                print(f"τ = {args.tau} | Feasible: NO ({r['termination']})")
                print(f"Deadlines: {r['deadlines_by_group']}")
