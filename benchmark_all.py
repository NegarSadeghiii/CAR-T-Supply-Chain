"""
CAR-T Supply Chain — Benchmark: Hard-Deadline MIP vs ColGen vs Benders
======================================================================
All three solve the SAME problem: hard deadline, pure cost minimization.
Optimality gap is measured against the hard-deadline MIP exact solution.
"""

import os, time, json, sys, re

from hard_deadline_model import run_experiment as run_mip_hard
from cart_colgen import run_experiment as run_colgen
from cart_benders import run_experiment as run_benders

URGENCY = {
    'high':   {'fraction': 0.20, 'base_due': 18, 'tolerance': 1},
    'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
    'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
}

SIZES = [
    ('N=5',  'Data_N5.dat'),
    ('N=10', 'Data_N10.dat'),
    ('N=15', 'Data_N15.dat'),
]
TAUS = [0.0, 1.0]
MIP_TIME_LIMIT = 600
CG_TIME_LIMIT = 300
BD_TIME_LIMIT = 300


def run_safe(func, label, **kwargs):
    try:
        t0 = time.time()
        r = func(**kwargs)
        r['wall_time'] = round(time.time() - t0, 2)
        return r
    except Exception as e:
        return {
            'solved': False, 'feasible': False,
            'reason': str(e)[:120],
            'wall_time': round(time.time() - t0, 2),
        }


def fmt_cost(c):
    return '${:,.0f}'.format(c) if c else '---'


def fmt_time(t):
    if t is None: return '---'
    return '%.1fs' % t if t < 60 else '%.1fm' % (t / 60)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    all_results = []

    print()
    print('=' * 90)
    print('  CAR-T SUPPLY CHAIN — SOLVER BENCHMARK (HARD DEADLINE)')
    print('  All methods solve: min cost s.t. TRT_p <= D_p for all p')
    print('  Transport: Air (1d, ~$3500/leg) | Ground (4d, ~$1200/leg)')
    print('=' * 90)

    for label, fname in SIZES:
        fpath = os.path.join(base, fname)

        # Get patient count and time horizon for MIP
        with open(fpath) as f:
            txt = f.read()
        n_patients = len(re.findall(r'p\d+\s+c\d+\s+\d+\s+1', txt))

        for tau in TAUS:
            print()
            print('-' * 70)
            print('  %s | tau=%.1f | %d patients' % (label, tau, n_patients))
            print('-' * 70)

            row = {'size': label, 'tau': tau, 'n': n_patients}

            # --- Hard-Deadline MIP (exact baseline) ---
            print('  [MIP]    Solving...', end='', flush=True)
            r_mip = run_safe(run_mip_hard, 'MIP',
                             tau=tau, data_file=fpath,
                             urgency_config=URGENCY,
                             solver_name='appsi_highs')
            mip_solved = r_mip.get('feasible', r_mip.get('solved', False))
            mip_cost = r_mip.get('total_cost') if mip_solved else None
            mip_time = r_mip.get('solve_time', r_mip.get('wall_time'))
            mip_trt = r_mip.get('avg_trt')
            if mip_solved:
                print(' DONE | Cost: %s | TRT: %.2f | Time: %s' % (
                    fmt_cost(mip_cost), mip_trt, fmt_time(mip_time)))
            else:
                reason = r_mip.get('termination', r_mip.get('reason', '?'))
                print(' FAILED (%s) | Time: %s' % (str(reason)[:50], fmt_time(mip_time)))
            row['mip'] = r_mip

            # --- Column Generation ---
            print('  [ColGen] Solving...', end='', flush=True)
            r_cg = run_safe(run_colgen, 'ColGen',
                            tau=tau, data_file=fpath,
                            urgency_config=URGENCY,
                            time_limit=CG_TIME_LIMIT)
            cg_solved = r_cg.get('solved', False)
            cg_cost = r_cg.get('total_cost') if cg_solved else None
            cg_time = r_cg.get('solve_time', r_cg.get('wall_time'))
            cg_trt = r_cg.get('avg_trt')
            if cg_solved:
                n_plans = r_cg.get('num_plans', 0)
                opt = 'OPTIMAL' if r_cg.get('optimal') else 'FEASIBLE'
                print(' DONE | Cost: %s | TRT: %.2f | Plans: %d | Time: %s | %s' % (
                    fmt_cost(cg_cost), cg_trt, n_plans, fmt_time(cg_time), opt))
            else:
                reason = r_cg.get('reason', r_cg.get('termination', '?'))
                print(' FAILED (%s) | Time: %s' % (str(reason)[:50], fmt_time(cg_time)))
            row['colgen'] = r_cg

            # --- Benders ---
            print('  [Benders] Solving...', end='', flush=True)
            r_bd = run_safe(run_benders, 'Benders',
                            tau=tau, data_file=fpath,
                            urgency_config=URGENCY,
                            time_limit=BD_TIME_LIMIT)
            bd_solved = r_bd.get('solved', False)
            bd_cost = r_bd.get('total_cost') if bd_solved else None
            bd_time = r_bd.get('solve_time', r_bd.get('wall_time'))
            bd_trt = r_bd.get('avg_trt')
            if bd_solved:
                iters = r_bd.get('iterations', 0)
                cuts = r_bd.get('num_cuts', 0)
                print(' DONE | Cost: %s | TRT: %.2f | Iter: %d | Cuts: %d | Time: %s' % (
                    fmt_cost(bd_cost), bd_trt, iters, cuts, fmt_time(bd_time)))
            else:
                reason = r_bd.get('reason', r_bd.get('termination', '?'))
                print(' FAILED (%s) | Time: %s' % (str(reason)[:50], fmt_time(bd_time)))
            row['benders'] = r_bd

            # --- Optimality Gap ---
            print()
            if mip_cost:
                ref = mip_cost
                print('  Optimality gaps vs Hard-Deadline MIP ($%s):' % '{:,.0f}'.format(ref))
                for name, cost, t in [('ColGen', cg_cost, cg_time), ('Benders', bd_cost, bd_time)]:
                    if cost:
                        gap = (cost - ref) / ref * 100
                        speedup = mip_time / t if t and t > 0 else float('inf')
                        print('    %8s: %s  gap=%+.2f%%  speedup=%.0fx' % (
                            name, fmt_cost(cost), gap, speedup))
                    else:
                        print('    %8s: FAILED' % name)
            else:
                # MIP failed — compare CG vs BD
                print('  MIP failed. Comparing ColGen vs Benders:')
                if cg_cost and bd_cost:
                    best = min(cg_cost, bd_cost)
                    for name, cost in [('ColGen', cg_cost), ('Benders', bd_cost)]:
                        gap = (cost - best) / best * 100
                        print('    %8s: %s  gap=%+.2f%%' % (name, fmt_cost(cost), gap))

            all_results.append(row)

    # === SUMMARY TABLE ===
    print()
    print()
    print('=' * 95)
    print('  SUMMARY TABLE')
    print('=' * 95)
    print()
    print('%6s %4s | %12s %7s | %12s %7s %7s | %12s %7s %5s %7s' % (
        'Size', 'tau',
        'MIP_Cost', 'MIP_T',
        'CG_Cost', 'CG_T', 'CG_Gap',
        'BD_Cost', 'BD_T', 'Iter', 'BD_Gap'))
    print('-' * 95)

    for row in all_results:
        r_m = row['mip']
        r_c = row['colgen']
        r_b = row['benders']

        m_ok = r_m.get('feasible', r_m.get('solved', False))
        c_ok = r_c.get('solved', False)
        b_ok = r_b.get('solved', False)

        mc = '%12s' % fmt_cost(r_m.get('total_cost')) if m_ok else '%12s' % 'FAIL'
        mt = '%7s' % fmt_time(r_m.get('solve_time', r_m.get('wall_time')))

        cc = '%12s' % fmt_cost(r_c.get('total_cost')) if c_ok else '%12s' % 'FAIL'
        ct = '%7s' % fmt_time(r_c.get('solve_time', r_c.get('wall_time')))

        bc = '%12s' % fmt_cost(r_b.get('total_cost')) if b_ok else '%12s' % 'FAIL'
        bt = '%7s' % fmt_time(r_b.get('solve_time', r_b.get('wall_time')))
        bi = '%5s' % str(r_b.get('iterations', '-'))

        # Gaps vs MIP
        ref = r_m.get('total_cost') if m_ok else None
        if ref and c_ok:
            cg_gap = '%+6.1f%%' % ((r_c['total_cost'] - ref) / ref * 100)
        else:
            cg_gap = '%7s' % '---'
        if ref and b_ok:
            bd_gap = '%+6.1f%%' % ((r_b['total_cost'] - ref) / ref * 100)
        else:
            bd_gap = '%7s' % '---'

        print('%6s %4.1f | %s %s | %s %s %s | %s %s %s %s' % (
            row['size'], row['tau'],
            mc, mt, cc, ct, cg_gap, bc, bt, bi, bd_gap))

    # Save
    out = os.path.join(base, 'benchmark_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print()
    print('Results saved to benchmark_results.json')


if __name__ == '__main__':
    main()
