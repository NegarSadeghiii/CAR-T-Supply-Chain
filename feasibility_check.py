"""
CAR-T Supply Chain — Little's Law Feasibility Pre-Screening
============================================================
Checks whether a (data, urgency, tau) configuration is feasible
BEFORE launching the expensive MIP solver.

Two checks:
  1. Physical feasibility:  D_p >= min_TRT for all patients
  2. Capacity feasibility:  Little's Law — peak arrival rate
     won't cause queue waits that exceed deadline slack

Returns: feasible (bool), diagnosis (dict with details)
"""

import re, random, math
from collections import Counter
from itertools import combinations


def parse_dat_light(filepath):
    """Parse .dat file — lightweight, no Pyomo needed."""
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


def check_feasibility(data_file, tau=0.0, urgency_config=None,
                      process_config=None, random_seed=42, verbose=True):
    """
    Pre-screening feasibility check using Little's Law.

    Returns (feasible: bool, report: dict)
    """
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS

    data = parse_dat_light(data_file)

    P = data['p']
    M = data['m']
    J = data['j']
    N = len(P)

    TLS  = int(process_config.get('tls', 1))
    TMFE = int(process_config.get('tmfe', 7))
    TQC  = int(process_config.get('tqc', 7))
    MAX_FAC = int(process_config.get('max_facilities', 2))

    TT1 = {j: int(data['TT1'][j]) for j in J}
    TT3 = {j: int(data['TT3'][j]) for j in J}
    FCAP = {m: int(data['FCAP'][m]) for m in M}

    min_tt1 = min(TT1.values())
    min_tt3 = min(TT3.values())
    min_trt = TLS + min_tt1 + TMFE + TQC + min_tt3

    # --- Patient arrivals ---
    arrivals = {}  # patient -> arrival day
    centers = {}   # patient -> center
    for (p, c, t), v in data['INC'].items():
        if v == 1:
            arrivals[p] = t
            centers[p] = c

    arrival_days = sorted(arrivals.values())
    time_span = max(arrival_days) - min(arrival_days) + 1 if arrival_days else 1

    # --- Assign urgency groups ---
    num_high = int(round(N * urgency_config['high']['fraction']))
    num_med  = int(round(N * urgency_config['medium']['fraction']))
    random.seed(random_seed)
    shuffled = P[:]
    random.shuffle(shuffled)

    group_map = {}
    deadlines = {}
    for idx, p in enumerate(shuffled):
        g = 'high' if idx < num_high else ('medium' if idx < num_high + num_med else 'low')
        cfg = urgency_config[g]
        group_map[p] = g
        deadlines[p] = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)

    # ============================================================
    # TRANSPORT MODE ANALYSIS
    # ============================================================
    # TRT depends on which transport modes are used.
    # j1 = fast (1 day, expensive), j2 = slow (2 days, cheap)
    # The solver may pick slow transport to save cost, adding days to TRT.

    # Compute TRT for each transport combination
    transport_combos = {}
    for j_out in J:
        for j_ret in J:
            trt = TLS + TT1[j_out] + TMFE + TQC + TT3[j_ret]
            transport_combos[(j_out, j_ret)] = trt

    fastest_trt = min(transport_combos.values())  # j1+j1
    slowest_trt = max(transport_combos.values())   # j2+j2
    transport_penalty = slowest_trt - fastest_trt  # days added by slow transport

    # Parse transport costs for cost-speed tradeoff
    U1_data = data.get('U1', {})
    U3_data = data.get('U3', {})

    # ============================================================
    # CHECK 1: Physical feasibility
    # ============================================================
    # Must check against BOTH fastest and slowest transport options:
    # - If deadline < fastest_trt: infeasible no matter what
    # - If deadline < slowest_trt: only feasible with fast (expensive) transport
    phys_violations = []
    transport_forced = []  # patients that MUST use fast transport
    for p in P:
        dl = deadlines[p]
        if dl < fastest_trt:
            phys_violations.append({
                'patient': p,
                'group': group_map[p],
                'deadline': dl,
                'min_trt': fastest_trt,
                'gap': fastest_trt - dl,
            })
        elif dl < slowest_trt:
            transport_forced.append({
                'patient': p,
                'group': group_map[p],
                'deadline': dl,
                'must_use_faster_than': dl,
                'feasible_combos': [(jo, jr) for (jo, jr), t in transport_combos.items() if t <= dl],
            })

    phys_feasible = len(phys_violations) == 0

    # ============================================================
    # CHECK 2: Capacity feasibility via Little's Law
    # ============================================================

    # Compute slack using the REALISTIC TRT (slow transport, since solver minimizes cost)
    # If solver picks slow transport: slack = deadline - slowest_trt
    # If forced to pick fast transport: slack = deadline - fastest_trt
    group_slack_fast = {}
    group_slack_slow = {}
    for g in urgency_config:
        dl = urgency_config[g]['base_due'] + urgency_config[g]['tolerance'] * (1.0 - tau)
        group_slack_fast[g] = dl - fastest_trt
        group_slack_slow[g] = dl - slowest_trt

    # Use slow-transport slack as the realistic scenario (solver picks cheap)
    group_slack = group_slack_slow
    min_slack = min(group_slack.values())

    # Compute peak arrival rates at various windows
    peak_analysis = {}
    for window in [TMFE, TMFE * 2, 1]:
        max_in_window = 0
        peak_start = 0
        for start in range(min(arrival_days), max(arrival_days) - window + 2):
            count = sum(1 for a in arrival_days if start <= a < start + window)
            if count > max_in_window:
                max_in_window = count
                peak_start = start
        peak_analysis[window] = {
            'window_days': window,
            'peak_count': max_in_window,
            'peak_start': peak_start,
            'wip_by_littles_law': max_in_window * TMFE / window if window > 0 else max_in_window,
        }

    # Average metrics
    avg_rate = N / time_span
    avg_wip = avg_rate * TMFE

    # Build all facility pairs ranked by cost (cheapest first) AND capacity (largest first)
    CIM = data.get('CIM', {})
    CVM = data.get('CVM', {})
    fac_sorted = sorted(FCAP.items(), key=lambda x: -x[1])

    all_pairs = []
    for combo in combinations(M, MAX_FAC):
        total_cap = sum(FCAP[m] for m in combo)
        total_cost = sum(CIM.get(m, 0) + CVM.get(m, 0) for m in combo)
        min_cap = min(FCAP[m] for m in combo)
        all_pairs.append({
            'facilities': combo,
            'total_cap': total_cap,
            'min_cap': min_cap,
            'total_cost': total_cost,
        })

    cheapest_pair = min(all_pairs, key=lambda x: x['total_cost'])
    largest_pair = max(all_pairs, key=lambda x: x['total_cap'])
    best_pair = (largest_pair['facilities'], largest_pair['total_cap'])
    best_total_cap = largest_pair['total_cap']

    # Little's Law check: during peak window, WIP must fit in capacity
    peak_tmfe = peak_analysis[TMFE]
    peak_wip = peak_tmfe['wip_by_littles_law']

    # Per-center burst rates
    center_arrivals = {}
    for p in P:
        c = centers.get(p, '?')
        day = arrivals.get(p, 0)
        center_arrivals.setdefault(c, []).append(day)

    center_peaks = {}
    for c, days in center_arrivals.items():
        days.sort()
        max_in_window = 0
        for start in range(min(days), max(days) + 1):
            count = sum(1 for d in days if start <= d < start + TMFE)
            max_in_window = max(max_in_window, count)
        center_peaks[c] = max_in_window

    # Queue wait estimation per facility pair using Little's Law
    # Key insight: the solver minimizes cost, so it may pick CHEAP facilities
    # with small FCAP. We must check BOTH cheapest and largest pairs.
    # If peak WIP > min_cap of chosen pair, patients at that facility queue.
    # Queue wait ≈ ceil((peak_arriving - FCAP) / FCAP) * TMFE-ish
    # Simpler: excess patients wait 1 manufacturing cycle per FCAP overflow
    queue_wait_issues = []
    for pair_info in sorted(all_pairs, key=lambda x: x['total_cost']):
        facs = pair_info['facilities']
        total_cap = pair_info['total_cap']
        min_cap = pair_info['min_cap']
        total_cost = pair_info['total_cost']

        # Worst case: all peak patients route to the SMALLER facility
        worst_case_wip = peak_tmfe['peak_count']  # raw patient count in window
        if worst_case_wip > min_cap:
            excess = worst_case_wip - min_cap
            # Each excess patient waits ~TMFE days for a slot to free up
            # More precisely: batches of FCAP are processed every TMFE days
            est_queue_wait = math.ceil(excess / min_cap) * TMFE
        else:
            excess = 0
            est_queue_wait = 0

        queue_wait_issues.append({
            'facilities': facs,
            'total_cap': total_cap,
            'min_cap': min_cap,
            'total_cost': round(total_cost),
            'peak_patients': worst_case_wip,
            'excess': excess,
            'est_queue_wait': est_queue_wait,
            'min_slack': round(min_slack, 1),
            'feasible': est_queue_wait <= min_slack,
        })

    # Feasible if at least one pair works, but flag if cheapest pair doesn't
    cap_feasible = any(q['feasible'] for q in queue_wait_issues)
    cheapest_feasible = queue_wait_issues[0]['feasible'] if queue_wait_issues else True
    cost_capacity_conflict = cap_feasible and not cheapest_feasible

    # ============================================================
    # CHECK 3: Per-center load balance
    # ============================================================
    # If one center sends too many patients in a burst,
    # and they all route to the same facility, check single-facility cap
    center_issues = []
    max_single_fcap = max(FCAP.values())
    for c, peak in center_peaks.items():
        wip = peak  # all arrive within TMFE window, so WIP ≈ count
        if wip > max_single_fcap:
            center_issues.append({
                'center': c,
                'peak_patients_in_window': peak,
                'max_single_fcap': max_single_fcap,
                'needs_splitting': True,
            })

    # ============================================================
    # OVERALL VERDICT
    # ============================================================
    overall_feasible = phys_feasible and cap_feasible

    report = {
        'tau': tau,
        'num_patients': N,
        'min_trt': fastest_trt,
        'max_trt_slow_transport': slowest_trt,
        'transport_penalty': transport_penalty,
        'transport_combos': transport_combos,
        'transport_forced_patients': len(transport_forced),
        'trt_breakdown': {
            'TLS': TLS, 'TT_out_min': min_tt1,
            'T_MF': TMFE, 'T_QC': TQC, 'TT_ret_min': min_tt3,
        },
        'deadlines': {g: round(urgency_config[g]['base_due'] + urgency_config[g]['tolerance'] * (1 - tau), 1)
                      for g in urgency_config},
        'slack': {g: round(v, 1) for g, v in group_slack.items()},
        'physical_feasibility': {
            'feasible': phys_feasible,
            'violations': phys_violations,
        },
        'capacity_feasibility': {
            'feasible': cap_feasible,
            'avg_arrival_rate': round(avg_rate, 3),
            'avg_wip': round(avg_wip, 1),
            'peak_in_tmfe_window': peak_tmfe,
            'best_facility_pair': {
                'facilities': best_pair[0],
                'combined_cap': best_pair[1],
            },
            'queue_analysis': queue_wait_issues,
            'center_peaks': center_peaks,
            'center_issues': center_issues,
        },
        'overall_feasible': overall_feasible,
        'cost_capacity_conflict': cost_capacity_conflict,
        'cheapest_pair': cheapest_pair,
        'largest_pair': largest_pair,
    }

    if verbose:
        _print_report(report)

    return overall_feasible, report


def _print_report(r):
    """Pretty-print the feasibility report."""
    print()
    print('=' * 70)
    print('  LITTLE\'S LAW FEASIBILITY PRE-SCREENING')
    print('=' * 70)
    print()
    print('  Patients: %d    tau: %.1f' % (r['num_patients'], r['tau']))
    b = r['trt_breakdown']
    print('  Min TRT (fast transport): %d + %d + %d + %d + %d = %d days' % (
        b['TLS'], b['TT_out_min'], b['T_MF'], b['T_QC'], b['TT_ret_min'], r['min_trt']))
    print('  TRT with slow transport:  %d days (+%d days transport penalty)' % (
        r['max_trt_slow_transport'], r['transport_penalty']))
    if r['transport_forced_patients'] > 0:
        print('  ** %d patients MUST use fast transport to meet deadline **' % r['transport_forced_patients'])
    print()

    # Deadlines and slack
    print('  %-8s  %8s  %12s  %12s' % ('Group', 'Deadline', 'Slack(fast)', 'Slack(slow)'))
    print('  ' + '-' * 46)
    for g in ('high', 'medium', 'low'):
        dl = r['deadlines'].get(g, '?')
        sl_fast = dl - r['min_trt'] if isinstance(dl, (int, float)) else '?'
        sl_slow = dl - r['max_trt_slow_transport'] if isinstance(dl, (int, float)) else '?'
        flag = ''
        if isinstance(sl_slow, (int, float)) and sl_slow < 0:
            flag = '  ** must use fast transport'
        if isinstance(sl_fast, (int, float)) and sl_fast < 0:
            flag = '  *** INFEASIBLE'
        print('  %-8s  %8s  %12.1f  %12.1f%s' % (g, dl, sl_fast, sl_slow, flag))
    print()

    # Check 1
    pf = r['physical_feasibility']
    if pf['feasible']:
        print('  [PASS] Physical feasibility: all deadlines >= min TRT')
    else:
        print('  [FAIL] Physical feasibility: %d patients have deadline < min TRT' % len(pf['violations']))
        for v in pf['violations'][:5]:
            print('         %s (%s): deadline=%.1f < min_trt=%d (gap=%.1f days)' % (
                v['patient'], v['group'], v['deadline'], v['min_trt'], v['gap']))
    print()

    # Check 2
    cf = r['capacity_feasibility']
    peak = cf['peak_in_tmfe_window']
    print('  Arrival pattern:')
    print('    Average rate: %.3f patients/day' % cf['avg_arrival_rate'])
    print('    Average WIP (Little\'s Law): %.1f patients' % cf['avg_wip'])
    print('    Peak in %d-day window: %d patients (days %d-%d)' % (
        peak['window_days'], peak['peak_count'], peak['peak_start'],
        peak['peak_start'] + peak['window_days'] - 1))
    print('    Peak WIP (Little\'s Law): %.1f patients' % peak['wip_by_littles_law'])
    print()

    # Cheapest vs largest pair
    cp = r.get('cheapest_pair', {})
    lp = r.get('largest_pair', {})
    print('  Cheapest pair: %s (FCAP=%d, min=%d, cost=$%s)' % (
        '+'.join(cp.get('facilities', [])), cp.get('total_cap', 0),
        cp.get('min_cap', 0), '{:,}'.format(int(cp.get('total_cost', 0)))))
    print('  Largest pair:  %s (FCAP=%d, min=%d, cost=$%s)' % (
        '+'.join(lp.get('facilities', [])), lp.get('total_cap', 0),
        lp.get('min_cap', 0), '{:,}'.format(int(lp.get('total_cost', 0)))))
    print()

    if cf['queue_analysis']:
        # Show cheapest pair analysis (what solver likely picks)
        qa_cheap = cf['queue_analysis'][0]  # sorted by cost
        print('  Queue analysis (cheapest pair — likely solver choice):')
        print('    Peak patients in MFG window: %d' % qa_cheap['peak_patients'])
        print('    Smallest facility FCAP: %d' % qa_cheap['min_cap'])
        if qa_cheap['excess'] > 0:
            print('    Excess if all peak -> small facility: %d patients' % qa_cheap['excess'])
            print('    Estimated queue wait: %d days' % qa_cheap['est_queue_wait'])
            print('    Min deadline slack: %.1f days' % qa_cheap['min_slack'])
            if qa_cheap['feasible']:
                print('    [PASS] Queue wait fits within slack')
            else:
                print('    [FAIL] Queue wait EXCEEDS slack -> solver may pick this and fail!')
        else:
            print('    [PASS] Peak fits in smallest facility')

        # Check if largest pair would work
        qa_large = next((q for q in cf['queue_analysis']
                        if set(q['facilities']) == set(lp.get('facilities', []))), None)
        if qa_large and qa_large['feasible'] and not qa_cheap['feasible']:
            print()
            print('    ** COST-CAPACITY CONFLICT **')
            print('    Cheapest pair is infeasible but largest pair works.')
            print('    The solver minimizes cost, so it may pick the cheap')
            print('    pair and then fail to find a feasible schedule.')
            print('    Extra cost for feasibility: $%s' % '{:,}'.format(
                int(lp.get('total_cost', 0) - cp.get('total_cost', 0))))
    print()

    # Check 3
    if cf['center_issues']:
        print('  [WARN] Center load balance issues:')
        for ci in cf['center_issues']:
            print('    %s: peak %d patients in MFG window > max single FCAP %d' % (
                ci['center'], ci['peak_patients_in_window'], ci['max_single_fcap']))
            print('    -> Must split across facilities')
    print()

    # Overall
    print('  ' + '=' * 40)
    if r['overall_feasible']:
        print('  VERDICT: LIKELY FEASIBLE')
        min_sl = min(r['slack'].values())
        if min_sl <= 2:
            print('  (tight — capacity congestion could still cause issues)')
    else:
        print('  VERDICT: INFEASIBLE')
        if not pf['feasible']:
            print('  -> Deadlines below physical minimum')
        if not cf['feasible']:
            print('  -> Capacity cannot absorb peak demand within slack')
        print()
        print('  Suggested fixes:')
        if not pf['feasible']:
            min_gap = max(v['gap'] for v in pf['violations'])
            print('    1. Increase high base_due by at least %.0f days' % math.ceil(min_gap))
            print('    2. Reduce T_MF or T_QC by at least %.0f days' % math.ceil(min_gap))
        if not cf['feasible'] and cf['queue_analysis']:
            qa = cf['queue_analysis'][0]
            print('    3. Increase FCAP at top facilities by %.0f' % math.ceil(qa['excess']))
            print('    4. Increase deadline tolerance to allow %.1f+ days slack' % qa['est_queue_wait'])
    print('  ' + '=' * 40)
    print()


# ── CLI ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser(description='Little\'s Law Feasibility Check')
    parser.add_argument('--data', default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--high-base-due', type=float, default=18)
    parser.add_argument('--tmfe', type=int, default=7)
    parser.add_argument('--tqc', type=int, default=7)
    args = parser.parse_args()

    data_file = args.data or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N50.dat')

    urgency = {
        'high':   {'fraction': 0.20, 'base_due': args.high_base_due, 'tolerance': 1},
        'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
        'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
    }
    process = {'tls': 1, 'tmfe': args.tmfe, 'tqc': args.tqc, 'max_facilities': 2}

    if args.sweep:
        print()
        print('SWEEP: tau from 0.0 to 1.0')
        print('%5s  %8s  %6s  %6s  %6s  %10s  %10s  %s' % (
            'tau', 'Feasible', 'H_DL', 'H_slk', 'PkPat', 'Cheap_FCAP', 'Large_FCAP', 'Note'))
        print('-' * 75)
        for tau_val in [i / 10.0 for i in range(11)]:
            ok, rpt = check_feasibility(data_file, tau=tau_val,
                                         urgency_config=urgency,
                                         process_config=process, verbose=False)
            h_dl = rpt['deadlines']['high']
            h_sl = rpt['slack']['high']
            pk = rpt['capacity_feasibility']['peak_in_tmfe_window']['peak_count']
            cp_cap = rpt['cheapest_pair']['min_cap']
            lp_cap = rpt['largest_pair']['min_cap']
            conflict = rpt.get('cost_capacity_conflict', False)
            status = 'YES' if ok else 'NO'
            note = 'COST-CAP CONFLICT' if conflict else ''
            if not rpt['physical_feasibility']['feasible']:
                note = 'PHYS INFEASIBLE'
            print('%5.1f  %8s  %6.1f  %6.1f  %6d  %10d  %10d  %s' % (
                tau_val, status, h_dl, h_sl, pk, cp_cap, lp_cap, note))
    elif args.tau is not None:
        check_feasibility(data_file, tau=args.tau,
                          urgency_config=urgency, process_config=process)
    else:
        # Default: show full report for tau=0 and tau=1
        for t in [0.0, 0.5, 1.0]:
            check_feasibility(data_file, tau=t,
                              urgency_config=urgency, process_config=process)
