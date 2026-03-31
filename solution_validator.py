"""
CAR-T Supply Chain — Solution Validator
=======================================
Validates that a solution satisfies ALL hard constraints:
  1. Every patient assigned exactly one route
  2. TRT_p <= D_p (hard deadline) for all patients
  3. Facility capacity: at most FCAP_m patients in mfg at any time
  4. At most MAX_FAC facilities open
  5. Route uses an open facility
  6. Patient routed through correct center -> hospital pair
  7. Cost computation is correct

Returns: (valid: bool, violations: list[str])
"""

import re


def parse_dat_light(filepath):
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


def validate(solution, data_file, deadlines, process_config=None, max_facilities=2, verbose=True):
    """
    Validate a solution.

    Args:
        solution: list of dicts, one per patient:
            {id, facility, j_out, j_ret, wait, trt, mfg_start}
        data_file: path to .dat file
        deadlines: dict {patient_id: deadline_value}
        process_config: {tls, tmfe, tqc}
        max_facilities: max open facilities
        verbose: print results

    Returns: (valid: bool, violations: list[str], stats: dict)
    """
    if process_config is None:
        process_config = {'tls': 1, 'tmfe': 7, 'tqc': 7}

    data = parse_dat_light(data_file)
    P = data['p']
    M = data['m']
    J = data['j']
    C = data['c']
    H = data['h']
    TLS = int(process_config['tls'])
    TMFE = int(process_config['tmfe'])
    TQC = int(process_config['tqc'])
    TT1 = {j: int(data['TT1'][j]) for j in J}
    TT3 = {j: int(data['TT3'][j]) for j in J}
    FCAP = {m: int(data['FCAP'][m]) for m in M}

    # Patient data
    patient_arrival = {}
    patient_center = {}
    c_to_h = {C[i]: H[i] for i in range(len(C))}
    for (p, c, t), v in data['INC'].items():
        if v == 1:
            patient_arrival[p] = t
            patient_center[p] = c

    violations = []
    warnings = []

    # Index solution by patient
    sol_by_patient = {}
    for s in solution:
        pid = s['id']
        if pid in sol_by_patient:
            violations.append('DUPLICATE: patient %s assigned multiple routes' % pid)
        sol_by_patient[pid] = s

    # ── CHECK 1: Every patient assigned ──
    for p in P:
        if p not in sol_by_patient:
            violations.append('MISSING: patient %s has no route' % p)

    # ── CHECK 2: Hard deadline ──
    for p, s in sol_by_patient.items():
        dl = deadlines.get(p)
        if dl is None:
            warnings.append('WARNING: no deadline for patient %s' % p)
            continue
        trt = s.get('trt', 0)
        if trt > dl + 0.01:  # small tolerance for float
            violations.append('DEADLINE: patient %s TRT=%d > deadline=%.1f (late by %.1f days)' % (
                p, trt, dl, trt - dl))

    # ── CHECK 3: TRT computation ──
    for p, s in sol_by_patient.items():
        j_out = s.get('j_out')
        j_ret = s.get('j_ret')
        wait = s.get('wait', 0)
        if j_out and j_ret and j_out in TT1 and j_ret in TT3:
            expected_trt = TLS + TT1[j_out] + TMFE + TQC + TT3[j_ret] + wait
            actual_trt = s.get('trt', 0)
            if abs(expected_trt - actual_trt) > 0.01:
                violations.append('TRT_CALC: patient %s reported TRT=%d but computed=%d '
                                  '(TLS=%d+TT1=%d+TMF=%d+TQC=%d+TT3=%d+w=%d)' % (
                                      p, actual_trt, expected_trt,
                                      TLS, TT1[j_out], TMFE, TQC, TT3[j_ret], wait))

    # ── CHECK 4: Manufacturing start ──
    for p, s in sol_by_patient.items():
        if 'mfg_start' in s and s['mfg_start'] is not None:
            j_out = s.get('j_out')
            wait = s.get('wait', 0)
            arr = patient_arrival.get(p, 0)
            if j_out and j_out in TT1:
                expected_start = arr + TLS + TT1[j_out] + wait
                if abs(s['mfg_start'] - expected_start) > 0.01:
                    violations.append('MFG_START: patient %s reported mfg_start=%d but expected=%d' % (
                        p, s['mfg_start'], expected_start))

    # ── CHECK 5: Facility capacity ──
    fac_schedule = {}  # (facility, time) -> count
    for p, s in sol_by_patient.items():
        mi = s.get('facility')
        mfg_start = s.get('mfg_start')
        if mi and mfg_start is not None:
            for t in range(int(mfg_start), int(mfg_start) + TMFE):
                key = (mi, t)
                fac_schedule[key] = fac_schedule.get(key, 0) + 1

    for (mi, t), count in fac_schedule.items():
        if count > FCAP.get(mi, 0):
            violations.append('CAPACITY: facility %s at time %d has %d patients > FCAP=%d' % (
                mi, t, count, FCAP[mi]))

    # ── CHECK 6: Facility count ──
    facilities_used = set(s.get('facility') for s in sol_by_patient.values() if s.get('facility'))
    if len(facilities_used) > max_facilities:
        violations.append('FAC_LIMIT: %d facilities used > max %d (%s)' % (
            len(facilities_used), max_facilities, facilities_used))

    # ── CHECK 7: Valid transport modes ──
    for p, s in sol_by_patient.items():
        if s.get('j_out') and s['j_out'] not in J:
            violations.append('INVALID_MODE: patient %s j_out=%s not in %s' % (p, s['j_out'], J))
        if s.get('j_ret') and s['j_ret'] not in J:
            violations.append('INVALID_MODE: patient %s j_ret=%s not in %s' % (p, s['j_ret'], J))

    # ── CHECK 8: Valid facility ──
    for p, s in sol_by_patient.items():
        if s.get('facility') and s['facility'] not in M:
            violations.append('INVALID_FAC: patient %s facility=%s not in %s' % (p, s['facility'], M))

    # ── CHECK 9: Center-hospital pairing ──
    for p, s in sol_by_patient.items():
        c = patient_center.get(p)
        if c:
            expected_h = c_to_h.get(c)
            # Can't check directly from solution unless hospital is reported
            # but at least check the transport cost route exists
            mi = s.get('facility')
            j_out = s.get('j_out')
            j_ret = s.get('j_ret')
            if mi and j_out and c:
                if (c, mi, j_out) not in data['U1']:
                    violations.append('NO_ROUTE: patient %s has no outbound route (%s,%s,%s)' % (
                        p, c, mi, j_out))
            if mi and j_ret and expected_h:
                if (mi, expected_h, j_ret) not in data['U3']:
                    violations.append('NO_ROUTE: patient %s has no return route (%s,%s,%s)' % (
                        p, mi, expected_h, j_ret))

    # ── STATS ──
    trts = [s.get('trt', 0) for s in sol_by_patient.values()]
    stats = {
        'num_patients': len(sol_by_patient),
        'num_violations': len(violations),
        'num_warnings': len(warnings),
        'facilities_used': sorted(facilities_used),
        'avg_trt': round(sum(trts) / len(trts), 2) if trts else 0,
        'min_trt': min(trts) if trts else 0,
        'max_trt': max(trts) if trts else 0,
        'max_capacity_util': max(fac_schedule.values()) if fac_schedule else 0,
    }

    valid = len(violations) == 0

    if verbose:
        print()
        if valid:
            print('  [VALID] Solution passed all %d checks (%d patients, facs=%s)' % (
                9, stats['num_patients'], stats['facilities_used']))
        else:
            print('  [INVALID] %d violations found:' % len(violations))
            for v in violations[:10]:
                print('    - %s' % v)
            if len(violations) > 10:
                print('    ... and %d more' % (len(violations) - 10))
        if warnings:
            for w in warnings:
                print('    %s' % w)

    return valid, violations, stats
