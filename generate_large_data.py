"""
Large-scale data generator for CAR-T supply chain experiments.

Generates N-patient .dat files within a single 90-day trimester window,
matching the problem structure described in the source paper:
  - One trimester = 90 days planning horizon (+30 day buffer = 120 total)
  - Arrivals drawn uniformly over [1, 90]
  - Max 8 patients/day per leukapheresis site (4 sites → 32/day max → 2880 max N)
  - Clinic assignment cycles c1→c2→c3→c4
  - Patient IDs: p1, p2, ..., pN

Usage: python generate_large_data.py <N> [seed]
"""
import re, sys, os, random

TRIMESTER   = 90    # planning window (days)
BUFFER      = 30    # post-last-arrival buffer
MAX_PER_SITE_PER_DAY = 8
CLINICS     = ['c1', 'c2', 'c3', 'c4']

def generate_large(num_patients, seed=42,
                   input_file='Data200_profileA.dat', output_file=None):
    with open(input_file) as f:
        txt = f.read()

    # Validate capacity constraint
    max_n = MAX_PER_SITE_PER_DAY * len(CLINICS) * TRIMESTER
    if num_patients > max_n:
        raise ValueError(f'N={num_patients} exceeds max capacity '
                         f'({MAX_PER_SITE_PER_DAY} patients/site/day × '
                         f'{len(CLINICS)} sites × {TRIMESTER} days = {max_n})')

    rng = random.Random(seed)

    # Track daily arrivals per clinic to respect the 8/day/site limit
    daily = {c: {d: 0 for d in range(1, TRIMESTER + 1)} for c in CLINICS}
    selected = []

    for i in range(num_patients):
        clinic = CLINICS[i % len(CLINICS)]
        # Pick a random day that still has capacity
        avail = [d for d in range(1, TRIMESTER + 1)
                 if daily[clinic][d] < MAX_PER_SITE_PER_DAY]
        if not avail:
            raise ValueError(f'Capacity exhausted for clinic {clinic} at patient {i+1}')
        day = rng.choice(avail)
        daily[clinic][day] += 1
        selected.append((f'p{i+1}', clinic, str(day)))

    # Sort by arrival day for readability
    selected.sort(key=lambda x: (int(x[2]), x[1]))

    time_horizon = TRIMESTER + BUFFER

    if output_file is None:
        output_file = f'Data_N{num_patients}.dat'

    lines = []
    lines.append('# Auto-generated large-scale trimester data')
    lines.append(f'# {num_patients} patients over {TRIMESTER}-day trimester, '
                 f'time horizon {time_horizon}')
    lines.append('')
    lines.append('set c := c1 c2 c3 c4;')
    lines.append('set h := h1 h2 h3 h4;')
    lines.append('set j := j1 j2;')
    lines.append('set m := m1 m2 m3 m4 m5 m6;')
    lines.append(f'set p := {" ".join(p for p, _, _ in selected)};')
    lines.append('')

    for param in ['CIM', 'CVM', 'FCAP', 'TT1', 'TT3', 'U1', 'U3']:
        m = re.search(rf'param {param}\s*:=(.*?);', txt, re.DOTALL)
        if m:
            lines.append(f'param {param} :={m.group(1)};')
            lines.append('')

    lines.append('param INC :=')
    for p, c, d in selected:
        lines.append(f'{p} {c} {d} 1')
    lines.append(';')
    lines.append('')

    for scalar in ['FMAX', 'FMIN', 'TAD', 'TLS']:
        m = re.search(rf'param {scalar}\s*:=\s*(\S+)\s*;', txt)
        if m:
            lines.append(f'param {scalar} := {m.group(1)} ;')

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    arr_days = [int(d) for _, _, d in selected]
    print(f'Generated {output_file}: {num_patients} patients, '
          f'arrivals day {min(arr_days)}–{max(arr_days)}, '
          f'time horizon {time_horizon} days')
    return output_file


if __name__ == '__main__':
    n    = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    generate_large(n, seed=seed)
