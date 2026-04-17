"""
Experiment 3: Computational Scalability
=========================================
Measures solve time vs N for all three models at fixed σ_L = 0.20.
Each (model, N) combination is run 3 times; median time is reported.

N ∈ {5, 10, 15, 20, 25, 30, 40, 50, 75}
Output: experiment_scalability.html
"""
import os, sys, time, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cart_colgen          import run_experiment          as det_run
from cart_colgen_ccp      import run_experiment_ccp      as ccp_run, DEFAULT_PROCESS_CCP
from cart_colgen_twostage import run_experiment_twostage as ts_run,  DEFAULT_PROCESS_2S

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SIGMA      = 0.20
N_REPS     = 3      # repetitions per (model, N) for stable timing
TIME_LIMIT = 300

N_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200]

def dat_file(n):
    return os.path.join(BASE_DIR, f'Data_N{n}.dat')

# ── single timed run ──────────────────────────────────────────────────────────

def timed_run(solver, n):
    df  = dat_file(n)
    cfg_ccp = dict(DEFAULT_PROCESS_CCP); cfg_ccp['sigma_L'] = SIGMA
    cfg_ts  = dict(DEFAULT_PROCESS_2S);  cfg_ts['sigma_L']  = SIGMA

    t0 = time.perf_counter()
    if solver == 'det':
        r = det_run(data_file=df, time_limit=TIME_LIMIT)
    elif solver == 'ccp':
        r = ccp_run(data_file=df, process_config=cfg_ccp, time_limit=TIME_LIMIT)
    else:
        r = ts_run(data_file=df,  process_config=cfg_ts,  time_limit=TIME_LIMIT)
    elapsed = time.perf_counter() - t0

    return {
        'solved':    r.get('solved', False),
        'wall_time': round(elapsed, 3),
        'gen_time':  round(r.get('gen_time', 0), 3),
        'solve_time':round(r.get('solve_time', 0), 3),
        'num_plans': r.get('num_plans', 0),
        'cost':      round(r.get('total_cost', 0) / 1e6, 4),
    }

# ── sweep ─────────────────────────────────────────────────────────────────────

def run_sweep():
    results = {}   # results[solver][n] = aggregated dict

    for solver, label in [('det', 'Deterministic'), ('ccp', 'CCP'), ('ts', 'Two-Stage')]:
        results[solver] = {}
        print(f'\n── {label} ──────────────────────────────')

        for n in N_VALUES:
            if not os.path.exists(dat_file(n)):
                print(f'  N={n:3d}  SKIP (no data file)')
                continue

            times, plans = [], None
            status = 'ok'
            for rep in range(N_REPS):
                r = timed_run(solver, n)
                if not r['solved']:
                    status = 'infeasible' if r['wall_time'] < TIME_LIMIT - 5 else 'timeout'
                    break
                times.append(r['wall_time'])
                plans = r['num_plans']

            if times:
                med = round(statistics.median(times), 3)
                mn  = round(min(times), 3)
                mx  = round(max(times), 3)
                results[solver][n] = {
                    'status': 'solved', 'median': med, 'min': mn, 'max': mx,
                    'num_plans': plans, 'times': times,
                }
                print(f'  N={n:3d}  median={med:7.3f}s  '
                      f'[{mn:.3f}–{mx:.3f}]  plans={plans}')
            else:
                results[solver][n] = {'status': status, 'median': None,
                                      'min': None, 'max': None, 'num_plans': None}
                print(f'  N={n:3d}  {status.upper()}')

    return results

# ── HTML ──────────────────────────────────────────────────────────────────────

COLORS = {
    'det': {'line': '#475569', 'fill': 'rgba(71,85,105,0.1)'},
    'ccp': {'line': '#3A7D8C', 'fill': 'rgba(58,125,140,0.1)'},
    'ts':  {'line': '#2C3E6B', 'fill': 'rgba(44,62,107,0.1)'},
}
LABELS = {'det': 'Deterministic ColGen', 'ccp': 'CCP (α=0.10)', 'ts': 'Two-Stage (δ=2.0)'}

def build_chart_js(results, y_field='median'):
    datasets = []
    for solver in ['det', 'ccp', 'ts']:
        pts = []
        for n in N_VALUES:
            r = results[solver].get(n, {})
            pts.append(r.get(y_field))   # None if infeasible/missing
        datasets.append({
            'label':           LABELS[solver],
            'data':            pts,
            'borderColor':     COLORS[solver]['line'],
            'backgroundColor': COLORS[solver]['fill'],
            'pointBackgroundColor': COLORS[solver]['line'],
            'pointRadius':     5,
            'tension':         0.3,
            'spanGaps':        False,
        })
    return json.dumps(datasets)

def summary_table(results):
    header = ('<tr style="background:#1E3A5F;color:#fff">'
              '<th style="padding:8px 12px">N</th>'
              '<th style="padding:8px 12px">Plans (Det)</th>'
              + ''.join(f'<th style="padding:8px 12px">{LABELS[s]}<br><small>median (s)</small></th>'
                        for s in ['det','ccp','ts'])
              + '</tr>')
    rows = [header]
    for n in N_VALUES:
        det_r = results['det'].get(n, {})
        plans = det_r.get('num_plans', '—')

        def cell(s):
            r = results[s].get(n, {})
            if r.get('status') == 'solved':
                t = r['median']
                bg = ('#fee2e2' if t > 60 else '#fef9c3' if t > 10 else '#f0fdf4')
                return (f'<td style="text-align:center;background:{bg};'
                        f'font-weight:600;padding:6px 12px">{t:.3f}s</td>')
            elif r.get('status') in ('infeasible', 'timeout'):
                return (f'<td style="text-align:center;color:#9ca3af;'
                        f'font-style:italic;padding:6px 12px">'
                        f'{r["status"].capitalize()}</td>')
            return '<td style="text-align:center;color:#9ca3af;padding:6px 12px">—</td>'

        rows.append(
            f'<tr style="border-bottom:1px solid #e2e8f0">'
            f'<td style="font-weight:700;padding:6px 12px;background:#f8fafc">N={n}</td>'
            f'<td style="text-align:center;padding:6px 12px">{plans}</td>'
            + ''.join(cell(s) for s in ['det','ccp','ts'])
            + '</tr>')
    return '<table style="border-collapse:collapse;width:100%;font-size:12px">' + ''.join(rows) + '</table>'

def plans_table(results):
    header = ('<tr style="background:#1E3A5F;color:#fff">'
              '<th style="padding:8px 12px">N</th>'
              + ''.join(f'<th style="padding:8px 12px">{LABELS[s]}<br><small># plans</small></th>'
                        for s in ['det','ccp','ts'])
              + '</tr>')
    rows = [header]
    for n in N_VALUES:
        def pcell(s):
            r = results[s].get(n, {})
            p = r.get('num_plans')
            if p is None:
                return '<td style="text-align:center;color:#9ca3af;padding:6px 12px">—</td>'
            return f'<td style="text-align:center;padding:6px 12px">{p:,}</td>'
        rows.append(
            f'<tr style="border-bottom:1px solid #e2e8f0">'
            f'<td style="font-weight:700;padding:6px 12px;background:#f8fafc">N={n}</td>'
            + ''.join(pcell(s) for s in ['det','ccp','ts'])
            + '</tr>')
    return '<table style="border-collapse:collapse;width:100%;font-size:12px">' + ''.join(rows) + '</table>'

def build_html(results):
    chart_ds  = build_chart_js(results)
    n_labels  = json.dumps(N_VALUES)
    sum_table = summary_table(results)
    pln_table = plans_table(results)
    raw_json  = json.dumps(
        {s: {str(n): v for n, v in nr.items()} for s, nr in results.items()}, indent=2)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 3: Computational Scalability</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body  {{ font-family:system-ui,sans-serif; margin:0; padding:24px 32px;
           background:#f8fafc; color:#1e293b; }}
  h1    {{ color:#1E3A5F; margin-bottom:4px; }}
  h2    {{ color:#1E3A5F; margin:0 0 12px; }}
  .card {{ background:#fff; border-radius:10px; padding:24px; margin-bottom:28px;
           box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .subtitle {{ color:#64748b; font-size:14px; margin-bottom:20px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  details summary {{ cursor:pointer;font-weight:600;color:#1E3A5F;font-size:13px }}
  pre {{ background:#f1f5f9;padding:12px;border-radius:6px;font-size:11px;
         overflow-x:auto;max-height:300px }}
</style>
</head>
<body>
<h1>Experiment 3 — Computational Scalability</h1>
<p class="subtitle">
  Solve time vs. N for all three models at fixed σ_L = {SIGMA}.
  Each point = median of {N_REPS} runs. Time limit: {TIME_LIMIT}s.
  <br>Cell shading:
  <span style="background:#f0fdf4;padding:1px 6px;border-radius:3px">fast (&lt;10s)</span>
  <span style="background:#fef9c3;padding:1px 6px;border-radius:3px">moderate (10–60s)</span>
  <span style="background:#fee2e2;padding:1px 6px;border-radius:3px">slow (&gt;60s)</span>
</p>

<div class="card">
  <h2>Solve Time vs N</h2>
  <div style="position:relative;height:380px">
    <canvas id="scaleChart"></canvas>
  </div>
</div>

<div class="card grid2">
  <div>
    <h2>Timing Summary (median seconds)</h2>
    {sum_table}
  </div>
  <div>
    <h2>Plan Count vs N</h2>
    {pln_table}
    <p style="font-size:11px;color:#64748b;margin-top:8px">
      Plans scale ≈ linearly with N.
      Two-Stage generates more plans than CCP due to looser deadline filter
      (expediting recourse relaxes the feasibility threshold).
    </p>
  </div>
</div>

<div class="card">
  <details>
    <summary>Raw JSON results</summary>
    <pre>{raw_json}</pre>
  </details>
</div>

<script>
new Chart(document.getElementById('scaleChart').getContext('2d'), {{
  type: 'line',
  data: {{
    labels: {n_labels},
    datasets: {chart_ds}
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{ callbacks: {{
        label: ctx => ctx.dataset.label + ': ' +
          (ctx.raw !== null ? ctx.raw.toFixed(3) + 's' : 'n/a')
      }} }}
    }},
    scales: {{
      x: {{ title: {{ display:true, text:'Number of patients (N)' }} }},
      y: {{
        type: 'logarithmic',
        title: {{ display:true, text:'Solve time (seconds, log scale)' }},
        ticks: {{
          callback: v => v >= 1 ? v + 's' : v.toFixed(2) + 's'
        }}
      }}
    }}
  }}
}});
</script>
</body>
</html>'''

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Computational Scalability Sweep')
    print(f'σ_L = {SIGMA}  |  reps = {N_REPS}  |  time limit = {TIME_LIMIT}s')
    print(f'N   = {N_VALUES}')
    total = len(N_VALUES) * 3 * N_REPS
    print(f'Total solver calls: {total}')

    results = run_sweep()

    out = os.path.join(BASE_DIR, 'experiment_scalability.html')
    with open(out, 'w') as f:
        f.write(build_html(results))
    print(f'\nSaved → {out}')
