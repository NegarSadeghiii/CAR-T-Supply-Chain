"""
Experiment 1: Feasibility Phase Diagram
========================================
Runs CCP and Two-Stage solvers across a grid of N x sigma_L values.
Each cell records: Solved / Infeasible / Timeout
Results saved to experiment_feasibility.html

NOTE ON LARGE-N CRASHES
-----------------------
For N >= 500 the solver (HiGHS/CBC) can exhaust available memory mid-run,
causing the Python process to be killed by the OS before it exits cleanly.
This is expected behaviour and is NOT a data-loss risk: every completed run
is written to experiment_feasibility_checkpoint.json immediately after it
finishes, so restarting the script resumes from exactly where it stopped.

If the sweep dies before finishing, simply re-run:
    python experiment_feasibility.py
and it will skip all cached entries and only run the missing ones.

To regenerate the HTML from a complete (or partial) checkpoint without
re-running any solvers:
    python - <<'EOF'
    import json, experiment_feasibility as ef
    with open('experiment_feasibility_checkpoint.json') as f:
        raw = json.load(f)
    results = {s: {int(n): {float(sg): v for sg, v in sv.items()}
                   for n, sv in nv.items()} for s, nv in raw.items()}
    open('experiment_feasibility.html', 'w').write(ef.build_html(results))
    EOF
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cart_colgen          import run_experiment          as det_run
from cart_colgen_ccp      import run_experiment_ccp      as ccp_run, DEFAULT_PROCESS_CCP
from cart_colgen_twostage import run_experiment_twostage as ts_run,  DEFAULT_PROCESS_2S
from cart_colgen_flexwait import run_experiment_flexwait as fw_run

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# Time limit scales with N
def time_limit_for(n):
    if n <= 75:  return 120
    if n <= 200: return 120
    if n <= 500: return 300   # 5 min — correctly scaled: N=250→1000/yr, N=500→2000/yr
    return 60   # large N (>500 per trimester) either infeasible fast or not worth longer

N_VALUES     = [5, 10, 15, 20, 25, 30, 50, 75, 100, 200, 250, 500, 1000, 2000]
SIGMA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CHECKPOINT   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'experiment_feasibility_checkpoint.json')

# ── helpers ──────────────────────────────────────────────────────────────────

def dat_file(n):
    return os.path.join(BASE_DIR, f'Data_N{n}.dat')

def run_one(solver, n, sigma):
    """Run a single solver/N/sigma combination. Returns status dict."""
    df  = dat_file(n)
    tlim = time_limit_for(n)
    if not os.path.exists(df):
        return {'status': 'missing', 'cost': None, 'time': None, 'plans': None}

    t0 = time.time()
    try:
        if solver == 'det':
            r = det_run(data_file=df, time_limit=tlim)
        elif solver == 'ccp':
            cfg = dict(DEFAULT_PROCESS_CCP); cfg['sigma_L'] = sigma
            r = ccp_run(data_file=df, process_config=cfg, time_limit=tlim)
        elif solver == 'ts':
            cfg = dict(DEFAULT_PROCESS_2S); cfg['sigma_L'] = sigma
            r = ts_run(data_file=df, process_config=cfg, time_limit=tlim)
        else:  # fw — sigma-invariant like det
            r = fw_run(data_file=df, time_limit=tlim)
        elapsed = time.time() - t0

        if not r.get('solved'):
            # Check if it timed out or was infeasible
            status = 'timeout' if elapsed >= tlim - 2 else 'infeasible'
            return {'status': status, 'cost': None, 'time': round(elapsed, 1), 'plans': r.get('num_plans')}
        return {
            'status':  'solved',
            'cost':    round(r.get('total_cost', 0) / 1e6, 4),
            'time':    round(elapsed, 1),
            'plans':   r.get('num_plans'),
            'n_open':  len(r.get('facilities_open', [])),
            'facilities': r.get('facilities_open', []),
        }
    except Exception as e:
        elapsed = time.time() - t0
        status = 'timeout' if elapsed >= tlim - 2 else 'error'
        return {'status': status, 'cost': None, 'time': round(elapsed, 1), 'plans': None, 'err': str(e)[:80]}

# ── main sweep ────────────────────────────────────────────────────────────────

def run_sweep():
    # Load checkpoint if it exists (resume interrupted run)
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            raw = json.load(f)
        results = {s: {int(n): {float(sg): v for sg, v in sv.items()}
                       for n, sv in nv.items()}
                   for s, nv in raw.items()}
        print(f'Resuming from checkpoint ({CHECKPOINT})')
    else:
        results = {}

    total = len(N_VALUES) * len(SIGMA_VALUES) * 4
    done  = 0

    for solver in ['det', 'ccp', 'ts', 'fw']:
        if solver not in results:
            results[solver] = {}
        for n in N_VALUES:
            if n not in results[solver]:
                results[solver][n] = {}
            for sigma in SIGMA_VALUES:
                done += 1
                if sigma in results[solver][n]:
                    # Already computed — skip
                    r = results[solver][n][sigma]
                    print(f'[{done:3d}/{total}] {solver.upper():4s} N={n:4d} σ={sigma:.2f} '
                          f'... (cached) {r["status"].upper()}')
                    continue

                # DET and FW are sigma-invariant: reuse first sigma result for all others
                if solver in ('det', 'fw') and results[solver][n]:
                    first_sigma = next(iter(results[solver][n]))
                    r = results[solver][n][first_sigma]
                    results[solver][n][sigma] = r
                    print(f'[{done:3d}/{total}] {solver.upper():4s} N={n:4d} σ={sigma:.2f} '
                          f'... (reused) {r["status"].upper()}')
                    with open(CHECKPOINT, 'w') as f:
                        json.dump({s: {str(n2): {str(sg): v for sg, v in sv.items()}
                                       for n2, sv in nv.items()}
                                   for s, nv in results.items()}, f)
                    continue

                tag = f'[{done:3d}/{total}] {solver.upper():4s} N={n:4d} σ={sigma:.2f}'
                print(tag, '...', end=' ', flush=True)
                r = run_one(solver, n, sigma)
                results[solver][n][sigma] = r
                status_str = r['status'].upper()
                cost_str   = f"${r['cost']:.3f}M" if r['cost'] else '—'
                time_str   = f"{r['time']}s"       if r['time'] else '?'
                print(f"{status_str:12s} {cost_str:12s} {time_str}", flush=True)

                # Save checkpoint after every completed run
                with open(CHECKPOINT, 'w') as f:
                    json.dump({s: {str(n2): {str(sg): v for sg, v in sv.items()}
                                   for n2, sv in nv.items()}
                               for s, nv in results.items()}, f)

    return results

# ── HTML generation ───────────────────────────────────────────────────────────

STATUS_COLOR = {
    'solved':     '#22c55e',   # green
    'infeasible': '#ef4444',   # red
    'timeout':    '#f59e0b',   # amber
    'missing':    '#9ca3af',   # grey
    'error':      '#8b5cf6',   # purple
}
STATUS_LABEL = {
    'solved':     'Solved',
    'infeasible': 'Infeasible',
    'timeout':    'Timeout',
    'missing':    'No data',
    'error':      'Error',
}

def cell_html(r, show_cost=True):
    status = r['status']
    color  = STATUS_COLOR.get(status, '#9ca3af')
    label  = STATUS_LABEL.get(status, status)
    cost   = f'<br><span style="font-size:10px">${r["cost"]:.3f}M</span>' if (show_cost and r.get('cost')) else ''
    time_  = f'<br><span style="font-size:9px;opacity:0.8">{r["time"]}s</span>' if r.get('time') else ''
    return (f'<td style="background:{color};color:#fff;text-align:center;'
            f'padding:8px 4px;font-size:11px;font-weight:600;border:1px solid #fff">'
            f'{label}{cost}{time_}</td>')

def grid_table_html(solver_results, title, show_cost=True):
    rows = [f'<h3 style="margin:24px 0 8px;color:#1E3A5F">{title}</h3>']
    rows.append('<table style="border-collapse:collapse;font-family:monospace;width:100%">')
    # header
    rows.append('<tr><th style="padding:6px 10px;background:#1E3A5F;color:#fff">N \\ σ</th>')
    for s in SIGMA_VALUES:
        rows.append(f'<th style="padding:6px 8px;background:#1E3A5F;color:#fff;min-width:80px">{s:.2f}</th>')
    rows.append('</tr>')
    # rows
    for n in N_VALUES:
        rows.append(f'<tr><td style="padding:6px 10px;background:#f1f5f9;font-weight:700;border:1px solid #e2e8f0">N={n}</td>')
        for s in SIGMA_VALUES:
            r = solver_results.get(n, {}).get(s, {'status': 'missing', 'cost': None, 'time': None})
            rows.append(cell_html(r, show_cost=show_cost))
        rows.append('</tr>')
    rows.append('</table>')
    return '\n'.join(rows)

def legend_html():
    items = ''.join(
        f'<span style="display:inline-flex;align-items:center;margin-right:18px">'
        f'<span style="width:14px;height:14px;background:{c};border-radius:3px;margin-right:6px"></span>'
        f'{STATUS_LABEL[s]}</span>'
        for s, c in STATUS_COLOR.items() if s != 'error'
    )
    return f'<div style="margin:12px 0;font-size:12px">{items}</div>'

def build_html(results):
    det_grid = grid_table_html(results.get('det', {}), 'Deterministic ColGen (wait=0)',    show_cost=True)
    fw_grid  = grid_table_html(results.get('fw',  {}), 'Deterministic ColGen (flex wait)', show_cost=True)
    ccp_grid = grid_table_html(results.get('ccp', {}), 'CCP (α = 0.10)',                   show_cost=True)
    ts_grid  = grid_table_html(results.get('ts',  {}), 'Two-Stage Stochastic (δ = 2.0)',   show_cost=True)

    # Summary counts
    def counts(solver_res):
        flat = [r for nr in solver_res.values() for r in nr.values()]
        return {s: sum(1 for r in flat if r['status'] == s) for s in STATUS_COLOR}

    def count_row(label, solver_res):
        c = counts(solver_res)
        total = sum(c.values())
        solved_pct = round(100 * c.get('solved', 0) / max(total, 1))
        return (f'<tr><td style="padding:6px 12px;font-weight:600">{label}</td>'
                + ''.join(f'<td style="padding:6px 12px;text-align:center">{c.get(s,0)}</td>'
                          for s in ['solved', 'infeasible', 'timeout', 'missing'])
                + f'<td style="padding:6px 12px;text-align:center;font-weight:600">{solved_pct}%</td></tr>')

    summary_table = f'''
<table style="border-collapse:collapse;font-family:sans-serif;margin:16px 0;font-size:13px">
  <tr style="background:#1E3A5F;color:#fff">
    <th style="padding:8px 12px">Model</th>
    <th style="padding:8px 12px">Solved</th>
    <th style="padding:8px 12px">Infeasible</th>
    <th style="padding:8px 12px">Timeout</th>
    <th style="padding:8px 12px">Missing</th>
    <th style="padding:8px 12px">Success rate</th>
  </tr>
  {count_row('Deterministic (wait=0)', results.get('det', {}))}
  {count_row('Deterministic (flex wait)', results.get('fw', {}))}
  {count_row('CCP', results.get('ccp', {}))}
  {count_row('Two-Stage', results.get('ts', {}))}
</table>'''

    results_json = json.dumps({
        solver: {str(n): {str(s): v for s, v in sv.items()} for n, sv in nv.items()}
        for solver, nv in results.items()
    }, indent=2)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 1: Feasibility Phase Diagram</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px 32px; background: #f8fafc; color: #1e293b; }}
  h1   {{ color: #1E3A5F; margin-bottom: 4px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 28px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .subtitle {{ color: #64748b; font-size: 14px; margin-bottom: 20px; }}
  details summary {{ cursor:pointer; font-weight:600; color:#1E3A5F; font-size:13px; margin-top:16px }}
  pre {{ background:#f1f5f9; padding:12px; border-radius:6px; font-size:11px;
         overflow-x:auto; max-height:300px; }}
</style>
</head>
<body>
<h1>Experiment 1 — Feasibility Phase Diagram</h1>
<p class="subtitle">
  N × σ grid showing solver feasibility for each model.
  Time limit: ≤75→120s / ≤200→60s / &gt;200→45s per run. &nbsp;|&nbsp;
  N ∈ {{{', '.join(str(n) for n in N_VALUES)}}} &nbsp;|&nbsp;
  σ_L ∈ {{{', '.join(f'{s:.2f}' for s in SIGMA_VALUES)}}}
</p>

<div class="card">
  <h2 style="margin-top:0;color:#1E3A5F">Summary</h2>
  {summary_table}
  {legend_html()}
</div>

<div class="card">
  {det_grid}
  <p style="font-size:12px;color:#64748b;margin-top:8px">
    Deterministic model ignores σ — result is σ-invariant (same cell repeated across columns).
    wait=0 means each patient's manufacturing slot is rigidly fixed by arrival time.
  </p>
</div>

<div class="card">
  {fw_grid}
  <p style="font-size:12px;color:#64748b;margin-top:8px">
    Same deterministic model but patients may wait up to their full deadline slack before
    manufacturing begins. This gives the solver flexibility to spread load across time,
    resolving infeasibility caused by clustered arrivals.
  </p>
</div>

<div class="card">
  {ccp_grid}
  <p style="font-size:12px;color:#64748b;margin-top:8px">
    CCP tightens the manufacturing slot to the (1-α)-quantile of T_MF ~ LogNormal(μ, σ²).
    Higher σ → longer effective TMFE → tighter feasibility window → infeasibility at large N.
  </p>
</div>

<div class="card">
  {ts_grid}
  <p style="font-size:12px;color:#64748b;margin-top:8px">
    Two-Stage model uses expediting recourse. Higher σ increases expediting probability
    but may still reach infeasibility when no plan meets the deadline even with recourse.
  </p>
</div>

<div class="card">
  <details>
    <summary>Raw JSON results</summary>
    <pre>{results_json}</pre>
  </details>
</div>

</body>
</html>'''
    return html

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Feasibility Phase Diagram Sweep')
    print(f'N values   : {N_VALUES}')
    print(f'σ values   : {SIGMA_VALUES}')
    print(f'Time limits: ≤75→120s  ≤200→60s  >200→45s')
    print(f'Total runs : {len(N_VALUES) * len(SIGMA_VALUES) * 4}')
    print()

    results = run_sweep()

    out_path = os.path.join(BASE_DIR, 'experiment_feasibility.html')
    with open(out_path, 'w') as f:
        f.write(build_html(results))

    print(f'\nSaved → {out_path}')
