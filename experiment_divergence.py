"""
Experiment 2: Decision Divergence Analysis
==========================================
Tracks how first-stage decisions (facility open/close, patient assignments,
route choices) diverge from the deterministic baseline as σ_L increases.

Compares: Deterministic vs CCP vs Two-Stage
At: N ∈ {15, 30, 50}, σ_L ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}

Output: experiment_divergence.html
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cart_colgen          import run_experiment          as det_run
from cart_colgen_ccp      import run_experiment_ccp      as ccp_run, DEFAULT_PROCESS_CCP
from cart_colgen_twostage import run_experiment_twostage as ts_run,  DEFAULT_PROCESS_2S

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
N_VALUES     = [15, 30, 50]
SIGMA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALL_FACILITIES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']

# ── helpers ───────────────────────────────────────────────────────────────────

def dat_file(n):
    return os.path.join(BASE_DIR, f'Data_N{n}.dat')

def decision_vec(result):
    """Extract (facility_set, {pid: (fac, j_out, j_ret)}) from a result."""
    if not result.get('solved'):
        return None, None
    fac_set  = set(result.get('facilities_open', []))
    pat_map  = {p['id']: (p['facility'], p['j_out'], p['j_ret'])
                for p in result.get('patients', [])}
    return fac_set, pat_map

def divergence(det_fac, det_pat, cmp_fac, cmp_pat):
    """Compute divergence metrics between deterministic and a comparison model."""
    if cmp_fac is None:
        return None

    # Facility divergence
    all_fac  = set(ALL_FACILITIES)
    det_open = det_fac & all_fac
    cmp_open = cmp_fac & all_fac
    fac_agree  = sum(1 for f in all_fac if (f in det_open) == (f in cmp_open))
    fac_div_pct = round(100 * (1 - fac_agree / len(all_fac)), 1)
    jaccard = (len(det_open & cmp_open) / len(det_open | cmp_open)
               if det_open | cmp_open else 1.0)

    # Patient-level divergence (only patients present in both)
    common = set(det_pat) & set(cmp_pat)
    fac_switch  = sum(1 for pid in common if det_pat[pid][0] != cmp_pat[pid][0])
    route_switch = sum(1 for pid in common
                       if det_pat[pid][1:] != cmp_pat[pid][1:])
    full_switch  = sum(1 for pid in common if det_pat[pid] != cmp_pat[pid])
    n = len(common)

    return {
        'fac_div_pct':   fac_div_pct,
        'jaccard':       round(jaccard, 3),
        'fac_switch_pct':   round(100 * fac_switch  / n, 1) if n else 0,
        'route_switch_pct': round(100 * route_switch / n, 1) if n else 0,
        'full_switch_pct':  round(100 * full_switch  / n, 1) if n else 0,
        'det_open':  sorted(det_fac),
        'cmp_open':  sorted(cmp_fac),
        'n_patients': n,
    }

# ── sweep ─────────────────────────────────────────────────────────────────────

def run_sweep():
    data = {}  # data[n][sigma] = {'det': ..., 'ccp': ..., 'ts': ...}

    for n in N_VALUES:
        data[n] = {}
        df = dat_file(n)

        # Deterministic — run once (σ-invariant)
        print(f'\n[N={n}] Running deterministic...', end=' ', flush=True)
        det_r = det_run(data_file=df, time_limit=120)
        det_fac, det_pat = decision_vec(det_r)
        print(f"{'OK' if det_r.get('solved') else 'FAIL'}  "
              f"facilities={sorted(det_fac or [])}  "
              f"cost=${det_r.get('total_cost',0)/1e6:.3f}M")

        for sigma in SIGMA_VALUES:
            print(f'  σ={sigma:.2f}', end='  ', flush=True)
            row = {'det_fac': sorted(det_fac or []),
                   'det_cost': round(det_r.get('total_cost', 0) / 1e6, 4)}

            # CCP
            cfg = dict(DEFAULT_PROCESS_CCP); cfg['sigma_L'] = sigma
            ccp_r  = ccp_run(data_file=df, process_config=cfg, time_limit=120)
            ccp_fac, ccp_pat = decision_vec(ccp_r)
            row['ccp'] = divergence(det_fac, det_pat, ccp_fac, ccp_pat)
            row['ccp_cost']   = round(ccp_r.get('total_cost', 0) / 1e6, 4) if ccp_r.get('solved') else None
            row['ccp_solved'] = ccp_r.get('solved', False)
            row['ccp_fac']    = sorted(ccp_fac) if ccp_fac else None

            # Two-Stage
            cfg2 = dict(DEFAULT_PROCESS_2S); cfg2['sigma_L'] = sigma
            ts_r   = ts_run(data_file=df, process_config=cfg2, time_limit=120)
            ts_fac, ts_pat = decision_vec(ts_r)
            row['ts']  = divergence(det_fac, det_pat, ts_fac, ts_pat)
            row['ts_cost']   = round(ts_r.get('total_cost', 0) / 1e6, 4) if ts_r.get('solved') else None
            row['ts_solved'] = ts_r.get('solved', False)
            row['ts_fac']    = sorted(ts_fac) if ts_fac else None

            data[n][sigma] = row

            ccp_tag = f"CCP={'OK' if row['ccp_solved'] else 'INF'} fac={row['ccp_fac']}"
            ts_tag  = f"TS={'OK'  if row['ts_solved']  else 'INF'} fac={row['ts_fac']}"
            print(f"{ccp_tag}   {ts_tag}")

    return data

# ── HTML ──────────────────────────────────────────────────────────────────────

PALETTE = {'det': '#475569', 'ccp': '#3A7D8C', 'ts': '#2C3E6B'}

def pct_cell(val, warn=20, danger=40):
    if val is None:
        return '<td style="text-align:center;color:#9ca3af;font-style:italic">n/a</td>'
    bg = ('#fee2e2' if val >= danger else '#fef9c3' if val >= warn else '#f0fdf4')
    fg = ('#b91c1c' if val >= danger else '#854d0e' if val >= warn else '#166534')
    return (f'<td style="text-align:center;background:{bg};color:{fg};'
            f'font-weight:600;padding:6px 10px">{val:.1f}%</td>')

def fac_badges(facs, det_facs):
    if facs is None:
        return '<td style="color:#9ca3af;text-align:center;font-style:italic">n/a</td>'
    det_set = set(det_facs)
    cmp_set = set(facs)
    badges = []
    for f in ALL_FACILITIES:
        in_det = f in det_set
        in_cmp = f in cmp_set
        if in_det and in_cmp:
            badges.append(f'<span style="background:#2C3E6B;color:#fff;border-radius:4px;'
                          f'padding:2px 5px;font-size:10px;margin:1px">{f}</span>')
        elif in_cmp and not in_det:
            badges.append(f'<span style="background:#ef4444;color:#fff;border-radius:4px;'
                          f'padding:2px 5px;font-size:10px;margin:1px">+{f}</span>')
        elif in_det and not in_cmp:
            badges.append(f'<span style="background:#f59e0b;color:#fff;border-radius:4px;'
                          f'padding:2px 5px;font-size:10px;margin:1px">−{f}</span>')
    return '<td style="padding:4px 8px">' + ' '.join(badges) + '</td>'

def build_table(n, n_data):
    rows = []
    header = '''<tr style="background:#1E3A5F;color:#fff">
      <th style="padding:8px 12px">σ_L</th>
      <th style="padding:8px 12px">Model</th>
      <th style="padding:8px 12px">Facilities open</th>
      <th style="padding:8px 12px">Facility<br>divergence</th>
      <th style="padding:8px 12px">Fac. assignment<br>switch</th>
      <th style="padding:8px 12px">Route<br>switch</th>
      <th style="padding:8px 12px">Any decision<br>switch</th>
      <th style="padding:8px 12px">Cost ($M)</th>
    </tr>'''
    rows.append(header)

    for sigma in SIGMA_VALUES:
        row = n_data.get(sigma, {})
        det_facs = row.get('det_fac', [])

        for model in ['ccp', 'ts']:
            label   = 'CCP (α=0.10)' if model == 'ccp' else 'Two-Stage (δ=2)'
            solved  = row.get(f'{model}_solved', False)
            div     = row.get(model)
            cost    = row.get(f'{model}_cost')
            facs    = row.get(f'{model}_fac')
            color   = PALETTE[model]

            sigma_cell = (f'<td rowspan="2" style="text-align:center;font-weight:700;'
                          f'font-size:13px;padding:8px;background:#f8fafc;'
                          f'border-top:2px solid #e2e8f0">{sigma:.2f}</td>'
                          if model == 'ccp' else '')
            border = 'border-top:2px solid #e2e8f0' if model == 'ccp' else ''

            if not solved:
                rows.append(
                    f'<tr style="{border}">{sigma_cell}'
                    f'<td style="color:{color};font-weight:600;padding:6px 10px">{label}</td>'
                    f'<td colspan="5" style="color:#9ca3af;text-align:center;font-style:italic">Infeasible</td>'
                    f'<td style="text-align:center;color:#9ca3af">—</td></tr>')
            else:
                rows.append(
                    f'<tr style="{border}">{sigma_cell}'
                    f'<td style="color:{color};font-weight:600;padding:6px 10px">{label}</td>'
                    + fac_badges(facs, det_facs)
                    + pct_cell(div['fac_div_pct']      if div else None)
                    + pct_cell(div['fac_switch_pct']   if div else None)
                    + pct_cell(div['route_switch_pct'] if div else None)
                    + pct_cell(div['full_switch_pct']  if div else None)
                    + f'<td style="text-align:center;padding:6px 10px">'
                    + (f'${cost:.3f}M' if cost else '—') + '</td></tr>')

    return f'''
<h3 style="color:#1E3A5F;margin:28px 0 8px">N = {n} patients
  <span style="font-size:12px;font-weight:400;color:#64748b;margin-left:8px">
    Det. baseline: facilities = {n_data.get(SIGMA_VALUES[0], {}).get("det_fac", [])}
    &nbsp;|&nbsp; cost = ${n_data.get(SIGMA_VALUES[0], {}).get("det_cost", 0):.3f}M
  </span>
</h3>
<table style="border-collapse:collapse;width:100%;font-size:12px;font-family:sans-serif">
  {''.join(rows)}
</table>'''

def build_chart_data(data):
    """Build Chart.js datasets for facility-switch % vs sigma, per N."""
    datasets = []
    colors_ccp = ['#3A7D8C', '#2E6470', '#22505A']
    colors_ts  = ['#2C3E6B', '#1E2D50', '#111C35']
    for i, n in enumerate(N_VALUES):
        ccp_pts = [data[n][s]['ccp']['fac_switch_pct']
                   if data[n][s].get('ccp_solved') and data[n][s].get('ccp') else None
                   for s in SIGMA_VALUES]
        ts_pts  = [data[n][s]['ts']['fac_switch_pct']
                   if data[n][s].get('ts_solved')  and data[n][s].get('ts')  else None
                   for s in SIGMA_VALUES]
        datasets.append({'label': f'CCP N={n}', 'data': ccp_pts,
                         'borderColor': colors_ccp[i], 'backgroundColor': colors_ccp[i],
                         'borderDash': [6, 3], 'pointStyle': 'circle'})
        datasets.append({'label': f'2S N={n}',  'data': ts_pts,
                         'borderColor': colors_ts[i],  'backgroundColor': colors_ts[i],
                         'pointStyle': 'rectRot'})
    return json.dumps(datasets)

def build_html(data):
    tables = '\n'.join(build_table(n, data[n]) for n in N_VALUES)
    chart_datasets = build_chart_data(data)
    sigma_labels   = json.dumps(SIGMA_VALUES)
    results_json   = json.dumps(
        {str(n): {str(s): v for s, v in nd.items()} for n, nd in data.items()}, indent=2)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 2: Decision Divergence</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body  {{ font-family:system-ui,sans-serif; margin:0; padding:24px 32px;
           background:#f8fafc; color:#1e293b; }}
  h1    {{ color:#1E3A5F; margin-bottom:4px; }}
  .card {{ background:#fff; border-radius:10px; padding:24px; margin-bottom:28px;
           box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .subtitle {{ color:#64748b; font-size:14px; margin-bottom:20px; }}
  .legend-dot {{ display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px }}
  details summary {{ cursor:pointer;font-weight:600;color:#1E3A5F;font-size:13px;margin-top:16px }}
  pre {{ background:#f1f5f9;padding:12px;border-radius:6px;font-size:11px;
         overflow-x:auto;max-height:300px }}
  .legend-box {{ font-size:12px; margin:12px 0 }}
</style>
</head>
<body>
<h1>Experiment 2 — Decision Divergence Analysis</h1>
<p class="subtitle">
  How much do CCP and Two-Stage first-stage decisions differ from the deterministic baseline as σ_L increases?<br>
  <strong>Facility switch</strong>: patient assigned to a different facility than deterministic.
  <strong>Route switch</strong>: different (j_out, j_ret) pair.
  <strong>Any switch</strong>: either facility or route changed.
</p>

<div class="card">
  <h2 style="margin-top:0;color:#1E3A5F">Facility Assignment Switch vs σ_L</h2>
  <p style="font-size:12px;color:#64748b">% of patients assigned to a different facility than the deterministic solution</p>
  <div style="position:relative;height:320px">
    <canvas id="divChart"></canvas>
  </div>
  <div class="legend-box">
    <span style="border-bottom:2px dashed #3A7D8C;padding-bottom:1px;margin-right:4px">&nbsp;&nbsp;&nbsp;</span> CCP &nbsp;&nbsp;
    <span style="border-bottom:2px solid #2C3E6B;padding-bottom:1px;margin-right:4px">&nbsp;&nbsp;&nbsp;</span> Two-Stage &nbsp;&nbsp;
    Shading: N=15 (light) → N=50 (dark)
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0;color:#1E3A5F">Detailed Decision Tables</h2>
  <p style="font-size:12px;color:#64748b;margin-bottom:16px">
    Facility badges: <span style="background:#2C3E6B;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px">m1</span> = open in both &nbsp;
    <span style="background:#ef4444;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px">+m4</span> = added vs det &nbsp;
    <span style="background:#f59e0b;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px">−m1</span> = removed vs det &nbsp;&nbsp;
    Cell shading: <span style="background:#f0fdf4;padding:1px 5px;border-radius:3px">low</span>
    <span style="background:#fef9c3;padding:1px 5px;border-radius:3px">≥20%</span>
    <span style="background:#fee2e2;padding:1px 5px;border-radius:3px">≥40%</span>
  </p>
  {tables}
</div>

<div class="card">
  <details>
    <summary>Raw JSON results</summary>
    <pre>{results_json}</pre>
  </details>
</div>

<script>
const ctx = document.getElementById('divChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {sigma_labels},
    datasets: {chart_datasets}
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
      tooltip: {{ callbacks: {{
        label: ctx => ctx.dataset.label + ': ' + (ctx.raw !== null ? ctx.raw.toFixed(1) + '%' : 'n/a')
      }} }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'σ_L (manufacturing uncertainty)' }} }},
      y: {{ title: {{ display: true, text: '% patients with facility switch' }},
            min: 0, max: 100,
            ticks: {{ callback: v => v + '%' }} }}
    }}
  }}
}});
</script>
</body>
</html>'''

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Decision Divergence Analysis')
    print(f'N values : {N_VALUES}')
    print(f'σ values : {SIGMA_VALUES}')
    print(f'Runs     : {len(N_VALUES) * len(SIGMA_VALUES) * 2 + len(N_VALUES)} '
          f'(det×N + ccp×grid + ts×grid)')

    data = run_sweep()

    out = os.path.join(BASE_DIR, 'experiment_divergence.html')
    with open(out, 'w') as f:
        f.write(build_html(data))
    print(f'\nSaved → {out}')
