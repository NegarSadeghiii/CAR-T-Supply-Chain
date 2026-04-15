"""
Experiment 4: Cost-Reliability Tradeoff Curve
===============================================
x-axis: OOS violation rate (% scenarios with ≥1 deadline miss)
y-axis: normalized first-stage cost (cost / deterministic baseline)

Sweeps:
  - CCP:        vary α ∈ {0.02, 0.05, 0.10, 0.15, 0.20}  at N=15, σ=0.20
  - Two-Stage:  vary δ ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}  at N=15, σ=0.20
  - σ curve:    vary σ ∈ {0.05,0.10,0.15,0.20,0.25,0.30}  at N=15 for TS (δ=2)
                vary σ ∈ {0.05,0.10,0.15,0.20}            at N=15 for CCP (α=0.10)

OOS: n_sim=2000 held-out scenarios, seed=77
Output: experiment_tradeoff.html
"""
import os, sys, math, random, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cart_colgen          import run_experiment          as det_run
from cart_colgen_ccp      import run_experiment_ccp      as ccp_run, DEFAULT_PROCESS_CCP
from cart_colgen_twostage import run_experiment_twostage as ts_run,  DEFAULT_PROCESS_2S

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N        = 15
N_SIM    = 2000
SEED     = 77
DAT      = os.path.join(BASE_DIR, f'Data_N{N}.dat')

ALPHA_VALUES = [0.02, 0.05, 0.10, 0.15, 0.20]
DELTA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SIGMA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SIGMA_CCP    = [0.05, 0.10, 0.15, 0.20]   # CCP infeasible beyond 0.20

# ── OOS validation ────────────────────────────────────────────────────────────

def oos_det(det_r, tmfe, sigma, n_sim=N_SIM, seed=SEED):
    """Deterministic has no recourse: violation if T_MF > mfg_budget."""
    rng   = random.Random(seed)
    mu_L  = math.log(tmfe) - 0.5 * sigma ** 2
    pats  = det_r.get('patients', [])
    viols = []
    for _ in range(n_sim):
        nv = 0
        for p in pats:
            # budget = how long manufacturing can take before deadline missed
            budget = p['deadline'] - (p['trt'] - tmfe)
            if rng.lognormvariate(mu_L, sigma) > budget:
                nv += 1
        viols.append(nv)
    pct_viol = round(sum(1 for v in viols if v > 0) / n_sim * 100, 1)
    return pct_viol

def oos_ccp(ccp_r, tmfe, sigma, n_sim=N_SIM, seed=SEED):
    """CCP has no recourse: violation if T_MF > extended budget."""
    rng      = random.Random(seed)
    mu_L     = math.log(tmfe) - 0.5 * sigma ** 2
    pats     = ccp_r.get('patients', [])
    tmfe_eff = ccp_r.get('tmfe_eff', tmfe)
    budgets  = {p['id']: p['deadline'] - (p['trt'] - tmfe_eff) for p in pats}
    viols    = []
    for _ in range(n_sim):
        nv = 0
        for p in pats:
            if rng.lognormvariate(mu_L, sigma) > budgets.get(p['id'], 999):
                nv += 1
        viols.append(nv)
    return round(sum(1 for v in viols if v > 0) / n_sim * 100, 1)

def oos_ts(ts_r, tmfe, sigma, delta, pi_map, n_sim=N_SIM, seed=SEED):
    """Two-Stage: expedite if T_MF > K; violate if T_MF > K + delta."""
    rng   = random.Random(seed)
    mu_L  = math.log(tmfe) - 0.5 * sigma ** 2
    pats  = ts_r.get('patients', [])
    ts_base_cost = ts_r.get('total_cost', 0) - ts_r.get('total_exp_expedite_cost', 0)
    costs, viols = [], []
    for _ in range(n_sim):
        ec, nv = 0, 0
        for p in pats:
            K    = p.get('slack')
            if K is None: continue
            tmf  = rng.lognormvariate(mu_L, sigma)
            pi_p = pi_map.get(p.get('group', 'low'), 1500)
            if tmf > K:
                ec += pi_p
                if tmf > K + delta:
                    nv += 1
        costs.append(ts_base_cost + ec)
        viols.append(nv)
    pct_viol  = round(sum(1 for v in viols if v > 0) / n_sim * 100, 1)
    mean_cost = round(statistics.mean(costs) / 1e6, 4)
    return pct_viol, mean_cost

# ── sweep ─────────────────────────────────────────────────────────────────────

def run_sweep():
    tmfe = DEFAULT_PROCESS_2S['tmfe']

    # ── Deterministic baseline ────────────────────────────────────────────────
    print('Running deterministic baseline...')
    det_r     = det_run(data_file=DAT, time_limit=120)
    det_cost  = det_r.get('total_cost', 0) / 1e6
    det_viols = {s: oos_det(det_r, tmfe, s) for s in SIGMA_VALUES}
    print(f'  cost=${det_cost:.3f}M  '
          + '  '.join(f'σ={s:.2f}→{v:.1f}%' for s, v in det_viols.items()))

    # ── CCP: vary α ───────────────────────────────────────────────────────────
    print('\nCCP α sweep (σ=0.20):')
    sigma_ref  = 0.20
    ccp_alpha  = []
    for alpha in ALPHA_VALUES:
        cfg = dict(DEFAULT_PROCESS_CCP); cfg['sigma_L'] = sigma_ref; cfg['alpha'] = alpha
        r   = ccp_run(data_file=DAT, process_config=cfg, time_limit=120)
        if not r.get('solved'):
            print(f'  α={alpha:.2f}  INFEASIBLE')
            continue
        viol      = oos_ccp(r, tmfe, sigma_ref)
        cost_norm = round(r['total_cost'] / 1e6 / det_cost, 4)
        ccp_alpha.append({'alpha': alpha, 'sigma': sigma_ref,
                          'cost_norm': cost_norm, 'cost_M': round(r['total_cost']/1e6, 4),
                          'viol_pct': viol, 'tmfe_eff': round(r.get('tmfe_eff', 0), 3)})
        print(f'  α={alpha:.2f}  viol={viol:.1f}%  cost_norm={cost_norm:.4f}  tmfe_eff={r.get("tmfe_eff",0):.2f}')

    # ── CCP: vary σ ───────────────────────────────────────────────────────────
    print('\nCCP σ sweep (α=0.10):')
    alpha_ref = 0.10
    ccp_sigma = []
    for sigma in SIGMA_CCP:
        cfg = dict(DEFAULT_PROCESS_CCP); cfg['sigma_L'] = sigma; cfg['alpha'] = alpha_ref
        r   = ccp_run(data_file=DAT, process_config=cfg, time_limit=120)
        if not r.get('solved'):
            print(f'  σ={sigma:.2f}  INFEASIBLE'); continue
        viol      = oos_ccp(r, tmfe, sigma)
        cost_norm = round(r['total_cost'] / 1e6 / det_cost, 4)
        ccp_sigma.append({'alpha': alpha_ref, 'sigma': sigma,
                          'cost_norm': cost_norm, 'cost_M': round(r['total_cost']/1e6, 4),
                          'viol_pct': viol})
        print(f'  σ={sigma:.2f}  viol={viol:.1f}%  cost_norm={cost_norm:.4f}')

    # ── Two-Stage: vary δ ─────────────────────────────────────────────────────
    print('\nTwo-Stage δ sweep (σ=0.20):')
    ts_delta = []
    for delta in DELTA_VALUES:
        cfg = dict(DEFAULT_PROCESS_2S); cfg['sigma_L'] = sigma_ref; cfg['delta'] = delta
        r   = ts_run(data_file=DAT, process_config=cfg, time_limit=120)
        if not r.get('solved'):
            print(f'  δ={delta:.1f}  INFEASIBLE'); continue
        pi_map    = r.get('pi_map', {'high': 8000, 'medium': 4000, 'low': 1500})
        viol, oos_cost = oos_ts(r, tmfe, sigma_ref, delta, pi_map)
        cost_norm = round(r['total_cost'] / 1e6 / det_cost, 4)
        oos_norm  = round(oos_cost / det_cost, 4)
        ts_delta.append({'delta': delta, 'sigma': sigma_ref,
                         'cost_norm': cost_norm, 'oos_norm': oos_norm,
                         'cost_M': round(r['total_cost']/1e6, 4), 'oos_M': oos_cost,
                         'viol_pct': viol,
                         'avg_p_exp': round(r.get('avg_prob_expedite', 0)*100, 1)})
        print(f'  δ={delta:.1f}  viol={viol:.1f}%  cost_norm={cost_norm:.4f}  '
              f'oos_norm={oos_norm:.4f}  P(exp)={r.get("avg_prob_expedite",0)*100:.1f}%')

    # ── Two-Stage: vary σ ─────────────────────────────────────────────────────
    print('\nTwo-Stage σ sweep (δ=2.0):')
    delta_ref = 2.0
    ts_sigma  = []
    for sigma in SIGMA_VALUES:
        cfg = dict(DEFAULT_PROCESS_2S); cfg['sigma_L'] = sigma; cfg['delta'] = delta_ref
        r   = ts_run(data_file=DAT, process_config=cfg, time_limit=120)
        if not r.get('solved'):
            print(f'  σ={sigma:.2f}  INFEASIBLE'); continue
        pi_map    = r.get('pi_map', {'high': 8000, 'medium': 4000, 'low': 1500})
        viol, oos_cost = oos_ts(r, tmfe, sigma, delta_ref, pi_map)
        cost_norm = round(r['total_cost'] / 1e6 / det_cost, 4)
        oos_norm  = round(oos_cost / det_cost, 4)
        ts_sigma.append({'delta': delta_ref, 'sigma': sigma,
                         'cost_norm': cost_norm, 'oos_norm': oos_norm,
                         'cost_M': round(r['total_cost']/1e6, 4), 'oos_M': oos_cost,
                         'viol_pct': viol})
        print(f'  σ={sigma:.2f}  viol={viol:.1f}%  cost_norm={cost_norm:.4f}  oos_norm={oos_norm:.4f}')

    # deterministic sigma sweep for reference
    det_sigma_pts = [{'sigma': s, 'viol_pct': det_viols[s],
                      'cost_norm': 1.0} for s in SIGMA_VALUES]

    return {
        'det_cost_M':   round(det_cost, 4),
        'det_sigma':    det_sigma_pts,
        'ccp_alpha':    ccp_alpha,
        'ccp_sigma':    ccp_sigma,
        'ts_delta':     ts_delta,
        'ts_sigma':     ts_sigma,
    }

# ── HTML ──────────────────────────────────────────────────────────────────────

def build_html(data):
    det_cost_M = data['det_cost_M']

    # Chart 1: α sweep (CCP) + δ sweep (TS) — main tradeoff
    def pt(d, xk, yk, label_k=None):
        lbl = str(round(d.get(label_k, ''), 3)) if label_k else ''
        return {'x': d[xk], 'y': d[yk], 'label': lbl}

    ccp_alpha_pts = [pt(d, 'viol_pct', 'cost_norm', 'alpha') for d in data['ccp_alpha']]
    ts_delta_pts  = [pt(d, 'viol_pct', 'cost_norm', 'delta') for d in data['ts_delta']]
    det_pt        = [{'x': data['det_sigma'][3]['viol_pct'], 'y': 1.0, 'label': 'Det'}]

    # Chart 2: σ sweep — how uncertainty shifts the curve
    ccp_sigma_pts = [pt(d, 'viol_pct', 'cost_norm', 'sigma') for d in data['ccp_sigma']]
    ts_sigma_pts  = [pt(d, 'viol_pct', 'cost_norm', 'sigma') for d in data['ts_sigma']]

    def table_rows(rows, cols):
        out = []
        for r in rows:
            out.append('<tr>' + ''.join(
                f'<td style="padding:6px 12px;text-align:center">{r.get(c, "—")}</td>'
                for c in cols) + '</tr>')
        return ''.join(out)

    def th(headers):
        return ('<tr style="background:#1E3A5F;color:#fff">'
                + ''.join(f'<th style="padding:8px 12px">{h}</th>' for h in headers)
                + '</tr>')

    ccp_alpha_rows = table_rows(
        [{'α': d['alpha'], 'σ': d['sigma'], 'TMFE_eff': d['tmfe_eff'],
          'Cost ($M)': d['cost_M'], 'Cost / Det': d['cost_norm'],
          'OOS viol %': d['viol_pct']} for d in data['ccp_alpha']],
        ['α', 'σ', 'TMFE_eff', 'Cost ($M)', 'Cost / Det', 'OOS viol %'])

    ts_delta_rows = table_rows(
        [{'δ': d['delta'], 'σ': d['sigma'], 'P(exp)%': d['avg_p_exp'],
          'Cost ($M)': d['cost_M'], 'Cost / Det': d['cost_norm'],
          'OOS viol %': d['viol_pct']} for d in data['ts_delta']],
        ['δ', 'σ', 'P(exp)%', 'Cost ($M)', 'Cost / Det', 'OOS viol %'])

    ts_sigma_rows = table_rows(
        [{'σ': d['sigma'], 'Cost ($M)': d['cost_M'], 'Cost / Det': d['cost_norm'],
          'OOS viol %': d['viol_pct']} for d in data['ts_sigma']],
        ['σ', 'Cost ($M)', 'Cost / Det', 'OOS viol %'])

    raw_json = json.dumps(data, indent=2)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 4: Cost-Reliability Tradeoff</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body  {{ font-family:system-ui,sans-serif; margin:0; padding:24px 32px;
           background:#f8fafc; color:#1e293b; }}
  h1    {{ color:#1E3A5F; margin-bottom:4px; }}
  h2    {{ color:#1E3A5F; margin:0 0 10px; }}
  .card {{ background:#fff; border-radius:10px; padding:24px; margin-bottom:28px;
           box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .subtitle {{ color:#64748b; font-size:14px; margin-bottom:20px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  table  {{ border-collapse:collapse; width:100%; font-size:12px; }}
  details summary {{ cursor:pointer; font-weight:600; color:#1E3A5F; font-size:13px }}
  pre {{ background:#f1f5f9;padding:12px;border-radius:6px;font-size:11px;
         overflow-x:auto;max-height:300px }}
</style>
</head>
<body>
<h1>Experiment 4 — Cost-Reliability Tradeoff</h1>
<p class="subtitle">
  N = {N} patients &nbsp;|&nbsp; OOS: n_sim = {N_SIM} scenarios, seed = {SEED}
  &nbsp;|&nbsp; Deterministic baseline cost = ${det_cost_M:.3f}M
  <br>
  <strong>x-axis:</strong> % of scenarios with ≥1 deadline violation &nbsp;|&nbsp;
  <strong>y-axis:</strong> first-stage cost normalised by deterministic baseline
</p>

<div class="card">
  <h2>Tradeoff: Varying Model Parameters (σ = 0.20)</h2>
  <p style="font-size:12px;color:#64748b">
    CCP: varying α (miss-probability budget) &nbsp;|&nbsp;
    Two-Stage: varying δ (days saved by expediting) &nbsp;|&nbsp;
    Point labels show parameter value
  </p>
  <div style="position:relative;height:400px">
    <canvas id="chart1"></canvas>
  </div>
</div>

<div class="card">
  <h2>Tradeoff: Varying Uncertainty Level σ_L</h2>
  <p style="font-size:12px;color:#64748b">
    CCP (α=0.10) and Two-Stage (δ=2.0) at fixed parameters, as σ increases.
    Shows how the feasible operating region shifts under higher uncertainty.
  </p>
  <div style="position:relative;height:360px">
    <canvas id="chart2"></canvas>
  </div>
</div>

<div class="card">
  <h2>Detailed Results</h2>
  <div class="grid2">
    <div>
      <h3 style="color:#3A7D8C;margin:0 0 8px">CCP — α sweep (σ=0.20)</h3>
      <table>
        {th(['α', 'σ', 'TMFE_eff', 'Cost ($M)', 'Cost / Det', 'OOS viol %'])}
        {ccp_alpha_rows}
      </table>
    </div>
    <div>
      <h3 style="color:#2C3E6B;margin:0 0 8px">Two-Stage — δ sweep (σ=0.20)</h3>
      <table>
        {th(['δ', 'σ', 'P(exp)%', 'Cost ($M)', 'Cost / Det', 'OOS viol %'])}
        {ts_delta_rows}
      </table>
    </div>
  </div>
  <div style="margin-top:20px">
    <h3 style="color:#2C3E6B;margin:0 0 8px">Two-Stage — σ sweep (δ=2.0)</h3>
    <table style="max-width:500px">
      {th(['σ', 'Cost ($M)', 'Cost / Det', 'OOS viol %'])}
      {ts_sigma_rows}
    </table>
  </div>
</div>

<div class="card">
  <details>
    <summary>Raw JSON results</summary>
    <pre>{raw_json}</pre>
  </details>
</div>

<script>
// ── Chart 1: parameter sweep ──────────────────────────────────────────────────
const ccp_alpha_pts = {json.dumps(ccp_alpha_pts)};
const ts_delta_pts  = {json.dumps(ts_delta_pts)};
const det_pt        = {json.dumps(det_pt)};

function labelPlugin(id) {{
  return {{
    id,
    afterDatasetsDraw(chart) {{
      const {{ctx}} = chart;
      ctx.save();
      chart.data.datasets.forEach((ds, di) => {{
        chart.getDatasetMeta(di).data.forEach((el, i) => {{
          const pt = ds.data[i];
          if (!pt || pt.label == null) return;
          ctx.font = 'bold 10px sans-serif';
          ctx.fillStyle = ds.borderColor;
          ctx.textAlign = 'left';
          ctx.fillText(pt.label, el.x + 6, el.y - 4);
        }});
      }});
      ctx.restore();
    }}
  }};
}}

new Chart(document.getElementById('chart1').getContext('2d'), {{
  type: 'scatter',
  data: {{
    datasets: [
      {{ label: 'CCP (vary α)', data: ccp_alpha_pts,
         borderColor: '#3A7D8C', backgroundColor: '#3A7D8C',
         pointRadius: 7, pointStyle: 'circle' }},
      {{ label: 'Two-Stage (vary δ)', data: ts_delta_pts,
         borderColor: '#2C3E6B', backgroundColor: '#2C3E6B',
         pointRadius: 7, pointStyle: 'rectRot' }},
      {{ label: 'Deterministic', data: det_pt,
         borderColor: '#475569', backgroundColor: '#475569',
         pointRadius: 9, pointStyle: 'triangle' }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{ callbacks: {{
        label: ctx => {{
          const p = ctx.raw;
          return `${{ctx.dataset.label}} — viol: ${{p.x}}%  cost: ${{p.y.toFixed(3)}}x`;
        }}
      }} }}
    }},
    scales: {{
      x: {{ title: {{ display:true, text:'OOS violation rate (% scenarios with ≥1 miss)' }},
            min: 0 }},
      y: {{ title: {{ display:true, text:'Normalised cost (÷ deterministic baseline)' }},
            ticks: {{ callback: v => v.toFixed(2) + 'x' }} }}
    }}
  }},
  plugins: [labelPlugin('labels1')]
}});

// ── Chart 2: σ sweep ──────────────────────────────────────────────────────────
const ccp_sigma_pts = {json.dumps(ccp_sigma_pts)};
const ts_sigma_pts  = {json.dumps(ts_sigma_pts)};

new Chart(document.getElementById('chart2').getContext('2d'), {{
  type: 'scatter',
  data: {{
    datasets: [
      {{ label: 'CCP α=0.10 (vary σ)', data: ccp_sigma_pts,
         borderColor: '#3A7D8C', backgroundColor: '#3A7D8C',
         pointRadius: 7, pointStyle: 'circle',
         showLine: true, tension: 0.3 }},
      {{ label: 'Two-Stage δ=2.0 (vary σ)', data: ts_sigma_pts,
         borderColor: '#2C3E6B', backgroundColor: '#2C3E6B',
         pointRadius: 7, pointStyle: 'rectRot',
         showLine: true, tension: 0.3 }},
      {{ label: 'Deterministic (vary σ)',
         data: {json.dumps([{'x': d['viol_pct'], 'y': 1.0, 'label': str(d['sigma'])} for d in data['det_sigma']])},
         borderColor: '#475569', backgroundColor: '#475569',
         pointRadius: 6, pointStyle: 'triangle',
         showLine: true, tension: 0.3 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{ callbacks: {{
        label: ctx => {{
          const p = ctx.raw;
          return `${{ctx.dataset.label}} — σ=${{p.label}}  viol: ${{p.x}}%  cost: ${{p.y?.toFixed(3)}}x`;
        }}
      }} }}
    }},
    scales: {{
      x: {{ title: {{ display:true, text:'OOS violation rate (%)' }}, min: 0 }},
      y: {{ title: {{ display:true, text:'Normalised cost' }},
            ticks: {{ callback: v => v.toFixed(2) + 'x' }} }}
    }}
  }},
  plugins: [labelPlugin('labels2')]
}});
</script>
</body>
</html>'''

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Cost-Reliability Tradeoff  N={N}  n_sim={N_SIM}')
    print(f'CCP α sweep  : {ALPHA_VALUES}')
    print(f'TS  δ sweep  : {DELTA_VALUES}')
    print(f'σ sweep      : {SIGMA_VALUES}')

    data = run_sweep()

    out = os.path.join(BASE_DIR, 'experiment_tradeoff.html')
    with open(out, 'w') as f:
        f.write(build_html(data))
    print(f'\nSaved → {out}')
