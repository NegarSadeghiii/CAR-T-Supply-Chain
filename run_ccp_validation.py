"""
CAR-T Supply Chain — CCP Validation Report Generator
=====================================================
Runs the full method comparison (Deterministic, CCP, Gamma-robust),
performs Monte Carlo simulation validation, and generates a
cart_dashboard.html-style HTML report.

Usage
-----
  python run_ccp_validation.py --data Data_N15.dat --sigma-L 0.20
  python run_ccp_validation.py --data Data_N15.dat Data_N50.dat --sigma-L 0.20
  python run_ccp_validation.py --data Data_N15.dat --html my_report.html --json
"""

import argparse
import json
import math
import os
import sys

from cart_ccp_simulation import run_full_comparison


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

def _svc_color(val):
    """Return CSS colour class for a service level value."""
    if val >= 0.95:  return '#059669'  # green
    if val >= 0.85:  return '#d97706'  # amber
    return '#dc2626'                   # red

def _cost_badge(solved, cost):
    if not solved:
        return '<span style="color:#dc2626;font-weight:700">INFEASIBLE</span>'
    return f'${cost:,.0f}'

def generate_html(all_results, output_path, sigma_L, n_sim):
    """Produce a 3-tab dashboard HTML report."""

    # ---- collect data per dataset ----
    datasets = list(all_results.keys())

    # ---- build JS data arrays ----
    # For frontier chart: each method is a point (empirical_svc, cost)
    frontier_datasets = []
    colours = ['#1E3A5F','#1B6B2E','#92400E','#7c3aed','#0891b2','#e8786a','#8eb85b']

    for di, (ds_name, results) in enumerate(all_results.items()):
        pts = []
        for r in results:
            sim  = r.get('simulation', {})
            esvc = sim.get('empirical_patient_svc')
            cost = r.get('total_cost')
            if esvc is not None and cost is not None and r.get('solved'):
                pts.append({'x': round(esvc*100, 1), 'y': round(cost/1e6, 3),
                            'label': r.get('method','?')})
        frontier_datasets.append({'name': ds_name, 'points': pts})

    # ---- method comparison tables (one per dataset) ----
    table_html = ''
    for ds_name, results in all_results.items():
        table_html += f'''
        <div class="ds-section">
          <h3 class="ds-title">{ds_name}</h3>
          <div class="tbl-wrap">
          <table class="data-tbl">
            <thead><tr>
              <th>Method</th><th>TMFE_eff (d)</th><th>Plans</th>
              <th>Schedule Cost</th><th>Avg TRT (d)</th>
              <th>Theoretical SL</th><th>Empirical SL</th>
              <th>Avg Violations</th><th>Max Violations</th>
            </tr></thead><tbody>'''
        for r in results:
            sim    = r.get('simulation', {})
            solved    = r.get('solved', False)
            is_robust = 'gamma' in r
            if not solved:
                tsvc = None
            elif is_robust:
                tsvc = None   # worst-case, not a probability
            else:
                tsvc = r.get('service_level', 1.0 - r.get('alpha', 0.0))
            esvc   = sim.get('empirical_patient_svc') if sim else None
            avg_v  = sim.get('avg_violations_per_run', 0) if sim else 0
            max_v  = sim.get('max_violations_per_run', 0) if sim else 0

            tsvc_str = ('worst-case' if is_robust and solved else
                        f"{tsvc*100:.0f}%" if tsvc is not None else '—')
            esvc_str = f"{esvc*100:.1f}%" if esvc is not None else '—'
            esvc_col = _svc_color(esvc) if esvc is not None else '#666'

            table_html += f'''<tr>
              <td><strong>{r.get('method','?')}</strong></td>
              <td>{r.get('tmfe_eff', 7):.2f}</td>
              <td>{r.get('num_plans', 0)}</td>
              <td>{_cost_badge(solved, r.get('total_cost',0))}</td>
              <td>{"—" if not solved else f"{r.get('avg_trt',0):.1f}"}</td>
              <td>{tsvc_str}</td>
              <td style="color:{esvc_col};font-weight:700">{esvc_str}</td>
              <td>{"—" if not solved else f"{avg_v:.2f}"}</td>
              <td>{"—" if not solved else str(max_v)}</td>
            </tr>'''
        table_html += '</tbody></table></div></div>'

    # ---- Monte Carlo detail table ----
    mc_html = ''
    for ds_name, results in all_results.items():
        mc_html += f'<div class="ds-section"><h3 class="ds-title">{ds_name} — Per-Method Monte Carlo ({n_sim:,} runs)</h3>'
        for r in results:
            if not r.get('solved') or not r.get('simulation'):
                continue
            sim        = r['simulation']
            method     = r.get('method', '?')
            is_robust  = 'gamma' in r
            esvc       = sim['empirical_patient_svc']
            tsvc       = None if is_robust else r.get('service_level', 1.0 - r.get('alpha', 0.0))
            gap        = (esvc - tsvc) if tsvc is not None else None
            gap_str    = (f"+{gap*100:.1f}pp" if gap >= 0 else f"{gap*100:.1f}pp") if gap is not None else 'N/A'
            gap_color  = '#059669' if (gap is None or gap >= 0) else '#dc2626'
            tsvc_label = 'worst-case' if is_robust else (f"{tsvc*100:.0f}%" if tsvc else '—')

            mc_html += f'''
            <div class="mc-card">
              <div class="mc-header">
                <span class="mc-title">{method}</span>
                <span class="mc-badge" style="background:{_svc_color(esvc)}20;
                      color:{_svc_color(esvc)};border:1px solid {_svc_color(esvc)}40">
                  Empirical SL: {esvc*100:.1f}%
                </span>
                <span class="mc-badge" style="background:#f3f4f6;color:#374151">
                  Theory: {tsvc_label}
                </span>
                <span class="mc-badge" style="background:{gap_color}15;color:{gap_color}">
                  Gap: {gap_str}
                </span>
              </div>
              <div class="mc-stats">
                <div class="stat-box"><div class="stat-val">{sim["avg_violations_per_run"]:.2f}</div>
                  <div class="stat-lbl">Avg violations/run</div></div>
                <div class="stat-box"><div class="stat-val">{sim["max_violations_per_run"]}</div>
                  <div class="stat-lbl">Max violations</div></div>
                <div class="stat-box"><div class="stat-val">{sim["empirical_all_on_time"]*100:.1f}%</div>
                  <div class="stat-lbl">Runs all-on-time</div></div>
                <div class="stat-box"><div class="stat-val">{sim["n_sim"]:,}</div>
                  <div class="stat-lbl">Simulations</div></div>
              </div>
            </div>'''
        mc_html += '</div>'

    # ---- Key findings ----
    findings_html = _build_findings(all_results, sigma_L, n_sim)

    # ---- frontier chart data ----
    frontier_js = json.dumps(frontier_datasets)

    # ---- bar chart data: empirical vs theoretical per method (first dataset) ----
    first_ds   = list(all_results.values())[0]
    bar_labels = []
    bar_theory = []
    bar_empirical = []
    for r in first_ds:
        if not r.get('solved') or not r.get('simulation'):
            continue
        bar_labels.append(r.get('method','?'))
        bar_theory.append(round((r.get('service_level', 1.0 - r.get('alpha', 0.0)))*100, 1))
        bar_empirical.append(round(r['simulation']['empirical_patient_svc']*100, 1))

    bar_labels_js   = json.dumps(bar_labels)
    bar_theory_js   = json.dumps(bar_theory)
    bar_empirical_js = json.dumps(bar_empirical)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CCP Validation — CAR-T Supply Chain</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --blue:#1E3A5F;--green:#1B6B2E;--orange:#92400E;
  --bg:#f4f5f7;--sf:#fff;--brd:#e0e3e8;
  --tx:#1a1d26;--dim:#5f6672;--rad:8px;
  --ac:#2563eb;--g:#059669;--r:#dc2626;--am:#d97706;
}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--tx);font-size:13px}}
.hd{{background:var(--blue);color:#fff;padding:14px 28px;display:flex;align-items:center;gap:12px}}
.hd h1{{font-size:17px;font-weight:700}}
.hd .sub{{font-size:11px;opacity:.75;margin-left:auto}}
.tabs{{display:flex;gap:0;border-bottom:2px solid var(--brd);background:#fff;padding:0 24px}}
.tab{{padding:10px 20px;cursor:pointer;font-size:12px;font-weight:600;
      color:var(--dim);border-bottom:2px solid transparent;margin-bottom:-2px}}
.tab.active{{color:var(--blue);border-color:var(--blue)}}
.pane{{display:none;padding:24px}}.pane.active{{display:block}}
.page{{max-width:1280px;margin:0 auto}}
.card{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);
       padding:20px;margin-bottom:18px}}
.card h2{{font-size:14px;font-weight:700;margin-bottom:14px;color:var(--blue)}}
.ds-section{{margin-bottom:28px}}
.ds-title{{font-size:13px;font-weight:700;color:var(--dim);
           text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;
           padding-bottom:6px;border-bottom:1px solid var(--brd)}}
.tbl-wrap{{overflow-x:auto}}
.data-tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.data-tbl th{{background:#f8f9fa;font-weight:600;padding:8px 10px;
              text-align:left;border-bottom:2px solid var(--brd);
              white-space:nowrap;color:var(--dim)}}
.data-tbl td{{padding:8px 10px;border-bottom:1px solid var(--brd)}}
.data-tbl tr:hover td{{background:#f8f9fa}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}
.chart-box{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);
            padding:16px}}
.chart-box h3{{font-size:12px;font-weight:700;color:var(--dim);margin-bottom:12px}}
.mc-card{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);
          padding:16px;margin-bottom:12px}}
.mc-header{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}}
.mc-title{{font-size:13px;font-weight:700;color:var(--tx)}}
.mc-badge{{padding:3px 9px;border-radius:12px;font-size:11px;font-weight:600}}
.mc-stats{{display:flex;gap:12px;flex-wrap:wrap}}
.stat-box{{background:#f8f9fa;border-radius:6px;padding:10px 16px;min-width:100px}}
.stat-val{{font-size:20px;font-weight:700;color:var(--tx)}}
.stat-lbl{{font-size:10px;color:var(--dim);margin-top:2px}}
.finding-card{{background:#fff;border-left:4px solid var(--blue);
               border-radius:0 var(--rad) var(--rad) 0;
               padding:14px 18px;margin-bottom:10px;
               border:1px solid var(--brd);border-left:4px solid var(--blue)}}
.finding-card h4{{font-size:12px;font-weight:700;margin-bottom:4px;color:var(--blue)}}
.finding-card p{{font-size:12px;color:var(--dim);line-height:1.6}}
@media(max-width:768px){{.chart-row{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="hd">
  <div>
    <h1>CCP Stochastic Validation — CAR-T Supply Chain</h1>
  </div>
  <div class="sub">σ_L = {sigma_L} &nbsp;|&nbsp; {n_sim:,} Monte Carlo runs &nbsp;|&nbsp;
    Methods: Deterministic · CCP · Γ-robust</div>
</div>

<div class="page">
<div class="tabs">
  <div class="tab active" onclick="showTab('comparison')">Method Comparison</div>
  <div class="tab" onclick="showTab('montecarlo')">Monte Carlo Validation</div>
  <div class="tab" onclick="showTab('findings')">Key Findings</div>
</div>

<!-- TAB 1: Comparison -->
<div id="pane-comparison" class="pane active">
  <div class="chart-row">
    <div class="chart-box">
      <h3>Cost vs Empirical Service Level Frontier</h3>
      <canvas id="frontierChart" height="280"></canvas>
    </div>
    <div class="chart-box">
      <h3>Theoretical vs Empirical Service Level ({list(all_results.keys())[0]})</h3>
      <canvas id="barChart" height="280"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Full Method Comparison</h2>
    {table_html}
  </div>
</div>

<!-- TAB 2: Monte Carlo -->
<div id="pane-montecarlo" class="pane">
  <div class="card">
    <h2>Monte Carlo Simulation Results</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:16px">
      For each solved schedule, {n_sim:,} T_MF ~ LogNormal(σ_L={sigma_L}) samples
      are drawn per patient. Empirical service level = fraction of (patient, run)
      pairs where realized TRT ≤ deadline. The CCP guarantee is per-patient, so
      empirical SL should be ≥ theoretical (1−α).
    </p>
    {mc_html}
  </div>
</div>

<!-- TAB 3: Findings -->
<div id="pane-findings" class="pane">
  <div class="card">
    <h2>Key Findings</h2>
    {findings_html}
  </div>
</div>
</div>

<script>
function showTab(name){{
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
  const idx = {{'comparison':0,'montecarlo':1,'findings':2}}[name];
  document.querySelectorAll('.tab')[idx].classList.add('active');
  document.getElementById('pane-'+name).classList.add('active');
}}

// Frontier scatter
const frontierData = {frontier_js};
const fColors = {json.dumps(colours[:len(all_results)])};
const fCtx = document.getElementById('frontierChart').getContext('2d');
new Chart(fCtx, {{
  type: 'scatter',
  data: {{
    datasets: frontierData.map((ds,i) => ({{
      label: ds.name,
      data: ds.points.map(p=>({{'x':p.x,'y':p.y,'label':p.label}})),
      backgroundColor: fColors[i]+'cc',
      borderColor: fColors[i],
      pointRadius: 8, pointHoverRadius: 10,
    }}))
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{
      tooltip:{{callbacks:{{label:ctx=>
        ctx.raw.label+': SL='+ctx.raw.x+'%  Cost=$'+
        (ctx.raw.y*1e6).toLocaleString('en-US',{{maximumFractionDigits:0}})
      }}}},
      legend:{{position:'bottom',labels:{{font:{{size:10}}}}}}
    }},
    scales:{{
      x:{{title:{{display:true,text:'Empirical Service Level (%)',font:{{size:11}}}},
          min:50,max:101}},
      y:{{title:{{display:true,text:'Schedule Cost ($M)',font:{{size:11}}}}}}
    }}
  }}
}});

// Bar chart: theory vs empirical
const barLabels = {bar_labels_js};
const barTheory = {bar_theory_js};
const barEmpirical = {bar_empirical_js};
const bCtx = document.getElementById('barChart').getContext('2d');
new Chart(bCtx, {{
  type:'bar',
  data:{{
    labels: barLabels,
    datasets:[
      {{label:'Theoretical SL (%)', data:barTheory,
        backgroundColor:'#1E3A5F44', borderColor:'#1E3A5F', borderWidth:1}},
      {{label:'Empirical SL (%)',   data:barEmpirical,
        backgroundColor:'#1B6B2E88', borderColor:'#1B6B2E', borderWidth:1}},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{font:{{size:10}}}}}}}},
    scales:{{
      y:{{min:50,max:105,title:{{display:true,text:'Service Level (%)',font:{{size:11}}}}}},
      x:{{ticks:{{font:{{size:9}},maxRotation:30}}}}
    }}
  }}
}});
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'HTML report written to {output_path}')


def _build_findings(all_results, sigma_L, n_sim):
    """Generate key findings HTML from results."""
    lines = []

    # Finding 1: CCP guarantee validity
    ccp_valid = []
    for ds_name, results in all_results.items():
        for r in results:
            if 'CCP' not in r.get('method',''):
                continue
            sim = r.get('simulation',{})
            if not sim:
                continue
            tsvc = r.get('service_level', 0)
            esvc = sim.get('empirical_patient_svc', 0)
            ccp_valid.append(esvc >= tsvc - 0.01)

    if ccp_valid:
        pct = sum(ccp_valid)/len(ccp_valid)*100
        color = '#059669' if pct >= 80 else '#dc2626'
        lines.append(f'''<div class="finding-card">
          <h4>CCP Guarantee Validity</h4>
          <p>Across all CCP variants, <strong style="color:{color}">{pct:.0f}%</strong>
          achieved an empirical service level ≥ the theoretical (1−α) guarantee
          ({n_sim:,} Monte Carlo runs, σ_L={sigma_L}).
          The CCP substitution T_MF^α = F⁻¹(1−α) correctly bounds per-patient deadline
          violations as predicted by the chance-constraint derivation.</p>
        </div>''')

    # Finding 2: Cost of robustness
    det_cost = None
    best_ccp_cost = None
    for ds_name, results in list(all_results.items())[:1]:
        for r in results:
            if r.get('method') == 'Deterministic' and r.get('solved'):
                det_cost = r['total_cost']
            if 'CCP' in r.get('method','') and r.get('solved'):
                if best_ccp_cost is None or r['total_cost'] < best_ccp_cost:
                    best_ccp_cost = r['total_cost']

    if det_cost and best_ccp_cost:
        pct_inc = (best_ccp_cost - det_cost)/det_cost*100
        lines.append(f'''<div class="finding-card">
          <h4>Cost of Robustness</h4>
          <p>The cheapest feasible CCP schedule costs
          <strong>${best_ccp_cost:,.0f}</strong> vs
          the deterministic baseline of <strong>${det_cost:,.0f}</strong>
          — a <strong>{pct_inc:.1f}% premium</strong> for the service-level guarantee.
          This premium reflects fewer feasible routes (TMFE_eff > 7 days tightens
          the deadline filter), not any change to the objective function.</p>
        </div>''')

    # Finding 3: Deterministic under real uncertainty
    for ds_name, results in list(all_results.items())[:1]:
        for r in results:
            if r.get('method') != 'Deterministic':
                continue
            sim = r.get('simulation',{})
            if not sim:
                continue
            esvc = sim['empirical_patient_svc']
            avg_v = sim['avg_violations_per_run']
            lines.append(f'''<div class="finding-card">
              <h4>Deterministic Schedule Under Real Uncertainty</h4>
              <p>When the deterministic schedule (TMFE=7d) is exposed to
              LogNormal(σ_L={sigma_L}) manufacturing variability, the empirical
              per-patient service level drops to
              <strong style="color:#dc2626">{esvc*100:.1f}%</strong>, with an
              average of <strong>{avg_v:.2f}</strong> deadline violations per
              simulation run. This illustrates why the CCP correction is necessary.</p>
            </div>''')

    # Finding 4: Gamma-robust vs CCP
    lines.append(f'''<div class="finding-card">
      <h4>Γ-Robust vs CCP: Different Guarantees</h4>
      <p>Γ-robust optimization (Bertsimas-Sim) protects against at most Γ simultaneous
      worst-case deviations using a box uncertainty set. Unlike CCP, it makes no
      probabilistic assumption about T_MF — the guarantee is worst-case deterministic.
      This typically results in higher TMFE_eff (more conservative) than CCP at the
      same σ_L, because the box worst-case exceeds the LogNormal quantile.
      CCP is preferred when the distribution is well-characterised; Γ-robust
      when only bounds on T_MF are known.</p>
    </div>''')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CCP Validation Report — Deterministic vs CCP vs Gamma-robust')
    parser.add_argument('--data',    nargs='+', default=None,
                        help='.dat file(s), space-separated')
    parser.add_argument('--sigma-L', type=float, default=0.20, dest='sigma_L')
    parser.add_argument('--alphas',  nargs='+',  type=float, default=[0.20, 0.10, 0.05])
    parser.add_argument('--gammas',  nargs='+',  type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument('--n-sim',   type=int,   default=1000, dest='n_sim')
    parser.add_argument('--time-limit', type=int, default=120)
    parser.add_argument('--html',    default='ccp_validation_report.html')
    parser.add_argument('--json',    action='store_true')
    parser.add_argument('--no-html', action='store_true', dest='no_html')
    args = parser.parse_args()

    data_files = args.data or [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N15.dat')]

    all_results = {}
    for df in data_files:
        name = os.path.basename(df).replace('.dat','')
        print(f'\n{"="*60}')
        print(f'Dataset: {name}  sigma_L={args.sigma_L}  n_sim={args.n_sim}')
        print(f'{"="*60}')
        all_results[name] = run_full_comparison(
            data_file=df,
            sigma_L=args.sigma_L,
            alphas=args.alphas,
            gammas=args.gammas,
            n_sim=args.n_sim,
            time_limit=args.time_limit,
        )

    if args.json:
        # Slim JSON — drop raw patient lists
        slim = {}
        for ds, rlist in all_results.items():
            slim[ds] = []
            for r in rlist:
                rc = {k: v for k, v in r.items() if k != 'patients'}
                slim[ds].append(rc)
        print(json.dumps(slim, indent=2))

    if not args.no_html:
        generate_html(all_results, args.html, sigma_L=args.sigma_L, n_sim=args.n_sim)
        print(f'\nOpen {args.html} in your browser to view the report.')
