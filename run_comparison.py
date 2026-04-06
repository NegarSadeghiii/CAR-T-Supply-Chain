"""
CAR-T Supply Chain — Full Method Comparison Report
===================================================
Runs Deterministic, CCP, and Two-Stage Expediting across datasets
and generates a single HTML dashboard comparing all scenarios.

Usage
-----
  python run_comparison.py
  python run_comparison.py --data Data_N15.dat Data_N50.dat
  python run_comparison.py --data Data_N15.dat --alphas 0.05 0.10 0.20 --deltas 1 2 3
"""

import argparse, json, math, os, statistics

# ── solvers ──────────────────────────────────────────────────────────────────
from cart_colgen        import run_experiment as det_run, DEFAULT_URGENCY
from cart_colgen_ccp    import run_experiment_ccp, DEFAULT_PROCESS_CCP
from cart_colgen_twostage import run_experiment_twostage, DEFAULT_PROCESS_2S

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── runner ────────────────────────────────────────────────────────────────────

def run_all(data_file, alphas, sigma_L, deltas, epsilon, time_limit=120):
    name = os.path.basename(data_file).replace('.dat', '')
    rows = []

    # 1. Deterministic
    print(f"  [{name}] Deterministic ...", flush=True)
    r = det_run(data_file=data_file, time_limit=time_limit)
    rows.append({**r,
        'method': 'Deterministic', 'group': 'det',
        'label': 'Deterministic',
        'sigma_L': 0, 'alpha': None, 'delta': None,
        'tmfe_eff': 7, 'prob_expedite': None,
    })
    _print_row(rows[-1])

    # 2. CCP variants
    for alpha in alphas:
        cfg = dict(DEFAULT_PROCESS_CCP)
        cfg['alpha']   = alpha
        cfg['sigma_L'] = sigma_L
        label = f'CCP  α={alpha}'
        print(f"  [{name}] {label} ...", flush=True)
        r = run_experiment_ccp(data_file=data_file, process_config=cfg,
                               time_limit=time_limit)
        rows.append({**r,
            'method': label, 'group': 'ccp',
            'label': label,
            'delta': None, 'prob_expedite': None,
        })
        _print_row(rows[-1])

    # 3. Two-stage variants
    for delta in deltas:
        cfg = dict(DEFAULT_PROCESS_2S)
        cfg['sigma_L'] = sigma_L
        cfg['delta']   = delta
        cfg['epsilon'] = epsilon
        label = f'2S-Exp  δ={delta}d'
        print(f"  [{name}] {label} ...", flush=True)
        r = run_experiment_twostage(data_file=data_file, process_config=cfg,
                                    time_limit=time_limit)
        rows.append({**r,
            'method': label, 'group': '2s',
            'label': label,
            'alpha': None, 'tmfe_eff': 7,
        })
        _print_row(rows[-1])

    return {'dataset': name, 'rows': rows}


def _print_row(r):
    s = r.get('solved', False)
    cost = f"${r['total_cost']:,.0f}" if s else 'INFEASIBLE'
    trt  = f"{r['avg_trt']:.1f}d"    if s else '—'
    print(f"    → {cost}  TRT={trt}  plans={r.get('num_plans',0)}")


# ── HTML generator ─────────────────────────────────────────────────────────────

def _c(v, lo, hi):
    """Interpolate green→amber→red for a value in [lo,hi]."""
    if v is None: return '#9ca3af'
    t = max(0, min(1, (v - lo) / (hi - lo + 1e-9)))
    if t < 0.5:
        r,g = int(34+t*2*200), int(197-t*2*60)
    else:
        r,g = int(234+(t-0.5)*2*0), int(137-(t-0.5)*2*137)
    return f'rgb({r},{g},60)'

METHOD_COLORS = {'det': '#1E3A5F', 'ccp': '#1B6B2E', '2s': '#92400E'}
METHOD_BG     = {'det': '#eff6ff',  'ccp': '#f0fdf4',  '2s': '#fff7ed'}
METHOD_BORDER = {'det': '#bfdbfe',  'ccp': '#bbf7d0',  '2s': '#fed7aa'}

def generate_html(datasets, output_path, sigma_L, alphas, deltas):
    # collect all rows for JS charts
    all_rows = [r for ds in datasets for r in ds['rows']]
    solved   = [r for r in all_rows if r.get('solved')]

    # build summary cards HTML
    cards_html = _summary_cards(datasets)

    # build per-dataset tables
    tables_html = ''
    for ds in datasets:
        tables_html += _dataset_table(ds)

    # build chart data
    chart_data = json.dumps([{
        'label':  r['label'],
        'group':  r['group'],
        'cost':   r['total_cost'] / 1e6,
        'trt':    r['avg_trt'],
        'plans':  r['num_plans'],
        'dataset':r.get('dataset', ''),
    } for r in solved])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CAR-T Stochastic Methods — Full Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--blue:#1E3A5F;--green:#1B6B2E;--orange:#92400E;
      --bg:#f4f5f7;--sf:#fff;--brd:#e0e3e8;--tx:#1a1d26;--dim:#5f6672;--rad:8px}}
body{{font-family:"Inter",system-ui,sans-serif;background:var(--bg);color:var(--tx);font-size:13px;line-height:1.6}}
.hd{{background:var(--blue);color:#fff;padding:14px 28px;display:flex;align-items:center;gap:16px}}
.hd h1{{font-size:17px;font-weight:700}}
.hd .sub{{font-size:11px;opacity:.7;margin-left:auto}}
.tabs{{display:flex;background:#fff;border-bottom:2px solid var(--brd);padding:0 24px}}
.tab{{padding:10px 20px;cursor:pointer;font-size:12px;font-weight:600;color:var(--dim);
      border-bottom:2px solid transparent;margin-bottom:-2px;transition:.15s}}
.tab.a{{color:var(--blue);border-color:var(--blue)}}
.pane{{display:none;max-width:1280px;margin:0 auto;padding:22px 24px 40px}}
.pane.a{{display:block}}
.card{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);
       padding:18px 22px;margin-bottom:18px}}
.card h2{{font-size:14px;font-weight:700;color:var(--blue);margin-bottom:14px;
          padding-bottom:8px;border-bottom:1px solid var(--brd)}}
.summary-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}}
@media(max-width:900px){{.summary-grid{{grid-template-columns:1fr}}}}
.sm-card{{border-radius:var(--rad);padding:14px 18px;border:1px solid}}
.sm-card h4{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}}
.stat-row{{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px}}
.stat-row .v{{font-weight:700}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}
@media(max-width:800px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-box{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);padding:16px}}
.chart-box h3{{font-size:12px;font-weight:700;color:var(--dim);margin-bottom:12px}}
.chart-wrap{{position:relative;height:260px;overflow:hidden}}
.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th{{background:#f1f5f9;font-weight:700;padding:8px 10px;text-align:left;
         border-bottom:2px solid var(--brd);color:var(--dim);white-space:nowrap}}
.tbl td{{padding:8px 10px;border-bottom:1px solid var(--brd)}}
.tbl tr:hover td{{background:#f8fafc}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700}}
.ds-title{{font-size:12px;font-weight:700;color:var(--dim);text-transform:uppercase;
           letter-spacing:.5px;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid var(--brd)}}
.ds-section{{margin-bottom:24px}}
.inf{{color:#dc2626;font-weight:700}}
</style>
</head>
<body>
<div class="hd">
  <div>
    <h1>CAR-T Supply Chain — Stochastic Methods Comparison</h1>
    <div style="font-size:11px;opacity:.7;margin-top:3px">
      Deterministic &nbsp;·&nbsp; CCP (α={alphas}) &nbsp;·&nbsp;
      Two-Stage Expediting (δ={deltas}d) &nbsp;·&nbsp; σ_L={sigma_L}
    </div>
  </div>
  <div class="sub">{len(datasets)} dataset(s) &nbsp;|&nbsp;
    {sum(len(ds["rows"]) for ds in datasets)} scenarios</div>
</div>

<div class="tabs">
  <div class="tab a" onclick="show(\'summary\')">Summary</div>
  <div class="tab" onclick="show(\'results\')">Results Table</div>
  <div class="tab" onclick="show(\'charts\')">Charts</div>
  <div class="tab" onclick="show(\'findings\')">Key Findings</div>
</div>

<!-- TAB 1: Summary -->
<div id="pane-summary" class="pane a">
  <div class="summary-grid">{cards_html}</div>
  <div class="chart-grid">
    <div class="chart-box"><h3>Schedule Cost by Method</h3>
      <div class="chart-wrap"><canvas id="costChart"></canvas></div></div>
    <div class="chart-box"><h3>Avg Treatment Time by Method</h3>
      <div class="chart-wrap"><canvas id="trtChart"></canvas></div></div>
  </div>
</div>

<!-- TAB 2: Results Table -->
<div id="pane-results" class="pane">
  <div class="card"><h2>All Scenarios</h2>{tables_html}</div>
</div>

<!-- TAB 3: Charts -->
<div id="pane-charts" class="pane">
  <div class="chart-grid">
    <div class="chart-box"><h3>Cost vs Plans Available</h3>
      <div class="chart-wrap"><canvas id="scatterChart"></canvas></div></div>
    <div class="chart-box"><h3>Plans Generated per Method</h3>
      <div class="chart-wrap"><canvas id="plansChart"></canvas></div></div>
  </div>
</div>

<!-- TAB 4: Findings -->
<div id="pane-findings" class="pane">
  <div class="card"><h2>Key Findings</h2>
  {_findings(datasets, sigma_L)}
  </div>
</div>

<script>
function show(n){{
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("a"));
  document.querySelectorAll(".pane").forEach(p=>p.classList.remove("a"));
  const m={{summary:0,results:1,charts:2,findings:3}};
  document.querySelectorAll(".tab")[m[n]].classList.add("a");
  document.getElementById("pane-"+n).classList.add("a");
  // Let Chart.js re-measure after the pane becomes visible
  requestAnimationFrame(()=>window.dispatchEvent(new Event("resize")));
}}

const DATA = {chart_data};
const GC = {{det:"#1E3A5F",ccp:"#1B6B2E","2s":"#92400E"}};
const labels = DATA.map(d=>d.label);
const costs  = DATA.map(d=>d.cost);
const trts   = DATA.map(d=>d.trt);
const plans  = DATA.map(d=>d.plans);
const bgs    = DATA.map(d=>GC[d.group]+"99");
const bds    = DATA.map(d=>GC[d.group]);

function mkBar(id, labels, data, label, fmt){{
  new Chart(document.getElementById(id),{{
    type:"bar",
    data:{{labels,datasets:[{{label,data,backgroundColor:bgs,borderColor:bds,borderWidth:1}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}}}},
      scales:{{y:{{ticks:{{callback:v=>fmt(v)}}}},
               x:{{ticks:{{font:{{size:9}},maxRotation:35}}}}}}}}
  }});
}}

mkBar("costChart", labels, costs, "Cost ($M)", v=>"$"+v.toFixed(2)+"M");
mkBar("trtChart",  labels, trts,  "Avg TRT (d)", v=>v.toFixed(1)+"d");
mkBar("plansChart",labels, plans, "Plans", v=>v);

// Force correct sizing after initial render
requestAnimationFrame(()=>window.dispatchEvent(new Event("resize")));

// Scatter: cost vs plans
new Chart(document.getElementById("scatterChart"),{{
  type:"scatter",
  data:{{datasets:[
    {{label:"Deterministic",
     data:DATA.filter(d=>d.group=="det").map(d=>({{"x":d.plans,"y":d.cost}})),
     backgroundColor:"#1E3A5F99",borderColor:"#1E3A5F",pointRadius:8}},
    {{label:"CCP",
     data:DATA.filter(d=>d.group=="ccp").map(d=>({{"x":d.plans,"y":d.cost}})),
     backgroundColor:"#1B6B2E99",borderColor:"#1B6B2E",pointRadius:8}},
    {{label:"2S-Expediting",
     data:DATA.filter(d=>d.group=="2s").map(d=>({{"x":d.plans,"y":d.cost}})),
     backgroundColor:"#92400E99",borderColor:"#92400E",pointRadius:8}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{position:"bottom",labels:{{font:{{size:10}}}}}}}},
    scales:{{
      x:{{title:{{display:true,text:"Plans Generated"}}}},
      y:{{title:{{display:true,text:"Cost ($M)"}},
         ticks:{{callback:v=>"$"+v.toFixed(2)+"M"}}}}}}}}
}});
</script>
</body></html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'\nHTML report → {output_path}')


def _summary_cards(datasets):
    groups = {'det': [], 'ccp': [], '2s': []}
    for ds in datasets:
        for r in ds['rows']:
            if r.get('solved'):
                groups[r['group']].append(r)

    cards = ''
    specs = [
        ('det', 'Deterministic', '#1E3A5F', '#eff6ff', '#bfdbfe'),
        ('ccp', 'CCP Stochastic',  '#1B6B2E', '#f0fdf4', '#bbf7d0'),
        ('2s',  'Two-Stage Expediting', '#92400E', '#fff7ed', '#fed7aa'),
    ]
    for gid, title, col, bg, brd in specs:
        rows = groups[gid]
        if not rows:
            cards += f'<div class="sm-card" style="background:{bg};border-color:{brd}"><h4 style="color:{col}">{title}</h4><p style="color:#9ca3af;font-size:12px">No feasible solutions</p></div>'
            continue
        avg_cost  = sum(r['total_cost'] for r in rows) / len(rows)
        avg_trt   = sum(r['avg_trt']    for r in rows) / len(rows)
        avg_plans = sum(r['num_plans']  for r in rows) / len(rows)
        n_feasible = len(rows)
        n_total    = sum(1 for ds in datasets for r in ds['rows'] if r['group'] == gid)

        extra = ''
        if gid == 'ccp':
            alphas_used = sorted(set(r.get('alpha',0) for r in rows if r.get('alpha')))
            extra = f'<div class="stat-row"><span>Service levels</span><span class="v">{[f"{int((1-a)*100)}%" for a in alphas_used]}</span></div>'
        elif gid == '2s':
            avg_pexp = sum(r.get('avg_prob_expedite',0) for r in rows) / len(rows)
            extra = f'<div class="stat-row"><span>Avg P(expedite)</span><span class="v">{avg_pexp*100:.1f}%</span></div>'

        cards += f'''<div class="sm-card" style="background:{bg};border-color:{brd}">
          <h4 style="color:{col}">{title}</h4>
          <div class="stat-row"><span>Avg cost</span><span class="v">${avg_cost:,.0f}</span></div>
          <div class="stat-row"><span>Avg TRT</span><span class="v">{avg_trt:.1f}d</span></div>
          <div class="stat-row"><span>Avg plans</span><span class="v">{avg_plans:.0f}</span></div>
          <div class="stat-row"><span>Feasible scenarios</span><span class="v">{n_feasible}/{n_total}</span></div>
          {extra}
        </div>'''
    return cards


def _dataset_table(ds):
    h = f'<div class="ds-section"><div class="ds-title">{ds["dataset"]}</div>'
    h += '<div style="overflow-x:auto"><table class="tbl"><thead><tr>'
    for col in ['Method','Plans','Filter/TMFE_eff','Schedule Cost','Avg TRT',
                'Solve Time','Extra Info']:
        h += f'<th>{col}</th>'
    h += '</tr></thead><tbody>'
    for r in ds['rows']:
        s   = r.get('solved', False)
        grp = r.get('group','det')
        col = METHOD_COLORS[grp]
        bg  = METHOD_BG[grp]
        cost_s = f"${r['total_cost']:,.0f}" if s else '<span class="inf">INFEASIBLE</span>'
        trt_s  = f"{r['avg_trt']:.1f}d" if s else '—'
        time_s = f"{r.get('solve_time',0):.1f}s" if s else '—'
        plans_s = str(r.get('num_plans',0)) if s else '0'

        if grp == 'ccp':
            teff  = r.get('tmfe_eff', 7)
            extra = f"α={r.get('alpha','?')}  TMFE_eff={teff:.2f}d"
        elif grp == '2s':
            pexp = r.get('avg_prob_expedite', 0)
            eexp = r.get('total_exp_expedite_cost', 0)
            filt = r.get('filter_threshold', 0)
            extra = f"Filter={filt:.2f}d  P(exp)={pexp*100:.1f}%  E[$exp]=${eexp:,.0f}"
        else:
            extra = 'Nominal T_MF = 7d'

        filter_s = f"{r.get('filter_threshold',r.get('tmfe_eff',7)):.2f}d"

        h += f'''<tr>
          <td><span class="badge" style="background:{bg};color:{col};border:1px solid {METHOD_BORDER[grp]}">{r["method"]}</span></td>
          <td>{plans_s}</td><td>{filter_s}</td>
          <td>{cost_s}</td><td>{trt_s}</td><td>{time_s}</td>
          <td style="font-size:11px;color:var(--dim)">{extra}</td>
        </tr>'''
    h += '</tbody></table></div></div>'
    return h


def _findings(datasets, sigma_L):
    all_rows = [r for ds in datasets for r in ds['rows']]
    det  = next((r for r in all_rows if r['group']=='det' and r.get('solved')), None)
    ccps = [r for r in all_rows if r['group']=='ccp' and r.get('solved')]
    twos = [r for r in all_rows if r['group']=='2s'  and r.get('solved')]

    f = ''
    style = 'border-left:4px solid {c};padding:12px 16px;background:{b};border-radius:0 8px 8px 0;margin-bottom:12px;border:1px solid {br};border-left:4px solid {c}'

    if det and twos:
        best2s = min(twos, key=lambda r: r['total_cost'])
        premium = (best2s['total_cost'] - det['total_cost']) / det['total_cost'] * 100
        pexp = best2s.get('avg_prob_expedite', 0)
        f += f'<div style="{style.format(c="#1B6B2E",b="#f0fdf4",br="#bbf7d0")}">'
        f += f'<strong>Two-Stage Expediting vs Deterministic:</strong> The best two-stage scenario costs '
        f += f'<strong>${best2s["total_cost"]:,.0f}</strong> vs deterministic <strong>${det["total_cost"]:,.0f}</strong> '
        f += f'— a premium of only <strong>{premium:.1f}%</strong>. On average, '
        f += f'<strong>{pexp*100:.1f}%</strong> of patients require expedited return, meaning the active '
        f += f'recourse is rarely triggered but available when needed.</div>'

    if det and ccps:
        best_ccp = min(ccps, key=lambda r: r['total_cost'])
        prem_ccp = (best_ccp['total_cost'] - det['total_cost']) / det['total_cost'] * 100
        f += f'<div style="{style.format(c="#92400E",b="#fff7ed",br="#fed7aa")}">'
        f += f'<strong>CCP vs Deterministic:</strong> The best feasible CCP scenario costs '
        f += f'<strong>${best_ccp["total_cost"]:,.0f}</strong> — a premium of <strong>{prem_ccp:.1f}%</strong>. '
        f += f'CCP pre-emptively inflates TMFE to {best_ccp.get("tmfe_eff",7):.2f}d, restricting routes and '
        f += f'raising facility costs. This front-loads robustness into the plan-selection stage.</div>'

    if ccps and twos:
        infeas_ccp = sum(1 for ds in datasets for r in ds['rows'] if r['group']=='ccp' and not r.get('solved'))
        infeas_2s  = sum(1 for ds in datasets for r in ds['rows'] if r['group']=='2s'  and not r.get('solved'))
        f += f'<div style="{style.format(c="#1E3A5F",b="#eff6ff",br="#bfdbfe")}">'
        f += f'<strong>Feasibility:</strong> CCP had <strong>{infeas_ccp}</strong> infeasible scenario(s) '
        f += f'(strict deadline filter eliminates too many routes). Two-Stage Expediting had '
        f += f'<strong>{infeas_2s}</strong> infeasible scenario(s) (relaxed filter via δ). '
        f += f'The expediting recourse creates more room for feasible routes.</div>'

    f += f'<div style="{style.format(c="#5b21b6",b="#f5f3ff",br="#ddd6fe")}">'
    f += f'<strong>Modelling philosophy:</strong> All three methods use σ_L={sigma_L} for T_MF uncertainty. '
    f += f'Deterministic ignores it. CCP enforces hard probabilistic deadlines (no recourse). '
    f += f'Two-Stage Expediting allows the schedule to breathe — routes are selected with recourse in mind, '
    f += f'and the cost of uncertainty is explicit in the objective as expected expediting cost.</div>'

    return f


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full stochastic comparison report')
    parser.add_argument('--data',    nargs='+', default=['Data_N15.dat'])
    parser.add_argument('--sigma-L', type=float, default=0.20, dest='sigma_L')
    parser.add_argument('--alphas',  nargs='+',  type=float, default=[0.20, 0.10, 0.05])
    parser.add_argument('--deltas',  nargs='+',  type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--time-limit', type=int, default=120)
    parser.add_argument('--html',    default='comparison_report.html')
    args = parser.parse_args()

    datasets = []
    for df in args.data:
        if not os.path.isabs(df):
            df = os.path.join(BASE_DIR, df)
        print(f'\n=== {os.path.basename(df)} ===')
        ds = run_all(df, args.alphas, args.sigma_L, args.deltas,
                     args.epsilon, args.time_limit)
        datasets.append(ds)

    generate_html(datasets, args.html, args.sigma_L, args.alphas, args.deltas)
    print(f'\nDone. Open {args.html} in your browser.')
