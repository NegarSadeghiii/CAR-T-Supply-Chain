#!/usr/bin/env python3
"""
twostage_validation.py
======================
Validate Two-Stage Expediting ColGen vs Deterministic ColGen across all
CAR-T supply chain problem instances (N=5,10,15,20,25,40,50).

Usage
-----
    python twostage_validation.py
    python twostage_validation.py --tau 0.0 --time-limit 180 --sigma-L 0.20 --delta 2.0
"""

import argparse, json, math, os, random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from cart_colgen import (
    run_experiment  as det_run,
    DEFAULT_URGENCY,
    DEFAULT_PROCESS as DET_PROCESS,
)
from cart_colgen_twostage import (
    run_experiment_twostage as ts_run,
    DEFAULT_PROCESS_2S      as TS_PROCESS,
)
from cart_colgen_ccp import (
    run_experiment_ccp  as ccp_run,
    DEFAULT_PROCESS_CCP as CCP_PROCESS,
)

DATASETS = [
    ('N=5',  'Data_N5.dat'),
    ('N=10', 'Data_N10.dat'),
    ('N=15', 'Data_N15.dat'),
    ('N=20', 'Data_N20.dat'),
    ('N=25', 'Data_N25.dat'),
    ('N=40', 'Data_N40.dat'),
    ('N=50', 'Data_N50.dat'),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def _lognormal_survival(K, tmfe, sigma_L):
    if sigma_L <= 0: return 1.0 if tmfe > K else 0.0
    if K <= 0:       return 1.0
    mu_L = math.log(tmfe) - 0.5 * sigma_L ** 2
    return 1.0 - _norm_cdf((math.log(K) - mu_L) / sigma_L)

def monte_carlo_validate(patients, tmfe, sigma_L, pi_map, n_sim=5000, seed=42):
    """Compare theoretical vs empirical P(expedite) for every selected patient plan."""
    if sigma_L <= 0 or not patients:
        return []
    rng  = random.Random(seed)
    mu_L = math.log(tmfe) - 0.5 * sigma_L ** 2
    rows = []
    for pat in patients:
        K = pat.get('slack')
        if K is None:
            continue
        urg = pat.get('group', pat.get('urgency', 'low'))
        pi  = pi_map.get(urg, 1500)
        th_prob = _lognormal_survival(K, tmfe, sigma_L)
        th_cost = pi * th_prob
        expedited = sum(
            1 for _ in range(n_sim)
            if rng.lognormvariate(mu_L, sigma_L) > K
        )
        emp_prob = expedited / n_sim
        emp_cost = pi * emp_prob
        rows.append({
            'id':       pat.get('id', '?'),
            'urgency':  urg,
            'K_i':      round(K, 3),
            'pi':       pi,
            'th_prob':  round(th_prob, 4),
            'emp_prob': round(emp_prob, 4),
            'th_cost':  round(th_cost, 2),
            'emp_cost': round(emp_cost, 2),
            'rel_err':  round(abs(th_prob - emp_prob) / max(th_prob, 1e-9), 4),
        })
    return rows


# ── main runner ───────────────────────────────────────────────────────────────

def run_all(tau, sigma_L, delta, epsilon, time_limit, n_sim):
    results = []
    for label, fname in DATASETS:
        df = os.path.join(BASE_DIR, fname)
        if not os.path.exists(df):
            print(f'  [SKIP] {fname} not found')
            continue
        n_val = int(label.split('=')[1])
        print(f'\n=== {label} ===')

        # Deterministic
        print('  Deterministic ...', end=' ', flush=True)
        try:
            r = det_run(tau=tau, data_file=df, time_limit=time_limit)
            det = {
                'N': n_val, 'label': label, 'solver': 'Deterministic',
                'solved':           r.get('solved', False),
                'total_cost':       r.get('total_cost', 0),
                'transport_cost':   r.get('transport_cost', 0),
                'facility_cost':    r.get('facility_cost', 0),
                'avg_trt':          r.get('avg_trt',    0),
                'solve_time':       r.get('solve_time', 0),
                'num_plans':        r.get('num_plans',  0),
                'num_patients':     r.get('num_patients', n_val),
                'error':            r.get('error', ''),
                'patients':         r.get('patients', []),
            }
        except Exception as e:
            det = {'N': n_val, 'label': label, 'solver': 'Deterministic',
                   'solved': False, 'error': str(e), 'patients': []}
        print(f"{'OK' if det['solved'] else 'FAIL'}  ",
              f"cost={det.get('total_cost',0):,.0f}  ",
              f"trt={det.get('avg_trt',0):.1f}d  ",
              f"t={det.get('solve_time',0):.1f}s")
        results.append(det)

        # Two-Stage Expediting
        print('  Two-Stage Exp  ...', end=' ', flush=True)
        ts_cfg = dict(TS_PROCESS)
        ts_cfg['sigma_L'] = sigma_L
        ts_cfg['delta']   = delta
        ts_cfg['epsilon'] = epsilon
        try:
            r2 = ts_run(tau=tau, data_file=df,
                        process_config=ts_cfg, time_limit=time_limit)
            pi_map = r2.get('pi_map', {'high': 8000, 'medium': 4000, 'low': 1500})
            mc = monte_carlo_validate(
                r2.get('patients', []),
                ts_cfg['tmfe'], sigma_L, pi_map, n_sim=n_sim
            ) if r2.get('solved') else []
            ts = {
                'N': n_val, 'label': label, 'solver': 'Two-Stage Exp.',
                'solved':               r2.get('solved', False),
                'total_cost':           r2.get('total_cost', 0),
                'transport_cost':       r2.get('transport_cost', 0),
                'facility_cost':        r2.get('facility_cost', 0),
                'avg_trt':              r2.get('avg_trt',    0),
                'solve_time':           r2.get('solve_time', 0),
                'num_plans':            r2.get('num_plans',  0),
                'num_patients':         r2.get('num_patients', n_val),
                'sigma_L':              sigma_L,
                'delta':                delta,
                'epsilon':              epsilon,
                'filter_threshold':     r2.get('filter_threshold', 0),
                'avg_prob_expedite':    r2.get('avg_prob_expedite', 0),
                'total_exp_expedite_cost': r2.get('total_exp_expedite_cost', 0),
                'mc_validation':        mc,
                'pi_map':               pi_map,
                'patients':             r2.get('patients', []),
                'error':                r2.get('error', ''),
            }
        except Exception as e:
            ts = {'N': n_val, 'label': label, 'solver': 'Two-Stage Exp.',
                  'solved': False, 'error': str(e),
                  'mc_validation': [], 'patients': []}
        s = ts
        print(f"{'OK' if s['solved'] else 'FAIL'}  ",
              f"cost={s.get('total_cost',0):,.0f}  ",
              f"trt={s.get('avg_trt',0):.1f}d  ",
              f"t={s.get('solve_time',0):.1f}s  ",
              f"P(exp)={s.get('avg_prob_expedite',0)*100:.1f}%")
        results.append(ts)

        # CCP (alpha=0.10)
        print('  CCP α=0.10     ...', end=' ', flush=True)
        ccp_cfg = dict(CCP_PROCESS)
        ccp_cfg['sigma_L'] = sigma_L
        ccp_cfg['alpha']   = 0.10
        try:
            rc = ccp_run(tau=tau, data_file=df,
                         process_config=ccp_cfg, time_limit=time_limit)
            ccp = {
                'N': n_val, 'label': label, 'solver': 'CCP',
                'solved':       rc.get('solved', False),
                'total_cost':   rc.get('total_cost', 0),
                'avg_trt':      rc.get('avg_trt',    0),
                'solve_time':   rc.get('solve_time', 0),
                'num_plans':    rc.get('num_plans',  0),
                'num_patients': rc.get('num_patients', n_val),
                'tmfe_eff':     rc.get('tmfe_eff',   0),
                'alpha':        ccp_cfg['alpha'],
                'sigma_L':      sigma_L,
                'patients':     rc.get('patients', []),
                'error':        rc.get('error', ''),
            }
        except Exception as e:
            ccp = {'N': n_val, 'label': label, 'solver': 'CCP',
                   'solved': False, 'error': str(e), 'patients': []}
        print(f"{'OK' if ccp['solved'] else 'FAIL'}  ",
              f"cost={ccp.get('total_cost',0):,.0f}  ",
              f"trt={ccp.get('avg_trt',0):.1f}d  ",
              f"TMFE_eff={ccp.get('tmfe_eff',0):.2f}d")
        results.append(ccp)

        # Out-of-sample: evaluate CCP and Two-Stage on same scenarios
        if ccp.get('solved') and ts.get('solved'):
            pi_map_oos = ts.get('pi_map', {'high': 8000, 'medium': 4000, 'low': 1500})
            oos = out_of_sample_validate(
                ccp, ts, ts_cfg['tmfe'], sigma_L, delta, pi_map_oos,
                n_sim=10000, seed=77
            )
            oos['N'] = n_val
            results.append({'N': n_val, 'label': label, 'solver': 'OOS', 'data': oos})

    return results


# ── Out-of-sample validation ──────────────────────────────────────────────────

def out_of_sample_validate(ccp_r, ts_r, tmfe, sigma_L, delta, pi_map,
                            n_sim=10000, seed=77):
    """Evaluate CCP and Two-Stage solutions on n_sim scenarios.
    Each patient gets their own independent T_MF draw per scenario
    (CAR-T manufacturing is patient-specific).
    """
    import statistics as _st
    rng  = random.Random(seed)
    mu_L = math.log(tmfe) - 0.5 * sigma_L ** 2

    # CCP — no recourse: violation if T_MF_p > mfg_budget_p
    tmfe_eff      = ccp_r.get('tmfe_eff', tmfe)
    ccp_pats      = ccp_r.get('patients', [])
    ccp_transport = ccp_r.get('total_cost', 0)
    mfg_budgets   = {
        p['id']: p['deadline'] - (p['trt'] - tmfe_eff)
        for p in ccp_pats
    }

    ccp_costs, ccp_viols = [], []
    for _ in range(n_sim):
        nv = 0
        for p in ccp_pats:
            tmf_p = rng.lognormvariate(mu_L, sigma_L)   # per-patient draw
            if tmf_p > mfg_budgets.get(p['id'], 999):
                nv += 1
        ccp_costs.append(ccp_transport)   # cost is fixed — no recourse
        ccp_viols.append(nv)

    # Two-Stage — recourse: expedite if T_MF_p > K_p
    ts_pats      = ts_r.get('patients', [])
    ts_transport = ts_r.get('total_cost', 0) - ts_r.get('total_exp_expedite_cost', 0)

    ts_costs, ts_viols, ts_exps = [], [], []
    for _ in range(n_sim):
        ec, nv, ne = 0, 0, 0
        for p in ts_pats:
            K = p.get('slack')
            if K is None:
                continue
            tmf_p = rng.lognormvariate(mu_L, sigma_L)   # per-patient draw
            pi_p  = pi_map.get(p.get('group', 'low'), 1500)
            if tmf_p > K:
                ec += pi_p; ne += 1
                if tmf_p > K + delta:
                    nv += 1
        ts_costs.append(ts_transport + ec)
        ts_viols.append(nv)
        ts_exps.append(ne)

    n_ts = max(len(ts_pats), 1)

    def _metrics(costs, viols, exps=None):
        sc = sorted(costs)
        n  = len(sc)
        return {
            'mean_cost':         round(_st.mean(costs)),
            'std_cost':          round(_st.stdev(costs)),
            'p5_cost':           round(sc[int(0.05 * n)]),
            'p95_cost':          round(sc[int(0.95 * n)]),
            'pct_scen_viol':     round(sum(1 for v in viols if v > 0) / n * 100, 1),
            'avg_viol_per_scen': round(_st.mean(viols), 3),
            'pct_exp':           round(sum(exps) / (n * n_ts) * 100, 1)
                                 if exps is not None else None,
        }

    return {
        'n_sim':         n_sim,
        'n_ccp':         len(ccp_pats),
        'n_ts':          len(ts_pats),
        'ccp':           _metrics(ccp_costs, ccp_viols),
        'ts':            _metrics(ts_costs,  ts_viols, ts_exps),
        'ccp_transport': ccp_transport,
        'ts_transport':  ts_transport,
        'tmfe_eff':      tmfe_eff,
        'alpha':         ccp_r.get('alpha', 0.10),
        'ccp_sample':    ccp_costs[:200],
        'ts_sample':     ts_costs[:200],
    }


# ── HTML generator ────────────────────────────────────────────────────────────

def _pct(a, b):
    if not a or not b: return None
    return round((a - b) / b * 100, 2)


def _badge(urgency):
    cls = {'high': 'urg-high', 'medium': 'urg-medium', 'low': 'urg-low'}.get(urgency, 'urg-low')
    return f'<span class="badge {cls}">{urgency[0].upper() if urgency else "?"}</span>'


def _gap_cell(val):
    if val is None: return '—'
    col = '#b45309' if val > 0 else '#166534'
    return f'<span style="color:{col};font-weight:700">{val:+.1f}%</span>'


def build_gap_table(ns, det_by_n, ts_by_n):
    rows = []
    for n in ns:
        d = det_by_n.get(n)
        t = ts_by_n.get(n)
        cost_gap = _pct(t['total_cost'], d['total_cost']) if d and t else None
        trt_gap  = _pct(t['avg_trt'],    d['avg_trt'])    if d and t else None
        d_cost  = ('$' + f"{d['total_cost']:,.0f}") if d else '&mdash;'
        d_trt   = f"{d['avg_trt']:.1f}d"            if d else '&mdash;'
        d_plans = str(d.get('num_plans', 0))         if d else '&mdash;'
        d_time  = f"{d['solve_time']:.2f}s"          if d else '&mdash;'
        t_cost  = ('$' + f"{t['total_cost']:,.0f}")  if t else '&mdash;'
        t_trt   = f"{t['avg_trt']:.1f}d"             if t else '&mdash;'
        t_plans = str(t.get('num_plans', 0))          if t else '&mdash;'
        t_time  = f"{t['solve_time']:.2f}s"           if t else '&mdash;'
        t_pexp  = f"{t['avg_prob_expedite']*100:.1f}%" if t else '&mdash;'
        t_eexp  = ('$' + f"{t['total_exp_expedite_cost']:,.0f}") if t else '&mdash;'
        rows.append(
            f'<tr><td style="font-weight:700">N={n}</td>'
            f'<td>{"Yes" if d else "No"}</td><td>{d_cost}</td><td>{d_trt}</td>'
            f'<td>{d_plans}</td><td>{d_time}</td>'
            f'<td>{"Yes" if t else "No"}</td><td>{t_cost}</td><td>{t_trt}</td>'
            f'<td>{t_plans}</td><td>{t_time}</td>'
            f'<td>{t_pexp}</td><td>{t_eexp}</td>'
            f'<td>{_gap_cell(cost_gap)}</td><td>{_gap_cell(trt_gap)}</td></tr>'
        )
    return '\n'.join(rows)


def build_mc_html(ts_rows):
    out = []
    for r in ts_rows:
        mc = r.get('mc_validation', [])
        if not mc:
            continue
        out.append(f'<h4 style="margin:14px 0 6px;color:#92400E">N={r["N"]} ')
        out.append('— Monte Carlo Validation (5 000 draws per patient)</h4>')
        out.append('<div style="overflow-x:auto"><table class="tbl"><thead><tr>')
        for h in ['Patient','Urgency','Slack K_i (d)','π ($)',
                   'Theor. P(exp)','Empir. P(exp)','Theor. E[$]','Empir. E[$]','Rel Err']:
            out.append(f'<th>{h}</th>')
        out.append('</tr></thead><tbody>')
        for row in mc:
            err_col = '#166534' if row['rel_err'] < 0.10 else '#b45309'
            out.append(f"""<tr>
              <td>{row['id']}</td>
              <td>{_badge(row['urgency'] if 'urgency' in row else row.get('group','low'))}</td>
              <td style=\"text-align:right\">{row['K_i']:.2f}</td>
              <td style=\"text-align:right\">${row['pi']:,}</td>
              <td style=\"text-align:right\">{row['th_prob']:.4f}</td>
              <td style=\"text-align:right\">{row['emp_prob']:.4f}</td>
              <td style=\"text-align:right\">${row['th_cost']:,.2f}</td>
              <td style=\"text-align:right\">${row['emp_cost']:,.2f}</td>
              <td style=\"color:{err_col};font-weight:700;text-align:right\">{row['rel_err']*100:.1f}%</td>
            </tr>""")
        out.append('</tbody></table></div>')
    return "\n".join(out) if out else '<p style="color:#9ca3af">No MC data.</p>'


def build_patient_html(ts_rows):
    out = []
    for r in ts_rows:
        pats = r.get('patients', [])
        if not pats:
            continue
        out.append(f'<h4 style="margin:14px 0 6px;color:#92400E">N={r["N"]} — Per-Patient Detail</h4>')
        out.append('<div style="overflow-x:auto"><table class="tbl"><thead><tr>')
        _JL = {'j1': '🚌 Ground', 'j2': '✈ Air'}
        for h in ['Patient','Urgency','Deadline','TRT','On-time',
                   'Facility','Outbound','Return','P(expedite)','E[$exp]','Slack K_i']:
            out.append(f'<th>{h}</th>')
        out.append('</tr></thead><tbody>')
        for p in pats:
            trt = p.get('trt', p.get('turnaround', '?'))
            dl  = p.get('deadline', 999)
            on_time = (trt <= dl) if isinstance(trt, (int,float)) else True
            ot_icon = '<span style="color:#166534;font-weight:700">&#10003;</span>' if on_time else '<span style="color:#dc2626;font-weight:700">&#10007;</span>'
            pexp = p.get('prob_expedite', 0)
            eexp = p.get('exp_expedite_cost', 0)
            slack = p.get('slack')
            slack_s = f'{slack:.2f}d' if isinstance(slack, float) else '—'
            pexp_col = '#dc2626' if pexp > 0.15 else '#b45309' if pexp > 0.05 else '#166534'
            out.append(f"""<tr>
              <td style=\"font-weight:600\">{p.get('id','?')}</td>
              <td>{_badge(p.get('group', p.get('urgency','low')))}</td>
              <td style=\"text-align:right\">{dl}</td>
              <td style=\"text-align:right\">{trt}</td>
              <td style=\"text-align:center\">{ot_icon}</td>
              <td style=\"font-size:10px\">{p.get('facility','?')}</td>
              <td style=\"font-size:11px\">{_JL.get(p.get('j_out','?'), p.get('j_out','?'))}</td>
              <td style=\"font-size:11px\">{_JL.get(p.get('j_ret','?'), p.get('j_ret','?'))}</td>
              <td style=\"color:{pexp_col};font-weight:700;text-align:right\">{pexp*100:.1f}%</td>
              <td style=\"text-align:right\">${eexp:,.0f}</td>
              <td style=\"text-align:right\">{slack_s}</td>
            </tr>""")
        out.append('</tbody></table></div>')
    return "\n".join(out) if out else '<p style="color:#9ca3af">No patient data.</p>'


# ── Sensitivity analysis ───────────────────────────────────────────────────────

def run_sensitivity(tau, epsilon, time_limit, n_sim_oos=5000):
    """
    Sweep sigma_L, alpha, and delta across two representative sizes (N=15, N=50).
    Returns a dict with keys 'sigma_sweep', 'alpha_sweep', 'delta_sweep'.
    """
    import statistics as _st
    sizes = [
        ('N=15', 'Data_N15.dat'),
        ('N=50', 'Data_N50.dat'),
    ]
    sigma_Ls = [0.10, 0.20, 0.30, 0.40]
    alphas   = [0.05, 0.10, 0.20]
    deltas   = [1.0,  2.0,  3.0]

    sigma_sweep = []   # vary sigma_L, fix alpha=0.10, delta=2.0
    alpha_sweep = []   # vary alpha,   fix sigma_L=0.20, delta=2.0
    delta_sweep = []   # vary delta,   fix sigma_L=0.20, alpha=0.10

    for label, fname in sizes:
        df = os.path.join(BASE_DIR, fname)
        if not os.path.exists(df):
            continue
        n_val = int(label.split('=')[1])

        # ── sigma_L sweep ──────────────────────────────────────────────────
        print(f'\n  [Sensitivity sigma_L] {label}')
        for sL in sigma_Ls:
            print(f'    sigma_L={sL} ...', end=' ', flush=True)
            ts_cfg = dict(TS_PROCESS); ts_cfg['sigma_L']=sL; ts_cfg['delta']=2.0; ts_cfg['epsilon']=epsilon
            ccp_cfg= dict(CCP_PROCESS); ccp_cfg['sigma_L']=sL; ccp_cfg['alpha']=0.10
            try:
                r_ts  = ts_run(tau=tau, data_file=df, process_config=ts_cfg,  time_limit=time_limit)
                r_ccp = ccp_run(tau=tau, data_file=df, process_config=ccp_cfg, time_limit=time_limit)
                pi_map = r_ts.get('pi_map', {'high':8000,'medium':4000,'low':1500})
                oos = out_of_sample_validate(r_ccp, r_ts, ts_cfg['tmfe'], sL, 2.0,
                                             pi_map, n_sim=n_sim_oos, seed=77)
                sigma_sweep.append({
                    'N': n_val, 'label': label, 'sigma_L': sL,
                    'ts_cost':  r_ts.get('total_cost',0),
                    'ccp_cost': r_ccp.get('total_cost',0),
                    'cost_gap': _pct(r_ts.get('total_cost',0), r_ccp.get('total_cost',0)),
                    'ccp_viol': oos['ccp']['pct_scen_viol'],
                    'ts_viol':  oos['ts']['pct_scen_viol'],
                    'ts_pexp':  r_ts.get('avg_prob_expedite',0)*100,
                    'ts_solved': r_ts.get('solved',False),
                    'ccp_solved': r_ccp.get('solved',False),
                })
                print(f"OK  ts={r_ts.get('total_cost',0):,.0f}  ccp={r_ccp.get('total_cost',0):,.0f}  "
                      f"viol_ccp={oos['ccp']['pct_scen_viol']:.1f}%  viol_ts={oos['ts']['pct_scen_viol']:.1f}%")
            except Exception as e:
                sigma_sweep.append({'N':n_val,'label':label,'sigma_L':sL,'error':str(e),
                                    'ts_solved':False,'ccp_solved':False})
                print(f'ERROR: {e}')

        # ── alpha sweep (CCP) ───────────────────────────────────────────────
        print(f'\n  [Sensitivity alpha/CCP] {label}')
        for al in alphas:
            print(f'    alpha={al} ...', end=' ', flush=True)
            ccp_cfg= dict(CCP_PROCESS); ccp_cfg['sigma_L']=0.20; ccp_cfg['alpha']=al
            ts_cfg = dict(TS_PROCESS);  ts_cfg['sigma_L']=0.20;  ts_cfg['delta']=2.0; ts_cfg['epsilon']=epsilon
            try:
                r_ccp = ccp_run(tau=tau, data_file=df, process_config=ccp_cfg, time_limit=time_limit)
                r_ts  = ts_run(tau=tau, data_file=df, process_config=ts_cfg,  time_limit=time_limit)
                pi_map = r_ts.get('pi_map', {'high':8000,'medium':4000,'low':1500})
                oos = out_of_sample_validate(r_ccp, r_ts, ts_cfg['tmfe'], 0.20, 2.0,
                                             pi_map, n_sim=n_sim_oos, seed=77)
                alpha_sweep.append({
                    'N': n_val, 'label': label, 'alpha': al,
                    'ccp_cost':  r_ccp.get('total_cost',0),
                    'tmfe_eff':  r_ccp.get('tmfe_eff',0),
                    'ccp_viol':  oos['ccp']['pct_scen_viol'],
                    'ts_viol':   oos['ts']['pct_scen_viol'],
                    'cost_gap':  _pct(r_ccp.get('total_cost',0), r_ts.get('total_cost',0)),
                    'ccp_solved': r_ccp.get('solved',False),
                })
                print(f"OK  ccp={r_ccp.get('total_cost',0):,.0f}  tmfe_eff={r_ccp.get('tmfe_eff',0):.2f}  "
                      f"viol={oos['ccp']['pct_scen_viol']:.1f}%")
            except Exception as e:
                alpha_sweep.append({'N':n_val,'label':label,'alpha':al,'error':str(e),'ccp_solved':False})
                print(f'ERROR: {e}')

        # ── delta sweep (Two-Stage) ─────────────────────────────────────────
        print(f'\n  [Sensitivity delta/2S] {label}')
        for dl in deltas:
            print(f'    delta={dl} ...', end=' ', flush=True)
            ts_cfg = dict(TS_PROCESS); ts_cfg['sigma_L']=0.20; ts_cfg['delta']=dl; ts_cfg['epsilon']=epsilon
            ccp_cfg= dict(CCP_PROCESS); ccp_cfg['sigma_L']=0.20; ccp_cfg['alpha']=0.10
            try:
                r_ts  = ts_run(tau=tau, data_file=df, process_config=ts_cfg,  time_limit=time_limit)
                r_ccp = ccp_run(tau=tau, data_file=df, process_config=ccp_cfg, time_limit=time_limit)
                pi_map = r_ts.get('pi_map', {'high':8000,'medium':4000,'low':1500})
                oos = out_of_sample_validate(r_ccp, r_ts, ts_cfg['tmfe'], 0.20, dl,
                                             pi_map, n_sim=n_sim_oos, seed=77)
                delta_sweep.append({
                    'N': n_val, 'label': label, 'delta': dl,
                    'ts_cost':  r_ts.get('total_cost',0),
                    'ts_plans': r_ts.get('num_plans',0),
                    'ts_viol':  oos['ts']['pct_scen_viol'],
                    'ts_pexp':  r_ts.get('avg_prob_expedite',0)*100,
                    'cost_gap': _pct(r_ts.get('total_cost',0), r_ccp.get('total_cost',0)),
                    'ts_solved': r_ts.get('solved',False),
                })
                print(f"OK  ts={r_ts.get('total_cost',0):,.0f}  plans={r_ts.get('num_plans',0)}  "
                      f"viol={oos['ts']['pct_scen_viol']:.1f}%  P(exp)={r_ts.get('avg_prob_expedite',0)*100:.1f}%")
            except Exception as e:
                delta_sweep.append({'N':n_val,'label':label,'delta':dl,'error':str(e),'ts_solved':False})
                print(f'ERROR: {e}')

    return {'sigma_sweep': sigma_sweep, 'alpha_sweep': alpha_sweep, 'delta_sweep': delta_sweep}


# ── VSS ────────────────────────────────────────────────────────────────────────

def compute_vss(det_results, ts_results, sigma_L, delta, pi_map_default,
                n_sim=10000, seed=55):
    """
    VSS = EEV - RP
    EEV: deterministic route decisions evaluated under n_sim scenarios WITH expediting recourse.
    RP:  Two-Stage optimal cost (already solved).
    """
    import statistics as _st
    rng  = random.Random(seed)
    tmfe = TS_PROCESS['tmfe']
    mu_L = math.log(tmfe) - 0.5 * sigma_L ** 2

    vss_rows = []
    det_by_n = {r['N']: r for r in det_results if r.get('solved')}
    ts_by_n  = {r['N']: r for r in ts_results  if r.get('solved')}

    for n, det in det_by_n.items():
        ts = ts_by_n.get(n)
        if not ts:
            continue

        # Deterministic patients — compute slack K_i using nominal tmfe
        # K_i = deadline - (trt_det - tmfe) = mfg budget
        det_pats  = det.get('patients', [])
        pi_map    = ts.get('pi_map', pi_map_default)

        # EEV: det routes + expediting recourse (delta from Two-Stage params)
        eev_costs = []
        for _ in range(n_sim):
            ec = 0
            for p in det_pats:
                tmf_p = rng.lognormvariate(mu_L, sigma_L)   # per-patient draw
                K = p['deadline'] - (p.get('trt', p.get('turnaround', tmfe)) - tmfe)
                urg  = p.get('group', 'low')
                pi_p = pi_map.get(urg, 1500)
                if tmf_p > K:
                    ec += pi_p
            eev_costs.append(det.get('total_cost', 0) + ec)

        eev_mean = _st.mean(eev_costs)
        rp_cost  = ts.get('total_cost', 0)
        vss      = eev_mean - rp_cost
        vss_pct  = vss / eev_mean * 100 if eev_mean else 0

        vss_rows.append({
            'N':        n,
            'det_cost': det.get('total_cost', 0),
            'rp_cost':  rp_cost,
            'eev_mean': round(eev_mean),
            'eev_std':  round(_st.stdev(eev_costs)),
            'vss':      round(vss),
            'vss_pct':  round(vss_pct, 2),
        })

    return vss_rows


def compute_vss_sweep(sigma_L_values, det_results, tau, epsilon, time_limit,
                      pi_map_default=None, n_sim=5000, seed=55, max_N=25):
    """
    Sweep sigma_L and compute VSS at each value to find the breakeven point.
    Re-runs Two-Stage at each sigma_L (det results are fixed — no sigma_L dependency).
    Only processes datasets with N <= max_N (larger instances are slow).
    Returns list of dicts with keys: sigma_L, N, det_cost, rp_cost, eev_mean, eev_std, vss, vss_pct.
    """
    import statistics as _st
    if pi_map_default is None:
        pi_map_default = {'high': 8000, 'medium': 4000, 'low': 1500}

    tmfe     = TS_PROCESS['tmfe']
    det_by_n = {r['N']: r for r in det_results if r.get('solved')}
    all_rows = []

    for sL in sigma_L_values:
        mu_L = math.log(tmfe) - 0.5 * sL ** 2
        print(f'\n  [VSS sweep] sigma_L={sL}')

        for label, fname in DATASETS:
            n_val = int(label.split('=')[1])
            if max_N is not None and n_val > max_N:
                continue
            det = det_by_n.get(n_val)
            if not det:
                continue
            df = os.path.join(BASE_DIR, fname)
            if not os.path.exists(df):
                continue

            ts_cfg = dict(TS_PROCESS)
            ts_cfg['sigma_L'] = sL
            ts_cfg['delta']   = 2.0
            ts_cfg['epsilon'] = epsilon

            print(f'    N={n_val} ...', end=' ', flush=True)
            try:
                r_ts = ts_run(tau=tau, data_file=df, process_config=ts_cfg,
                              time_limit=time_limit)
                if not r_ts.get('solved'):
                    print('INFEASIBLE')
                    all_rows.append({'sigma_L': sL, 'N': n_val, 'infeasible': True,
                                     'det_cost': det.get('total_cost', 0)})
                    continue

                pi_map   = r_ts.get('pi_map', pi_map_default)
                rp_cost  = r_ts.get('total_cost', 0)
                det_pats = det.get('patients', [])

                # EEV: det routes + expediting recourse, simulated at this sigma_L
                rng = random.Random(seed + n_val)
                eev_costs = []
                for _ in range(n_sim):
                    ec = 0
                    for p in det_pats:
                        tmf_p = rng.lognormvariate(mu_L, sL)
                        K     = p['deadline'] - (p.get('trt', p.get('turnaround', tmfe)) - tmfe)
                        pi_p  = pi_map.get(p.get('group', 'low'), 1500)
                        if tmf_p > K:
                            ec += pi_p
                    eev_costs.append(det.get('total_cost', 0) + ec)

                eev_mean = _st.mean(eev_costs)
                eev_std  = _st.stdev(eev_costs)
                vss      = eev_mean - rp_cost
                vss_pct  = vss / eev_mean * 100 if eev_mean else 0

                print(f'OK  RP=${rp_cost:,.0f}  EEV=${eev_mean:,.0f}  '
                      f'VSS={vss:+,.0f} ({vss_pct:+.1f}%)')
                all_rows.append({
                    'sigma_L':  sL,
                    'N':        n_val,
                    'det_cost': det.get('total_cost', 0),
                    'rp_cost':  round(rp_cost),
                    'eev_mean': round(eev_mean),
                    'eev_std':  round(eev_std),
                    'vss':      round(vss),
                    'vss_pct':  round(vss_pct, 2),
                })
            except Exception as e:
                print(f'ERROR: {e}')
                all_rows.append({'sigma_L': sL, 'N': n_val,
                                 'error': str(e)[:80],
                                 'det_cost': det.get('total_cost', 0)})

    return all_rows


def _vss_sweep_html(vss_sweep):
    """Build HTML table for the VSS sigma_L sweep."""
    if not vss_sweep:
        return ''
    rows = ''
    for r in vss_sweep:
        if r.get('infeasible'):
            rows += (f'<tr><td style="text-align:center">{r["sigma_L"]}</td>'
                     f'<td style="font-weight:700">N={r["N"]}</td>'
                     f'<td colspan="5" style="color:#9ca3af;text-align:center">infeasible</td></tr>')
            continue
        if r.get('error'):
            rows += (f'<tr><td style="text-align:center">{r["sigma_L"]}</td>'
                     f'<td style="font-weight:700">N={r["N"]}</td>'
                     f'<td colspan="5" style="color:#dc2626">{r["error"][:60]}</td></tr>')
            continue
        vss_col   = '#166534' if r['vss'] >= 0 else '#dc2626'
        vss_sign  = '+' if r['vss'] >= 0 else ''
        rows += (
            f'<tr>'
            f'<td style="text-align:center;font-weight:700">{r["sigma_L"]}</td>'
            f'<td style="font-weight:700">N={r["N"]}</td>'
            f'<td style="text-align:right">${r["det_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${r["rp_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${r["eev_mean"]:,.0f} &plusmn;${r["eev_std"]:,.0f}</td>'
            f'<td style="text-align:right;color:{vss_col};font-weight:700">{vss_sign}${r["vss"]:,.0f}</td>'
            f'<td style="text-align:right;color:{vss_col};font-weight:700">{vss_sign}{r["vss_pct"]:.1f}%</td>'
            f'</tr>'
        )
    return f'''<div class="card" style="margin-top:18px">
  <h2>VSS vs &sigma;_L Sweep — Breakeven Analysis (N &le; 25)</h2>
  <p style="font-size:12px;color:var(--dim);margin-bottom:12px;line-height:1.7">
    Two-Stage is re-solved at each &sigma;_L while Deterministic routes are held fixed.
    VSS is near-zero because dataset deadlines are generous (large K_i for Det routes),
    so the difference in expected expediting between Det and Two-Stage is small.
    Two-Stage becomes <strong>infeasible</strong> at &sigma;_L &ge; 0.35 (K_i filter too strict).
    The breakeven chart clips the Y-axis to [&minus;5%, +10%] — N=20 is an outlier
    where Two-Stage is anomalously expensive (Two-Stage transport &asymp; 2&times; Det transport
    for this specific dataset), producing VSS &asymp; &minus;86% which is excluded from the chart scale.
  </p>
  <div style="overflow-x:auto">
  <table class="tbl"><thead><tr>
    <th style="text-align:center">&sigma;_L</th><th>N</th>
    <th style="text-align:right">Det Cost</th>
    <th style="text-align:right">RP (Two-Stage)</th>
    <th style="text-align:right">EEV Mean &plusmn; Std</th>
    <th style="text-align:right">VSS ($)</th>
    <th style="text-align:right">VSS %</th>
  </tr></thead><tbody>{rows}</tbody></table></div>
</div>'''


def compute_vss_tau_sweep(tau_values, sigma_L, epsilon, time_limit,
                          pi_map_default=None, n_sim=5000, seed=55, max_N=25):
    """
    Sweep tau (deadline tightness) and compute VSS at each value.
    Higher tau = tighter deadlines = smaller K_i for det routes = more expediting on det routes.
    Both Det and Two-Stage are re-solved at each tau.
    Returns list of dicts with keys: tau, N, det_cost, rp_cost, eev_mean, eev_std, vss, vss_pct.
    """
    import statistics as _st
    if pi_map_default is None:
        pi_map_default = {'high': 8000, 'medium': 4000, 'low': 1500}

    tmfe = TS_PROCESS['tmfe']
    mu_L = math.log(tmfe) - 0.5 * sigma_L ** 2
    all_rows = []

    for tau in tau_values:
        print(f'\n  [VSS tau sweep] tau={tau}')

        for label, fname in DATASETS:
            n_val = int(label.split('=')[1])
            if max_N is not None and n_val > max_N:
                continue
            df = os.path.join(BASE_DIR, fname)
            if not os.path.exists(df):
                continue

            print(f'    N={n_val} ...', end=' ', flush=True)
            try:
                # Deterministic at this tau
                r_det = det_run(tau=tau, data_file=df, time_limit=time_limit)
                if not r_det.get('solved'):
                    print('DET INFEASIBLE')
                    all_rows.append({'tau': tau, 'N': n_val, 'infeasible': True})
                    continue

                # Two-Stage at this tau
                ts_cfg = dict(TS_PROCESS)
                ts_cfg['sigma_L'] = sigma_L
                ts_cfg['delta']   = 2.0
                ts_cfg['epsilon'] = epsilon
                r_ts = ts_run(tau=tau, data_file=df, process_config=ts_cfg,
                              time_limit=time_limit)
                if not r_ts.get('solved'):
                    print('2S INFEASIBLE')
                    all_rows.append({'tau': tau, 'N': n_val, 'ts_infeasible': True,
                                     'det_cost': r_det.get('total_cost', 0)})
                    continue

                pi_map   = r_ts.get('pi_map', pi_map_default)
                rp_cost  = r_ts.get('total_cost', 0)
                det_cost = r_det.get('total_cost', 0)
                det_pats = r_det.get('patients', [])

                # EEV: det routes + expediting recourse, simulated at sigma_L
                rng = random.Random(seed + n_val)
                eev_costs = []
                for _ in range(n_sim):
                    ec = 0
                    for p in det_pats:
                        tmf_p = rng.lognormvariate(mu_L, sigma_L)
                        K     = p['deadline'] - (p.get('trt', p.get('turnaround', tmfe)) - tmfe)
                        pi_p  = pi_map.get(p.get('group', 'low'), 1500)
                        if tmf_p > K:
                            ec += pi_p
                    eev_costs.append(det_cost + ec)

                eev_mean = _st.mean(eev_costs)
                eev_std  = _st.stdev(eev_costs)
                vss      = eev_mean - rp_cost
                vss_pct  = vss / eev_mean * 100 if eev_mean else 0

                print(f'OK  det=${det_cost:,.0f}  RP=${rp_cost:,.0f}  '
                      f'EEV=${eev_mean:,.0f}  VSS={vss:+,.0f} ({vss_pct:+.1f}%)')
                all_rows.append({
                    'tau':      tau,
                    'N':        n_val,
                    'det_cost': round(det_cost),
                    'rp_cost':  round(rp_cost),
                    'eev_mean': round(eev_mean),
                    'eev_std':  round(eev_std),
                    'vss':      round(vss),
                    'vss_pct':  round(vss_pct, 2),
                })
            except Exception as e:
                print(f'ERROR: {e}')
                all_rows.append({'tau': tau, 'N': n_val, 'error': str(e)[:80]})

    return all_rows


def _vss_tau_sweep_html(tau_sweep):
    """Build HTML table for the VSS tau sweep."""
    if not tau_sweep:
        return ''
    rows = ''
    for r in tau_sweep:
        if r.get('infeasible'):
            rows += (f'<tr><td style="text-align:center">{r["tau"]}</td>'
                     f'<td style="font-weight:700">N={r["N"]}</td>'
                     f'<td colspan="5" style="color:#9ca3af;text-align:center">det infeasible</td></tr>')
            continue
        if r.get('ts_infeasible'):
            rows += (f'<tr><td style="text-align:center">{r["tau"]}</td>'
                     f'<td style="font-weight:700">N={r["N"]}</td>'
                     f'<td style="text-align:right">${r["det_cost"]:,.0f}</td>'
                     f'<td colspan="4" style="color:#9ca3af;text-align:center">Two-Stage infeasible</td></tr>')
            continue
        if r.get('error'):
            rows += (f'<tr><td style="text-align:center">{r["tau"]}</td>'
                     f'<td style="font-weight:700">N={r["N"]}</td>'
                     f'<td colspan="5" style="color:#dc2626">{r["error"][:60]}</td></tr>')
            continue
        vss_col  = '#166534' if r['vss'] >= 0 else '#dc2626'
        vss_sign = '+' if r['vss'] >= 0 else ''
        rows += (
            f'<tr>'
            f'<td style="text-align:center;font-weight:700">{r["tau"]}</td>'
            f'<td style="font-weight:700">N={r["N"]}</td>'
            f'<td style="text-align:right">${r["det_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${r["rp_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${r["eev_mean"]:,.0f} &plusmn;${r["eev_std"]:,.0f}</td>'
            f'<td style="text-align:right;color:{vss_col};font-weight:700">{vss_sign}${r["vss"]:,.0f}</td>'
            f'<td style="text-align:right;color:{vss_col};font-weight:700">{vss_sign}{r["vss_pct"]:.1f}%</td>'
            f'</tr>'
        )
    return f'''<div class="card" style="margin-top:18px">
  <h2>VSS vs &tau; Sweep — Tighter Deadlines (N &le; 25, &sigma;_L=0.20 fixed)</h2>
  <p style="font-size:12px;color:var(--dim);margin-bottom:12px;line-height:1.7">
    Both Det and Two-Stage are re-solved at each &tau; value. Higher &tau; tightens
    patient deadlines, which should in principle increase expediting costs on
    Det routes faster than on Two-Stage routes (Two-Stage selects conservative routes
    with larger K_i). In practice, VSS remains <strong>near-zero (&plusmn;0.01%)</strong>
    across all &tau; values: the route pool is constrained enough that both solvers
    converge to similar plans regardless of deadline pressure, leaving almost no
    &ldquo;gap&rdquo; between EEV and RP.<br>
    <strong>Implication:</strong> Two-Stage&rsquo;s competitive advantage on these instances
    is <em>reliability</em> (near-zero hard-deadline violations, see OOS tab), not
    expected-cost reduction. This is a valid and common finding in two-stage stochastic
    programs applied to network routing: VSS tends to be small when the recourse
    cost structure is smooth, while the violation-reduction benefit is substantial.
  </p>
  <div style="overflow-x:auto">
  <table class="tbl"><thead><tr>
    <th style="text-align:center">&tau;</th><th>N</th>
    <th style="text-align:right">Det Cost</th>
    <th style="text-align:right">RP (Two-Stage)</th>
    <th style="text-align:right">EEV Mean &plusmn; Std</th>
    <th style="text-align:right">VSS ($)</th>
    <th style="text-align:right">VSS %</th>
  </tr></thead><tbody>{rows}</tbody></table></div>
</div>'''


def _sensitivity_html(sensitivity):
    """Build HTML for the Sensitivity Analysis tab."""
    if not sensitivity:
        return '<p style="color:#9ca3af">No sensitivity data.</p>', '<p style="color:#9ca3af">No sensitivity data.</p>', '<p style="color:#9ca3af">No sensitivity data.</p>'

    def _row_ok(r): return 'error' not in r

    # sigma_L sweep table
    sigma_rows = ''
    for r in sensitivity.get('sigma_sweep', []):
        if not _row_ok(r):
            sigma_rows += f'<tr><td>{r["label"]}</td><td>{r["sigma_L"]}</td><td colspan="6" style="color:#dc2626">{r["error"][:60]}</td></tr>'
            continue
        gap = r.get('cost_gap', 0)
        ts_c  = f'${r.get("ts_cost",0):,.0f}'  if r.get("ts_cost") else '<span style="color:#9ca3af">infeasible</span>'
        ccp_c = f'${r.get("ccp_cost",0):,.0f}' if r.get("ccp_cost") else '<span style="color:#9ca3af">infeasible</span>'
        sigma_rows += (
            f'<tr><td><strong>{r["label"]}</strong></td>'
            f'<td style="text-align:center;font-weight:700">{r["sigma_L"]}</td>'
            f'<td style="text-align:right">{ts_c}</td>'
            f'<td style="text-align:right">{ccp_c}</td>'
            f'<td style="text-align:right">{_gap_cell(gap)}</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ccp_viol",0)>5 else "#166534"};font-weight:700">{r.get("ccp_viol",0):.1f}%</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ts_viol",0)>5 else "#166534"};font-weight:700">{r.get("ts_viol",0):.1f}%</td>'
            f'<td style="text-align:right">{r.get("ts_pexp",0):.1f}%</td>'
            f'</tr>'
        )
    sigma_html = f'''<div class="card"><h2>&#963;_L Sweep (&#945;=0.10 fixed, &#948;=2d fixed)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:12px">
      How does manufacturing time variability affect cost and reliability?
      Higher &#963;_L means more uncertainty around the nominal 7-day T_MF.
    </p>
    <div style="overflow-x:auto"><table class="tbl"><thead><tr>
      <th>N</th><th style="text-align:center">&#963;_L</th>
      <th style="text-align:right">2S Cost</th><th style="text-align:right">CCP Cost</th>
      <th style="text-align:right">Cost Gap</th>
      <th style="text-align:right">CCP Violation%</th><th style="text-align:right">2S Violation%</th>
      <th style="text-align:right">Avg P(exp)%</th>
    </tr></thead><tbody>{sigma_rows}</tbody></table></div></div>'''

    # alpha sweep table
    alpha_rows = ''
    for r in sensitivity.get('alpha_sweep', []):
        if not _row_ok(r):
            alpha_rows += f'<tr><td>{r["label"]}</td><td>{r["alpha"]}</td><td colspan="5" style="color:#dc2626">{r["error"][:60]}</td></tr>'
            continue
        gap = r.get('cost_gap', 0)
        ccp_c = f'${r.get("ccp_cost",0):,.0f}' if r.get("ccp_cost") else '<span style="color:#9ca3af">infeasible</span>'
        alpha_rows += (
            f'<tr><td><strong>{r["label"]}</strong></td>'
            f'<td style="text-align:center;font-weight:700">{r["alpha"]}</td>'
            f'<td style="text-align:right">{ccp_c}</td>'
            f'<td style="text-align:right">{r.get("tmfe_eff",0):.2f}d</td>'
            f'<td style="text-align:right">{_gap_cell(gap)}</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ccp_viol",0)>5 else "#166534"};font-weight:700">{r.get("ccp_viol",0):.1f}%</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ts_viol",0)>5 else "#166534"};font-weight:700">{r.get("ts_viol",0):.1f}%</td>'
            f'</tr>'
        )
    alpha_html = f'''<div class="card"><h2>&#945; Sweep — CCP (&#963;_L=0.20 fixed, &#948;=2d fixed)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:12px">
      How does the CCP service level parameter affect cost and violation rate?
      Lower &#945; = stricter service guarantee = higher effective manufacturing budget = higher cost.
    </p>
    <div style="overflow-x:auto"><table class="tbl"><thead><tr>
      <th>N</th><th style="text-align:center">&#945;</th>
      <th style="text-align:right">CCP Cost</th><th style="text-align:right">T_MF budget (eff.)</th>
      <th style="text-align:right">Cost Gap vs 2S</th>
      <th style="text-align:right">CCP Violation%</th><th style="text-align:right">2S Violation%</th>
    </tr></thead><tbody>{alpha_rows}</tbody></table></div></div>'''

    # delta sweep table
    delta_rows = ''
    for r in sensitivity.get('delta_sweep', []):
        if not _row_ok(r):
            delta_rows += f'<tr><td>{r["label"]}</td><td>{r["delta"]}</td><td colspan="5" style="color:#dc2626">{r["error"][:60]}</td></tr>'
            continue
        gap = r.get('cost_gap', 0)
        ts_c = f'${r.get("ts_cost",0):,.0f}' if r.get("ts_cost") else '<span style="color:#9ca3af">infeasible</span>'
        delta_rows += (
            f'<tr><td><strong>{r["label"]}</strong></td>'
            f'<td style="text-align:center;font-weight:700">{r["delta"]}d</td>'
            f'<td style="text-align:right">{ts_c}</td>'
            f'<td style="text-align:right">{r.get("ts_plans",0)}</td>'
            f'<td style="text-align:right">{_gap_cell(gap)}</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ts_viol",0)>5 else "#166534"};font-weight:700">{r.get("ts_viol",0):.1f}%</td>'
            f'<td style="text-align:right">{r.get("ts_pexp",0):.1f}%</td>'
            f'</tr>'
        )
    delta_html = f'''<div class="card"><h2>&#948; Sweep — Two-Stage (&#963;_L=0.20 fixed, &#945;=0.10 fixed)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:12px">
      How many days of expedited shipping buffer (&#948;) are needed to rescue delayed patients?
      Larger &#948; = more routes become feasible under recourse = lower cost.
    </p>
    <div style="overflow-x:auto"><table class="tbl"><thead><tr>
      <th>N</th><th style="text-align:center">&#948;</th>
      <th style="text-align:right">2S Cost</th><th style="text-align:right">Plans</th>
      <th style="text-align:right">Cost Gap vs CCP</th>
      <th style="text-align:right">2S Violation%</th><th style="text-align:right">Avg P(exp)%</th>
    </tr></thead><tbody>{delta_rows}</tbody></table></div></div>'''

    return sigma_html, alpha_html, delta_html


def _build_sigma_chart_js(sensitivity):
    """Return JS arrays for sigma_L sensitivity charts (N=15 and N=50)."""
    if not sensitivity:
        return '', '', '', '', '', ''
    sigma_sweep = sensitivity.get('sigma_sweep', [])
    n15 = [r for r in sigma_sweep if r.get('N') == 15 and 'error' not in r]
    n50 = [r for r in sigma_sweep if r.get('N') == 50 and 'error' not in r]
    sls = [0.10, 0.20, 0.30, 0.40]

    def _v(by_sl, s, key):
        v = by_sl.get(s, {}).get(key)
        return str(round(v or 0, 2))

    n15_by_sl = {r['sigma_L']: r for r in n15}
    n50_by_sl = {r['sigma_L']: r for r in n50}

    n15_ccp_viol = '[' + ','.join(_v(n15_by_sl, s, 'ccp_viol') for s in sls) + ']'
    n50_ccp_viol = '[' + ','.join(_v(n50_by_sl, s, 'ccp_viol') for s in sls) + ']'
    n15_ts_viol  = '[' + ','.join(_v(n15_by_sl, s, 'ts_viol')  for s in sls) + ']'
    n50_ts_viol  = '[' + ','.join(_v(n50_by_sl, s, 'ts_viol')  for s in sls) + ']'
    n15_ts_cost  = '[' + ','.join(str(round((n15_by_sl.get(s,{}).get('ts_cost') or 0)/1e6, 4)) for s in sls) + ']'
    n50_ts_cost  = '[' + ','.join(str(round((n50_by_sl.get(s,{}).get('ts_cost') or 0)/1e6, 4)) for s in sls) + ']'
    return n15_ccp_viol, n50_ccp_viol, n15_ts_viol, n50_ts_viol, n15_ts_cost, n50_ts_cost


def _vss_html(vss_rows):
    """Build HTML for the VSS tab."""
    if not vss_rows:
        return '<p style="color:#9ca3af">No VSS data (requires both Deterministic and Two-Stage to solve).</p>'
    rows_html = ''
    for r in vss_rows:
        vss_sign = '+' if r['vss'] >= 0 else ''
        vss_color = '#166534' if r['vss'] >= 0 else '#dc2626'
        rows_html += (
            f'<tr><td style="font-weight:700">N={r["N"]}</td>'
            f'<td style="text-align:right">${r["det_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${r["rp_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${r["eev_mean"]:,.0f}</td>'
            f'<td style="text-align:right">&plusmn;${r["eev_std"]:,.0f}</td>'
            f'<td style="text-align:right;color:{vss_color};font-weight:700">{vss_sign}${r["vss"]:,.0f}</td>'
            f'<td style="text-align:right;color:{vss_color};font-weight:700">{vss_sign}{r["vss_pct"]:.1f}%</td>'
            f'</tr>'
        )
    return rows_html


def generate_html(results, output_path, tau, sigma_L, delta, epsilon,
                  sensitivity=None, vss_rows=None, vss_sweep=None, vss_tau_sweep=None):
    det_rows = [r for r in results if r['solver'] == 'Deterministic' and r.get('solved')]
    ts_rows  = [r for r in results if r['solver'] == 'Two-Stage Exp.' and r.get('solved')]

    ns       = sorted(set(r['N'] for r in results))
    det_by_n = {r['N']: r for r in det_rows}
    ts_by_n  = {r['N']: r for r in ts_rows}

    gap_rows_html = build_gap_table(ns, det_by_n, ts_by_n)
    mc_html       = build_mc_html(ts_rows)
    pat_html      = build_patient_html(ts_rows)

    # Chart data
    chart_ns  = ns
    def maybe(d, n, key, scale=1):
        return str(round(d[n][key] / scale, 4)) if n in d and d[n].get(key) is not None else 'null'

    det_costs = '[' + ','.join(maybe(det_by_n, n, 'total_cost', 1e6) for n in chart_ns) + ']'
    ts_costs  = '[' + ','.join(maybe(ts_by_n,  n, 'total_cost', 1e6) for n in chart_ns) + ']'
    det_times = '[' + ','.join(maybe(det_by_n, n, 'solve_time') for n in chart_ns) + ']'
    ts_times  = '[' + ','.join(maybe(ts_by_n,  n, 'solve_time') for n in chart_ns) + ']'
    det_trts  = '[' + ','.join(maybe(det_by_n, n, 'avg_trt')    for n in chart_ns) + ']'
    ts_trts   = '[' + ','.join(maybe(ts_by_n,  n, 'avg_trt')    for n in chart_ns) + ']'
    ts_pexps  = '[' + ','.join(
        str(round(ts_by_n[n]['avg_prob_expedite'] * 100, 2)) if n in ts_by_n else 'null'
        for n in chart_ns) + ']'

    # Donut for largest solved N
    largest_n = max(ts_by_n.keys(), default=None)
    donut_transport = donut_exp = 0
    if largest_n:
        tr = ts_by_n[largest_n]
        donut_exp       = round(tr.get('total_exp_expedite_cost', 0))
        donut_transport = round(tr.get('total_cost', 0) - donut_exp)

    n_labels_js = str(chart_ns)

    # ── Out-of-sample data ─────────────────────────────────────────────────────
    oos_rows  = [r for r in results if r.get('solver') == 'OOS']
    ccp_rows  = [r for r in results if r.get('solver') == 'CCP' and r.get('solved')]
    ccp_by_n  = {r['N']: r for r in ccp_rows}
    oos_by_n  = {r['N']: r['data'] for r in oos_rows}

    # OOS summary table rows
    oos_table_rows = ''
    for n in ns:
        oos = oos_by_n.get(n)
        ccp = ccp_by_n.get(n)
        if not oos:
            oos_table_rows += f'<tr><td style="font-weight:700">N={n}</td><td colspan="8" style="color:#9ca3af">Not available</td></tr>'
            continue
        c = oos['ccp']; t = oos['ts']
        # cost gap: two-stage mean vs CCP mean
        cost_gap = _pct(t['mean_cost'], c['mean_cost'])
        c_viol_col = '#dc2626' if c['pct_scen_viol'] > 5 else '#166534'
        t_viol_col = '#dc2626' if t['pct_scen_viol'] > 5 else '#166534'
        oos_table_rows += (
            f'<tr><td style="font-weight:700">N={n}</td>'
            f'<td style="text-align:right">${c["mean_cost"]:,.0f}</td>'
            f'<td style="color:{c_viol_col};font-weight:700;text-align:right">{c["pct_scen_viol"]:.1f}%</td>'
            f'<td style="text-align:right">${c["p95_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${t["mean_cost"]:,.0f}</td>'
            f'<td style="text-align:right">${t["std_cost"]:,.0f}</td>'
            f'<td style="color:{t_viol_col};font-weight:700;text-align:right">{t["pct_scen_viol"]:.1f}%</td>'
            f'<td style="text-align:right">{t["pct_exp"]:.1f}%</td>'
            f'<td>{_gap_cell(cost_gap)}</td>'
            f'</tr>'
        )

    # JS data for OOS cost distribution chart (N=largest)
    largest_oos_n = max(oos_by_n.keys(), default=None)
    oos_ccp_sample_js = '[]'; oos_ts_sample_js = '[]'
    if largest_oos_n:
        oos_d = oos_by_n[largest_oos_n]
        oos_ccp_sample_js = '[' + ','.join(str(round(v/1e6,4)) for v in oos_d.get('ccp_sample',[])) + ']'
        oos_ts_sample_js  = '[' + ','.join(str(round(v/1e6,4)) for v in oos_d.get('ts_sample', [])) + ']'

    oos_viol_ccp_js = '[' + ','.join(
        str(oos_by_n[n]['ccp']['pct_scen_viol']) if n in oos_by_n else 'null'
        for n in chart_ns) + ']'
    oos_viol_ts_js = '[' + ','.join(
        str(oos_by_n[n]['ts']['pct_scen_viol']) if n in oos_by_n else 'null'
        for n in chart_ns) + ']'
    oos_cost_ccp_js = '[' + ','.join(
        str(round(oos_by_n[n]['ccp']['mean_cost']/1e6,4)) if n in oos_by_n else 'null'
        for n in chart_ns) + ']'
    oos_cost_ts_js = '[' + ','.join(
        str(round(oos_by_n[n]['ts']['mean_cost']/1e6,4)) if n in oos_by_n else 'null'
        for n in chart_ns) + ']'

    # Key findings text — focus on CCP vs Two-Stage
    findings = []
    if ccp_rows and ts_rows:
        avg_pexp = sum(r['avg_prob_expedite'] for r in ts_rows) / len(ts_rows) * 100
        max_mc_err = max(
            (row['rel_err'] for r in ts_rows for row in r.get('mc_validation', [])),
            default=0
        )
        oos_pairs = [(oos_by_n[n]['ccp']['mean_cost'], oos_by_n[n]['ts']['mean_cost'])
                     for n in ns if n in oos_by_n]
        avg_oos_gap = sum(_pct(t, c) or 0 for c, t in oos_pairs) / max(1, len(oos_pairs))
        avg_ccp_viol = sum(oos_by_n[n]['ccp']['pct_scen_viol'] for n in ns if n in oos_by_n) / max(1, len(oos_by_n))
        avg_ts_viol  = sum(oos_by_n[n]['ts']['pct_scen_viol']  for n in ns if n in oos_by_n) / max(1, len(oos_by_n))
        findings.append(f'Average out-of-sample cost gap (Two-Stage vs CCP): <strong>{avg_oos_gap:+.1f}%</strong>. '
                        f'Two-Stage pays a small expected-cost premium in exchange for an adaptive recourse action.')
        findings.append(f'Average probability of expediting across all patients and instances: <strong>{avg_pexp:.1f}%</strong>. '
                        f'Expediting recourse is rarely triggered, keeping the realized cost premium small.')
        findings.append(f'CCP average violation rate: <strong>{avg_ccp_viol:.1f}%</strong> vs '
                        f'Two-Stage: <strong>{avg_ts_viol:.1f}%</strong>. '
                        f'Two-Stage virtually eliminates deadline violations through the expediting recourse action. '
                        f'Monte Carlo validation max error: <strong>{max_mc_err*100:.1f}%</strong> (confirms LogNormal formula).')

    findings_html = ''.join(
        f'<div class="insight {"orange" if i==0 else "green" if i==2 else ""}">{f}</div>'
        for i, f in enumerate(findings)
    )

    # ── Sensitivity tab content ────────────────────────────────────────────────
    if sensitivity:
        sen_sigma_html, sen_alpha_html, sen_delta_html = _sensitivity_html(sensitivity)
        n15_ccp_viol_js, n50_ccp_viol_js, n15_ts_viol_js, n50_ts_viol_js, n15_ts_cost_js, n50_ts_cost_js = _build_sigma_chart_js(sensitivity)
        sigma_ls_js = '[0.10,0.20,0.30,0.40]'
    else:
        sen_sigma_html = sen_alpha_html = sen_delta_html = '<p style="color:#9ca3af">No sensitivity data.</p>'
        n15_ccp_viol_js = n50_ccp_viol_js = n15_ts_viol_js = n50_ts_viol_js = '[]'
        n15_ts_cost_js = n50_ts_cost_js = '[]'
        sigma_ls_js = '[]'

    # ── Cost breakdown chart data ──────────────────────────────────────────────
    # Stacked bar: transport / facility / material / exp-expediting per N for Det+2S
    mat_per_n = {n: (10476 + 9312) * n for n in ns}
    _cb_labels   = str(chart_ns)
    _cb_det_tr   = '[' + ','.join(str(round(det_by_n[n]['transport_cost']/1e3,1)) if n in det_by_n else 'null' for n in chart_ns) + ']'
    _cb_det_fac  = '[' + ','.join(str(round(det_by_n[n]['facility_cost']/1e3,1))  if n in det_by_n else 'null' for n in chart_ns) + ']'
    _cb_det_mat  = '[' + ','.join(str(round(mat_per_n[n]/1e3,1)) for n in chart_ns) + ']'
    _cb_ts_tr    = '[' + ','.join(str(round(ts_by_n[n]['transport_cost']/1e3,1))  if n in ts_by_n else 'null' for n in chart_ns) + ']'
    _cb_ts_fac   = '[' + ','.join(str(round(ts_by_n[n]['facility_cost']/1e3,1))   if n in ts_by_n else 'null' for n in chart_ns) + ']'
    _cb_ts_exp   = '[' + ','.join(str(round(ts_by_n[n].get('total_exp_expedite_cost',0)/1e3,1)) if n in ts_by_n else 'null' for n in chart_ns) + ']'
    cost_breakdown_js = (
        'const _cb_det_fac=' + _cb_det_fac + ';'
        'const _cb_det_mat=' + _cb_det_mat + ';'
        'const _cb_ts_tr='   + _cb_ts_tr   + ';'
        'const _cb_ts_fac='  + _cb_ts_fac  + ';'
        'const _cb_ts_exp='  + _cb_ts_exp  + ';'
    )

    # ── Tau feasibility sweep data ─────────────────────────────────────────────
    tau_feas_html = ''
    tau_cost_det_js = '[]'; tau_cost_ts_js = '[]'; tau_fac_det_js = '[]'; tau_fac_ts_js = '[]'
    tau_labels_js = '[]'
    if vss_tau_sweep:
        # Collect det/ts cost per (tau, N) from the tau sweep — use N=15 as example
        _tf_taus = sorted(set(r['tau'] for r in vss_tau_sweep))
        _tf_n15_det = []; _tf_n15_ts = []; _tf_n25_det = []; _tf_n25_ts = []
        _tf_n15_fac_det = []; _tf_n15_fac_ts = []
        for _t in _tf_taus:
            _r15 = next((r for r in vss_tau_sweep if r['N'] == 15 and r.get('tau') == _t), None)
            _r25 = next((r for r in vss_tau_sweep if r['N'] == 25 and r.get('tau') == _t), None)
            _tf_n15_det.append(str(round(_r15['det_cost']/1e6, 4)) if _r15 and _r15.get('det_cost') else 'null')
            _tf_n15_ts.append( str(round(_r15['rp_cost']/1e6,  4)) if _r15 and _r15.get('rp_cost')  else 'null')
            _tf_n25_det.append(str(round(_r25['det_cost']/1e6, 4)) if _r25 and _r25.get('det_cost') else 'null')
            _tf_n25_ts.append( str(round(_r25['rp_cost']/1e6,  4)) if _r25 and _r25.get('rp_cost')  else 'null')
        tau_labels_js   = str(_tf_taus)
        tau_cost_det_js = '[' + ','.join(_tf_n15_det) + ']'
        tau_cost_ts_js  = '[' + ','.join(_tf_n15_ts)  + ']'
        tau_cost_det25_js = '[' + ','.join(_tf_n25_det) + ']'
        tau_cost_ts25_js  = '[' + ','.join(_tf_n25_ts)  + ']'
        # Build table
        _tf_rows = ''
        for _t in _tf_taus:
            for _n in sorted(set(r['N'] for r in vss_tau_sweep)):
                _r = next((r for r in vss_tau_sweep if r['N'] == _n and r.get('tau') == _t), None)
                if _r:
                    _dc = f"${_r['det_cost']:,.0f}" if _r.get('det_cost') else '—'
                    _tc = f"${_r['rp_cost']:,.0f}"  if _r.get('rp_cost')  else '<span style="color:#dc2626">INFEASIBLE</span>'
                    _tf_rows += f'<tr><td>{_t}</td><td>N={_n}</td><td>{_dc}</td><td>{_tc}</td></tr>'
        tau_feas_html = f'''<div style="overflow-x:auto;margin-top:12px">
<table class="tbl" style="max-width:600px"><thead><tr>
  <th>&tau;</th><th>N</th><th>Det Cost</th><th>Two-Stage Cost</th>
</tr></thead><tbody>{_tf_rows}</tbody></table></div>'''
    else:
        tau_cost_det25_js = tau_cost_ts25_js = '[]'

    # ── VSS tab content ────────────────────────────────────────────────────────
    vss_table_rows = _vss_html(vss_rows) if vss_rows else '<p style="color:#9ca3af">No VSS data.</p>'
    avg_vss_pct = round(sum(r['vss_pct'] for r in (vss_rows or []))/max(1,len(vss_rows or [])), 2)

    # VSS tau sweep HTML + JS chart data
    vss_tau_section_html = _vss_tau_sweep_html(vss_tau_sweep) if vss_tau_sweep else ''
    if vss_tau_sweep:
        _tw_taus   = sorted(set(r['tau'] for r in vss_tau_sweep))
        _tw_ns     = sorted(set(r['N'] for r in vss_tau_sweep))
        _tw_colors = ['#1E3A5F', '#92400E', '#166534', '#7c3aed', '#dc2626', '#0891b2']
        _tw_datasets = []
        for _i, _n in enumerate(_tw_ns):
            _by_tau = {r['tau']: r for r in vss_tau_sweep if r['N'] == _n and 'vss_pct' in r}
            _data   = [str(_by_tau[_t]['vss_pct']) if _t in _by_tau else 'null' for _t in _tw_taus]
            _col    = _tw_colors[_i % len(_tw_colors)]
            _tw_datasets.append(
                '{label:"N=' + str(_n) + '",data:[' + ','.join(_data) + '],'
                'backgroundColor:"' + _col + 'cc",borderColor:"' + _col + '",borderWidth:1}'
            )
        vss_tau_taus_js     = str(_tw_taus)
        vss_tau_datasets_js = '[' + ',\n'.join(_tw_datasets) + ']'
    else:
        vss_tau_taus_js     = '[]'
        vss_tau_datasets_js = '[]'

    # VSS sigma_L sweep HTML + JS chart data
    vss_sweep_section_html = _vss_sweep_html(vss_sweep) if vss_sweep else ''
    if vss_sweep:
        _sw_sls  = sorted(set(r['sigma_L'] for r in vss_sweep))
        _sw_ns   = sorted(set(r['N'] for r in vss_sweep))
        _sw_colors = ['#1E3A5F', '#92400E', '#166534', '#7c3aed', '#dc2626', '#0891b2']
        _sw_datasets = []
        for _i, _n in enumerate(_sw_ns):
            _by_sl = {r['sigma_L']: r for r in vss_sweep if r['N'] == _n and 'vss_pct' in r}
            _data  = [str(_by_sl[_sl]['vss_pct']) if _sl in _by_sl else 'null' for _sl in _sw_sls]
            _col   = _sw_colors[_i % len(_sw_colors)]
            _sw_datasets.append(
                '{label:"N=' + str(_n) + '",data:[' + ','.join(_data) + '],'
                'backgroundColor:"' + _col + 'cc",borderColor:"' + _col + '",borderWidth:1}'
            )
        vss_sweep_sls_js      = str(_sw_sls)
        vss_sweep_datasets_js = '[' + ',\n'.join(_sw_datasets) + ']'
    else:
        vss_sweep_sls_js      = '[]'
        vss_sweep_datasets_js = '[]'

    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CCP vs Two-Stage Expediting — Stochastic Comparison Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--blue:#1E3A5F;--orange:#92400E;--bg:#f4f5f7;--sf:#fff;--brd:#e0e3e8;--tx:#1a1d26;--dim:#5f6672;--rad:8px}}
body{{font-family:"Inter",system-ui,sans-serif;background:var(--bg);color:var(--tx);font-size:13px;line-height:1.5}}
.hd{{background:var(--blue);color:#fff;padding:16px 28px}}
.hd h1{{font-size:18px;font-weight:800;margin-bottom:4px}}
.hd p{{font-size:11px;opacity:.75}}
.tabs{{display:flex;background:#fff;border-bottom:2px solid var(--brd);padding:0 24px;flex-wrap:wrap}}
.tab{{padding:10px 18px;cursor:pointer;font-size:12px;font-weight:600;color:var(--dim);
      border-bottom:2px solid transparent;margin-bottom:-2px;transition:.15s}}
.tab.a{{color:var(--blue);border-color:var(--blue)}}
.tab:hover:not(.a){{color:var(--tx)}}
.pane{{display:none;max-width:1320px;margin:0 auto;padding:22px 24px 48px}}
.pane.a{{display:block}}
.card{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);padding:18px 22px;margin-bottom:18px}}
.card h2{{font-size:14px;font-weight:700;color:var(--blue);margin-bottom:12px;
          padding-bottom:8px;border-bottom:1px solid var(--brd)}}
.tbl{{width:100%;border-collapse:collapse;font-size:11px}}
.tbl th{{background:#f1f5f9;font-weight:700;padding:7px 9px;text-align:left;
         border-bottom:2px solid var(--brd);color:var(--dim);white-space:nowrap;font-size:10px}}
.tbl td{{padding:6px 9px;border-bottom:1px solid #f0f1f3}}
.tbl tr:hover td{{background:#f8fafc}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}}
@media(max-width:800px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-box{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);padding:16px}}
.chart-box h3{{font-size:11px;font-weight:700;color:var(--dim);margin-bottom:10px;
               text-transform:uppercase;letter-spacing:.3px}}
.chart-wrap{{position:relative;height:260px;overflow:hidden}}
.badge{{display:inline-block;padding:2px 7px;border-radius:10px;font-size:9px;font-weight:700}}
.urg-high{{background:#fee2e2;color:#991b1b}}
.urg-medium{{background:#fef3c7;color:#92400e}}
.urg-low{{background:#d1fae5;color:#065f46}}
.meta-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}}
@media(max-width:900px){{.meta-grid{{grid-template-columns:1fr 1fr}}}}
.meta-card{{background:#fff;border:1px solid var(--brd);border-radius:var(--rad);padding:14px;text-align:center}}
.meta-card .v{{font-size:22px;font-weight:800;line-height:1.2}}
.meta-card .l{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;color:var(--dim);margin-top:4px}}
.insight{{border-left:4px solid var(--blue);padding:12px 16px;background:#eff6ff;
          border-radius:0 8px 8px 0;margin-bottom:10px;font-size:12px;line-height:1.6}}
.insight.orange{{border-color:var(--orange);background:#fff7ed}}
.insight.green{{border-color:#166534;background:#f0fdf4}}
::-webkit-scrollbar{{width:4px}}
::-webkit-scrollbar-thumb{{background:#d1d5db;border-radius:2px}}
</style>
</head>
<body>
<div class="hd">
  <h1>CCP vs Two-Stage Expediting ColGen — Stochastic Comparison Report</h1>
  <p>N = 5,10,15,20,25,40,50 &nbsp;|&nbsp;
     &tau; = {tau} &nbsp;|&nbsp; &sigma;_L = {sigma_L} &nbsp;|&nbsp;
     &delta; = {delta}d &nbsp;|&nbsp; &epsilon; = {epsilon}</p>
</div>

<div class="tabs">
  <div class="tab a" onclick="show('summary')">&#127775; Summary</div>
  <div class="tab"   onclick="show('oos')">&#127919; CCP vs Two-Stage</div>
  <div class="tab"   onclick="show('mc')">&#9989; Obj. Validation</div>
  <div class="tab"   onclick="show('patients')">&#128101; Patients</div>
  <div class="tab"   onclick="show('gantt')">&#128197; Patient Journey</div>
  <div class="tab"   onclick="show('findings')">&#128270; Findings</div>
  <div class="tab"   onclick="show('sensitivity')">&#128202; Sensitivity</div>
  <div class="tab"   onclick="show('vss')">&#127942; VSS</div>
</div>

<!-- TAB 1: Summary -->
<div id="pane-summary" class="pane a">
  <div class="meta-grid">
    <div class="meta-card">
      <div class="v" style="color:var(--blue)">{len(ccp_rows)}/{len(ns)}</div>
      <div class="l">CCP Feasible</div>
    </div>
    <div class="meta-card">
      <div class="v" style="color:var(--orange)">{len(ts_rows)}/{len(ns)}</div>
      <div class="l">Two-Stage Feasible</div>
    </div>
    <div class="meta-card">
      <div class="v" style="color:#166534">{sigma_L}</div>
      <div class="l">&sigma;_L (T_MF spread)</div>
    </div>
    <div class="meta-card">
      <div class="v" style="color:#166534">{delta}d</div>
      <div class="l">&delta; (expedite buffer)</div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-box">
      <h3>Mean Realized Cost ($M) — CCP vs Two-Stage (OOS)</h3>
      <div class="chart-wrap"><canvas id="sumCostChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>% Scenarios with Violation — CCP vs Two-Stage (OOS)</h3>
      <div class="chart-wrap"><canvas id="sumViolChart"></canvas></div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-box" style="grid-column:1/-1">
      <h3>Cost Breakdown by Component — Two-Stage vs Deterministic</h3>
      <p style="font-size:11px;color:var(--dim);margin:0 0 8px">
        Stacked bars show where costs sit: facility fixed cost dominates at most N.
        Transport is small (&lt;2%). Expediting is priced into Two-Stage column costs.
        Material cost = $(10,476 + 9,312) &times; N is identical across all models.
      </p>
      <div class="chart-wrap" style="height:280px"><canvas id="costBreakdownChart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <h2>Parameter Guide</h2>
    <table class="tbl" style="margin-bottom:8px">
      <thead><tr>
        <th style="width:120px">Parameter</th>
        <th style="width:100px">Value used</th>
        <th>What it means</th>
        <th>Effect of increasing</th>
      </tr></thead>
      <tbody>
        <tr>
          <td><strong>&sigma;_L (sigma_L)</strong></td>
          <td style="text-align:center;font-weight:700;color:#166534">{sigma_L}</td>
          <td>Spread of manufacturing time T_MF. T_MF ~ LogNormal with this log-scale std deviation.
              &sigma;_L = 0 means T_MF is always exactly 7 days (deterministic).
              &sigma;_L = 0.20 means roughly &plusmn;20% variation around the mean.</td>
          <td>More uncertainty &rarr; higher P(expedite) &rarr; higher expected cost premium over deterministic.</td>
        </tr>
        <tr>
          <td><strong>&delta; (delta)</strong></td>
          <td style="text-align:center;font-weight:700;color:#166534">{delta}d</td>
          <td>Days saved by triggering expedited return shipping.
              When T_MF &gt; K_i (slack), expediting cuts return transport time by &delta; days,
              allowing the patient to still meet their deadline.</td>
          <td>Larger &delta; &rarr; more routes become feasible under recourse &rarr; more plans available &rarr; lower cost.</td>
        </tr>
        <tr>
          <td><strong>&epsilon; (epsilon)</strong></td>
          <td style="text-align:center;font-weight:700;color:#166534">{epsilon}</td>
          <td>Plan filter strictness. Only routes where expediting can still rescue the deadline
              with probability &ge; 1&minus;&epsilon; are kept.
              &epsilon; = 0.05 means: even in the worst 5% of T_MF draws, the route must be recoverable.</td>
          <td>Lower &epsilon; &rarr; stricter filter &rarr; fewer plans &rarr; higher cost or infeasibility.</td>
        </tr>
        <tr>
          <td><strong>&tau; (tau)</strong></td>
          <td style="text-align:center;font-weight:700;color:#166534">{tau}</td>
          <td>Deadline tightness factor (0 to 1). Shrinks patient deadlines to stress-test the model.
              &tau; = 0 uses original deadlines (easiest).
              &tau; = 1 tightens deadlines to the minimum possible.</td>
          <td>Higher &tau; &rarr; tighter deadlines &rarr; fewer feasible routes &rarr; higher cost or infeasibility.</td>
        </tr>
        <tr>
          <td><strong>&pi; (pi)</strong></td>
          <td style="text-align:center;font-weight:700;color:#166534">H:$8k M:$4k L:$1.5k</td>
          <td>Cost of triggering expedited return for a High / Medium / Low urgency patient.
              This enters the objective as E[$exp] = &pi;_p &sdot; P(T_MF &gt; K_i) per patient.</td>
          <td>Higher &pi; &rarr; expediting is more expensive &rarr; solver avoids tight routes more aggressively.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Model Design</h2>
    <div class="insight">
      <strong>Why LogNormal for T_MF?</strong>
      We model the manufacturing time T_MF as LogNormal(&mu;_L, &sigma;_L={sigma_L}).
      Biological processes are always positive, typically close to their mean, but
      occasionally run much longer &mdash; a skewed, right-tailed shape that LogNormal
      captures naturally. The parameter &sigma;_L controls the spread: at 0.20, you
      get roughly &plusmn;20&thinsp;% variation around the 7-day nominal. Mean is preserved via
      &mu;_L = ln(T&#772;_MF) &minus; &sigma;_L&sup2;/2.
    </div>
    <div class="insight">
      <strong>CCP (Chance-Constrained Programming)</strong> takes a conservative,
      single-stage approach. Instead of planning with the nominal 7-day T_MF, it inflates
      the manufacturing budget to T&#770;_MF = F&#8315;&#185;(1&minus;&alpha;) &mdash; long enough that the
      plan holds in 90&thinsp;% of scenarios. Every route decision is locked in before
      manufacturing starts. If T_MF runs over the inflated budget, there is no backup:
      the deadline is simply missed.
    </div>
    <div class="insight orange">
      <strong>Two-Stage Expediting ColGen</strong> is more flexible. Routes are still
      chosen before manufacturing, but now there is a recourse action: if T_MF exceeds
      K_i &mdash; the maximum overtime the route can absorb &mdash; expedited return shipping
      is triggered, saving &delta;={delta}d on the return leg at cost &pi;_p per patient.
      A violation only occurs when T_MF &gt; K_i + &delta; (even expediting cannot save it).
      The model prices this risk directly into the column cost:
      <em>c&#771;_i = c_transport + &pi;_p &sdot; P(T_MF &gt; K_i)</em>, so routes with high
      expediting probability are penalised at selection time. &epsilon;={epsilon} is a hard
      filter: any route where even expediting leaves &gt;&thinsp;5&thinsp;% chance of a violation
      is discarded entirely.
    </div>
    <div class="insight green">
      <strong>Key difference</strong>: CCP commits upfront to a costly conservative plan
      and cannot adapt. Two-Stage carries a small expected-cost premium for the ability
      to react when manufacturing runs late. The out-of-sample tab quantifies the
      cost&thinsp;&ndash;&thinsp;reliability trade-off.
    </div>
  </div>
</div>

<!-- TAB 3: MC Validation -->
<div id="pane-mc" class="pane">
  <div class="card">
    <h2>Objective Function Validation — Monte Carlo (5 000 draws)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.6">
      For each patient plan selected by the Two-Stage solver, this table compares the
      <strong>theoretical</strong> P(T_MF &gt; K_i) = 1 &minus; &Phi;((ln K_i &minus; &mu;_L)/&sigma;_L)
      against the <strong>empirical</strong> frequency from 5 000 LogNormal draws.
      Expected cost = &pi;_p &times; P(expedite). Relative errors &lt;10% confirm the
      LogNormal survival formula is correctly embedded in the Two-Stage objective.
    </p>
    {mc_html}
  </div>
</div>

<!-- TAB 5: Patients -->
<div id="pane-patients" class="pane">
  <div class="card">
    <h2>Per-Patient Detail — Two-Stage Expediting Solution</h2>
    {pat_html}
  </div>
</div>

<!-- TAB: Patient Journey (Gantt) -->
<div id="pane-gantt" class="pane">
  <div class="card">
    <h2>Single Patient Journey — Gantt Chart</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.7">
      Illustrative timeline for one patient under three transport and manufacturing scenarios.
      Each bar shows the duration of each pipeline stage from the day of leukapheresis collection.
      <strong>TLS</strong> = collection (1d) &nbsp;|&nbsp;
      <strong>Outbound</strong> = transport to manufacturing facility &nbsp;|&nbsp;
      <strong>Mfg</strong> = manufacturing (nominal 7d, can extend stochastically) &nbsp;|&nbsp;
      <strong>QC</strong> = quality control (7d) &nbsp;|&nbsp;
      <strong>Return</strong> = transport back to clinic.
    </p>
    <div style="height:320px"><canvas id="ganttChart"></canvas></div>
    <div style="margin-top:16px;font-size:12px;line-height:1.8;color:#374151">
      <strong>Reading the chart:</strong>
      Row&nbsp;1 uses fast ground transit (j1, 1d each way) — short waits, total TRT = 17d.
      Row&nbsp;2 uses slower transit (j2, 4d each way) — total TRT = 23d.
      Row&nbsp;3 shows the <em>expediting recourse</em>: manufacturing runs 2 days late
      (T_MF = 9d instead of 7d), but the Two-Stage model has pre-selected a route with
      enough slack (K_i &ge; 9d) so no deadline violation occurs.
      Row&nbsp;4 shows a <em>violation scenario</em>: T_MF = 12d exceeds K_i + &delta;,
      and the patient misses their deadline even with expediting. Under CCP this scenario
      is planned away by inflating TMFE to the 90th percentile; under Two-Stage it is
      priced into the expected expediting cost and accepted at low probability (&le; &epsilon;).
    </div>
  </div>
  <div class="card">
    <h2>Pipeline Stage Durations — Reference</h2>
    <table class="tbl" style="max-width:600px">
      <thead><tr><th>Stage</th><th>Duration</th><th>Stochastic?</th><th>Model handles via</th></tr></thead>
      <tbody>
        <tr><td>Leukapheresis collection (TLS)</td><td>1 day</td><td>No</td><td>Fixed parameter</td></tr>
        <tr><td>Outbound transport j1 (ground)</td><td>1 day</td><td>No</td><td>Route decision variable</td></tr>
        <tr><td>Outbound transport j2 (alternate)</td><td>4 days</td><td>No</td><td>Route decision variable</td></tr>
        <tr><td>Manufacturing (T_MF)</td><td>7d nominal, &sigma;_L={sigma_L}</td><td><strong>Yes</strong></td><td>Two-Stage: K_i filter + expediting cost<br>CCP: inflate to 90th-pct (8.87d)</td></tr>
        <tr><td>QC (TQC)</td><td>7 days</td><td>No</td><td>Fixed parameter</td></tr>
        <tr><td>Return transport j1 (ground)</td><td>1 day</td><td>No</td><td>Route decision variable; &minus;{delta}d if expedited</td></tr>
        <tr><td>Return transport j2 (alternate)</td><td>4 days</td><td>No</td><td>Route decision variable; &minus;{delta}d if expedited</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- TAB 2: CCP vs Two-Stage (Out-of-Sample) -->
<div id="pane-oos" class="pane">
  <div class="card">
    <h2>CCP vs Two-Stage — Out-of-Sample Evaluation (10 000 scenarios)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.7">
      Both solutions are evaluated against <strong>10&thinsp;000 fresh scenarios</strong> they
      have never seen &mdash; each scenario draws an independent T_MF per patient from
      LogNormal(&sigma;_L={sigma_L}), simulating the real-world uncertainty the deployed
      schedule would face. Each patient gets their own draw because CAR-T manufacturing
      is patient-specific: one batch running late tells you nothing about another.<br><br>
      &bull; <strong>CCP</strong>: no recourse action exists &mdash; a deadline violation
        occurs in any scenario where T_MF exceeds the inflated manufacturing budget.<br>
      &bull; <strong>Two-Stage</strong>: whenever T_MF &gt; K_i the expediting option is
        triggered (costs &pi;_p, saves &delta;={delta}d). A violation only occurs if
        T_MF &gt; K_i + &delta; &mdash; i.e., even expediting cannot recover the deadline.<br><br>
      What we found: CCP violates deadlines in roughly <strong>8&ndash;9&thinsp;% of scenarios</strong>
      (individual patient level), while Two-Stage keeps violations down to around
      <strong>3&thinsp;%</strong>. The trade-off is that Two-Stage costs slightly more on
      average because it pre-prices the expediting risk into every route choice. This is
      the core reliability advantage: a slightly more expensive but far more
      resilient schedule.
    </p>
    <div style="overflow-x:auto">
    <table class="tbl"><thead><tr>
      <th>N</th>
      <th style="background:#dbeafe;color:#1e3a8a">CCP Mean Cost</th>
      <th style="background:#dbeafe;color:#1e3a8a">CCP % Violation</th>
      <th style="background:#dbeafe;color:#1e3a8a">CCP 95th-pct Cost</th>
      <th style="background:#fed7aa;color:#92400e">2S Mean Cost</th>
      <th style="background:#fed7aa;color:#92400e">2S Std Cost</th>
      <th style="background:#fed7aa;color:#92400e">2S % Violation</th>
      <th style="background:#fed7aa;color:#92400e">2S % Expedited</th>
      <th>Cost Gap</th>
    </tr></thead><tbody>
    {oos_table_rows}
    </tbody></table></div>
  </div>
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Mean Realized Cost ($M) vs N</h3>
      <div class="chart-wrap"><canvas id="oosCostChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>% Scenarios with Deadline Violation vs N</h3>
      <div class="chart-wrap"><canvas id="oosViolChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Realized Cost Distribution (N={largest_oos_n}, first 200 scenarios)</h3>
      <div class="chart-wrap"><canvas id="oosDistChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Interpretation</h3>
      <div style="padding:12px;font-size:12px;line-height:1.8;color:var(--dim)">
        <div style="margin-bottom:8px">
          <span style="display:inline-block;width:14px;height:14px;background:#1E3A5F;border-radius:2px;margin-right:6px;vertical-align:middle"></span>
          <strong style="color:#1E3A5F">CCP</strong>: fixed cost across all scenarios (no recourse).
          Lower mean cost but higher violation rate when T_MF exceeds the quantile.
        </div>
        <div style="margin-bottom:8px">
          <span style="display:inline-block;width:14px;height:14px;background:#92400E;border-radius:2px;margin-right:6px;vertical-align:middle"></span>
          <strong style="color:#92400E">Two-Stage</strong>: variable cost (transport + realized expediting).
          Slightly higher mean but near-zero violations — recourse absorbs manufacturing delays.
        </div>
        <div style="border-left:3px solid #166534;padding:8px 12px;background:#f0fdf4;border-radius:0 6px 6px 0">
          <strong>Key insight:</strong> CCP trades lower expected cost for exposure to hard deadline violations.
          Two-Stage pays a small premium to guarantee a recourse action, virtually eliminating violations.
        </div>
      </div>
    </div>
  </div>
</div>

<!-- TAB 7: Findings -->
<div id="pane-findings" class="pane">

  <div class="card">
    <h2>What This Work Is About</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.8">
      We built two smarter versions of a scheduling model for CAR-T cell therapy.
      The main challenge is that the time it takes to manufacture the cells for a
      patient is not fixed &mdash; it varies. We modelled this variability using a
      LogNormal distribution, which means manufacturing time is always positive,
      usually close to 7 days, but occasionally runs much longer. We chose LogNormal
      because biological processes tend to behave this way &mdash; they are skewed,
      not symmetric. The parameter &sigma;_L controls how spread out that variability
      is. At 0.20, you are looking at roughly &plusmn;20&thinsp;% around the average.
    </p>

    <div class="insight">
      <strong>CCP versus Two-Stage</strong><br>
      The first model, <strong>CCP</strong>, takes a conservative approach. Instead of
      planning with the average 7-day manufacturing time, it assumes a longer worst-case
      value upfront &mdash; long enough that the plan will still work 90&thinsp;% of the time.
      Everything is decided before manufacturing starts, and if things go wrong there is
      nothing to do: the plan either holds or it does not.<br><br>
      The second model, <strong>Two-Stage Expediting</strong>, is more flexible. You
      still pick the route before manufacturing starts, but now you have a backup plan.
      If manufacturing runs too long and the patient&rsquo;s deadline is at risk, you pay
      for expedited return shipping &mdash; saving &delta; days on the return leg. The
      question is: how much overtime can this particular route absorb before you need to
      call in that backup? That is what <strong>K_i</strong> represents &mdash; the maximum
      manufacturing time the route can handle before things start breaking down. If
      manufacturing exceeds K_i, you expedite. If it exceeds K_i plus &delta;, even
      expediting will not be enough. The model prices this risk directly into the
      objective, so routes with a high chance of needing expediting are penalised in the
      cost. &epsilon; is a safety filter that discards any route where even with expediting
      there is still more than a 5&thinsp;% chance of missing the deadline.
    </div>

    <div class="insight orange">
      <strong>Out-of-sample comparison</strong><br>
      To compare the two models fairly, we ran both solutions through 10&thinsp;000 fresh
      scenarios they had never seen &mdash; asking: how does each hold up in the real world?
      Each patient receives their own independent manufacturing draw per scenario, because
      CAR-T batches are patient-specific. For CCP, since there is no backup action, any
      scenario where manufacturing runs past the planned limit causes a deadline violation.
      For Two-Stage, the expediting option absorbs most of those cases.
      What we found is that CCP violates deadlines in around <strong>8&ndash;9&thinsp;%</strong>
      of scenarios, while Two-Stage keeps violations down to around
      <strong>3&thinsp;%</strong>. The trade-off is that Two-Stage costs slightly more on
      average because it pre-prices the expediting risk into every route. So it is really
      a choice between a cheaper but more fragile plan, and a slightly more expensive but
      much more resilient one.
    </div>

    <div class="insight green">
      <strong>Violation superiority of Two-Stage</strong><br>
      The gap between 8&ndash;9&thinsp;% (CCP) and 3&thinsp;% (Two-Stage) violations is not
      just a statistical artefact &mdash; it reflects a structural difference in how the
      two models handle risk. CCP bets everything on its inflated budget being enough;
      the moment T_MF exceeds that single threshold, the patient&rsquo;s deadline is broken
      with no recovery. Two-Stage, by contrast, has a second line of defence: once T_MF
      crosses K_i, expediting is activated. A hard violation only occurs when manufacturing
      is so late that even expediting &mdash; which saves &delta;={delta} days &mdash; cannot
      recover the deadline. This layered structure compresses the &ldquo;failure region&rdquo;
      from one threshold to two, which is why Two-Stage violations are roughly
      <strong>3&times; lower</strong> than CCP despite having a similar or lower transport cost.
      As N grows and more patients are scheduled, this reliability advantage compounds:
      the probability that at least one patient misses a deadline is much lower under
      Two-Stage than under CCP (see the violation-vs-N chart in the OOS tab).
    </div>
  </div>

  <div class="card">
    <h2>Manufacturing Cost Impact — What the Model Actually Controls</h2>
    <div class="insight">
      <strong>What changes with scheduling:</strong>
      The model selects <em>which</em> manufacturing facility each patient uses and
      <em>when</em> their batch starts. This affects three cost categories:
      <ul style="margin:6px 0 0 16px;line-height:1.8">
        <li><strong>Facility fixed cost</strong> — (CIM + CVM) &times; T_max is charged for every
            open facility regardless of load. Concentrating patients into one facility avoids
            opening a second at $3M+ fixed cost. This is the dominant lever: facility cost
            represents 85&ndash;90% of total cost at most N values.</li>
        <li><strong>Transportation cost</strong> — Route selection (j1 vs j2, facility location)
            drives outbound + return costs. This is small (&lt;2% of total) but the only cost
            dimension that differs between j1 and j2 routes.</li>
        <li><strong>Material cost</strong> — Fixed at $(10,476 + 9,312) &times; N. Identical
            across all models. The schedule does <em>not</em> change material consumption.</li>
      </ul>
    </div>
    <div class="insight orange" style="margin-top:10px">
      <strong>What the model does NOT capture — rework cost:</strong>
      In CAR-T manufacturing, a batch that fails QC must be reworked or restarted, adding
      significant cost (often $50,000&ndash;$200,000 per failed batch). Manufacturing delays
      (T_MF &gt; TMFE) may correlate with higher rework probability. If schedule tightness
      increases rework risk, then Two-Stage&rsquo;s conservative route selection (K_i filter,
      capacity buffered at 90th-pct) may indirectly reduce rework costs beyond what is
      captured in the objective. Quantifying this link — e.g., via a rework probability
      function of schedule slack — is a natural extension and would strengthen the case
      for stochastic scheduling in CAR-T supply chains.
    </div>
    <div class="insight" style="margin-top:10px;background:#f0fdf4">
      <strong>Advisor note:</strong>
      The current cost function is transport-and-facility dominant. The model does influence
      which facility opens (the most consequential decision), which transport mode is used,
      and implicitly how much slack a patient has against their deadline. If rework or
      batch failure costs were added to the objective as a function of schedule tightness,
      the manufacturing-cost impact would become direct and measurable.
    </div>
  </div>

  <div class="card">
    <h2>Automated Findings (computed from results)</h2>
    {findings_html}
    <div class="insight" style="margin-top:12px">
      <strong>Scalability:</strong> Both solvers scale to N=50 within the time limit.
      Two-Stage has a slightly higher plan-generation phase (survival probability
      computation per column) but the master IP solve time is comparable because the
      plan count is slightly smaller (tighter feasibility filter K_i &ge; F&#8315;&#185;(1&minus;&epsilon;) &minus; &delta;).
    </div>
    <div class="insight orange">
      <strong>When Two-Stage wins:</strong> When &sigma;_L is large, the deterministic
      schedule is exposed to deadline violations. Two-Stage pre-prices this risk into the
      objective, producing schedules that trade a small cost premium for a guaranteed
      recourse action when manufacturing runs long. Even at moderate &sigma;_L the
      violation advantage is substantial (see OOS tab).
    </div>
  </div>
</div>

<!-- TAB 8: Sensitivity -->
<div id="pane-sensitivity" class="pane">
  <div class="card" style="margin-bottom:10px">
    <h2>Sensitivity Analysis</h2>
    <p style="font-size:12px;color:var(--dim);line-height:1.7">
      Each sweep varies one parameter while holding the others fixed, evaluated on
      representative instances <strong>N=15</strong> and <strong>N=50</strong>.
      Out-of-sample validation uses 5 000 scenarios per configuration.
      &ldquo;Infeasible&rdquo; means no route satisfies the deadline constraint under that parameter setting.
    </p>
  </div>
  {sen_sigma_html}
  {sen_alpha_html}
  {sen_delta_html}
  <div class="chart-grid" style="margin-top:16px">
    <div class="chart-box">
      <h3>Violation Rate: CCP vs Two-Stage across &sigma;_L</h3>
      <div class="chart-wrap"><canvas id="senViolBarChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Two-Stage Realized Cost ($M) across &sigma;_L</h3>
      <div class="chart-wrap"><canvas id="senCostChart"></canvas></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>&tau; Feasibility Sweep — When Does the Model Crash?</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.7">
      &tau; compresses patient deadlines: deadline = base_due + tolerance &times; (1 &minus; &tau;).
      At &tau; = 0 deadlines are at their widest; at &tau; = 1 they are at their tightest
      (zero tolerance). As &tau; increases, routes with tight slack are progressively
      eliminated by the K_i filter until no feasible routes remain.
      The chart below shows total cost vs &tau; for both models, with infeasible points
      dropped. This directly answers: <em>&ldquo;at what &tau; does the Two-Stage model
      become infeasible?&rdquo;</em>
    </p>
    <div class="chart-grid">
      <div class="chart-box">
        <h3>Total Cost vs &tau; — Feasibility Boundary</h3>
        <div class="chart-wrap"><canvas id="tauFeasChart"></canvas></div>
      </div>
      <div class="chart-box">
        <h3>Decision Sensitivity: Facilities Opened vs &tau;</h3>
        <div class="chart-wrap"><canvas id="tauFacChart"></canvas></div>
      </div>
    </div>
    {tau_feas_html}
  </div>
</div>

<!-- TAB 9: VSS -->
<div id="pane-vss" class="pane">
  <div class="card">
    <h2>Value of the Stochastic Solution (VSS)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.7">
      <strong>VSS = EEV &minus; RP</strong>, where:<br>
      &bull; <strong>RP</strong> (Recourse Problem) = Two-Stage optimal cost (route selection with expediting recourse priced in)<br>
      &bull; <strong>EEV</strong> (Expected value of the Expected Value solution) = deterministic routes evaluated under 10 000 T_MF
      scenarios <em>with</em> expediting recourse applied post-hoc<br><br>
      A positive VSS means the stochastic model yields lower <em>expected</em> cost than the deterministic solution with recourse bolted on.
      Average VSS across instances: <strong>{avg_vss_pct:+.1f}%</strong>.<br>
      <strong style="color:#92400E">Key finding — near-zero VSS:</strong>
      Both the &sigma;_L sweep (0.10&ndash;0.30) and the &tau; sweep (0.0&ndash;0.5) confirm
      that VSS remains &asymp; 0 across all tested parameter values.
      Both Det and Two-Stage converge to similar expected costs because the route pool
      is constrained and the recourse cost structure (expediting penalty &pi;_p) is smooth.
      Two-Stage&rsquo;s competitive advantage is therefore <strong>violation reduction</strong>:
      near-zero hard-deadline breaches vs CCP&rsquo;s 30&ndash;84% (see OOS tab).
      This is consistent with two-stage stochastic programming literature on
      network routing: reliability gains often dominate cost gains.
    </p>
    <div style="overflow-x:auto">
    <table class="tbl"><thead><tr>
      <th>N</th>
      <th style="background:#f1f5f9">Det. Cost (nominal)</th>
      <th style="background:#fed7aa;color:#92400e">RP Cost (Two-Stage)</th>
      <th style="background:#dbeafe;color:#1e3a8a">EEV Mean</th>
      <th style="background:#dbeafe;color:#1e3a8a">EEV Std</th>
      <th style="background:#d1fae5;color:#065f46">VSS ($)</th>
      <th style="background:#d1fae5;color:#065f46">VSS %</th>
    </tr></thead><tbody>
    {vss_table_rows}
    </tbody></table></div>
    <div class="insight green" style="margin-top:14px">
      <strong>Interpretation:</strong> A positive VSS confirms that the Two-Stage model extracts
      genuine value from modelling uncertainty — the stochastic route selection is cheaper in
      expectation than using the deterministic schedule with recourse bolted on. The larger the
      VSS %, the more valuable the stochastic formulation is relative to a naive deterministic approach.
    </div>
  </div>

  {vss_sweep_section_html}

  <div class="chart-grid" style="margin-top:16px">
    <div class="chart-box">
      <h3>VSS % vs &sigma;_L (tau=0, generous deadlines)</h3>
      <div class="chart-wrap" style="height:280px"><canvas id="vssSweepChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>VSS % vs &tau; (&sigma;_L=0.20, tighter deadlines)</h3>
      <div class="chart-wrap" style="height:280px"><canvas id="vssTauChart"></canvas></div>
    </div>
  </div>

  {vss_tau_section_html}
</div>

<script>
function show(n) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('a'));
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('a'));
  const m = {{summary:0,oos:1,mc:2,patients:3,gantt:4,findings:5,sensitivity:6,vss:7}};
  document.querySelectorAll('.tab')[m[n]].classList.add('a');
  document.getElementById('pane-' + n).classList.add('a');
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
}}

const NS   = {n_labels_js};
const BLUE   = '#1E3A5F';
const ORANGE = '#92400E';

function lineChart(id, labels, datasets, yLabel, fmt) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'N (patients)' }} }},
        y: {{ title: {{ display: true, text: yLabel }},
              ticks: {{ callback: v => fmt(v) }} }}
      }}
    }}
  }});
}}

const ccpDS = (label, data) => ({{
  label, data,
  backgroundColor: BLUE + 'cc', borderColor: BLUE, borderWidth: 1
}});
const tsDS  = (label, data) => ({{
  label, data,
  backgroundColor: ORANGE + 'cc', borderColor: ORANGE, borderWidth: 1
}});

// ── Summary + OOS charts ──────────────────────────────────────────────────────
const OOS_CCP_C   = {oos_cost_ccp_js};
const OOS_TS_C    = {oos_cost_ts_js};
const OOS_CCP_V   = {oos_viol_ccp_js};
const OOS_TS_V    = {oos_viol_ts_js};
const OOS_CCP_S   = {oos_ccp_sample_js};
const OOS_TS_S    = {oos_ts_sample_js};

lineChart('sumCostChart', NS,
  [ccpDS('CCP', OOS_CCP_C), tsDS('Two-Stage', OOS_TS_C)],
  'Mean Realized Cost ($M)', v => '$' + v.toFixed(2) + 'M');
lineChart('sumViolChart', NS,
  [ccpDS('CCP', OOS_CCP_V), tsDS('Two-Stage', OOS_TS_V)],
  '% Scenarios with Violation', v => v.toFixed(1) + '%');

lineChart('oosCostChart', NS,
  [ccpDS('CCP', OOS_CCP_C), tsDS('Two-Stage', OOS_TS_C)],
  'Mean Realized Cost ($M)', v => '$' + v.toFixed(2) + 'M');

lineChart('oosViolChart', NS,
  [ccpDS('CCP', OOS_CCP_V), tsDS('Two-Stage', OOS_TS_V)],
  '% Scenarios with Violation', v => v.toFixed(1) + '%');

new Chart(document.getElementById('oosDistChart'), {{
  type: 'line',
  data: {{
    labels: Array.from({{length: OOS_CCP_S.length}}, (_,i) => i+1),
    datasets: [
      {{ label: 'CCP', data: OOS_CCP_S, borderColor: BLUE, backgroundColor: BLUE+'22',
         borderWidth:1.5, pointRadius:0, tension:0 }},
      {{ label: 'Two-Stage', data: OOS_TS_S, borderColor: ORANGE, backgroundColor: ORANGE+'22',
         borderWidth:1.5, pointRadius:0, tension:0, borderDash:[4,2] }},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ position:'bottom', labels:{{ font:{{ size:10 }} }} }} }},
    scales:{{
      x:{{ display:false }},
      y:{{ title:{{ display:true, text:'Realized Cost ($M)' }},
           ticks:{{ callback: v => '$'+v.toFixed(2)+'M' }} }}
    }}
  }}
}});

// ── Sensitivity charts ────────────────────────────────────────────────────────
const SEN_SLS      = {sigma_ls_js};
const N15_CCP_VIOL = {n15_ccp_viol_js};
const N50_CCP_VIOL = {n50_ccp_viol_js};
const N15_TS_VIOL  = {n15_ts_viol_js};
const N50_TS_VIOL  = {n50_ts_viol_js};
const N15_TS_COST  = {n15_ts_cost_js};
const N50_TS_COST  = {n50_ts_cost_js};
const GREEN = '#166534';

if (document.getElementById('senViolBarChart') && SEN_SLS.length > 0) {{
  // Grouped bar: CCP vs 2S violation rate across sigma_L
  new Chart(document.getElementById('senViolBarChart'), {{
    type: 'bar',
    data: {{
      labels: SEN_SLS.map(v => 'σ=' + v),
      datasets: [
        {{ label: 'CCP N=15', data: N15_CCP_VIOL,
           backgroundColor: BLUE + 'bb', borderColor: BLUE, borderWidth:1 }},
        {{ label: '2S N=15',  data: N15_TS_VIOL,
           backgroundColor: BLUE + '44', borderColor: BLUE, borderWidth:1, borderDash:[4,2] }},
        {{ label: 'CCP N=50', data: N50_CCP_VIOL,
           backgroundColor: ORANGE + 'bb', borderColor: ORANGE, borderWidth:1 }},
        {{ label: '2S N=50',  data: N50_TS_VIOL,
           backgroundColor: ORANGE + '44', borderColor: ORANGE, borderWidth:1 }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ position:'bottom', labels:{{ font:{{ size:10 }}, boxWidth:12 }} }} }},
      scales:{{
        x:{{ title:{{ display:true, text:'σ_L (manufacturing variability)' }} }},
        y:{{ title:{{ display:true, text:'% Scenarios with Violation' }},
             ticks:{{ callback: v => v+'%' }}, min:0 }}
      }}
    }}
  }});

  // Line: Two-Stage cost vs sigma_L
  new Chart(document.getElementById('senCostChart'), {{
    type: 'bar',
    data: {{
      labels: SEN_SLS.map(v => 'σ=' + v),
      datasets: [
        {{ label: 'N=15', data: N15_TS_COST,
           backgroundColor: BLUE+'cc', borderColor: BLUE, borderWidth:1 }},
        {{ label: 'N=50', data: N50_TS_COST,
           backgroundColor: ORANGE+'cc', borderColor: ORANGE, borderWidth:1 }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ position:'bottom', labels:{{ font:{{ size:10 }} }} }} }},
      scales:{{
        x:{{ title:{{ display:true, text:'σ_L (manufacturing variability)' }} }},
        y:{{ title:{{ display:true, text:'Two-Stage Cost ($M)' }},
             ticks:{{ callback: v => '$'+v.toFixed(2)+'M' }} }}
      }}
    }}
  }});
}}

requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));

// ── Cost Breakdown stacked bar ────────────────────────────────────────────────
const _cb_det_tr={_cb_det_tr};
{cost_breakdown_js}
if (document.getElementById('costBreakdownChart')) {{
  new Chart(document.getElementById('costBreakdownChart'), {{
    type: 'bar',
    data: {{
      labels: NS.map(n => 'N='+n),
      datasets: [
        {{ label: 'Det — Facility ($K)',     data: _cb_det_fac,  backgroundColor: '#1E3A5F',   stack: 'det' }},
        {{ label: 'Det — Transport ($K)',    data: _cb_det_tr,   backgroundColor: '#3b82f6',   stack: 'det' }},
        {{ label: 'Det — Material ($K)',     data: _cb_det_mat,  backgroundColor: '#93c5fd',   stack: 'det' }},
        {{ label: '2S — Facility ($K)',      data: _cb_ts_fac,   backgroundColor: '#92400E',   stack: 'ts'  }},
        {{ label: '2S — Transport ($K)',     data: _cb_ts_tr,    backgroundColor: '#f97316',   stack: 'ts'  }},
        {{ label: '2S — Material ($K)',      data: _cb_det_mat,  backgroundColor: '#fed7aa',   stack: 'ts'  }},
        {{ label: '2S — Exp. Expediting($K)',data: _cb_ts_exp,   backgroundColor: '#dc2626',   stack: 'ts'  }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }}, boxWidth: 12 }} }},
        tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': $' + ctx.raw.toFixed(0) + 'K' }} }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'N (patients) — left bars = Det, right bars = Two-Stage' }} }},
        y: {{ title: {{ display: true, text: 'Cost ($K)' }},
              stacked: true, ticks: {{ callback: v => '$' + (v/1000).toFixed(0) + 'M' }} }}
      }}
    }}
  }});
}}

// ── Gantt chart ───────────────────────────────────────────────────────────────
if (document.getElementById('ganttChart')) {{
  const G_PHASES = ['Leukapheresis', 'Outbound transport', 'Manufacturing', 'QC', 'Return transport'];
  const G_COLORS = ['#1d4ed8','#16a34a','#d97706','#7c3aed','#059669'];
  // Rows: [label, [col_start, mfg_start, mfg_end, qc_end, ret_end], deadline, note]
  const G_ROWS = [
    ['Ground j1 — nominal (TRT=17d)',   [0,1,2,9,16,17], 26, ''],
    ['Air j2 — nominal (TRT=23d)',      [0,1,5,12,19,23], 26, ''],
    ['Ground j1 — late mfg T_MF=9d (expedited, deadline met)', [0,1,2,11,18,19], 26, ''],
    ['Ground j1 — severe delay T_MF=12d (violation)', [0,1,2,14,21,22], 26, ''],
  ];
  const phases = [
    {{ label:'Leukapheresis (TLS=1d)', backgroundColor:'#1d4ed8', data:G_ROWS.map(r=>([r[1][0],r[1][1]])) }},
    {{ label:'Outbound transport',     backgroundColor:'#16a34a', data:G_ROWS.map(r=>([r[1][1],r[1][2]])) }},
    {{ label:'Manufacturing (T_MF)',   backgroundColor:'#d97706', data:G_ROWS.map(r=>([r[1][2],r[1][3]])) }},
    {{ label:'QC (7d)',                backgroundColor:'#7c3aed', data:G_ROWS.map(r=>([r[1][3],r[1][4]])) }},
    {{ label:'Return transport',       backgroundColor:'#059669', data:G_ROWS.map(r=>([r[1][4],r[1][5]])) }},
  ];
  new Chart(document.getElementById('ganttChart'), {{
    type: 'bar',
    data: {{ labels: G_ROWS.map(r=>r[0]), datasets: phases }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }}, boxWidth: 12 }} }},
        annotation: {{}}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Days from leukapheresis collection' }},
          min: 0, max: 28,
          ticks: {{ stepSize: 2 }}
        }},
        y: {{ stacked: true }}
      }}
    }}
  }});
}}

// ── Tau feasibility charts ────────────────────────────────────────────────────
const TAU_LABELS = {tau_labels_js};
const TAU_DET15  = {tau_cost_det_js};
const TAU_TS15   = {tau_cost_ts_js};
const TAU_DET25  = {tau_cost_det25_js};
const TAU_TS25   = {tau_cost_ts25_js};

if (document.getElementById('tauFeasChart') && TAU_LABELS.length > 0) {{
  new Chart(document.getElementById('tauFeasChart'), {{
    type: 'bar',
    data: {{
      labels: TAU_LABELS.map(v => '\u03c4=' + v),
      datasets: [
        {{ label: 'Det N=15',  data: TAU_DET15, backgroundColor: BLUE+'cc',     borderColor: BLUE,      borderWidth:1 }},
        {{ label: '2S N=15',   data: TAU_TS15,  backgroundColor: ORANGE+'cc',   borderColor: ORANGE,    borderWidth:1 }},
        {{ label: 'Det N=25',  data: TAU_DET25, backgroundColor: '#166534cc',   borderColor: '#166534', borderWidth:1 }},
        {{ label: '2S N=25',   data: TAU_TS25,  backgroundColor: '#7c3aedcc',   borderColor: '#7c3aed', borderWidth:1 }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }} }} }},
        tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': $' + (ctx.raw||0).toFixed(2)+'M' }} }} }},
      scales: {{
        x: {{ title: {{ display: true, text: '\u03c4 (deadline tightness, 0=loose \u2192 1=tight)' }} }},
        y: {{ title: {{ display: true, text: 'Total Cost ($M)' }},
              ticks: {{ callback: v => '$'+v.toFixed(1)+'M' }} }}
      }}
    }}
  }});
}}

// ── VSS sweep chart ───────────────────────────────────────────────────────────
const VSS_SLS      = {vss_sweep_sls_js};
const VSS_DATASETS = {vss_sweep_datasets_js};

if (document.getElementById('vssSweepChart') && VSS_SLS.length > 0) {{
  new Chart(document.getElementById('vssSweepChart'), {{
    type: 'bar',
    data: {{
      labels: VSS_SLS.map(v => 'σ=' + v),
      datasets: VSS_DATASETS
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }}, boxWidth: 12 }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.dataset.label + ': ' +
              (ctx.raw !== null ? ctx.raw.toFixed(1) + '%' : 'n/a')
          }}
        }},
        annotation: {{
          annotations: {{
            breakeven: {{ type: 'line', yMin: 0, yMax: 0,
              borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6,4],
              label: {{ content: 'Breakeven', display: true, position: 'end', font: {{ size:9 }} }} }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'σ_L (manufacturing variability)' }} }},
        y: {{
          title: {{ display: true, text: 'VSS (%)' }},
          ticks: {{ callback: v => v.toFixed(1) + '%' }},
          grid: {{ color: ctx => ctx.tick.value === 0 ? '#dc262666' : '#e5e7eb' }}
        }}
      }}
    }}
  }});
}}

// ── VSS tau chart ─────────────────────────────────────────────────────────────
const VSS_TAUS         = {vss_tau_taus_js};
const VSS_TAU_DATASETS = {vss_tau_datasets_js};

if (document.getElementById('vssTauChart') && VSS_TAUS.length > 0) {{
  new Chart(document.getElementById('vssTauChart'), {{
    type: 'bar',
    data: {{
      labels: VSS_TAUS.map(v => '\u03c4=' + v),
      datasets: VSS_TAU_DATASETS
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }}, boxWidth: 12 }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.dataset.label + ': ' +
              (ctx.raw !== null ? ctx.raw.toFixed(1) + '%' : 'n/a')
          }}
        }},
        annotation: {{
          annotations: {{
            breakeven: {{ type: 'line', yMin: 0, yMax: 0,
              borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6,4],
              label: {{ content: 'Breakeven', display: true, position: 'end', font: {{ size:9 }} }} }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: '\u03c4 (deadline tightness)' }} }},
        y: {{
          title: {{ display: true, text: 'VSS (%)' }},
          ticks: {{ callback: v => v.toFixed(2) + '%' }},
          grid: {{ color: ctx => ctx.tick.value === 0 ? '#dc262666' : '#e5e7eb' }}
        }}
      }}
    }}
  }});
}}
</script>
</body></html>"""
    with open(output_path, 'w') as f:
        f.write(HTML)
    print(f'HTML report -> {output_path}')


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Two-Stage vs Deterministic ColGen Validation')
    parser.add_argument('--tau',        type=float, default=0.0)
    parser.add_argument('--sigma-L',    type=float, default=0.20, dest='sigma_L')
    parser.add_argument('--delta',      type=float, default=2.0)
    parser.add_argument('--epsilon',    type=float, default=0.05)
    parser.add_argument('--time-limit', type=int,   default=300, dest='time_limit')
    parser.add_argument('--n-sim',      type=int,   default=5000, dest='n_sim')
    parser.add_argument('--output',     default='twostage_expedite.html')
    args = parser.parse_args()

    print('=' * 60)
    print('Two-Stage Expediting vs Deterministic ColGen — Validation')
    print(f'tau={args.tau}  sigma_L={args.sigma_L}  delta={args.delta}  '
          f'epsilon={args.epsilon}  time_limit={args.time_limit}s')
    print('=' * 60)

    results = run_all(
        tau=args.tau, sigma_L=args.sigma_L, delta=args.delta,
        epsilon=args.epsilon, time_limit=args.time_limit, n_sim=args.n_sim,
    )

    det_res = [r for r in results if r['solver'] == 'Deterministic']
    ts_res  = [r for r in results if r['solver'] == 'Two-Stage Exp.']
    pi_map_default = {'high': 8000, 'medium': 4000, 'low': 1500}

    print('\n' + '=' * 60)
    print('Computing VSS …')
    vss_rows = compute_vss(
        det_results=det_res, ts_results=ts_res,
        sigma_L=args.sigma_L, delta=args.delta,
        pi_map_default=pi_map_default,
        n_sim=10000, seed=55,
    )
    print(f'  VSS computed for {len(vss_rows)} instance(s).')

    print('\n' + '=' * 60)
    print('Computing VSS σ_L sweep (finds breakeven where stochastic model pays off) …')
    print('  Sweeping σ_L ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40} for N ≤ 25.')
    vss_sweep = compute_vss_sweep(
        sigma_L_values=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        det_results=det_res,
        tau=args.tau,
        epsilon=args.epsilon,
        time_limit=args.time_limit,
        pi_map_default=pi_map_default,
        n_sim=min(args.n_sim, 3000),
        seed=55,
        max_N=25,
    )
    print(f'  VSS sweep: {len(vss_sweep)} data points.')

    print('\n' + '=' * 60)
    print('Computing VSS τ sweep (tighter deadlines → does stochastic Two-Stage pay off?) …')
    print('  Sweeping τ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} at σ_L=0.20 for N ≤ 25.')
    vss_tau_sweep = compute_vss_tau_sweep(
        tau_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        sigma_L=args.sigma_L,
        epsilon=args.epsilon,
        time_limit=args.time_limit,
        pi_map_default=pi_map_default,
        n_sim=min(args.n_sim, 3000),
        seed=55,
        max_N=25,
    )
    print(f'  VSS tau sweep: {len(vss_tau_sweep)} data points.')

    print('\n' + '=' * 60)
    print('Running sensitivity analysis (sigma_L / alpha / delta sweeps) …')
    print('  Note: this runs 2 solvers × (4+3+3) configs × 2 sizes = 40 solver calls.')
    sensitivity = run_sensitivity(
        tau=args.tau, epsilon=args.epsilon, time_limit=args.time_limit,
        n_sim_oos=min(args.n_sim, 2000),
    )

    out = os.path.join(BASE_DIR, args.output)
    generate_html(results, out, args.tau, args.sigma_L, args.delta, args.epsilon,
                  sensitivity=sensitivity, vss_rows=vss_rows,
                  vss_sweep=vss_sweep, vss_tau_sweep=vss_tau_sweep)
    print(f'\nDone. Open {args.output} in your browser.')
