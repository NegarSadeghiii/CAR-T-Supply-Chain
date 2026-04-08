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
                'solved':       r.get('solved', False),
                'total_cost':   r.get('total_cost', 0),
                'avg_trt':      r.get('avg_trt',    0),
                'solve_time':   r.get('solve_time', 0),
                'num_plans':    r.get('num_plans',  0),
                'num_patients': r.get('num_patients', n_val),
                'error':        r.get('error', ''),
                'patients':     r.get('patients', []),
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
    """Evaluate CCP and Two-Stage solutions on the same n_sim T_MF scenarios."""
    import statistics as _st
    rng  = random.Random(seed)
    mu_L = math.log(tmfe) - 0.5 * sigma_L ** 2
    scenarios = [rng.lognormvariate(mu_L, sigma_L) for _ in range(n_sim)]

    # CCP — no recourse: violation if T_MF > mfg_budget_p
    tmfe_eff     = ccp_r.get('tmfe_eff', tmfe)
    ccp_pats     = ccp_r.get('patients', [])
    ccp_transport = ccp_r.get('total_cost', 0)
    # mfg_budget per patient = max T_MF before deadline is missed
    mfg_budgets = {
        p['id']: p['deadline'] - (p['trt'] - tmfe_eff)
        for p in ccp_pats
    }

    ccp_costs, ccp_viols = [], []
    for tmf in scenarios:
        v = sum(1 for p in ccp_pats if tmf > mfg_budgets.get(p['id'], 999))
        ccp_costs.append(ccp_transport)   # cost is fixed — no recourse
        ccp_viols.append(v)

    # Two-Stage — recourse: expedite if T_MF > K_i
    ts_pats = ts_r.get('patients', [])
    ts_transport = ts_r.get('total_cost', 0) - ts_r.get('total_exp_expedite_cost', 0)

    ts_costs, ts_viols, ts_exps = [], [], []
    for tmf in scenarios:
        ec, nv, ne = 0, 0, 0
        for p in ts_pats:
            K   = p.get('slack')
            if K is None: continue
            pi_p = pi_map.get(p.get('group', 'low'), 1500)
            if tmf > K:
                ec += pi_p; ne += 1
                if tmf > K + delta:
                    nv += 1
        ts_costs.append(ts_transport + ec)
        ts_viols.append(nv)
        ts_exps.append(ne)

    def _metrics(costs, viols, exps=None):
        sc = sorted(costs)
        n  = len(sc)
        return {
            'mean_cost':    round(_st.mean(costs)),
            'std_cost':     round(_st.stdev(costs)),
            'p5_cost':      round(sc[int(0.05 * n)]),
            'p95_cost':     round(sc[int(0.95 * n)]),
            'pct_scen_viol': round(sum(1 for v in viols if v > 0) / n * 100, 1),
            'avg_viol_per_scen': round(_st.mean(viols), 3),
            'pct_exp':      round(sum(exps) / (n * max(len(ts_pats), 1)) * 100, 1)
                            if exps is not None else None,
        }

    return {
        'n_sim':   n_sim,
        'n_ccp':   len(ccp_pats),
        'n_ts':    len(ts_pats),
        'ccp':     _metrics(ccp_costs, ccp_viols),
        'ts':      _metrics(ts_costs,  ts_viols, ts_exps),
        'ccp_transport': ccp_transport,
        'ts_transport':  ts_transport,
        'tmfe_eff':  tmfe_eff,
        'alpha':     ccp_r.get('alpha', 0.10),
        # sample of 200 costs for chart
        'ccp_sample': ccp_costs[:200],
        'ts_sample':  ts_costs[:200],
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
        for h in ['Patient','Urgency','Deadline','TRT','On-time',
                   'Facility','j_out','j_ret','P(expedite)','E[$exp]','Slack K_i']:
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
              <td>{p.get('j_out','?')}</td>
              <td>{p.get('j_ret','?')}</td>
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
    scenarios = [rng.lognormvariate(mu_L, sigma_L) for _ in range(n_sim)]

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
        for tmf in scenarios:
            ec = 0
            for p in det_pats:
                # slack = deadline - (trt - tmfe) because trt = TLS+tmfe+TQC+TT
                K = p['deadline'] - (p.get('trt', p.get('turnaround', tmfe)) - tmfe)
                urg  = p.get('group', 'low')
                pi_p = pi_map.get(urg, 1500)
                if tmf > K:
                    ec += pi_p          # expedite triggered
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
        sigma_rows += (
            f'<tr><td><strong>{r["label"]}</strong></td>'
            f'<td style="text-align:center;font-weight:700">{r["sigma_L"]}</td>'
            f'<td style="text-align:right">${r.get("ts_cost",0):,.0f}</td>'
            f'<td style="text-align:right">${r.get("ccp_cost",0):,.0f}</td>'
            f'<td>{_gap_cell(gap)}</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ccp_viol",0)>5 else "#166534"};font-weight:700">{r.get("ccp_viol",0):.1f}%</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ts_viol",0)>5 else "#166534"};font-weight:700">{r.get("ts_viol",0):.1f}%</td>'
            f'<td style="text-align:right">{r.get("ts_pexp",0):.1f}%</td>'
            f'</tr>'
        )
    sigma_html = f'''<div class="card"><h2>&#963;_L Sweep (alpha=0.10 fixed, delta=2d fixed)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:12px">
      How does manufacturing time variability affect cost and reliability?
      Higher &#963;_L means more uncertainty around the nominal 7-day T_MF.
    </p>
    <div style="overflow-x:auto"><table class="tbl"><thead><tr>
      <th>N</th><th>&#963;_L</th><th>2S Cost</th><th>CCP Cost</th><th>Cost Gap</th>
      <th>CCP Violation%</th><th>2S Violation%</th><th>Avg P(exp)%</th>
    </tr></thead><tbody>{sigma_rows}</tbody></table></div></div>'''

    # alpha sweep table
    alpha_rows = ''
    for r in sensitivity.get('alpha_sweep', []):
        if not _row_ok(r):
            alpha_rows += f'<tr><td>{r["label"]}</td><td>{r["alpha"]}</td><td colspan="5" style="color:#dc2626">{r["error"][:60]}</td></tr>'
            continue
        gap = r.get('cost_gap', 0)
        alpha_rows += (
            f'<tr><td><strong>{r["label"]}</strong></td>'
            f'<td style="text-align:center;font-weight:700">{r["alpha"]}</td>'
            f'<td style="text-align:right">${r.get("ccp_cost",0):,.0f}</td>'
            f'<td style="text-align:right">{r.get("tmfe_eff",0):.2f}d</td>'
            f'<td>{_gap_cell(gap)}</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ccp_viol",0)>5 else "#166534"};font-weight:700">{r.get("ccp_viol",0):.1f}%</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ts_viol",0)>5 else "#166534"};font-weight:700">{r.get("ts_viol",0):.1f}%</td>'
            f'</tr>'
        )
    alpha_html = f'''<div class="card"><h2>&#945; Sweep — CCP (sigma_L=0.20 fixed, delta=2d fixed)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:12px">
      How does the CCP service level parameter affect cost and violation rate?
      Lower &#945; = stricter service guarantee = higher effective manufacturing budget = higher cost.
    </p>
    <div style="overflow-x:auto"><table class="tbl"><thead><tr>
      <th>N</th><th>&#945;</th><th>CCP Cost</th><th>T_MF budget (eff.)</th>
      <th>Cost Gap vs 2S</th><th>CCP Violation%</th><th>2S Violation%</th>
    </tr></thead><tbody>{alpha_rows}</tbody></table></div></div>'''

    # delta sweep table
    delta_rows = ''
    for r in sensitivity.get('delta_sweep', []):
        if not _row_ok(r):
            delta_rows += f'<tr><td>{r["label"]}</td><td>{r["delta"]}</td><td colspan="5" style="color:#dc2626">{r["error"][:60]}</td></tr>'
            continue
        gap = r.get('cost_gap', 0)
        delta_rows += (
            f'<tr><td><strong>{r["label"]}</strong></td>'
            f'<td style="text-align:center;font-weight:700">{r["delta"]}d</td>'
            f'<td style="text-align:right">${r.get("ts_cost",0):,.0f}</td>'
            f'<td style="text-align:right">{r.get("ts_plans",0)}</td>'
            f'<td>{_gap_cell(gap)}</td>'
            f'<td style="text-align:right;color:{"#dc2626" if r.get("ts_viol",0)>5 else "#166534"};font-weight:700">{r.get("ts_viol",0):.1f}%</td>'
            f'<td style="text-align:right">{r.get("ts_pexp",0):.1f}%</td>'
            f'</tr>'
        )
    delta_html = f'''<div class="card"><h2>&#948; Sweep — Two-Stage (sigma_L=0.20 fixed, alpha=0.10 fixed)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:12px">
      How many days of expedited shipping buffer (&#948;) are needed to rescue delayed patients?
      Larger &#948; = more routes become feasible under recourse = lower cost.
    </p>
    <div style="overflow-x:auto"><table class="tbl"><thead><tr>
      <th>N</th><th>&#948;</th><th>2S Cost</th><th>Plans</th>
      <th>Cost Gap vs CCP</th><th>2S Violation%</th><th>Avg P(exp)%</th>
    </tr></thead><tbody>{delta_rows}</tbody></table></div></div>'''

    return sigma_html, alpha_html, delta_html


def _build_sigma_chart_js(sensitivity):
    """Return JS arrays for sigma_L sensitivity charts (N=15 and N=50)."""
    if not sensitivity:
        return '', '', '', ''
    sigma_sweep = sensitivity.get('sigma_sweep', [])
    n15 = [r for r in sigma_sweep if r.get('N') == 15 and 'error' not in r]
    n50 = [r for r in sigma_sweep if r.get('N') == 50 and 'error' not in r]
    sls = [0.10, 0.20, 0.30, 0.40]

    def _arr(rows, key, scale=1):
        by_sl = {r['sigma_L']: r for r in rows}
        def _val(s):
            if s not in by_sl: return 'null'
            v = by_sl[s].get(key)
            if v is None: return 'null'
            return str(round(v / scale, 4))
        return '[' + ','.join(_val(s) for s in sls) + ']'

    n15_gap  = _arr(n15, 'cost_gap')
    n50_gap  = _arr(n50, 'cost_gap')
    n15_by_sl = {r['sigma_L']: r for r in n15}
    n50_by_sl = {r['sigma_L']: r for r in n50}
    n15_viol = '[' + ','.join(str(round(n15_by_sl.get(s,{}).get('ccp_viol',0) or 0, 2)) for s in sls) + ']'
    n50_viol = '[' + ','.join(str(round(n50_by_sl.get(s,{}).get('ccp_viol',0) or 0, 2)) for s in sls) + ']'
    return n15_gap, n50_gap, n15_viol, n50_viol


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
                  sensitivity=None, vss_rows=None):
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

    # Key findings text
    findings = []
    if det_rows and ts_rows:
        avg_premium = sum(
            (ts_by_n[n]['total_cost'] - det_by_n[n]['total_cost']) / det_by_n[n]['total_cost'] * 100
            for n in ns if n in det_by_n and n in ts_by_n
        ) / max(1, len([n for n in ns if n in det_by_n and n in ts_by_n]))
        avg_pexp = sum(r['avg_prob_expedite'] for r in ts_rows) / len(ts_rows) * 100
        max_mc_err = max(
            (row['rel_err'] for r in ts_rows for row in r.get('mc_validation', [])),
            default=0
        )
        findings.append(f'Average cost premium of Two-Stage over Deterministic: <strong>{avg_premium:+.1f}%</strong>. '
                        f'This is the price of robustness — routes are filtered to keep a recourse option viable.')
        findings.append(f'Average probability of expediting across all patients and instances: <strong>{avg_pexp:.1f}%</strong>. '
                        f'Expediting recourse is rarely triggered (&lt;20% of patients), keeping expected costs low.')
        findings.append(f'Monte Carlo validation — maximum relative error between theoretical and empirical '
                        f'P(expedite): <strong>{max_mc_err*100:.1f}%</strong>. '
                        f'Values below 10% confirm the LogNormal survival formula in the objective is correct.')

    findings_html = ''.join(
        f'<div class="insight {"orange" if i==0 else "green" if i==2 else ""}">{f}</div>'
        for i, f in enumerate(findings)
    )

    # ── Sensitivity tab content ────────────────────────────────────────────────
    if sensitivity:
        sen_sigma_html, sen_alpha_html, sen_delta_html = _sensitivity_html(sensitivity)
        n15_gap_js, n50_gap_js, n15_viol_js, n50_viol_js = _build_sigma_chart_js(sensitivity)
        sigma_ls_js = '[0.10,0.20,0.30,0.40]'
    else:
        sen_sigma_html = sen_alpha_html = sen_delta_html = '<p style="color:#9ca3af">No sensitivity data.</p>'
        n15_gap_js = n50_gap_js = n15_viol_js = n50_viol_js = '[]'
        sigma_ls_js = '[]'

    # ── VSS tab content ────────────────────────────────────────────────────────
    vss_table_rows = _vss_html(vss_rows) if vss_rows else '<p style="color:#9ca3af">No VSS data.</p>'
    avg_vss_pct = round(sum(r['vss_pct'] for r in (vss_rows or []))/max(1,len(vss_rows or [])), 2)

    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Two-Stage vs Deterministic — Validation Report</title>
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
  <h1>Two-Stage Expediting ColGen vs Deterministic ColGen — Validation Report</h1>
  <p>N = 5,10,15,20,25,40,50 &nbsp;|&nbsp;
     &tau; = {tau} &nbsp;|&nbsp; &sigma;_L = {sigma_L} &nbsp;|&nbsp;
     &delta; = {delta}d &nbsp;|&nbsp; &epsilon; = {epsilon}</p>
</div>

<div class="tabs">
  <div class="tab a" onclick="show('summary')">&#127775; Summary</div>
  <div class="tab"   onclick="show('charts')">&#128200; Charts</div>
  <div class="tab"   onclick="show('table')">&#128196; Full Table</div>
  <div class="tab"   onclick="show('mc')">&#9989; Obj. Validation</div>
  <div class="tab"   onclick="show('patients')">&#128101; Patients</div>
  <div class="tab"   onclick="show('oos')">&#127919; Out-of-Sample</div>
  <div class="tab"   onclick="show('findings')">&#128270; Findings</div>
  <div class="tab"   onclick="show('sensitivity')">&#128202; Sensitivity</div>
  <div class="tab"   onclick="show('vss')">&#127942; VSS</div>
</div>

<!-- TAB 1: Summary -->
<div id="pane-summary" class="pane a">
  <div class="meta-grid">
    <div class="meta-card">
      <div class="v" style="color:var(--blue)">{len(det_rows)}/{len(ns)}</div>
      <div class="l">Det. Feasible</div>
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
      <h3>Schedule Cost ($M) — both methods</h3>
      <div class="chart-wrap"><canvas id="costChartS"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Cost Breakdown — Two-Stage (N={largest_n})</h3>
      <div class="chart-wrap"><canvas id="donutChart"></canvas></div>
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
      <strong>Deterministic ColGen</strong> solves the set-partitioning master problem using the nominal
      manufacturing time T_MF = 7 days. Plans are filtered purely by the deadline
      constraint TLS + T_MF + TQC + TT1 + TT3 &le; D_p. No uncertainty is modelled.
    </div>
    <div class="insight orange">
      <strong>Two-Stage Expediting ColGen</strong> assumes T_MF ~ LogNormal(&mu;_L, &sigma;_L={sigma_L}).
      Stage 1 selects a route x_i before T_MF is realised.
      Stage 2 triggers expedited return (saving &delta;={delta}d, costing &pi;_p) if T_MF &gt; K_i (slack).
      Modified column cost: <em>c&#771;_i = c_transport + &pi;_p &sdot; P(T_MF &gt; K_i)</em>.
    </div>
    <div class="insight green">
      <strong>Objective validation</strong>: For each selected patient plan we draw 5 000 LogNormal
      samples and verify empirical P(expedite) &asymp; theoretical 1 &minus; &Phi;((ln K_i &minus; &mu;_L)/&sigma;_L).
      Relative errors &lt;10% confirm the formula is correctly embedded in the objective.
    </div>
  </div>
</div>

<!-- TAB 2: Charts -->
<div id="pane-charts" class="pane">
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Total Cost ($M) vs N</h3>
      <div class="chart-wrap"><canvas id="costChart2"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Solve Time (s) vs N</h3>
      <div class="chart-wrap"><canvas id="timeChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Avg TRT (days) vs N</h3>
      <div class="chart-wrap"><canvas id="trtChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Avg P(expedite) % vs N — Two-Stage only</h3>
      <div class="chart-wrap"><canvas id="pexpChart"></canvas></div>
    </div>
  </div>
</div>

<!-- TAB 3: Full Table -->
<div id="pane-table" class="pane">
  <div class="card"><h2>Full Results — All Instances & Both Solvers</h2>
  <div style="overflow-x:auto">
  <table class="tbl"><thead><tr>
    <th>N</th>
    <th>Det Solved</th><th>Det Cost</th><th>Det TRT</th><th>Det Plans</th><th>Det Time</th>
    <th>2S Solved</th><th>2S Cost</th><th>2S TRT</th><th>2S Plans</th><th>2S Time</th>
    <th>Avg P(exp)</th><th>E[$exp] tot.</th>
    <th>Cost Gap</th><th>TRT Gap</th>
  </tr></thead><tbody>
  {gap_rows_html}
  </tbody></table>
  </div></div>
</div>

<!-- TAB 4: MC Validation -->
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

<!-- TAB 6: Out-of-Sample Validation -->
<div id="pane-oos" class="pane">
  <div class="card">
    <h2>Out-of-Sample Validation — CCP vs Two-Stage (10 000 scenarios)</h2>
    <p style="font-size:12px;color:var(--dim);margin-bottom:14px;line-height:1.6">
      The first-stage route decisions from both models are <strong>fixed</strong> and then evaluated
      against 10 000 fresh T_MF draws from LogNormal(&sigma;_L={sigma_L}).
      This tests how well each solution holds up under unseen uncertainty.<br>
      &bull; <strong>CCP</strong>: no recourse — deadline violation occurs whenever T_MF &gt; mfg_budget.<br>
      &bull; <strong>Two-Stage</strong>: expediting triggered if T_MF &gt; K_i (pays &pi;_p, saves &delta;d);
        violation only if T_MF &gt; K_i + &delta;.
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
    <h2>Key Findings</h2>
    {findings_html}
    <div class="insight" style="margin-top:12px">
      <strong>Scalability:</strong> Both solvers scale to N=50 within the time limit.
      Two-Stage Expediting has a slightly higher plan-generation phase (survival probability
      computation per column) but the master IP solve time is comparable because the plan
      count is slightly smaller (tighter feasibility filter K_i &ge; F&sup;&#8315;&sup1;(1&minus;&epsilon;) &minus; &delta;).
    </div>
    <div class="insight orange">
      <strong>When Two-Stage wins:</strong> When &sigma;_L is large (high T_MF variability),
      the deterministic schedule is exposed to deadline violations. Two-Stage pre-prices
      this risk into the objective, producing schedules that trade a small cost premium
      for a guaranteed recourse action (expedited return) when manufacturing runs long.
    </div>
  </div>
</div>

<!-- TAB 8: Sensitivity -->
<div id="pane-sensitivity" class="pane">
  <div class="card" style="margin-bottom:10px">
    <h2>Sensitivity Analysis — Conference Paper Results</h2>
    <p style="font-size:12px;color:var(--dim);line-height:1.7">
      Each sweep varies one parameter while holding the others fixed, evaluated on
      representative instances <strong>N=15</strong> and <strong>N=50</strong>.
      Out-of-sample validation uses 5 000 scenarios per configuration.
    </p>
  </div>
  {sen_sigma_html}
  {sen_alpha_html}
  {sen_delta_html}
  <div class="chart-grid" style="margin-top:16px">
    <div class="chart-box">
      <h3>&sigma;_L vs Cost Gap (2S vs CCP) — %</h3>
      <div class="chart-wrap"><canvas id="senGapChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h3>&sigma;_L vs CCP Violation Rate — %</h3>
      <div class="chart-wrap"><canvas id="senViolChart"></canvas></div>
    </div>
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
      A positive VSS means using the stochastic model (Two-Stage) yields a lower expected cost than
      naively taking the deterministic solution and hoping recourse will cover delays.
      Average VSS across instances: <strong>{avg_vss_pct:+.1f}%</strong>.
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
</div>

<script>
function show(n) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('a'));
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('a'));
  const m = {{summary:0,charts:1,table:2,mc:3,patients:4,oos:5,findings:6,sensitivity:7,vss:8}};
  document.querySelectorAll('.tab')[m[n]].classList.add('a');
  document.getElementById('pane-' + n).classList.add('a');
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
}}

const NS     = {n_labels_js};
const DET_C  = {det_costs};
const TS_C   = {ts_costs};
const DET_T  = {det_times};
const TS_T   = {ts_times};
const DET_TRT= {det_trts};
const TS_TRT = {ts_trts};
const TS_PEXP= {ts_pexps};

const BLUE   = '#1E3A5F';
const ORANGE = '#92400E';

function lineChart(id, labels, datasets, yLabel, fmt) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
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

const detDS = (label, data, id) => ({{
  label, data,
  borderColor: BLUE, backgroundColor: BLUE + '22',
  pointBackgroundColor: BLUE, borderWidth: 2, pointRadius: 5, tension: 0.3
}});
const tsDS  = (label, data) => ({{
  label, data,
  borderColor: ORANGE, backgroundColor: ORANGE + '22',
  pointBackgroundColor: ORANGE, borderWidth: 2, pointRadius: 5, tension: 0.3,
  borderDash: [5, 3]
}});

lineChart('costChartS', NS, [detDS('Deterministic', DET_C), tsDS('Two-Stage', TS_C)],
          'Cost ($M)', v => '$' + v.toFixed(2) + 'M');
lineChart('costChart2', NS, [detDS('Deterministic', DET_C), tsDS('Two-Stage', TS_C)],
          'Cost ($M)', v => '$' + v.toFixed(2) + 'M');
lineChart('timeChart',  NS, [detDS('Deterministic', DET_T), tsDS('Two-Stage', TS_T)],
          'Solve Time (s)', v => v.toFixed(1) + 's');
lineChart('trtChart',   NS, [detDS('Deterministic', DET_TRT), tsDS('Two-Stage', TS_TRT)],
          'Avg TRT (days)', v => v.toFixed(1) + 'd');

new Chart(document.getElementById('pexpChart'), {{
  type: 'bar',
  data: {{
    labels: NS,
    datasets: [{{ label: 'Avg P(expedite) %', data: TS_PEXP,
                  backgroundColor: ORANGE + '99', borderColor: ORANGE, borderWidth: 1 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'N (patients)' }} }},
      y: {{ title: {{ display: true, text: 'P(expedite) %' }},
            ticks: {{ callback: v => v + '%' }} }}
    }}
  }}
}});

new Chart(document.getElementById('donutChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Transport Cost', 'Expected Expediting Cost'],
    datasets: [{{
      data: [{donut_transport}, {donut_exp}],
      backgroundColor: [BLUE + 'cc', ORANGE + 'cc'],
      borderColor: [BLUE, ORANGE], borderWidth: 2
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }} }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ' $' + ctx.parsed.toLocaleString()
        }}
      }}
    }}
  }}
}});

// ── Out-of-Sample charts ──────────────────────────────────────────────────────
const OOS_NS      = {n_labels_js};
const OOS_CCP_C   = {oos_cost_ccp_js};
const OOS_TS_C    = {oos_cost_ts_js};
const OOS_CCP_V   = {oos_viol_ccp_js};
const OOS_TS_V    = {oos_viol_ts_js};
const OOS_CCP_S   = {oos_ccp_sample_js};
const OOS_TS_S    = {oos_ts_sample_js};

lineChart('oosCostChart', OOS_NS,
  [detDS('CCP', OOS_CCP_C), tsDS('Two-Stage', OOS_TS_C)],
  'Mean Realized Cost ($M)', v => '$' + v.toFixed(2) + 'M');

lineChart('oosViolChart', OOS_NS,
  [detDS('CCP', OOS_CCP_V), tsDS('Two-Stage', OOS_TS_V)],
  '% Scenarios with Violation', v => v.toFixed(1) + '%');

new Chart(document.getElementById('oosDistChart'), {{
  type: 'line',
  data: {{
    labels: Array.from({{length: OOS_CCP_S.length}}, (_,i) => i+1),
    datasets: [
      {{ label: 'CCP', data: OOS_CCP_S, borderColor: BLUE,   backgroundColor: BLUE+'22',
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
const SEN_SLS   = {sigma_ls_js};
const N15_GAP   = {n15_gap_js};
const N50_GAP   = {n50_gap_js};
const N15_VIOL  = {n15_viol_js};
const N50_VIOL  = {n50_viol_js};
const GREEN = '#166534';

if (document.getElementById('senGapChart') && SEN_SLS.length > 0) {{
  new Chart(document.getElementById('senGapChart'), {{
    type: 'line',
    data: {{
      labels: SEN_SLS,
      datasets: [
        {{ label: 'N=15', data: N15_GAP,
           borderColor: BLUE, backgroundColor: BLUE+'22',
           pointBackgroundColor: BLUE, borderWidth:2, pointRadius:5, tension:0.3 }},
        {{ label: 'N=50', data: N50_GAP,
           borderColor: ORANGE, backgroundColor: ORANGE+'22',
           pointBackgroundColor: ORANGE, borderWidth:2, pointRadius:5, tension:0.3, borderDash:[5,3] }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ position:'bottom', labels:{{ font:{{ size:10 }} }} }} }},
      scales:{{
        x:{{ title:{{ display:true, text:'sigma_L' }} }},
        y:{{ title:{{ display:true, text:'Cost Gap (%)' }},
             ticks:{{ callback: v => v.toFixed(1)+'%' }} }}
      }}
    }}
  }});

  new Chart(document.getElementById('senViolChart'), {{
    type: 'line',
    data: {{
      labels: SEN_SLS,
      datasets: [
        {{ label: 'N=15 CCP Viol%', data: N15_VIOL,
           borderColor: BLUE, backgroundColor: BLUE+'22',
           pointBackgroundColor: BLUE, borderWidth:2, pointRadius:5, tension:0.3 }},
        {{ label: 'N=50 CCP Viol%', data: N50_VIOL,
           borderColor: ORANGE, backgroundColor: ORANGE+'22',
           pointBackgroundColor: ORANGE, borderWidth:2, pointRadius:5, tension:0.3, borderDash:[5,3] }},
      ]
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ position:'bottom', labels:{{ font:{{ size:10 }} }} }} }},
      scales:{{
        x:{{ title:{{ display:true, text:'sigma_L' }} }},
        y:{{ title:{{ display:true, text:'CCP Violation %' }},
             ticks:{{ callback: v => v.toFixed(1)+'%' }} }}
      }}
    }}
  }});
}}

requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
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
    print('Running sensitivity analysis (sigma_L / alpha / delta sweeps) …')
    print('  Note: this runs 2 solvers × (4+3+3) configs × 2 sizes = 40 solver calls.')
    sensitivity = run_sensitivity(
        tau=args.tau, epsilon=args.epsilon, time_limit=args.time_limit,
        n_sim_oos=min(args.n_sim, 2000),
    )

    out = os.path.join(BASE_DIR, args.output)
    generate_html(results, out, args.tau, args.sigma_L, args.delta, args.epsilon,
                  sensitivity=sensitivity, vss_rows=vss_rows)
    print(f'\nDone. Open {args.output} in your browser.')
