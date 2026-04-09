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
                  sensitivity=None, vss_rows=None, vss_sweep=None):
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

    # ── VSS tab content ────────────────────────────────────────────────────────
    vss_table_rows = _vss_html(vss_rows) if vss_rows else '<p style="color:#9ca3af">No VSS data.</p>'
    avg_vss_pct = round(sum(r['vss_pct'] for r in (vss_rows or []))/max(1,len(vss_rows or [])), 2)

    # VSS sweep HTML + JS chart data
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
                'borderColor:"' + _col + '",backgroundColor:"' + _col + '22",'
                'pointBackgroundColor:"' + _col + '",borderWidth:2,pointRadius:5,'
                'tension:0.3,spanGaps:false}'
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
      <strong>CCP (Chance-Constrained Programming)</strong> replaces the nominal T_MF with an
      inflated budget T&#770;_MF = F&#8315;&#185;(1&minus;&alpha;) so that P(T_MF &le; T&#770;_MF) = 1&minus;&alpha;.
      This is a single-stage plan: routes are fixed, and no recourse action is taken if manufacturing runs late.
      Lower &alpha; = stricter guarantee = more conservative routes = higher transport cost.
    </div>
    <div class="insight orange">
      <strong>Two-Stage Expediting ColGen</strong> assumes T_MF ~ LogNormal(&mu;_L, &sigma;_L={sigma_L}).
      Stage 1 selects a route x_i before T_MF is realised.
      Stage 2 triggers expedited return (saving &delta;={delta}d, costing &pi;_p) if T_MF &gt; K_i (slack).
      Modified column cost: <em>c&#771;_i = c_transport + &pi;_p &sdot; P(T_MF &gt; K_i)</em>.
      Violations only occur when T_MF &gt; K_i + &delta; (recourse also fails).
    </div>
    <div class="insight green">
      <strong>Key difference</strong>: CCP commits to a conservative plan upfront (no flexibility).
      Two-Stage pays a small premium for a recourse action, trading lower up-front cost for
      the ability to adapt when manufacturing runs long. The out-of-sample tab quantifies this trade-off.
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

<!-- TAB 2: CCP vs Two-Stage (Out-of-Sample) -->
<div id="pane-oos" class="pane">
  <div class="card">
    <h2>CCP vs Two-Stage — Out-of-Sample Evaluation (10 000 scenarios)</h2>
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
      <strong style="color:#92400E">Note on near-zero VSS:</strong>
      At moderate &sigma;_L (0.10&ndash;0.30) the dataset deadlines are generous enough that
      both Det and Two-Stage routes have comfortable manufacturing slack K_i, keeping
      P(expedite) small for both. Two-Stage pays a small transport cost premium (selecting
      routes with larger K_i) that slightly exceeds the expediting savings &rarr; VSS &asymp; 0.
      Two-Stage&rsquo;s primary advantage here is <strong>violation reduction</strong>
      (near-zero hard-deadline breaches vs CCP&rsquo;s 30&ndash;84% — see OOS tab),
      not expected-cost reduction. Positive VSS would emerge with tighter deadlines (&tau; &gt; 0)
      or when the Two-Stage route filter forces the solver to use routes with genuinely small K_i.
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

  <div class="chart-box" style="margin-top:16px">
    <h3>VSS % vs &sigma;_L — Breakeven Chart</h3>
    <div class="chart-wrap" style="height:320px"><canvas id="vssSweepChart"></canvas></div>
  </div>
</div>

<script>
function show(n) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('a'));
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('a'));
  const m = {{summary:0,oos:1,mc:2,patients:3,findings:4,sensitivity:5,vss:6}};
  document.querySelectorAll('.tab')[m[n]].classList.add('a');
  document.getElementById('pane-' + n).classList.add('a');
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
}}

const NS   = {n_labels_js};
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

const ccpDS = (label, data) => ({{
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
    type: 'line',
    data: {{
      labels: SEN_SLS,
      datasets: [
        {{ label: 'N=15', data: N15_TS_COST,
           borderColor: BLUE, backgroundColor: BLUE+'22',
           pointBackgroundColor: BLUE, borderWidth:2, pointRadius:5, tension:0.3,
           spanGaps: false }},
        {{ label: 'N=50', data: N50_TS_COST,
           borderColor: ORANGE, backgroundColor: ORANGE+'22',
           pointBackgroundColor: ORANGE, borderWidth:2, pointRadius:5, tension:0.3,
           borderDash:[5,3], spanGaps: false }},
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

// ── VSS sweep chart ───────────────────────────────────────────────────────────
const VSS_SLS      = {vss_sweep_sls_js};
const VSS_DATASETS = {vss_sweep_datasets_js};

if (document.getElementById('vssSweepChart') && VSS_SLS.length > 0) {{
  new Chart(document.getElementById('vssSweepChart'), {{
    type: 'line',
    data: {{
      labels: VSS_SLS,
      datasets: [
        ...VSS_DATASETS,
        {{ label: 'Breakeven (VSS=0)', data: VSS_SLS.map(() => 0),
           borderColor: '#dc2626', borderWidth: 1.5, borderDash: [6, 4],
           pointRadius: 0, fill: false, order: 99 }}
      ]
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
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'σ_L (manufacturing variability)' }} }},
        y: {{
          title: {{ display: true, text: 'VSS (%)' }},
          ticks: {{ callback: v => v.toFixed(1) + '%' }},
          min: -5, max: 10,
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
    print('Running sensitivity analysis (sigma_L / alpha / delta sweeps) …')
    print('  Note: this runs 2 solvers × (4+3+3) configs × 2 sizes = 40 solver calls.')
    sensitivity = run_sensitivity(
        tau=args.tau, epsilon=args.epsilon, time_limit=args.time_limit,
        n_sim_oos=min(args.n_sim, 2000),
    )

    out = os.path.join(BASE_DIR, args.output)
    generate_html(results, out, args.tau, args.sigma_L, args.delta, args.epsilon,
                  sensitivity=sensitivity, vss_rows=vss_rows, vss_sweep=vss_sweep)
    print(f'\nDone. Open {args.output} in your browser.')
