"""
CAR-T Supply Chain — Interactive Feasibility Explorer
======================================================
Run with:  streamlit run app.py
"""
import os, sys, json, time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(BASE_DIR, 'experiment_feasibility_checkpoint.json')

MODELS = {
    'Deterministic (wait=0)': 'det',
    'Deterministic (flex wait)': 'fw',
    'CCP (α=0.10)': 'ccp',
    'Two-Stage (δ=2.0)': 'ts',
}
SIGMA_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
N_VALUES     = [5, 10, 15, 20, 25, 30, 50, 75, 100, 200, 500, 1000, 2000]

STATUS_COLOR = {
    'solved':     '#22c55e',
    'infeasible': '#ef4444',
    'timeout':    '#f59e0b',
    'missing':    '#d1d5db',
    'error':      '#8b5cf6',
    'running':    '#3b82f6',
}

# ── load / save checkpoint ────────────────────────────────────────────────────

@st.cache_data(ttl=2)
def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return {}
    with open(CHECKPOINT) as f:
        raw = json.load(f)
    return {s: {int(n): {float(sg): v for sg, v in sv.items()}
                for n, sv in nv.items()}
            for s, nv in raw.items()}

def save_result(solver, n, sigma, result):
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            raw = json.load(f)
    else:
        raw = {}
    raw.setdefault(solver, {}).setdefault(str(n), {})[str(sigma)] = result
    with open(CHECKPOINT, 'w') as f:
        json.dump(raw, f)
    load_checkpoint.clear()

# ── run solver ────────────────────────────────────────────────────────────────

def run_solver(solver_key, n, sigma, time_limit):
    from cart_colgen          import run_experiment          as det_run
    from cart_colgen_ccp      import run_experiment_ccp      as ccp_run, DEFAULT_PROCESS_CCP
    from cart_colgen_twostage import run_experiment_twostage as ts_run,  DEFAULT_PROCESS_2S
    from cart_colgen_flexwait import run_experiment_flexwait as fw_run

    df = os.path.join(BASE_DIR, f'Data_N{n}.dat')
    if not os.path.exists(df):
        return {'status': 'missing', 'cost': None, 'time': None, 'plans': None}

    t0 = time.time()
    try:
        if solver_key == 'det':
            r = det_run(data_file=df, time_limit=time_limit)
        elif solver_key == 'fw':
            r = fw_run(data_file=df, time_limit=time_limit)
        elif solver_key == 'ccp':
            cfg = dict(DEFAULT_PROCESS_CCP); cfg['sigma_L'] = sigma
            r = ccp_run(data_file=df, process_config=cfg, time_limit=time_limit)
        else:
            cfg = dict(DEFAULT_PROCESS_2S); cfg['sigma_L'] = sigma
            r = ts_run(data_file=df, process_config=cfg, time_limit=time_limit)
        elapsed = round(time.time() - t0, 1)

        if not r.get('solved'):
            status = 'timeout' if elapsed >= time_limit - 2 else 'infeasible'
            return {'status': status, 'cost': None, 'time': elapsed,
                    'plans': r.get('num_plans')}
        return {
            'status':     'solved',
            'cost':       round(r.get('total_cost', 0) / 1e6, 4),
            'time':       elapsed,
            'plans':      r.get('num_plans'),
            'n_open':     len(r.get('facilities_open', [])),
            'facilities': r.get('facilities_open', []),
            'avg_trt':    r.get('avg_trt'),
        }
    except Exception as e:
        return {'status': 'error', 'cost': None,
                'time': round(time.time() - t0, 1), 'err': str(e)[:120]}

# ── heatmap ───────────────────────────────────────────────────────────────────

def build_heatmap(results, solver_key):
    sr = results.get(solver_key, {})
    z, text, colors = [], [], []
    status_order = {'solved': 3, 'timeout': 2, 'infeasible': 1, 'missing': 0, 'error': 0}
    color_map    = {'solved': '#22c55e', 'timeout': '#f59e0b',
                    'infeasible': '#ef4444', 'missing': '#e5e7eb', 'error': '#8b5cf6'}

    for n in N_VALUES:
        row_z, row_t, row_c = [], [], []
        for s in SIGMA_VALUES:
            r = sr.get(n, {}).get(s, {'status': 'missing'})
            st_  = r.get('status', 'missing')
            val  = status_order.get(st_, 0)
            cost = f"${r['cost']:.3f}M" if r.get('cost') else '—'
            t    = f"{r['time']}s" if r.get('time') else ''
            row_z.append(val)
            row_t.append(f"N={n}, σ={s}<br>{st_.capitalize()}<br>{cost}  {t}")
            row_c.append(color_map.get(st_, '#e5e7eb'))
        z.append(row_z)
        text.append(row_t)
        colors.append(row_c)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(s) for s in SIGMA_VALUES],
        y=[f'N={n}' for n in N_VALUES],
        text=text,
        hovertemplate='%{text}<extra></extra>',
        colorscale=[
            [0.00, '#e5e7eb'],
            [0.33, '#ef4444'],
            [0.66, '#f59e0b'],
            [1.00, '#22c55e'],
        ],
        showscale=False,
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        xaxis_title='σ (manufacturing uncertainty)',
        yaxis_title='N (patients per trimester)',
        height=460,
        margin=dict(l=80, r=20, t=20, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    return fig

# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='CAR-T Feasibility Explorer', layout='wide')
st.title('CAR-T Supply Chain — Feasibility Explorer')
st.caption('Select a model to view its feasibility grid, then run any new (N, σ) combination.')

# Legend
cols = st.columns(4)
for col, (label, color) in zip(cols, [
    ('Solved', '#22c55e'), ('Infeasible', '#ef4444'),
    ('Timeout', '#f59e0b'), ('No data', '#e5e7eb')
]):
    col.markdown(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{color};border-radius:3px;margin-right:6px;vertical-align:middle"></span>'
        f'**{label}**', unsafe_allow_html=True)

st.divider()

# Model tabs
tab_labels = list(MODELS.keys())
tabs = st.tabs(tab_labels)

results = load_checkpoint()

for tab, (model_label, solver_key) in zip(tabs, MODELS.items()):
    with tab:
        st.plotly_chart(build_heatmap(results, solver_key),
                        use_container_width=True, key=f'heat_{solver_key}')

        st.markdown('#### Run a new combination')
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

        n_input     = c1.selectbox('N (patients/trimester)',
                                   N_VALUES + [250, 300, 400],
                                   key=f'n_{solver_key}')
        sigma_input = c2.selectbox('σ', SIGMA_VALUES,
                                   key=f's_{solver_key}')
        tlim        = c3.number_input('Time limit (s)', min_value=10,
                                      max_value=7200, value=120,
                                      key=f't_{solver_key}')

        # Check if cached
        cached = results.get(solver_key, {}).get(n_input, {}).get(sigma_input)
        if cached:
            c4.markdown('<br>', unsafe_allow_html=True)
            c4.info('Cached ↓')
            _display = cached
        else:
            _display = None

        run_btn = st.button('▶ Run', key=f'run_{solver_key}')

        if run_btn:
            dat_path = os.path.join(BASE_DIR, f'Data_N{n_input}.dat')
            if not os.path.exists(dat_path):
                st.error(f'Data_N{n_input}.dat not found. Generate it first.')
            else:
                with st.spinner(f'Running {model_label} N={n_input} σ={sigma_input} …'):
                    res = run_solver(solver_key, n_input, sigma_input, int(tlim))
                save_result(solver_key, n_input, sigma_input, res)
                _display = res
                results = load_checkpoint()
                st.rerun()

        if _display:
            st_ = _display.get('status', '?')
            color = STATUS_COLOR.get(st_, '#9ca3af')
            st.markdown(
                f'<div style="background:{color};color:#fff;padding:12px 20px;'
                f'border-radius:8px;margin-top:8px">'
                f'<b>{st_.upper()}</b>'
                + (f' — Cost: <b>${_display["cost"]:.3f}M</b>' if _display.get("cost") else '')
                + (f' | Facilities: <b>{", ".join(_display.get("facilities", []))}</b>'
                   if _display.get('facilities') else '')
                + (f' | Avg TRT: <b>{_display.get("avg_trt")} days</b>'
                   if _display.get('avg_trt') else '')
                + (f' | Plans: {_display.get("plans")}' if _display.get('plans') else '')
                + (f' | Time: {_display.get("time")}s' if _display.get('time') else '')
                + '</div>',
                unsafe_allow_html=True
            )
