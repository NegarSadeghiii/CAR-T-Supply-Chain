"""
CAR-T Supply Chain — Integrated API Server
ColGen + Benders + MIP benchmark, scenario analysis, single experiments.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, json, threading, uuid, traceback, re
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__, static_folder=None)
CORS(app)

from cart_colgen import (run_experiment as colgen_run, DEFAULT_URGENCY,
                         DEFAULT_PROCESS, parse_dat)
from cart_benders import run_experiment as benders_run
from generate_small_data import generate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
jobs = {}


def _urg(raw):
    if not raw: return None
    return {g: {'fraction': raw[g].get('fraction', DEFAULT_URGENCY[g]['fraction']),
                'base_due': raw[g].get('base_due', DEFAULT_URGENCY[g]['base_due']),
                'tolerance': raw[g].get('tolerance', DEFAULT_URGENCY[g]['tolerance'])}
            for g in ('high','medium','low') if g in raw}

def _proc(raw):
    if not raw: return None
    return {k: raw.get(k, DEFAULT_PROCESS[k]) for k in DEFAULT_PROCESS}

def _ensure_data(n):
    f = os.path.join(BASE_DIR, f'Data_N{n}.dat')
    if not os.path.exists(f):
        generate(input_file=os.path.join(BASE_DIR,'Data200_profileA.dat'),
                 output_file=f, num_patients=n)
    return f


def _get_arrivals(data_file):
    """Parse patient arrival days from data file."""
    with open(data_file) as f:
        txt = f.read()
    m = re.search(r'param INC\s*:=(.*?);', txt, re.DOTALL)
    arrivals = {}
    if m:
        for p, c, t, v in re.findall(r'(\S+)\s+(\S+)\s+(\d+)\s+(\d+)', m.group(1)):
            if int(v) == 1:
                arrivals[p] = int(t)
    return arrivals


def _normalize(r, method, data_file=None):
    """Normalize solver output to a consistent format for the UI."""
    if not r.get('solved') and not r.get('feasible'):
        r['solved'] = False
        r.setdefault('solver', method)
        return r

    r['solved'] = True
    r['solver'] = method
    r.setdefault('real_cost', r.get('total_cost', 0))
    r.setdefault('num_patients', len(r.get('patients', [])))

    # Get arrivals for patient data enrichment
    arrivals = _get_arrivals(data_file) if data_file else {}

    # Normalize patient entries
    patients = r.get('patients', [])
    num_late = 0
    total_lateness = 0
    for p in patients:
        pid = p.get('id', '')
        p.setdefault('turnaround', p.get('trt', 0))
        p.setdefault('trt', p.get('turnaround', 0))
        p.setdefault('transport_out', p.get('j_out', '?'))
        p.setdefault('transport_ret', p.get('j_ret', '?'))
        p.setdefault('arrival_day', arrivals.get(pid, 0))
        p.setdefault('completion_day', p['arrival_day'] + p['turnaround'])
        dl = p.get('deadline', 999)
        late = max(0, p['turnaround'] - dl)
        p.setdefault('lateness', late)
        p.setdefault('on_time', late < 0.01)
        if late > 0.01:
            num_late += 1
            total_lateness += late

    r.setdefault('num_late', num_late)
    r.setdefault('total_lateness', total_lateness)
    r.setdefault('all_on_time', num_late == 0)

    # Per-patient cost
    if r['num_patients'] > 0:
        r['per_patient_cost'] = round(r['real_cost'] / r['num_patients'], 2)

    return r


def _run_solver(method, **kwargs):
    """Dispatch to the appropriate solver."""
    if method == 'benders':
        return benders_run(**kwargs)
    elif method == 'mip':
        from hard_deadline_model import run_experiment as mip_run
        r = mip_run(tau=kwargs.get('tau', 0),
                    data_file=kwargs.get('data_file'),
                    urgency_config=kwargs.get('urgency_config'),
                    time_limit=kwargs.get('time_limit', 120))
        if r.get('feasible'):
            r['solved'] = True
        return r
    else:  # colgen (default)
        return colgen_run(**kwargs)


@app.route('/')
def index(): return send_from_directory(BASE_DIR, 'ui.html')

@app.route('/api/run', methods=['POST'])
def run_single():
    p = request.json or {}
    n = p.get('n', 25)
    method = p.get('method', 'colgen')
    _ensure_data(n)
    df = os.path.join(BASE_DIR, f'Data_N{n}.dat')
    try:
        r = _run_solver(method, tau=p.get('tau', 0), data_file=df,
                        urgency_config=_urg(p.get('urgency')),
                        process_config=_proc(p.get('process')),
                        time_limit=p.get('time_limit', 120))
        r = _normalize(r, method, df)
        return jsonify(r)
    except Exception as e:
        return jsonify({'solved': False, 'error': str(e), 'solver': method})

@app.route('/api/generate-data', methods=['POST'])
def gen_data():
    n = request.json.get('num_patients',10)
    _ensure_data(n)
    return jsonify({'ok':True,'n':n})

@app.route('/api/scenarios', methods=['POST'])
def run_scenarios():
    """Run multiple scenarios in parallel."""
    scens = request.json.get('scenarios', [])
    method = request.json.get('method', 'colgen')
    if not scens: return jsonify([])

    for s in scens:
        _ensure_data(s.get('n', 25))

    def run_one(s):
        n = s.get('n', 25)
        df = os.path.join(BASE_DIR, f"Data_N{n}.dat")
        m = s.get('method', method)
        try:
            r = _run_solver(m, tau=s.get('tau', 0), data_file=df,
                            urgency_config=_urg(s.get('urgency')),
                            process_config=_proc(s.get('process')),
                            time_limit=s.get('time_limit', 120))
            r = _normalize(r, m, df)
            r['_label'] = s.get('label', '')
            r['_idx'] = s.get('idx', 0)
            return r
        except Exception as e:
            return {'solved': False, 'error': str(e), '_label': s.get('label', ''),
                    '_idx': s.get('idx', 0), 'solver': m}

    with ThreadPoolExecutor(max_workers=min(len(scens), os.cpu_count() or 4)) as ex:
        futures = {ex.submit(run_one, s): s.get('idx', i) for i, s in enumerate(scens)}
        results = {}
        for f in as_completed(futures):
            idx = futures[f]
            results[idx] = f.result()
    return jsonify([results[i] for i in sorted(results)])

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Compare MIP vs ColGen vs Benders across patient counts."""
    p = request.json or {}
    sizes = p.get('sizes', [5, 10, 15, 25])
    tau = p.get('tau', 0.0)
    time_limit = p.get('time_limit', 120)
    mip_limit = min(time_limit, 600)  # cap MIP at 10 min

    results = []
    for n in sizes:
        _ensure_data(n)
        df = os.path.join(BASE_DIR, f'Data_N{n}.dat')
        entry = {'n': n, 'tau': tau}

        # Column Generation
        try:
            r_cg = colgen_run(tau=tau, data_file=df, time_limit=time_limit)
            r_cg = _normalize(r_cg, 'colgen', df)
            entry['colgen'] = {
                'solved': r_cg.get('solved', False),
                'cost': r_cg.get('real_cost', 0),
                'avg_trt': r_cg.get('avg_trt', 0),
                'per_patient_cost': r_cg.get('per_patient_cost', 0),
                'solve_time': r_cg.get('solve_time', 0),
                'optimal': r_cg.get('optimal', False),
                'num_plans': r_cg.get('num_plans', 0),
                'facilities': r_cg.get('facilities_open', []),
            }
        except Exception as e:
            entry['colgen'] = {'solved': False, 'error': str(e)}

        # Benders
        try:
            r_bd = benders_run(tau=tau, data_file=df, time_limit=time_limit)
            r_bd = _normalize(r_bd, 'benders', df)
            entry['benders'] = {
                'solved': r_bd.get('solved', False),
                'cost': r_bd.get('real_cost', 0),
                'avg_trt': r_bd.get('avg_trt', 0),
                'per_patient_cost': r_bd.get('per_patient_cost', 0),
                'solve_time': r_bd.get('solve_time', 0),
                'iterations': r_bd.get('iterations', 0),
                'num_cuts': r_bd.get('num_cuts', 0),
                'facilities': r_bd.get('facilities_open', []),
            }
        except Exception as e:
            entry['benders'] = {'solved': False, 'error': str(e)}

        # MIP (hard deadline) — may timeout for large N
        try:
            from hard_deadline_model import run_experiment as mip_run
            r_mip = mip_run(tau=tau, data_file=df,
                            urgency_config=_urg(None),
                            time_limit=mip_limit)
            r_mip = _normalize(r_mip, 'mip', df)
            entry['mip'] = {
                'solved': r_mip.get('solved', False),
                'cost': r_mip.get('real_cost', 0),
                'avg_trt': r_mip.get('avg_trt', 0),
                'per_patient_cost': r_mip.get('per_patient_cost', 0),
                'solve_time': r_mip.get('solve_time', 0),
                'facilities': r_mip.get('facilities_open', []),
            }
        except Exception as e:
            entry['mip'] = {'solved': False, 'error': str(e)}

        # Compute optimality gaps vs ColGen (best method)
        cg_cost = entry['colgen'].get('cost', 0) if entry['colgen'].get('solved') else 0
        if cg_cost > 0:
            for m in ['benders', 'mip']:
                if entry[m].get('solved') and entry[m]['cost'] > 0:
                    entry[f'gap_{m}_pct'] = round(
                        (entry[m]['cost'] - cg_cost) / cg_cost * 100, 2)
            # Speedup
            cg_time = entry['colgen'].get('solve_time', 0.001)
            for m in ['benders', 'mip']:
                if entry[m].get('solved') and entry[m].get('solve_time', 0) > 0:
                    entry[f'speedup_{m}'] = round(
                        entry[m]['solve_time'] / max(cg_time, 0.001), 0)

        results.append(entry)
    return jsonify(results)


@app.route('/api/feasibility', methods=['POST'])
def feasibility_boundary():
    """Sweep tau in fine steps to find the exact feasibility boundary."""
    p = request.json or {}
    n = p.get('n', 25)
    steps = p.get('steps', 21)  # 0.00, 0.05, 0.10 ... 1.00
    uc = _urg(p.get('urgency'))
    pc = _proc(p.get('process'))

    _ensure_data(n)
    df = os.path.join(BASE_DIR, f'Data_N{n}.dat')

    taus = [round(i / (steps - 1), 4) for i in range(steps)]
    results = []

    def run_one(tau):
        try:
            r = colgen_run(tau=tau, data_file=df, urgency_config=uc,
                           process_config=pc, time_limit=30)
            r = _normalize(r, 'colgen', df)
            return {'tau': tau, 'solved': r.get('solved', False),
                    'cost': r.get('real_cost', 0), 'avg_trt': r.get('avg_trt', 0),
                    'num_late': r.get('num_late', 0), 'total_lateness': r.get('total_lateness', 0),
                    'all_on_time': r.get('all_on_time', False),
                    'num_patients': r.get('num_patients', n)}
        except Exception as e:
            return {'tau': tau, 'solved': False, 'error': str(e)}

    # Run all in parallel
    with ThreadPoolExecutor(max_workers=min(steps, os.cpu_count() or 4)) as ex:
        futures = {ex.submit(run_one, t): t for t in taus}
        res_map = {}
        for f in as_completed(futures):
            t = futures[f]
            res_map[t] = f.result()

    results = [res_map[t] for t in taus]

    # Find boundary
    feasible_taus = [r['tau'] for r in results if r.get('all_on_time')]
    infeasible_taus = [r['tau'] for r in results if r.get('solved') and not r.get('all_on_time')]
    boundary = max(feasible_taus) if feasible_taus else None
    first_late = min(infeasible_taus) if infeasible_taus else None

    return jsonify({
        'results': results,
        'boundary_tau': boundary,
        'first_late_tau': first_late,
        'n': n,
    })


# ══════ RL ENDPOINTS ══════

rl_jobs = {}

@app.route('/api/rl/train', methods=['POST'])
def rl_train():
    """Start RL training in background."""
    p = request.json or {}
    n = p.get('n', 25)
    tau = p.get('tau', 0.0)
    agents = p.get('agents', ['PPO', 'DQN', 'A2C'])
    timesteps = p.get('timesteps', 50000)
    curriculum = p.get('curriculum', True)

    _ensure_data(n)
    job_id = str(uuid.uuid4())[:8]
    rl_jobs[job_id] = {'status': 'running', 'progress': 0, 'message': 'Starting...',
                       'results': {}, 'agents': agents}

    def worker():
        try:
            from rl_agents.train_rl import train_all
            def prog(frac, msg):
                rl_jobs[job_id]['progress'] = round(frac, 3)
                rl_jobs[job_id]['message'] = msg

            results = train_all(
                data_file=os.path.join(BASE_DIR, f'Data_N{n}.dat'),
                tau=tau, agent_types=agents,
                total_timesteps=timesteps,
                curriculum=curriculum,
                progress_callback=prog)

            rl_jobs[job_id]['results'] = {
                k: {kk: vv for kk, vv in v.items() if kk != 'metrics'}
                for k, v in results.items()}
            rl_jobs[job_id]['status'] = 'complete'
            rl_jobs[job_id]['progress'] = 1.0
            rl_jobs[job_id]['message'] = 'Training complete'
        except Exception as e:
            rl_jobs[job_id]['status'] = 'error'
            rl_jobs[job_id]['message'] = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/rl/status/<job_id>')
def rl_status(job_id):
    """Poll training progress."""
    if job_id not in rl_jobs:
        return jsonify({'error': 'not found'}), 404

    job = rl_jobs[job_id]
    resp = {
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'agents': job['agents'],
    }

    # Include training logs — ALWAYS read from disk (live during training)
    if job['status'] in ('complete', 'running'):
        if job['status'] == 'complete':
            resp['results'] = job['results']
        curves = {}
        log_dir = os.path.join(BASE_DIR, 'rl_models')
        if os.path.exists(log_dir):
            for agent in job['agents']:
                for d in os.listdir(log_dir):
                    if d.startswith(agent.lower()):
                        log_file = os.path.join(log_dir, d, 'training_log.json')
                        if os.path.exists(log_file):
                            try:
                                with open(log_file) as f:
                                    data = json.load(f)
                                if len(data) > 200:
                                    step = len(data) // 200
                                    data = data[::step]
                                curves[agent] = data
                            except: pass
        resp['curves'] = curves

    return jsonify(resp)


@app.route('/api/rl/eval', methods=['POST'])
def rl_eval():
    """Evaluate trained models against CP-SAT."""
    p = request.json or {}
    n = p.get('n', 25)
    tau = p.get('tau', 0.0)
    agents = p.get('agents', ['PPO', 'DQN', 'A2C'])
    episodes = p.get('episodes', 10)

    _ensure_data(n)
    try:
        from rl_agents.eval_rl import evaluate_all
        results = evaluate_all(
            data_file=os.path.join(BASE_DIR, f'Data_N{n}.dat'),
            tau=tau, agent_types=agents, n_episodes=episodes)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/rl/models')
def rl_models():
    """List available trained models."""
    model_dir = os.path.join(BASE_DIR, 'rl_models')
    models = []
    if os.path.exists(model_dir):
        for d in sorted(os.listdir(model_dir)):
            cfg_file = os.path.join(model_dir, d, 'config.json')
            if os.path.exists(cfg_file):
                with open(cfg_file) as f:
                    cfg = json.load(f)
                cfg['dir'] = d
                models.append(cfg)
    return jsonify(models)


if __name__ == '__main__':
    import multiprocessing
    cores = multiprocessing.cpu_count()
    print(f'\n  CAR-T Supply Chain Server')
    print(f'  ColGen + Benders + MIP ({cores} cores)')
    print(f'  http://localhost:5050\n')
    app.run(host='127.0.0.1', port=5050, debug=False)
