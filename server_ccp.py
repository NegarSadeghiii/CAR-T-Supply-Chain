"""
CAR-T Supply Chain — CCP Stochastic UI Server
Serves ui_ccp.html and provides API endpoints for:
  - Single CCP / deterministic / Gamma-robust experiment
  - Sensitivity sweep (alpha x sigma_L grid)
  - Full comparison + Monte Carlo validation (background job)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, threading, uuid, traceback

app = Flask(__name__, static_folder=None)
CORS(app)

from cart_colgen_ccp import (
    run_experiment_ccp, run_sensitivity_ccp,
    DEFAULT_PROCESS_CCP, DEFAULT_PROCESS_DET,
    compute_tmfe_quantile,
)
from cart_ccp_simulation import (
    run_full_comparison, run_experiment_robust, simulate_schedule,
)
from cart_colgen import parse_dat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
jobs = {}   # background job store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dat(n):
    return os.path.join(BASE_DIR, f'Data_N{n}.dat')

def _cfg_ccp(p, sigma_L=None):
    cfg = dict(DEFAULT_PROCESS_CCP)
    cfg['alpha']   = float(p.get('alpha',   0.10))
    cfg['sigma_L'] = float(p.get('sigma_L', 0.20) if sigma_L is None else sigma_L)
    cfg['tmfe']    = float(p.get('tmfe', 7))
    cfg['tls']     = int(p.get('tls', 1))
    cfg['tqc']     = int(p.get('tqc', 7))
    return cfg

def _slim(r):
    """Strip large patient list for job polling payloads."""
    return {k: v for k, v in r.items() if k != 'patients'}


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'ui_ccp.html')


# ---------------------------------------------------------------------------
# Routes — single experiment
# ---------------------------------------------------------------------------

@app.route('/api/ccp/run', methods=['POST'])
def ccp_run():
    p      = request.json or {}
    n      = int(p.get('n', 15))
    method = p.get('method', 'ccp')   # 'ccp' | 'deterministic' | 'robust'
    df     = _dat(n)

    if not os.path.exists(df):
        return jsonify({'solved': False, 'error': f'Data_N{n}.dat not found'})

    try:
        if method == 'deterministic':
            cfg = dict(DEFAULT_PROCESS_DET)
            cfg['sigma_L'] = 0.0
            cfg['alpha']   = 0.50
            r = run_experiment_ccp(data_file=df, process_config=cfg, time_limit=120)
            r['method'] = 'Deterministic'
        elif method == 'robust':
            gamma = float(p.get('gamma', 1.0))
            cfg   = _cfg_ccp(p)
            r = run_experiment_robust(gamma=gamma, data_file=df,
                                      process_config=cfg, time_limit=120)
        else:  # ccp
            cfg = _cfg_ccp(p)
            r   = run_experiment_ccp(data_file=df, process_config=cfg, time_limit=120)
            r['method'] = f"CCP(α={cfg['alpha']})"

        # Attach quick simulation if solved
        if r.get('solved') and r.get('patients'):
            sigma_L = float(p.get('sigma_L', 0.20))
            n_sim   = int(p.get('n_sim', 500))
            sim = simulate_schedule(
                patients=r['patients'],
                tmfe_eff=r['tmfe_eff'],
                tmfe_nominal=r.get('tmfe_nominal', 7),
                sigma_L=sigma_L,
                n_sim=n_sim, seed=42,
            )
            r['simulation'] = sim

        return jsonify(r)
    except Exception as e:
        return jsonify({'solved': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


# ---------------------------------------------------------------------------
# Routes — sensitivity sweep
# ---------------------------------------------------------------------------

@app.route('/api/ccp/sensitivity', methods=['POST'])
def ccp_sensitivity():
    p       = request.json or {}
    n       = int(p.get('n', 15))
    alphas  = p.get('alphas',  [0.20, 0.10, 0.05, 0.01])
    sigma_Ls= p.get('sigma_Ls',[0.10, 0.20, 0.30, 0.50])
    df      = _dat(n)

    if not os.path.exists(df):
        return jsonify({'error': f'Data_N{n}.dat not found'})

    try:
        results = run_sensitivity_ccp(
            data_file=df, alphas=alphas, sigma_Ls=sigma_Ls,
            time_limit=60, include_deterministic=True,
        )
        # Slim each result
        return jsonify([_slim(r) for r in results])
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()})


# ---------------------------------------------------------------------------
# Routes — full validation (background job)
# ---------------------------------------------------------------------------

@app.route('/api/ccp/validate', methods=['POST'])
def ccp_validate():
    p      = request.json or {}
    n      = int(p.get('n', 15))
    sigma_L= float(p.get('sigma_L', 0.20))
    alphas = p.get('alphas',  [0.20, 0.10, 0.05])
    gammas = p.get('gammas',  [0.5,  1.0,  1.5])
    n_sim  = int(p.get('n_sim', 1000))
    df     = _dat(n)

    if not os.path.exists(df):
        return jsonify({'error': f'Data_N{n}.dat not found'})

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        'status':   'running',
        'progress': 0.0,
        'message':  'Starting…',
        'results':  [],
        'total':    1 + len(alphas) + len(gammas),
        'done':     0,
    }

    def worker():
        try:
            all_results = []
            total = jobs[job_id]['total']

            def _step(r):
                all_results.append(_slim(r))
                jobs[job_id]['done']    += 1
                jobs[job_id]['progress'] = jobs[job_id]['done'] / total
                jobs[job_id]['results']  = all_results
                jobs[job_id]['message']  = f"Done {jobs[job_id]['done']}/{total}: {r.get('method','?')}"

            # Deterministic
            jobs[job_id]['message'] = 'Running Deterministic…'
            cfg = dict(DEFAULT_PROCESS_DET); cfg['sigma_L']=0.0; cfg['alpha']=0.50
            r = run_experiment_ccp(data_file=df, process_config=cfg, time_limit=120)
            r['method'] = 'Deterministic'
            if r.get('solved') and r.get('patients'):
                r['simulation'] = simulate_schedule(r['patients'], r['tmfe_eff'],
                    r.get('tmfe_nominal',7), sigma_L, n_sim=n_sim, seed=42)
            _step(r)

            # CCP variants
            for alpha in alphas:
                jobs[job_id]['message'] = f'Running CCP(α={alpha})…'
                cfg = _cfg_ccp({'alpha': alpha, 'sigma_L': sigma_L})
                r = run_experiment_ccp(data_file=df, process_config=cfg, time_limit=120)
                r['method'] = f'CCP(α={alpha})'
                if r.get('solved') and r.get('patients'):
                    r['simulation'] = simulate_schedule(r['patients'], r['tmfe_eff'],
                        r.get('tmfe_nominal',7), sigma_L, n_sim=n_sim, seed=42)
                _step(r)

            # Gamma-robust variants
            for gamma in gammas:
                jobs[job_id]['message'] = f'Running Γ-robust(Γ={gamma})…'
                cfg = _cfg_ccp({'sigma_L': sigma_L})
                r = run_experiment_robust(gamma=gamma, data_file=df,
                                          process_config=cfg, time_limit=120)
                if r.get('solved') and r.get('patients'):
                    r['simulation'] = simulate_schedule(r['patients'], r['tmfe_eff'],
                        r.get('tmfe_nominal',7), sigma_L, n_sim=n_sim, seed=42)
                _step(r)

            jobs[job_id]['status']   = 'complete'
            jobs[job_id]['progress'] = 1.0
            jobs[job_id]['message']  = 'Complete'

        except Exception as e:
            jobs[job_id]['status']  = 'error'
            jobs[job_id]['message'] = str(e)
            jobs[job_id]['traceback'] = traceback.format_exc()

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/ccp/job/<job_id>')
def ccp_job(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'not found'}), 404
    j = jobs[job_id]
    return jsonify({
        'status':   j['status'],
        'progress': j['progress'],
        'message':  j['message'],
        'results':  j['results'],
        'done':     j['done'],
        'total':    j['total'],
    })


# ---------------------------------------------------------------------------
# Routes — data info
# ---------------------------------------------------------------------------

@app.route('/api/ccp/datasets')
def datasets():
    avail = []
    for n in [5, 10, 15, 20, 25, 40, 50]:
        if os.path.exists(_dat(n)):
            avail.append(n)
    return jsonify(avail)


if __name__ == '__main__':
    import multiprocessing
    cores = multiprocessing.cpu_count()
    print(f'\n  CAR-T CCP Stochastic UI Server')
    print(f'  CCP · Gamma-Robust · Monte Carlo Validation ({cores} cores)')
    print(f'  http://localhost:5051\n')
    app.run(host='0.0.0.0', port=5051, debug=False)
