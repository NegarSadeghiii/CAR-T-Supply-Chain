"""Shared machinery for the INFORMS runs.

Everything about the optimisation model itself comes from the notebook via
informs_nb; this module only adds (a) a solve wrapper that logs status, gap and
runtime, (b) the policy definitions, which are precedence constraints layered
on top of the unchanged model, and (c) the per-run bookkeeping.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import informs_nb

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_informs")
CACHE = os.path.join(RESULTS, "_cache")

SOLVER_NAME = "appsi_highs"
MIP_GAP = 1e-6
TIME_LIMIT = 300.0

# cost tolerance for "same cost as the cheapest plan"  (prompt: C_min*1.001)
COST_TOL = 1e-3

_NS = None


def ns():
    """The notebook's namespace, loaded once."""
    global _NS
    if _NS is None:
        _NS = informs_nb.load_checked(verbose=False)
    return _NS


def nb(name):
    return ns()[name]


# -----------------------------------------------------------------------------
# Solving, with every solve logged
# -----------------------------------------------------------------------------
def solve_logged(model, label=""):
    """Solve and return (ok, log).  Mirrors the notebook's status guard -- a
    solution is loaded only when one exists -- and adds the gap/runtime record
    the experiment protocol asks for."""
    opt = pyo.SolverFactory(SOLVER_NAME)
    if not opt.available(exception_flag=False):
        return False, dict(label=label, status="solver unavailable",
                           gap=np.nan, runtime=0.0, optimal=False)
    opt.config.time_limit = TIME_LIMIT
    opt.config.mip_gap = MIP_GAP
    t0 = time.perf_counter()
    res = opt.solve(model, tee=False, load_solutions=False)
    dt = time.perf_counter() - t0
    tc = res.solver.termination_condition
    ok = tc == pyo.TerminationCondition.optimal
    if ok:
        model.solutions.load_from(res)
    try:
        ub, lb = float(res.problem.upper_bound), float(res.problem.lower_bound)
        gap = abs(ub - lb) / max(abs(ub), 1e-9)
    except Exception:
        gap = np.nan
    return ok, dict(label=label, status=f"{SOLVER_NAME}: {tc}", gap=gap,
                    runtime=dt, optimal=bool(ok))


def set_objective(m, expr, sense):
    """Replace the model's objective without touching a single constraint."""
    m.OBJ.deactivate()
    if hasattr(m, "OBJ_ALT"):
        m.del_component(m.OBJ_ALT)
    m.OBJ_ALT = pyo.Objective(expr=expr, sense=sense)
    return m


def total_trt(m):
    return sum(m.TRT[p] for p in m.P)


# -----------------------------------------------------------------------------
# Policy precedence constraints
#
# start_p = sum_{f,t} t * INM[p,f,t]            (the day patient p enters a line)
# z[p,f]  = sum_t INM[p,f,t]                    (1 when p is made at facility f)
#
# A precedence only binds when both patients share a facility, which is itself a
# decision, so it is written with a big-M that switches off unless z_pf = z_qf = 1.
# M = horizon is valid because every start day lies in 1..horizon.
# -----------------------------------------------------------------------------
def _start(m, p):
    return sum(t * m.INM[p, f, t] for f in m.M for t in m.T)


def _at(m, p, f):
    return sum(m.INM[p, f, t] for t in m.T)


def _precede(m, pairs, tag):
    """start_p <= start_q whenever p and q are built at the same facility."""
    big_m = float(max(m.T))
    for p, q in pairs:
        for f in m.M:
            m.con.add(_start(m, p) - _start(m, q)
                      <= big_m * (2 - _at(m, p, f) - _at(m, q, f)))
    m._policy_pairs = (tag, len(pairs))
    return m


def add_fifo(m):
    """Patients enter a line in arrival order.  Patients collected on the same
    day are left unordered -- the rule only speaks about a_p < a_q."""
    pat = m._inst["patients"]
    pairs = [(p, q) for p in m.P for q in m.P
             if p != q and pat[p]["t0"] < pat[q]["t0"]]
    return _precede(m, pairs, "fifo")


def add_high_risk_first(m):
    """Every high-risk patient enters a line no later than any low-risk patient
    built at the same facility."""
    pat = m._inst["patients"]
    pairs = [(p, q) for p in m.P for q in m.P
             if pat[p]["g"] == "High" and pat[q]["g"] == "Low"]
    return _precede(m, pairs, "high_risk_first")


POLICY_PRECEDENCE = {"fifo": add_fifo, "high_risk_first": add_high_risk_first}

# Every policy gets the SAME secondary preference -- minimise total turnaround
# -- so a difference in ELIG between two policies is a difference in what the
# policy allows, not in how its ties happened to be broken.
POLICIES = ["cost_speed", "fifo", "high_risk_first", "elig_min", "elig_max"]


def min_cost(inst, label="cost_min"):
    m = nb("build_model")(inst, objective="cost")
    ok, log = solve_logged(m, label)
    return (m, log) if ok else (None, log)


def run_policy(inst, policy, cost_cap, label=None):
    """Return (model, log).  cost_cap=None means no budget constraint."""
    label = label or policy
    build = nb("build_model")
    logs = []

    if policy == "elig_min":
        m = build(inst, objective="elig", cost_cap=cost_cap)
        set_objective(m, m.ELIG, pyo.minimize)
        ok, log = solve_logged(m, label)
        return (m if ok else None), [log]

    if policy == "elig_max":
        ma = build(inst, objective="elig", cost_cap=cost_cap)
        ok, log = solve_logged(ma, label + ":stage1")
        logs.append(log)
        if not ok:
            return None, logs
        e_star = float(pyo.value(ma.ELIG))
        mb = build(inst, objective="elig", cost_cap=cost_cap)
        mb.con.add(mb.ELIG >= e_star - 1e-9)
        set_objective(mb, total_trt(mb), pyo.minimize)
        ok2, log2 = solve_logged(mb, label + ":stage2")
        logs.append(log2)
        return (mb if ok2 else ma), logs

    # cost_speed / fifo / high_risk_first: cheapest plan, then fastest schedule
    m = build(inst, objective="cost", cost_cap=cost_cap)
    if policy in POLICY_PRECEDENCE:
        POLICY_PRECEDENCE[policy](m)
    set_objective(m, total_trt(m), pyo.minimize)
    ok, log = solve_logged(m, label)
    return (m if ok else None), [log]


# -----------------------------------------------------------------------------
# Reading a solved model
# -----------------------------------------------------------------------------
def summarise(m):
    d = nb("itinerary")(m)
    hi, lo = d[d.group == "High"], d[d.group == "Low"]
    return dict(cost=float(pyo.value(m.TOTCOST)),
                ELIG=float(pyo.value(m.ELIG)),
                CTM=float(pyo.value(m.CTM)),
                TTC=float(pyo.value(m.TTC)),
                HOLD=float(pyo.value(m.HOLD)),
                total_wait_days=float(d.W.sum()),
                total_TRT=float(d.TRT.sum()),
                mean_TRT_H=float(hi.TRT.mean()),
                mean_TRT_L=float(lo.TRT.mean()),
                max_TRT=float(d.TRT.max()))


def patient_frame(m):
    d = nb("itinerary")(m).copy()
    d = d.rename(columns={"a_p": "collection_day", "mode_out": "out_mode",
                          "mode_return": "ret_mode", "start": "start_day",
                          "W": "wait_days", "delivery": "delivery_day",
                          "sigma": "S"})
    return d[["patient", "group", "facility", "out_mode", "ret_mode",
              "collection_day", "arrival", "start_day", "wait_days",
              "delivery_day", "TRT", "S"]]


def lever_counts(d_ref, d_new):
    """How many patients the eligibility-aware plan moves, lever by lever."""
    a = d_ref.set_index("patient")
    b = d_new.set_index("patient").reindex(a.index)
    order = lambda d: d.start_day.rank(method="min")
    return dict(n_start_order_changed=int((order(a) != order(b)).sum()),
                n_start_day_changed=int((a.start_day != b.start_day).sum()),
                n_mode_changed=int(((a.out_mode != b.out_mode)
                                    | (a.ret_mode != b.ret_mode)).sum()),
                n_facility_changed=int((a.facility != b.facility).sum()))


# -----------------------------------------------------------------------------
# Weibull helpers
# -----------------------------------------------------------------------------
def S(t, k, eta):
    t = np.asarray(t, dtype=float)
    return np.exp(-np.power(np.maximum(t, 0.0) / eta, k))


def inversion_day(curves, t_max=400):
    """The day urgency changes hands.

    The optimiser compares one-day losses, Delta_g(t) = S_g(t) - S_g(t+1), not
    survival levels.  With a steeper high-risk shape the two densities cross
    twice: Delta_H starts below Delta_L (both densities leave zero, the flatter
    one first), rises above it, then falls below again once the high-risk group
    is largely lost.  Only the SECOND crossing matters -- before it a day saved
    is worth more to a high-risk patient, after it to a low-risk one -- so the
    first crossing, which sits at t ~ 0, is deliberately skipped.

    Returns inf when the high-risk group stays the more urgent one throughout,
    and 0.0 when it is never the more urgent one.
    """
    H, L = curves["High"], curves["Low"]
    t = np.arange(0, t_max)
    d = ((S(t, H["k"], H["eta"]) - S(t + 1, H["k"], H["eta"]))
         - (S(t, L["k"], L["eta"]) - S(t + 1, L["k"], L["eta"])))
    if not (d > 0).any():
        return 0.0
    peak = int(np.argmax(d))
    after = np.where(d[peak:] < 0)[0]
    return float(peak + after[0]) if len(after) else float("inf")


# -----------------------------------------------------------------------------
# Instances and caching
# -----------------------------------------------------------------------------
def instance(seed=42, window=None, curves=None, **over):
    """BASE, with the seed / window / Weibull curves overridden."""
    cfg = dict(nb("BASE")); cfg["seed"] = int(seed)
    if window is not None:
        cfg["window"] = int(window)
    if curves is not None:
        cfg.update(k_high=curves["High"]["k"], eta_high=curves["High"]["eta"],
                   k_low=curves["Low"]["k"], eta_low=curves["Low"]["eta"])
    cfg.update(over)
    return nb("build_instance")(cfg), cfg


def cache_key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True,
                                   default=str).encode()).hexdigest()[:16]


def cached(key, fn):
    """Run fn() once; a resumed run reads the stored result instead."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = fn()
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    return out


def _pyomo_version():
    import pyomo.version
    return str(pyomo.version.version)


def solver_banner():
    return dict(solver=SOLVER_NAME, mip_gap=MIP_GAP, time_limit_s=TIME_LIMIT,
                cost_tolerance=COST_TOL,
                pyomo=_pyomo_version(),
                numpy=np.__version__, pandas=pd.__version__)


def md_table(df, index=False, floatfmt="{:g}"):
    """Markdown table without pulling in tabulate.

    pandas' to_markdown needs an optional dependency that is not guaranteed to
    be installed where these scripts run, and a summary file is not worth a
    hard dependency.
    """
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [" ".join(str(x) for x in c if str(x) != "")
                     for c in d.columns]
    if index:
        d = d.reset_index()

    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    head = [str(c) for c in d.columns]
    body = [[cell(v) for v in row] for row in d.itertuples(index=False)]
    width = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
             for i, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(
        c.ljust(w) for c, w in zip(cells, width)) + " |"
    return "\n".join([line(head),
                      "| " + " | ".join("-" * w for w in width) + " |",
                      *(line(r) for r in body)])
