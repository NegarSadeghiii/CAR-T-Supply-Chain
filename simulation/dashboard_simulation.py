"""
simulation/dashboard_simulation.py
====================================

Discrete-event simulation adapter for the Opt→Sim→Policy dashboard.

This module sits between the policy layer and the metrics layer:

    Optimizer baseline plan
         |
         v
    Policy (policy_hooks.py) adjusts manufacturing_start_policy
         |
         v
  --> run_sim(patient_schedule)  <-- this module
         |
         v
    Realized per-patient timing + facility utilisation

Key design decision:
    The simulation's entry point for each patient is `manufacturing_start_policy`
    (the day the sample *arrives* at the facility gate, ready to queue).
    This is distinct from:
        • manufacturing_start_baseline  – what the optimizer decided
        • manufacturing_start_policy    – what the policy layer decided
        • manufacturing_start_realized  – when the patient actually enters
                                          the machine (after queuing)

    This three-way split is the core hook for later RL integration:
    an RL agent would modify manufacturing_start_policy before calling run_sim().

Stochastic assumptions (documented):
    • Manufacturing duration  : Triangular(min=4, mode=7, max=10) days
                                mean = 7 d, variance ≈ 1.5 d²
    • QC duration             : Triangular(min=4, mode=7, max=10) days
    • QC batch failure         : Bernoulli(p=0.05) → repeat mfg + QC
    • Eligibility / transport  : Deterministic (TT3 from plan)
    • Resource capacity        : FCAP[m] concurrent patients per facility

Usage:
    from simulation.dashboard_simulation import run_sim

    result = run_sim(patient_schedule, n_replications=20)
    # result['per_patient']  – list of per-patient averaged outcomes
    # result['facility_util'] – dict: facility -> avg utilisation fraction
    # result['kpis']          – aggregate KPIs
"""

import simpy
import numpy as np
from collections import defaultdict

# ── Distribution parameters ────────────────────────────────────────────────────
MFG_MIN,  MFG_MODE,  MFG_MAX  = 4.0, 7.0, 10.0
QC_MIN,   QC_MODE,   QC_MAX   = 4.0, 7.0, 10.0
QC_FAIL_PROB = 0.05


# ── Patient process ────────────────────────────────────────────────────────────

def _patient_process(env, info, fac_resources, rng, rep_out):
    """SimPy generator for one patient in one replication.

    Parameters
    ----------
    env           : simpy.Environment
    info          : dict  with keys:
                      patient_id, facility, manufacturing_start_policy (absolute day),
                      arrival (absolute day), deadline (TRT limit in days),
                      tt3 (return transport days), fcap (int, unused here — on resource)
    fac_resources : dict  facility_id -> simpy.Resource(capacity=FCAP[m])
    rng           : np.random.Generator  (seeded per replication)
    rep_out       : dict  patient_id -> result dict (written at process end)
    """
    pid      = info['patient_id']
    arrival  = float(info['arrival'])
    deadline = float(info['deadline'])
    facility = info['facility']
    tt3      = float(info.get('tt3', 1))
    release  = float(info['manufacturing_start_policy'])   # scheduled gate time

    # ── Wait until the policy-scheduled release time ───────────────────────────
    # This represents: sample collection → TLS → outbound transport.
    # The policy may have shifted this earlier or later than the optimizer said.
    if env.now < release:
        yield env.timeout(release - env.now)

    gate_time = env.now   # time sample is at facility gate, starts queueing

    # ── Manufacturing + QC loop (retry on QC failure) ─────────────────────────
    mfg_start_realized = gate_time   # updated if patient has to wait in queue
    n_remanufactures   = 0

    # First manufacturing attempt
    with fac_resources[facility].request() as req:
        yield req                                      # wait for available slot
        mfg_start_realized = env.now

        mfg_time = float(rng.triangular(MFG_MIN, MFG_MODE, MFG_MAX))
        yield env.timeout(mfg_time)
    # Slot released here — another patient can enter.

    # QC + optional retry loop
    while True:
        qc_time = float(rng.triangular(QC_MIN, QC_MODE, QC_MAX))
        yield env.timeout(qc_time)

        if rng.random() < QC_FAIL_PROB:
            # Batch failed → back to manufacturing
            n_remanufactures += 1
            with fac_resources[facility].request() as req2:
                yield req2
                mfg_time2 = float(rng.triangular(MFG_MIN, MFG_MODE, MFG_MAX))
                yield env.timeout(mfg_time2)
            # Slot released, will loop to QC again
        else:
            break   # QC passed

    # ── Return transport (deterministic) ──────────────────────────────────────
    qc_end_realized = env.now
    yield env.timeout(tt3)

    completion_realized  = env.now
    total_time_realized  = completion_realized - arrival     # = realized TRT
    wait_at_facility     = mfg_start_realized - gate_time   # pure queueing delay
    lateness_realized    = max(0.0, total_time_realized - deadline)

    rep_out[pid] = {
        'manufacturing_start_realized': mfg_start_realized,
        'mfg_end_realized':             mfg_start_realized + (mfg_time if n_remanufactures == 0
                                                               else mfg_time2),  # last run's time
        'qc_end_realized':              qc_end_realized,
        'completion_realized':          completion_realized,
        'total_time_realized':          total_time_realized,
        'wait_at_facility':             wait_at_facility,
        'lateness_realized':            lateness_realized,
        'on_time_realized':             lateness_realized < 0.01,
        'n_remanufactures':             n_remanufactures,
    }


# ── Single replication ─────────────────────────────────────────────────────────

def _run_one_replication(patient_schedule, fcap_map, seed):
    """Execute one complete simulation replication.

    Returns dict: patient_id -> per-patient result dict.
    """
    rng = np.random.default_rng(seed)
    env = simpy.Environment()

    # One SimPy Resource per facility (capacity = FCAP[m])
    fac_resources = {
        fac: simpy.Resource(env, capacity=max(1, int(cap)))
        for fac, cap in fcap_map.items()
    }

    rep_out = {}
    for info in patient_schedule:
        env.process(_patient_process(env, info, fac_resources, rng, rep_out))

    env.run()   # runs until all processes complete
    return rep_out


# ── Public API ─────────────────────────────────────────────────────────────────

def run_sim(patient_schedule, n_replications=20, base_seed=42):
    """Run multi-replication DES and return averaged per-patient outcomes.

    Parameters
    ----------
    patient_schedule : list[dict]
        One dict per patient.  Required keys:
          patient_id                  – str
          facility                    – str  (e.g. 'm1')
          manufacturing_start_policy  – int  policy-adjusted gate day (abs.)
          arrival                     – int  patient arrival day (abs.)
          deadline                    – float  TRT deadline (days from arrival)
          tt3                         – int  return transport days
          fcap                        – int  facility capacity (used to build fcap_map)

    n_replications : int   number of independent simulation runs
    base_seed      : int   replication i uses seed base_seed + i

    Returns
    -------
    dict with keys:
        per_patient     – list of per-patient averaged dicts (see below)
        facility_util   – {facility: avg_utilisation_fraction}
        kpis            – aggregate KPI dict
        n_replications  – int
    """
    if not patient_schedule:
        return {'per_patient': [], 'facility_util': {}, 'kpis': {}, 'n_replications': 0}

    # Build FCAP map from schedule (each patient carries their facility's capacity)
    fcap_map = {}
    for p in patient_schedule:
        fac = p['facility']
        if fac not in fcap_map:
            fcap_map[fac] = max(1, int(p.get('fcap', 4)))

    # Collect per-replication, per-patient results
    all_rep_data = {p['patient_id']: [] for p in patient_schedule}

    # Track facility busy-time across replications for utilisation
    fac_busy_days = defaultdict(list)   # facility -> [busy_days_per_rep]

    for rep in range(n_replications):
        rep_out = _run_one_replication(patient_schedule, fcap_map, base_seed + rep)

        for pid, r in rep_out.items():
            all_rep_data[pid].append(r)

        # Estimate facility utilisation this replication
        fac_patient_times = defaultdict(list)
        for p in patient_schedule:
            pid = p['patient_id']
            if pid in rep_out:
                fac_patient_times[p['facility']].append(
                    rep_out[pid]['total_time_realized'])

        # Busy fraction ≈ sum(mfg_times) / (FCAP * horizon)
        # Use a simple approximation: total mfg time / (FCAP * span)
        horizon = max((r['completion_realized'] for r in rep_out.values()), default=30)
        for fac, times in fac_patient_times.items():
            cap = fcap_map.get(fac, 1)
            # Each patient occupies ~TMFE days in the facility
            busy = sum(MFG_MODE for _ in times)   # approx using mode
            span = max(horizon, 1)
            fac_busy_days[fac].append(min(1.0, busy / (cap * span)))

    # ── Aggregate across replications ─────────────────────────────────────────
    per_patient = []
    total_lateness    = 0.0
    on_time_sum       = 0.0
    total_remfg       = 0.0
    total_wait        = 0.0

    for p in patient_schedule:
        pid  = p['patient_id']
        reps = all_rep_data.get(pid, [])
        if not reps:
            continue

        def _mean(key):
            return float(np.mean([r[key] for r in reps]))
        def _std(key):
            return float(np.std([r[key] for r in reps]))

        mfg_s_mean   = _mean('manufacturing_start_realized')
        comp_mean    = _mean('completion_realized')
        late_mean    = _mean('lateness_realized')
        wait_mean    = _mean('wait_at_facility')
        remfg_mean   = _mean('n_remanufactures')
        on_time_rate = float(np.mean([r['on_time_realized'] for r in reps]))

        per_patient.append({
            'patient_id':                      pid,
            # ── Three manufacturing start times (the core visibility requirement) ──
            'manufacturing_start_baseline':    p.get('manufacturing_start_baseline'),
            'manufacturing_start_policy':      p['manufacturing_start_policy'],
            'manufacturing_start_realized':    round(mfg_s_mean, 2),
            'manufacturing_start_realized_std': round(_std('manufacturing_start_realized'), 2),
            # ── Other realized timing ────────────────────────────────────────────
            'qc_end_realized':                 round(_mean('qc_end_realized'), 2),
            'completion_realized':             round(comp_mean, 2),
            'completion_realized_std':         round(_std('completion_realized'), 2),
            'wait_at_facility':                round(wait_mean, 2),
            'lateness_realized':               round(late_mean, 2),
            'on_time_rate':                    round(on_time_rate, 3),
            'n_remanufactures':                round(remfg_mean, 3),
            # ── Pass-through from schedule ───────────────────────────────────────
            'policy_adjustment':               p.get('policy_adjustment', 0),
            'adjustment_reason':               p.get('adjustment_reason', 'baseline'),
            'urgency_group':                   p.get('urgency_group', '?'),
            'facility':                        p['facility'],
            'arrival':                         p['arrival'],
            'deadline':                        p['deadline'],
            'deadline_day':                    p.get('deadline_day', p['arrival'] + p['deadline']),
        })

        total_lateness += late_mean
        on_time_sum    += on_time_rate
        total_remfg    += remfg_mean
        total_wait     += wait_mean

    n = len(per_patient)

    # ── Facility utilisation ───────────────────────────────────────────────────
    facility_util = {
        fac: round(float(np.mean(vals)), 3)
        for fac, vals in fac_busy_days.items()
    }

    # ── Aggregate KPIs ─────────────────────────────────────────────────────────
    kpis = {
        'mean_lateness':          round(total_lateness / n, 3) if n else 0,
        'on_time_pct':            round(on_time_sum / n * 100, 1) if n else 0,
        'mean_wait_at_facility':  round(total_wait / n, 3) if n else 0,
        'mean_remanufactures':    round(total_remfg / n, 4) if n else 0,
        'n_replications':         n_replications,
    }

    return {
        'per_patient':    per_patient,
        'facility_util':  facility_util,
        'kpis':           kpis,
        'n_replications': n_replications,
    }
