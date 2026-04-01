"""
policy_hooks.py
===============

Policy layer for the CAR-T Opt → Sim → Policy dashboard.

This module sits between the optimizer and the simulation:

    Optimizer baseline plan
         │
         ▼  apply_policy(plan, policy_config, fcap_map)
    Policy-adjusted plan   ← THIS MODULE
         │
         ▼  simulation.dashboard_simulation.run_sim(adjusted_plan)
    Realised outcomes

What a "policy" does here
--------------------------
A policy receives the optimizer's baseline plan (a list of patient dicts, each
containing manufacturing_start_baseline and urgency metadata) and returns an
adjusted plan where `manufacturing_start_policy` may differ from
`manufacturing_start_baseline`.

The simulation then treats `manufacturing_start_policy` as the "release time"
for each patient — the day the sample arrives at the facility gate.

RL integration hook
-------------------
Any of the rule-based policy functions below can be replaced by an RL agent.
The interface is identical:

    adjusted_plan = rl_agent.act(baseline_plan, fcap_map)

where the agent returns the same dict structure with `manufacturing_start_policy`
set according to its learned policy.  The reward signal would be derived from
the simulation KPIs (lateness, utilisation, re-manufactures).

Available policies
------------------
  'baseline'         – no changes (identity pass-through)
  'urgency-shift'    – earlier starts for high-urgency patients
  'congestion-aware' – greedily re-sequence patients to avoid facility overload

Public interface
----------------
    from policy_hooks import apply_policy

    adjusted = apply_policy(
        baseline_plan,                     # list[dict] from optimizer
        policy_name = 'urgency-shift',
        policy_params = {'shift_high': -2},
        fcap_map = {'m1': 4, 'm2': 10},
    )
    # Each dict in adjusted has:
    #   manufacturing_start_policy   – int  (may differ from baseline)
    #   policy_adjustment            – int  (days shifted; <0 = earlier)
    #   adjustment_reason            – str  (human-readable explanation)
"""

from collections import defaultdict
import copy


# ── Registry ───────────────────────────────────────────────────────────────────

_POLICY_REGISTRY = {}

def _register(name):
    """Decorator to register a policy function by name."""
    def decorator(fn):
        _POLICY_REGISTRY[name] = fn
        return fn
    return decorator


# ── Public entry point ────────────────────────────────────────────────────────

def apply_policy(baseline_plan, policy_name='baseline',
                 policy_params=None, fcap_map=None):
    """Apply a named policy to adjust manufacturing start times.

    Parameters
    ----------
    baseline_plan : list[dict]
        Per-patient dicts from the optimizer.  Each must contain:
          patient_id, urgency_group, facility,
          manufacturing_start_baseline, earliest_mfg_start,
          optimizer_wait, arrival, deadline.
    policy_name   : str   one of 'baseline', 'urgency-shift', 'congestion-aware'
    policy_params : dict  policy-specific parameters (see each policy below)
    fcap_map      : dict  {facility_id: int capacity}  (required for congestion-aware)

    Returns
    -------
    list[dict]
        Same structure as baseline_plan, each dict augmented with:
          manufacturing_start_policy   – int  possibly adjusted start day
          policy_adjustment            – int  days shifted (negative = earlier)
          adjustment_reason            – str
    """
    if policy_params is None:
        policy_params = {}
    if fcap_map is None:
        fcap_map = {}

    fn = _POLICY_REGISTRY.get(policy_name)
    if fn is None:
        raise ValueError(
            f"Unknown policy '{policy_name}'. "
            f"Available: {list(_POLICY_REGISTRY.keys())}"
        )

    # Deep-copy so baseline_plan is never mutated
    plan = copy.deepcopy(baseline_plan)
    return fn(plan, policy_params, fcap_map)


# ── Policy: baseline (identity) ───────────────────────────────────────────────

@_register('baseline')
def _policy_baseline(plan, params, fcap_map):
    """No adjustment — passes the optimizer plan through unchanged.

    Useful as a control condition: run sim with this policy to see what
    happens to the optimizer's plan under stochastic process times.
    """
    for p in plan:
        p['manufacturing_start_policy'] = p['manufacturing_start_baseline']
        p['policy_adjustment']          = 0
        p['adjustment_reason']          = 'baseline: no change'
    return plan


# ── Policy: urgency-shift ─────────────────────────────────────────────────────

@_register('urgency-shift')
def _policy_urgency_shift(plan, params, fcap_map):
    """Shift manufacturing start times by urgency group.

    High-urgency patients are moved earlier (negative shift) to reduce their
    deadline risk.  Low-urgency patients may be delayed to free slots.

    Parameters (policy_params)
    --------------------------
    shift_high   : int  days to shift high-urgency patients (default -2, i.e. 2 days earlier)
    shift_medium : int  days to shift medium-urgency patients (default 0)
    shift_low    : int  days to shift low-urgency patients (default +1, i.e. 1 day later)

    Constraint: manufacturing_start_policy >= earliest_mfg_start
    (cannot start before the sample has physically arrived at the facility)

    RL hook: An RL agent replaces the fixed shift_* constants with
    state-dependent actions.  The state would include urgency, deadline
    proximity, and current facility load.
    """
    shift_map = {
        'high':   params.get('shift_high',   -2),
        'medium': params.get('shift_medium',  0),
        'low':    params.get('shift_low',     1),
    }

    for p in plan:
        grp      = p.get('urgency_group', 'medium')
        baseline = p['manufacturing_start_baseline']
        shift    = shift_map.get(grp, 0)

        # Floor: sample cannot arrive at facility before earliest_mfg_start
        # earliest_mfg_start = arrival + TLS + TT1  (wait = 0)
        # = manufacturing_start_baseline - optimizer_wait
        earliest = p.get('earliest_mfg_start', baseline - p.get('optimizer_wait', 0))

        new_start  = max(earliest, baseline + shift)
        actual_adj = new_start - baseline

        p['manufacturing_start_policy'] = new_start
        p['policy_adjustment']          = actual_adj
        p['adjustment_reason'] = (
            f'urgency-shift ({grp}): {actual_adj:+d} days'
            + (' [floor applied]' if new_start != baseline + shift else '')
        )

    return plan


# ── Policy: congestion-aware ──────────────────────────────────────────────────

@_register('congestion-aware')
def _policy_congestion_aware(plan, params, fcap_map):
    """Greedily re-sequence patients to avoid simultaneous facility overload.

    Algorithm
    ---------
    1. Sort patients by urgency (high → medium → low) so higher-priority
       patients are scheduled first.
    2. For each patient (in priority order), find the earliest day >= their
       baseline start such that adding one patient does not exceed FCAP[m].
    3. Lower-priority patients are delayed if needed; higher-priority patients
       may be advanced if a slot is free earlier than their baseline.

    Parameters (policy_params)
    --------------------------
    allow_advance : bool  if True, patients may also be moved earlier when a
                          free slot exists before their baseline day (default True)
    max_delay     : int   maximum days a patient may be delayed (default 5)

    RL hook: An RL agent would replace step 2 with a learned scheduling
    action — given the current facility state (which slots are taken),
    choose the release day for each patient to maximise a reward signal
    (e.g. weighted on-time rate minus re-manufacture cost).
    """
    allow_advance = params.get('allow_advance', True)
    max_delay     = params.get('max_delay', 5)

    urgency_order = {'high': 0, 'medium': 1, 'low': 2}

    # Sort by urgency then by baseline start (earlier first within same urgency)
    sorted_plan = sorted(
        enumerate(plan),
        key=lambda ip: (urgency_order.get(ip[1].get('urgency_group', 'medium'), 1),
                        ip[1]['manufacturing_start_baseline'])
    )

    # facility_day_allocated[facility][day] = count of patients allocated that day
    facility_day_allocated = defaultdict(lambda: defaultdict(int))

    allocation = {}  # original index -> new_start_day

    for orig_idx, p in sorted_plan:
        fac      = p['facility']
        cap      = fcap_map.get(fac, 4)
        baseline = p['manufacturing_start_baseline']
        earliest = p.get('earliest_mfg_start', baseline - p.get('optimizer_wait', 0))

        # Search window: [search_start, baseline + max_delay]
        search_start = earliest if allow_advance else baseline
        search_end   = baseline + max_delay

        chosen_day = None
        # Prefer baseline day; if full, search later; if allow_advance also search earlier
        candidates = []
        if allow_advance and search_start < baseline:
            candidates.extend(range(search_start, baseline))
        candidates.append(baseline)
        candidates.extend(range(baseline + 1, search_end + 1))

        for day in candidates:
            if facility_day_allocated[fac][day] < cap:
                chosen_day = day
                break

        if chosen_day is None:
            chosen_day = baseline + max_delay   # fallback: use max delay

        facility_day_allocated[fac][chosen_day] += 1
        allocation[orig_idx] = chosen_day

    # Apply allocations back to plan (in original order)
    for orig_idx, p in enumerate(plan):
        new_start  = allocation.get(orig_idx, p['manufacturing_start_baseline'])
        actual_adj = new_start - p['manufacturing_start_baseline']

        p['manufacturing_start_policy'] = new_start
        p['policy_adjustment']          = actual_adj

        if actual_adj < 0:
            p['adjustment_reason'] = (
                f'congestion-aware: advanced {abs(actual_adj)}d '
                f'(earlier slot at {p["facility"]})'
            )
        elif actual_adj > 0:
            p['adjustment_reason'] = (
                f'congestion-aware: delayed {actual_adj}d '
                f'(capacity={fcap_map.get(p["facility"], "?")} at {p["facility"]})'
            )
        else:
            p['adjustment_reason'] = f'congestion-aware: no change (slot available)'

    return plan


# ── Utility: list available policies ──────────────────────────────────────────

def list_policies():
    """Return the names and docstrings of all registered policies."""
    return {
        name: (fn.__doc__ or '').strip().split('\n')[0]
        for name, fn in _POLICY_REGISTRY.items()
    }
