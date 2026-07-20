"""
patient_simulator.py — patient-level outcome simulator that (STEP 1) tags every
loss by its cause and (STEP 2) supports a "sickest first" prioritization lever.

For each patient in each simulated scenario we track: which facility made their
cells, whether the batch succeeded or failed, how long they waited, and whether
they survived the wait. Every high-/medium-/low-urgency loss is tagged as:

  (a) LOST DURING THE NORMAL WAIT — the batch succeeded, but the standard
      ~6-week time from collection to treatment outran how long the patient could
      survive. This is a property of the wait itself, NOT of factory crowding:
      more capacity does not shorten the normal schedule, so it cannot fix (a).
      Treating high-urgency patients FASTER (out of turn) is the only lever.
  (b) LOST AFTER A FAILURE — the batch failed and the re-make (plus any backlog
      at a busy factory) pushed the total wait past what the patient could
      survive, or they could no longer undergo a re-collection. Spare capacity
      shortens the backlog and so can reduce (b).

Nesting. Both channels scale with the decline-speed knob and vanish at speed 0,
so the model reduces exactly to the original (loss only from failed batches that
cannot be re-collected).

Waiting-time model (corrected after model review — see TECHNICAL_APPENDIX.md).
  * Normal vein-to-vein wait T_V2V (days) is an EXPLICIT parameter (base 28 d =
    4 weeks; commercial CAR-T is generally 3-5 weeks). Normal-wait mortality is a
    HAZARD RATE `h_norm_u = -ln(1 - WAIT_DEATH_REF_u) / T_REF` applied over the
    modelled wait, so it scales with T_V2V. WAIT_DEATH_REF = {H:0.15,M:0.05,L:0.02}
    over T_REF = 42 d recovers the earlier fixed probabilities at T_V2V = 42.
  * The re-make decline kernel is applied to the EXTRA exposure only (delay ABOVE
    the normal wait), matching how lambda was calibrated (Delta = 12 d -> PFS HR
    1.64). extra_i = DELTA_REMAKE + congestion backlog.
  * Compounding: every patient faces the normal-wait hazard over T_V2V; a FAILED
    patient additionally faces the extra-delay hazard:
        S_success_i = exp(-kappa * h_norm_u * T_V2V * e_i)
        S_failed_i  = S_success_i * exp(-kappa * HR_u * (extra_i / lambda)^gamma)
    (e_i is the priority exposure, mean 1; FIFO -> e_i = 1, which reproduces the
    Issue-3 baseline formulas exactly.)

Prioritization lever. Each factory serves patients in some order. Under FIFO
`e_i = 1` for all (equal sharing) — reproduces the no-lever model. Under a
survival-based rule the sickest are served first: their effective wait exposure
`e_i` falls toward 0 (shorter effective vein-to-vein and less backlog), and
lower-urgency patients absorb it (up to ~2, per-facility mean preserved at 1) —
the same total waiting, reallocated to protect the sickest.
"""

from __future__ import annotations

import numpy as np

from declineprob import calibrate_lambda, HR_TIER
from clearing_function import DEFAULT_CLEARING
from value_of_endogeneity import _solve_recourse

_URGENCY_RANK = {"H": 0, "M": 1, "L": 2}

# Reference normal-wait mortality by urgency over T_REF days (duration-explicit).
# Converted to a hazard rate so it scales with the modelled vein-to-vein time.
# FLAGGED CLINICAL ASSUMPTION — sensitivity-tested (E5).
WAIT_DEATH_REF = {"H": 0.15, "M": 0.05, "L": 0.02}
T_REF = 42.0                 # days the reference probabilities correspond to (6 weeks)
T_V2V_DEFAULT = 28.0         # base normal vein-to-vein wait (4 weeks)
DELTA_REMAKE_DEFAULT = 12.0  # extra days a re-make adds (observed increment, Cohet 2023)

# Backward-compatibility alias (previous name); values now interpreted at T_REF.
BASE_WAIT_DEATH_6WK = WAIT_DEATH_REF

# Survival-rate eligibility cutoff for the priority-rule family. A patient whose
# survival rate at therapy falls below this is "ineligible" and, under a
# Threshold rule, joins the higher-priority group. Value 0.75 follows the
# prioritization framework of Tseng et al. (2024) (SimPAC). Sensitivity-testable.
ELIGIBILITY_CUTOFF = 0.75


def _rule_exposure(assign, signal, n_f, rule, threshold):
    """
    Per-patient wait exposure e_i under a priority RULE (generalizes the earlier
    sickest-first switch). `signal[i]` is the patient's survival rate under FIFO
    (lower = sicker); `rule` is "FIFO" or "THRESHOLD"; `threshold` in (0,1].

    - FIFO: e_i = 1 for all — everyone shares any re-make backlog equally
      (reproduces the no-lever / no-prioritization model exactly).
    - THRESHOLD: within each facility, patients with survival rate < `threshold`
      form the priority group, served lowest-survival-first; they take the front
      (fast) slots with exposure ramping 0 -> 2(k-1)/(n-1), and the remaining
      patients (FIFO) share the leftover waiting uniformly at exposure > 1. Total
      exposure per facility is conserved (mean 1), so the lever reallocates
      waiting, it does not create or remove it. `threshold = 1.0` puts everyone
      in the priority group -> pure lowest-survival-first (== sickest-first).

    Within a facility the survival signal is monotincreasing from H to L, so
    ordering by survival reproduces the tier ordering; `threshold = 1.0` therefore
    reproduces the earlier sickest-first exposure exactly.
    """
    n_p = len(assign)
    e = np.ones(n_p)
    if rule == "FIFO":
        return e
    for m in range(n_f):
        idx = [i for i in range(n_p) if assign[i] == m]
        n = len(idx)
        if n <= 1:
            continue                                    # lone patient: no queue -> e=1
        prio = sorted((i for i in idx if signal[i] < threshold), key=lambda i: (signal[i], i))
        rest = [i for i in idx if signal[i] >= threshold]
        k = len(prio)
        if k == 0:
            continue                                    # nobody prioritized -> FIFO (e=1)
        for r, i in enumerate(prio):
            e[i] = 2.0 * r / (n - 1)                     # front / fast slots
        sum_prio = k * (k - 1) / (n - 1)                # = sum_{r=0}^{k-1} 2r/(n-1)
        if rest:
            e_rest = (n - sum_prio) / len(rest)          # uniform, > 1, conserves mean 1
            for i in rest:
                e[i] = e_rest
    return e


def _fifo_survival_signal(assign, hr, h_norm_rate, t_v2v, p_by_fac, extra_bar,
                          lam, gamma, kappa):
    """
    Per-patient expected survival rate under FIFO (exposure 1), used as the
    priority SIGNAL. Compounds the normal-wait hazard (all patients) with the
    extra-delay hazard (failed patients) at the facility's mean extra delay:
        S_succ  = exp(-kappa h_norm_rate_i T_V2V)
        S_fail  = S_succ * exp(-kappa hr_i (extra_bar_m / lam)^gamma)
        signal  = p_m S_succ + (1-p_m) S_fail.
    Monotone-decreasing from tier L to H within a facility (kappa>0). At kappa=0
    it is 1 for everyone (no deterioration), so no patient is ever prioritized.
    """
    s_succ = np.exp(-kappa * h_norm_rate * t_v2v)
    s_fail = s_succ * np.exp(-kappa * hr * (np.asarray(extra_bar)[assign] / lam) ** gamma)
    pm = np.asarray(p_by_fac)[assign]
    return pm * s_succ + (1.0 - pm) * s_fail


def simulate_patients(inst, design, tiers, tier_idx, Y, B, U, *,
                      kappa, gamma, clearing=DEFAULT_CLEARING, priority=False,
                      rule=None, threshold=1.0, signal=None,
                      t_v2v=T_V2V_DEFAULT, delta_remake=DELTA_REMAKE_DEFAULT,
                      base_wait_death=None, hr_tier=None):
    """
    Score a fixed design on out-of-sample scenarios, returning plain patient
    outcomes with each loss split into cause (a) / (b), by urgency tier.

    Waiting-time model (see module docstring): normal vein-to-vein wait `t_v2v`
    days with a hazard-rate normal-wait mortality; a failed patient additionally
    faces the extra-delay kernel over `extra_i = delta_remake + congestion`, with
    the kernel applied to the EXTRA exposure only (matching lambda's calibration).

    Priority rule (Tseng et al. 2024 framework):
      rule="FIFO"       — no survival-based prioritization (== no-lever model).
      rule="THRESHOLD"  — patients with survival rate < `threshold` are served
                          lowest-survival-first; `threshold=1.0` == sickest-first.
      signal            — per-patient survival rate (lower = sicker) used to order.
    Backward compatibility: `priority=True`  -> rule="THRESHOLD", threshold=1.0;
    `priority=False` -> rule="FIFO". Passing `rule` overrides `priority`.

    Optional overrides (default to module constants; results unchanged unless set):
      t_v2v         : normal vein-to-vein wait (days).
      delta_remake  : extra days a re-make adds, before congestion (Issue 4).
      base_wait_death : dict{H,M,L} reference normal-wait mortality (over T_REF).
      hr_tier         : dict{H,M,L} delay hazard ratios.
    """
    if rule is None:
        rule = "THRESHOLD" if priority else "FIFO"
    bwd = WAIT_DEATH_REF if base_wait_death is None else base_wait_death
    hrt = HR_TIER if hr_tier is None else hr_tier
    z = np.asarray(design["z"], float)
    C = np.asarray(design["C"], float)
    x = np.asarray(design["x"], float)
    N, n_p, n_f = Y.shape
    lam = calibrate_lambda(gamma)
    hr = np.array([hrt[t] for t in tiers])
    # Normal-wait mortality as a HAZARD RATE (per day), from the reference
    # probability over T_REF; cumulative hazard = h_norm_rate * (effective wait).
    h_norm_rate = np.array([-np.log(1.0 - bwd[t]) / T_REF for t in tiers])
    rho_cancel = np.asarray(inst.rho_cancel, float)
    tau = clearing.tau_proc

    assign = x.argmax(axis=1)
    a_m = np.bincount(assign, minlength=n_f).astype(float)
    Csafe = np.maximum(C, 1e-9)
    Yass = Y[:, np.arange(n_p), assign]
    F = 1.0 - Yass                                             # 1 if batch failed

    # Mean congestion backlog per facility (rule-independent) -> priority signal.
    surge_all = np.stack([np.bincount(assign[F[o] > 0.5], minlength=n_f).astype(float)
                          for o in range(N)])                 # (N, n_f)
    Dbar = np.maximum(0.0, clearing.remake_delay((a_m[None, :] + surge_all) / Csafe[None, :])
                      - tau).mean(axis=0)                      # (n_f,) congestion beyond tau
    if signal is None:
        signal = _fifo_survival_signal(assign, hr, h_norm_rate, t_v2v,
                                       np.asarray(inst.p, float), delta_remake + Dbar,
                                       lam, gamma, kappa)
    e = _rule_exposure(assign, np.asarray(signal, float), n_f, rule, threshold)

    still = np.zeros((N, n_p))                                 # re-collection eligibility (failed)
    cause_a = np.zeros((N, n_p))                               # lost on a successful batch
    surv_rate = np.zeros(n_p)                                  # expected survival at therapy
    # Normal-wait survival over the effective vein-to-vein wait T_V2V * e_i
    # (priority shortens the effective wait for the sickest). Faces ALL patients.
    surv_norm = np.exp(-kappa * h_norm_rate * t_v2v * e)       # (n_p,) = S_success
    for o in range(N):
        surge = np.zeros(n_f)
        for i in range(n_p):
            if F[o, i] > 0.5:
                surge[assign[i]] += 1.0
        rho = (a_m + surge) / Csafe
        D = np.maximum(0.0, clearing.remake_delay(rho) - tau)  # congestion backlog per factory
        # Extra exposure for a failed patient: base re-make increment + its share
        # of congestion (priority shortens the backlog share). Kernel on extra only.
        extra = delta_remake + D[assign] * e
        surv_fail = surv_norm * np.exp(-kappa * hr * (extra / lam) ** gamma)  # compound
        for i in range(n_p):
            if F[o, i] > 0.5:
                still[o, i] = B[o, i] * (1.0 if U[o, i] < surv_fail[i] else 0.0)
                surv_rate[i] += surv_fail[i]
            else:
                if U[o, i] >= surv_norm[i]:
                    cause_a[o, i] = 1.0
                surv_rate[i] += surv_norm[i]
    surv_rate /= N                                             # per-patient survival rate at therapy

    # Re-collection / re-manufacture for failed batches (cost-driven -> sickest
    # salvaged first). Cancellations here are cause (b).
    r_cancel, s2 = _solve_recourse(inst, z, C, x, Y, still)

    stage1 = (sum(inst.f[m] * z[m] for m in range(n_f))
              + sum(inst.pi[m] * C[m] for m in range(n_f))
              + sum(inst.c[m] * x[i, m] for i in range(n_p) for m in range(n_f)))
    cause_a_cost = float((cause_a * rho_cancel[None, :]).sum() / N)
    total_cost = float(stage1 + s2 + cause_a_cost)

    sizes = {t: len(tier_idx[t]) for t in ("H", "M", "L")}
    ca = {t: float(cause_a[:, tier_idx[t]].sum() / N) for t in ("H", "M", "L")}
    cb = {t: float(r_cancel[:, tier_idx[t]].sum() / N) for t in ("H", "M", "L")}
    lost = {t: ca[t] + cb[t] for t in ("H", "M", "L")}
    total_lost = sum(lost.values())
    treated = n_p - total_lost
    assigned = x.sum(axis=0)
    return {
        "total_cost": total_cost,
        "expected_stage2_cost": float(s2 + cause_a_cost),
        "lost_by_tier": lost,
        "cause_a_by_tier": ca,          # lost during the normal wait (batch succeeded)
        "cause_b_by_tier": cb,          # lost after a failure
        "total_lost": total_lost,
        "high_urgency_lost": lost["H"],
        "treated_share": treated / n_p,
        "treated_share_by_tier": {t: 1.0 - lost[t] / sizes[t] for t in ("H", "M", "L")},
        "cost_per_treated": total_cost / max(treated, 1e-9),
        "spare_capacity_built": float((C - assigned).clip(0).sum()),
        "facilities_open": int((z > 0.5).sum()),
        "open_facilities": [m for m in range(n_f) if z[m] > 0.5],
        "capacity": C.tolist(),
        "rule": rule,
        "threshold": float(threshold),
        "t_v2v": float(t_v2v),
        "delta_remake": float(delta_remake),
        "survival_rate_by_patient": surv_rate.tolist(),   # expected survival at therapy
        "signal_by_patient": np.asarray(signal, float).tolist(),
    }
