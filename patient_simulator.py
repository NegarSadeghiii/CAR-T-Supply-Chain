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

Calibration.
  * Normal-wait (6-week) baseline mortality by urgency — FLAGGED CLINICAL
    ASSUMPTION (BASE_WAIT_DEATH_6WK below): high-urgency r/r disease progresses
    fastest during the manufacturing wait. Values are order-of-magnitude clinical
    estimates, sensitivity-testable.
  * Re-make decline uses the calibrated proportional-hazards kernel
    (declineprob.py): Weibull scale fixed to the Dulobdas 2025 delayed-vs-control
    PFS hazard ratio 1.64 at the Cohet 2023 +12-day re-make delay.

Prioritization lever (`priority`). Each factory processes its patients in some
order. With `priority=False` everyone waits the normal schedule (exposure 1) and
shares any re-make backlog equally. With `priority=True` high-urgency patients go
FIRST — their wait exposure drops toward 0 and the delay shifts onto
lower-urgency patients (exposure up to ~2, average preserved): the same total
waiting, reallocated to protect the sickest. `priority=False` reproduces the
model without the lever exactly.
"""

from __future__ import annotations

import numpy as np

from declineprob import calibrate_lambda, HR_TIER
from clearing_function import DEFAULT_CLEARING
from value_of_endogeneity import _solve_recourse

_URGENCY_RANK = {"H": 0, "M": 1, "L": 2}

# Baseline probability a patient of each urgency is lost during the standard
# ~6-week manufacturing wait (progression/deterioration before a *successful*
# product is ready). FLAGGED CLINICAL ASSUMPTION — sensitivity-tested.
BASE_WAIT_DEATH_6WK = {"H": 0.15, "M": 0.05, "L": 0.02}


def _priority_exposure(assign, tiers, n_f, priority):
    """
    Per-patient wait exposure e_i. Without priority e_i=1 for all. With priority,
    patients at each factory are ordered sickest-first and exposure runs linearly
    0 (front) -> 2 (back), average 1 — high-urgency yield the least waiting, low-
    urgency the most.
    """
    n_p = len(assign)
    e = np.ones(n_p)
    if not priority:
        return e
    for m in range(n_f):
        idx = [i for i in range(n_p) if assign[i] == m]
        if len(idx) <= 1:
            for i in idx:
                e[i] = 0.0
            continue
        idx.sort(key=lambda i: (_URGENCY_RANK[tiers[i]], i))
        n = len(idx)
        for r, i in enumerate(idx):
            e[i] = 2.0 * r / (n - 1)
    return e


def simulate_patients(inst, design, tiers, tier_idx, Y, B, U, *,
                      kappa, gamma, clearing=DEFAULT_CLEARING, priority=False):
    """
    Score a fixed design on out-of-sample scenarios, returning plain patient
    outcomes with each loss split into cause (a) / (b), by urgency tier.
    """
    z = np.asarray(design["z"], float)
    C = np.asarray(design["C"], float)
    x = np.asarray(design["x"], float)
    N, n_p, n_f = Y.shape
    lam = calibrate_lambda(gamma)
    hr = np.array([HR_TIER[t] for t in tiers])
    h_norm = np.array([-np.log(1.0 - BASE_WAIT_DEATH_6WK[t]) for t in tiers])   # baseline hazard
    rho_cancel = np.asarray(inst.rho_cancel, float)
    tau = clearing.tau_proc

    assign = x.argmax(axis=1)
    a_m = np.bincount(assign, minlength=n_f).astype(float)
    Csafe = np.maximum(C, 1e-9)
    Yass = Y[:, np.arange(n_p), assign]
    F = 1.0 - Yass                                             # 1 if batch failed
    e = _priority_exposure(assign, tiers, n_f, priority)       # wait exposure per patient

    still = np.zeros((N, n_p))                                 # re-collection eligibility (failed)
    cause_a = np.zeros((N, n_p))                               # lost on a successful batch
    # Cause (a): normal-wait mortality (design-independent), reallocated by priority.
    surv_norm = np.exp(-kappa * h_norm * e)                    # (n_p,)
    for o in range(N):
        surge = np.zeros(n_f)
        for i in range(n_p):
            if F[o, i] > 0.5:
                surge[assign[i]] += 1.0
        rho = (a_m + surge) / Csafe
        D = np.maximum(0.0, clearing.remake_delay(rho) - tau)  # re-make backlog days per factory
        # Cause (b): failed patient survives re-make (base + its share of backlog).
        wait_b = tau + D[assign] * e                           # priority shortens backlog for sickest
        surv_fail = np.exp(-kappa * hr * (wait_b / lam) ** gamma)
        for i in range(n_p):
            if F[o, i] > 0.5:
                still[o, i] = B[o, i] * (1.0 if U[o, i] < surv_fail[i] else 0.0)
            else:
                if U[o, i] >= surv_norm[i]:
                    cause_a[o, i] = 1.0

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
    }
