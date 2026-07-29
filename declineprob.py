"""
declineprob.py — discretized proportional-hazards eligibility/survival kernel
for delay-driven clinical deterioration in the dynamic CAR-T SP.

The v1 model treats re-collection eligibility B_i(omega) ~ Bernoulli(beta_u) as
INDEPENDENT of time. This module makes eligibility ENDOGENOUS to accrued wait:
a patient who waits longer (because of a re-manufacture and/or congestion delay)
is more likely to have deteriorated past the point of re-collection feasibility.

Model (Weibull proportional hazards, discretized into epochs)
------------------------------------------------------------
Baseline cumulative hazard:      H0(t) = (t / lambda)^gamma
Per-epoch decline probability for a still-eligible tier-u patient over accrued
wait [t_{k-1}, t_k]:
    DeclineProb_u(k) = 1 - exp( -HR_u * [ (t_k/lambda)^gamma - (t_{k-1}/lambda)^gamma ] )

Eligibility is anchored to the v1 Bernoulli(beta_u) at zero *extra* delay and
degraded by any *extra* wait `delay` (days) beyond the nominal primary schedule:
    StillEligibleProb_u(delay) = beta_u * exp( -kappa * HR_u * (delay/lambda)^gamma )
  - kappa = 0  -> StillEligibleProb_u = beta_u exactly (i.e., v1).  <-- nesting knob
  - kappa = 1  -> calibrated delay-sensitivity.
  - kappa > 1  -> stress test.

Calibration (sources match Phase5_Referenced_Calibration_Table / calibration_table.md)
-------------------------------------------------------------------------------------
  - beta_u ordering 0.55 / 0.70 / 0.80 (H/M/L): Roth et al. 2018 (sicker patients
    tolerate delay less well). Used as the baseline eligibility anchor; the case
    study passes its own beta_u via the Instance.
  - HR_u tier ratios 1.6 / 1.5 / 1.4 (H/M/L): ordered around the observed
    1.4-1.6 delayed-manufacture hazard-ratio range (Dulobdas 2025 delayed-vs-
    control PFS analysis; Cohet 2023 remanufacture-delay cohort).
  - lambda (Weibull scale): calibrated so that the observed remanufacture delay
    (~+12 days, Cohet 2023) at the mid-tier hazard ratio reproduces the
    delayed-vs-control PFS hazard ratio 1.64 (Dulobdas 2025):
        HR_anchor * (DELAY_ANCHOR / lambda)^gamma = ln(HR_PFS)
    => lambda = DELAY_ANCHOR / ( ln(HR_PFS) / HR_anchor )^(1/gamma).
  - gamma (Weibull shape): NOT identified from two endpoints. Treated as a SWEPT
    sensitivity parameter (gamma=1 memoryless / constant hazard; gamma>1
    accelerating). We do NOT claim gamma is fitted — see
    TECHNICAL_APPENDIX.md §5 (flagged assumptions) for the sweep.

All numeric constants below carry a source tag; every one is a
sensitivity-tested assumption where flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# --- Calibration anchors (sources in calibration_table.md) -------------------
DELAY_ANCHOR_DAYS: float = 12.0     # observed remanufacture delay, Cohet 2023
HR_PFS_DULOBDAS: float = 1.64       # delayed-vs-control PFS hazard ratio, Dulobdas 2025

# Tier hazard ratios, ordered around the observed 1.4-1.6 delay-HR range.
# ASSUMPTION (sensitivity-tested): the exact spread is not identified; only the
# ordering HR_H > HR_M > HR_L and the ~1.4-1.6 band are data-anchored.
HR_TIER: dict[str, float] = {"H": 1.6, "M": 1.5, "L": 1.4}
HR_ANCHOR: float = HR_TIER["M"]     # mid-tier used for the lambda calibration

# Baseline (nominal-wait) eligibility anchor, Roth et al. 2018 (fallback anchor;
# the case study supplies its own beta_u through the Instance).
BETA_ANCHOR: dict[str, float] = {"H": 0.55, "M": 0.70, "L": 0.80}


def h0(t, lam: float, gamma: float):
    """Weibull baseline cumulative hazard H0(t) = (t/lambda)^gamma (t >= 0)."""
    t = np.asarray(t, dtype=float)
    return np.power(np.clip(t, 0.0, None) / lam, gamma)


def calibrate_lambda(gamma: float,
                     delay_anchor: float = DELAY_ANCHOR_DAYS,
                     hr_pfs: float = HR_PFS_DULOBDAS,
                     hr_anchor: float = HR_ANCHOR) -> float:
    """
    Weibull scale lambda (days) such that the anchor-tier decline over the
    observed remanufacture delay reproduces the Dulobdas PFS hazard ratio:
        hr_anchor * (delay_anchor / lambda)^gamma = ln(hr_pfs).
    lambda depends on gamma, so it is recomputed for every swept gamma.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    target = math.log(hr_pfs) / hr_anchor          # required (delay/lambda)^gamma
    return delay_anchor / (target ** (1.0 / gamma))


def decline_prob(t_prev, t_cur, hr: float, lam: float, gamma: float):
    """
    Per-epoch decline probability over accrued wait [t_prev, t_cur] for a
    still-eligible patient with tier hazard ratio `hr`:
        1 - exp( -hr * [H0(t_cur) - H0(t_prev)] ).
    """
    dH = h0(t_cur, lam, gamma) - h0(t_prev, lam, gamma)
    return 1.0 - np.exp(-hr * np.clip(dH, 0.0, None))


def extra_survival(delay, hr: float, lam: float, gamma: float, kappa: float):
    """
    Survival of the *extra* delay-driven decline over `delay` days (beyond the
    nominal schedule), scaled by the delay-sensitivity knob kappa:
        exp( -kappa * hr * (delay/lambda)^gamma ).
    kappa = 0 -> 1.0 (no extra decline; collapses to v1).
    """
    return np.exp(-kappa * hr * h0(delay, lam, gamma))


def still_eligible_prob(beta_u, delay, hr: float, lam: float,
                        gamma: float, kappa: float):
    """
    Endogenous still-eligible probability:
        StillEligibleProb_u(delay) = beta_u * extra_survival(delay, ...).
    kappa = 0 returns beta_u exactly (v1); larger delay / kappa lowers it.
    """
    return np.asarray(beta_u, float) * extra_survival(delay, hr, lam, gamma, kappa)


@dataclass(frozen=True)
class DeclineKernel:
    """Bundles a calibrated PH kernel for a given gamma and delay-sensitivity kappa."""
    gamma: float = 1.0
    kappa: float = 1.0
    hr_tier: dict = field(default_factory=lambda: dict(HR_TIER))
    lam: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "lam", calibrate_lambda(self.gamma))

    def still_eligible_prob(self, beta_u, delay, tier: str):
        return still_eligible_prob(beta_u, delay, self.hr_tier[tier],
                                   self.lam, self.gamma, self.kappa)

    def survival_multiplier(self, delay, tier: str):
        """extra_survival only (the beta_u-independent delay factor)."""
        return extra_survival(delay, self.hr_tier[tier], self.lam,
                              self.gamma, self.kappa)


if __name__ == "__main__":
    # Quick demonstration of the calibrated scales and endpoint recovery.
    print("Calibrated Weibull scale lambda (days) by gamma:")
    for g in (1.0, 1.5, 2.0):
        lam = calibrate_lambda(g)
        # Recover the anchor: mid-tier survival over 12 d should equal 1/1.64.
        recov = math.exp(-HR_ANCHOR * (DELAY_ANCHOR_DAYS / lam) ** g)
        print(f"  gamma={g}: lambda={lam:6.2f} d   "
              f"mid-tier 12d survival={recov:.4f} (target {1/HR_PFS_DULOBDAS:.4f})")
    print("\nStillEligibleProb at kappa=0 equals beta_u (nesting):")
    for tier, b in BETA_ANCHOR.items():
        p = still_eligible_prob(b, 30.0, HR_TIER[tier], calibrate_lambda(1.0), 1.0, 0.0)
        print(f"  tier {tier}: beta={b} -> {float(p):.4f}")
