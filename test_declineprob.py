"""
test_declineprob.py — unit tests for the proportional-hazards decline kernel.

Focus: the gamma=1 memoryless property, gamma>1 acceleration, calibration
endpoint recovery, and the kappa=0 -> beta_u nesting anchor.
Run: python test_declineprob.py
"""

from __future__ import annotations

import math

import numpy as np

from declineprob import (
    h0, calibrate_lambda, decline_prob, extra_survival, still_eligible_prob,
    DeclineKernel, HR_TIER, BETA_ANCHOR, DELAY_ANCHOR_DAYS, HR_PFS_DULOBDAS,
    HR_ANCHOR,
)

TOL = 1e-12


def test_gamma1_memoryless():
    """gamma=1 => constant hazard: decline over a window depends only on its width."""
    lam = calibrate_lambda(1.0)
    d = 7.0
    base = decline_prob(0.0, d, hr=1.5, lam=lam, gamma=1.0)
    for a in (0.0, 5.0, 20.0, 100.0):
        shifted = decline_prob(a, a + d, hr=1.5, lam=lam, gamma=1.0)
        assert abs(float(shifted) - float(base)) < 1e-12, (
            f"gamma=1 not memoryless at a={a}: {shifted} vs {base}")
    print("[PASS] gamma=1 memoryless (window-width invariant)")


def test_gamma_gt1_accelerates():
    """gamma>1 => later windows of equal width carry MORE decline (aging)."""
    lam = calibrate_lambda(2.0)
    d = 7.0
    early = float(decline_prob(0.0, d, hr=1.5, lam=lam, gamma=2.0))
    late = float(decline_prob(20.0, 20.0 + d, hr=1.5, lam=lam, gamma=2.0))
    assert late > early + 1e-6, f"gamma>1 should accelerate: late {late} !> early {early}"
    print(f"[PASS] gamma=2 accelerates (early={early:.4f} < late={late:.4f})")


def test_calibration_recovers_anchor():
    """Mid-tier survival over the 12-day anchor delay equals 1/HR_PFS for every gamma."""
    for gamma in (1.0, 1.5, 2.0):
        lam = calibrate_lambda(gamma)
        surv = math.exp(-HR_ANCHOR * (DELAY_ANCHOR_DAYS / lam) ** gamma)
        assert abs(surv - 1.0 / HR_PFS_DULOBDAS) < 1e-9, (
            f"anchor not recovered at gamma={gamma}: {surv}")
    print(f"[PASS] calibration recovers Dulobdas anchor (1/{HR_PFS_DULOBDAS}="
          f"{1/HR_PFS_DULOBDAS:.4f}) for gamma in {{1.0,1.5,2.0}}")


def test_kappa0_is_v1():
    """kappa=0 => StillEligibleProb == beta_u exactly, any delay/gamma/tier."""
    for gamma in (1.0, 1.5, 2.0):
        lam = calibrate_lambda(gamma)
        for tier, beta in BETA_ANCHOR.items():
            for delay in (0.0, 12.0, 40.0, 200.0):
                p = still_eligible_prob(beta, delay, HR_TIER[tier], lam, gamma, kappa=0.0)
                assert abs(float(p) - beta) < TOL, (
                    f"kappa=0 not beta_u (tier {tier}, delay {delay}): {p} vs {beta}")
    print("[PASS] kappa=0 collapses to beta_u exactly (v1 nesting anchor)")


def test_monotone_in_delay_and_kappa():
    """Eligibility is non-increasing in delay and in kappa (more delay/sensitivity = worse)."""
    lam = calibrate_lambda(1.5)
    beta = BETA_ANCHOR["H"]
    delays = np.array([0.0, 5.0, 12.0, 25.0, 60.0])
    p = still_eligible_prob(beta, delays, HR_TIER["H"], lam, 1.5, kappa=1.0)
    assert np.all(np.diff(p) <= 1e-12), f"not monotone in delay: {p}"
    ks = [0.0, 0.5, 1.0, 2.0]
    pk = [float(still_eligible_prob(beta, 20.0, HR_TIER["H"], lam, 1.5, kappa=k)) for k in ks]
    assert all(pk[j] >= pk[j + 1] - 1e-12 for j in range(len(pk) - 1)), f"not monotone in kappa: {pk}"
    print("[PASS] eligibility monotone decreasing in delay and in kappa")


def test_tier_ordering():
    """At equal delay, higher-hazard tiers lose more eligibility (H worst)."""
    lam = calibrate_lambda(1.0)
    mult = {t: float(extra_survival(20.0, HR_TIER[t], lam, 1.0, 1.0)) for t in "HML"}
    assert mult["H"] < mult["M"] < mult["L"], f"tier hazard ordering violated: {mult}"
    print(f"[PASS] tier decline ordering H<M<L survival ({mult['H']:.3f}<"
          f"{mult['M']:.3f}<{mult['L']:.3f})")


def test_kernel_dataclass():
    """DeclineKernel wires gamma->lambda and reproduces the functional API."""
    k = DeclineKernel(gamma=1.5, kappa=1.0)
    assert abs(k.lam - calibrate_lambda(1.5)) < 1e-12
    direct = float(still_eligible_prob(0.55, 18.0, HR_TIER["H"], k.lam, 1.5, 1.0))
    viacls = float(k.still_eligible_prob(0.55, 18.0, "H"))
    assert abs(direct - viacls) < TOL
    print("[PASS] DeclineKernel matches functional API")


if __name__ == "__main__":
    test_gamma1_memoryless()
    test_gamma_gt1_accelerates()
    test_calibration_recovers_anchor()
    test_kappa0_is_v1()
    test_monotone_in_delay_and_kappa()
    test_tier_ordering()
    test_kernel_dataclass()
    print("\n7 / 7 declineprob tests passed.")
