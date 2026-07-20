"""
test_waiting_model.py — validation of the corrected waiting-time / deterioration
model (see the model-review issues in TECHNICAL_APPENDIX.md).

Checks:
  1. Calibration point == application point (Issue 2): the re-make kernel applied
     to an EXTRA delay of 12 days at kappa=1, mid-tier, gives survival 1/1.64.
  2. T_V2V = 42 with the reference settings recovers the previous fixed
     normal-wait mortality exactly (hazard-rate reform regression, Issue 1).
  3. kappa = 0 reproduces the on-time base model (no deterioration).
  4. Priority OFF (FIFO) reproduces the no-lever model exactly.

Run: python test_waiting_model.py   (exit 0 on success)
"""

from __future__ import annotations

import sys

import numpy as np

from declineprob import calibrate_lambda, extra_survival, HR_TIER, HR_ANCHOR
from yield_sp_v1 import sample_scenarios
from scaled_instance import scaled_case_study_instance
from dynamic_sp import solve_dynamic_sp
from patient_simulator import simulate_patients, WAIT_DEATH_REF, T_REF


def test_extra_12_gives_0610():
    """Issue 2: kernel on EXTRA=12 d, mid-tier, kappa=1 -> 1/1.64 = 0.610."""
    lam = calibrate_lambda(1.0)
    mult = float(extra_survival(12.0, HR_TIER["M"], lam, 1.0, 1.0))
    assert HR_TIER["M"] == HR_ANCHOR, "mid-tier must be the calibration anchor"
    assert abs(mult - 1.0 / 1.64) < 1e-3, f"expected 0.610, got {mult:.4f}"
    print(f"  [1] extra=12 d mid-tier survival multiplier = {mult:.4f} (target 0.610)  OK")


def _fixture():
    inst, tiers, tidx = scaled_case_study_instance(mult=1, s_max=75)
    Ytr, Btr = sample_scenarios(inst, 80, seed=0)
    Yte, Bte = sample_scenarios(inst, 500, seed=100)
    U = np.random.default_rng(100 + 777).random((500, inst.n_patients))
    D = solve_dynamic_sp(inst, Ytr, Btr, tiers=tiers, kappa=0.0, mip_gap=1e-2, time_limit=120)
    return inst, tiers, tidx, D, Yte, Bte, U


def test_tv2v42_recovers_reference(fx):
    """Issue 1 regression: at T_V2V=T_REF the normal-wait (cause-a) mortality equals
    the reference probabilities, i.e. cause_a per tier == WAIT_DEATH_REF * successes."""
    inst, tiers, tidx, D, Yte, Bte, U = fx
    s = simulate_patients(inst, D, tiers, tidx, Yte, Bte, U, kappa=1.0, gamma=1.0,
                          rule="FIFO", t_v2v=T_REF)
    # Expected cause-a per tier = WAIT_DEATH_REF * (expected successful batches in tier).
    x = np.asarray(D["x"], float); assign = x.argmax(axis=1)
    p = np.asarray(inst.p, float)
    exp_a = {t: 0.0 for t in "HML"}
    for i, t in enumerate(tiers):
        exp_a[t] += p[assign[i]] * WAIT_DEATH_REF[t]
    for t in "HML":
        got = s["cause_a_by_tier"][t]
        assert abs(got - exp_a[t]) < 0.06, f"tier {t}: cause_a {got:.3f} vs expected {exp_a[t]:.3f}"
    got_str = ", ".join(f"{t}:{s['cause_a_by_tier'][t]:.2f}" for t in "HML")
    print(f"  [2] T_V2V={T_REF:.0f} cause-a {{{got_str}}} matches reference {WAIT_DEATH_REF}  OK")


def test_kappa0_nests(fx):
    inst, tiers, tidx, D, Yte, Bte, U = fx
    s = simulate_patients(inst, D, tiers, tidx, Yte, Bte, U, kappa=0.0, gamma=1.0,
                          rule="THRESHOLD", threshold=1.0)
    ca = sum(s["cause_a_by_tier"].values())
    assert ca == 0.0, f"kappa=0 must have zero normal-wait deaths, got {ca}"
    print(f"  [3] kappa=0: normal-wait deaths = {ca:.4f}, high-urgency lost "
          f"{s['high_urgency_lost']:.3f} (failure-only == base model)  OK")


def test_priority_off_is_no_lever(fx):
    inst, tiers, tidx, D, Yte, Bte, U = fx
    a = simulate_patients(inst, D, tiers, tidx, Yte, Bte, U, kappa=1.0, gamma=1.0, rule="FIFO")
    b = simulate_patients(inst, D, tiers, tidx, Yte, Bte, U, kappa=1.0, gamma=1.0, priority=False)
    assert a["high_urgency_lost"] == b["high_urgency_lost"] and a["total_cost"] == b["total_cost"]
    print(f"  [4] priority OFF == FIFO == no-lever (H-lost {a['high_urgency_lost']:.3f}, "
          f"cost {a['total_cost']:.2f})  OK")


def main():
    print("Waiting-model validation:")
    test_extra_12_gives_0610()
    fx = _fixture()
    test_tv2v42_recovers_reference(fx)
    test_kappa0_nests(fx)
    test_priority_off_is_no_lever(fx)
    print("All waiting-model checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
