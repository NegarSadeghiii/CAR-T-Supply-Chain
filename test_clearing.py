"""
test_clearing.py — unit tests for the congestion->delay clearing function.
Checks: baseline at zero congestion, monotonicity, convexity, flat below the
first breakpoint, and validation of malformed parameters.
Run: python test_clearing.py
"""

from __future__ import annotations

import numpy as np

from clearing_function import ClearingFunction, DEFAULT_CLEARING, TAU_PROC_DAYS


def test_baseline_at_zero():
    assert abs(float(DEFAULT_CLEARING.remake_delay(0.0)) - TAU_PROC_DAYS) < 1e-12
    print(f"[PASS] RemakeDelay(0) = tau_proc = {TAU_PROC_DAYS} d")


def test_flat_below_first_breakpoint():
    """With a_1=0, delay is flat (= tau_proc) up to the first breakpoint."""
    cf = DEFAULT_CLEARING
    b1 = cf.breakpoints[0]
    for rho in (0.0, 0.3, b1 - 1e-6, b1):
        assert abs(float(cf.remake_delay(rho)) - cf.tau_proc) < 1e-9, rho
    print(f"[PASS] flat (= tau_proc) for rho <= first breakpoint {cf.breakpoints[0]}")


def test_monotone_increasing():
    cf = DEFAULT_CLEARING
    rho = np.linspace(0.0, 1.6, 200)
    d = cf.remake_delay(rho)
    assert np.all(np.diff(d) >= -1e-12), "delay must be non-decreasing in rho"
    print("[PASS] RemakeDelay monotone non-decreasing in utilization")


def test_convexity():
    """Second differences >= 0 (convex): congestion hurts more as it saturates."""
    cf = DEFAULT_CLEARING
    rho = np.linspace(0.0, 1.6, 400)
    d = cf.remake_delay(rho)
    second = np.diff(d, 2)
    assert np.all(second >= -1e-9), f"not convex: min 2nd diff {second.min()}"
    # marginal slope should be non-decreasing across breakpoints
    slopes = [float(cf.marginal_slope(r)) for r in (0.5, 0.8, 1.0)]
    assert slopes[0] <= slopes[1] <= slopes[2], f"marginal slope not increasing: {slopes}"
    print(f"[PASS] convex; marginal slope increases across breakpoints {slopes}")


def test_saturation_escalates():
    """A fully utilized facility roughly doubles turnaround; overload escalates."""
    cf = DEFAULT_CLEARING
    d1 = float(cf.remake_delay(1.0))
    d12 = float(cf.remake_delay(1.2))
    assert d1 > cf.tau_proc, "rho=1 should exceed baseline"
    assert d12 > d1, "overload (rho>1) should exceed rho=1"
    print(f"[PASS] saturation escalates: delay(1.0)={d1:.1f}d < delay(1.2)={d12:.1f}d")


def test_validation():
    ok = 0
    try:
        ClearingFunction(breakpoints=(0.9, 0.7))          # not increasing
    except ValueError:
        ok += 1
    try:
        ClearingFunction(breakpoints=(0.7,), inc_slopes=(0.0,))  # wrong length
    except ValueError:
        ok += 1
    try:
        ClearingFunction(breakpoints=(0.7, 0.9), inc_slopes=(0.0, 45.0, -1.0))  # non-convex
    except ValueError:
        ok += 1
    assert ok == 3, f"expected 3 validation errors, got {ok}"
    print("[PASS] rejects non-increasing breakpoints, wrong slope count, non-convex slopes")


if __name__ == "__main__":
    test_baseline_at_zero()
    test_flat_below_first_breakpoint()
    test_monotone_increasing()
    test_convexity()
    test_saturation_escalates()
    test_validation()
    print("\n6 / 6 clearing-function tests passed.")
