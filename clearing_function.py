"""
clearing_function.py — piecewise-linear congestion -> delay (clearing) function.

A re-manufacturing job's turnaround grows with facility congestion. We model the
re-make delay as a convex piecewise-linear function of utilization
rho_m = (slots consumed) / Slots[m]:

    RemakeDelay(rho) = tau_proc + sum_s a_s * max(0, rho - b_{s-1}),   b_0 = 0

with breakpoints 0 < b_1 < b_2 < ... and non-negative incremental slopes a_s so
the marginal delay is non-decreasing (convex) — congestion hurts more as the
facility saturates. This is the standard clearing-function shape from
production-planning (Karmarkar 1989; Missbauer & Uzsoy 2011).

Calibration / status
--------------------
  - tau_proc ~ 19 days: baseline re-manufacture processing time
    (~7 d production + ~7 d QC + 1-2 d transport; Avramescu et al. 2022/2023,
    calibration_table.md "Cross-references for cost parameters").
  - Breakpoints (b_s) and slopes (a_s): DECLARED STRUCTURAL ASSUMPTION. There is
    no public CDMO congestion-vs-delay dataset, so these are exposed as
    parameters and are flagged as sensitivity-tested in RESULTS.md. The default
    is tuned so a fully utilized facility (rho=1) roughly doubles turnaround
    (~40 d) and an overloaded facility (rho>1) escalates steeply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Baseline re-manufacture processing time (days). Source: Avramescu 2022/2023.
TAU_PROC_DAYS: float = 19.0


@dataclass(frozen=True)
class ClearingFunction:
    """
    Convex piecewise-linear congestion->delay map.

    Parameters (all sensitivity-sweepable STRUCTURAL ASSUMPTIONS except tau_proc):
      tau_proc     : baseline processing delay at zero congestion (days).
      breakpoints  : increasing utilizations b_1 < b_2 < ... where slope steepens.
      inc_slopes   : incremental slopes (a_1, a_2, ...) with len = len(breakpoints)+1,
                     a_1 for [0, b_1], and each subsequent a_s ADDED above b_{s-1}.
                     Non-negative a_s (s>=2) => convex (non-decreasing marginal delay).
    """
    tau_proc: float = TAU_PROC_DAYS
    breakpoints: tuple = (0.70, 0.90)          # ASSUMPTION (sweepable)
    inc_slopes: tuple = (0.0, 45.0, 90.0)      # ASSUMPTION (sweepable), convex

    def __post_init__(self):
        b = np.asarray(self.breakpoints, float)
        a = np.asarray(self.inc_slopes, float)
        if len(a) != len(b) + 1:
            raise ValueError("inc_slopes must have len(breakpoints)+1 entries")
        if np.any(np.diff(b) <= 0) or np.any(b <= 0):
            raise ValueError("breakpoints must be strictly increasing and > 0")
        if np.any(a[1:] < 0):
            raise ValueError("incremental slopes a_s (s>=2) must be >= 0 for convexity")

    def remake_delay(self, rho):
        """
        RemakeDelay(rho) in days. Accepts scalars or arrays; rho is clipped at 0.
        Convex, non-decreasing, equals tau_proc for rho <= b_1 when a_1 = 0.
        """
        rho = np.asarray(rho, float)
        b_prev = np.concatenate(([0.0], np.asarray(self.breakpoints, float)))
        a = np.asarray(self.inc_slopes, float)
        delay = np.full(rho.shape, self.tau_proc, dtype=float)
        for a_s, b_s in zip(a, b_prev):
            delay = delay + a_s * np.clip(rho - b_s, 0.0, None)
        return delay

    def marginal_slope(self, rho):
        """Left-to-right marginal slope at rho (for convexity checks/plots)."""
        rho = np.asarray(rho, float)
        b_prev = np.concatenate(([0.0], np.asarray(self.breakpoints, float)))
        a = np.asarray(self.inc_slopes, float)
        slope = np.zeros(rho.shape, float)
        for a_s, b_s in zip(a, b_prev):
            slope = slope + a_s * (rho > b_s)
        return slope


# Central-calibration instance used across the dynamic experiments.
DEFAULT_CLEARING = ClearingFunction()


if __name__ == "__main__":
    cf = DEFAULT_CLEARING
    print(f"tau_proc = {cf.tau_proc} d; breakpoints = {cf.breakpoints}; "
          f"inc_slopes = {cf.inc_slopes}")
    for rho in (0.0, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5):
        print(f"  rho={rho:4.2f} -> RemakeDelay = {float(cf.remake_delay(rho)):6.2f} d  "
              f"(marginal {float(cf.marginal_slope(rho)):.0f} d/util)")
