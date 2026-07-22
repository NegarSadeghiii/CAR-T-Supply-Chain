"""
scaled_instance.py — build larger, capacity-tight variants of the case-study
network WITHOUT modifying case_study.py.

Two knobs:
  * mult   — tile the 50-patient cohort `mult` times (preserving the tier mix and
             geography) to raise demand until facilities run near full capacity.
  * s_max  — installed capacity per facility (default 75, the Wan et al. 2026
             Maxcap, matching case_study.py). The tight, two-facility regime is
             reached by scaling demand via `mult`, not by shrinking capacity.

Everything else (costs, yields, tier parameters) is inherited unchanged from
build_case_study_instance().
"""

from __future__ import annotations

import copy

import numpy as np

from case_study import build_case_study_instance, TIERS


def scaled_case_study_instance(mult: int = 1, s_max: int = 75):
    """
    Return (instance, tiers, tier_idx) for a `mult`x-tiled cohort with per-facility
    capacity `s_max`. tiers is a length-n_patients list of "H"/"M"/"L"; tier_idx
    maps each tier to its patient indices.
    """
    base = build_case_study_instance()
    inst = copy.deepcopy(base)
    k = int(mult)
    inst.n_patients = base.n_patients * k
    inst.beta = np.tile(np.asarray(base.beta), k)
    inst.rho_cancel = np.tile(np.asarray(base.rho_cancel), k)
    inst.s_max = np.full(base.n_facilities, int(s_max))

    tiers = list(TIERS) * k
    tier_idx = {t: [i for i, tt in enumerate(tiers) if tt == t] for t in ("H", "M", "L")}
    return inst, tiers, tier_idx


if __name__ == "__main__":
    for mult in (1, 2, 3):
        inst, tiers, tidx = scaled_case_study_instance(mult=mult, s_max=75)
        cap = inst.n_facilities * inst.s_max[0]
        print(f"mult={mult}: {inst.n_patients} patients, s_max={inst.s_max[0]}, "
              f"tier sizes H/M/L = {len(tidx['H'])}/{len(tidx['M'])}/{len(tidx['L'])}, "
              f"all-facility capacity={cap}, demand/cap={inst.n_patients/cap:.0%}, "
              f"demand/2-facility-cap={inst.n_patients/(2*inst.s_max[0]):.0%}")
