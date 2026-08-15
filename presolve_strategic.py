"""
presolve_strategic.py
=====================

Pre-solve (and cache) every strategic design the experiments need:

  * the frozen operating network at N = 100 and N = 200, alpha = $500K --
    one solve per scale, shared by all five policies;
  * the Exp C value-of-life sweep at N = 200 -- the ONLY experiment that
    re-solves the strategic model, once per value of a life.

Results land in ``results/cache``; ``run_experiments.py`` re-uses them.
"""

from __future__ import annotations

import os
import sys

import cart_data as cd
import strategic as st

ROOT = os.path.dirname(os.path.abspath(__file__))
VALUE_OF_LIFE = [0.0, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6]


def main(argv=None):
    dat = os.path.join(ROOT, "Data200_profileA.dat")
    net = cd.load_network(dat)
    jobs = []
    for n in (100, 200):
        jobs.append((n, st.ALPHA_STRATEGIC))
    for a in VALUE_OF_LIFE:                      # Exp C, primary scale only
        if a != st.ALPHA_STRATEGIC:
            jobs.append((200, a))

    for n, alpha in jobs:
        inst = cd.build_instance(dat, mult=n // 50)
        print(f"=== N={n} alpha={alpha:,.0f}", flush=True)
        try:
            plan = st.frozen_plan(net, inst, alpha=alpha,
                                  time_limit=2400 if n == 200 else 1200)
            print(f"    opened={'+'.join(plan.opened)} "
                  f"load={plan.offered_load():.3f} "
                  f"cost=${plan.facility_cost:,.0f}", flush=True)
        except Exception as exc:                 # noqa: BLE001
            print(f"    FAILED: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
