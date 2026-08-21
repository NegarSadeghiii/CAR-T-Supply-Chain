"""
strategic_survival_probe.py
===========================

The design-sensitivity PROBE, used only for Exp C.

Objective:   the cost design's objective plus alpha * sum_p rho_u(p) (1 - S_p)
Constraints: the cost design plus the survival extension (27)-(31) and the
             manufacturing queue (32)-(35), with ND relaxed to 42 days.

This variant exists for one purpose: to sweep the value of a life and locate
the valuation at which the cost-optimal network changes. It never supplies the
network for the main results -- that is always ``strategic_cost_design`` -- and
alpha and rho are combined nowhere else in the study.

The MILP is ``ishipment_survival.build_extension``. Plans produced here are
tagged ``variant = "survival_probe"`` so that a probe solve cannot be used as a
design solve by accident.

``ishipment_survival`` is imported lazily so that importing this module needs no
solver licence.
"""

from __future__ import annotations

import os

import cart_data as cd
import strategic as st

# ND relaxed from 18 to 42 days -- part of the survival extension (31).
ND_PROBE = 42

# The operating anchor: the therapy's own worth, ~$500K.
ALPHA_OPERATING = 500_000.0

VARIANT = "survival_probe"


def frozen_plan(net, inst, alpha=ALPHA_OPERATING, nd=ND_PROBE, max_fac=None,
                time_limit=None, use_cache=True, mip_gap=1e-4):
    """Solve (or re-use) the survival probe at one value of a life."""
    return st.build_frozen_plan(net, inst, kind="extension", alpha=float(alpha),
                                nd=nd, max_fac=max_fac, time_limit=time_limit,
                                use_cache=use_cache, mip_gap=mip_gap,
                                variant=VARIANT)


def load_scale(n, dat=None, **kw):
    """Build the N-patient instance and solve the probe on it."""
    root = os.path.dirname(os.path.abspath(__file__))
    dat = dat or os.path.join(root, "Data200_profileA.dat")
    net = cd.load_network(dat)
    inst = cd.build_instance(dat, mult=n // 50)
    return net, inst, frozen_plan(net, inst, **kw)
