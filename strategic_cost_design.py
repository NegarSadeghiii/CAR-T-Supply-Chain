"""
strategic_cost_design.py
========================

The CORE strategic model: cost-based network design.

Objective:   min Z = sum_p CTM_p + sum_p TTC_p + (C_mat + C_QC)|P|
Constraints: (2)-(26) only.

It contains **no clinical-loss term, no survival extension (27)-(31) and no
manufacturing queue (32)-(35)**.  Manufacturing therefore starts on the day
material arrives, exactly as in the base model, and the question of *who waits*
does not arise here at all -- it is decided operationally, per epoch, by the
policy.

This variant produces the network on which every main result runs: the open
facilities, their capacity, the patient-to-facility assignment m(i) and the
transport modes.

The MILP itself is ``ishipment_survival.build_baseline``, which is already a
verified restatement of the base cost model; this module only names it, freezes
its solution into the structure the operational layer consumes, and records the
provenance so that a probe solve can never be mistaken for a design solve.

``ishipment_survival`` is imported lazily so that importing this module needs no
solver licence.
"""

from __future__ import annotations

import os

import cart_data as cd
import strategic as st

# The turnaround cap of the base cost model.  The relaxation to 42 days is part
# of the survival extension (31) and therefore belongs to the probe, not here.
ND_COST_DESIGN = 18

VARIANT = "cost_design"


def frozen_plan(net, inst, nd=ND_COST_DESIGN, max_fac=None, time_limit=None,
                use_cache=True, mip_gap=1e-4):
    """Solve (or re-use) the cost design and freeze it for the operational layer.

    The returned plan is tagged ``variant = "cost_design"``; the operational
    layer refuses any plan that is not.
    """
    return st.build_frozen_plan(net, inst, kind="baseline", alpha=0.0, nd=nd,
                                max_fac=max_fac, time_limit=time_limit,
                                use_cache=use_cache, mip_gap=mip_gap,
                                variant=VARIANT)


def load_scale(n, dat=None, **kw):
    """Build the N-patient instance and freeze its cost-design network."""
    root = os.path.dirname(os.path.abspath(__file__))
    dat = dat or os.path.join(root, "Data200_profileA.dat")
    net = cd.load_network(dat)
    inst = cd.build_instance(dat, mult=n // 50)
    return net, inst, frozen_plan(net, inst, **kw)
