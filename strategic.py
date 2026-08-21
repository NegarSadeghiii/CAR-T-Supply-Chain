"""
strategic.py
============

The frozen strategic layer for the survival-aware MPC study.

The strategic MILP (1)-(35) is solved ONCE per demand scale, at
``ALPHA_STRATEGIC = $500K per life``, and its solution freezes -- for every
policy alike --

  * which facilities are open (and therefore the facility cost),
  * the link structure and the assignment m(i),
  * the transport modes of each patient's two legs (taken from the strategic
    solution, NOT forced to the fastest mode j1),
  * FCAP_m, the concurrent-slot capacity the simulation queues against,
  * and, for ``static_survival``, the pre-computed serving order implied by the
    strategic manufacturing-start days INM.

Sharing one frozen network across all five policies is what makes the policy
comparison like-for-like and lets common random numbers line up.  Exp C is the
only experiment that re-solves the strategic model (once per value of life).

``ishipment_survival`` is imported lazily inside the functions that need it so
that importing this module never pulls in Pyomo or a solver licence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cart_data as cd

# The confirmed operating point: alpha is MONETISED at $500K per life and
# enters the STRATEGIC objective only (doc, "Finalized experimental design").
# rho_u stays a triage priority weight, not a life-value multiplier, and the
# per-epoch problem (P3) is never monetised.
ALPHA_STRATEGIC = 500_000.0

# Manufacturing yield by facility (doc, parameter table: "by mode").
YIELD_P = {"m1": 0.85, "m4": 0.85,
           "m2": 0.92, "m5": 0.92,
           "m3": 0.95, "m6": 0.95}


@dataclass
class PatientPlan:
    """Everything the simulation inherits about one patient from the solve."""
    pid: str
    tier: str
    c: str
    h: str
    t0: int                 # first leukapheresis (collection) day
    facility: str           # m(i), frozen
    mode_in: str            # LS -> MS mode, frozen
    mode_out: str           # MS -> hospital mode, frozen
    tt1: int
    tt3: int
    u1: float               # LS -> MS unit transport cost, per attempt
    u3: float               # MS -> hospital unit transport cost, on delivery
    strategic_start: int    # INM start day chosen by the strategic solve
    p_pass: float           # facility yield


@dataclass
class FrozenPlan:
    n: int
    alpha: float
    max_facilities: int
    opened: list
    facility_cost: float
    fcap: dict
    c_material: float                                # $/attempt, base-model data
    c_qc: float                                      # $/attempt, base-model data
    variant: str = "cost_design"                     # cost_design | survival_probe
    nd: int = 18
    patients: dict = field(default_factory=dict)     # pid -> PatientPlan
    static_order: list = field(default_factory=list)  # pids by strategic start
    source: str = ""
    solve: dict = field(default_factory=dict)

    @property
    def by_facility(self) -> dict:
        out = {m: [] for m in self.opened}
        for p in self.patients.values():
            out.setdefault(p.facility, []).append(p.pid)
        return out

    def offered_load(self, tmfe: int = 7) -> float:
        """Arrivals per day divided by sustainable starts per day.

        Capacity is FCAP slots each held for T_MFE days, i.e. sum(FCAP)/T_MFE
        starts per day; demand is the arrival density over the arrival window.
        """
        days = [p.t0 for p in self.patients.values()]
        span = max(days) - min(days) + 1
        cap = sum(self.fcap[m] for m in self.opened) / tmfe
        return (len(self.patients) / span) / cap


def build_frozen_plan(net, inst, kind, alpha=0.0, nd=None, max_fac=None,
                      time_limit=None, use_cache=True, mip_gap=1e-4,
                      variant="cost_design"):
    """Solve one strategic variant and freeze its solution.

    ``kind`` selects the MILP: "baseline" is the cost design (2)-(26);
    "extension" is the survival probe, which adds (27)-(35). ``variant`` is
    recorded on the plan so the operational layer can refuse a probe solve.

    The on-disk cache is keyed by kind, scale, alpha, facility cap and ND, so a
    probe solve and a design solve can never collide.
    """
    import ishipment_survival as ish

    max_fac = max_fac or ish.max_facilities_for(inst.n)
    time_limit = time_limit or {50: 900, 100: 1800, 200: 3600}.get(inst.n, 3600)
    blob = ish.run_design(net, inst, kind, alpha=float(alpha), nd=nd,
                          max_fac=max_fac, time_limit=time_limit,
                          mip_gap=mip_gap, use_cache=use_cache)
    if "summary" not in blob:
        raise RuntimeError(f"strategic solve failed: {blob['solve']}")

    plan = FrozenPlan(
        n=inst.n, alpha=float(alpha), max_facilities=max_fac,
        opened=list(blob["summary"]["opened"]),
        facility_cost=float(blob["summary"]["facility_cost"]),
        fcap={m: int(net.FCAP[m]) for m in net.m},
        c_material=float(net.C_material), c_qc=float(net.CQC),
        variant=variant, nd=int(blob["ND"]),
        source=blob["key"], solve=blob["solve"],
    )
    for r in blob["rows"]:
        m, j_in, j_out = r["facility"], r["mode_in"], r["mode_out"]
        plan.patients[r["pid"]] = PatientPlan(
            pid=r["pid"], tier=r["tier"], c=r["c"], h=r["h"], t0=int(r["t0"]),
            facility=m, mode_in=j_in, mode_out=j_out,
            tt1=int(net.TT1[j_in]), tt3=int(net.TT3[j_out]),
            u1=float(net.U1[(r["c"], m, j_in)]),
            u3=float(net.U3[(m, r["h"], j_out)]),
            strategic_start=int(r["start"]), p_pass=YIELD_P[m],
        )
    # static_survival dispatches in this order and never re-solves.
    plan.static_order = [p.pid for p in sorted(plan.patients.values(),
                                               key=lambda p: (p.strategic_start,
                                                              p.pid))]
    return plan


def frozen_plan(net, inst, alpha=ALPHA_STRATEGIC, **kw):
    """Backwards-compatible alias for the survival-probe plan.

    Retained so that existing callers keep working; new code should use
    ``strategic_cost_design.frozen_plan`` or
    ``strategic_survival_probe.frozen_plan`` explicitly.
    """
    return build_frozen_plan(net, inst, kind="extension", alpha=alpha,
                             variant="survival_probe", **kw)


def load_scale(n, dat=None, **kw):
    """Convenience: build the N-patient instance and freeze its strategic solve."""
    root = os.path.dirname(os.path.abspath(__file__))
    dat = dat or os.path.join(root, "Data200_profileA.dat")
    net = cd.load_network(dat)
    inst = cd.build_instance(dat, mult=n // 50)
    return net, inst, frozen_plan(net, inst, **kw)
