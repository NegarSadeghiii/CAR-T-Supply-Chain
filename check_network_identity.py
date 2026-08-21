"""
check_network_identity.py
=========================

Validation gate for the strategic split.

The claim under test: because the design is survival-insensitive across the
operationally relevant range, the network produced by ``strategic_cost_design``
should be IDENTICAL to the alpha = $500K network the main results currently use
-- same open facilities, same capacity, same patient-to-facility assignment,
same transport modes.

If the two differ, the difference must be attributed before anything is re-run,
so this script also solves two intermediate variants that separate the possible
causes:

  cost design, ND = 18   the cost model as specified (no queue, base deadline)
  cost design, ND = 42   isolates the effect of relaxing the deadline alone
  probe, alpha = 0       isolates the effect of adding the queue at zero alpha
  probe, alpha = $500K   the network the main results use

Reading across those four rows says whether any difference is caused by the
deadline, by the queue, or genuinely by survival.
"""

from __future__ import annotations

import os
import sys

import cart_data as cd
import strategic as st


def describe(plan):
    return {
        "opened": "+".join(plan.opened),
        "capacity": sum(plan.fcap[m] for m in plan.opened),
        "assign": {pid: p.facility for pid, p in plan.patients.items()},
        "mode_in": {pid: p.mode_in for pid, p in plan.patients.items()},
        "mode_out": {pid: p.mode_out for pid, p in plan.patients.items()},
    }


def diff(a, b, label_a, label_b):
    """Report every dimension on which two frozen networks disagree."""
    out = []
    if a["opened"] != b["opened"]:
        out.append(f"opened: {label_a}={a['opened']}  {label_b}={b['opened']}")
    if a["capacity"] != b["capacity"]:
        out.append(f"capacity: {label_a}={a['capacity']}  {label_b}={b['capacity']}")
    for key in ("assign", "mode_in", "mode_out"):
        n_diff = sum(1 for pid in a[key] if a[key][pid] != b[key].get(pid))
        if n_diff:
            out.append(f"{key}: {n_diff}/{len(a[key])} patients differ")
    return out


def main(scales=(200, 100)):
    root = os.path.dirname(os.path.abspath(__file__))
    dat = os.path.join(root, "Data200_profileA.dat")
    net = cd.load_network(dat)

    for n in scales:
        inst = cd.build_instance(dat, mult=n // 50)
        print(f"\n{'='*72}\nN = {n}\n{'='*72}", flush=True)

        variants = [
            ("cost design, ND=18", dict(kind="baseline", alpha=0.0, nd=18,
                                        variant="cost_design")),
            ("cost design, ND=42", dict(kind="baseline", alpha=0.0, nd=42,
                                        variant="cost_design")),
            ("probe, alpha=0", dict(kind="extension", alpha=0.0, nd=42,
                                    variant="survival_probe")),
            ("probe, alpha=$500K", dict(kind="extension", alpha=500_000.0, nd=42,
                                        variant="survival_probe")),
        ]
        got = {}
        for label, kw in variants:
            plan = st.build_frozen_plan(net, inst, **kw)
            got[label] = describe(plan)
            print(f"  {label:22s} opened={got[label]['opened']:9s} "
                  f"capacity={got[label]['capacity']:3d} slots", flush=True)

        ref = "probe, alpha=$500K"
        print(f"\n  Identity check against the network in current use ({ref}):")
        for label, _ in variants[:-1]:
            d = diff(got[label], got[ref], label, ref)
            verdict = "IDENTICAL" if not d else "DIFFERS"
            print(f"    {label:22s} -> {verdict}")
            for line in d:
                print(f"        {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
