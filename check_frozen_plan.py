"""
check_frozen_plan.py
====================

Does re-solving the strategic layer on a full-licence solver change the plan the
operational study is frozen on?

The MPC study freezes one strategic solve per demand scale, at
``strategic.ALPHA_STRATEGIC`` ($500K per life in the reference tier), and every
policy inherits from it: the open facilities, FCAP, the assignment m(i), both
transport modes, and -- for ``static_survival`` -- the serving order implied by
the strategic manufacturing-start days.  If any of that moves, the whole
operational study has to be re-run.  If none of it moves, the new design-layer
numbers can be adopted on their own.

This script answers that question and nothing else.  It

  * reads the COMMITTED solve from ``results/cache`` (solved under HiGHS),
  * re-solves the same model on whatever solver the machine has, at a tighter
    gap, caching under a separate ``_recheck`` key so the committed blob is
    never overwritten,
  * compares the two plans field by field and prints a verdict.

Run it on a machine with a full Gurobi licence::

    python check_frozen_plan.py                  # N = 100 and 200
    python check_frozen_plan.py --scales 200     # just one scale
    python check_frozen_plan.py --mip-gap 1e-6

Nothing here writes to ``results/`` other than the ``_recheck`` cache entries.
"""

from __future__ import annotations

import argparse
import json
import os

import cart_data as cd

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "results", "cache")

# The fields a FrozenPlan/PatientPlan actually inherits from the solve.  Every
# other field (tt1, tt3, u1, u3, p_pass) is a deterministic lookup from these,
# so comparing these is comparing the whole plan.
PATIENT_FIELDS = ("facility", "mode_in", "mode_out", "start")


def plan_from_blob(blob):
    """The plan-relevant projection of a solved design."""
    if "summary" not in blob:
        raise RuntimeError(f"solve did not produce a solution: {blob['solve']}")
    s = blob["summary"]
    rows = {r["pid"]: {f: r[f] for f in PATIENT_FIELDS} for r in blob["rows"]}
    order = [p for p, _ in sorted(rows.items(), key=lambda kv: (kv[1]["start"], kv[0]))]
    return {
        "opened": list(s["opened"]),
        "capacity": s["capacity_opened"],
        "facility_cost": s["facility_cost"],
        "objective": blob["solve"].get("objective"),
        "gap": blob["solve"].get("gap"),
        "wall_s": blob["solve"].get("wall_s"),
        "solver": blob["solve"].get("solver"),
        "patients": rows,
        "static_order": order,
    }


def compare(old, new):
    """Field-by-field diff of two plans.  Returns (verdict, list of lines)."""
    out, same = [], True

    if old["opened"] != new["opened"]:
        same = False
        out.append(f"  opened        : {'+'.join(old['opened'])}  ->  {'+'.join(new['opened'])}")
    else:
        out.append(f"  opened        : {'+'.join(old['opened'])}   (unchanged)")

    for key, label in (("capacity", "capacity"), ("facility_cost", "facility cost")):
        if abs(float(old[key]) - float(new[key])) > 1e-6:
            same = False
            out.append(f"  {label:<14}: {old[key]:,}  ->  {new[key]:,}")
        else:
            out.append(f"  {label:<14}: {old[key]:,}   (unchanged)")

    missing = set(old["patients"]) ^ set(new["patients"])
    if missing:
        same = False
        out.append(f"  !! patient sets differ on {len(missing)} ids")

    n = 0
    for f in PATIENT_FIELDS:
        d = [p for p in old["patients"] if p in new["patients"]
             and old["patients"][p][f] != new["patients"][p][f]]
        n = max(n, len(d))
        if d:
            same = False
            ex = ", ".join(f"{p}: {old['patients'][p][f]}->{new['patients'][p][f]}"
                           for p in d[:3])
            out.append(f"  {f:<14}: {len(d)}/{len(old['patients'])} patients differ   [{ex}...]")
        else:
            out.append(f"  {f:<14}: all {len(old['patients'])} patients identical")

    if old["static_order"] != new["static_order"]:
        same = False
        moved = sum(1 for a, b in zip(old["static_order"], new["static_order"]) if a != b)
        out.append(f"  static_order  : DIFFERS ({moved} positions changed)"
                   f"   -> static_survival would dispatch differently")
    else:
        out.append("  static_order  : identical   -> static_survival unaffected")

    return same, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--mip-gap", type=float, default=1e-6)
    ap.add_argument("--time-limit", type=int, default=3600)
    ap.add_argument("--dat", default="Data200_profileA.dat")
    args = ap.parse_args(argv)

    import ishipment_survival as ish
    import strategic as st

    net = cd.load_network(os.path.join(ROOT, args.dat))
    verdicts = {}

    for n in args.scales:
        alpha = st.ALPHA_STRATEGIC
        max_fac = ish.max_facilities_for(n)
        key = f"extension_n{n}_alpha{alpha:g}_fac{max_fac}_nd{ish.ND_EXT}"
        committed = os.path.join(CACHE, key + ".json")

        print("=" * 78)
        print(f"N = {n}   alpha_ref = ${alpha:,.0f}   CON1 <= {max_fac}")
        print("=" * 78)
        if not os.path.exists(committed):
            print(f"  no committed solve at {key} -- nothing to compare against\n")
            continue

        old = plan_from_blob(json.load(open(committed)))
        print(f"  committed : obj={old['objective']:,.2f}  gap={old['gap']:.2e}  "
              f"wall={old['wall_s']:.0f}s  solver={old['solver']}")

        inst = cd.build_instance(os.path.join(ROOT, args.dat), mult=n // 50)
        blob = ish.run_design(net, inst, "extension", alpha=float(alpha),
                              max_fac=max_fac, time_limit=args.time_limit,
                              mip_gap=args.mip_gap, use_cache=True, tag="recheck")
        new = plan_from_blob(blob)
        print(f"  re-solved : obj={new['objective']:,.2f}  gap={new['gap']:.2e}  "
              f"wall={new['wall_s']:.0f}s  solver={new['solver']}")

        d = old["objective"] - new["objective"]
        rel = abs(d) / max(1.0, abs(old["objective"]))
        print(f"  objective : committed - re-solved = {d:,.2f}  ({rel:.2e} relative)"
              + ("   committed was NOT optimal" if d > 1.0 else ""))
        print()

        same, lines = compare(old, new)
        for ln in lines:
            print(ln)
        verdicts[n] = same
        print(f"\n  VERDICT: plan is {'IDENTICAL' if same else 'DIFFERENT'}\n")

    print("=" * 78)
    if not verdicts:
        print("nothing compared")
    elif all(verdicts.values()):
        print("ALL PLANS IDENTICAL")
        print("  The operational study's inputs are unchanged. Adopt the new")
        print("  design-layer numbers; the MPC study does NOT need re-running.")
    else:
        moved = [n for n, s in verdicts.items() if not s]
        print(f"PLAN CHANGED AT N = {', '.join(map(str, moved))}")
        print("  The operational study is frozen on a plan that no longer matches")
        print("  the optimal design. The MPC study DOES need re-running at those")
        print("  scales, along with every figure derived from it.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
