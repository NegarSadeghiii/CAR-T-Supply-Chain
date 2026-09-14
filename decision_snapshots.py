"""
decision_snapshots.py
=====================

Instrumentation for the OPERATIONAL layer: what each policy actually chooses at
a decision epoch, and why.

Nothing here re-implements a policy.  Every choice reported is produced by
calling the shipped selection code itself --

    fifo             ``policies.Fifo.select``       (sort on (t0, pid))
    survival_index   ``per_epoch.index_choice``     (P8)
    adaptive_mpc     ``per_epoch.solve_epoch``      (P1)-(P6) MILP

-- on the SAME simulator state, so the three are directly comparable.  One
policy ACTS (it drives the trajectory); the other two are evaluated
counterfactually on the state that policy produced.  Both selection routines
are pure functions of ``(cands, t, busy, fcap, n_start)``, so evaluating them
off-trajectory changes nothing in the simulation.

For every epoch the script records the epoch state (day, facility, free slots,
the ready queue W_t and each candidate's decision-relevant attributes), the
three choices, and -- when the choices differ -- the value of BOTH choices
under the MPC's own look-ahead objective (P3), which is the only common yardstick
the two survival-aware policies share.

Usage
-----
    python3 decision_snapshots.py --scale 50 --seed 0 --acting adaptive_mpc
    python3 decision_snapshots.py --scale 200 --seed 0 --acting all
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import cart_data as cd
import per_epoch as pe
import policies as pol
import simulation as sim
import strategic as st

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "results", "snapshots")

PROBED = ("fifo", "survival_index", "adaptive_mpc")


# ---------------------------------------------------------------------------
# Per-candidate decision attributes, all read from the shipped code
# ---------------------------------------------------------------------------
def candidate_row(sim_, c, t, cfg):
    """Everything that enters any of the three selection rules, for one patient."""
    r = sim_.rec[c.pid]
    elapsed = c.elapsed_at_delivery(t, cfg.tmfe, cfg.tqc)       # (P1)
    s_now = c.survival_if_started(t, cfg.tmfe, cfg.tqc)         # (P2)
    s_next = c.survival_if_started(t + 1, cfg.tmfe, cfg.tqc)
    s_defer = c.survival_if_started(t + cfg.lookahead, cfg.tmfe, cfg.tqc)
    return {
        "pid": c.pid,
        "tier": c.tier,
        "alpha_w": cd.ALPHA_W[c.tier],
        "w_risk": cd.W_RISK[c.tier],
        "t0": c.t0,                       # first collection -- the survival clock origin
        "age": t - c.t0,                  # a_i, accrued wait at this epoch
        "tt3": c.tt3,                     # frozen MS -> hospital transport time
        "attempt_next": r.attempts + 1,   # 1 = first make, 2+ = remake
        "ready_day": sim_.ready_at[c.pid],
        "hold": t - sim_.ready_at[c.pid],  # days already queued since ready
        "elapsed_at_delivery": elapsed,
        "S_start_today": s_now,
        "S_start_tomorrow": s_next,
        "S_start_t_plus_H": s_defer,
        # (P8) the survival index: alpha_u * [S_i(t) - S_i(t+1)]
        "index_score": cd.ALPHA_W[c.tier] * (s_now - s_next),
        # (P3) the MPC's own coefficient for starting TODAY: alpha_u * [S_i(t) - d_i]
        "mpc_gain_today": cd.ALPHA_W[c.tier] * (s_now - s_defer),
    }


def epoch_objective(cands, t, busy, fcap, cfg, forced_today):
    """Value of (P3)'s look-ahead objective when ``forced_today`` starts today.

    The same MILP ``per_epoch.solve_epoch`` maximises, with today's start set
    pinned to ``forced_today`` and the rest of the window left free.  This is
    how an alternative choice is scored on the MPC's own yardstick.
    """
    from pyomo.environ import (ConcreteModel, Var, Binary, Objective,
                               ConstraintList, maximize, value)
    import ishipment_survival as ish

    taus, gain, free = pe._epoch_data(cands, t, busy, fcap, cfg.tmfe, cfg.tqc,
                                      cfg.lookahead)
    forced = set(forced_today)
    mdl = ConcreteModel(name="epoch objective at a pinned today-set")
    mdl.x = Var([(c.pid, tau) for c in cands for tau in taus], within=Binary)
    mdl.con = ConstraintList()
    for c in cands:
        mdl.con.add(sum(mdl.x[c.pid, tau] for tau in taus) <= 1)
        mdl.con.add(mdl.x[c.pid, taus[0]] == (1 if c.pid in forced else 0))
    for tau, window in pe._capacity_rows(taus, cfg.tmfe):
        mdl.con.add(sum(mdl.x[c.pid, tp] for c in cands for tp in window)
                    <= max(free[tau], 0))
    mdl.OBJ = Objective(expr=sum(gain[c.pid][tau] * mdl.x[c.pid, tau]
                                 for c in cands for tau in taus),
                        sense=maximize)
    out = ish.solve(mdl, time_limit=60, mip_gap=1e-9, solver_pref="highs")
    if out.status not in ("optimal", "feasible"):
        return None
    return sum(gain[c.pid][tau] * value(mdl.x[c.pid, tau])
               for c in cands for tau in taus)


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------
class Probe:
    """Runs ``acting`` and records what all three policies would do each epoch."""

    def __init__(self, acting, score_disagreements=True):
        self.acting_name = acting
        self.acting = pol.build(acting)
        self.name = acting
        self.rules = {n: pol.build(n) for n in PROBED}
        self.epochs = []
        self.score_disagreements = score_disagreements

    def prepare(self, sim_):
        for p in list(self.rules.values()) + [self.acting]:
            if hasattr(p, "prepare"):
                p.prepare(sim_)

    def select(self, sim_, m, t, n_start, cands):
        cfg = sim_.cfg
        choices = {n: list(r.select(sim_, m, t, n_start, cands))
                   for n, r in self.rules.items()}
        rows = sorted((candidate_row(sim_, c, t, cfg) for c in cands),
                      key=lambda r: -r["index_score"])
        free_profile = [sim_.plan.fcap[m] - sim_.busy[m][tau]
                        for tau in range(t, t + cfg.lookahead + 1)]
        rec = {
            "day": t, "facility": m, "fcap": sim_.plan.fcap[m],
            "busy_today": sim_.busy[m][t],
            "free_slots_today": sim_.plan.fcap[m] - sim_.busy[m][t],
            "n_start": n_start,
            "n_waiting": len(cands),
            "free_profile_window": free_profile,
            "candidates": rows,
            "choice": {n: sorted(v) for n, v in choices.items()},
            "acting": self.acting_name,
        }
        same_si_mpc = set(choices["survival_index"]) == set(choices["adaptive_mpc"])
        rec["agree_index_vs_mpc"] = same_si_mpc
        rec["agree_fifo_vs_mpc"] = (set(choices["fifo"])
                                    == set(choices["adaptive_mpc"]))
        if not same_si_mpc and self.score_disagreements:
            busy, fcap = sim_.busy[m], sim_.plan.fcap[m]
            rec["obj_mpc_choice"] = epoch_objective(
                cands, t, busy, fcap, cfg, choices["adaptive_mpc"])
            rec["obj_index_choice"] = epoch_objective(
                cands, t, busy, fcap, cfg, choices["survival_index"])
        self.epochs.append(rec)
        return self.acting.select(sim_, m, t, n_start, cands)



# ---------------------------------------------------------------------------
# Why the two survival-aware rules coincide -- read off the shipped formulas
# ---------------------------------------------------------------------------
def equivalence_report(cfg=None):
    """Compare the ranking (P8) induces with the ranking (P1)-(P6) induces.

    With ``cart_data.survival`` the kernel is exponential,
    ``S_u(e) = rho_u ** e`` with ``rho_u = (1 - w_u) ** (1 / eta)``, so the two
    scores the code computes factor the same way:

        index_score  = alpha_u * [S_u(e) - S_u(e+1)]    = A_i * (1 - rho_u)
        mpc gain(τ)  = alpha_u * [S_u(e+τ-t) - S_u(e+H)]
                     = A_i * (rho_u ** (τ-t) - rho_u ** H)

    with the whole patient dependence collapsed into ONE scalar
    ``A_i = alpha_u * S_u(e_i(t))``.  Exchanging two candidates between today
    and a slot ``k`` days later changes the MPC objective by
    ``A_i (1 - rho_i**k) - A_j (1 - rho_j**k)``, so the MPC ranks on
    ``A_i * m_u(k)`` where ``m_u(k) = 1 - rho_u ** k`` -- the same form as the
    index, which is the special case ``k = 1``.

    Two candidates of the SAME tier therefore rank identically under both rules
    (same ``rho``, so both reduce to ranking on ``A_i``).  Two candidates of
    DIFFERENT tiers can only be ordered differently if ``A_i / A_j`` falls in
    the narrow band between the two thresholds.  This function prints those
    bands and the elapsed time needed to reach them.
    """
    import math
    cfg = cfg or sim.SimConfig()
    rho = {u: (1 - cd.W_RISK[u]) ** (1 / cd.ETA) for u in cd.TIER_ORDER}
    print("rho_u = (1 - w_u) ** (1/eta):",
          {u: round(r, 8) for u, r in rho.items()})
    print("\nm_u(k) = 1 - rho_u**k   (index uses k = 1; the MPC uses the gap "
          "to the next free slot)")
    print("  k  " + "  ".join(f"{u:>12}" for u in cd.TIER_ORDER))
    for k in range(1, cfg.lookahead + 1):
        print(f"{k:>3}  " + "  ".join(f"{1 - rho[u] ** k:12.8f}"
                                      for u in cd.TIER_ORDER))
    print("\nOrder-flip bands (a flip needs A_i/A_j inside the band):")
    for i, j in (("H", "M"), ("H", "L"), ("M", "L")):
        t1 = (1 - rho[j]) / (1 - rho[i])
        tk = (1 - rho[j] ** cfg.lookahead) / (1 - rho[i] ** cfg.lookahead)
        lo, hi = sorted((t1, tk))
        need = lo / (cd.ALPHA_W[i] / cd.ALPHA_W[j])        # required S_i/S_j
        e = math.log(need) / math.log(rho[i] / rho[j])     # at equal elapsed
        print(f"  {i} vs {j}: index {t1:.8f}  mpc(k={cfg.lookahead}) {tk:.8f}"
              f"  width {100 * (hi - lo) / lo:.3f} %"
              f"  -> needs elapsed ~{e:.0f} d")
    print(f"\nThe simulation's own drain cap is {cfg.drain_cap} d and the S_min "
          f"gate removes patients long before that, so no realisable state "
          f"reaches those bands: on this calibration (P8) and (P1)-(P6) induce "
          f"the SAME ranking, and can differ only by breaking exact ties.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def rescale_load(plan, load, tmfe=7):
    """Exp B's load lever: compress the arrival window onto the fixed network.

    Same arithmetic as ``run_experiments.span_for_load`` -- capacity is
    sum(FCAP)/T_MFE starts per day, so the span that realises ``load`` is
    n / (load * capacity).  Patients keep their tier, facility and modes.
    """
    import copy
    cap = sum(plan.fcap[m] for m in plan.opened) / tmfe
    span = max(2, int(round(plan.n / (load * cap))))
    out = copy.deepcopy(plan)
    days = sorted({p.t0 for p in plan.patients.values()})
    lo, hi = min(days), max(days)
    for p in out.patients.values():
        frac = (p.t0 - lo) / (hi - lo) if hi > lo else 0.0
        p.t0 = 1 + int(round(frac * (span - 1)))
    return out



def run(scale, seed, acting, cfg=None, plan=None):
    if plan is None:
        _, _, plan = st.load_scale(scale)
    cfg = cfg or sim.SimConfig(epoch_solver="highs")

    probe = Probe(acting)
    res = sim.simulate(plan, probe, seed=seed, cfg=cfg)
    return plan, res, probe


def summarise(probe):
    eps = probe.epochs
    contested = [e for e in eps if e["n_waiting"] > e["n_start"]]
    dis_si = [e for e in contested if not e["agree_index_vs_mpc"]]
    dis_fifo = [e for e in contested if not e["agree_fifo_vs_mpc"]]
    ties = []
    for e in dis_si:
        a, b = e.get("obj_mpc_choice"), e.get("obj_index_choice")
        if a is not None and b is not None:
            ties.append(a - b)
    return {
        "acting": probe.acting_name,
        "epochs_total": len(eps),
        "epochs_contested": len(contested),
        "epochs_forced": len(eps) - len(contested),
        "index_vs_mpc_disagree": len(dis_si),
        "index_vs_mpc_disagree_share": (len(dis_si) / len(contested)
                                        if contested else 0.0),
        "fifo_vs_mpc_disagree": len(dis_fifo),
        "fifo_vs_mpc_disagree_share": (len(dis_fifo) / len(contested)
                                       if contested else 0.0),
        "max_objective_gap_mpc_minus_index": max(ties) if ties else 0.0,
        "mean_objective_gap_mpc_minus_index": (sum(ties) / len(ties)
                                               if ties else 0.0),
        "disagreements_with_zero_objective_gap": sum(1 for g in ties
                                                     if abs(g) < 1e-9),
    }


def write_outputs(scale, seed, probe, res, tag=""):
    os.makedirs(OUTDIR, exist_ok=True)
    stem = f"N{scale}_seed{seed}_{probe.acting_name}{tag}"
    with open(os.path.join(OUTDIR, f"epochs_{stem}.json"), "w") as f:
        json.dump({"acting": probe.acting_name, "scale": scale, "seed": seed,
                   "summary": summarise(probe),
                   "metrics": {k: v for k, v in res.metrics.items()
                               if isinstance(v, (int, float))},
                   "epochs": probe.epochs}, f, indent=1)
    rows = []
    for e in probe.epochs:
        rows.append({
            "day": e["day"], "facility": e["facility"],
            "free_slots": e["free_slots_today"], "n_start": e["n_start"],
            "n_waiting": e["n_waiting"],
            "contested": e["n_waiting"] > e["n_start"],
            "fifo": "|".join(e["choice"]["fifo"]),
            "survival_index": "|".join(e["choice"]["survival_index"]),
            "adaptive_mpc": "|".join(e["choice"]["adaptive_mpc"]),
            "agree_index_vs_mpc": e["agree_index_vs_mpc"],
            "agree_fifo_vs_mpc": e["agree_fifo_vs_mpc"],
            "obj_mpc_choice": e.get("obj_mpc_choice"),
            "obj_index_choice": e.get("obj_index_choice"),
        })
    with open(os.path.join(OUTDIR, f"epochs_{stem}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return stem


def print_snapshot(e, top=8):
    kind = ("contested" if e["n_waiting"] > e["n_start"]
            else "forced: every waiting patient starts")
    print(f"\n--- day {e['day']}  facility {e['facility']}  "
          f"FCAP {e['fcap']}  busy {e['busy_today']}  "
          f"free slots {e['free_slots_today']}  ->  n_start {e['n_start']}  "
          f"| waiting |W_t| = {e['n_waiting']}  ({kind})")
    print(f"    free slots over the MPC window [t, t+H]: {e['free_profile_window']}")
    hdr = (f"    {'pid':>8} {'tier':>4} {'t0':>4} {'age':>4} {'tt3':>3} "
           f"{'att':>3} {'hold':>4} {'elapsed':>7} {'S(start today)':>14} "
           f"{'index score':>12} {'mpc gain':>10}")
    print(hdr)
    for r in e["candidates"][:top]:
        print(f"    {r['pid']:>8} {r['tier']:>4} {r['t0']:>4} {r['age']:>4} "
              f"{r['tt3']:>3} {r['attempt_next']:>3} {r['hold']:>4} "
              f"{r['elapsed_at_delivery']:>7} {r['S_start_today']:>14.6f} "
              f"{r['index_score']:>12.6f} {r['mpc_gain_today']:>10.6f}")
    if len(e["candidates"]) > top:
        print(f"    ... {len(e['candidates']) - top} further waiting patients")
    for n in PROBED:
        print(f"    {n:>16s} -> {', '.join(e['choice'][n])}")
    if "obj_mpc_choice" in e:
        print(f"    (P3) objective: mpc {e['obj_mpc_choice']:.8f}  "
              f"index {e['obj_index_choice']:.8f}  "
              f"gap {e['obj_mpc_choice'] - e['obj_index_choice']:+.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--acting", default="adaptive_mpc",
                    choices=list(PROBED) + ["all"])
    ap.add_argument("--show", type=int, default=3,
                    help="how many contested epochs to print in full")
    ap.add_argument("--fail-rate", type=float, default=None,
                    help="Exp E override: common batch failure rate 1 - p")
    ap.add_argument("--lookahead", type=int, default=None,
                    help="MPC window H (default per_epoch.LOOKAHEAD_H = 7)")
    ap.add_argument("--load", type=float, default=None,
                    help="Exp B override: compress arrivals to this offered load")
    ap.add_argument("--seeds", type=int, default=1,
                    help="sweep seeds seed .. seed + seeds - 1")
    ap.add_argument("--theory", action="store_true",
                    help="print the (P8) vs (P1)-(P6) ranking comparison and exit")
    args = ap.parse_args(argv)

    if args.theory:
        equivalence_report()
        return 0

    _, _, plan = st.load_scale(args.scale)
    if args.load is not None:
        plan = rescale_load(plan, args.load)
    cfg = sim.SimConfig(epoch_solver="highs")
    if args.fail_rate is not None:
        cfg.fail_rate = args.fail_rate
    if args.lookahead is not None:
        cfg.lookahead = args.lookahead
    tag = ""
    if args.fail_rate is not None:
        tag += f"_q{args.fail_rate:g}"
    if args.lookahead is not None:
        tag += f"_H{args.lookahead}"
    if args.load is not None:
        tag += f"_load{args.load:g}"
    print(f"frozen plan: N={plan.n} opened={'+'.join(plan.opened)} "
          f"FCAP={ {m: plan.fcap[m] for m in plan.opened} } "
          f"offered load={plan.offered_load():.2f}  "
          f"fail_rate={cfg.fail_rate}  H={cfg.lookahead}")

    acts = list(PROBED) if args.acting == "all" else [args.acting]
    summaries = []
    for acting in acts:
      for seed in range(args.seed, args.seed + args.seeds):
        plan_, res, probe = run(args.scale, seed, acting, cfg=cfg, plan=plan)
        stem = write_outputs(args.scale, seed, probe, res, tag=tag)
        s = summarise(probe)
        s["seed"] = seed
        s["fail_rate"] = cfg.fail_rate
        s["lookahead"] = cfg.lookahead
        s["offered_load"] = plan.offered_load()
        summaries.append(s)
        print(f"\n=== acting policy: {acting} seed {seed}  ({stem}) ===")
        print(f"  epochs {s['epochs_total']}  contested {s['epochs_contested']}"
              f"  forced {s['epochs_forced']}")
        print(f"  survival_index vs adaptive_mpc disagree on "
              f"{s['index_vs_mpc_disagree']} / {s['epochs_contested']} contested "
              f"({100 * s['index_vs_mpc_disagree_share']:.1f} %)")
        print(f"  fifo vs adaptive_mpc disagree on {s['fifo_vs_mpc_disagree']} "
              f"/ {s['epochs_contested']} contested "
              f"({100 * s['fifo_vs_mpc_disagree_share']:.1f} %)")
        print(f"  (P3) objective gap on disagreements: "
              f"mean {s['mean_objective_gap_mpc_minus_index']:+.3e}  "
              f"max {s['max_objective_gap_mpc_minus_index']:+.3e}  "
              f"exact ties {s['disagreements_with_zero_objective_gap']}")
        contested = [e for e in probe.epochs if e["n_waiting"] > e["n_start"]]
        dis = [e for e in contested if not e["agree_index_vs_mpc"]]
        for e in dis[:args.show]:
            print_snapshot(e)
        for e in contested[:args.show]:
            print_snapshot(e)
    with open(os.path.join(OUTDIR,
                           f"summary_N{args.scale}_seed{args.seed}{tag}.json"),
              "w") as f:
        json.dump(summaries, f, indent=1)
    tot_c = sum(s["epochs_contested"] for s in summaries)
    tot_d = sum(s["index_vs_mpc_disagree"] for s in summaries)
    tot_t = sum(s["disagreements_with_zero_objective_gap"] for s in summaries)
    print(f"\nOVERALL: contested epochs {tot_c}; survival_index vs adaptive_mpc "
          f"differ on {tot_d} ({100 * tot_d / tot_c if tot_c else 0:.1f} %), "
          f"of which {tot_t} are exact ties in the (P3) objective; "
          f"{tot_d - tot_t} are genuine differences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
