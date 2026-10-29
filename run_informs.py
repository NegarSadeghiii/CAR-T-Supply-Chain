#!/usr/bin/env python3
"""INFORMS runs for the eligibility-aware i-SHIPMENT extension.

    python run_informs.py all          # everything, in order
    python run_informs.py screen       # section 0 only
    python run_informs.py exp1 exp2    # any subset, in the order given

Every stage caches each solve under results_informs/_cache, so an interrupted
run resumes instead of restarting.  The model itself is the notebook's --
nothing here edits a constraint except the policy precedences, which are added
on top of an unchanged model.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import informs_core as ic
import informs_figs as fg

R = ic.RESULTS
os.makedirs(R, exist_ok=True)

# -----------------------------------------------------------------------------
# Section 0 -- candidate curves
# -----------------------------------------------------------------------------
CURVE_SETS = {
    "S1": {"High": dict(k=3.0, eta=30.0), "Low": dict(k=1.5, eta=90.0)},
    "S2": {"High": dict(k=3.0, eta=25.0), "Low": dict(k=1.5, eta=80.0)},
    "S3": {"High": dict(k=3.0, eta=35.0), "Low": dict(k=1.5, eta=70.0)},
}
DEFAULT_SET = "S1"
MIN_GAIN = 0.05


def faster_decline(curves):
    """Experiment 2's second decline case: high risk falls off faster, low
    risk untouched."""
    out = {g: dict(d) for g, d in curves.items()}
    out["High"] = dict(k=3.0, eta=22.0)
    return out


FROZEN = os.path.join(R, "frozen_curves.json")


def frozen_curves():
    if not os.path.exists(FROZEN):
        raise SystemExit("run `python run_informs.py screen` first")
    with open(FROZEN) as f:
        return json.load(f)["curves"]


# -----------------------------------------------------------------------------
# One evaluation, cached
# -----------------------------------------------------------------------------
def evaluate(cfg, policy, cap, tag):
    """Solve one (instance, policy, budget) and return summary + patient rows."""
    key = ic.cache_key("v3", tag, cfg, policy, cap)

    def run():
        inst = ic.nb("build_instance")(cfg)
        m, logs = ic.run_policy(inst, policy, cap)
        if m is None:
            return dict(ok=False, logs=logs, summary=None, patients=None)
        return dict(ok=True, logs=logs, summary=ic.summarise(m),
                    patients=ic.patient_frame(m).to_dict("records"))

    return ic.cached(key, run)


def cost_min_of(cfg, tag):
    key = ic.cache_key("v3", tag, cfg, "cost_min")

    def run():
        inst = ic.nb("build_instance")(cfg)
        m, log = ic.min_cost(inst)
        if m is None:
            return dict(ok=False, log=log, cost=None)
        return dict(ok=True, log=log, cost=float(pyo.value(m.TOTCOST)))

    return ic.cached(key, run)


def log_rows(res, **extra):
    out = []
    for l in (res.get("logs") or [res.get("log")] if res.get("log") else
              res.get("logs") or []):
        out.append({**extra, **l})
    return out


def flatten(res, **extra):
    """One CSV row from one evaluation."""
    logs = res.get("logs") or []
    row = dict(extra)
    if res["ok"]:
        row.update(res["summary"])
    row["status"] = "; ".join(l["status"] for l in logs) or "n/a"
    row["gap"] = max([l["gap"] for l in logs if l["gap"] == l["gap"]] or [np.nan])
    row["runtime"] = sum(l["runtime"] for l in logs)
    row["optimal"] = all(l["optimal"] for l in logs) if logs else False
    row["feasible"] = bool(res["ok"])
    return row


# =============================================================================
# Section 0 -- screen, then freeze
# =============================================================================
def screen():
    t0 = time.time()
    print("== section 0: screening candidate curves ==", flush=True)
    rows = []
    for name, curves in CURVE_SETS.items():
        _, cfg = ic.instance(seed=42, curves=curves)
        cm = cost_min_of(cfg, "screen")
        if not cm["ok"]:
            rows.append(dict(set=name, status="cost-min infeasible"))
            continue
        c_min = cm["cost"]
        cap = c_min * (1 + ic.COST_TOL)
        cs = evaluate(cfg, "cost_speed", cap, "screen")
        em = evaluate(cfg, "elig_max", cap, "screen")
        max_trt = max(cs["summary"]["max_TRT"], em["summary"]["max_TRT"])
        rows.append(dict(
            set=name,
            eta_H=curves["High"]["eta"], k_H=curves["High"]["k"],
            eta_L=curves["Low"]["eta"], k_L=curves["Low"]["k"],
            C_min=c_min,
            ELIG_cost_speed=cs["summary"]["ELIG"],
            E_max=em["summary"]["ELIG"],
            same_cost_gain=em["summary"]["ELIG"] - cs["summary"]["ELIG"],
            inversion_day=ic.inversion_day(curves),
            max_TRT=max_trt,
            inversion_after_max_TRT=ic.inversion_day(curves) > max_trt))
        print(f"  {name}: gain {rows[-1]['same_cost_gain']:+.3f}  "
              f"inversion day {rows[-1]['inversion_day']:g}  "
              f"max TRT {max_trt:g}", flush=True)

    tab = pd.DataFrame(rows)
    ok = tab[tab.inversion_after_max_TRT & (tab.same_cost_gain > MIN_GAIN)]
    if len(ok):
        choice = ok.loc[ok.same_cost_gain.idxmax(), "set"]
        why = (f"largest same-cost gain ({ok.same_cost_gain.max():.3f}) among "
               f"the sets whose inversion day falls after the largest observed "
               f"turnaround time")
    else:
        choice = DEFAULT_SET
        why = (f"no candidate set reached a same-cost gain above "
               f"{MIN_GAIN:.2f} with its inversion day after the largest "
               f"observed turnaround time, so the default set is used anyway")
    curves = CURVE_SETS[choice]
    with open(FROZEN, "w") as f:
        json.dump(dict(set=choice, curves=curves,
                       faster_decline=faster_decline(curves),
                       reason=why, solver=ic.solver_banner()), f, indent=2)
    tab.to_csv(os.path.join(R, "00_curve_screen.csv"), index=False)

    with open(os.path.join(R, "00_curve_choice.md"), "w") as f:
        f.write("# Curve choice\n\n")
        f.write("These Weibull curves are **illustrative showcase values** "
                "chosen to make the scheduling mechanism visible. They are not "
                "clinical estimates and no clinical claim rests on them.\n\n")
        f.write(f"**Frozen set: {choice}** - "
                f"high risk eta={curves['High']['eta']:g}, "
                f"k={curves['High']['k']:g}; "
                f"low risk eta={curves['Low']['eta']:g}, "
                f"k={curves['Low']['k']:g}.\n\n")
        f.write(f"Chosen because it has the {why}.\n\n")
        f.write("Every later run and every figure caption uses this set. "
                "Experiment 2's faster-decline case changes only the high-risk "
                f"scale to eta={faster_decline(curves)['High']['eta']:g}.\n\n")
        f.write("## Screening table (base instance, seed 42)\n\n")
        f.write(ic.md_table(tab.round(4)))
        f.write("\n\n`inversion_day` is the day the one-day eligibility loss "
                "of a high-risk patient falls below that of a low-risk "
                "patient; past it the optimiser gains more by speeding up a "
                "low-risk patient. It must sit after the largest turnaround "
                "time the solutions actually reach, or the showcase would be "
                "demonstrating the opposite of its intent.\n")
    print(f"  frozen: {choice}  ({time.time()-t0:.0f}s)", flush=True)
    return curves


# =============================================================================
# Experiment 1 -- same-cost schedule comparison
# =============================================================================
EXP1_SEEDS = list(range(1, 11))


def exp1():
    curves = frozen_curves()
    print("== experiment 1: same-cost policy comparison ==", flush=True)
    runs, pats, metrics = [], [], []
    for seed in EXP1_SEEDS:
        t0 = time.time()
        _, cfg = ic.instance(seed=seed, curves=curves)
        cm = cost_min_of(cfg, "exp1")
        if not cm["ok"]:
            print(f"  seed {seed}: cost-min infeasible", flush=True)
            continue
        c_min = cm["cost"]
        cap = c_min * (1 + ic.COST_TOL)
        runs.append(dict(seed=seed, policy="cost_min", cost=c_min,
                         status=cm["log"]["status"], gap=cm["log"]["gap"],
                         runtime=cm["log"]["runtime"],
                         optimal=cm["log"]["optimal"], feasible=True,
                         budget="C_min"))

        got = {}
        for pol in ic.POLICIES:
            res = evaluate(cfg, pol, cap, "exp1")
            row = flatten(res, seed=seed, policy=pol, budget="C_min")
            if not res["ok"]:
                # the prompt's fallback: report the policy's own cheapest plan
                res2 = evaluate(cfg, pol, None, "exp1")
                row = flatten(res2, seed=seed, policy=pol,
                              budget="own_min_cost")
                row["note"] = "infeasible at C_min; own cheapest plan reported"
            runs.append(row)
            if row.get("feasible"):
                got[pol] = res if res["ok"] else res2
                src = got[pol]
                for r in src["patients"]:
                    pats.append(dict(seed=seed, policy=pol, **r))

        # extra budgets for the eligibility-aware plan
        paid = evaluate(cfg, "elig_max", c_min * 1.05, "exp1")
        runs.append(flatten(paid, seed=seed, policy="elig_max",
                            budget="1.05*C_min"))
        ceil = evaluate(cfg, "elig_max", None, "exp1")
        runs.append(flatten(ceil, seed=seed, policy="elig_max",
                            budget="no_cap"))

        if "cost_speed" in got and "elig_max" in got:
            ref = pd.DataFrame(got["cost_speed"]["patients"])
            new = pd.DataFrame(got["elig_max"]["patients"])
            e_cs = got["cost_speed"]["summary"]["ELIG"]
            e_mx = got["elig_max"]["summary"]["ELIG"]
            e_mn = (got["elig_min"]["summary"]["ELIG"]
                    if "elig_min" in got else np.nan)
            metrics.append(dict(
                seed=seed, C_min=c_min,
                ELIG_cost_speed=e_cs, E_max=e_mx, E_min=e_mn,
                same_cost_gain=e_mx - e_cs,
                same_cost_range=e_mx - e_mn,
                paid_gain=(paid["summary"]["ELIG"] - e_cs
                           if paid["ok"] else np.nan),
                ceiling_gap=(ceil["summary"]["ELIG"] - e_cs
                             if ceil["ok"] else np.nan),
                ceiling_cost=ceil["summary"]["cost"] if ceil["ok"] else np.nan,
                **ic.lever_counts(ref, new)))
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(R, "exp1_runs.csv"), index=False)
    pd.DataFrame(pats).to_csv(os.path.join(R, "exp1_patients.csv"), index=False)
    met = pd.DataFrame(metrics)
    met.to_csv(os.path.join(R, "exp1_metrics.csv"), index=False)

    pol_runs = runs[runs.policy.isin(ic.POLICIES) &
                    (runs.budget.isin(["C_min", "own_min_cost"]))]
    fg.fig_exp1_dotplot(pol_runs, curves,
                        os.path.join(R, "fig_exp1_dotplot.png"))

    if len(met):
        pick = int(met.loc[(met.same_cost_gain -
                            met.same_cost_gain.median()).abs().idxmin(), "seed"])
        _, cfg = ic.instance(seed=pick, curves=curves)
        inst = ic.nb("build_instance")(cfg)
        c_min = cost_min_of(cfg, "exp1")["cost"]
        cap = c_min * (1 + ic.COST_TOL)
        m_ref, _ = ic.run_policy(inst, "cost_speed", cap)
        m_new, _ = ic.run_policy(inst, "elig_max", cap)
        fg.fig_exp1_swimlane_pair(
            m_ref, m_new, inst, curves,
            os.path.join(R, "fig_exp1_swimlane_pair.png"),
            ic.nb("stages"), seed=pick)
        print(f"  swimlane pair drawn for seed {pick}", flush=True)
    return met


# =============================================================================
# Experiment 2 -- congestion x decline
# =============================================================================
EXP2_SEEDS = list(range(1, 11))


def exp2():
    curves = frozen_curves()
    declines = {"frozen set": curves, "faster decline": faster_decline(curves)}
    print("== experiment 2: congestion x decline ==", flush=True)
    runs, gains = [], []
    for dec_name, dec in declines.items():
        for cong, window in fg.CONGESTION_ORDER:
            for seed in EXP2_SEEDS:
                _, cfg = ic.instance(seed=seed, window=window, curves=dec)
                cm = cost_min_of(cfg, "exp2")
                if not cm["ok"]:
                    runs.append(dict(decline=dec_name, congestion=cong,
                                     window=window, seed=seed,
                                     policy="cost_min", feasible=False,
                                     status=cm["log"]["status"]))
                    continue
                c_min = cm["cost"]
                cap = c_min * (1 + ic.COST_TOL)
                common = dict(decline=dec_name, congestion=cong,
                              window=window, seed=seed)
                cs = evaluate(cfg, "cost_speed", cap, "exp2")
                mx = evaluate(cfg, "elig_max", cap, "exp2")
                pd5 = evaluate(cfg, "elig_max", c_min * 1.05, "exp2")
                runs.append(flatten(cs, policy="cost_speed", budget="C_min",
                                    **common))
                runs.append(flatten(mx, policy="elig_max", budget="C_min",
                                    **common))
                runs.append(flatten(pd5, policy="elig_max",
                                    budget="1.05*C_min", **common))
                if cs["ok"] and mx["ok"]:
                    gains.append(dict(
                        **common, C_min=c_min,
                        ELIG_cost_speed=cs["summary"]["ELIG"],
                        E_max=mx["summary"]["ELIG"],
                        same_cost_gain=(mx["summary"]["ELIG"]
                                        - cs["summary"]["ELIG"]),
                        paid_gain=(pd5["summary"]["ELIG"]
                                   - cs["summary"]["ELIG"]
                                   if pd5["ok"] else np.nan)))
            print(f"  {dec_name} / {cong} ({window} d) done", flush=True)

    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(R, "exp2_runs.csv"), index=False)
    g = pd.DataFrame(gains)
    g.to_csv(os.path.join(R, "exp2_gains.csv"), index=False)
    if len(g):
        fg.fig_exp2_gain(g, curves, os.path.join(R, "fig_exp2_gain.png"),
                         list(declines))
    return g


# =============================================================================
# B1 -- frontier
# =============================================================================
def upper_hull_flags(cost, elig):
    """True for points strictly below the concave envelope (unsupported)."""
    pts = sorted(zip(cost, elig))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop the middle point when it is not above the chord
            if (x2 - x1) * (p[1] - y1) >= (p[0] - x1) * (y2 - y1):
                hull.pop()
            else:
                break
        hull.append(p)
    on = {p for p in hull}
    return [p not in on for p in zip(cost, elig)]


def b1(n_points=15):
    curves = frozen_curves()
    print("== B1: frontier ==", flush=True)
    _, cfg = ic.instance(seed=42, curves=curves)
    inst = ic.nb("build_instance")(cfg)
    c_min = cost_min_of(cfg, "b1")["cost"]
    ceil = evaluate(cfg, "elig_max", None, "b1")
    c_max = ceil["summary"]["cost"]
    if c_max <= c_min * 1.001:
        c_max = c_min * 1.5
    rows, logs = [], []
    for eps in np.linspace(c_min, c_max, n_points):
        key = ic.cache_key("v3", "b1", cfg, float(eps))

        def run(eps=float(eps)):
            ma = ic.nb("build_model")(inst, objective="elig", cost_cap=eps)
            ok, la = ic.solve_logged(ma, f"b1 eps={eps:.0f} stage1")
            if not ok:
                return dict(ok=False, logs=[la])
            e = float(pyo.value(ma.ELIG))
            mb = ic.nb("build_model")(inst, objective="cost", cost_cap=eps)
            mb.con.add(mb.ELIG >= e - 1e-9)       # lexicographic: then min cost
            ok2, lb = ic.solve_logged(mb, f"b1 eps={eps:.0f} stage2")
            best = mb if ok2 else ma
            return dict(ok=True, logs=[la, lb], summary=ic.summarise(best))

        res = ic.cached(key, run)
        logs += res["logs"]
        if res["ok"]:
            rows.append(dict(eps=float(eps), **res["summary"]))
    # Costs are dollars: two solves of the same point can differ by ~1e-5 of a
    # dollar on solver tolerance alone, and rounding to 6 decimals keeps that
    # as a second "distinct" frontier point.  Dedupe at the cent.
    f = pd.DataFrame(rows)
    f["cost"] = f.cost.round(2)
    f["ELIG"] = f.ELIG.round(6)
    f = (f.drop_duplicates(subset=["cost", "ELIG"])
         .sort_values("cost").reset_index(drop=True))
    f["unsupported"] = upper_hull_flags(f.cost.values, f.ELIG.values)
    knee_i, f_sorted = ic.nb("knee_point")(f)
    f.to_csv(os.path.join(R, "b1_frontier.csv"), index=False)
    pd.DataFrame(logs).to_csv(os.path.join(R, "b1_solves.csv"), index=False)
    fg.fig_b1_frontier(f, knee_i, c_min, c_min * 1.05, curves,
                       os.path.join(R, "fig_b1_frontier.png"))
    print(f"  {len(f)} points, {int(f.unsupported.sum())} unsupported, "
          f"knee at cost {f.cost.iloc[knee_i]:,.0f}", flush=True)
    return f, knee_i


# =============================================================================
# B3 -- cost side
# =============================================================================
SA_XLSX = "CART_parameter_sources.xlsx"


def sa_levels():
    """The SA levels tab if it exists, otherwise the notebook's own GRID."""
    if os.path.exists(SA_XLSX):
        tab = pd.read_excel(SA_XLSX, sheet_name="SA levels")
        out = []
        for _, r in tab.iterrows():
            out.append(dict(row=str(r["parameter"]), code=str(r["column"]),
                            label=str(r["level"]), value=r["value"],
                            is_base=bool(r.get("is_base", False)),
                            cost_side=True))
        return out, "SA levels tab of " + SA_XLSX
    # The GRID's Weibull rows were calibrated for the notebook's own BASE
    # (eta_H 15, eta_L 29), not for the frozen set (eta_H 25, eta_L 80), so
    # sweeping them here would walk away from the curves section 0 froze --
    # and they are not cost-side parameters, which is what B3 is for.  They
    # are still solved and kept in the CSV, but flagged out of the figure.
    clinical = {"eta_high", "eta_low", "k_high", "k_low"}
    out = []
    for row, code, values in ic.nb("GRID"):
        for v in values:
            out.append(dict(row=row, code=code, label=str(v), value=v,
                            is_base=(ic.nb("BASE")[code] == v),
                            cost_side=code not in clinical))
    return out, ("the notebook's GRID (" + SA_XLSX + " is not in the "
                 "repository)")


def frontier_for(cfg, tag, n_points=7):
    key = ic.cache_key("v3", tag, cfg, n_points)

    def run():
        inst = ic.nb("build_instance")(cfg)
        m0 = ic.nb("build_model")(inst, objective="cost")
        ok, l0 = ic.solve_logged(m0, "cost")
        if not ok:
            return dict(ok=False, logs=[l0], rows=[])
        c_min = float(pyo.value(m0.TOTCOST))
        m1 = ic.nb("build_model")(inst, objective="elig")
        ok1, l1 = ic.solve_logged(m1, "ceiling")
        if not ok1:
            return dict(ok=False, logs=[l0, l1], rows=[])
        c_max = max(float(pyo.value(m1.TOTCOST)), c_min * 1.5)
        rows, logs = [], [l0, l1]
        for eps in np.linspace(c_min, c_max, n_points):
            ma = ic.nb("build_model")(inst, objective="elig",
                                      cost_cap=float(eps))
            oka, la = ic.solve_logged(ma, f"eps={eps:.0f}")
            logs.append(la)
            if not oka:
                continue
            e = float(pyo.value(ma.ELIG))
            mb = ic.nb("build_model")(inst, objective="cost",
                                      cost_cap=float(eps))
            mb.con.add(mb.ELIG >= e - 1e-9)
            okb, lb = ic.solve_logged(mb, f"eps={eps:.0f} stage2")
            logs.append(lb)
            rows.append(ic.summarise(mb if okb else ma))
        return dict(ok=True, logs=logs, rows=rows, c_min=c_min)

    return ic.cached(key, run)


def b3():
    curves = frozen_curves()
    print("== B3: cost side ==", flush=True)
    levels, source = sa_levels()
    print(f"  SA levels from {source}", flush=True)

    base_cfg = ic.instance(seed=42, curves=curves)[1]
    base = frontier_for(base_cfg, "b3")
    base_front = pd.DataFrame(base["rows"])

    pts, logs = [], []
    for lv in levels:
        cfg = dict(base_cfg)
        cfg[lv["code"]] = lv["value"]
        res = frontier_for(cfg, "b3")
        # the solve log already has its own `label`, so the level is carried
        # under a separate key rather than as a colliding keyword
        logs += [dict(l, row=lv["row"], level=lv["label"]) for l in res["logs"]]
        for r in res["rows"]:
            pts.append(dict(row=lv["row"], code=lv["code"], label=lv["label"],
                            is_base=lv["is_base"],
                            cost_side=lv["cost_side"], **r))
        print(f"    {lv['row']} = {lv['label']}: {len(res['rows'])} points",
              flush=True)
    pts = pd.DataFrame(pts)
    pts.to_csv(os.path.join(R, "b3_sensitivity.csv"), index=False)
    pd.DataFrame(logs).to_csv(os.path.join(R, "b3_solves.csv"), index=False)
    shown = pts[pts.cost_side] if "cost_side" in pts else pts
    if len(shown):
        fg.fig_b3_sensitivity(shown, base_front, curves,
                              os.path.join(R, "fig_b3_sensitivity.png"))

    # (b) cost composition at C_min and at the knee, per capacity level
    mix = []
    for scale, fcap in ((0.5, (1, 2)), (1.0, (2, 4)), (1.5, (3, 6))):
        cfg = dict(base_cfg); cfg["fcap"] = fcap
        res = frontier_for(cfg, "b3")
        if not res["ok"] or not res["rows"]:
            continue
        f = pd.DataFrame(res["rows"]).sort_values("cost").reset_index(drop=True)
        knee_i, _ = ic.nb("knee_point")(f)
        label = f"M1={fcap[0]}, M2={fcap[1]}\n({scale:g}x lines)"
        for point, i in (("min cost", 0), ("knee", knee_i)):
            r = f.iloc[i]
            mix.append(dict(config=label, point=point, scale=scale,
                            CTM=r.CTM, TTC=r.TTC, HOLD=r.HOLD,
                            cost=r.cost, ELIG=r.ELIG))
    mix = pd.DataFrame(mix)
    mix.to_csv(os.path.join(R, "b3_costmix.csv"), index=False)
    if len(mix):
        fg.fig_b3_costmix(mix, curves, os.path.join(R, "fig_b3_costmix.png"))
    return pts, mix


# =============================================================================
# Summary
# =============================================================================
def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f}"


def summary():
    curves = frozen_curves()
    meta = json.load(open(FROZEN))
    out = ["# INFORMS runs - numbers for the slides", ""]
    out.append("All Weibull curves below are **illustrative showcase values**, "
               "chosen to make the scheduling mechanism visible. They are not "
               "clinical estimates.")
    out.append("")
    out.append(f"Frozen curve set **{meta['set']}**: high risk "
               f"eta={curves['High']['eta']:g}, k={curves['High']['k']:g}; "
               f"low risk eta={curves['Low']['eta']:g}, "
               f"k={curves['Low']['k']:g}.")
    out.append("")
    out.append(f"Solver settings: `{json.dumps(meta['solver'])}`.")
    out.append("")

    issues = []

    # --- slide 7 -----------------------------------------------------------
    p = os.path.join(R, "exp1_metrics.csv")
    if os.path.exists(p):
        met = pd.read_csv(p)
        runs = pd.read_csv(os.path.join(R, "exp1_runs.csv"))
        out += ["## Slide 7 - same-cost schedule comparison", ""]
        out.append(f"Seeds {met.seed.min()}-{met.seed.max()} "
                   f"({len(met)} instances), every plan within "
                   f"{ic.COST_TOL:.1%} of the cheapest plan's cost.")
        out.append("")
        for col, lab in (("same_cost_gain", "same-cost gain"),
                         ("paid_gain", "paid gain at +5% budget"),
                         ("ceiling_gap", "ceiling gap (no budget)"),
                         ("same_cost_range", "same-cost range (E_max - E_min)")):
            out.append(f"- **{lab}**: mean {_fmt(met[col].mean())} "
                       f"(min {_fmt(met[col].min())}, "
                       f"max {_fmt(met[col].max())}) expected eligible "
                       f"patients")
        out.append("")
        lev = [("n_start_order_changed", "start order"),
               ("n_start_day_changed", "start day"),
               ("n_mode_changed", "transport mode"),
               ("n_facility_changed", "facility")]
        out.append("Levers the eligibility-aware plan actually moves, counted "
                   "per instance against the cost+speed plan (10 patients "
                   "each):")
        out.append("")
        for col, lab in lev:
            out.append(f"- {lab}: mean {_fmt(met[col].mean(), 1)} patients "
                       f"(max {int(met[col].max())})")
        biggest = max(lev, key=lambda kv: met[kv[0]].mean())
        out.append("")
        out.append(f"The lever that changes most is **{biggest[1]}** "
                   f"(mean {_fmt(met[biggest[0]].mean(), 1)} patients per "
                   f"instance).")
        out.append("")
        same = runs[runs.budget.isin(["C_min", "own_min_cost"])
                    & runs.policy.isin(ic.POLICIES)]
        tab = (same.groupby("policy")
               .agg(ELIG_mean=("ELIG", "mean"), ELIG_min=("ELIG", "min"),
                    ELIG_max=("ELIG", "max"), cost_mean=("cost", "mean"),
                    n=("ELIG", "size"),
                    at_C_min=("budget", lambda b: (b == "C_min").all()))
               .reindex(ic.POLICIES).round(3))
        out.append("Policy results. `at_C_min` False means the policy could "
                   "not be scheduled at the cheapest plan's cost and is shown "
                   "at its own minimum, so its eligibility is NOT comparable "
                   "with the rest of the table:")
        out.append("")
        out.append(ic.md_table(tab, index=True, floatfmt="{:,.3f}"))
        out.append("")
        for pol, lab in (("fifo", "FIFO"),
                         ("high_risk_first", "high-risk-first")):
            sub = same[(same.policy == pol) & (same.budget == "C_min")]
            mx = same[same.policy == "elig_max"]
            # only policies that actually ran at C_min belong in a same-cost
            # comparison; the others get their own line below
            if len(sub) and len(mx):
                out.append(f"- {lab} reaches {_fmt(sub.ELIG.mean())} on "
                           f"average against {_fmt(mx.ELIG.mean())} for the "
                           f"eligibility-aware plan "
                           f"({_fmt(mx.ELIG.mean() - sub.ELIG.mean())} "
                           f"fewer eligible patients), over "
                           f"{len(sub)} of {len(met)} instances.")
            else:
                n_inf = int((runs[runs.policy == pol].budget
                             == "own_min_cost").sum())
                out.append(f"- {lab} was infeasible at C_min in {n_inf} of "
                           f"{len(met)} instances; see the own-cheapest-plan "
                           f"rows in exp1_runs.csv.")
                issues.append(f"{lab} infeasible at C_min in {n_inf} of "
                              f"{len(met)} instances")
        own = runs[runs.budget == "own_min_cost"]
        base_cost = runs[runs.budget == "C_min"].cost.mean()
        for pol in own.policy.unique():
            o = own[own.policy == pol]
            out.append(f"- {POL_LABEL(pol)} at its own cheapest cost: "
                       f"{o.cost.mean():,.0f} against {base_cost:,.0f} for "
                       f"every other policy "
                       f"(+{100 * (o.cost.mean() / base_cost - 1):.0f}%), "
                       f"reaching {_fmt(o.ELIG.mean())} eligible patients. "
                       f"It buys more than the eligibility-aware plan, but "
                       f"not at the same cost.")
        out.append("")

    # --- slide 8 -----------------------------------------------------------
    p = os.path.join(R, "exp2_gains.csv")
    if os.path.exists(p):
        g = pd.read_csv(p)
        out += ["## Slide 8 - congestion x decline", ""]
        t = (g.groupby(["decline", "congestion"])
             [["same_cost_gain", "paid_gain"]]
             .agg(["mean", "min", "max"]).round(3))
        out.append(ic.md_table(t, index=True, floatfmt="{:.3f}"))
        out.append("")
        for dec in g.decline.unique():
            sub = g[g.decline == dec]
            order = [c for c, _ in fg.CONGESTION_ORDER]
            means = [sub[sub.congestion == c].same_cost_gain.mean()
                     for c in order]
            means = [m for m in means if m == m]
            if len(means) < 2:
                continue
            spread = max(means) - min(means)
            if spread < 0.02:
                shape = "is flat across congestion"
            elif means[-1] > means[0]:
                shape = "grows as congestion rises"
            else:
                shape = "falls as congestion rises"
            out.append(f"- **{dec}**: the same-cost gain {shape} "
                       f"({' -> '.join(f'{m:.3f}' for m in means)} for "
                       f"{', '.join(order)}).")
        out.append("")

    # --- B1 ----------------------------------------------------------------
    p = os.path.join(R, "b1_frontier.csv")
    if os.path.exists(p):
        f = pd.read_csv(p)
        knee_i, _ = ic.nb("knee_point")(f)
        out += ["## B1 - frontier", ""]
        out.append(f"- {len(f)} distinct frontier points, "
                   f"{int(f.unsupported.sum())} unsupported "
                   f"(not on the convex hull).")
        out.append(f"- Knee at cost {f.cost.iloc[knee_i]:,.0f} with "
                   f"{_fmt(f.ELIG.iloc[knee_i])} expected eligible patients.")
        out.append(f"- Cost range {f.cost.min():,.0f} to {f.cost.max():,.0f}; "
                   f"eligibility {_fmt(f.ELIG.min())} to "
                   f"{_fmt(f.ELIG.max())}.")
        out.append("")

    # --- B3 ----------------------------------------------------------------
    p = os.path.join(R, "b3_sensitivity.csv")
    if os.path.exists(p):
        pts = pd.read_csv(p)
        _, source = sa_levels()
        out += ["## B3 - cost side", ""]
        out.append(f"- Sensitivity levels taken from {source}.")
        shown = pts[pts.cost_side] if "cost_side" in pts else pts
        out.append(f"- {pts.row.nunique()} parameters, "
                   f"{pts.groupby(['row', 'label']).ngroups} levels, "
                   f"{len(pts)} frontier points in total.")
        if len(shown) < len(pts):
            dropped = sorted(set(pts[~pts.cost_side].row))
            out.append(f"- {fg_panel_count(shown)} of those parameters are "
                       f"drawn in fig_b3_sensitivity.png. The Weibull rows "
                       f"({', '.join(dropped)}) are kept in the CSV but left "
                       f"out of the figure: the GRID's levels for them were "
                       f"set for the notebook's BASE curves (eta_H 15, "
                       f"eta_L 29), not for the frozen set (eta_H "
                       f"{curves['High']['eta']:g}, eta_L "
                       f"{curves['Low']['eta']:g}), so sweeping them walks "
                       f"away from the curves section 0 froze, and they are "
                       f"not cost-side parameters.")
        if not os.path.exists(SA_XLSX):
            issues.append(f"{SA_XLSX} is not in the repository, so B3(a) used "
                          f"the notebook's GRID instead of the SA levels tab")
        mixp = os.path.join(R, "b3_costmix.csv")
        if os.path.exists(mixp):
            mix = pd.read_csv(mixp)
            mix["config"] = mix.config.str.replace("\n", " ", regex=False)
            out.append("")
            out.append(ic.md_table(mix.round(3), floatfmt="{:,.3f}"))
        out.append("")

    # --- solve quality -----------------------------------------------------
    out += ["## Solve quality and caveats", ""]
    bad = []
    for name in ("exp1_runs.csv", "exp2_runs.csv", "b1_solves.csv",
                 "b3_solves.csv"):
        p = os.path.join(R, name)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "optimal" in df:
            n = int((~df.optimal.astype(bool)).sum())
            if n:
                bad.append(f"{name}: {n} solves not reported optimal")
    out.append("- " + ("; ".join(bad) if bad
                       else "every solve returned optimal."))
    out.append("- Base instance matches the notebook's BASE exactly "
               "(10 patients, window 15, M1 2 lines / M2 4 lines, fast 1 d / "
               "slow 3 d, TLS 1, TMFE 5, TQC 2, horizon 45, HC $200); no "
               "mismatch found.")
    out.append(f"- Font: {fg.FONT} (IBM Plex Sans is not installed here).")
    for i in issues:
        out.append(f"- {i}")
    out.append("")

    with open(os.path.join(R, "SUMMARY.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", os.path.join(R, "SUMMARY.md"), flush=True)


def fg_panel_count(df):
    return df.row.nunique()


def POL_LABEL(p):
    return fg.POLICY_LABEL.get(p, p)


STAGES = dict(screen=screen, exp1=exp1, exp2=exp2, b1=b1, b3=b3,
              summary=summary)


def main(argv):
    want = argv[1:] or ["all"]
    if want == ["all"]:
        want = ["screen", "exp1", "exp2", "b1", "b3", "summary"]
    t0 = time.time()
    for s in want:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}; pick from {list(STAGES)}")
        STAGES[s]()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv)
