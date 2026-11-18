"""Pilot with continuous patient arrivals: build the 4 settings, solve Baseline (FIFO) and Patient-aware, write results.

Usage (from the repository root or from pilot_continuous/):
    python run_pilot.py                     # all settings, solver from the config block below
    python run_pilot.py --settings P1 P2    # only some settings
    python run_pilot.py --settings P5       # P5 "demand surge" (not part of the default run; outputs in results/P5_surge/)
    python run_pilot.py --count-only        # instance sizes and FIFO constraint counts, no solving
    python run_pilot.py --rerun             # ignore cached solves and solve again
    python run_pilot.py --smoke             # tiny 5-patient test with HiGHS (writes to results_smoke/)
    python run_pilot.py --smoke --solver gurobi   # the same test with Gurobi (checks gurobipy and the license)
    --solver gurobi|gams|highs overrides SOLVER from the configuration block

Every solve is cached in results/cache/ as soon as it finishes, so an interrupted run resumes where it stopped.
"""
# ================================================================================================================
# CONFIGURATION — all solver choices are here
# ================================================================================================================
SOLVER = "gurobi"             # "gurobi" (default; Gurobi called directly through gurobipy), "gams" or "highs"
GAMS_SUBSOLVER = "cplex"      # only for SOLVER = "gams": "cplex", "gurobi" or "cbc"
GAMS_EXECUTABLE = None        # None = find "gams" on PATH; or a full path, e.g. r"C:\GAMS\48\gams.exe"
TIME_LIMIT_S = 1800           # per solve (GAMS option reslim)
MIP_GAP = 0.0                 # relative gap (GAMS option optcr); optca is set to 0
THREADS = 0                   # 0 = solver default (option not passed); otherwise number of threads
FIFO_FORM = "timeindexed"     # Baseline FIFO: "timeindexed" (default) or "bigM"
KEEP_GAMS_FILES = False       # True keeps the .gms/.lst files GAMS writes (for debugging)
SHOW_SOLVER_LOG = True        # print the Gurobi/HiGHS log while solving

# ---- pilot design ----------------------------------------------------------------------------------------------
ARRIVAL_DAYS = 30             # D: patients arrive on days 1..D
ND = 41                       # maximum turnaround (days); horizon NT = D + ND + 2
URGENT_PROB = 0.30            # each patient is urgent with this probability
SEED = 1                      # one replication; the same seed gives the same patients for the same demand
LAMBDA_D = 150_000.0          # Patient-aware weight on E_D (lambda_U = 0)
LATE_FROM_DAY = 8             # extra metrics for patients collected on this day or later (removes the empty start)
SETTINGS = {                  # name: annual demand (patients/year) and lines (Philadelphia, Houston, Los Angeles)
    "P1": dict(annual_demand=1000, lines=(10, 10, 10)),   # overall utilisation about 0.64
    "P2": dict(annual_demand=1000, lines=(4, 10, 10)),    # about 0.80
    "P3": dict(annual_demand=2000, lines=(31, 4, 10)),    # about 0.85
    "P4": dict(annual_demand=2000, lines=(31, 4, 4)),     # about 0.98
    # P5 "demand surge": nonhomogeneous Poisson arrivals. Base rate = P1/P2 rate (1000/yr), multiplied by
    # surge_factor on collection days surge_start..surge_end. Run it by name: python run_pilot.py --settings P5.
    # optional=True keeps it out of the default run; its outputs go to results/<out_subdir>/.
    "P5": dict(annual_demand=1000, lines=(14, 3, 7), surge_factor=1.5, surge_start=11, surge_end=20,
               optional=True, out_subdir="P5_surge"),
}
MODELS = ["Baseline", "Patient-aware"]
# ================================================================================================================

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import instances as inst_mod   # noqa: E402
import model as mdl_mod        # noqa: E402

WORKBOOK = HERE.parent / "CART_instance_30patients_cap4.xlsx"   # network data (same workbook as CART_doe_main.ipynb)
RESULTS_DIR = HERE / "results"


# ----------------------------------------------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------------------------------------------
def _status(tc_name, has_solution):
    if tc_name == "optimal" and has_solution:
        return "optimal"
    if tc_name in ("maxTimeLimit", "maxIterations", "maxEvaluations", "userInterrupt", "interrupted"):
        return "time limit, feasible" if has_solution else "time limit, no solution"
    if tc_name in ("infeasible", "infeasibleOrUnbounded"):
        return "infeasible"
    return f"{tc_name}{', feasible' if has_solution else ', no solution'}"


def solve_gams(mdl):
    from pyomo.opt import SolutionStatus
    if GAMS_EXECUTABLE:
        import pyomo.common
        pyomo.common.Executable("gams").set_path(GAMS_EXECUTABLE)
    opt = pyo.SolverFactory("gams")            # GAMS shell interface: writes a .gms file and calls the gams executable
    if not opt.available(exception_flag=False):
        raise RuntimeError("GAMS not found: put the GAMS folder on PATH or set GAMS_EXECUTABLE in run_pilot.py")
    add = [f"option reslim={TIME_LIMIT_S};", f"option optcr={MIP_GAP};", "option optca=0;"]
    if THREADS:
        add.append(f"option threads={THREADS};")
    t0 = time.time()
    res = opt.solve(mdl, solver=GAMS_SUBSOLVER, tee=True, load_solutions=False, add_options=add,
                    keepfiles=KEEP_GAMS_FILES)
    secs = time.time() - t0
    tc = res.solver.termination_condition
    soln = res.solution(0) if len(res.solution) else None
    has_sol = soln is not None and soln.status in (SolutionStatus.optimal, SolutionStatus.feasible,
                                                   SolutionStatus.locallyOptimal)
    if has_sol:
        mdl.solutions.load_from(res)
        has_sol = all(v.value is not None and not math.isnan(v.value) for v in mdl.X.values())
    obj = res.problem.upper_bound if has_sol else float("nan")   # minimisation: upper = objective, lower = bound
    bound = res.problem.lower_bound
    return dict(status=_status(tc.name, has_sol), has_solution=has_sol, objective=obj,
                best_bound=float(bound) if bound is not None else float("nan"), solve_s=secs,
                solver=f"gams/{GAMS_SUBSOLVER}", termination=str(tc.name))


def solve_appsi(mdl, which):
    """Gurobi (gurobipy) or HiGHS (highspy) through Pyomo's appsi interface."""
    from pyomo.contrib.appsi.base import TerminationCondition as TC
    if which == "gurobi":
        from pyomo.contrib.appsi.solvers import Gurobi
        opt = Gurobi()
        avail = opt.available()
        if not avail:
            raise RuntimeError(f"Gurobi not usable ({avail.name}): install gurobipy and a Gurobi license (see README)")
        if avail.name == "LimitedLicense":
            print("  WARNING: Gurobi is running with the size-limited pip license (2,000 variables/constraints); "
                  "the pilot settings need your full license (gurobi.lic)")
        opt.gurobi_options = {"Seed": 0, "MIPGapAbs": 1e-9}
        if THREADS:
            opt.gurobi_options["Threads"] = THREADS
    else:
        from pyomo.contrib.appsi.solvers import Highs
        opt = Highs()
        opt.highs_options = {"random_seed": 0, "mip_abs_gap": 1e-9}
        if THREADS:
            opt.highs_options["threads"] = THREADS
    opt.config.time_limit = TIME_LIMIT_S
    opt.config.mip_gap = MIP_GAP
    opt.config.load_solution = False
    opt.config.stream_solver = SHOW_SOLVER_LOG
    t0 = time.time()
    res = opt.solve(mdl)
    secs = time.time() - t0
    tc = res.termination_condition
    has_sol = res.best_feasible_objective is not None and tc in (TC.optimal, TC.maxTimeLimit, TC.interrupted,
                                                                 TC.maxIterations)
    if has_sol:
        res.solution_loader.load_vars()
    bound = res.best_objective_bound
    return dict(status=_status(tc.name, has_sol), has_solution=has_sol,
                objective=res.best_feasible_objective if has_sol else float("nan"),
                best_bound=float(bound) if bound is not None else float("nan"), solve_s=secs, solver=which,
                termination=str(tc.name))


def solve(mdl):
    if SOLVER == "gams":
        return solve_gams(mdl)
    if SOLVER in ("gurobi", "highs"):
        return solve_appsi(mdl, SOLVER)
    raise ValueError(f"unknown SOLVER {SOLVER!r}")


# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------
def instance_for(net, cfg, out_dir):
    """Generate the instance once per (demand, seed, D, urgent probability) and reuse its CSV afterwards."""
    D = cfg.get("D", ARRIVAL_DAYS)
    name = f"demand{cfg['annual_demand']}_D{D}_urgent{int(round(URGENT_PROB * 100))}_seed{cfg.get('seed', SEED)}"
    if cfg.get("surge_start") is not None:
        name += f"_surge{cfg['surge_factor']:g}x_d{cfg['surge_start']}-{cfg['surge_end']}"
    if cfg.get("max_patients"):
        name += f"_first{cfg['max_patients']}"
    path = out_dir / "instances" / f"{name}.csv"
    if path.exists():
        return inst_mod.load_instance(path), path
    df = inst_mod.generate_instance(net, D, cfg["annual_demand"], URGENT_PROB, cfg.get("seed", SEED),
                                    max_patients=cfg.get("max_patients"), rate=rates_for(cfg))
    inst_mod.save_instance(df, path)
    return df, path


def rates_for(cfg):
    """Per-day arrival rate vector of a setting (constant unless the setting has a surge)."""
    D = cfg.get("D", ARRIVAL_DAYS)
    if cfg.get("surge_start") is None:
        return inst_mod.daily_rates(D, cfg["annual_demand"])
    return inst_mod.daily_rates(D, cfg["annual_demand"], cfg["surge_factor"], cfg["surge_start"], cfg["surge_end"])


def surge_groups(cfg):
    """(label, first day, last day) of the pre-surge, surge and post-surge collection-day groups."""
    D, a, b = cfg.get("D", ARRIVAL_DAYS), cfg["surge_start"], cfg["surge_end"]
    return [(f"pre-surge (days 1-{a - 1})", 1, a - 1), (f"surge (days {a}-{b})", a, b),
            (f"post-surge (days {b + 1}-{D})", b + 1, D)]


def queue_by_day(d, tab):
    """Patients collected but not yet started, per assigned facility and day: collection day <= day < start."""
    days = np.arange(1, d["NT"] + 1)
    out = pd.DataFrame({"day": days})
    for m in d["M"]:
        sub = tab[tab.facility == m]
        out[m] = [int(((sub["collection day"] <= t) & (sub.start > t)).sum()) for t in days]
    return out


def plot_queue(name, d, cfg, queues, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2 = "#52514e"
    color = {"Baseline": "#eb6834", "Patient-aware": "#2a78d6"}
    fig, axes = plt.subplots(1, len(d["M"]), figsize=(13, 3.8), sharey=True)
    for ax, m in zip(axes, d["M"]):
        ax.axvspan(cfg["surge_start"] - 0.5, cfg["surge_end"] + 0.5, color="#e4e3df", alpha=0.6, lw=0,
                   label="surge days")
        for model_name, q in queues.items():
            ax.step(q.day, q[m], where="post", lw=2, color=color[model_name], label=model_name,
                    ls="-" if model_name == "Baseline" else "--")
        ax.set_title(f"{m} {d['M_CITY'][m].split(',')[0]} ({d['FCAP'][m]} lines)", loc="left", fontsize=11)
        ax.set_xlabel("day")
    axes[0].set_ylabel("patients collected, not yet started")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{name}: queue per facility per day (shaded = surge, rate x{cfg['surge_factor']:g})",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scope_metrics(tab, mask):
    t = tab[mask]
    u, r = t[t["class"] == "U"], t[t["class"] == "R"]
    return {"patients": len(t), "total cost": t["plan cost"].sum(), "E_D": t["pi_D at delivery"].sum(),
            "E_U": t["pi_U at delivery"].sum(),
            "mean wait urgent": u.W_p.mean() if len(u) else np.nan, "max wait urgent": u.W_p.max() if len(u) else np.nan,
            "mean wait regular": r.W_p.mean() if len(r) else np.nan, "max wait regular": r.W_p.max() if len(r) else np.nan}


def jdump(obj, path):
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(obj, indent=1, default=conv))


def plot_waiting(name, d, waits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid, surf = "#52514e", "#e4e3df", "#fcfcfb"
    color = {"Baseline": "#eb6834", "Patient-aware": "#2a78d6"}
    plt.rcParams.update({"figure.facecolor": surf, "axes.facecolor": surf, "savefig.facecolor": surf,
                         "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": grid,
                         "font.size": 10})
    fig, axes = plt.subplots(1, len(d["M"]), figsize=(13, 3.8), sharey=True)
    for ax, m in zip(axes, d["M"]):
        for model_name, w in waits.items():
            ax.step(w.day, w[m], where="post", lw=2, color=color[model_name], label=model_name)
        ax.axvline(d["D"], color=ink2, lw=0.8, ls=":")
        ax.set_title(f"{m} {d['M_CITY'][m].split(',')[0]} ({d['FCAP'][m]} lines)", loc="left", fontsize=11)
        ax.set_xlabel("day")
    axes[0].set_ylabel("patients waiting at the facility")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{name}: patients waiting per day (dotted line = last arrival day)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------
def run(settings, out_dir, rerun=False, count_only=False):
    for sub in ["cache", "plans", "figures", "instances"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"solver: {SOLVER}{'/' + GAMS_SUBSOLVER if SOLVER == 'gams' else ''}  time limit {TIME_LIMIT_S}s  "
          f"gap {MIP_GAP}  threads {THREADS or 'solver default'}  FIFO form {FIFO_FORM}")
    net = inst_mod.load_network(WORKBOOK)
    run_rows, set_rows, check_rows, group_rows, queue_frames = [], [], [], [], []

    for name, cfg in settings.items():
        D, nd = cfg.get("D", ARRIVAL_DAYS), cfg.get("ND", ND)
        fifo_form = cfg.get("fifo_form", FIFO_FORM)
        patients, ipath = instance_for(net, cfg, out_dir)
        d = inst_mod.make_data(net, patients, nd, cfg["lines"], D)
        rho = inst_mod.utilisation(net, patients, cfg["lines"], D, cfg["annual_demand"], rate=rates_for(cfg))
        print(f"\n{'=' * 100}\n{name}: demand {cfg['annual_demand']}/yr, lines (Phil, Hou, LA) = {tuple(cfg['lines'])}, "
              f"ND {nd}, D {D}, NT {d['NT']}, N = {len(patients)} patients "
              f"({(patients['class'] == 'U').sum()} urgent), instance {ipath.name}")
        print("  " + "  ".join(f"{k} = {v:.3f}" for k, v in rho.items()))
        if cfg.get("surge_start") is not None:
            base = cfg["annual_demand"] / 365.0
            for lab, r_ in [("normal", base), (f"surge x{cfg['surge_factor']:g}", base * cfg["surge_factor"])]:
                loads = inst_mod.expected_loads(net, cfg["lines"], r_)
                print(f"  expected load, {lab} days ({r_:.3f} patients/day; per facility = nearest by distance): "
                      + "  ".join(f"{k} {v:.3f}" for k, v in loads.items()))
            print("  patients collected: " + "  ".join(
                f"{lab}: {int(patients['collection day'].between(a, b).sum())}" for lab, a, b in surge_groups(cfg)))
        plans = mdl_mod.enumerate_plans(d)
        with_plan = {k[0] for k in plans}
        no_plan = [p for p in d["P"] if p not in with_plan]
        chk = f"every patient has an admissible plan within ND ({nd})"
        print(f"  [{'PASS' if not no_plan else 'FAIL'}] {chk}"
              + (f" — no admissible plan for {len(no_plan)} patient(s): {', '.join(no_plan[:10])}" if no_plan else ""))
        if not count_only:
            check_rows.append(dict(setting=name, model="(instance)", check=chk, result="PASS" if not no_plan else "FAIL"))
        if count_only:
            idx = mdl_mod.index_plans(plans)
            n_cap = sum(1 for m in d["M"] for t in range(1, d["NT"] + 1)
                        if any((m, tt) in idx["by_mt"] for tt in range(max(1, t - d["T_MFE"] + 1), t + 1)))
            for form in mdl_mod.FIFO_FORMS:
                n, nnz = mdl_mod.fifo_size(d, plans, form, idx)
                print(f"  FIFO {form:<11s}: {n:>10,} constraints, {nnz:>13,} nonzeros")
            print(f"  variables {len(plans):,}; one-plan + capacity constraints {len(d['P']) + n_cap:,}")
            continue

        tabs, results = {}, {}
        for kind in MODELS:
            cache = out_dir / "cache" / f"{name}__{kind}.json"
            plan_path = out_dir / "plans" / f"{name}__{kind}.csv"
            if cache.exists() and not rerun:
                r = json.loads(cache.read_text())
                print(f"  {kind}: cached ({r['status']}) — delete {cache.name} or use --rerun to solve again")
            else:
                if no_plan:
                    r = dict(status="infeasible", has_solution=False, objective=float("nan"), best_bound=float("nan"),
                             solve_s=0.0, solver="none", termination="no admissible plan",
                             n_variables=len(plans), n_constraints=0, n_fifo_constraints=0, fifo_form="", build_s=0.0,
                             note=f"patients with no admissible plan (TRT <= ND impossible): {', '.join(no_plan[:10])}")
                    r["checks"] = {}
                else:
                    m, info = mdl_mod.build_model(d, plans, kind, fifo_form=fifo_form, lambda_D=LAMBDA_D)
                    print(f"  {kind}: {info['n_variables']:,} variables, {info['n_constraints']:,} constraints"
                          + (f" of which FIFO ({fifo_form}) {info['n_fifo_constraints']:,}" if kind == "Baseline" else "")
                          + f"  [built in {info['build_s']:.1f}s] — solving ...", flush=True)
                    r = dict(solve(m), **info)
                    r["checks"] = {}
                    if r["has_solution"]:
                        tab = mdl_mod.chosen_plans(d, plans, m)
                        r.update(total_cost=pyo.value(m.cost), E_D=pyo.value(m.ED), E_U=pyo.value(m.EU))
                        r["checks"] = mdl_mod.check_solution(d, plans, tab, r["E_D"], fifo=(kind == "Baseline"))
                        tab.drop(columns=["j1", "j2"]).to_csv(plan_path, index=False)
                r["mip_gap"] = (abs(r["objective"] - r["best_bound"]) / max(abs(r["objective"]), 1e-10)
                                if r["has_solution"] and not math.isnan(r["best_bound"]) else float("nan"))
                r.update(setting=name, model=kind)
                jdump(r, cache)
            results[kind] = r
            msg = (f"  {kind}: {r['status'].upper()}  objective {r['objective']:,.2f}  bound {r['best_bound']:,.2f}  "
                   f"gap {r['mip_gap']:.2e}  [{r['solve_s']:.1f}s]" if r["has_solution"]
                   else f"  {kind}: {r['status'].upper()} — no solution ({r.get('termination', '')}). {r.get('note', '')}")
            print(msg)
            for chk, ok in r["checks"].items():
                print(f"    [{'PASS' if ok else 'FAIL'}] {chk}")
                check_rows.append(dict(setting=name, model=kind, check=chk, result="PASS" if ok else "FAIL"))
            if r["has_solution"] and plan_path.exists():
                tabs[kind] = pd.read_csv(plan_path)
            run_rows.append({"setting": name, "model": kind, "N": len(patients), "urgent": int((patients["class"] == "U").sum()),
                             "annual demand": cfg["annual_demand"], "lines": str(tuple(cfg["lines"])), "ND": nd, "D": D,
                             "NT": d["NT"], **rho, "status": r["status"], "objective": r["objective"],
                             "best bound": r["best_bound"], "MIP gap": r["mip_gap"], "solve time (s)": r["solve_s"],
                             "build time (s)": r.get("build_s"), "variables": r.get("n_variables"),
                             "constraints": r.get("n_constraints"), "FIFO constraints": r.get("n_fifo_constraints"),
                             "FIFO form": r.get("fifo_form"), "total cost": r.get("total_cost", np.nan),
                             "E_D": r.get("E_D", np.nan), "E_U": r.get("E_U", np.nan), "solver": r.get("solver")})
        if not any(r["has_solution"] for r in results.values()):
            print(f"  >>> {name} IS INFEASIBLE (no solution for either model) — continuing with the next setting")
        elif not all(r["has_solution"] for r in results.values()):
            bad = [k for k, r in results.items() if not r["has_solution"]]
            print(f"  >>> {name}: no solution for {', '.join(bad)} — comparison not possible, continuing")

        # comparison, all patients and patients collected on day LATE_FROM_DAY or later
        if all(k in tabs for k in MODELS):
            for scope, fn in [("all patients", lambda t: t["collection day"] >= 1),
                              (f"collected day >= {LATE_FROM_DAY}", lambda t: t["collection day"] >= LATE_FROM_DAY)]:
                b, a = scope_metrics(tabs["Baseline"], fn(tabs["Baseline"])), scope_metrics(tabs["Patient-aware"], fn(tabs["Patient-aware"]))
                diff = b["E_D"] - a["E_D"]
                row = {"setting": name, "scope": scope, "patients": b["patients"],
                       "E_D Baseline": b["E_D"], "E_D Patient-aware": a["E_D"], "difference in E_D": diff,
                       "relative reduction": diff / b["E_D"] if b["E_D"] > 0 else np.nan,
                       "total cost Baseline": b["total cost"], "total cost Patient-aware": a["total cost"],
                       "extra cost (USD)": a["total cost"] - b["total cost"],
                       "extra cost (%)": 100 * (a["total cost"] - b["total cost"]) / b["total cost"] if b["total cost"] else np.nan,
                       "E_U Baseline": b["E_U"], "E_U Patient-aware": a["E_U"]}
                for key in ["mean wait urgent", "max wait urgent", "mean wait regular", "max wait regular"]:
                    row[f"{key}, Baseline"], row[f"{key}, Patient-aware"] = b[key], a[key]
                set_rows.append(row)
        if tabs:
            waits = {k: mdl_mod.waiting_by_day(d, t) for k, t in tabs.items()}
            for k, w in waits.items():
                w.to_csv(out_dir / "plans" / f"{name}__{k}__waiting_by_day.csv", index=False)
            plot_waiting(name, d, waits, out_dir / "figures" / f"waiting_{name}.png")
        if cfg.get("surge_start") is not None and tabs:
            # queue per facility per day (Baseline required; Patient-aware alongside when available)
            queues = {k: queue_by_day(d, t) for k, t in tabs.items()}
            q = queues["Baseline"] if "Baseline" in queues else None
            wide = pd.DataFrame({"day": next(iter(queues.values())).day})
            for k, qq in queues.items():
                for m in d["M"]:
                    wide[f"{k} {m} {d['M_CITY'][m].split(',')[0]}"] = qq[m]
            wide.insert(0, "setting", name)
            wide.insert(2, "period", ["surge" if cfg["surge_start"] <= t <= cfg["surge_end"] else
                                      ("pre-surge" if t < cfg["surge_start"] else
                                       ("post-surge" if t <= D else "after arrivals")) for t in wide.day])
            queue_frames.append(wide)
            plot_queue(name, d, cfg, queues, out_dir / "figures" / f"queue_{name}.png")
            # pre-surge / surge / post-surge groups
            if all(k in tabs for k in MODELS):
                for lab, a, b in surge_groups(cfg):
                    row = {"setting": name, "group": lab}
                    vals = {}
                    for k in MODELS:
                        t = tabs[k][tabs[k]["collection day"].between(a, b)]
                        vals[k] = dict(n=len(t), ED=t["pi_D at delivery"].sum(),
                                       mean_wait=t.W_p.mean() if len(t) else np.nan,
                                       max_wait=t.W_p.max() if len(t) else np.nan,
                                       non_nearest=int((t.facility != t.site.map(net["NEAREST"])).sum()))
                    bb, aa = vals["Baseline"], vals["Patient-aware"]
                    row.update({"patients": bb["n"], "E_D Baseline": bb["ED"], "E_D Patient-aware": aa["ED"],
                                "E_D reduction": bb["ED"] - aa["ED"],
                                "E_D reduction (%)": 100 * (bb["ED"] - aa["ED"]) / bb["ED"] if bb["ED"] > 0 else np.nan,
                                "mean wait, Baseline": bb["mean_wait"], "mean wait, Patient-aware": aa["mean_wait"],
                                "max wait, Baseline": bb["max_wait"], "max wait, Patient-aware": aa["max_wait"],
                                "sent to non-nearest facility, Baseline": bb["non_nearest"],
                                "sent to non-nearest facility, Patient-aware": aa["non_nearest"]})
                    group_rows.append(row)

    if count_only:
        return
    pd.set_option("display.width", 250, "display.max_columns", 60)
    runs = pd.DataFrame(run_rows)
    runs.to_csv(out_dir / "summary_runs.csv", index=False)
    sets = pd.DataFrame(set_rows)
    sets.to_csv(out_dir / "summary_settings.csv", index=False)
    checks = pd.DataFrame(check_rows, columns=["setting", "model", "check", "result"])
    checks.to_csv(out_dir / "checks.csv", index=False)
    print(f"\n{'=' * 100}\nRUNS ({out_dir / 'summary_runs.csv'})")
    print(runs[["setting", "model", "N", "rho overall (realised)", "status", "MIP gap", "solve time (s)", "variables",
                "constraints", "FIFO constraints", "total cost", "E_D", "E_U"]].to_string(index=False))
    if len(sets):
        print(f"\nSETTINGS ({out_dir / 'summary_settings.csv'})")
        print(sets.set_index(["setting", "scope"]).T.to_string(float_format=lambda x: f"{x:,.4f}"))
    if group_rows:
        groups = pd.DataFrame(group_rows)
        groups.to_csv(out_dir / "summary_surge_groups.csv", index=False)
        print(f"\nSURGE GROUPS by collection day ({out_dir / 'summary_surge_groups.csv'})")
        print(groups.set_index(["setting", "group"]).T.to_string(float_format=lambda x: f"{x:,.4f}"))
    if queue_frames:
        queue = pd.concat(queue_frames, ignore_index=True)
        queue.to_csv(out_dir / "queue_by_day.csv", index=False)
        print(f"\nQUEUE per facility per day: patients collected but not yet started ({out_dir / 'queue_by_day.csv'})")
        for name_, qf in queue.groupby("setting", sort=False):
            busy = qf.drop(columns=["setting", "day", "period"]).sum(axis=1) > 0
            last = int(qf.day[busy].max()) if busy.any() else 0
            print(f"{name_} (days 1-{last}; later days have no queue)")
            print(qf[qf.day <= last].drop(columns=["setting"]).to_string(index=False))
    n_fail = int((checks.result == "FAIL").sum())
    print(f"\nCHECKS: {int((checks.result == 'PASS').sum())} PASS, {n_fail} FAIL")
    if n_fail:
        print(checks[checks.result == "FAIL"].to_string(index=False))
    return runs, sets, checks


def main():
    global SOLVER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", help="run only these settings (default: all)")
    ap.add_argument("--rerun", action="store_true", help="ignore cached solves")
    ap.add_argument("--count-only", action="store_true", help="print sizes and FIFO constraint counts, do not solve")
    ap.add_argument("--smoke", action="store_true", help="tiny 5-patient test with HiGHS")
    ap.add_argument("--out", help="output folder (default: pilot_continuous/results)")
    ap.add_argument("--solver", choices=["gurobi", "gams", "highs"], help="override SOLVER from the configuration block")
    args = ap.parse_args()

    if args.solver:
        SOLVER = args.solver
    if args.smoke:
        SOLVER = args.solver or "highs"
        out = Path(args.out) if args.out else HERE / "results_smoke"
        small = dict(annual_demand=1000, D=3, max_patients=5, lines=(1, 1, 1))
        settings = {"S1_timeindexed": dict(small, fifo_form="timeindexed"),
                    "S1_bigM": dict(small, fifo_form="bigM"),
                    "S2_no_capacity": dict(small, lines=(0, 0, 0)),      # infeasible on purpose (solver path)
                    "S3_ND16": dict(small, ND=16),                       # infeasible on purpose (no admissible plan)
                    "S4_surge": dict(annual_demand=100, lines=(1, 1, 1),  # tiny surge instance: tests P5 reporting
                                     surge_factor=2.0, surge_start=11, surge_end=20)}
        runs, _, checks = run(settings, out, rerun=True)
        b = runs[runs.model == "Baseline"].set_index("setting")
        same = abs(b.loc["S1_timeindexed", "objective"] - b.loc["S1_bigM", "objective"]) <= 1e-6
        print(f"\nSMOKE: Baseline objective timeindexed = bigM: {'PASS' if same else 'FAIL'} "
              f"({b.loc['S1_timeindexed', 'objective']:,.2f} vs {b.loc['S1_bigM', 'objective']:,.2f})")
        inf_ok = (runs[runs.setting.isin(["S2_no_capacity", "S3_ND16"])].status == "infeasible").all()
        print(f"SMOKE: infeasible settings reported as infeasible: {'PASS' if inf_ok else 'FAIL'}")
        expected_inf = checks.setting.isin(["S2_no_capacity", "S3_ND16"])     # infeasible on purpose
        surge_ok = (out / "summary_surge_groups.csv").exists() and (out / "queue_by_day.csv").exists()
        print(f"SMOKE: surge reporting written (groups and queue): {'PASS' if surge_ok else 'FAIL'}")
        ok = same and inf_ok and surge_ok and (checks[~expected_inf].result == "PASS").all()
        print(f"SMOKE OVERALL: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    unknown = set(args.settings or []) - set(SETTINGS)
    if unknown:
        ap.error(f"unknown setting(s): {', '.join(sorted(unknown))}; choose from {', '.join(SETTINGS)}")
    # default run = settings without optional=True (P1-P4); optional ones (P5) only when named
    chosen = {k: v for k, v in SETTINGS.items() if (k in args.settings if args.settings else not v.get("optional"))}
    base_out = Path(args.out) if args.out else RESULTS_DIR
    groups = {}                                            # settings with their own output folder run separately
    for k, v in chosen.items():
        groups.setdefault(v.get("out_subdir"), {})[k] = v
    for sub, sets in groups.items():
        run(sets, base_out / sub if sub else base_out, rerun=args.rerun, count_only=args.count_only)


if __name__ == "__main__":
    main()
