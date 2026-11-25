"""Main experimental design with the epsilon-constraint model.

Per instance (factors A-D x replication) nine solves:
  Baseline-FIFO                     min cost with FIFO                                                  (1 solve)
  Baseline-MinRisk                  (a) min cost -> C*; (b) min E_D s.t. cost <= C*(1+1e-6); E_D = ED_minrisk   (2)
  ED_cap (diagnostic)               min E_D subject to capacity only (no cost term)                     (1)
  PatientAware (eps=...)            min cost s.t. E_D <= eps, no FIFO, for 5 values of eps evenly spaced
                                    from ED_minrisk (cheapest) down to ED_cap (safest)                  (5)
Rows and files are labelled by the numeric eps (3 decimals). The column eps_step (1 = ED_minrisk ... 5 = ED_cap)
only records the position of a level so that levels can be averaged across instances.

Usage
  python run_doe.py                          # full design (480 instances), HiGHS, 1 process
  python run_doe.py --workers 4              # 4 instances in parallel (1 solver thread each)
  python run_doe.py --solver gurobi          # Gurobi through gurobipy (needs a full license)
  python run_doe.py --smoke                  # 2 instances only, written to results_doe_smoke/
  python run_doe.py --tables-only            # rebuild all_runs.csv, tables and checks from the cache
  python figures.py                          # trade-off figure, after the run
Every finished instance is cached in results_doe/_cache/, so an interrupted run resumes where it stopped.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import instances as I
import model as M

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_doe"
TIME_LIMIT = 120.0           # seconds per solve
TIE_TOL = 1e-6               # Baseline-MinRisk stage (b): cost <= C* * (1 + TIE_TOL)
N_EPS = 5                    # PatientAware levels from ED_minrisk down to ED_cap
MIDDLE_STEP = 3              # level used by Tables 1, 2 and 4
BASE_CASE = dict(ND=41, N=30, layout="C3", share=0.30)
KEYS = [(ND, N, layout, share, r) for ND in I.ND_LEVELS for N in I.N_LEVELS for layout in I.LAYOUTS
        for share in I.SHARES for r in I.REPS]
SMOKE_KEYS = [(41, 30, "C3", 0.30, 1), (52, 60, "C4", 0.30, 1)]
pa_label = lambda eps: f"PatientAware (eps={eps:.3f})"


def instance_name(key):
    ND, N, layout, share, r = key
    return f"ND{ND}_N{N}_{layout}_U{int(round(share * 100))}_r{r:02d}"


# ----------------------------------------------------------------------------------------------------------------
# One instance
# ----------------------------------------------------------------------------------------------------------------
def run_instance(job):
    key, out_dir, solver, time_limit, threads = job
    name = instance_name(key)
    cache = Path(out_dir) / "_cache" / f"{name}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    ND, N, layout, share, r = key
    t_start = time.time()
    d = I.make_data(str(out_dir), ND, N, layout, share, r)
    plans = M.enumerate_plans(d)
    common = dict(instance=name, ND=ND, N=N, layout=layout, lines=str(I.LAYOUTS[layout]), urgent_share=share, rep=r,
                  NT=d["NT"], n_urgent=sum(g == "U" for g in d["G"].values()), n_plans=len(plans))
    rows, plan_frames, checks = [], [], []

    def chk(model, check, ok, note=""):
        checks.append(dict(instance=name, model=model, check=check,
                           result=ok if isinstance(ok, str) else ("PASS" if ok else "FAIL"), note=note))

    def record(model, s, mdl, fifo=False, eps=None, eps_step=np.nan, extra=None):
        row = dict(common, model=model, eps=eps if eps is not None else np.nan, eps_step=eps_step,
                   status=s["status"], objective=s["objective"], best_bound=s["best_bound"], mip_gap=s["mip_gap"],
                   solve_s=s["solve_s"], n_fifo_constraints=len(mdl.FIFO), **(extra or {}))
        chk(model, "solve optimal or logged", True if s["status"] == "optimal"
            else f"LOGGED ({s['status']}, gap {s['mip_gap']:.2e})" if s["has_solution"] else f"LOGGED ({s['status']})")
        if s["has_solution"]:
            tab = M.plan_table(d, plans, mdl)
            row.update(M.plan_metrics(tab))
            for c, ok in M.check_plan(d, plans, tab, mdl, fifo=fifo, eps=eps).items():
                chk(model, c, ok)
            plan_frames.append(tab.assign(instance=name, model=model))
        rows.append(row)
        return row

    # 1. Baseline-FIFO
    mf = M.build_model(d, plans, fifo=True)
    r_fifo = record("Baseline-FIFO", M.solve(M.new_solver(solver, time_limit, threads), mf), mf, fifo=True)

    # 2. Baseline-MinRisk (two solves), 3. ED_cap, 4. PatientAware: one shared model and solver object
    mdl = M.build_model(d, plans, fifo=False)
    opt = M.new_solver(solver, time_limit, threads)
    sa = M.solve(opt, mdl, "cost")
    if sa["has_solution"]:
        C_star = sa["objective"]
        sb = M.solve(opt, mdl, "ED", cost_cap=C_star * (1 + TIE_TOL))
        s_min = dict(sb, solve_s=sa["solve_s"] + sb["solve_s"],
                     status=sb["status"] if sa["status"] == "optimal" else f"stage a {sa['status']}; stage b {sb['status']}")
    else:
        C_star, s_min = np.nan, sa
    r_min = record("Baseline-MinRisk", s_min, mdl, extra={"C* (stage a)": C_star, "stage a gap": sa["mip_gap"]})
    ED_minrisk = r_min.get("E_D", np.nan)

    s_cap = M.solve(opt, mdl, "ED")
    r_cap = record("ED_cap (diagnostic)", s_cap, mdl)
    ED_cap = r_cap.get("E_D", np.nan)

    pa_rows = []
    if np.isfinite(ED_minrisk) and np.isfinite(ED_cap):
        for step in range(1, N_EPS + 1):
            eps = ED_minrisk - (step - 1) / (N_EPS - 1) * (ED_minrisk - ED_cap)
            pa_rows.append(record(pa_label(eps), M.solve(opt, mdl, "cost", eps=eps), mdl, eps=eps, eps_step=step))
    for row in rows:
        row["ED_minrisk"], row["ED_cap"] = ED_minrisk, ED_cap

    # PatientAware vs Baseline-MinRisk
    for row in pa_rows:
        if "cost" in row and "cost" in r_min:
            extra, red = row["cost"] - r_min["cost"], r_min["E_D"] - row["E_D"]
            row.update({"extra cost vs MinRisk (USD)": extra, "extra cost vs MinRisk (%)": 100 * extra / r_min["cost"],
                        "E_D reduction vs MinRisk": red, "E_D reduction vs MinRisk (%)": 100 * red / r_min["E_D"],
                        "cost per protected patient (USD)": extra / red if red > 1e-9 else np.nan})

    # instance-level checks
    if "cost" in r_fifo and "cost" in r_min:
        chk("Baseline-MinRisk", "Baseline-MinRisk cost <= Baseline-FIFO cost", r_min["cost"] <= r_fifo["cost"] + 1e-6,
            f"{r_min['cost']:,.2f} vs {r_fifo['cost']:,.2f}")
    if pa_rows and "cost" in pa_rows[0] and "cost" in r_min:
        top = pa_rows[0]
        chk(top["model"], "PatientAware at eps=ED_minrisk has the cost of Baseline-MinRisk (limit inactive)",
            abs(top["cost"] - r_min["cost"]) <= 1e-6 * max(1.0, r_min["cost"]), f"{top['cost']:,.2f} vs {r_min['cost']:,.2f}")
    solved = [r_ for r_ in pa_rows if "cost" in r_]
    if len(solved) >= 2:
        chk("PatientAware", "along the eps levels E_D falls (does not rise)",
            all(b["E_D"] <= a["E_D"] + 1e-6 for a, b in zip(solved, solved[1:])), " > ".join(f"{x['E_D']:.4f}" for x in solved))
        chk("PatientAware", "along the eps levels cost rises (does not fall)",
            all(b["cost"] >= a["cost"] - 1e-6 * max(1.0, a["cost"]) for a, b in zip(solved, solved[1:])),
            " < ".join(f"{x['cost']:,.0f}" for x in solved))
    if len(pa_rows) < N_EPS:
        chk("PatientAware", "all eps levels solved", f"LOGGED (no eps levels: ED_minrisk or ED_cap missing)")

    out = dict(rows=rows, plans=pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame(),
               checks=checks, seconds=time.time() - t_start)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------------------------
METRICS = ["extra cost vs MinRisk (USD)", "extra cost vs MinRisk (%)", "E_D reduction vs MinRisk",
           "E_D reduction vs MinRisk (%)", "E_U", "cost per protected patient (USD)"]
FMT = {"extra cost vs MinRisk (USD)": "{:,.0f}", "extra cost vs MinRisk (%)": "{:.2f}%", "E_D reduction vs MinRisk": "{:.3f}",
       "E_D reduction vs MinRisk (%)": "{:.1f}%", "E_U": "{:.3f}", "cost per protected patient (USD)": "{:,.0f}"}


def rep_stats(df, group_cols):
    """Average over everything except group_cols and rep, then mean / min / max over the replications.
    Cost per protected patient of a replication = its mean extra cost / its mean E_D reduction (ratio of means)."""
    base = [c for c in METRICS if c != "cost per protected patient (USD)"]
    per_rep = df.groupby(group_cols + ["rep"])[base].mean()
    red = per_rep["E_D reduction vs MinRisk"]
    per_rep["cost per protected patient (USD)"] = (per_rep["extra cost vs MinRisk (USD)"] / red).where(red > 1e-9)
    g = per_rep.groupby(level=list(range(len(group_cols))))
    return g.mean(), g.min(), g.max(), g.count()["E_U"]


def fmt_cell(m, lo, hi, f):
    return "–" if pd.isna(m) else f"{f.format(m)} [{f.format(lo)} to {f.format(hi)}]"


def to_tables(mean, lo, hi, n):
    txt = pd.DataFrame({c: [fmt_cell(a, b, c_, FMT[c]) for a, b, c_ in zip(mean[c], lo[c], hi[c])] for c in METRICS},
                       index=mean.index)
    txt["replications"] = n
    csv = pd.concat({"mean": mean[METRICS], "min": lo[METRICS], "max": hi[METRICS]}, axis=1)
    csv.columns = [f"{m} ({s})" for s, m in csv.columns]
    csv["replications"] = n
    return txt, csv


def make_tables(ALL, out_dir):
    mid = ALL[ALL.eps_step == MIDDLE_STEP]
    out = {}
    # Table 1: main effects
    t_txt, t_csv = [], []
    for factor, label in [("ND", "ND"), ("N", "Demand N"), ("layout", "Capacity layout"), ("urgent_share", "Urgent share")]:
        txt, csv = to_tables(*rep_stats(mid, [factor]))
        idx = [f"{label} = {f'{int(round(v * 100))}%' if factor == 'urgent_share' else v}"
               + (f" {I.LAYOUTS[v]}" if factor == "layout" else "") for v in txt.index]
        txt.index = csv.index = idx
        t_txt.append(txt); t_csv.append(csv)
    out["table1_main_effects"] = (pd.concat(t_txt), pd.concat(t_csv))
    # Table 2: by capacity layout x (ND, demand), urgent share 30%
    txt, csv = to_tables(*rep_stats(mid[np.isclose(mid.urgent_share, 0.30)], ["layout", "ND", "N"]))
    out["table2_by_layout"] = (txt, csv)
    # Table 4 (appendix): all 48 combinations
    txt, csv = to_tables(*rep_stats(mid, ["ND", "N", "layout", "urgent_share"]))
    out["table4_appendix_48_combinations"] = (txt, csv)
    # Table 3: trade-off curve of the base case
    bc = ALL[(ALL.ND == BASE_CASE["ND"]) & (ALL.N == BASE_CASE["N"]) & (ALL.layout == BASE_CASE["layout"])
             & np.isclose(ALL.urgent_share, BASE_CASE["share"])]
    rows = []
    for model, sub in [("Baseline-FIFO", bc[bc.model == "Baseline-FIFO"]), ("Baseline-MinRisk", bc[bc.model == "Baseline-MinRisk"])] + \
                      [(None, bc[bc.eps_step == s]) for s in range(1, N_EPS + 1)]:
        if not len(sub):
            continue
        lab = model or f"PatientAware (mean eps={sub.eps.mean():.3f})"
        rows.append({"model": lab, "eps_step": sub.eps_step.iloc[0] if model is None else np.nan,
                     "replications": int(sub.cost.notna().sum()), "eps (mean)": sub.eps.mean(),
                     "cost (mean)": sub.cost.mean(), "cost (min)": sub.cost.min(), "cost (max)": sub.cost.max(),
                     "E_D (mean)": sub.E_D.mean(), "E_D (min)": sub.E_D.min(), "E_D (max)": sub.E_D.max(),
                     "E_U (mean)": sub.E_U.mean()})
    t3 = pd.DataFrame(rows)
    pa = t3[t3.eps_step.notna()].copy()
    slope = (pa["cost (mean)"].diff() / -pa["E_D (mean)"].diff())
    t3["slope vs previous level (USD per expected ineligible patient avoided)"] = slope.reindex(t3.index)
    out["table3_tradeoff_base_case"] = (t3, t3)
    for nm, (txt, csv) in out.items():
        csv.to_csv(out_dir / f"{nm}.csv", index=nm != "table3_tradeoff_base_case")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------
COLS = ["instance", "ND", "N", "layout", "lines", "urgent_share", "rep", "model", "eps", "eps_step", "ED_minrisk", "ED_cap",
        "status", "objective", "best_bound", "mip_gap", "cost", "E_D", "E_U",
        "extra cost vs MinRisk (USD)", "extra cost vs MinRisk (%)", "E_D reduction vs MinRisk", "E_D reduction vs MinRisk (%)",
        "cost per protected patient (USD)", "mean wait urgent", "max wait urgent", "mean wait regular", "max wait regular",
        "non-nearest assignments", "max TRT", "solve_s", "C* (stage a)", "stage a gap", "n_plans", "n_fifo_constraints",
        "NT", "n_urgent"]


def collect(keys, out_dir):
    results = []
    for k in keys:
        cache = out_dir / "_cache" / f"{instance_name(k)}.pkl"
        if cache.exists():
            with open(cache, "rb") as fh:
                results.append(pickle.load(fh))
    return results


def report(results, out_dir, n_expected):
    ALL = pd.DataFrame([row for res in results for row in res["rows"]]).reindex(columns=COLS)
    ALL.to_csv(out_dir / "all_runs.csv", index=False)
    plans = [res["plans"] for res in results if len(res["plans"])]
    if plans:
        pd.concat(plans, ignore_index=True).to_csv(out_dir / "all_plans.csv.gz", index=False, compression="gzip")
    CHK = pd.DataFrame([c for res in results for c in res["checks"]])
    CHK.to_csv(out_dir / "checks.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 400)
    print(f"\n{'=' * 110}\n{len(results)} of {n_expected} instances in the results ({len(ALL)} rows) -> {out_dir}")
    print("\nSOLVE STATUS")
    print(ALL.assign(level=ALL.eps_step.fillna(0).astype(int).astype(str).where(ALL.eps_step.notna(), ALL.model))
          .groupby(["level", "status"]).size().unstack(fill_value=0).to_string())
    print("\nsolve time (s) by model type and N:")
    typ = ALL.model.where(~ALL.model.str.startswith("PatientAware"), "PatientAware (any eps)")
    print(ALL.assign(type=typ).groupby(["type", "N"]).solve_s.describe()[["mean", "50%", "max"]].round(2).to_string())
    tl = ALL[ALL.status.str.startswith("time limit") | ALL.status.str.startswith("stage a")]
    if len(tl):
        print(f"\n{len(tl)} solves stopped at the time limit:")
        print(tl[["instance", "model", "status", "mip_gap", "solve_s"]].to_string(index=False))
    if len(ALL) and ALL.eps_step.notna().any():
        tables = make_tables(ALL, out_dir)
        titles = {"table1_main_effects": f"TABLE 1 - main effects at the middle eps level (step {MIDDLE_STEP} of {N_EPS}); "
                                         "mean [min to max] over replications",
                  "table2_by_layout": "TABLE 2 - by capacity layout x (ND, demand), urgent 30%, middle eps level",
                  "table3_tradeoff_base_case": f"TABLE 3 - trade-off curve, base case ND {BASE_CASE['ND']}, N {BASE_CASE['N']}, "
                                               f"{BASE_CASE['layout']}, urgent 30% (means over replications)",
                  "table4_appendix_48_combinations": "TABLE 4 (appendix) - all 48 combinations, middle eps level"}
        for nm, (txt, _) in tables.items():
            print(f"\n{titles[nm]}")
            print(txt.to_string(index=nm != "table3_tradeoff_base_case", float_format=lambda x: f"{x:,.4f}"))
    print("\nCHECKS (checks.csv)")
    res_type = CHK.result.where(~CHK.result.str.startswith("LOGGED"), "LOGGED")
    print(CHK.assign(r=res_type).groupby(["check", "r"]).size().unstack(fill_value=0).to_string())
    bad = CHK[CHK.result == "FAIL"]
    print(f"\n{len(bad)} FAIL, {int((res_type == 'LOGGED').sum())} LOGGED, {int((CHK.result == 'PASS').sum())} PASS")
    if len(bad):
        print(bad.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", choices=["highs", "gurobi"], default="highs")
    ap.add_argument("--workers", type=int, default=1, help="instances solved in parallel (default 1)")
    ap.add_argument("--threads", type=int, help="solver threads per solve (default: solver default with 1 worker, 1 otherwise)")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="seconds per solve (default 120)")
    ap.add_argument("--out", help="output folder (default results_doe/, results_doe_smoke/ with --smoke)")
    ap.add_argument("--smoke", action="store_true", help="2 instances only (base case r1 and the largest, ND52 N60 C4 30%% r1)")
    ap.add_argument("--tables-only", action="store_true", help="no solving: rebuild outputs from the cache")
    ap.add_argument("--rerun", action="store_true", help="ignore and overwrite the cache")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (ROOT / "results_doe_smoke" if args.smoke else OUT_DIR)
    (out_dir / "_cache").mkdir(parents=True, exist_ok=True)
    I.write_instances(out_dir)                       # the 20 patient lists (kept if they already exist)
    keys = SMOKE_KEYS if args.smoke else KEYS
    if args.tables_only:
        report(collect(keys, out_dir), out_dir, len(keys))
        return
    if args.rerun:
        for k in keys:
            (out_dir / "_cache" / f"{instance_name(k)}.pkl").unlink(missing_ok=True)
    threads = args.threads if args.threads is not None else (0 if args.workers == 1 else 1)
    jobs = [(k, str(out_dir), args.solver, args.time_limit, threads) for k in keys]
    n_cached = sum((out_dir / "_cache" / f"{instance_name(k)}.pkl").exists() for k in keys)
    print(f"{len(keys)} instances x 9 solves, solver {args.solver}, {args.workers} worker(s), threads per solve "
          f"{threads or 'solver default'}, time limit {args.time_limit:.0f}s; {n_cached} instances already cached -> {out_dir}",
          flush=True)
    t0 = time.time()
    results = []

    def progress(i, res):
        st = [r["status"] for r in res["rows"]]
        bad = [f"{r['model']}: {r['status']} (gap {r['mip_gap']:.2e})" for r in res["rows"] if r["status"] != "optimal"]
        print(f"[{i:>3}/{len(jobs)}] {res['rows'][0]['instance']:<24s} {res['seconds']:7.1f}s  "
              f"{st.count('optimal')}/{len(st)} results optimal (9 solves)  elapsed {time.time() - t0:,.0f}s"
              + (f"  NOT OPTIMAL: {'; '.join(bad)}" if bad else ""), flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(run_instance, jobs), start=1):
                results.append(res)
                progress(i, res)
    else:
        for i, job in enumerate(jobs, start=1):
            res = run_instance(job)
            results.append(res)
            progress(i, res)
    print(f"done in {time.time() - t0:,.0f}s")
    report(collect(keys, out_dir), out_dir, len(keys))


if __name__ == "__main__":
    main()
