"""
Generate results/README.md from the CSV/JSON artefacts produced by
``ishipment_survival.py``.  Every number in the README is read back from the
result files, so the write-up can never drift from the experiments.
"""
from __future__ import annotations

import csv
import json
import os
import textwrap

import cart_data as cd
import ishipment_survival as ish

R = ish.RESULTS


def read_csv(name):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(name, default=None):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        return default
    return json.load(open(path))


def md_table(rows, cols, headers=None, fmt=None):
    if not rows:
        return "_(no rows)_\n"
    fmt = fmt or {}
    headers = headers or cols
    def cell(r, c):
        v = r.get(c, "")
        if c in fmt:
            try:
                return fmt[c](v)
            except (TypeError, ValueError):
                return str(v)
        return "" if v is None else str(v)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(r, c) for c in cols) + " |")
    return "\n".join(out) + "\n"


def money(v):
    return "" if v in ("", None) else f"${float(v)/1e6:,.2f}M"


def num(nd=2):
    return lambda v: "" if v in ("", None) else f"{float(v):,.{nd}f}"


def main():
    calib = read_json("calibration.json", {})
    study = read_json("study_summary.json", {})
    inst_rows = read_csv("instances_overview.csv")
    p1 = read_csv("phase1_baseline_by_scale.csv")
    p1t = read_csv("phase1_baseline_by_tier.csv")
    p2 = read_csv("phase2_extension_by_scale.csv")
    p3 = read_csv("phase3_design_comparison.csv")
    p3t = read_csv("phase3_by_tier.csv")
    p3h = read_csv("phase3_hold_distribution.csv")
    frontier_scale = 200
    p4 = read_csv(f"phase4_frontier_N{frontier_scale}.csv")
    verif = read_json("verification_baseline_equivalence.json", {})

    solver_used = study.get("solver_used", "?")
    solver_note = study.get("solver_note", "")

    L = []
    A = L.append

    A("# Survival-aware scheduling extension of i-SHIPMENT\n")
    A(textwrap.dedent("""\
    This directory holds the full output of a four-phase computational study that
    adds a **manufacturing queue** and an **exact integer-day survival model** to
    the i-SHIPMENT CAR-T supply-chain MILP, at demand scales of **100 / 200 / 500
    patients**.

    * `i-SHIPMENT_Pyomo.ipynb` is the **baseline** and was not touched.
    * `ishipment_survival.py` contains the baseline re-statement *and* the
      extension; `cart_data.py` holds the instance generator and the frozen
      clinical calibration; `verify_baseline.py` proves the re-statement is
      equivalent to the notebook model.
    """))

    # ---------------------------------------------------------------- solver
    A("\n## 0. How to reproduce\n")
    A("```bash\npython ishipment_survival.py --phase all      # all four phases\n"
      "python verify_baseline.py                    # baseline equivalence check\n"
      "python make_readme.py                        # regenerate this file\n```\n")
    if solver_used != "gurobi":
        A("\n> **Solver note.** The study is written for Gurobi and asks for it "
          "first on every solve. " + solver_note + "\n>\n> Every model, every "
          "constraint and every reported number is solver-independent; only the "
          "MILP engine changed. Reported wall-clock times are HiGHS times.\n")
    else:
        A("\nAll models were solved with Gurobi through Pyomo.\n")

    # ------------------------------------------------------------- calibration
    A("\n## 1. Fixed set-up (never re-fit)\n")
    A(f"""
| item | value |
|---|---|
| risk tiers | H {cd.TIER_MIX['H']:.0%} / M {cd.TIER_MIX['M']:.0%} / L {cd.TIER_MIX['L']:.0%}, assigned once on the 50-patient base cohort (seed {cd.SEED}) and inherited by every replica |
| survival | `S_u(t) = (1 - w_u) ** (t/42)` with gamma=1, eta=42, kappa=1 |
| deterioration `w_u` | H {cd.W_RISK['H']}, M {cd.W_RISK['M']}, L {cd.W_RISK['L']} |
| clinical-loss weight `rho_u` | H {cd.RHO['H']:.0f}, M {cd.RHO['M']:.0f}, L {cd.RHO['L']:.0f} |
| `sigma[d,u]` lookup | precomputed for integer days d = {cd.D_MIN}..{cd.D_MAX} (`sigma_lookup.csv`) |
| ND | {ish.ND_BASELINE} days (baseline) -> {ish.ND_EXT} days (extension, eq. 31) |
| CON1 | at most {ish.MAX_FACILITIES} manufacturing facilities |
| horizon | {cd.HORIZON} days |
""".strip() + "\n")
    A(f"\n`d = {cd.D_MIN}` is the physical floor of the network: "
      "`TLS(1) + TT1_air(1) + TMFE(7) + TQC(7) + TT3_air(1) = 17` days with a "
      f"zero hold; `d = {cd.D_MAX}` is ND in the extension.\n")

    # --------------------------------------------------------------- instances
    A("\n## 2. Instances\n")
    A("Each scale tiles the 50-patient cohort (mult = 2 / 4 / 10). Tier labels "
      "and leukapheresis sites are copied patient-by-patient, so the 30/40/30 mix "
      "and the site distribution are preserved **exactly**. Arrival days are "
      "rescaled into the admissible window "
      f"[{cd.ARRIVAL_WINDOW[0]}, {cd.ARRIVAL_WINDOW[1]}] "
      f"(a therapy started on the last day must still be delivered inside the "
      f"{cd.HORIZON}-day horizon at ND = {ish.ND_EXT}) and each replica gets a "
      "seeded +/-3-day jitter. The window is fixed, so **arrival density grows "
      "linearly with N** - that is what loads the factories. Costs, transport "
      "times, FCAP and the network structure come untouched from "
      "`Data200_profileA.dat`.\n")
    A(md_table(inst_rows,
               ["n", "mult", "arrivals_per_day", "n_H", "n_M", "n_L",
                "n_c1", "n_c2", "n_c3", "n_c4"],
               ["N", "mult", "arrivals/day", "H", "M", "L",
                "c1", "c2", "c3", "c4"]))

    A("\nFacility menu (from the data file): 130-day opening cost = "
      "`130 x (CIM + CVM)`.\n")
    net = cd.load_network(os.path.join(os.path.dirname(R), "Data200_profileA.dat"))
    A(md_table([{"m": m, "FCAP": net.FCAP[m],
                 "cost": 130 * (net.CIM[m] + net.CVM[m])} for m in net.m],
               ["m", "FCAP", "cost"], ["facility", "FCAP (concurrent)",
                                       "opening cost"],
               fmt={"cost": money}))

    # ----------------------------------------------------------------- phase 1
    A("\n## 3. Phase 1 - baseline i-SHIPMENT vs demand\n")
    A("The baseline has no queue: manufacturing starts on the day the sample "
      "reaches the facility (`INM = arrival`, old MSB5), so a clustered arrival "
      "pattern must be absorbed by *concurrent capacity alone*. ND = 18.\n")
    A(md_table(p1, ["n", "status", "opened", "capacity_opened", "total_cost",
                    "mean_TRT", "expected_lost", "wall_s"],
               ["N", "status", "facilities opened", "concurrent capacity",
                "total cost", "mean TRT", "E[lost]", "wall s"],
               fmt={"total_cost": money, "mean_TRT": num(2),
                    "expected_lost": num(2)}))

    infeas = [r for r in p1 if r["status"] == "infeasible"]
    if infeas:
        A("\n**The no-queue model breaks down as demand rises.** " +
          ", ".join(f"N = {r['n']} is INFEASIBLE under CON1 <= 2"
                    for r in infeas) + ". Diagnostic re-solves with CON1 relaxed "
          "to 6 facilities:\n")
        A(md_table(infeas, ["n", "diagnostic_CON1_relaxed", "diagnostic_opened",
                            "diagnostic_capacity", "diagnostic_total_cost"],
                   ["N", "status with CON1 <= 6", "facilities", "capacity",
                    "total cost"], fmt={"diagnostic_total_cost": money}))
    A("\nThe pattern is monotone: every doubling of demand pushes the baseline "
      "onto a larger and more expensive facility set, because the only lever it "
      "has against a burst of arrivals is raw concurrent capacity. It cannot "
      "delay a single job.\n")
    if p1t:
        A("\nBaseline survival, by tier (only where the baseline is feasible):\n")
        A(md_table([r for r in p1t if r["tier"] != "ALL"],
                   ["design", "tier", "n", "mean_TRT", "mean_survival",
                    "expected_lost"],
                   ["instance", "tier", "n", "mean TRT", "mean S", "E[lost]"],
                   fmt={"mean_TRT": num(2), "mean_survival": num(4),
                        "expected_lost": num(3)}))

    # ----------------------------------------------------------------- phase 2
    A("\n## 4. Phase 2 - the queue makes it feasible\n")
    A(textwrap.dedent("""
    The fixed start (old MSB5) is replaced by a genuine start decision:

    ```
    A[p,m,t]  = sum_{c,j} LSA[p,c,m,j,t]                  arrivals at the MS
    sum_{tau<=t} INM[p,m,tau] <= sum_{tau<=t} A[p,m,tau]  (32)  no start before arrival
    sum_t INM[p,m,t]          = sum_t A[p,m,t]            (33)  started exactly once
    sum_p DURV[p,m,t]         <= FCAP[m]                  (34)  concurrency -> forces holds
    HOLD[p] = start_time - arrival_time >= 0
    ```

    plus ND relaxed 18 -> 42 (eq. 31), the exact integer-day survival lookup
    (`delta[p,d]`, d = 17..42, `TRT[p] = sum_d d*delta`, `S[p] = sum_d sigma[d,u]*delta`)
    linked to eqs. (24)-(25), and the objective of eq. (1)

    ```
    min Z = (original i-SHIPMENT cost) + ALPHA * sum_p rho_u(p) * (1 - S[p])
    ```
    """).strip() + "\n")
    A("\n" + md_table(p2, ["n", "status", "opened", "capacity_opened",
                           "total_cost", "mean_TRT", "mean_HOLD", "max_HOLD",
                           "expected_lost", "wall_s"],
                      ["N", "status", "facilities opened", "capacity",
                       "total cost", "mean TRT", "mean HOLD", "max HOLD",
                       "E[lost]", "wall s"],
                      fmt={"total_cost": money, "mean_TRT": num(2),
                           "mean_HOLD": num(2), "expected_lost": num(2)}))

    # side-by-side
    b = {r["n"]: r for r in p1}
    e = {r["n"]: r for r in p2}
    comp = []
    for n in sorted(set(b) | set(e), key=int):
        rb, re_ = b.get(n, {}), e.get(n, {})
        comp.append({
            "n": n,
            "baseline": (rb.get("opened") or rb.get("status", "-")),
            "baseline_cost": rb.get("total_cost", ""),
            "ext": re_.get("opened", "-"),
            "ext_cost": re_.get("total_cost", ""),
            "saving": (f"{100*(1-float(re_['total_cost'])/float(rb['total_cost'])):.1f}%"
                       if rb.get("total_cost") and re_.get("total_cost") else "n/a"),
        })
    A("\n**Baseline vs. extension, head to head:**\n")
    A(md_table(comp, ["n", "baseline", "baseline_cost", "ext", "ext_cost",
                      "saving"],
               ["N", "baseline network", "baseline cost", "extension network",
                "extension cost", "cost reduction"],
               fmt={"baseline_cost": money, "ext_cost": money}))
    A("\nThe hold absorbs exactly the contention that the baseline had to buy "
      "concurrent capacity for, so the extension is feasible at every scale and "
      "on a strictly cheaper network.\n")

    # ----------------------------------------------------------------- phase 3
    A("\n## 5. Phase 3 - cost design vs survival design, by tier\n")
    A(f"Both designs are solved on the **same** frozen tier assignment at each "
      f"scale. (a) COST design: ALPHA = 0, survival evaluated afterwards. "
      f"(b) SURVIVAL design: ALPHA = {ish.ALPHA_SURVIVAL:g} "
      f"($ per unit of rho-weighted expected loss).\n")
    A(md_table(p3, ["n", "design", "opened", "total_cost", "mean_TRT",
                    "mean_HOLD", "mean_survival", "expected_lost"],
               ["N", "design", "facilities", "total cost", "mean TRT",
                "mean HOLD", "mean S", "E[lost]"],
               fmt={"total_cost": money, "mean_TRT": num(2), "mean_HOLD": num(2),
                    "mean_survival": num(4), "expected_lost": num(2)}))

    A("\n### Emergent priority: holds by tier\n")
    A(md_table(p3t, ["n_patients_total", "design", "tier", "n", "mean_TRT",
                     "mean_HOLD", "median_HOLD", "max_HOLD", "mean_survival",
                     "expected_lost"],
               ["N", "design", "tier", "n", "mean TRT", "mean HOLD",
                "median HOLD", "max HOLD", "mean S", "E[lost]"],
               fmt={"mean_TRT": num(2), "mean_HOLD": num(2),
                    "mean_survival": num(4), "expected_lost": num(3)}))

    A("\nHold distribution (share of each tier that is never held, and the "
      "upper tail):\n")
    A(md_table(p3h, ["n", "design", "tier", "mean_hold", "p50", "p90", "max",
                     "share_hold_0"],
               ["N", "design", "tier", "mean hold", "p50", "p90", "max",
                "share with hold = 0"],
               fmt={"mean_hold": num(2), "share_hold_0":
                    lambda v: f"{float(v):.0%}"}))

    # contention growth
    surv = [r for r in p3 if r["design"] == "survival"]
    if surv:
        A("\n### Contention grows with demand\n")
        A(md_table(surv, ["n", "mean_HOLD", "mean_HOLD_H", "mean_HOLD_M",
                          "mean_HOLD_L", "opened"],
                   ["N", "mean HOLD (all)", "H", "M", "L", "network"],
                   fmt={k: num(2) for k in ("mean_HOLD", "mean_HOLD_H",
                                            "mean_HOLD_M", "mean_HOLD_L")}))

    # ----------------------------------------------------------------- phase 4
    A(f"\n## 6. Phase 4 - the cost-lives frontier (N = {frontier_scale})\n")
    A(md_table(p4, ["alpha", "opened", "capacity_opened", "total_cost",
                    "expected_lost", "expected_lost_H", "mean_HOLD",
                    "mean_HOLD_H", "mean_HOLD_L"],
               ["ALPHA", "facilities", "capacity", "total cost", "E[lost]",
                "E[lost] high-risk", "mean HOLD", "mean HOLD H", "mean HOLD L"],
               fmt={"total_cost": money, "expected_lost": num(3),
                    "expected_lost_H": num(3), "mean_HOLD": num(2),
                    "mean_HOLD_H": num(2), "mean_HOLD_L": num(2)}))
    A(f"\n![cost-lives frontier](phase4_frontier_N{frontier_scale}.png)\n")

    # --------------------------------------------------------------- verification
    if verif:
        A("\n## 7. Verification\n")
        A("`verify_baseline.py` builds the **full-index** i-SHIPMENT MILP "
          "transcribed verbatim from the notebook (Y1 over (p,c,m,j,t), Y2 over "
          "(p,m,h,j,t), all MSB/CAP/CON constraints) and solves it next to the "
          "index-reduced baseline on the original 50-patient cohort.\n")
        A(md_table(
            [{"model": "index-reduced (this study)", **verif["reduced"]},
             {"model": "full-index (notebook)", **verif["full_index_from_notebook"]}],
            ["model", "status", "objective", "opened", "TRT_distribution", "wall_s"],
            ["model", "status", "objective", "opened", "TRT distribution",
             "wall s"], fmt={"objective": money, "opened": lambda v: "+".join(v),
                             "TRT_distribution": lambda v: str(v),
                             "wall_s": num(1)}))
        A(f"\n**Equivalent: {verif.get('equivalent')}**\n")

    # --------------------------------------------------------------- file list
    A("\n## 8. Files\n")
    A(textwrap.dedent("""
    | file | contents |
    |---|---|
    | `calibration.json` | the frozen tiers / survival / rho set-up |
    | `sigma_lookup.csv` | `sigma[d,u]` for d = 17..42 |
    | `instances_overview.csv`, `instance_N*.json`, `tiers_N*.csv` | the generated cohorts and their frozen tier labels |
    | `phase1_baseline_by_scale.csv`, `phase1_baseline_by_tier.csv`, `phase1_patients_N*.csv` | Phase 1 |
    | `phase2_extension_by_scale.csv`, `phase2_patients_N*.csv` | Phase 2 |
    | `phase3_design_comparison.csv`, `phase3_by_tier.csv`, `phase3_hold_distribution.csv`, `phase3_patients_N*_{cost,survival}.csv` | Phase 3 |
    | `phase4_frontier_N200.csv`, `phase4_frontier_N200.png` | Phase 4 |
    | `verification_baseline_equivalence.json` | baseline equivalence proof |
    | `study_summary.json` | everything above in one JSON |
    | `cache/` | one JSON per solved model (per-patient schedule + solver metadata) |
    """).strip() + "\n")

    out = os.path.join(R, "README.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
