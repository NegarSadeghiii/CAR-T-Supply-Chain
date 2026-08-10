# Research-
Soft Due-Date Extension — CAR-T Supply Chain Optimisation
This notebook replicates the original Pyomo formulation (i-SHIPMENT_Pyomo.ipynb) and adds patient-specific soft due dates.

For each patient p a nonneg lateness variable LATE[p] and a penalty weight PEN[p] are introduced. The due date DUE[p] is a mutable parameter defaulting to the global max turnaround time ND. The model stays feasible even when due dates are tight — lateness is allowed but penalised.

Requirements: Data200_profileA.dat in the working directory, Gurobi (or any MILP solver)

---

## Survival-aware scheduling extension (queue + survival)

A second, independent extension lives in this repository. It keeps
`i-SHIPMENT_Pyomo.ipynb` as the untouched **baseline** and adds, in a copy:

* a real **manufacturing queue** — `INM` becomes a start decision constrained by
  arrival (eq. 32), started-once (eq. 33) and concurrent capacity (eq. 34), so a
  job can be **held**;
* an **exact integer-day survival model** — `delta[p,d]`, `d = 17..42`, with
  `S_u(t) = (1 - w_u)**(t/42)`;
* the objective `min Z = (original i-SHIPMENT cost) + ALPHA * Σ_p ρ_u(p)·(1 − S[p])`.

The study runs at demand scales of 100 / 200 / 500 patients. CON1 follows
i-SHIPMENT's own demand-dependent configuration - the centralised 2-facility
network at N = 100/200, relaxed to 3 facilities at N = 500 - so the baseline is
never compared against the extension on a tighter facility budget than the paper
would give it. At N = 500 the headline result is **capacity substitution**: the
baseline needs three plants, the extension serves the same 500 patients on two.

| file | role |
|---|---|
| `cart_data.py` | `.dat` parser, scaled instance generator, frozen clinical calibration |
| `ishipment_survival.py` | baseline restatement + queue/survival extension + all four phases |
| `verify_baseline.py` | solves the full-index notebook MILP next to the reduced form and checks they agree |
| `test_extension.py` | property tests for the queue, the capacity constraint and the survival lookup |
| `make_readme.py` | regenerates `results/README.md` from the result files |
| `results/` | per-scale CSVs, the cost–lives frontier PNG, and the full write-up |

```bash
python ishipment_survival.py --phase all
```

See **[`results/README.md`](results/README.md)** for the findings.
