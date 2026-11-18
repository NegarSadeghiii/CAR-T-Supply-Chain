# Pilot: continuous patient arrivals

Code for a pilot of the compact CAR-T scheduling model with patients arriving continuously over 30 days.
The model is the same as in `CART_doe_main.ipynb`:

- **Baseline**: minimise total cost, with the FIFO rule at every facility.
- **Patient-aware**: minimise total cost + λ_D · E_D (λ_D = 150,000, λ_U = 0), no FIFO.

The pilot has **not** been run. Only a 5-patient smoke test was run, with HiGHS (all checks PASS).

## Files

| File | Content |
|---|---|
| `model.py` | Compact model: plan variable `X[p, k]`, `k = (m, j1, j2, t)`; one plan per patient; time-indexed capacity; Baseline with FIFO in two forms (`bigM`, `timeindexed`); Patient-aware; solution checks |
| `instances.py` | Network data from `../CART_instance_30patients_cap4.xlsx` and the continuous-arrival instance generator |
| `run_pilot.py` | Configuration block, the four pilot settings, solving, caching, results, plots |
| `results/instances/` | The two pilot instances, already generated (seed 1): 74 patients (1,000/yr) and 147 patients (2,000/yr). `run_pilot.py` reuses these files, so you get exactly these patients whatever your numpy version. Delete them to regenerate. |

## Pilot settings

ND = 41, arrivals on days 1..30 (horizon NT = 30 + 41 + 2 = 73), each patient urgent with probability 0.30, seed 1,
one replication. Solves per setting: Baseline (FIFO) and Patient-aware at λ_D = 150,000.

| Setting | Demand (/yr) | Lines (Philadelphia, Houston, LA) | ρ overall, expected | ρ overall, this instance |
|---|---|---|---|---|
| P1 | 1,000 | (10, 10, 10) | 0.64 | 0.58 |
| P2 | 1,000 | (4, 10, 10) | 0.80 | 0.72 |
| P3 | 2,000 | (31, 4, 10) | 0.85 | 0.76 |
| P4 | 2,000 | (31, 4, 4) | 0.98 | 0.88 |

ρ = patients per day × T_MFE ÷ lines. The per-facility ρ (patients whose nearest facility it is) is printed for every
setting and written to `summary_runs.csv`.

## Step by step (Anaconda)

### 1. Create the environment

Open the Anaconda Prompt and go to the repository folder (the one that contains `CART_instance_30patients_cap4.xlsx`):

```bash
conda create -n cart-pilot -c conda-forge python=3.11 pyomo pandas numpy openpyxl matplotlib
```

```bash
conda activate cart-pilot
```

Optional, only needed for `SOLVER = "highs"` or the smoke test:

```bash
pip install highspy
```

The smoke test here used Pyomo 6.10.1, numpy 2.5.3, pandas 3.0.5, openpyxl 3.1.5, matplotlib 3.11.2, highspy.

### 2. Point Pyomo to GAMS

Pyomo calls the `gams` executable (shell interface: it writes a `.gms` file and runs it). Either:

- **Add the GAMS folder to PATH** for this environment (Windows example; use your GAMS version folder):

  ```bash
  conda env config vars set PATH="C:\GAMS\48;%PATH%"
  ```

  then `conda deactivate` and `conda activate cart-pilot` again; or
- **Set the full path** in the configuration block of `run_pilot.py`:
  `GAMS_EXECUTABLE = r"C:\GAMS\48\gams.exe"`.

Check that Pyomo finds GAMS:

```bash
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('gams').available(exception_flag=False))"
```

This must print `True`. Your GAMS license must include the subsolver you choose (CPLEX by default).

### 3. Choose the solver

All choices are in the configuration block at the top of `run_pilot.py`:

| Setting | Default | Meaning |
|---|---|---|
| `SOLVER` | `"gams"` | `"gams"` or `"highs"` |
| `GAMS_SUBSOLVER` | `"cplex"` | `"cplex"`, `"gurobi"` or `"cbc"` (passed as `solver=` to Pyomo's GAMS interface) |
| `GAMS_EXECUTABLE` | `None` | full path to `gams` if it is not on PATH |
| `TIME_LIMIT_S` | `1800` | per solve; GAMS `option reslim` |
| `MIP_GAP` | `0.0` | relative gap; GAMS `option optcr` (`optca` is set to 0) |
| `THREADS` | `0` | 0 = solver default (option not passed); otherwise GAMS `option threads` |
| `FIFO_FORM` | `"timeindexed"` | Baseline FIFO form: `"timeindexed"` or `"bigM"` |
| `KEEP_GAMS_FILES` | `False` | `True` keeps the `.gms`/`.lst` files for debugging |

### 4. Check model sizes first (no solving)

```bash
python pilot_continuous/run_pilot.py --count-only
```

### 5. Optional: smoke test (5 patients, HiGHS, a few seconds)

```bash
python pilot_continuous/run_pilot.py --smoke
```

It must end with `SMOKE OVERALL: PASS`. Its output goes to `pilot_continuous/results_smoke/` (not committed).

### 6. Run the pilot

Start with the smallest setting to see build and solve times on your machine:

```bash
python pilot_continuous/run_pilot.py --settings P1
```

Then the rest (P1 is taken from the cache):

```bash
python pilot_continuous/run_pilot.py
```

### 7. Resume an interrupted run

Each solve is saved in `results/cache/<setting>__<model>.json` as soon as it finishes; the per-patient plan goes to
`results/plans/`. Run the same command again: finished solves are loaded from the cache and only the missing ones are
solved. To solve something again, delete its cache file, or use `--rerun` to ignore the whole cache.
A solve that was interrupted in the middle starts again from scratch (only finished solves are cached).

## Model sizes (from `--count-only`, before presolve)

| Setting | Patients | Variables | FIFO `bigM`: constraints / nonzeros | FIFO `timeindexed`: constraints / nonzeros |
|---|---|---|---|---|
| P1, P2 | 74 | 20,308 | 7,744 / 4.3 million | 110,218 / 7.4 million |
| P3, P4 | 147 | 40,396 | 29,802 / 16.4 million | 406,129 / 27.0 million |

The one-plan and capacity constraints add only 247 (P1, P2) or 325 (P3, P4) rows; FIFO dominates the size of the
Baseline. The Patient-aware model has no FIFO and is small.

Expect the Baseline of P3 and P4 to need a lot of memory and time **before** the solver starts: Pyomo builds the model
in Python and the GAMS interface writes it as a text `.gms` file. Run P1 first and watch memory.

## FIFO forms

Order: p before q if a_p < a_q, or a_p = a_q and p has the smaller index. z_pm = Σ X over p's plans at m.

- `bigM`: S_p ≤ S_q + NT·(2 − z_pm − z_qm) with S_p = Σ t·X_pk.
- `timeindexed`: Σ_{k of q at m, t ≤ τ} X_qk ≤ Σ_{k of p at m, t ≤ τ} X_pk + (1 − z_pm), written equivalently as
  Σ_{k of q at m, t ≤ τ} X_qk + Σ_{k of p at m, t > τ} X_pk ≤ 1.

Only constraints that can bind are generated; dropping the others does not change the feasible set:
- A pair (p, q, m) is generated only if both patients have admissible plans at m and p's latest start at m is after
  q's earliest start at m. Otherwise S_p ≤ S_q holds for every choice.
- `timeindexed`: τ runs over q's admissible start days at m that are before p's latest start at m. For any other τ the
  constraint is empty or follows from the constraint at q's own start day.

The two forms give the same Baseline optimum (checked in the smoke test).

## Instance generator

On each day d = 1..30 the number of new patients is Poisson(annual demand / 365). For each new patient, in turn:
- region: East 60%, Central 13%, West 27%;
- hospital within the region, in proportion to the Wan et al. (2026) demand weights: Boston 24, New York 29,
  Philadelphia 14, Baltimore 23, Durham 12, Chicago 14, Houston 14, Denver 30, Los Angeles 37, Palo Alto 28;
- urgent with probability 0.30.

All of this is drawn from one `numpy.random.default_rng(seed)` stream. Patients are numbered in arrival order.
The same seed and demand give the same patients, so P1 and P2 share one instance, and P3 and P4 share the other.

## Outputs (`pilot_continuous/results/`)

| File | Content |
|---|---|
| `summary_runs.csv` | One row per setting and model: N, ρ overall and per facility (expected and realised), status, objective, best bound, MIP gap, solve and build time, variables, constraints, FIFO constraints, total cost, E_D, E_U |
| `summary_settings.csv` | Per setting, for all patients and for patients collected on day 8 or later: E_D of both models, difference, relative reduction, extra cost (USD and %), E_U, mean and max wait of urgent and regular patients |
| `checks.csv` | Every check, PASS/FAIL |
| `plans/<setting>__<model>.csv` | Per-patient plan |
| `plans/<setting>__<model>__waiting_by_day.csv` | Patients waiting at each facility on each day |
| `figures/waiting_<setting>.png` | Patients waiting per facility per day, Baseline vs Patient-aware |
| `cache/` | One JSON per finished solve (resume) |
| `instances/` | Patient lists |

Checks run automatically after each solve:
- every patient has exactly one admissible plan;
- capacity is respected every day;
- TRT ≤ ND;
- FIFO holds in the Baseline;
- E_D recomputed from the plans equals the solver value (1e-6).

A setting with no feasible solution is reported as INFEASIBLE and the run continues with the next one.
Because only the Baseline has FIFO, the Patient-aware extra cost can be negative; that is not an error.
