# Main experimental design — ε-constraint model

This branch runs the main design (DOE) with the ε-constraint version of the compact CAR-T scheduling model.
Every model minimises total cost. Expected ineligible patients (E_D) is limited from above (E_D ≤ ε), and the
model is solved repeatedly from the cheapest solution to the safest one. E_U is never optimised but is reported
for every solution. There is no λ anywhere.

**The experiments have not been run yet.** Only a 2-instance smoke test was run with HiGHS; all checks PASS.

## Models per instance (9 solves)

| Model | What it solves | Solves |
|---|---|---|
| `Baseline-FIFO` | min cost, with FIFO at every facility | 1 |
| `Baseline-MinRisk` | (a) min cost → C\*; (b) min E_D s.t. cost ≤ C\*(1 + 1e-6). Its E_D is `ED_minrisk` | 2 |
| `ED_cap (diagnostic)` | min E_D subject to capacity only (no cost term): the safest achievable E_D | 1 |
| `PatientAware (eps=…)` | min cost s.t. E_D ≤ ε, no FIFO, for 5 values of ε evenly spaced from `ED_minrisk` down to `ED_cap` | 5 |

Each PatientAware row is labelled by its numeric ε, rounded to 3 decimals. The column `eps_step` (1 = `ED_minrisk`
… 5 = `ED_cap`) only records the position of a level, so that levels can be averaged across instances.
The constraint is written as E_D ≤ ε + 1e-7, a numerical slack because both endpoints equal the E_D of an
existing plan.

## Design

Full factorial, 48 combinations × 10 replications = 480 instances, 4,320 solves.

| Factor | Levels |
|---|---|
| A. ND | 41, 52 |
| B. Demand | 30 or 60 patients, collection days 1..10 |
| C. Lines (Philadelphia, Houston, LA) | C1 (10,10,10), C2 (4,10,10), C3 (4,4,10), C4 (4,4,4) |
| D. Urgent share | 10%, 20%, 30% (nested urgent sets) |

The patients are generated exactly as in the earlier DOE: seed 1000·N + r, regional hospital shares, nested
urgent sets. The 20 patient lists in `results_doe/instances/` are identical to the earlier DOE's, and the runs
use these files.

The fixed network data come from `CART_instance_30patients_cap4.xlsx`. Horizon NT = last collection day + ND + 2.
Time limit 120 s per solve; any timeout is logged with its MIP gap.

## How to run

Use Python 3.10+ with `pyomo pandas numpy openpyxl matplotlib highspy` (and `gurobipy` for Gurobi). For example,
in a conda environment:

```bash
conda create -n cart-doe -c conda-forge python=3.11 pyomo pandas numpy openpyxl matplotlib -y
```

```bash
conda activate cart-doe
```

```bash
pip install highspy gurobipy
```

From the repository folder:

1. Smoke test, 2 instances, about 2–3 minutes; output in `results_doe_smoke/`:

   ```bash
   python run_doe.py --smoke
   ```

2. Full design:

   ```bash
   python run_doe.py --workers 4
   ```

   With Gurobi (needs a full license; the pip license is limited to 2,000 variables):

   ```bash
   python run_doe.py --workers 4 --solver gurobi
   ```

3. Figure, after the run:

   ```bash
   python figures.py
   ```

On a Mac, prefix long runs with `caffeinate -i` so the computer does not sleep.

**Resume:** every finished instance is cached in `results_doe/_cache/`. Run the same command again and it
continues where it stopped. `--rerun` ignores the cache; `--tables-only` rebuilds all outputs from the cache
without solving.

| Option | Default | Meaning |
|---|---|---|
| `--solver` | `highs` | `highs` or `gurobi` |
| `--workers` | 1 | instances solved in parallel |
| `--threads` | solver default with 1 worker, 1 otherwise | solver threads per solve |
| `--time-limit` | 120 | seconds per solve |
| `--out` | `results_doe/` | output folder |

**Expected run time (HiGHS):**
- In the smoke test, a 30-patient instance took about 12 s for all 9 solves.
- The largest instance (60 patients, C4, ND 52) took about 130 s, 90 s of it the Baseline-FIFO solve.
- 60-patient instances dominate: about **6–7 hours with 1 worker**, about **2 hours with `--workers 4`** on a
  4-core machine.
- Some 60-patient Baseline-FIFO solves may reach the 120 s limit; they are logged with their gap.
- Gurobi is usually much faster.

## Outputs (`results_doe/`)

| File | Content |
|---|---|
| `all_runs.csv` | One row per instance × model (× ε): factors, ε, `ED_minrisk`, `ED_cap`, status, objective, best bound, MIP gap, cost, E_D, E_U, extra cost and E_D reduction vs Baseline-MinRisk (USD, %), cost per protected patient, mean and max wait of urgent and regular patients, non-nearest assignments, solve time |
| `all_plans.csv.gz` | Every per-patient plan |
| `table1_main_effects.csv` | Per factor level, at the middle ε level (step 3): extra cost over Baseline-MinRisk (USD, %), E_D reduction (absolute, %), E_U, cost per protected patient. Mean, min and max over the 10 replications |
| `table2_by_layout.csv` | The same metrics by capacity layout × (ND, demand), urgent share 30% |
| `table3_tradeoff_base_case.csv` | Base case (ND 41, demand 30, C3, 30% urgent): mean cost and E_D over the replications for Baseline-FIFO, Baseline-MinRisk and the 5 ε levels, with the slope between consecutive levels (USD per expected ineligible patient avoided) |
| `table4_appendix_48_combinations.csv` | Table 1's metrics for all 48 combinations |
| `checks.csv` | Every check, PASS / FAIL / LOGGED |
| `figures/tradeoff_base_case.png` and `.pdf` | From `figures.py`: cost vs E_D for the base case, with Baseline-MinRisk and Baseline-FIFO as separate points |
| `instances/` | The 20 patient lists |

For each factor level, every replication contributes the average over the other factors' combinations. The
tables show the mean, min and max of those 10 values. Cost per protected patient of a replication is its mean
extra cost divided by its mean E_D reduction.

**Checks:**
- every solve is optimal, or logged with its status and gap;
- Baseline-MinRisk cost ≤ Baseline-FIFO cost;
- PatientAware at ε = `ED_minrisk` has the same cost as Baseline-MinRisk (the limit is inactive there);
- along the 5 ε levels, E_D does not rise and cost does not fall;
- every plan is admissible, capacity is respected, TRT ≤ ND, E_D recomputed equals the solver value,
  E_D ≤ ε, and FIFO holds in Baseline-FIFO.

## Files

| File | Content |
|---|---|
| `model.py` | Compact model builders (plan variable X[p,k], one plan per patient, capacity, FIFO, cost cap, E_D ≤ ε), solver wrapper (HiGHS / Gurobi), plan checks |
| `instances.py` | Fixed network data, design factors, instance generation |
| `run_doe.py` | Experiment loop (caching and resume, parallel workers), `all_runs.csv`, tables, checks |
| `figures.py` | Trade-off figure |
| `CART_instance_30patients*.xlsx` | Network and scenario workbooks (the DOE reads `CART_instance_30patients_cap4.xlsx`) |
| `CART_reformulation_fifo.ipynb`, `reference_no_fifo/`, `results_reformulation_fifo/` | Earlier six-scenario FIFO study (from branch `reformulation-with-FIFO-in-baseline`), kept for reference |
