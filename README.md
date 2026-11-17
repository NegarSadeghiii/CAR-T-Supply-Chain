# Compact reformulation with FIFO in the Baseline

This branch holds only what is needed to run the compact CAR-T scheduling reformulation with a
first-in-first-out (FIFO) rule added to the **Baseline** model and to compare it with the same model without FIFO.
It was started as an orphan branch, so it shares no history or files with the other branches.

Every scenario uses ND = 40, Philadelphia 4 lines, Los Angeles 10 lines, Patients column I (30% urgent),
lambda_D = 150,000 and lambda_U = 0. The Patient-aware model has no FIFO rule.

## Files and where they come from

### Notebook (new on this branch)

| File | Origin |
|---|---|
| `CART_reformulation_fifo.ipynb` | New. Self-contained notebook; its code is adapted from `CART_reformulation.ipynb` on branch `reformulation`, with the FIFO constraints added to the Baseline. Contains its executed outputs. |
| `README.md` | New. This file. |

### Scenario workbooks (read-only inputs)

| File | Scenario | Origin |
|---|---|---|
| `CART_instance_30patients.xlsx` | Houston 10 / T_MFE 7 | branch `reformulation` (commit 7aa60b5), copied unchanged |
| `CART_instance_30patients_cap4.xlsx` | Houston 4 / T_MFE 7 | branch `reformulation` (commit 7aa60b5), copied unchanged |
| `CART_instance_30patients_cap2.xlsx` | Houston 2 / T_MFE 7 | branch `reformulation` (commit 7aa60b5), copied unchanged |
| `CART_instance_30patients_cap4_tmfe14.xlsx` | Houston 4 / T_MFE 14 | branch `reformulation` (commit 7aa60b5), copied unchanged |
| `CART_instance_30patients_cap2_tmfe14.xlsx` | Houston 2 / T_MFE 14 | branch `reformulation` (commit 7aa60b5), copied unchanged |
| `CART_instance_30patients_cap10_tmfe5.xlsx` | Houston 10 / T_MFE 5 | branch `reformulation` (commit 7aa60b5), copied unchanged |

### Reference results without FIFO (read only for the comparison)

| File | Origin |
|---|---|
| `reference_no_fifo/six_scenarios_reformulation.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/six_scenarios_reformulation.csv` |
| `reference_no_fifo/plans/Houston_10_T_MFE_5__Baseline.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_10_T_MFE_5__Baseline.csv` |
| `reference_no_fifo/plans/Houston_10_T_MFE_5__Patient-aware.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_10_T_MFE_5__Patient-aware.csv` |
| `reference_no_fifo/plans/Houston_10_T_MFE_7__Baseline.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_10_T_MFE_7__Baseline.csv` |
| `reference_no_fifo/plans/Houston_10_T_MFE_7__Patient-aware.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_10_T_MFE_7__Patient-aware.csv` |
| `reference_no_fifo/plans/Houston_2_T_MFE_14__Baseline.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_2_T_MFE_14__Baseline.csv` |
| `reference_no_fifo/plans/Houston_2_T_MFE_14__Patient-aware.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_2_T_MFE_14__Patient-aware.csv` |
| `reference_no_fifo/plans/Houston_2_T_MFE_7__Baseline.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_2_T_MFE_7__Baseline.csv` |
| `reference_no_fifo/plans/Houston_2_T_MFE_7__Patient-aware.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_2_T_MFE_7__Patient-aware.csv` |
| `reference_no_fifo/plans/Houston_4_T_MFE_14__Baseline.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_4_T_MFE_14__Baseline.csv` |
| `reference_no_fifo/plans/Houston_4_T_MFE_14__Patient-aware.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_4_T_MFE_14__Patient-aware.csv` |
| `reference_no_fifo/plans/Houston_4_T_MFE_7__Baseline.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_4_T_MFE_7__Baseline.csv` |
| `reference_no_fifo/plans/Houston_4_T_MFE_7__Patient-aware.csv` | branch `reformulation` (commit 7aa60b5), copied unchanged from `results_reformulation/plans/Houston_4_T_MFE_7__Patient-aware.csv` |

### Outputs of `CART_reformulation_fifo.ipynb` (new on this branch)

| File | Content |
|---|---|
| `results_reformulation_fifo/six_scenarios_fifo.csv` | Six-scenario table: Baseline with FIFO, Patient-aware without FIFO |
| `results_reformulation_fifo/comparison_baseline_with_vs_without_fifo.csv` | Baseline cost, E_D and relative reduction, without vs with FIFO |
| `results_reformulation_fifo/runs_size_and_time.csv` | Variables, FIFO constraints, total constraints, build and solve time per run |
| `results_reformulation_fifo/diagnostic_residual_ties_fifo.csv` | Extra diagnostic: range of Baseline E_D over same-cost plans that satisfy FIFO |
| `results_reformulation_fifo/checks.csv` | Every PASS/FAIL check |
| `results_reformulation_fifo/plans/Houston_10_T_MFE_5__Baseline.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_10_T_MFE_5__Patient-aware.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_10_T_MFE_7__Baseline.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_10_T_MFE_7__Patient-aware.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_2_T_MFE_14__Baseline.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_2_T_MFE_14__Patient-aware.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_2_T_MFE_7__Baseline.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_2_T_MFE_7__Patient-aware.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_4_T_MFE_14__Baseline.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_4_T_MFE_14__Patient-aware.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_4_T_MFE_7__Baseline.csv` | Per-patient plan of that run |
| `results_reformulation_fifo/plans/Houston_4_T_MFE_7__Patient-aware.csv` | Per-patient plan of that run |
