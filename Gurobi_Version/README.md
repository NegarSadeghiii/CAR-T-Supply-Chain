# Gurobi_Version — local Gurobi backend

A clean, runnable Gurobi implementation of the CAR-T resilient two-stage
stochastic program (yield uncertainty + three-action recourse
re-manufacture / subcontract / cancel + capacity-disruption channel with the
corrected non-negative capacity floor `(1 − ξ)·C`). It solves the same model the
rest of the repo solves — the default backend elsewhere is HiGHS; this folder
lets you reproduce and report results with Gurobi instead, for speed and
because a full Gurobi license handles the full 50-patient instance comfortably.

The model itself is **not** re-derived here: `gurobi_model.py` imports the
repo's `Instance`, scenario sampler, disruption sampler, and the canonical
Gurobi builder (`yield_sp_v1.build_sp_model`), so there is a single source of
truth for the model.

## Contents

| File | What it does |
|------|--------------|
| `gurobi_model.py`   | Gurobi solve function `solve_sp_gurobi(...)` (same return dict as the HiGHS backend) + `subset_instance` helper. |
| `parity_check.py`   | Confirms Gurobi and HiGHS return the same optimal objective within `1e-6` on the toy instance and a small case-study subset; prints a like-for-like solve-time comparison. |
| `run_experiments.py`| Solves the case study and the disruption/correlation sweep with Gurobi, writing JSON to `results/`. |
| `results/`          | Output JSON (`case_study_gurobi.json`, `correlation_sweep_gurobi.json`). |

## Requirements

- Python 3.10+
- `numpy`
- `gurobipy` **and a Gurobi license installed on the machine** (`gurobi.lic`,
  a WLS license, or an academic named-user license). Gurobi picks this up
  automatically from the environment — **no license key or credential is stored
  in this repo**, and none is needed in the code.
- The repo root must be importable (the scripts add it to `sys.path`
  automatically), and the HiGHS backend (`highspy`) is needed only for
  `parity_check.py`.

## How to run

From the repository root:

```bash
# 1. Solver parity + timing (Gurobi vs HiGHS), small instances
python Gurobi_Version/parity_check.py

# 2. Case study + correlation sweep with Gurobi
python Gurobi_Version/run_experiments.py

# 3. Force the full 50-patient instance (needs an UNRESTRICTED license)
FULL_INSTANCE=1 python Gurobi_Version/run_experiments.py
#   or:  python Gurobi_Version/run_experiments.py --full
```

## A note on license size limits

The free **size-limited** `gurobipy` from PyPI caps model size (~2000
variables/constraints) and **cannot** build the full 50-patient × 200-scenario
model — it raises `GurobiError: Model too large for size-limited license`. The
repo's HiGHS backend was chosen for exactly this reason.

- `run_experiments.py` **auto-detects** this: it tries the full instance and, on
  a size-limit error, transparently falls back to the largest case-study subset
  that fits (recording `"instance": "subset"` in the JSON so results are never
  silently mislabeled).
- With a **full/unrestricted** Gurobi license, pass `--full` (or set
  `FULL_INSTANCE=1`) to solve the real 50-patient instance and reproduce the
  headline numbers with Gurobi.

Both solvers are exact MILP solvers: on the same model, scenarios, and
tolerance they return the same optimum (see `parity_check.py`). Gurobi is used
for speed and reporting, not for a different answer.
