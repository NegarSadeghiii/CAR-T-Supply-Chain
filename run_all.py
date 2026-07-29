"""
run_all.py — reproduce the entire dynamic-model extension end to end.

Runs (all randomness seeded; see calibration_table.md):
  1. Unit tests            : declineprob, clearing function.
  2. Nesting validation    : dynamic_sp(kappa=0) == yield_sp_v1 on the toy
                             (total ~ 2.774, VSS ~ 0.154). MUST pass first.
  3. Causes + priority     : causes_priority_experiment.py — the main deliverable.
                             STEP 1 (why high-urgency patients are lost: normal
                             wait vs after a failure) and STEP 2 (does putting the
                             sickest first protect them when factories are full).
                             numbers -> results/causes_priority_results.json.
  4. Technical validation  : deterioration_experiment.py — internal-model
                             diagnostics behind TECHNICAL_APPENDIX.md.

Publication figures are produced separately by the manuscript pipeline
(`python paper/make_assets.py`).

Usage:
  python run_all.py            # full reproduction (minutes)
  python run_all.py --quick    # fast smoke run
Exit code is non-zero if the unit tests or the nesting gate fail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _run(cmd, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    r = subprocess.run([sys.executable] + cmd, cwd=_HERE)
    if r.returncode != 0:
        print(f"\n!! FAILED: {label} (exit {r.returncode})")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast VoE smoke run")
    args = ap.parse_args()

    # 1. Unit tests for the calibration kernels.
    _run(["test_declineprob.py"], "1a. Unit tests — decline kernel")
    _run(["test_clearing.py"], "1b. Unit tests — clearing function")

    # 2. Nesting gate (must reproduce v1 before any experiment is trusted).
    _run(["test_nesting.py"], "2. Nesting validation — dynamic_sp(kappa=0) == v1")

    # 2b. Waiting-model / kernel corrections (extra=12->0.610, T_V2V=42 regression,
    #     kappa=0 nesting, priority OFF == no-lever).
    _run(["test_waiting_model.py"], "2b. Waiting-model validation (kernel corrections)")

    # 3. Main deliverable (RESULTS.md): why high-urgency patients are lost
    #    (Step 1) and whether putting the sickest first protects them when
    #    factories are full (Step 2). Writes results/causes_priority_results.json.
    cp_cmd = (["causes_priority_experiment.py", "--quick"]
              if args.quick else ["causes_priority_experiment.py"])
    _run(cp_cmd, "3. Causes + sickest-first experiment (patient outcomes)")

    # 4. Technical validation behind TECHNICAL_APPENDIX.md.
    det_cmd = (["deterioration_experiment.py", "--quick"]
               if args.quick else ["deterioration_experiment.py"])
    _run(det_cmd, "4. Technical validation (nesting, surrogate, benefit curve)")

    print(f"\n{'='*70}\nAll stages completed. See RESULTS.md and "
          f"results/causes_priority_results.json\n{'='*70}")


if __name__ == "__main__":
    main()
