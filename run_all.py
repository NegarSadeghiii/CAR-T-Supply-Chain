"""
run_all.py — reproduce the entire dynamic-model extension end to end.

Runs (all randomness seeded; see calibration_table.md):
  1. Unit tests            : declineprob, clearing function.
  2. Nesting validation    : dynamic_sp(kappa=0) == yield_sp_v1 on the toy
                             (total ~ 2.774, VSS ~ 0.154). MUST pass first.
  3. VoE experiment        : value_of_endogeneity.py (designs, simulator, sweep,
                             figures -> figures/, numbers -> results/).

Usage:
  python run_all.py            # full reproduction (minutes)
  python run_all.py --quick    # fast smoke run of the VoE sweep
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

    # 3. Value-of-Endogeneity experiment (designs + simulator + sweep + figures).
    # --quick skips figures so it never overwrites the full-run figures.
    voe_cmd = (["value_of_endogeneity.py", "--quick", "--no-figures"]
               if args.quick else ["value_of_endogeneity.py"])
    _run(voe_cmd, "3. Value of Endogeneity experiment")

    print(f"\n{'='*70}\nAll stages completed. See RESULTS.md, figures/figureD*, "
          f"results/value_of_endogeneity*.json\n{'='*70}")


if __name__ == "__main__":
    main()
