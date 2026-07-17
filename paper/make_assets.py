"""
paper/make_assets.py — regenerate EVERY manuscript asset, seeded, from one
command:

    python paper/make_assets.py            # full run (minutes)
    python paper/make_assets.py --quick    # fast smoke run (smaller N, coarse grid)
    python paper/make_assets.py --figures-only   # rebuild assets from cached raw

Pipeline
  1. run the full seeded experiment set (experiments.py) -> paper/data/raw/*.json
  2. assemble the master payload                          -> paper/results/results.json
  3. tidy CSVs (one per experiment)                       -> paper/data/processed/
  4. LaTeX tables (booktabs, with source column)          -> paper/tables/
  5. traceability ledger                                  -> paper/results/numbers.md
  6. publication + technical figures                      -> paper/figures[/technical]

Everything downstream of step 1 is a pure function of results.json, so
--figures-only rebuilds CSVs/tables/ledger/figures from the cached raw run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

import experiments          # noqa: E402
import emit                 # noqa: E402
import figures              # noqa: E402

RAW = _HERE / "data" / "raw"
PROC = _HERE / "data" / "processed"
RES = _HERE / "results"
FIG = _HERE / "figures"
FIGT = FIG / "technical"
TAB = _HERE / "tables"


def _load_cached():
    P = {}
    for name in ("busy", "low", "three_plans", "cap_sweep", "e5_bwd",
                 "gamma_grid", "hr_grid", "validation"):
        P[name] = json.loads((RAW / f"{name}.json").read_text())
    P["meta"] = json.loads((RAW / "run_meta.json").read_text())
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--figures-only", action="store_true",
                    help="rebuild assets from cached paper/data/raw (no re-solve)")
    args = ap.parse_args()

    for d in (PROC, RES, FIG / "png", FIG / "pdf", FIGT / "png", FIGT / "pdf", TAB):
        d.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        print("Loading cached raw run …")
        P = _load_cached()
    else:
        print("Running full experiment set (seeded) …")
        P = experiments.run(quick=args.quick)

    print("Writing master results.json …")
    emit.write_master_json(P, RES / "results.json")
    print("Writing processed CSVs …")
    emit.write_csvs(P, PROC)
    print("Writing LaTeX tables …")
    emit.write_tables(P, TAB)
    print("Writing traceability ledger …")
    emit.write_numbers_md(P, RES / "numbers.md")
    print("Rendering figures …")
    figures.make_all(P, FIG, FIGT)
    print("\nAll assets written under paper/.")


if __name__ == "__main__":
    main()
