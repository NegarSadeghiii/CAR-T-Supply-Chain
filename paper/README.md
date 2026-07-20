# Paper assets — CAR-T dynamic deterioration & sickest-first prioritization

Self-contained, fully reproducible asset bundle for the manuscript. Every
number the paper cites is traceable to a seeded experiment run and tagged with
its source in `results/numbers.md`.

## Reproduce everything with one command

From the repository root:

```bash
python paper/make_assets.py           # full seeded run (minutes)
python paper/make_assets.py --quick   # fast smoke run (smaller N, coarse grid)
python paper/make_assets.py --figures-only   # rebuild all assets from cached raw
```

This runs the full experiment set, then writes the master results, CSVs, LaTeX
tables, the traceability ledger, and all figures. Seeds are fixed (training seed
0, out-of-sample seed 100); the MILP backend is HiGHS.

## Layout

```
paper/
  make_assets.py            one command to regenerate everything
  refs.bib                  BibTeX for every cited source
  src/
    experiments.py          seeded experiment set (validation, E1-E5, sensitivity)
    paper_style.py          shared matplotlib style (colorblind-safe, journal sizes)
    figures.py              F1-F5 (main) + T1-T3 (technical)
    emit.py                 CSVs, LaTeX tables, traceability ledger
  data/
    raw/                    scenario-level outputs + run metadata (JSON)
    processed/              one tidy CSV per experiment (numbers behind each figure)
  results/
    results.json            master machine-readable results
    numbers.md              TRACEABILITY LEDGER — every parameter & result with a source tag
  figures/                  publication figures (vector PDF + 300-dpi PNG)
    technical/              validation / internal-model figures
  tables/                   LaTeX table fragments (booktabs): parameters + results
```

## Experiments

| ID | Question | Main asset |
|---|---|---|
| Validation | decline speed 0 == base model; priority OFF == no-lever model; planning estimate vs detailed simulation | `VALIDATION.csv`, `T1` |
| E1 | on-time plan: what it ASSUMES vs what ACTUALLY happens (high-urgency lost, cost), by tier | `E1_prediction_gap.csv` |
| E2 | why patients are lost: normal wait vs after failure (busy 150p, low 50p) | `E2_loss_causes.csv`, `F2` |
| E3 | three plans @150p: on-time / decline-aware-spare / sickest-first | `E3_three_plans.csv`, `F3` |
| E4 | capacity-cap sweep s_max in {40,55,75} | `E4_capacity_cap.csv`, `F4` |
| E5 | robustness to the normal-wait mortality anchor | `E5_bwd_robustness.csv`, `F5` |
| E6 | priority-rule family (Tseng et al. 2024): FIFO vs Threshold-X | `E6_priority_rules.csv`, `F6a/F6b` |
| Sensitivity | decline-curve shape (gamma) and tier hazard-ratio grids | `SENS_*.csv`, `T2`, `T3` |
| Issue 1 | normal vein-to-vein wait `T_V2V` sweep {21,28,35,42} d | `ISSUE1_tv2v_sweep.csv` |
| Issue 4 | re-make delay: observed 12 d vs full cycle (= T_V2V) | `ISSUE4_remake_delay.csv` |
| Issue 5 | implemented / not-implemented scope | `ISSUE5_implemented.csv`, `tables/implemented.tex` |
| Issue 6 | failure rate × T_V2V: where waiting vs failure dominates | `ISSUE6_failure_x_tv2v.csv`, `F7` |
| Issue 7 | OLD-vs-NEW headline comparison | `numbers.md`, `tables/old_vs_new.tex` |

The waiting-time model was corrected after review (explicit `T_V2V`, hazard-rate
normal wait, extra-only kernel, compound hazards); see `TECHNICAL_APPENDIX.md
§4b` and validate with `python test_waiting_model.py`.

## Figures

- **F1** high-urgency lost vs decline speed, with the real-decline-rate marker
- **F2** why high-urgency patients are lost: normal wait vs after a failure (busy vs low)
- **F3** three-plan comparison (high-urgency lost; who pays for it)
- **F4** capacity-cap sweep
- **F5** robustness of the two headline numbers to the normal-wait mortality anchor
- **T1-T3** planning-vs-simulation, decline-shape, hazard-ratio (technical)

## Provenance & citations

`results/numbers.md` is the source of truth for every cited value, each tagged
`[data-anchored]`, `[literature-derived]`, `[calibrated from multiple]`,
`[assumption (sensitivity-tested)]`, or `[synthetic]`, carrying the provenance
from the repository calibration table (`calibration_table.md`). The two clinical
anchors (`dulobdas2025` PFS HR 1.64; `cohet2023` +12-day vein-to-vein delay) are
verified against their sources. Network/cost citation metadata in `refs.bib`
follows the calibration table and should be confirmed against final published
copies before submission.
