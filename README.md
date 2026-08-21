# Research-

Survival-aware CAR-T supply-chain scheduling, built on the i-SHIPMENT MILP
(Triantafyllou et al., 2022). `i-SHIPMENT_Pyomo.ipynb` is the untouched
**baseline**; everything here extends it in a copy.

The work has two layers, and they compose:

1. a **strategic** layer that adds a manufacturing queue and an exact
   integer-day survival model to the MILP, and
2. an **operational** layer that freezes the strategic network and executes a
   scheduling policy through a daily discrete-event simulation with stochastic
   manufacturing yield.

The full formulation is `Survival_Aware_iSHIPMENT_Formulation.docx`.


## 1. Strategic layer — queue + survival extension

Keeps the baseline and adds, in a copy:

* a real **manufacturing queue** — `INM` becomes a start decision constrained by
  arrival (eq. 32), started-once (eq. 33) and concurrent capacity (eq. 34), so a
  job can be **held**;
* an **exact integer-day survival model** — `delta[p,d]`, `d = 17..42`, with
  `S_u(t) = (1 - w_u)**(t/42)`;
* the objective `min Z = (original i-SHIPMENT cost) + Σ_p α_u(p)·(1 − S[p])`,
  with `α_u` the dollar value of one life lost in tier u -- the model's only
  life-value parameter (there is no separate priority weight).

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
| `ext_figure.py`, `value_figures.py` | phase 1-4 figures and sweeps |
| `make_readme.py` | regenerates `results/README.md` from the result files |
| `results/` | per-scale CSVs, the cost–lives frontier PNG, and the full write-up |

```bash
python ishipment_survival.py --phase all
```

See **[`results/README.md`](results/README.md)** for the findings.


## 2. Operational layer — survival-aware MPC simulation

The strategic MILP is solved **once per demand scale** (at α = $500K per life)
and frozen: open facilities, the assignment m(i), transport modes and FCAP stop
being decisions. A daily discrete-event simulation then runs the clock over the
130-day horizon, realises the uncertainty the deterministic model omits —
manufacturing yield, Bernoulli(p_m) at release testing — and at every decision
epoch asks a policy who starts next. On failure the recourse is a fresh
leukapheresis, gated on projected survival at delivery.

Survival is evaluated on **one continuous clock from the first collection**, so a
remade patient carries the deterioration of every earlier attempt.

### The five policies

| policy | what it does |
|---|---|
| `fifo` | serve in arrival order — no objective |
| `survival_index` | greedy (P8): serve the largest one-day loss of life value α_u·ΔS |
| `static_survival` | the survival schedule optimised ONCE up front, dispatched greedily, never re-solved |
| `adaptive_mpc` | the survival schedule re-optimised at EVERY decision epoch on the observed state, via (P1)–(P6) |
| `best_achievable` | the loss if all manufacturing outcomes were known in advance — a perfect-information MILP per replication, the lower-bound reference |

All five see the same frozen network and, through common random numbers keyed on
`(seed, patient, attempt)`, the same batch failures: differences between policies
are scheduling, never luck.

| file | role |
|---|---|
| `strategic.py` | freezes one strategic solve per scale and hands it to the simulation |
| `per_epoch.py` | the per-epoch optimisation (P1)–(P6), the closed-form survival index (P8), and the S_min futility gate |
| `simulation.py` | the daily discrete-event loop, yield realisation, re-collection recourse, common random numbers |
| `policies.py` | the five policies and the perfect-information MILP |
| `run_experiments.py` | Exp 0 and A–E; writes every figure to `figures/` |
| `presolve_strategic.py` | caches the strategic designs the experiments need |
| `figures/`, `results/mpc/` | the figures, per-experiment CSVs and the write-up |

```bash
python presolve_strategic.py                      # cache the strategic designs
python run_experiments.py --exp all --scale 200   # primary
python run_experiments.py --exp all --scale 100   # confirmation
```

### Experiments and figures

| experiment | figure | question |
|---|---|---|
| Exp 0 | `fig0_tier_survival_curves` | how much does a day of delay cost each tier? |
| Exp A | `figA1_schedule_gantt_fifo_vs_mpc`, `figA2_holdtime_by_tier` | who gets served first, and where does the waiting go? |
| Exp B | `figB1_clinical_loss_vs_offered_load` | when does survival-aware scheduling matter? |
| Exp C | `figC1_value_of_life_frontier` | at what value of a life does the network design change? |
| Exp D | `figD1_policy_cost_vs_loss`, `figD2_gap_to_best_achievable` | how do the policies rank against the best possible? |
| Exp E | `figE1_expected_loss_vs_failure_rate` | what does the adaptive layer buy as batches fail? |

Reported metrics are only these: expected high-risk patients lost (primary),
total expected patients lost, total cost / cost per therapy, mean hold by tier,
and the share of the `best_achievable` gap closed.

### Headline results (N = 200, 30 yield seeds, common random numbers)

`adaptive_mpc` cuts expected high-risk patients lost from **6.16 to 4.24** against
`fifo`, lowers total expected loss too (9.58 → 7.65), and closes **97 %** of the
gap to a proven-optimal perfect-information bound — at essentially unchanged cost
(+$945 on $15.65M, under 0.01 %). Prioritisation is emergent, not a rule: mean
hold for the high-risk tier falls from 9.5 d to 0.4 d while the low-risk tier
absorbs the wait (8.1 d → 23.6 d).

The advantage grows under stress in both directions — the high-risk gap over
`fifo` widens from 0.27 to 4.37 patients as offered load goes 0.7 → 1.5, and
`adaptive_mpc` pulls ahead of `static_survival` by 5.08 patients at a 30 %
manufacturing failure rate while being *identical* to it when nothing fails.

See **[`results/mpc/README.md`](results/mpc/README.md)** for the full tables,
the configuration, and the open modelling questions.
