# Survival-aware MPC: simulation study

Operational layer of the survival-aware i-SHIPMENT model
(`Survival_Aware_iSHIPMENT_Formulation.docx`). The strategic MILP (1)–(35) is
solved **once per demand scale** at α = $500K per life and frozen; a daily
discrete-event simulation then executes an operating policy on that network,
realising manufacturing yield as Bernoulli(p_m) at release testing.

| module | role |
|---|---|
| `strategic.py` | freezes one strategic solve: open facilities, m(i), transport modes, FCAP, the `static_survival` order |
| `per_epoch.py` | per-epoch optimisation (P1)–(P6), the survival index (P8), the S_min futility gate |
| `simulation.py` | the daily loop, re-collection recourse, common random numbers |
| `policies.py` | the five policies + the perfect-information MILP |
| `run_experiments.py` | Exp 0 and A–E, all figures |

```bash
python presolve_strategic.py                      # cache the strategic designs
python run_experiments.py --exp all --scale 200    # primary
python run_experiments.py --exp all --scale 100    # confirmation
```

## Configuration (all values as confirmed)

α = $500K/life (strategic objective and Exp C only — **not** in P3);
ρ = 3/2/1; w = 0.15/0.05/0.02; η = 42, γ = 1, κ = 1; H = 7 d;
S_min = 0.75 on **projected survival at delivery**; K_remake = 2 remakes
(3 attempts); no calendar backstop;
ρ_leuk = $5,000; T_horizon = 130 d; TLS/T_MFE/T_QC = 1/7/7;
p_m = 0.85 (m1/m4), 0.92 (m2/m5), 0.95 (m3/m6); N_rep = 30 seeds (0–29).

Survival runs on **one continuous clock from the first collection**:
`loss = 1 − S_u(delivery − t0)`. Expected loss is the probability sum
`Σ(1 − S)` on each realised timeline, with S = 0 for a lost patient.

Cost = facility (strategic) + transport (U1 per attempt, U3 on delivery)
+ (C_material + C_QC) per attempt + ρ_leuk per re-collection. Wasted attempts
are sunk — paid and reported. Cost per therapy divides by treated patients.

**Common random numbers.** The yield draw for attempt k of patient i is a pure
function of `(seed, pid, k)`, so the same batches fail under every policy, at
every offered load, and nested across the failure-rate sweep. Differences
between policies are scheduling, never luck.

## Frozen networks

| N | opened | slots | offered load | spillover past day 130 |
|---|---|---|---|---|
| 50 | m1 | 4 | 0.99 | 0.6 % |
| 100 | m1+m4 | 8 | 0.99 | 0.5–0.7 % |
| 200 | m1+m3 | 14 | 1.14 | 1.2–1.9 % |

All within the < 5 % spillover target; spilled patients are credited with their
realised survival, never zeroed.

## Exp D — policy benchmark (30 seeds, CRN) — *backstop-on baseline*

> Superseded by the **Backstop-off re-run** below, which is the current state.
> Kept for comparison: this is the run with the 90-day calendar backstop ON and
> K_remake read as 2 *attempts*.

N = 200, frozen m1+m3. `best_achievable` is a **proven-optimal**
perfect-information solve on every replication (23 s each, gap 0) — not a proxy.

| policy | clinical loss Σρ(1−S) | high-risk lost *(primary)* | all lost | cost | $/therapy | hold H/M/L | gap closed (H) |
|---|---|---|---|---|---|---|---|
| fifo | 26.98 | 6.63 | 10.76 | $15.608M | $78,645 | 9.2 / 7.4 / 7.8 | 0 % |
| static_survival | 25.03 | 5.43 | 10.70 | $15.572M | $79,049 | 2.9 / 4.6 / 16.1 | 64 % |
| survival_index | 23.25 | 4.81 | 11.13 | $15.554M | $79,309 | 0.4 / 2.7 / 19.8 | 97 % |
| adaptive_mpc | 23.22 | 4.81 | 11.10 | $15.554M | $79,294 | 0.4 / 2.7 / 19.8 | **97 %** |
| best_achievable | 22.93 | 4.75 | 11.00 | $15.483M | $78,959 | 0.1 / 1.9 / 19.6 | 100 % |

N = 100 confirms the ordering: fifo 3.62 → static 3.40 (32 %) → index/mpc 3.03
(88 %) → bound 2.95.

`adaptive_mpc` cuts high-risk loss by **27 %** against fifo and closes 97 % of
the gap to perfect information, for slightly *less* money.

## The trade, stated plainly — *as it stood with the backstop ON*

> **This no longer holds.** With the calendar backstop removed the total-lost
> penalty disappears entirely (non-high −0.012 at N = 200), so the paragraph
> below describes the old configuration, not the current one. See the
> backstop-off re-run.

On **total** patients lost the survival-aware policies are *worse* than fifo
(11.10 vs 10.76 at N = 200), and the gap widens with load and failure rate
(18.1 vs 14.2 at load 1.5). This is the ρ-weighted objective doing what it was
asked: it buys high-risk survival with low-risk delay. Both metrics are
reported everywhere; neither is hidden.

## Exp A / B / C / E

**All six experiments now run on one configuration** (backstop off, K_remake = 2
remakes). Exp 0 is calibration only and config-independent; A, B, C, D and E were
all re-run. The B and E numbers that stand are in the backstop-off section above;
A and C are current as written here.

**A** — `fifo` holds the high-risk tier 9.46 d, medium 7.53 d and low-risk
8.06 d — essentially flat, because arrival order ignores risk. `adaptive_mpc`
holds high-risk **0.40 d** and medium 2.78 d, pushing the wait onto low-risk
at 23.64 d. Priority is an *output* of the objective, not an input rule. The
spread matches Exp D exactly (same config, same seeds, same CRN). N = 100:
fifo 7.09 / 5.47 / 5.79, adaptive_mpc 0.89 / 2.21 / 16.60.

**B** — high-risk loss is flat near 4.26–4.43 for `adaptive_mpc` across offered
load 0.7 → 1.5, while fifo climbs 4.52 → 8.81 and `static_survival` 4.38 → 7.38.
The lever matters most when the system is loaded, and the gap over fifo widens
16× (0.27 → 4.37). Full table in the backstop-off section above. Load is varied
by compressing/expanding the arrival window on the fixed network; at load 0.7 the
window runs past day 130 and spillover reaches ~23 %.

**C** — the network holds at m1+m3 from α = 0 through $2M and **flips to m3+m6
at $5M** (20 slots, load 0.80, +$4.7M, high-risk loss 4.24 → 4.01). The design
threshold is unchanged by the backstop-off re-run — it is a property of the
strategic solve, and those were re-used from cache. The $500K operating point
sits well inside the stable region: at $500K the design is not yet the binding
lever, prioritisation is. At N = 100 the design never flips across the whole
sweep (m1+m4 throughout).

### What actually drives the network

α is design-inert over the operationally relevant range, but that does not mean
the survival extension is inert. Solving the strategic model in four
configurations separates the causes:

| variant | N = 200 | N = 100 |
|---|---|---|
| cost objective, no queue, ND = 18 | m2 (31 slots) | m1+m3 (14 slots) |
| cost objective, no queue, ND = 42 | m2 (31 slots) | m1+m3 (14 slots) |
| full model, α = 0 | m1+m3 (14 slots) | m1+m4 (8 slots) |
| full model, α = $500K | m1+m3 (14 slots) | m1+m4 (8 slots) |

Three readings. **Survival is design-inert**: α = 0 and α = $500K open the same
facilities at both scales. **The deadline is irrelevant**: ND = 18 and ND = 42
agree. **The manufacturing queue is what moves the design**: it is the only
remaining difference, and it moves it a long way. Denied the ability to hold
material, the cost model cannot smooth the arrival peak and must buy raw
capacity instead — $22.6M on m2 rather than $10.5M on m1+m3 at N = 200, at an
offered load of 0.51 rather than 1.14.

This is a result about the queue, not about survival, and it is why the survival
term is kept in the strategic objective rather than being split out: removing it
changes nothing about the design, while removing the queue changes the design,
the capacity and the congestion the whole operational study is defined against.

**E** — `adaptive_mpc` pulls away from `static_survival` as batches fail more
often: high-risk loss 3.85 vs 3.85 at a 0 % failure rate (identical — nothing to
react to), 4.32 vs 5.44 at 10 %, **6.51 vs 11.59 at 30 %**. Re-optimising on the
observed state is what the dynamic layer buys, and its value grows with the
failure rate. Full table in the backstop-off section above.

## Backstop-off re-run

Two changes, then Exp D, B and E re-run at both scales (30 seeds, CRN,
failures on). The network is unchanged, so the strategic solves were re-used
from cache — only simulation outcomes move.

* the **loose 90-day backstop is switched off entirely**: no calendar-based
  removal of any kind. A patient leaves only by failing the S_min gate on a
  required (re-)collection, by exhausting K_remake, or by being treated;
  everyone else keeps queueing and is credited with the survival their eventual
  delivery earns, horizon spillover included.
* **K_remake now means max REMAKES = 2** (three attempts), matching the
  parameter table. The yield draw is keyed on `(seed, patient, attempt)`, so the
  first two draws are unchanged and CRN stays nested against the old runs.

Everything below is the new state; the old (backstop-on) numbers are kept beside
it. Exp A and C were re-run on this configuration afterwards (their numbers are
in the section above); Exp 0 is calibration only and is config-independent.

### 1. Exp D — where the non-high excess went

N = 200, adaptive_mpc vs fifo, expected patients lost:

| | high-risk | non-high | total |
|---|---|---|---|
| **backstop ON** (old) | −1.822 | **+2.155** | **+0.332** |
| **backstop OFF** (new) | −1.917 | **−0.012** | **−1.929** |

The +2.16 non-high excess is gone — not traded elsewhere, gone. It was the
tier-blind backstop executing the low-risk patients `adaptive_mpc` deprioritises,
not a clinical cost of the ρ-weighted objective. N = 100 agrees: non-high
+0.404 → −0.003, total −0.184 → −0.671.

| policy | high-risk | non-high | all | removals | clinical loss |
|---|---|---|---|---|---|
| fifo | 6.161 | 3.422 | 9.582 | 0.10 | 24.52 |
| static_survival | 5.165 | 3.475 | 8.640 | 0.60 | 21.39 |
| survival_index | 4.244 | 3.411 | 7.655 | 0.07 | 18.31 |
| adaptive_mpc | **4.244** | 3.409 | **7.653** | 0.07 | 18.31 |
| best_achievable | 4.183 | 3.360 | 7.544 | 0.07 | 18.02 |

N = 100: fifo 3.068 / 1.664 / 4.733 · adaptive 2.400 / 1.662 / 4.062 · bound
2.318 / 1.671 / 3.989.

**adaptive_mpc now beats fifo on both metrics at both scales.** Removals are
near-zero throughout, so expected loss is now almost entirely survival decay
from waiting rather than death-by-rule. Two side effects worth knowing: the
S_min gate now does fire, but only under `static_survival` (0.53 per replication
at N = 200) — its rigid order lets a few patients decay far enough that a remake
fails the projected-delivery test, which is the gate biting exactly as intended;
and `best_achievable` records zero K_remake losses, because with foresight it
never spends a third attempt it knows will fail.

### 2. Exp B — high-risk lost vs offered load

N = 200 (new; old fifo/adaptive in brackets):

| load | fifo | static_survival | adaptive_mpc | fifo − adaptive |
|---|---|---|---|---|
| 0.70 | 4.523 [5.08] | 4.376 | 4.256 [4.83] | 0.267 |
| 0.85 | 4.796 [5.34] | 4.499 | 4.271 [4.84] | 0.525 |
| 1.00 | 5.303 [5.81] | 4.832 | 4.307 [4.87] | 0.996 |
| 1.15 | 6.304 [6.75] | 5.216 | 4.230 [4.80] | 2.073 |
| 1.30 | 7.420 [7.71] | 6.312 | 4.300 [4.87] | 3.119 |
| 1.50 | 8.806 [8.94] | 7.377 | 4.433 [4.98] | 4.372 |

N = 100 gap: 0.184 → 0.351 → 0.668 → 1.254 → 1.575 → 2.259.

**The gap widens monotonically with load — 16× at N = 200** (0.27 → 4.37) and
12× at N = 100. `adaptive_mpc` holds high-risk loss essentially flat (4.26 →
4.43) while fifo nearly doubles. `static_survival` degrades almost as badly as
fifo once load passes 1.0, so this is a case for the *adaptive* layer, not for
survival-awareness alone.

### 3. Exp E — high-risk lost vs failure rate

N = 200 (new; old in brackets):

| 1 − p | fifo | static_survival | adaptive_mpc | static − adaptive |
|---|---|---|---|---|
| 0.00 | 5.086 | 3.849 [3.85] | 3.849 [3.85] | **0.000** |
| 0.05 | 5.637 | 4.434 [4.61] | 4.060 [4.17] | 0.374 |
| 0.10 | 6.383 | 5.435 [6.24] | 4.319 [4.89] | 1.117 |
| 0.15 | 7.147 | 6.504 [7.48] | 4.611 [5.62] | 1.892 |
| 0.20 | 8.323 | 7.785 [8.95] | 4.995 [6.60] | 2.790 |
| 0.30 | 12.564 | 11.589 [12.89] | 6.510 [9.58] | **5.079** |

N = 100 static − adaptive: 0.000 → 0.106 → 0.260 → 0.620 → 0.853 → 1.700.

**Adaptive pulls further ahead of static as failures rise, monotonically, to
5.08 patients at a 30 % failure rate** — 14× the 0.37 gap at 5 %. At zero
failures the two are *identical* (3.849 both): with nothing to react to,
re-optimising buys nothing, exactly as theory predicts. Meanwhile
`static_survival`'s own advantage over fifo collapses (1.24 → 0.98) while
adaptive's grows (1.24 → 6.05). This is the sharpest evidence in the study for
the dynamic layer.

## Open issue — the S_min gate never fires (backstop-on runs)

`S_min = 0.75` is evaluated on projected survival at delivery, where the wait
for a slot is read from `busy_m(τ)` (in-progress jobs only), exactly as
specified. That occupancy never looks more than `T_MFE − 1 = 6` days ahead, so
the gate needs an accrued wait of **> 50 d (H), > 212 d (M), > 574 d (L)** at
the moment of failure. It fired **zero times** in every run at every scale.

Consequently every loss is either `k_remake` exhaustion (identical across
policies by CRN) or the *loose 90-day backstop* — and the backstop is
tier-blind, so it falls entirely on the low-risk patients that `adaptive_mpc`
deprioritises (20 backstop losses per 10 replications at N = 200, **100 %
low-risk**, vs 0 under fifo). That is a large part of why `adaptive_mpc`'s total
expected loss exceeds fifo's.

The doc's intent — "sicker/later patients cross the floor first, so they are
precisely the ones who cannot re-collect" — therefore did not bite as written.
Making it bite more broadly would mean letting `wait_for_slot` reflect **queue
position**, not just in-progress occupancy; `wait_for_slot` is unchanged.

**Superseded in part by the backstop-off re-run above.** With the backstop gone
and three attempts allowed, the gate does now fire — but only under
`static_survival`, whose rigid order lets patients decay far enough for a remake
to fail the projected-delivery test. Under the adaptive policies it still never
fires, and removals are near-zero, so expected loss is now almost entirely
survival decay rather than removal.

## Solver notes

Gurobi is imported lazily; no module import needs a licence. The container
carries only the size-limited Gurobi licence (2000 cols), which refuses the
strategic and perfect-information models — those fall back to HiGHS through the
same Pyomo plumbing the phase 1–4 scripts use. The small per-epoch (P1)–(P6)
models do solve under Gurobi.

`best_achievable` is dispatched exactly as its MILP planned, including left-idle
slots (77 slot-days at N = 50): holding capacity for a sicker patient arriving
tomorrow is part of what perfect information buys, and forcing it to be
non-idling would stop it being a lower bound. The four online policies are all
strictly non-idling.
