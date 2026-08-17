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
S_min = 0.75 on **projected survival at delivery**; K_remake = 2 attempts;
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

## Exp D — policy benchmark (30 seeds, CRN)

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

## The trade, stated plainly

On **total** patients lost the survival-aware policies are *worse* than fifo
(11.10 vs 10.76 at N = 200), and the gap widens with load and failure rate
(18.1 vs 14.2 at load 1.5). This is the ρ-weighted objective doing what it was
asked: it buys high-risk survival with low-risk delay. Both metrics are
reported everywhere; neither is hidden.

## Exp A / B / C / E

**A** — `fifo` holds the high-risk tier 9.2 d and low-risk 7.8 d; `adaptive_mpc`
holds high-risk **0.4 d** and low-risk 19.8 d. Priority is an *output* of the
objective, not an input rule.

**B** — high-risk loss is flat near 4.8–5.0 for the survival-aware policies
across offered load 0.7 → 1.5, while fifo climbs 5.1 → 8.9 and
`static_survival` 4.9 → 7.5. The lever matters most when the system is loaded.
Load is varied by compressing/expanding the arrival window on the fixed
network; at load 0.7 the window runs past day 130 and spillover reaches ~23 %.

**C** — the network holds at m1+m3 from α = 0 through $2M and **flips to m3+m6
at $5M** (20 slots, load 0.80, +$4.7M, high-risk loss 4.81 → 4.11). The $500K
operating point sits well inside the stable region: at $500K the design is not
yet the binding lever, prioritisation is.

**E** — `adaptive_mpc` pulls away from `static_survival` as batches fail more
often: high-risk loss 3.85 vs 3.85 at a 0 % failure rate, 4.89 vs 6.24 at 10 %,
**9.58 vs 12.89 at 30 %**. Re-optimising on the observed state is what the
dynamic layer buys, and its value grows with the failure rate.

## Open issue — the S_min gate never fires

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
precisely the ones who cannot re-collect" — therefore does not bite as written.
Making it bite would mean letting `wait_for_slot` reflect **queue position**, not
just in-progress occupancy. Left as specified pending a decision.

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
