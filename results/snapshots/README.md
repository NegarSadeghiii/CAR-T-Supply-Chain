# Patient selection at a decision epoch — what the code actually does

Produced by `decision_snapshots.py`, which instruments the operational layer and
records, at every decision epoch, the epoch state and what **each** policy would
choose on that **same** state. No policy is re-implemented here: every choice
below comes from calling the shipped selection code
(`policies.Fifo.select`, `per_epoch.index_choice`, `per_epoch.solve_epoch`).

```bash
python3 decision_snapshots.py --theory                              # the ranking algebra
python3 decision_snapshots.py --scale 200 --seed 0 --acting all     # snapshots + agreement stats
python3 decision_snapshots.py --scale 200 --acting adaptive_mpc --load 1.5 --fail-rate 0.30
```

## 1. Where the decision happens

`simulation.Simulator._decisions` (step 4 of the daily loop) runs one epoch per
**facility** per day, and only when that facility has a free slot and a non-empty
ready queue:

```python
free    = self.plan.fcap[m] - self.busy[m][t]     # slots free TODAY
cands   = self.candidates(m, t)                   # W_t restricted to facility m
n_start = min(free, len(cands))                   # non-idling
chosen  = policy.select(self, m, t, n_start, cands)
```

Three consequences that frame everything else:

* **The problem decomposes by facility.** `m(i)` is frozen by the strategic
  solve, so the policies never choose *where*, only *who*.
* **`n_start` is fixed before the policy is consulted.** All four online
  policies are strictly non-idling; they can only re-order the queue, never
  hold a slot back. Only `best_achievable` may idle.
* **A candidate is *ready*, not merely arrived** — `candidates()` filters on
  `ready_at[pid] <= t`, i.e. collection plus `TLS + TT1` has completed. A
  remade patient re-enters `ready_at` only after a fresh leukapheresis.

Each candidate carries `pid, tier, t0, tt3` (`per_epoch.Candidate`). `t0` is the
**first** collection day, so a remade patient carries the deterioration of every
earlier attempt into its next competition.

## 2. The three selection rules

### FIFO — `policies.Fifo.select`

```python
order = sorted(cands, key=lambda c: (c.t0, c.pid))
return [c.pid for c in order[:n_start]]
```

Arrival order on the **first** collection day, ties by patient id. Tier, accrued
wait and transport time are all invisible to it. Note it sorts on `t0`, not on
`ready_at`, so among patients ready today the one who was collected earliest
wins — which is what makes a twice-failed patient FIFO's top priority.

### Survival index — `per_epoch.index_choice` (P8)

```python
index_score = ALPHA_W[tier] * (S_i(t) - S_i(t+1))
ordered = sorted(cands, key=lambda c: (-index_score(c, t, tmfe, tqc), c.t0, c.pid))
```

where `S_i(τ) = survival(tier, τ + T_MFE + T_QC + tt3 − t0)`. The score is the
life-value-weighted survival lost to **one more day** of waiting. Ties break on
`(t0, pid)` — deterministic.

Two properties that are easy to miss and that drive the snapshots below:

* **Tier dominates, absolutely.** With `ALPHA_W = 3/2/1` and `w = 0.15/0.05/0.02`
  the per-day loss of a high-risk patient is ~24× a low-risk patient's and ~4.7×
  a medium's. Flipping that ordering on wait alone needs an elapsed spread of
  hundreds of days. In practice the rule is *lexicographic*: all H, then all M,
  then all L.
* **Inside a tier it runs the queue backwards.** `S` is convex-decreasing, so
  the one-day drop `S(e) − S(e+1)` is **largest at small elapsed**. The freshest
  patient has the steepest curve, so within a tier the index serves the
  **latest**-arriving patient first. Measured on the N=200, seed-0 run: at the
  130 contested epochs, the top-priority pick was the *minimum* elapsed within
  its tier in **124** of them. This is not a bug — it is what maximising the
  instantaneous derivative of a convex curve means — but it is the opposite of
  the "sickest-and-longest-waiting first" reading, and it is why low-risk hold
  times reach ~23 d while high-risk sits at 0.4 d.

### Adaptive MPC — `per_epoch.solve_epoch` (P1)–(P6)

A MILP over the window `[t, t+H]`, `H = 7`:

```
max  Σ_i Σ_τ  gain[i][τ] · x[i,τ],     gain[i][τ] = ALPHA_W[tier] · (S_i(τ) − S_i(t+H))
s.t. Σ_τ x[i,τ] ≤ 1                                         (P4) started at most once
     Σ_i Σ_{tp ∈ (τ−T_MFE, τ]} x[i,tp] ≤ max(free[τ], 0)    (P5) rolling-window capacity
     Σ_i x[i, t] == n_start                                  non-idling
```

`free[τ] = FCAP_m − busy_m[τ]` is read from live occupancy. Only `x[i, t]` is
implemented; the rest of the plan is discarded and re-solved next epoch. Two
short-circuits fire before the solver: an empty `cands` returns `[]`, and
`n_start >= len(cands)` returns everyone (no MILP built).

Deferral is charged at `d_i = S_i(t+H)`, so a patient the window cannot reach is
worth `gain = 0` — identical to not being scheduled.

## 3. Do the MPC and the index actually differ?

**No — not on this calibration.** They are order-equivalent by construction, and
the simulation confirms it.

### The algebra (`--theory`, read off the shipped formulas)

`cart_data.survival` is exponential: `S_u(e) = ρ_u**e`, `ρ_u = (1−w_u)**(1/42)`.
Both scores then factor through a *single* scalar `A_i = ALPHA_W[u] · S_u(e_i(t))`:

```
index_score  = A_i · (1 − ρ_u)
gain[i][τ]   = A_i · (ρ_u**(τ−t) − ρ_u**H)
```

Exchanging two candidates between today and a slot `k` days out changes the MPC
objective by `A_i(1 − ρ_i**k) − A_j(1 − ρ_j**k)`. So the MPC ranks on
`A_i · m_u(k)` with `m_u(k) = 1 − ρ_u**k`, and the index is exactly the `k = 1`
case. Therefore:

* **same tier → identical ranking** (same `ρ`, both reduce to ranking on `A_i`);
* **different tiers → the ranking can differ only if `A_i/A_j` lands in the
  band between `m_j(1)/m_i(1)` and `m_j(H)/m_i(H)`.** Those bands are 0.2–1.0 %
  wide and require an elapsed time of **588 d (H vs M), 939 d (H vs L),
  2195 d (M vs L)**. The simulation's own drain cap is 390 d and the largest
  elapsed-at-delivery observed in *any* recorded epoch — including the stressed
  runs — is **175 d**.

The MPC's look-ahead cannot rescue the difference either: `n_start` is imposed
from outside, so the extra information in `free[τ]` for `τ > t` only reorders
patients the MPC is not implementing anyway.

### The measurement

Agreement over recorded epochs (an epoch is *contested* when `|W_t| > n_start`;
otherwise every waiting patient starts and all policies trivially coincide):

| configuration | contested epochs | index ≠ MPC | of which exact ties in (P3) | genuine differences |
|---|---|---|---|---|
| N=50, seed 0 | 44 | 0 | — | **0** |
| N=200, seeds 0–2 × 3 acting policies | 1,170 | 87 (7.4 %) | 87 | **0** |
| N=200, 30 % failure rate, seeds 0–1 | 336 | 25 (7.4 %) | 25 | **0** |
| N=200, offered load 1.5 | 129 | 24 (18.6 %) | 24 | **0** |
| N=200, load 1.5 + 30 % failures | 164 | 19 (11.6 %) | 19 | **0** |

Every disagreement was scored on the MPC's *own* look-ahead objective by
re-solving the epoch MILP with today's start set pinned to the index's choice
(`epoch_objective`). The gap was **exactly 0.0 in all 155 cases**.

The disagreements are **decision-twins**: the cohort is built by tiling the
50-patient base cohort (`cart_data.build_instance`), so at N=200, **72 of 200
patients (36 %)** share their `(tier, t0, facility, tt3)` class with another
patient, in classes of up to 4. Twins have identical scores and identical gain
vectors; the index breaks the tie on `pid`, the MILP breaks it arbitrarily. At
N=50 only 8 % have a twin — and there the two policies disagree **zero** times.

### Outcome level, N=200, 10 seeds

| | index | MPC | difference |
|---|---|---|---|
| mean weighted clinical loss | 18.3305 | 18.3358 | +0.0053 (+0.03 %) |
| high-risk expected lost | identical to 4 dp in **every** seed | | 0.0000 |

No seed produced an identical **start schedule** — a twin swap happens in all
ten — yet three seeds land on the same loss to floating-point precision
(7e-15), and the rest differ in the 4th decimal with the sign flipping seed to
seed (MPC better on 4, worse on 4). That residue is **not** decision quality: CRN is keyed on `(seed, pid, attempt)`, so starting twin
`p38r2` instead of `p38r1` draws a *different* batch outcome. Breaking a tie
differently reshuffles luck. This is exactly the `18.31 vs 18.31` and
`4.244 vs 4.244` already recorded in `results/mpc/README.md` — now with the
mechanism established rather than observed.

## 4. Snapshots

All from `epochs_N200_seed0_adaptive_mpc.json` (frozen plan m1+m3, FCAP 4+10,
offered load 1.14, seed 0). `elapsed` = elapsed-at-delivery if started today =
`t + T_MFE + T_QC + tt3 − t0`; `att` = which attempt this would be.

### A — Tier ordering, all three tiers waiting (day 26, m3)

FCAP 10, busy 9 → **1 free slot**, `n_start = 1`, `|W_t| = 4`.

| pid | tier | t0 | age | tt3 | att | hold | elapsed | S(start today) | index score |
|---|---|---|---|---|---|---|---|---|---|
| p28r0 | H | 24 | 2 | 1 | 1 | 0 | 17 | 0.93634 | **0.010848** |
| p5r2 | M | 23 | 3 | 1 | 1 | 0 | 18 | 0.97826 | 0.002388 |
| p14r1 | L | 22 | 4 | 2 | 1 | 1 | 20 | 0.99043 | 0.000476 |
| p38r1 | L | 12 | 14 | 2 | 1 | 11 | 30 | 0.98567 | 0.000474 |

* **FIFO → p38r1.** Earliest `t0` (day 12), 11 days queued. Nothing else enters.
* **Survival index → p28r0.** Highest one-day weighted loss: `3 × (0.93634 −
  0.92547) = 0.010848`, 23× p38r1's. The H-tier curve is steep enough that two
  days of age beats fourteen.
* **Adaptive MPC → p28r0.** Same patient. Its coefficient for starting today is
  `3 × (S(17) − S(24)) = 0.075065`, again the largest; deferring p28r0 to the
  end of the window costs 22× what deferring p38r1 costs.

### B — Within-tier ordering is *backwards* (day 79, m3 — deepest queue in the run)

FCAP 10, busy 8 → **2 free slots**, `n_start = 2`, `|W_t| = 32`.

| pid | tier | t0 | age | hold | elapsed | index score |
|---|---|---|---|---|---|---|
| p44r3 | H | 77 | 2 | 0 | 17 | **0.010848** |
| p18r0 | M | 76 | 3 | 0 | 18 | **0.002388** |
| p18r2 | M | 76 | 3 | 0 | 18 | 0.002388 |
| p39r0 | M | 75 | 4 | 1 | 19 | 0.002385 |
| p42r3 | M | 68 | 11 | 8 | 26 | 0.002365 |
| p45r0 | L | 71 | 8 | 5 | 23 | 0.000476 |
| … 26 further waiting patients | | | | | |

* **FIFO → p49r1, p49r2** — the two earliest collections in a 32-deep queue.
* **Survival index → p18r0, p44r3.** The H patient collected **two days ago**,
  plus the M patient collected **three days ago** — the two *newest* arrivals in
  the queue. Every M patient who has waited 8–11 days is passed over, because
  their survival curve has already flattened and loses less per day.
* **Adaptive MPC → p18r0, p44r3.** Identical.

This is the clearest statement of the mechanism: the survival-aware policies buy
high-risk survival with low-risk *and stale* delay, and "stale" is doing real
work — the rule is anti-FIFO within a tier, not merely tier-weighted FIFO.

### C — A remake competing (day 27, m1)

FCAP 4, busy 3 → **1 free slot**, `n_start = 1`, `|W_t| = 4`.

| pid | tier | t0 | age | att | hold | elapsed | index score |
|---|---|---|---|---|---|---|---|
| p28r2 | H | 25 | 2 | 1 | 0 | 17 | **0.010848** |
| p14r2 | L | 22 | 5 | 1 | 2 | 21 | 0.000476 |
| p14r3 | L | 17 | 10 | 1 | 7 | 26 | 0.000475 |
| p25r2 | L | 9 | 18 | **2** | 0 | 34 | 0.000473 |

* **FIFO → p25r2.** A second-attempt patient with `t0 = 9` is FIFO's *top*
  priority — the failed batch pushed it back into the queue carrying the oldest
  collection date in the system.
* **Survival index → p28r2**, and **adaptive MPC → p28r2.** The remake is scored
  on one continuous clock from its **first** collection (elapsed 34 vs 17), so
  its already-flattened L-tier curve puts it *last*. The failure recourse costs
  it priority twice over: tier, then accumulated decay.

Neither survival-aware policy has any notion of "this patient already failed
once" beyond what the clock encodes. Whether that is the intended clinical
behaviour is a modelling question, not an implementation one.

### D — Where index and MPC diverge: an exact tie (day 17, m3)

FCAP 10, busy 9 → **1 free slot**, `n_start = 1`, `|W_t| = 6`.

| pid | tier | t0 | elapsed | index score |
|---|---|---|---|---|
| p19r0 | M | 14 | 18 | 0.0023877 |
| p12r2 | M | 14 | 18 | 0.0023877 |
| p25r0 | L | 13 | 20 | 0.0004764 |
| p38r1, p38r2, p38r3 | L | 12 | 21 | 0.0004763 |

* **FIFO → p38r1** (earliest `t0`, `pid` tie-break among three identical twins).
* **Survival index → p12r2**, **adaptive MPC → p19r0**.

p19r0 and p12r2 are decision-twins — same tier, same `t0`, same `tt3`, same
facility, bit-identical scores. The index breaks the tie on `pid`; the MILP
returns whichever optimal vertex HiGHS lands on. Re-solving the epoch with the
index's pick pinned gives objective **0.0385234524** — identical to the MPC's
own **0.0385234524**. All 155 recorded disagreements are of this form.

### E — Forced epoch (day 3, m1)

FCAP 4, busy 0 → 4 free, but `|W_t| = 2` → `n_start = 2`. Both patients start
under all three policies. 24 of the 154 epochs in this run are forced;
`solve_epoch` returns before building a MILP.

## 5. Answering the question

**Adaptive MPC does not make better per-epoch decisions than the survival index
on this model — it makes the same decisions.** That is a property of the
calibration, not a coincidence and not a coding error:

1. the survival kernel is a pure exponential, so both objectives factor through
   the same scalar `A_i = α_u S_u(e_i)`;
2. `n_start` is imposed non-idling from outside the policy, so the MPC has no
   freedom to hold capacity — the one decision where look-ahead would pay;
3. the look-ahead horizon `H = 7` equals `T_MFE`, so every day in the window is
   coupled to today by one rolling capacity row and the window carries no
   independent future decision;
4. nothing stochastic enters the epoch model — failures are observed *after* a
   start, and the MPC's window contains no failure model — so "re-optimising on
   the observed state" changes only the candidate set, which the index sees too.

The README's Exp E result (`adaptive_mpc` pulling away from `static_survival`
as failures rise, 3.85 → 6.51 vs 3.85 → 11.59) is entirely about **static
vs dynamic**, not about MILP vs index: the value comes from re-evaluating the
queue at all, and the index re-evaluates it just as adaptively at a fraction of
the cost. On this evidence the honest claim is *"the adaptive layer matters, and
a closed-form index captures all of it"* — which is a stronger, more publishable
result than a MILP that wins by 0.03 % of noise.

### What would make the MPC genuinely different

Each of these breaks one of the four conditions above, and none is a large
change to `per_epoch.py`:

* **let it idle** — drop the `Σ_i x[i,t] == n_start` equality to `≤ free[t]` and
  let it hold a slot for an arriving high-risk patient (this is the one
  scheduling freedom `best_achievable` has and the online policies do not — it
  idles 77 slot-days at N=50 — though its remaining 0.29-unit edge also buys
  foresight of which batches fail);
* **put arrivals in the window** — `candidates()` only ever passes *ready*
  patients; feeding the MPC the known pipeline (`ready_at[pid] > t`) would give
  the look-ahead something to look ahead *at*;
* **a non-exponential kernel** — any `S` whose log-derivative is not tier-constant
  (`γ ≠ 1` in `cart_data.survival`) breaks the factorisation and the two rules
  separate;
* **model the failure risk in-window** — an expected-value term `p_m` on the
  gain would let the MPC prefer a patient with slack for a remake, which the
  index cannot express.

Until one of those lands, `survival_index` should be reported as the operating
policy and `adaptive_mpc` as the certificate that it is optimal for (P1)–(P6) —
not as a separate, better policy.
