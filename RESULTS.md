# Dynamic Semi-Markov Decline-Probability Extension — Results

Extension of the v1 two-stage CAR-T stochastic program (`yield_sp_v1.py`) into a
**dynamic** model with **endogenous, delay-driven clinical deterioration**: the
re-collection eligibility that v1 draws as a time-independent
`B_i(ω) ~ Bernoulli(β_u)` is replaced by `StillEligible_i(ω)`, an outcome of a
proportional-hazards decline kernel evaluated at the patient's accrued wait,
where the wait includes congestion-driven re-manufacture delay.

Reproduce everything with `python run_all.py` (add `--quick` for a fast smoke
run). All randomness is seeded (in-sample seed 0; out-of-sample seed 100).
`yield_sp_v1.py`, `case_study.py`, and the disruption experiment are unchanged.

---

## 1. Nesting validation (gate) — PASS

`test_nesting.py`: with `kappa = 0` the decline kernel's extra-survival factor is
identically 1, so `StillEligible == B` for any input (including the hand-crafted
toy scenarios), and `dynamic_sp` reproduces v1 exactly on the toy instance:

| Quantity | Target (v1 §15) | dynamic_sp (kappa=0) |
|---|---|---|
| Stochastic SP total cost | 2.774 | **2.7738** |
| Deterministic baseline | 2.620 | **2.6200** |
| VSS (SP − deterministic) | 0.154 | **0.1538** |
| First-stage design (free solve) | z=[0,1], C=[0,3] | **identical** |

Fixed- and free-first-stage solves match v1 to `< 1e-6`. The dynamic model is a
strict superset of v1.

---

## 2. Calibration (sources match `calibration_table.md`)

### Decline kernel `declineprob.py` (Weibull proportional hazards)
`H0(t) = (t/λ)^γ`, per-epoch decline `1 − exp(−HR_u·[H0(t_k) − H0(t_{k−1})])`,
and `StillEligibleProb_u(delay) = β_u · exp(−κ·HR_u·(delay/λ)^γ)`.

| Parameter | Value | Status | Source |
|---|---|---|---|
| β_u (H/M/L) | 0.55 / 0.70 / 0.80 | anchor | Roth et al. 2018 (case study supplies its own β via the Instance) |
| HR_u (H/M/L) | 1.6 / 1.5 / 1.4 | **assumption (sensitivity-tested)** | ordered within the observed 1.4–1.6 delay-HR band (Dulobdas 2025 / Cohet 2023) |
| λ (Weibull scale) | **36.4 / 25.1 / 20.9 d** (γ=1.0 / 1.5 / 2.0) | calibrated | set so the mid-tier decline over the observed +12 d remake delay (Cohet 2023) reproduces the delayed-vs-control PFS hazard ratio 1.64 (Dulobdas 2025): `HR_M·(12/λ)^γ = ln 1.64` |
| γ (Weibull shape) | swept {1.0, 1.5, 2.0} | **NOT fitted** | not identified from two endpoints; γ=1 = memoryless, γ>1 = accelerating. Treated as a sensitivity parameter, **not** claimed as calibrated |

Verified: mid-tier survival over the 12-day anchor delay = `1/1.64 = 0.6098`
exactly for every γ (`test_declineprob.py`).

### Clearing function `clearing_function.py` (congestion → delay)
`RemakeDelay(ρ) = τ_proc + Σ_s a_s·max(0, ρ − b_{s−1})`, convex.

| Parameter | Value | Status | Source |
|---|---|---|---|
| τ_proc | 19 d | literature | baseline re-manufacture cycle, Avramescu et al. 2022/2023 |
| breakpoints b_s | (0.70, 0.90) | **assumption (sensitivity-tested)** | no public CDMO congestion data; declared structural assumption |
| slopes a_s | (0, 45, 90) d/util | **assumption (sensitivity-tested)** | convex; tuned so a fully utilized facility ≈ doubles turnaround (~41 d) |

`ρ_m` is facility utilization = (primary batches routed to m + per-scenario
failure surge) / capacity, so a re-manufacture queues behind a near-full line.

---

## 3. Value of Endogeneity (`value_of_endogeneity.py`)

`D_exo` = optimal design from the exogenous model (κ=0 == v1); `D_endo` =
optimal design from the endogenous model. Both are scored under a **common
endogenous Monte-Carlo simulator** (2000 OOS scenarios, seed 100) that applies
the true dynamics: primary yields → decision-dependent facility congestion →
re-manufacture delay → PH decline → forced cancellation. A cancelled patient is
untreated (mortality proxy), decomposed by tier.

`VoE = Cost(D_exo | true) − Cost(D_endo | true)`.

### Headline numbers (γ=1.0 unless noted)

| Effect | Result |
|---|---|
| **Optimism bias (dominant effect)** | Ignoring delay makes the exogenous model predict **18.0 M USD**, but the design's **true** cost under delay dynamics rises to **20–22.5 M USD (up to +25 %)** as κ grows. |
| **True tier-H mortality** | rises from **0.21** (the flat value the exogenous model assumes) to **0.48 cancellations/scenario — a ~2.3× under-estimate** the exogenous model is blind to (figure D2). |
| **VoE (design value)** | **positive but modest**, peaking at **0.44 M USD** (κ=0.25, γ=2.0) and **declining to 0 at high κ** (figure D1). `D_endo` buys spare capacity at the non-saturated facility m0 (C: 10→15→20) to cut its congestion delay. |
| **Tier-H mortality reduction from D_endo** | **≈ 0 in this calibration** — see structural caveat below. |

The VoE curve is an **inverted U**: at low κ delay barely matters (VoE→0); at
moderate κ the endogenous design profitably provisions spare capacity; at high κ
delay is so damaging that no capacity can restore eligibility, so again VoE→0.

### Why tier-H mortality is not reduced by D_endo (structural finding)
Tier-H patients are concentrated at the single fully-automated, highest-yield
facility **m2 (p=0.95)**, which is pinned at **s_max = 40 slots** and already
serves ~40 patients (ρ≈1). The only congestion-relieving lever — spare capacity —
is therefore **unavailable exactly where tier-H sit**: m2 cannot contract beyond
s_max, and moving tier-H to a facility with slack would raise their failure rate
(lower yield). D_endo instead relieves congestion for tier-M/L patients at m0,
which drives the (modest) cost VoE. The scientifically honest reading: **the
value of endogeneity here is primarily corrected risk prediction** (the
exogenous model badly under-states true cost and tier-H mortality), with a
smaller design-improvement component that is bounded by the s_max cap on the
tier-H facility. This is a property of the current calibration, not of the model.

---

## 4. Flagged assumptions (sensitivity-tested)

These are declared assumptions, not fitted quantities, and are exposed as
sweepable parameters in code:

1. **γ (Weibull shape)** — swept {1.0, 1.5, 2.0}; λ is recomputed per γ. Larger γ
   (accelerating decline) raises true tier-H mortality faster in κ (figure D2)
   and shifts the VoE peak (figure D1).
2. **Clearing-function breakpoints/slopes** — no public CDMO data; the defaults
   (0.70, 0.90) / (0, 45, 90) are a declared structural assumption. `ClearingFunction`
   exposes them for sweeps.
3. **Tier hazard ratios HR_u (1.6/1.5/1.4)** — only the ordering and the 1.4–1.6
   band are data-anchored; the exact spread is an assumption.
4. **Tier cancellation loss ρ_cancel (6/3/1 M USD)** — order-of-magnitude
   assumptions (calibration_table.md citation gap); they weight the delay-mortality
   surrogate and hence D_endo's provisioning.

---

## 5. Files

| File | Role |
|---|---|
| `declineprob.py` / `test_declineprob.py` | PH decline kernel + 7 unit tests (γ=1 memoryless, anchor recovery, κ=0 nesting) |
| `clearing_function.py` / `test_clearing.py` | convex congestion→delay + 6 unit tests |
| `dynamic_sp.py` | dynamic SAA MILP (HiGHS); `solve_dynamic_sp` (kappa knob), `solve_dynamic_sp_endogenous` (fixed-point + convex delay-mortality surrogate) |
| `test_nesting.py` | κ=0 reproduces v1 on the toy (gate) |
| `value_of_endogeneity.py` | D_exo vs D_endo under the common endogenous simulator; κ/γ sweep; figures D1/D2; results JSON |
| `run_all.py` | one-command reproduction (tests → nesting gate → experiment) |
| `figures/figureD1_voe_vs_kappa.{png,pdf}` | VoE vs κ (per γ) |
| `figures/figureD2_tierH_mortality_vs_kappa.{png,pdf}` | true tier-H mortality vs the exogenous assumption |
| `results/value_of_endogeneity_results.json` | full sweep numbers |
