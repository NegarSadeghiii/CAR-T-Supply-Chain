# Technical Appendix — validation and internal-model diagnostics

Validation checks and internal-parameter curves for the dynamic
deterioration model behind `RESULTS.md`. Technical parameter names are used here
and in code comments only; the main report uses plain healthcare language.

Glossary (technical ↔ plain):

| Technical | Plain (RESULTS.md) |
|---|---|
| `kappa` (delay-sensitivity) | how fast patients decline while waiting |
| `gamma` (Weibull shape) | internal decline-curve shape (swept, not fitted) |
| `D_exo` (kappa=0 optimum) | on-time plan |
| `D_endo` (endogenous optimum, fixed-point + delay-mortality surrogate) | decline-aware plan |
| tier-H / M / L cancellation | high- / medium- / low-urgency patients not treated in time |
| VoE = cost(D_exo) − cost(D_endo) under true dynamics | cost saved by planning for decline |
| calibrated `kappa = 1.0` | real decline rate (Dulobdas 2025 PFS HR 1.64) |
| cause (a): baseline-hazard death on a successful batch | lost during the normal wait |
| cause (b): post-failure re-make + congestion cancellation | lost after a manufacturing failure |
| priority exposure reallocation `e_i` (sickest-first) | putting the sickest first |

Instances: case-study network tiled to 50 / 100 / 150 patients
(`scaled_instance.py`), per-facility capacity `s_max ∈ {40, 55, 75}` (75 = Wan
2026 Maxcap). Backend HiGHS; design seed 0, out-of-sample seed 100; N_train=80
scenarios, N_oos=500. Calibration (`declineprob.py`): Weibull scale
`lambda(gamma)` = 36.4 / 25.1 / 20.9 d for gamma = 1.0 / 1.5 / 2.0, fixed so the
mid-tier decline over the +12 d re-manufacture delay (Cohet 2023) reproduces the
delayed-vs-control PFS hazard ratio 1.64 (Dulobdas 2025); tier hazard ratios
1.6 / 1.5 / 1.4; gamma swept, NOT fitted.

---

## 1. No-deterioration match (nesting): `kappa = 0` reproduces the original model

With the delay-sensitivity knob `kappa = 0`, the endogenous eligibility collapses
to the original time-independent Bernoulli, so `D_endo` must equal `D_exo` and the
dynamic SP must reproduce v1 exactly.

- **Toy instance** (`test_nesting.py`): total cost 2.7738 (target 2.774), VSS
  0.1538 (target 0.154), identical first-stage design, matches v1 to `<1e-6`.
- **Scaled case study** (`figures/technical/figureT1_nesting_kappa0.png`): at
  `kappa = 0`, `D_exo` and `D_endo` give identical total cost (49.703 M USD),
  identical high-urgency deaths (0.842), and identical spare capacity (0). The
  extension is a strict superset of v1.

---

## 2. Planning-model estimate vs detailed patient simulation

The endogenous design solve internalizes crowding deaths through a convex
delay-mortality **surrogate** on facility capacity: an expected delay-induced
cancellation cost `g_m(C_m)` that decreases as capacity is added (lower
utilization → shorter re-manufacture delay → higher post-delay eligibility). This
check confirms the surrogate tracks the detailed patient simulation.

Taking the on-time design and progressively adding spare capacity at its busiest
facility (`figures/technical/figureT2_planned_vs_simulated.png`), the planning
model's internal estimate of delay-caused cancellation cost tracks the simulated
value closely and monotonically:

| spare slots added | planned (surrogate) | simulated |
|---|---|---|
| 0 | 5.43 | 5.15 |
| 8 | 4.40 | 4.34 |
| 16 | 3.92 | 3.77 |
| 24 | 3.62 | 3.47 |

The surrogate slightly over-estimates the level (it prices each facility's
patients independently) but reproduces the shape and the marginal benefit of
spare capacity, which is what drives the design. This justifies using it inside
the MILP rather than an intractable bilinear congestion term.

---

## 3. Internal benefit-vs-decline-speed curve (inverted-U)

VoE = cost(`D_exo`) − cost(`D_endo`) under the true dynamics, vs `kappa`, in the
setting with capacity headroom (50-patient network, `s_max = 75`)
(`figures/technical/figureT3_value_of_endogeneity.png`):

- **Inverted-U in `kappa`.** At low `kappa` delay barely matters (VoE ≈ 0); at
  moderate `kappa` the endogenous design profitably buys spare capacity (peak VoE
  ≈ 0.43 M USD at `kappa ≈ 1` for gamma = 1; ≈ 1.2–2.0 M for gamma = 1.5–2.0 at
  `kappa ≈ 0.25`); at high `kappa` delay is so severe that no reachable amount of
  spare restores eligibility, so VoE falls back toward 0.
- **Saturation caveat.** In the capacity-tight 150-patient network every facility
  is pinned at `s_max`, so no spare can be built and VoE ≡ 0 for all `kappa`; the
  value of the endogenous model there is corrected prediction, not a changed
  design. The inverted-U therefore requires headroom (a facility below `s_max`).
- **gamma is swept, not fitted** — larger gamma (more accelerating decline) shifts
  the VoE peak to lower `kappa` and raises it.

---

## 4. Patient-level cause split and the sickest-first lever

`patient_simulator.py` scores a fixed design on the out-of-sample scenarios and
tags every tier-H/M/L cancellation by cause, and adds the priority lever behind
RESULTS.md Steps 1–2 (`causes_priority_experiment.py`, `figureP1–P6`).

**Two death channels.**

- **Cause (a) — baseline-hazard death on a successful batch.** Independent of the
  design: a patient of tier `t` survives the standard wait with probability
  `exp(-kappa · h0_t · e_i)`, where `h0_t = -ln(1 − p_t)` and
  `p_t = BASE_WAIT_DEATH_6WK[t]` is the flagged 6-week baseline mortality
  (`{H:0.15, M:0.05, L:0.02}`). `e_i` is the wait-exposure (below). Crowding /
  capacity do NOT enter (a) — only the priority lever, through `e_i`, does.
- **Cause (b) — post-failure re-make + congestion cancellation.** For a failed
  batch (`Y=0`) the re-collection-eligibility draw uses the calibrated PH kernel
  at wait `tau_proc + D_m · e_i`, where the per-facility re-make backlog
  `D_m = max(0, clearing.remake_delay(rho_m) − tau_proc)` and
  `rho_m = (assigned_m + failures_m)/C_m`. Survivors feed the same recourse LP as
  v1; its cancellations are cause (b). Spare capacity lowers `rho_m` and so lowers
  (b), but never (a).

**Nesting (`kappa = 0`).** `surv_norm ≡ 1` ⇒ cause (a) = 0 for all tiers, and the
post-failure survival ≡ 1 ⇒ the recourse LP reduces to v1's time-independent
Bernoulli. The simulator then reproduces the original model exactly: at
`kappa = 0` the busy 150-patient network gives high-urgency lost 0.842 and total
cost 49.70 M USD (50-patient: 17.51 M USD), matching `dynamic_sp(kappa=0)` and v1.

**Priority rule family (`rule` / `threshold`) — wait-exposure reallocation.**
Generalizes the earlier sickest-first switch to the prioritization framework of
Tseng et al. (2024) (SimPAC). The priority SIGNAL is each patient's *survival
rate* under FIFO (expected survival at therapy, `_fifo_survival_signal`);
urgency tiers only supply heterogeneous decline and clinical value.

- `rule = "FIFO"` → `e_i ≡ 1` (equal backlog sharing) → reproduces the
  no-prioritization model exactly.
- `rule = "THRESHOLD"`, cutoff `X` → patients with survival rate `< X` form the
  priority group, served lowest-survival-first: they take the front (fast) slots
  with `e_i` ramping `0 → 2(k-1)/(n-1)`, and the rest (FIFO) share the leftover
  waiting uniformly at `e_i > 1`. Per-facility mean exposure is exactly 1
  (budget-neutral: reallocates waiting, never creates/removes it).
- `threshold = 1.0` puts everyone in the priority group ⇒ pure
  lowest-survival-first. Within a facility the survival signal is monotone in
  tier (H lowest), so this ordering coincides with the tier ordering and
  `threshold = 1.0` reproduces the earlier sickest-first exposure exactly.

**Validation of the rule family** (busy 150-patient network, real rate; E6 /
`paper/`): `rule = "FIFO"` reproduces the no-prioritization outcome (high-urgency
lost 5.86, identical cost); `threshold = 1.0` (Threshold-100%) reproduces
sickest-first (high-urgency lost 5.86 → 2.39); and at decline speed 0 the survival
signal is 1 for everyone, so no patient is prioritized and every rule collapses to
the base model. `priority=True/False` remain as backward-compatible aliases for
Threshold-100% / FIFO.

---

## 5. Flagged assumptions (sensitivity-tested)

- **`gamma` (Weibull shape)** — swept {1.0, 1.5, 2.0}; `lambda` recomputed per
  gamma. Not identified from two endpoints; not claimed as fitted.
- **Clearing-function breakpoints/slopes** (`clearing_function.py`, defaults
  (0.70, 0.90) / (0, 45, 90) d, `tau_proc = 19` d) — declared structural
  assumption; no public CDMO congestion data. Exposed as sweepable parameters.
- **Tier hazard ratios** (1.6 / 1.5 / 1.4) — only ordering and the 1.4–1.6 band
  are data-anchored; the spread is an assumption.
- **Tier cancellation loss** (6 / 3 / 1 M USD) — order-of-magnitude assumptions;
  they weight the surrogate and thus how much spare the decline-aware plan builds.
- **Baseline 6-week wait mortality** `BASE_WAIT_DEATH_6WK` (`{H:0.15, M:0.05,
  L:0.02}`) — FLAGGED CLINICAL ASSUMPTION driving cause (a). Order-of-magnitude
  estimates of progression risk during the standard manufacturing wait; the
  ordering (high-urgency worst) is data-anchored, the levels are sensitivity-
  testable. Sets the size of the "normal wait" slice in Step 1 and the ceiling on
  what the sickest-first lever can save in Step 2.

---

## 6. Files

| File | Role |
|---|---|
| `patient_simulator.py` | patient-level cause split (a)/(b) + sickest-first lever (Steps 1–2) |
| `causes_priority_experiment.py` | main experiment: cause split, three-plan comparison, cap sweep |
| `causes_priority_figures.py` | plain-language main figures `figureP1–P6` |
| `results/causes_priority_results.json` | full numeric record for RESULTS.md |
| `scaled_instance.py` | tiled, capacity-parameterized case-study instances |
| `deterioration_experiment.py` | technical validation rerun (nesting, surrogate, benefit curve → `figures/technical/`) |
| `deterioration_figures.py` | technical (`figures/technical/`) validation figures |
| `results/deterioration_results.json` | technical numeric record |
| `dynamic_sp.py`, `declineprob.py`, `clearing_function.py`, `value_of_endogeneity.py` | model, kernels, planning simulator (unchanged core; see their own docstrings) |
| `test_nesting.py`, `test_declineprob.py`, `test_clearing.py` | validation tests |

Reproduce everything (unit tests → nesting gate → causes+priority experiment →
technical validation → figures): `python run_all.py`.
