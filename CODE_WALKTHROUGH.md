# CAR-T Model — Code Walkthrough

A plain-language guide to the logic, in the order you should read the files.
Each section says what the file is *for*, walks the key functions, and ends with
the one-line **defense** you'd give if asked "what does this do and why is it right?"

The whole system in one sentence:
> An **instance** defines the world → a **deterioration kernel** + a **congestion
> function** say how waiting harms patients → a **stochastic program** designs the
> network and provably reduces to the plain yield model when decline is switched
> off → an independent **simulator** scores any design and splits every loss into
> wait-caused vs failure-caused → the `paper/` folder runs it all and draws the figures.

Two ideas recur everywhere, so hold them from the start:
- **κ (kappa)** = a single dial for "how fast patients decline while waiting."
  κ = 0 means no decline (the model collapses to the classic yield model);
  κ = 1 is the rate calibrated to real survival data.
- **Design vs. policy.** A *design* is the optimizer's output — which factories
  open, how much capacity, who is assigned where. A *policy* is a rule applied
  later — the order patients are served. Keep these separate; most confusion
  comes from mixing them.

---

## 1. `case_study.py` — the world

**Role.** Defines the concrete network the whole study runs on: 4 candidate
factories, 15 hospitals, 50 patients in a High/Medium/Low urgency mix, plus every
cost, yield, and tier parameter. Also contains a self-contained HiGHS solver used
for its own standalone analysis (VSS / recourse mix); the paper doesn't call that
solver — it uses `dynamic_sp.py` — but the **`Instance` dataclass and
`build_case_study_instance()` are the shared foundation everything imports.**

**Walk it.**
- `Instance` (dataclass) — a bag holding the network: `f` (facility opening
  cost), `pi` (per-slot capacity cost), `c` (per-batch manufacturing cost),
  `s_max` (max capacity per factory, 75 = Wan 2026's real Maxcap), `p` (per-factory
  yield = probability a batch passes), `beta` (baseline re-collection feasibility
  by tier), `rho_cancel` (the clinical loss cost of losing a patient, by tier).
- `build_case_study_instance()` — fills all those numbers in. This is where your
  parameter values live; every one traces to `calibration_table.md`.
- `_highs_solve_sp(...)` and the `vz/vC/vx/vr/vs/vc` helpers — the standalone
  solver. Those helper functions are the **variable-index map**: HiGHS wants a
  flat list of columns, so each named decision (open factory `z_m`, capacity
  `C_m`, assign `x_{im}`, remake `r`, subcontract `s`, cancel `c`) is mapped to
  an integer column position. `n_sub_pp = n_f*(n_f-1)` counts the off-diagonal
  (source ≠ destination) subcontracting pairs.

**Defense.** "The instance is a US network parameterized from Wan (2026) geography
and clinical-literature costs; every value is in the calibration table."

---

## 2. `scaled_instance.py` — turning up demand

**Role.** Makes the network *busy* without editing `case_study.py`.

**Walk it.** `scaled_case_study_instance(mult, s_max)` copies the base 50-patient
instance and **tiles the cohort `mult` times** (mult = 3 → 150 patients), keeping
the same tier mix and geography, and sets capacity to the real `s_max = 75`.

**Why it matters.** Crowding-driven delay only bites near capacity. The honest way
to reach that regime is to *raise demand at real capacity* (this file), not to
shrink capacity artificially. The paper runs at `mult = 3`.

**Defense.** "We reach the capacity-tight regime by scaling demand to 150 patients
at the real per-factory capacity, not by rigging a small capacity."

---

## 3. `declineprob.py` — the deterioration kernel (the heart)

**Role.** Turns "waiting" into "probability the patient is still treatable." This
is the core novelty; know it cold.

**Walk it.**
- `h0(t) = (t/λ)^γ` — the Weibull baseline hazard. γ ("gamma") sets how sharply
  risk accelerates with waiting; λ ("lambda") is the time-scale.
- `calibrate_lambda(γ)` — the one line that makes the model non-tunable. It solves
  λ so that a patient experiencing the *observed* extra delay accrues *exactly* the
  *observed* published hazard ratio: `target = ln(HR_PFS)/HR_anchor`, then
  `λ = Δ / target^(1/γ)`. Δ = 12 days (the extra vein-to-vein time when a product
  is remade) and HR_PFS = 1.64 (delayed-vs-on-schedule survival) are the two
  published anchors. λ is recomputed for each γ (36.4 / 25.1 / 20.9 days).
- `extra_survival(delay) = exp(−κ·HR·(delay/λ)^γ)` — survival of the **extra**
  delay only (the days beyond the normal schedule). At delay 0 it is 1. The word
  "extra" is doing real work: this is the *new* piece the model adds.
- `still_eligible_prob = β_u × extra_survival` — total probability the patient can
  still be re-collected: baseline feasibility β_u (the old v1 term) times the
  survival of the extra delay. At κ = 0 this is exactly β_u → the classic model.

**Defense.** "λ is not fitted — it's pinned by requiring the model to reproduce a
published hazard ratio at a published delay. κ = 0 recovers the classic model."

---

## 4. `clearing_function.py` — congestion → delay

**Role.** A busy factory is a slow factory. This converts how full a factory is
into how many extra days a remake takes.

**Walk it.** `ClearingFunction.remake_delay(rho)` where `rho` = utilization
(slots used / capacity). It returns `tau_proc` (the baseline ~19-day remake time)
plus a **convex, piecewise-linear** climb as `rho` rises past breakpoints — so
delay grows slowly when the factory is quiet and steeply as it saturates. Below
the first breakpoint the delay is just `tau_proc` (no congestion penalty).

**Status to be honest about.** `tau_proc` is literature-anchored; the breakpoints
and slopes are a **declared structural assumption** (no public factory
congestion-vs-delay dataset exists) and are flagged as sensitivity-tested.

**Defense.** "Standard clearing-function shape from production planning; the one
assumption (the congestion curve) is disclosed and sensitivity-tested."

---

## 5. `yield_sp_v1.py` — the original model (your baseline)

**Role.** The plain two-stage stochastic program with **no deterioration**:
eligibility is a fixed coin-flip `B ~ Bernoulli(β_u)`, independent of time. Read
this before the dynamic model, because the dynamic one is defined as "this, with
one thing swapped." It's also your **nesting anchor**.

**Walk it.**
- `Instance` — same idea as case_study's, for the toy/validation examples.
- `sample_scenarios(instance, n, seed)` — draws `n` Monte-Carlo scenarios of
  which batches pass (`Y`) and which patients stay eligible (`B`). *Pure numpy —
  the paper imports only this and the `Instance` type.*
- `build_sp_model(...)` / `build_and_solve_sp(...)` — build and solve the v1 MILP
  (Gurobi). Stage 1 chooses `z, C, x`; stage 2 chooses recourse per scenario.
- `verify_toy_example()` — reproduces the worked example, the sanity check.

**Note.** The Gurobi import is what breaks figure regeneration on machines without
a license (the paper only needs `sample_scenarios`/`Instance`, which are pure) —
that's the known PNG issue; making the Gurobi import lazy fixes it.

**Defense.** "v1 is the classic yield model; it's the baseline our dynamic model
must reduce to when decline is off."

---

## 6. `dynamic_sp.py` — your model

**Role.** v1 with the fixed eligibility coin-flip **replaced by an outcome of the
decline kernel**, evaluated at the patient's accrued wait (including congestion
delay). This is the optimization contribution.

**Walk it.**
- `_still_from_assignment(...)` — the core computation. For each scenario: find
  which routed batches failed (`F`), compute each factory's utilization `rho` =
  (patients routed + failure surge) / capacity, turn that into remake days via the
  clearing function, and set `StillEligible = B AND Bernoulli(extra_survival(wait))`.
  At κ = 0 the survival factor is 1, so `StillEligible == B` → exactly v1.
- `solve_dynamic_sp(...)` — the design optimizer for a *fixed* eligibility
  (HiGHS). Same variable-index map pattern as case_study.
- `solve_dynamic_sp_endogenous(...)` — the important one. A design's capacity
  changes congestion, which changes delay, which changes eligibility — a feedback
  loop. This handles it by a **fixed-point iteration**: start from the κ = 0
  design, build a convex surrogate for delay-induced loss as a function of
  capacity (`_delay_surrogate_curves`, kept linear via `_lower_convex_hull`),
  re-solve, and repeat until the assignment stops changing. It returns whichever
  visited design has the lowest cost under its *own* true eligibility — so the
  endogenous design is never worse than the exogenous one, and at κ = 0 they're
  identical.

**Defense.** "The dynamic model nests v1 (κ = 0) exactly; the endogenous solve lets
the design respond to the congestion it creates, via a convex capacity surrogate
that keeps the problem linear."

---

## 7. `value_of_endogeneity.py` — does decline change the *design*?

**Role.** The experiment that answers the "is it real optimization or just a
heuristic?" question. It compares two designs under identical true dynamics:
`D_exo` (built ignoring decline, κ = 0) vs `D_endo` (built accounting for it).

**Walk it.**
- `simulate_endogenous(design, ...)` — scores a *fixed* design under the true
  loop: yields → decision-dependent congestion (`rho` from the design's own
  capacity and routing) → delay → decline → recourse LP. Returns cost and tier-H
  mortality.
- `run_experiment()` — sweeps κ and γ, computing `VoE = cost(D_exo) − cost(D_endo)`.
- `_solve_recourse(...)` — the pure stage-2 LP (remake / subcontract / cancel) for
  a given eligibility; **`patient_simulator.py` imports this**, so this file is
  load-bearing, not optional.

**What it shows.** With capacity headroom, `D_endo` **buys spare capacity** (e.g.
10 → 20 slots) to cushion decline, and VoE > 0 — decline genuinely changes the
optimal design. In the saturated base case there's no room, so `D_endo == D_exo`
and VoE = 0; there the value is *corrected prediction*, not a changed design.

**Defense.** "When there's capacity slack, planning for decline changes the optimal
design and saves cost/lives — that's the endogenous optimization result."

---

## 8. `patient_simulator.py` — scoring designs out of sample (where the numbers come from)

**Role.** Takes any design and measures what actually happens to patients on a
fresh scenario set, tags every loss by cause, and applies the priority rule. Your
presented numbers come from here.

**Walk it.**
- `_fifo_survival_signal(...)` — each patient's survival probability at therapy
  under first-come-first-served (lower = sicker). This is the ranking number.
- `_rule_exposure(...)` — turns that ranking into a **wait multiplier** `e_i`
  (mean 1). FIFO → everyone `e = 1`. Sickest-first → sort lowest-survival-first
  and spread multipliers linearly from 0 (served first, no wait) up to 2 (served
  last, double wait). The mean stays 1, so prioritizing *reallocates* waiting, it
  doesn't erase it — the sickest's short wait is paid for by the healthiest's
  longer wait.
- `simulate_patients(...)` — the main routine:
  - normal-wait survival `surv_norm = exp(−κ·h_norm·T_v2v·e_i)` faces **every**
    patient (the ordinary vein-to-vein wait, shortened for the sickest by `e_i`);
  - a failed patient additionally faces `surv_fail = surv_norm ×
    exp(−κ·HR·(extra/λ)^γ)`, where `extra` = base remake delay + its share of
    congestion — the two hazards **compound**;
  - each loss is tagged **cause (a)** lost during the normal wait (batch
    succeeded) or **cause (b)** lost after a failure;
  - it then solves the recourse LP for failed patients and totals cost.

**Defense.** "Independent out-of-sample scoring; survival is normal-wait and
extra-delay hazards compounded; every loss is attributed to wait vs. failure, and
prioritization only reshuffles waiting (mean exposure 1)."

---

## 9. `deterioration_experiment.py` — planned vs. simulated (validation support)

**Role.** Checks the fast planning-model estimate of delay-cost against the
detailed simulator, and sweeps decline speed. Its `planned_vs_simulated_deaths`
is imported by the paper pipeline to draw technical figure T1.

**Defense.** "Confirms the surrogate used inside the optimizer matches the detailed
simulation — the design isn't optimizing against a distorted proxy."

---

## 10. `paper/src/` — orchestration and figures (read last; no model logic)

- **`experiments.py`** — runs every experiment (E1 prediction gap, E2 loss causes,
  E3 three plans, E4 capacity sweep, E5 robustness, E6 priority-rule family, the
  γ / hazard-ratio sensitivities, and the failure×wait regime map). `_sim(...)` is
  a thin wrapper around `simulate_patients`; `priority=False/True` is the
  FIFO-vs-sickest-first switch.
- **`figures.py`** — draws F1–F7 and T1–T3 from the results payload. Labels only.
- **`emit.py`** — writes the CSVs, LaTeX tables, and `numbers.md` traceability
  ledger.
- **`make_assets.py`** — one command to run it all (`--figures-only` rebuilds from
  cached raw data).

**Defense.** "Everything in `paper/` just runs the model stack and renders it;
`numbers.md` traces every cited value back to a seeded experiment."

---

## Suggested first pass (30 minutes)

Read the **docstring + the one starred function** in each file, in order:
`build_case_study_instance` → `scaled_case_study_instance` → `calibrate_lambda` +
`extra_survival` → `remake_delay` → `sample_scenarios` → `_still_from_assignment`
+ `solve_dynamic_sp_endogenous` → `simulate_endogenous` → `simulate_patients`.
That path is the entire logic; everything else is plumbing or presentation.

## For the MDP work

The pieces that carry over unchanged as your environment: `declineprob.py` (state
transition — how health decays with wait), `clearing_function.py` (how the queue
creates delay), and `patient_simulator.py` (the evaluator). What changes is the
**decision**: today `_rule_exposure` imposes a *fixed* serving order; the MDP
replaces that with a policy chosen by solving the sequential problem.
