# Paper 1: experiment log (eligibility-aware CAR-T scheduling)

Last updated: 2026-11-12. This log records what was changed in the model and the data, and what each change did to the result.

## Starting point

- **Model:** "Updated Model_Patient tiers_revised_v3_pure.docx". Deterministic time-indexed MILP extending i-SHIPMENT, equations (1)–(22).
  - Baseline = objective (1): minimise cost only, no ordering rule.
  - Patient-aware = objective (2): cost + λ_D·E_D + λ_U·E_U, with λ_U = 0 in all runs so far.
  - Clinical states follow Liu (2021): q_RU = 0.01, q_RD = 0.003, q_UD = 0.02 per day.
- **Data:** the UK i-SHIPMENT instance.
- **Problem:** the health effect was small, with a maximum gain of about 0.3–0.6 expected patients out of 30.
- **Diagnosis:**
  - The model controls at most 7 days per patient (ND 24 minus the fastest turnaround of 17).
  - Transport time depended only on mode, so the choice of facility changed cost but not time.
  - Capacity was not competitive.
  - This is a structural limit, not a bug.

## How the runs are solved

- Each run is a single-objective MILP solved with HiGHS. This is a weighted-sum approach, not an ε-constraint method.
- **Baseline:** min cost (λ_D = 0).
- **Patient-aware:** min cost + λ_D·E_D.
- **"Best achievable":** min E_D only.
  - Earlier logs and outputs call this "upper bound". That name is wrong: it is a lower bound on E_D (and an upper bound on the possible gain). It has been renamed.
- **Tie range:** cost fixed at C*, then min and max E_D.
- **E_D** is the sum of each patient's probability of being ineligible on the delivery day. Because INH is binary, each patient contributes exactly one term. By linearity of expectation this equals the expected number of ineligible patients.
- **Reporting:** Difference = E_D Baseline − E_D Patient-aware; relative reduction = Difference ÷ E_D Baseline.

## Rule learned

**Health effect ≈ (queue) × (window) × (difference between patients).**

Three things are needed together:

1. **A queue.** Capacity must be below demand, so that someone has to wait.
2. **A window.** The waiting must be allowed to be long; this is set by ND minus the fixed time.
3. **Heterogeneity.** Waiting must hurt urgent patients more than regular ones (about 1.33%/day vs 0.53%/day around days 17–24).

Other findings:

- **Theoretical floor:** if every patient is delivered at the fastest possible turnaround, E_D = Σ π(g, minTRT, D).
  - This gives 4.07 for T_MFE 7 (minTRT 17) and 5.68 for T_MFE 14 (minTRT 24).
  - It is a provable lower bound, because D is absorbing, so a patient's risk can only grow with delivery time.
  - The model's best-achievable E_D sits above this floor because of capacity.
- The Liu rates are not raised. They already give about 15% loss over about 20 days, while Kwon (2023) reports 15% over a median of 41 days, so the Liu rates are harsher than real data.
- A manufacturing time that is random per patient but known in advance does not widen the window. A facility-dependent manufacturing time would.
- **The value of health-aware scheduling is non-monotone in congestion.** It is small with no queue, largest at intermediate congestion, and smaller again when every facility is congested.
- **Limitation:** the model is open-loop. It uses only the expected state at delivery, and the intermediate states do not affect any decision. Reacting to state changes would need a dynamic model (MDP or rolling-horizon re-optimisation), which is the natural next paper.

## Changes made, in order

### 1. Data: UK → US instance; model: route-dependent transport time

- **Sources:**
  - Wan et al. (2026), EJOR 333(3):807–822, shared data.
  - Avramescu et al. (2021), EvoApplications, US Kymriah centres.
  - Transport time = Haversine distance / 110 km/h.
- **Hospitals:** 10 of the 55 centres:
  - East: Boston, New York, Philadelphia, Baltimore, Durham, Chicago;
  - Central: Houston, Denver;
  - West: Los Angeles, Palo Alto.
- **Facilities:** fixed, at Wan candidate cities: Philadelphia (4 lines), Houston (10), Los Angeles (10). Line counts come from i-SHIPMENT.
- **Transport time rules:**
  - Fast mode: 1 day if the distance is ≤ 1,500 km, otherwise 2 days. World Courier reports 36–48 h.
  - Slow mode: ceil(driving hours / 10) days.
- **Transport cost:** three TrakCel tiers from i-SHIPMENT, assigned by distance.
- **Patients:** 30 patients; 18 of them in the East.
- **Model change:** TT1_j → TT1_cmj and TT2_j → TT2_mhj in eqs. (4), (10), (14).

### 2. Data: urgent share

- **Settings:** base 20%, scenarios 10% and 30%; the urgent sets are nested.
- **Justification:** Kwon et al. (2023), high-risk features at apheresis ranged from 6% to 60%.
- **All runs below use 30% urgent:** 9 patients, 6 of them in the East.

### 3. Run A: ND = 24, Houston 10 lines

| Quantity | Value |
|---|---|
| Baseline E_D | 4.5243 |
| Best achievable | 4.2027 |
| Gain at λ_D = 150k | 0.2459 |
| Extra cost | 7,876 USD |

### 4. Run B: ND = 40, Houston 10 lines (only ND changed)

| Quantity | Value |
|---|---|
| Baseline E_D | 4.5905 |
| Patient-aware E_D | 4.2784 |
| Gain | 0.3121 (6.8%) |
| Extra cost | 9,099 USD |
| Baseline wait | mean 1.50 days, max 10 |
| Tie range | 4.4919–4.6657 |

- **Lesson:** ND opens the window, but capacity was still loose, so the effect barely changed.

### 5. Run C (base case): ND = 40, Houston 4 lines

| Quantity | Value |
|---|---|
| Baseline E_D | 5.0564 |
| Patient-aware E_D | 4.4341 |
| Gain | 0.6223 (12.3%) |
| Extra cost | 16,832 USD |
| Best achievable | 4.2904 |
| Tie range | 4.7443–5.1867 |

- **Mechanism (rerouting):**
  - In the Baseline, the urgent patients p13, p16, p21 and p30 each wait 11–13 days at Philadelphia.
  - Patient-aware sends regular East patients to Houston and Los Angeles, which frees Philadelphia for the urgent ones.
  - Waiting at Philadelphia drops from mean 6.5 / max 18 days to mean 2.6 / max 8 days.

### 6. Run D: ND = 40, all facilities 4 lines

| Quantity | Value |
|---|---|
| Baseline E_D | 4.9928 (lucky tie; range 4.8882–5.3209) |
| Patient-aware E_D | 4.7035 |
| Best achievable | 4.6796 |

- **Mechanism (reordering the queue):** urgent wait 3.22 → 1.11 days, regular 4.10 → 3.81 days.
- **Why the gain shrank:** the floor rose, so losses become unavoidable.

### 7. Six-scenario comparison (ND = 40, Philadelphia 4, LA 10, 30% urgent, λ_D = 150k)

| | H10/T7 | H4/T7 | H2/T7 | H4/T14 | H2/T14 | H10/T5 |
|---|---|---|---|---|---|---|
| E_D, Baseline | 4.5905 | 5.0564 | 5.2263 | 6.8094 | 7.0220 | 4.2589 |
| E_D, Patient-aware | 4.2784 | 4.4341 | 4.5644 | 6.5381 | 6.7850 | 3.7456 |
| Difference | 0.3121 | 0.6223 | 0.6619 | 0.2713 | 0.2370 | 0.5134 |
| Relative reduction | 6.8% | 12.3% | 12.7% | 4.0% | 3.4% | 12.1% |
| Extra cost (USD) | 9,099 | 16,832 | 19,342 | 14,777 | 9,051 | 10,178 |
| Mean / max wait, Baseline (days) | 1.50 / 10 | 3.40 / 18 | 4.87 / 21 | 3.53 / 13 | 4.03 / 13 | 2.17 / 11 |
| Urgent mean wait, Baseline → Patient-aware (days) | 2.00 → 0.11 | 5.44 → 0.44 | 4.56 → 0.89 | 3.78 → 0.56 | 5.33 → 1.67 | 3.33 → 0.00 |

Notes:

- Houston 2 lines is a stress test; it is not an i-SHIPMENT value.
- T_MFE 5 and 14 are sensitivity values. i-SHIPMENT has 7 and 19, and 19 is infeasible with this demand at ND 40.
- No tie range was solved for the T_MFE 5 and 14 runs.

Findings:

- **Going from 4 to 2 Houston lines adds almost nothing** (12.3% → 12.7%). The benefit appears between 10 and 4 lines and then saturates.
- **T_MFE 14 with ND fixed at 40 cuts the gain to 3–4%.** The fixed time rises, the waiting window shrinks from 23 to 16 days, and the Baseline itself reroutes to Los Angeles. Only reordering is left.
- **T_MFE 5 with loose capacity gives a 12.1% gain.** The window is larger (25 days), and faster line turnover makes queuing at Philadelphia cheap for the Baseline (16 patients go there).
- **Shared driver:** the gain is large when the waiting window is large and the cost-only plan concentrates patients at the cheap nearby facility.
- **Caution:** differences between columns, for example 6.8% vs 12.1%, are of the same size as the tie ranges (0.17–0.52). A FIFO baseline is required before these comparisons are final.

## Open issues, not yet done

- **FIFO baseline**, to remove the tie-range ambiguity. This is the highest priority before any number is final.
- **Window-fixed sweep:** ND = T_MFE + 33, i.e. (5, 38), (10, 43), (14, 47). All are within Kwon's 41–52 day medians.
- **Presentation of E_D:** report the difference and the relative reduction. An optional Monte Carlo with common random numbers on the fixed plans would turn the result into integer counts of patients saved or lost.
- **Patient-flow page:** https://claude.ai/artifact/TMhbwNMR438CcoDhuuDCmK. It still uses placeholder schedules and needs flows_C.json (per-patient plans for Scenario C).
- **The workbook README sheet is stale.**
- **λ_U = 0 so far.**
- **Later / structural:** manufacturing or QC speed as a decision; the collection day as a decision; a dynamic (state-observed) version.

## Files

- **Device folder** "Literature Review Papers-PhD":
  - Updated Model_Patient tiers_revised_v3_pure.docx (route-dependent TT)
  - CART_instance_30patients.xlsx
- **GitHub branch Model_Code_version_3** (mirrored to claude/cart-supply-chain-formulation-42hq63):
  - CART_eligibility_US.ipynb
  - results folders: results_us_ND24/, results_us/, results_us_cap4/, results_us_cap4all/, results_us_cap2/, results_us_cap4_tmfe14/, results_us_cap2_tmfe14/, results_us_cap10_tmfe5/ (the last holds comparison_6_scenarios.csv)
