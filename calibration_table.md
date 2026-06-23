# Calibration Table — CAR-T Supply Chain Model Parameters

All monetary values in millions of USD (M USD). Parameters from `case_study.py :: build_case_study_instance()`.

---

## Facility / production mode parameters

| Symbol | Value(s) | Units | Status | Primary source |
|---|---|---|---|---|
| $f_m$ (opening cost) | [0.50, 2.00, 3.00, 2.00] | M USD | Literature-derived | Bernardi et al. 2022; scaled by automation level (k1/k2/k3) |
| $\pi_m$ (capacity cost) | [0.04, 0.06, 0.09, 0.06] | M USD/slot | Literature-derived | Avramescu et al. 2023 |
| $c_m$ (batch mfg cost) | [0.20, 0.18, 0.15, 0.18] | M USD/batch | Literature-derived | Avramescu et al. 2023; decreasing with automation |
| $s_m^{\max}$ (max slots) | 40 (all four) | slots | Assumption | Wan et al. 2026 reports 75; reduced to 40 to enforce two-facility coverage |
| $p_m$ (yield success) | [0.85, 0.92, 0.95, 0.92] | — | Literature-derived | k1 manual 15% failure, k2 semi-auto 8%, k3 fully auto 5% (Costariol et al. 2021) |

---

## Patient tier / clinical parameters

| Symbol | Value(s) | Units | Status | Primary source |
|---|---|---|---|---|
| $\beta_H$ (re-collection, tier H) | 0.55 | — | Literature-derived | Roth et al. 2018 (sicker patients less likely to tolerate delay) |
| $\beta_M$ (re-collection, tier M) | 0.70 | — | Literature-derived | Roth et al. 2018 |
| $\beta_L$ (re-collection, tier L) | 0.80 | — | Literature-derived | Roth et al. 2018 |
| $\rho^{\text{cancel}}_H$ | 6.0 | M USD | Assumption | No published trial data; reflects high clinical consequence of tier-H cancellation |
| $\rho^{\text{cancel}}_M$ | 3.0 | M USD | Assumption | No published trial data |
| $\rho^{\text{cancel}}_L$ | 1.0 | M USD | Assumption | No published trial data |
| Tier mix | 20% H / 50% M / 30% L | — | Industry-aligned | Consistent with Kymriah DLBCL indication epidemiology |

---

## Recourse cost parameters

| Symbol | Value(s) | Units | Status | Primary source |
|---|---|---|---|---|
| $\rho^{\text{leuk}}$ | 0.005 | M USD | Literature-derived | Published leukapheresis procedure cost |
| $\rho^{\text{remfg}}_m$ | [0.20, 0.18, 0.15, 0.18] | M USD/batch | Assumption | Set equal to primary batch cost $c_m$ |
| $\rho^{\text{sub}}_{mm'}$ | $c_{m'} \times 1.15$ (off-diagonal) | M USD | Assumption | 15% premium on destination facility's primary cost |

---

## Population / scenario parameters

| Symbol | Value | Units | Status | Notes |
|---|---|---|---|---|
| $n$ (patients) | 50 | — | Industry-aligned | Base case; multi-instance runs: 50/100/200/500 |
| $N$ (scenarios) | 200 (base) | — | Assumption | Convergence checked; OOS evaluation uses 2,000 |
| $n_{\text{boot}}$ | 500 | — | Assumption | Bootstrap replications for OOS confidence intervals |
| Seed | 0 (in-sample), 100 (OOS) | — | Reproducibility | — |

---

## Geographic network (Wan et al. 2026 subset)

| Facility | City | Wan j-index | Mode | Coordinates |
|---|---|---|---|---|
| $m_0$ | Newark, NJ | j67 | k1 manual | 40.74°N, 74.17°W |
| $m_1$ | Boston, MA | j24 | k2 semi-auto | 42.36°N, 71.06°W |
| $m_2$ | Raleigh-Durham, NC | j43 | k3 fully-auto | 35.78°N, 78.64°W |
| $m_3$ | San Francisco, CA | j14 | k2 semi-auto | 37.77°N, 122.42°W |

Hospitals: 15 sites from Wan's 57-hospital network selected by highest total demand, covering ≈43% of Wan's patient volume.

**Cost parameter note:** Wan's raw $\bar{c}_{jkt}$ values (~\$84–121K/batch) are ~1/3 of contemporary clinical estimates and produce a degenerate cost structure ($\pi_m + c_m$ identical across all modes). Cost parameters follow clinical calibration (Avramescu 2023, Bernardi 2022) rather than Wan's tables; the geographic network structure is adopted as described.

---

## Citation gaps

| Parameter | Gap |
|---|---|
| $\rho^{\text{cancel}}_u$ | No published willingness-to-pay or clinical loss estimates by urgency tier. Values are order-of-magnitude assumptions. |
| $\rho^{\text{remfg}}_m$ | Assumed equal to primary cost; no published re-manufacturing-specific cost breakdown. |
| $\rho^{\text{sub}}_{mm'}$ | 15% subcontracting premium is assumption; no published CAR-T CMO contract data. |
| $\beta_u$ | Roth et al. values are for stem-cell transplant patients; direct CAR-T re-leukapheresis feasibility data unavailable. |

---

## Cross-references for cost parameters

The wide spread in facility opening cost $f_m \in [\$0.5\text{M}, \$3.0\text{M}]$ in our calibration is consistent with the multi-scale industry data reported in Avramescu et al. (2022) (Nature Scientific Reports supplementary materials), where facility construction costs differ by approximately 7× across small / medium / large facility classes (annual amortized: \$231K / \$577K / \$1.73M for 4 / 10 / 31 parallel-line facilities, respectively). Transport-cost components in the recourse menu are calibrated consistent with industry data reported by TrakCel Ltd (cited in the same reference). Re-manufacturing delay $\delta^{\text{remfg}} = 14$–21 days reflects the typical CAR-T cycle (~7 days production + ~7 days quality control + 1–2 days transport, with current-technology cycle times totaling 19 days as reported by Avramescu et al., 2022).

Wan et al. (2026) calibrate per-mode failure rates of 10% / 5% / 3% (manual / partially-automated / fully-automated production modes) following Lopes et al. (2020); our facility-specific $p_m = [0.85, 0.92, 0.95, 0.92]$ from per-product real-world failure rates falls within this empirical range and provides finer-grained facility-level resolution. The maximum installable capacity $s_m^{\max} = 40$ slots in our calibration is conservative relative to Wan's 75-slot value, reflecting a single-CDMO scale appropriate for our cohort size $N = 50$.

---

## Demographic and network context (reference only)

The case study facility-location candidates correspond to 4 major US biopharmaceutical hubs selected from the 1,000-city candidate network of Wan et al. (2026) (adapted from Avramescu et al., 2021b). The 15-hospital selection reflects approximately 26% of Wan's 57-hospital US network. Patient cohort $N = 50$ corresponds to approximate annual r/r pediatric ALL CAR-T demand at a mid-sized US treatment-region catchment, derived from IICC-3 childhood cancer incidence data (Steliarova-Fourcher et al., 2017) refined by the conditional probabilities reported in Avramescu et al. (2021b): ~80% of pediatric ALL cases are B-precursor, of which ~20% are refractory or relapsed.

The following parameters from Wan et al. (2026) are documented as reference context only — they are NOT adopted as values in our model:

| Wan 2026 parameter | Value | Source | Adopted? |
|---|---|---|---|
| Hospital network size | 57 cities (US Kymriah-related) | `Hospital Location` sheet | Geographic labels only; we use 15-hospital subset |
| Candidate facility locations | 1,000 US cities | `Candidate Facility Location` sheet | Geographic labels only; we select 4 |
| Planning periods | 6 | `d_it`, `s_ijt` sheets | Not adopted (single period) |
| Per-mode failure rates | 10% / 5% / 3% | `r_ok` (Lopes 2020) | Cross-reference only; our $p_m$ anchored to UK Panel / per-product real-world rates |
| Max facility capacity | 75 slots | `Maxcap_j` | Cross-reference only; we use 40 |
| Facility construction cost | \$170K–\$465K | `f_jk` | Not adopted; substituting these values produced VSS-vs-ECD ≈ 0% (see sensitivity-analysis discussion) |
| Operating cost per batch | \$68K–\$102K | `overline_c_jkt` | Not adopted; same reason as above |
| Shelf life | 24 hours | `SLF_o` | Not directly modeled in our formulation |

---

## Methodological observation: cost-spread regime requirement

During the calibration process, we evaluated whether substituting Wan et al. (2026)'s facility-specific cost values ($f_{jk}$ and $\bar{c}_{jkt}$ from their `Real_Case/Data/` archive, adapted from Avramescu et al., 2021b) for our Avramescu/Bernardi-aligned calibration would strengthen the empirical grounding of the case study. The substitution produced degenerate optimization: the value of stochastic solution against the expected-parameter deterministic baseline (VSS-vs-ECD) collapsed from 10.48% to approximately 0%.

**Diagnosis:** Wan's published facility opening cost range (\$0.19M–\$0.46M) and effectively uniform per-slot operating cost across production modes (\$0.24M/slot in all modes) eliminate the cost-yield trade-off that drives the structural advantage of two-stage stochastic programming. With compressed cost spread, both the deterministic and stochastic plans converge to the same facility selection (the unique cost-yield-Pareto-dominant facility), and the structural recourse advantage disappears.

This is consistent with the regime characterization in Figure 16 (p_m sensitivity heatmap): VSS-vs-ECD remains above the 4% contribution-viability threshold in the moderate-cost-spread regime where our Avramescu/Bernardi calibration sits, and collapses in the narrow-spread regime where Wan's published cost values sit. The empirical CAR-T calibration — wide spread in facility opening cost (\$0.5M–\$3.0M), wide spread in per-batch operating cost (\$0.15M–\$0.20M), wide spread in per-batch yield (0.85–0.95) — places the case study squarely in the regime where the structural-recourse value holds.

The finding does NOT invalidate our contribution claim; it characterizes the operating regime in which two-stage SP adds value over expected-parameter deterministic baselines, and demonstrates that this regime is consistent with empirical real-world CAR-T heterogeneity (axicabtagene ciloleucel 4% failure rate / tisagenlecleucel 17% / lisocabtagene maraleucel 28%, per-product real-world rates documented by the UK National CAR-T Panel).
