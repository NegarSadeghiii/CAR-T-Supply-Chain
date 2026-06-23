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
