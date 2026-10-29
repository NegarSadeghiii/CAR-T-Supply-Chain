# INFORMS runs - numbers for the slides

All Weibull curves below are **illustrative showcase values**, chosen to make the scheduling mechanism visible. They are not clinical estimates.

Frozen curve set **S2**: high risk eta=25, k=3; low risk eta=80, k=1.5.

Solver settings: `{"solver": "appsi_highs", "mip_gap": 1e-06, "time_limit_s": 300.0, "cost_tolerance": 0.001, "pyomo": "6.10.1", "numpy": "2.4.6", "pandas": "3.0.6"}`.

## Slide 7 - same-cost schedule comparison

Seeds 1-10 (10 instances), every plan within 0.1% of the cheapest plan's cost.

- **same-cost gain**: mean 0.415 (min 0.165, max 0.604) expected eligible patients
- **paid gain at +5% budget**: mean 0.771 (min 0.440, max 1.248) expected eligible patients
- **ceiling gap (no budget)**: mean 1.375 (min 0.836, max 1.934) expected eligible patients
- **same-cost range (E_max - E_min)**: mean 0.799 (min 0.320, max 1.395) expected eligible patients

Levers the eligibility-aware plan actually moves, counted per instance against the cost+speed plan (10 patients each):

- start order: mean 5.4 patients (max 9)
- start day: mean 5.4 patients (max 9)
- transport mode: mean 0.4 patients (max 2)
- facility: mean 0.0 patients (max 0)

The lever that changes most is **start order** (mean 5.4 patients per instance).

Policy results. `at_C_min` False means the policy could not be scheduled at the cheapest plan's cost and is shown at its own minimum, so its eligibility is NOT comparable with the rest of the table:

| policy          | ELIG_mean | ELIG_min | ELIG_max | cost_mean  | n  | at_C_min |
| --------------- | --------- | -------- | -------- | ---------- | -- | -------- |
| cost_speed      | 8.093     | 7.533    | 8.638    | 23,360.000 | 10 | True     |
| fifo            | 8.065     | 7.509    | 8.510    | 23,360.000 | 10 | True     |
| high_risk_first | 9.460     | 9.412    | 9.474    | 43,060.000 | 10 | False    |
| elig_min        | 7.709     | 6.742    | 8.381    | 23,360.000 | 10 | True     |
| elig_max        | 8.508     | 8.137    | 8.831    | 23,360.000 | 10 | True     |

- FIFO reaches 8.065 on average against 8.508 for the eligibility-aware plan (0.443 fewer eligible patients), over 10 of 10 instances.
- high-risk-first was infeasible at C_min in 10 of 10 instances; see the own-cheapest-plan rows in exp1_runs.csv.
- high risk first at its own cheapest cost: 43,060 against 23,360 for every other policy (+84%), reaching 9.460 eligible patients. It buys more than the eligibility-aware plan, but not at the same cost.

## Slide 8 - congestion x decline

| decline        | congestion | same_cost_gain mean | same_cost_gain min | same_cost_gain max | paid_gain mean | paid_gain min | paid_gain max |
| -------------- | ---------- | ------------------- | ------------------ | ------------------ | -------------- | ------------- | ------------- |
| faster decline | base       | 0.479               | 0.215              | 0.643              | 0.947          | 0.597         | 1.463         |
| faster decline | high       | 0.941               | 0.210              | 1.790              | 1.648          | 0.943         | 2.538         |
| faster decline | low        | 0.143               | 0.000              | 0.471              | 0.482          | 0.275         | 0.896         |
| frozen set     | base       | 0.415               | 0.165              | 0.604              | 0.771          | 0.440         | 1.248         |
| frozen set     | high       | 0.861               | 0.189              | 1.681              | 1.437          | 0.782         | 2.309         |
| frozen set     | low        | 0.115               | 0.000              | 0.422              | 0.363          | 0.198         | 0.741         |

- **frozen set**: the same-cost gain grows as congestion rises (0.115 -> 0.415 -> 0.861 for low, base, high).
- **faster decline**: the same-cost gain grows as congestion rises (0.143 -> 0.479 -> 0.941 for low, base, high).

## B1 - frontier

- 9 distinct frontier points, 2 unsupported (not on the convex hull).
- Knee at cost 29,800 with 9.272 expected eligible patients.
- Cost range 25,000 to 39,800; eligibility 8.395 to 9.461.

## B3 - cost side

- Sensitivity levels taken from the notebook's GRID (CART_parameter_sources.xlsx is not in the repository).
- 11 parameters, 40 levels, 280 frontier points in total.
- 7 of those parameters are drawn in fig_b3_sensitivity.png. The Weibull rows (Scale eta, high risk, Scale eta, low risk, Shape k, high risk, Shape k, low risk) are kept in the CSV but left out of the figure: the GRID's levels for them were set for the notebook's BASE curves (eta_H 15, eta_L 29), not for the frozen set (eta_H 25, eta_L 80), so sweeping them walks away from the curves section 0 froze, and they are not cost-side parameters.

| config                  | point    | scale | CTM        | TTC       | HOLD      | cost       | ELIG  |
| ----------------------- | -------- | ----- | ---------- | --------- | --------- | ---------- | ----- |
| M1=1, M2=2 (0.5x lines) | min cost | 0.500 | 26,000.000 | 6,600.000 | 5,800.000 | 38,400.000 | 8.379 |
| M1=1, M2=2 (0.5x lines) | knee     | 0.500 | 30,000.000 | 8,200.000 | 3,400.000 | 41,600.000 | 9.097 |
| M1=2, M2=4 (1x lines)   | min cost | 1.000 | 10,000.000 | 6,400.000 | 8,600.000 | 25,000.000 | 8.395 |
| M1=2, M2=4 (1x lines)   | knee     | 1.000 | 18,000.000 | 8,600.000 | 2,800.000 | 29,400.000 | 9.245 |
| M1=3, M2=6 (1.5x lines) | min cost | 1.500 | 10,000.000 | 6,600.000 | 1,800.000 | 18,400.000 | 8.858 |
| M1=3, M2=6 (1.5x lines) | knee     | 1.500 | 14,000.000 | 8,800.000 | 400.000   | 23,200.000 | 9.351 |

## Solve quality and caveats

- every solve returned optimal.
- Base instance matches the notebook's BASE exactly (10 patients, window 15, M1 2 lines / M2 4 lines, fast 1 d / slow 3 d, TLS 1, TMFE 5, TQC 2, horizon 45, HC $200); no mismatch found.
- Font: DejaVu Sans (IBM Plex Sans is not installed here).
- high-risk-first infeasible at C_min in 10 of 10 instances
- CART_parameter_sources.xlsx is not in the repository, so B3(a) used the notebook's GRID instead of the SA levels tab

