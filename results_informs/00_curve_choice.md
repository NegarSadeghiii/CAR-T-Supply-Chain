# Curve choice

These Weibull curves are **illustrative showcase values** chosen to make the scheduling mechanism visible. They are not clinical estimates and no clinical claim rests on them.

**Frozen set: S2** - high risk eta=25, k=3; low risk eta=80, k=1.5.

Chosen because it has the largest same-cost gain (0.159) among the sets whose inversion day falls after the largest observed turnaround time.

Every later run and every figure caption uses this set. Experiment 2's faster-decline case changes only the high-risk scale to eta=22.

## Screening table (base instance, seed 42)

| set | eta_H | k_H | eta_L | k_L | C_min | ELIG_cost_speed | E_max  | same_cost_gain | inversion_day | max_TRT | inversion_after_max_TRT |
| --- | ----- | --- | ----- | --- | ----- | --------------- | ------ | -------------- | ------------- | ------- | ----------------------- |
| S1  | 30    | 3   | 90    | 1.5 | 25000 | 8.7722          | 8.8703 | 0.0981         | 45            | 32      | True                    |
| S2  | 25    | 3   | 80    | 1.5 | 25000 | 8.2362          | 8.3951 | 0.1589         | 37            | 32      | True                    |
| S3  | 35    | 3   | 70    | 1.5 | 25000 | 8.8111          | 8.8528 | 0.0417         | 50            | 32      | True                    |

`inversion_day` is the day the one-day eligibility loss of a high-risk patient falls below that of a low-risk patient; past it the optimiser gains more by speeding up a low-risk patient. It must sit after the largest turnaround time the solutions actually reach, or the showcase would be demonstrating the opposite of its intent.
