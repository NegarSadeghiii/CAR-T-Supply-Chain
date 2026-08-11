# Decision experiments - is the scheduling extension surprising?

Go/no-go for switching the INFORMS abstract from the yield SP to the survival-aware scheduling extension. Every number below comes from `results/ab_experiments/*.csv`; nothing is re-derived by hand.


## (a) Does scheduling substitute for capacity?

_Phase 5 produced no usable frontier._


## (b) Does centralization determine when scheduling matters?

Expected lives saved by scheduling alone, as CON1 is relaxed:

| N | CON1=2 | CON1=3 | CON1=4 | CON1=5 | CON1=6 | trend |
|---|---|---|---|---|---|---|
| 200 | 2.64 | 2.33 | 2.46 | 2.24 | 2.17 | decreasing overall (2.64 -> 2.17) |

CON1 is only an *upper bound*, so this table answers "what happens if more plants are permitted", not "what happens if more plants exist". Where the bound is not binding the optimiser simply keeps the network it already preferred and the row is flat by construction.


## (c) Did triage appear at gamma > 1?

_Phase 7 produced no usable runs._


## Verdict

* (a) scheduling-substitutes-for-capacity: **NOT RUN**
* (b) centralization sets the value of scheduling (CON1 sweep only): **SUPPORTED**
* (c) triage at gamma > 1: **NO**
