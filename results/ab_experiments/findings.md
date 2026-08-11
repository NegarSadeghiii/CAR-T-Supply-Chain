# Decision experiments - is the scheduling extension surprising?

Go/no-go for switching the INFORMS abstract from the yield SP to the survival-aware scheduling extension. Every number below comes from `results/ab_experiments/*.csv`; nothing is re-derived by hand.


## (a) Does scheduling substitute for capacity?

| N | network | lives saved by scheduling alone | what those cost | extra lives only buying capacity reaches | $/life to build | share of all reachable lives that scheduling gets |
|---|---|---|---|---|---|---|
| 200 | m1+m3 | 3.22 | $404,981 | 0.67 | $7.13M | 83% |


The first tranche is the striking one: at every scale a large share of the reachable lives costs **nothing at all** - those schedules are cost-indistinguishable from the cost-optimal design (within 0.1% of its total cost), they were simply never selected because a pure cost objective is indifferent among them:

| N | lives at zero added cost | cost delta | alpha that reaches it |
|---|---|---|---|
| 200 | 2.44 | $509 | 10000 |


**Verdict on (a).** At N = 200, holding the network fixed at the cost-optimal design and only re-scheduling saves **3.22 expected lives for $404,981** ($125,735 per life). The next 0.67 lives are reachable only by changing the network, at $7.13M per life - roughly **57 x** more expensive per life than the scheduling lives.


## (b) Does centralization determine when scheduling matters?

Expected lives saved by scheduling alone, as CON1 is relaxed:

| N | CON1=2 | CON1=3 | CON1=4 | CON1=5 | CON1=6 | trend |
|---|---|---|---|---|---|---|
| 200 | 2.64 | 2.33 | 2.46 | 2.24 | 2.17 | decreasing overall (2.64 -> 2.17) |

CON1 is only an *upper bound*, so this table answers "what happens if more plants are permitted", not "what happens if more plants exist". Where the bound is not binding the optimiser simply keeps the network it already preferred and the row is flat by construction.


The controlled version imposes the network with `fixed_facilities`, so capacity really does grow:

| N | plants | network | capacity | mean hold (cost design) | E[lost] cost | E[lost] survival | lives saved by scheduling |
|---|---|---|---|---|---|---|---|
| 200 | 1 | m2 | 31 | 0.01 | 6.636 | 6.355 | 0.281 |
| 200 | 2 | m1+m2 | 35 | 2.98 | 7.653 | 6.322 | 1.332 |
| 200 | 3 | m1+m2+m3 | 45 | 3.22 | 7.677 | 6.322 | 1.355 |
| 200 | 4 | m1+m2+m3+m4 | 49 | 2.73 | 7.518 | 6.322 | 1.197 |
| 200 | 5 | m1+m2+m3+m4+m6 | 59 | 3.00 | 7.671 | 6.322 | 1.349 |
| 200 | 6 | m1+m2+m3+m4+m5+m6 | 90 | 3.24 | 7.792 | 6.322 | 1.471 |

**N = 200:** scheduling saves 0.28 lives on the tightest imposed network (m2, capacity 31) and 1.47 on the widest (m1+m2+m3+m4+m5+m6, capacity 90) - no decline, with mean hold falling from 0.01 d to 3.24 d.


## (c) Did triage appear at gamma > 1?

| gamma | mean hold H | M | L | sickest-first order holds? | swap-feasible pairs | priority inversions (raw) | inversions (rho-weighted) |
|---|---|---|---|---|---|---|---|
| 1 | 0.033 | 3.150 | 13.950 | yes | 406 | 0 | 0 |
| 1.5 | 0.033 | 3.100 | 13.833 | yes | 399 | 0 | 0 |
| 2 | 0.033 | 3.163 | 13.617 | yes | 429 | 0 | 0 |

**Verdict on (c). No triage** - at every gamma tested the schedule stays strictly sickest-first: tier mean holds remain ordered H <= M <= L and no swap-feasible pair inverts. In a deterministic model with a single monotone risk index this is the expected outcome - triage requires either uncertainty or a non-monotone value of time, neither of which this formulation contains.


## Verdict

* (a) scheduling-substitutes-for-capacity: **SUPPORTED**
* (b) centralization sets the value of scheduling (forced-network experiment): **NOT SUPPORTED as a clean monotone trend**
* (c) triage at gamma > 1: **NO**
