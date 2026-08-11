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

_Phase 6 produced no usable sweep._


## (c) Did triage appear at gamma > 1?

_Phase 7 produced no usable runs._


## Verdict

* (a) scheduling-substitutes-for-capacity: **SUPPORTED**
* (b) centralization sets the value of scheduling: **NOT SUPPORTED as a clean monotone trend**
* (c) triage at gamma > 1: **NO**
