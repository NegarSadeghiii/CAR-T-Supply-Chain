# CAR-T Network Planning — Why High-Urgency Patients Are Lost, and How to Protect Them

Autologous CAR-T is made from each patient's own cells. Between collection and
treatment a patient waits about six weeks — and if the batch fails and has to be
re-made, they wait longer still. High-urgency patients can get sicker during that
wait, sometimes past the point where treatment is still possible. This report
answers two operational questions in plain terms — patients, deaths, waiting
time, and cost:

> **1. WHY are high-urgency patients being lost — is it the ordinary wait, or is
> it manufacturing failures and crowding?**
>
> **2. When every factory is already full, can putting the SICKEST PATIENTS FIRST
> protect high-urgency patients — in the situation where building more spare
> capacity cannot?**

Everything is run through the same detailed patient simulation (real
manufacturing failures, crowding at busy facilities, and patients declining while
they wait). All results are seeded and reproducible with `python run_all.py`.
Capacity limits use the real per-facility value (75 slots, Wan et al. 2026).
"Real decline rate" is the speed of patient decline matched to published survival
data (delayed-vs-on-time hazard ratio 1.64, Dulobdas 2025).

---

## Step 1 — Why are high-urgency patients being lost?

Every high-urgency patient we lose is tagged by cause:

- **Lost during the normal wait** — the batch was made successfully, but the
  standard ~6-week collection-to-treatment time outran how long the patient could
  survive. **More factory capacity does not fix this** — it doesn't shorten the
  normal schedule. The only lever is *speed*: treating the sickest sooner.
- **Lost after a manufacturing failure** — the batch failed, and the re-make plus
  any backlog at a busy factory pushed the total wait past survival. Spare
  capacity shortens that backlog and *can* reduce these losses.

At the real decline rate, high-urgency losses split as follows
(`figures/figureP1_why_high_urgency_lost.png`):

| Setting | High-urgency lost per cohort | During the normal wait | After a failure |
|---|---|---|---|
| **Busy network (150 patients)** | **5.86** | **4.19 (72%)** | 1.67 (28%) |
| **Low-demand network (50 patients)** | **2.04** | **1.36 (66%)** | 0.69 (34%) |

**Roughly two out of every three high-urgency patients we lose die during the
ordinary wait — before crowding or failure ever enters the picture.** This holds
in both the busy and the quiet network. The implication is decisive: **the main
fix is not more factories, it is speed and who-goes-first.** Adding capacity only
addresses the smaller "after a failure" slice.

For context, the same split for medium- and low-urgency patients in the busy
network (`figures/figureP2_cause_by_urgency.png`) runs the other way — most of
their losses come *after a failure*, because they decline slowly enough that the
normal wait rarely kills them:

| Urgency | Lost per cohort | During the normal wait | After a failure |
|---|---|---|---|
| High | 5.86 | 72% | 28% |
| Medium | 8.08 | 43% | 57% |
| Low | 3.31 | 25% | 75% |

This is exactly why a single network-wide capacity fix misses the high-urgency
problem: the sickest patients are lost to *time*, not to *crowding*.

---

## Step 2 — Does putting the sickest first protect high-urgency patients when factories are full?

Step 1 says the fix is speed, not capacity. Step 2 tests a "sickest first" lever:
high-urgency patients get first claim on the fastest, safest slots and on re-make
capacity, and lower-urgency patients yield their place when things are tight — the
*same* total waiting, reallocated to protect the sickest. We compare three plans
in the busy 150-patient network, all scored on the same patients:

1. **On-time plan** — builds the network assuming batches always come back on time.
2. **Decline-aware plan (spare capacity only)** — plans for patients declining
   during delays and is allowed to build spare capacity, but treats patients in
   the normal order.
3. **Decline-aware + sickest first** — the same plan, plus the sickest-first lever.

**The key finding first.** In the busy network every facility is already pinned at
its capacity limit, so **there is no room to build spare capacity — the
decline-aware plan (2) is forced to the exact same network as the on-time plan
(1), and loses exactly as many high-urgency patients (5.86).** This is the case
where "just build more" has nothing left to give.

Putting the sickest first is the only lever that still works
(`figures/figureP3_three_plans.png`):

| Busy network, real decline rate | On-time | Decline-aware (spare only) | Decline-aware **+ sickest first** |
|---|---|---|---|
| **High-urgency patients lost** | **5.86** | 5.86 | **2.39** |
| High-urgency treated in time | 80.5% | 80.5% | **92.0%** |
| Medium treated in time | 89.2% | 89.2% | 89.8% |
| Low treated in time | 92.6% | 92.6% | 91.1% |
| Total cost per patient | 0.72 | 0.72 | **0.54** |

> **Answer to the key question: YES.** When factories are full — the exact
> situation where extra spare capacity cannot help (plans 1 and 2 are identical) —
> putting the sickest first cuts high-urgency losses from **5.86 to 2.39 per
> cohort, a 59% reduction**, and lifts high-urgency treated-in-time from 80% to
> 92%.

### The price to lower-urgency patients is small

Reallocating waiting time to protect the sickest does push some delay onto
lower-urgency patients — but far less than it saves
(`figures/figureP4_tradeoff.png`):

| | Change with sickest-first |
|---|---|
| High-urgency patients **saved** | **+3.47 per cohort** |
| Extra lower-urgency patients lost | +0.25 per cohort (medium −0.45, low +0.70) |
| Low-urgency treated in time | 92.6% → 91.1% (−1.5 points) |
| Medium-urgency treated in time | 89.2% → 89.8% (essentially unchanged) |

For every extra lower-urgency patient delayed past their deadline, roughly **14
high-urgency patients are saved.** And because an untreated high-urgency patient
is the most costly outcome of all, protecting them **lowers total cost per
patient** (0.72 → 0.54) rather than raising it — the lever pays for itself.

---

## Supporting result — removing the artificial capacity cap

An earlier version of the model capped each facility at 40 slots, an artificial
value that forced patients onto a lower-quality third facility. Raising it to the
real 75-slot limit (100-patient network, on-time plan, real decline rate,
`figures/figureP6_capacity_cap.png`):

| Capacity limit per facility | High-urgency lost | Facilities opened |
|---|---|---|
| 40 slots (artificial cap) | 4.05 | 3 (forced onto a lower-quality site) |
| 55 slots | 3.93 | 2 |
| 75 slots (real limit) | 3.89 | 2 (both high-quality) |

Removing the artificial cap trims high-urgency losses (4.05 → 3.89) by keeping
patients off the failure-prone third site — but note the effect (~0.16 patients)
is small next to the sickest-first lever (~3.5 patients). Capacity choices move
the "after a failure" slice; only prioritization moves the larger "normal wait"
slice.

---

## Bottom line

1. **Most high-urgency losses are not about crowding — they happen during the
   ordinary six-week wait** (72% in the busy network, 66% in the quiet one). No
   amount of extra factory capacity shortens that wait.
2. **When factories are full, building spare capacity does nothing** — the
   decline-aware plan is forced to the same network as the on-time plan and loses
   the same 5.86 high-urgency patients.
3. **Putting the sickest first is the lever that works in that regime**: it cuts
   high-urgency losses by 59% (5.86 → 2.39), costs the average lower-urgency
   patient almost nothing (net +0.25 lost, ~14 high-urgency saved each), and
   *lowers* total cost per patient.

*Reproduce:* `python run_all.py`. Main figures: `figures/figureP1–P6`. Validation
checks and internal model diagnostics (in technical language) are in
`TECHNICAL_APPENDIX.md` and `figures/technical/`.
