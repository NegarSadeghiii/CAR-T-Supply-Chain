# CAR-T Network Planning — Patient Outcomes When Manufacturing Runs Late

Autologous CAR-T is made from each patient's own cells. When a batch fails and
has to be re-made, the patient waits — and high-urgency patients get sicker while
they wait, sometimes past the point where treatment is still possible. This
report asks a simple operational question:

> **If we plan the manufacturing network assuming batches always come back on
> time, how many high-urgency patients do we lose that a plan accounting for
> waiting-time decline would have saved — and at what cost?**

We compare two plans, both choosing where to open facilities and how much
capacity to build:

- **On-time plan** — assumes manufacturing always succeeds on time.
- **Decline-aware plan** — accounts for patients getting sicker during delays.

Both plans are then run through the same detailed patient simulation (real
manufacturing failures, crowding at busy facilities, and patients declining while
they wait) so we see what each plan *actually* delivers. All results are seeded
and reproducible with `python run_all.py`. Capacity limits use the real
per-facility value (75 slots, Wan et al. 2026) rather than the earlier
artificial 40. "Real decline rate" marks the speed of patient decline matched to
published survival data (delayed-vs-on-time hazard ratio 1.64, Dulobdas 2025).

---

## Headline: the on-time plan loses roughly twice as many high-urgency patients as it thinks

At the real decline rate, in a busy (near-capacity) network of 150 patients:

| | High-urgency patients lost per cohort | Patients treated in time |
|---|---|---|
| What the on-time plan **assumes** | **0.84** | ~96% |
| What the on-time plan **actually delivers** | **1.67** | 94.2% |

**The on-time plan loses about 2× more high-urgency patients than its own
numbers predict** — because it never accounts for the extra deaths caused by
manufacturing delays at busy facilities. In the smaller 50-patient network the
same gap appears: 0.32 assumed vs 0.69 actually lost. See
`figures/figureH1_high_urgency_lost.png` (and `..._low_demand.png`): the flat
dotted line is what the plan assumes; the red line is what really happens; the
gap between them is the hidden death toll. It grows steadily as patients decline
faster, and at the real decline rate it has already doubled.

---

## Building spare capacity saves high-urgency lives — when a facility has room to grow

The decline-aware plan protects high-urgency patients by **building spare
capacity at the facility treating them**, so re-made batches don't queue behind a
full production line. This is only possible when that facility has room to grow
under its capacity limit. Where it does (50-patient network, one busy facility
with headroom):

| At the real decline rate | On-time plan | Decline-aware plan |
|---|---|---|
| High-urgency patients lost | 0.69 | **0.61** (−12%) |
| Spare capacity built | 0 slots | **11 slots** |
| Total cost | 22.7 M USD | 22.2 M USD |
| Facilities opened | 1 | 1 |

So the decline-aware plan lost **fewer** high-urgency patients **and cost
slightly less** — building spare capacity paid for itself by avoiding the most
expensive outcome (an untreated high-urgency patient).
(`figures/figureH3_treated_in_time.png`, `figureH5_spare_capacity_built.png`.)

**Where facilities are already maxed out (150-patient network, every facility
full), there is no room to build spare, and the two plans converge** — the only
remaining option, opening an extra lower-quality facility, costs more lives to
manufacturing failures than it saves to delay. In that regime the value of the
decline-aware plan is in *revealing* the true toll, not changing the design.

---

## Removing the artificial capacity cap saves high-urgency lives

The earlier model capped each facility at 40 slots — an artificial value chosen
to force a two-facility answer, not a real constraint. Raising it to the real
limit of 75 slots (100-patient network, at the real decline rate):

| Capacity limit per facility | High-urgency patients lost (on-time plan) | Facilities opened |
|---|---|---|
| 40 slots (artificial cap) | **1.20** | 3 (forced onto a lower-quality site) |
| 55 slots | 1.07 | 2 |
| 75 slots (real limit) | **1.01** | 2 (both high-quality) |

Removing the artificial cap cuts high-urgency losses by **~16%** (1.20 → 1.01),
because the network no longer has to push patients onto a lower-yield facility
where batches fail more often. With the cap removed, the decline-aware plan can
then build spare capacity and trims losses further to **0.96**
(`figures/figureH4_capacity_cap_effect.png`).

---

## What the network actually does

- **On-time plan** concentrates patients on the one or two highest-quality
  facilities and builds **no spare capacity** (it sees no reason to).
- **Decline-aware plan** opens the same facilities but **builds spare capacity**
  (11–26 slots, where room exists) so re-manufactured batches for waiting
  patients clear faster.
- Under the artificial 40-slot cap, the on-time plan is *forced* to open a third,
  lower-quality facility, which is what drives its higher high-urgency losses.

---

## Bottom line

1. **A plan that assumes on-time manufacturing loses about twice as many
   high-urgency patients as it predicts.** The extra deaths come from patients
   declining while they wait for a re-made batch at a busy facility — an effect
   the on-time plan is blind to.
2. **Planning for that decline saves high-urgency lives at little or no extra
   cost, by building spare capacity** — provided the treating facility has room
   to grow.
3. **The artificial 40-slot capacity cap was itself costing high-urgency lives**
   (~16% more losses) by forcing patients onto lower-quality facilities; the real
   75-slot limit is both cheaper and safer.

*Reproduce:* `python run_all.py` (validation checks and internal model diagnostics
are in `TECHNICAL_APPENDIX.md` and `figures/technical/`). Figures: `figures/figureH1–H5`.
