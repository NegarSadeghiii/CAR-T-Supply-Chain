# Resilience figure guide (F1–F7, F6a)

Decision-focused figures for the capacity-disruption / correlation study. All regenerate from `figures/generate_resilience.py` (compute cached in `figures/resilience_data.json`). Backend: **HIGHS** (Gurobi when a full license is present, else HiGHS — identical optima by parity). Design solved at N=200 (MIP gap 1e-03); curves scored out-of-sample at N=1000 over 3 seeds, plotted as mean with a min–max band.

| Fig | File | Question it answers | Takeaway (from the numbers) |
|-----|------|---------------------|------------------------------|
| F1 | `figureR1_rerouting_under_disruption` | Does rerouting wake up under facility disruption — is the earlier "subcontracting is dominated" result reversed at facility level? | **Yes, for independent failures:** subcontracting rises to ~4.7/scenario as surviving facilities absorb load; under common-mode failures it stays ~0 and everything cancels. |
| F2 | `figureR2_subcontracting_vs_correlation` | Slope or cliff — how does rerouting value fall as ρ rises (q=0.10)? | **A steepening slope, not a sharp cliff:** E[sub] declines steadily 2.4→0.0, steepest around ρ≈0.6–0.8 where it collapses to ~0; cancellations rise roughly linearly 3.4→50.0. |
| F3 | `figureR3_price_of_correlation` | What does correlation cost in money and service, across severities? | Cost climbs steeply and service falls with ρ at every q; correlation, not severity alone, drives the loss. |
| F4 | `figureR4_hedging_frontier` | Under high correlation, which hedge is worth paying for — spare capacity or diversification? | **Diversification dominates:** best diversified point reaches 74% service at 48 M USD vs baseline 24%/107; spare capacity saturates fast (best 38% at 76). |
| F5 | `figureR5_cancellations_by_tier` | Who gets cancelled as ρ rises? | **Per-patient cancellation rates track each other across tiers** (all → 100% at ρ=1): high-urgency tier-H is *not* shielded — its 6× cancel-cost incentive is offset by its lower re-collection eligibility (β_H=0.55), so H cancels at a marginally higher rate than mid-tier M. |
| F6a | `figureR6a_input_convergence` | How many scenarios does the *input* sampling need (no optimization)? | In the correlated regime the KPIs are bimodal (a common shock either fires or not), so the running median only settles by k*≈569 (disrupted facilities) / k*≈569 (failed batches) scenarios — justifying N≈600+ and confirming N=1000 leaves margin. |
| F6 | `figureR6_scenario_stability` | Is training N large enough for stable out-of-sample results? | OOS cost & service stabilize at **N=100** (tol 0.5 M USD / 1 pp). The N=1000 point sits ~1.0 M above N=500 because those design solves hit the 180 s limit at ~3% MIP gap (a solver artefact, not sampling); N≤500 all solved to gap 0. |
| F7 | `figureR7_blip_diagnosis` | Is the q=0.15 non-monotonic subcontracting blip real or noise? | **resolved: no blip at N=1000** (see below). |

## What moved from the under-sampled version

- Curves are now scored at N=1000 (was 200) over 3 seeds with min–max bands, so the traces are smooth and the bands show the sampling is tight.
- **F6a (new)** justifies the scenario count directly from the draws: with a converged-and-stays criterion the running median settles only by ~569 (disrupted facilities) / ~569 (failed batches) samples — the bimodal common-shock KPIs need the full N≥500–1000 to stabilize.
- **F6** now spans N∈[50, 100, 200, 500, 1000] at 5 seeds and reports stabilization at N=100.
- **F4** is now a populated two-lever frontier (spare capacity 0–100% of s_max, diversification 0–75% of effective ρ) rather than three markers.
- **F7**: with N=1000 the blip is diagnosed as **resolved: no blip at N=1000**; the headline decline (F2) and the diversification result (F4) are unchanged from the under-sampled version — only the noise shrank.
