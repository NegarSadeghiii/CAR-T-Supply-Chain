# Syncing `Survival_Aware_iSHIPMENT_Formulation.docx` to the code

State of the code as of commit `23c1126` on `claude/cart-supply-chain-formulation-42hq63`.

Every claim below was checked against the source or against the recorded runs in
`results/mpc/`; the verification is named in each entry. Items are keyed to text
you can Ctrl+F in the .docx.

There are four groups:

* **A** — the strategic split was reverted; the doc still describes two models.
* **B** — ρ is gone; α is now per tier.
* **C** — drift that predates both changes: places the doc never matched the code.
* **D** — results now established that the doc could state.

---

## A. The strategic split was reverted — there is one strategic model

`strategic_cost_design.py`, `strategic_survival_probe.py` and
`check_network_identity.py` were deleted. `strategic.py` calls
`ishipment_survival.run_design(net, inst, "extension", alpha=ALPHA_STRATEGIC)`,
a single model that carries the survival term of (1), the survival extension
and the manufacturing queue. Every experiment, Exp C included, uses it.

Verified: `strategic.py:106-112`; `git show --stat edff3f6`.

| # | Find this | Problem | Replace with |
|---|---|---|---|
| A1 | Header italic: *"Survival is now handled operationally, so the strategic layer is split into two variants (see the architecture note below)… belong to the survival probe used only for Exp C, not to the core design model."* | Describes the reverted split. | *"Equations (2)–(26) are retained from i-SHIPMENT. The survival extension, the manufacturing queue and the second term of (1) are introduced here and are part of the single strategic model."* |
| A2 | The whole bold block **"Model architecture (revised): survival is operational, not strategic."** and its paragraph | Specifies two files that no longer exist. | Delete the block. If you want an architecture note, state the two *layers* (strategic MILP solved once and frozen; operational simulation with per-epoch MPC) — not two strategic variants. |
| A3 | Italic note under **Objective**: *"**Core vs probe.** In strategic_cost_design the objective is cost only… inert for design across the plausible α range."* | Same. | Delete. The design-inertness finding is real and belongs in Exp C or a results note (see D2), not attached to the objective. |
| A4 | Heading **"Survival extension (strategic_survival_probe only)"** | Parenthetical is false. | **"Survival extension"** |
| A5 | Heading **"Manufacturing queue (strategic_survival_probe only) — replaces the fixed start (7)"** | Same. | **"Manufacturing queue — replaces the fixed start (7)"** |
| A6 | Paragraph after (35): *"In strategic_survival_probe, the hold is the only new degree of freedom… In the core design (strategic_cost_design) this choice does not exist — who-waits is decided operationally by the per-epoch policy."* | Second half describes a model that no longer exists. | *"The hold is the only new degree of freedom. Capacity (34) forces some patients to wait; the clinical-loss term of (1) then chooses who waits. ND (31), relaxed to 42 d, bounds the maximum hold. Because downstream stages (8)–(11) shift with the chosen start, TRT_p — and hence S_p — pick up the wait automatically."* |
| A7 | **"Per-epoch operational optimization"**, first line: *"The strategic design (strategic_cost_design, cost-based) is solved ONCE…"* | Names the deleted file; "cost-based" is wrong — the frozen solve includes the survival term. | *"The strategic design is solved ONCE, offline, at the calibrated life values, to fix the network."* |
| A8 | **"Relationship to the strategic design"**: *"…with the cost-based strategic layer frozen… It reuses the survival kernel and the capacity structure, not the probe's objective."* | Same two errors. | *"…with the strategic layer frozen and the horizon shrunk to a look-ahead window. It reuses the survival kernel and the capacity structure."* |
| A9 | Simulation **"Role and overview"**: *"The strategic design (strategic_cost_design) is solved once…"* | Names the deleted file. | *"The strategic design is solved once…"* |
| A10 | Parameter table, row **"Survival weight alpha"**, status column: *"strategic_survival_probe only (Exp C); not in core model"* | Same. | See B7 — this row is also affected by the α change. |
| A11 | **"Confirmed configuration"**: *"α is MONETISED in strategic_survival_probe only… The operational layer uses the tier weights ρ_u without α — its objective is Σρ(1−Ŝ) — so α and ρ are never combined outside the probe."* | Describes both the split *and* the old α×ρ split of duties. | See B6 — this paragraph needs a full rewrite. |
| A12 | Exp C description: *"(uses strategic_survival_probe; not part of the main pipeline)"* | Same. | *"(the only experiment that re-solves the strategic layer)"* |

---

## B. ρ is removed; α is now per tier

`cart_data.RHO` is gone. The single life-value parameter is
`cart_data.ALPHA_TIER = {H: $1.5M, M: $1.0M, L: $0.5M}` — the dollar value of
one life lost in each tier. `ALPHA_W = α_u/α_ref` (= 3/2/1) carries the
vector's *shape*; the scalar `alpha` threaded through `run_design` is the
reference tier's value and carries its *level*, which is what Exp C sweeps.

This is a reparameterisation, not a model change: `ALPHA_W` equals the old ρ
vector exactly. Verified two ways — the objective identity
`obj = cost + alpha·Σ ALPHA_W[u](1−S)` holds on all 61 cached extension solves,
and re-running `fifo`, `survival_index`, `static_survival` at N=200 over the 30
recorded seeds reproduces clinical loss, high-risk lost, total lost and total
cost bit-identically.

| # | Find this | Replace with |
|---|---|---|
| B1 | **Parameters → New (survival)**: *"ρ_u (clinical value of a saved tier-u patient); α (cost–survival weight); breakpoints τ_k, σ_k = S(τ_k)"* | *"α_u (value of one life lost in tier u, in dollars — the only life-value parameter)"*. Drop ρ_u entirely. Drop the breakpoints too — see C1. |
| B2 | Objective (1): `… + α Σ_p ρ_u(p)(1 − S_p)` | `… + Σ_p α_u(p)(1 − S_p)` |
| B3 | (P3): `min Σ_{i∈W_t} ρ_u(i)(1 − Ŝ_i) + Σ_{i,τ} c^op_i x_{i,τ}` | `min Σ_{i∈W_t} α_u(i)(1 − Ŝ_i)` — ρ→α_u, and the c^op term is not implemented (C5). |
| B4 | (P8): `start i* = argmax ρ_u(i)[S_i(t) − S_i(t+1)]` | `start i* = argmax α_u(i)[S_i(t) − S_i(t+1)]` |
| B5 | **Emergent priority** paragraph: *"minimizing the operational objective Σρ(1−Ŝ)"* | *"minimizing the operational objective Σα_u(1−Ŝ)"*. Consider adding the bound in D3 — it turns this claim from an assertion into a test. |
| B6 | **Confirmed configuration** bold paragraph (the α/ρ division of duties) | *"All parameters confirmed as tabled. Life value is carried by a single per-tier parameter α_u = {H: \$1.5M, M: \$1.0M, L: \$0.5M}; there is no separate priority weight. The strategic objective (1) uses α_u directly. The per-epoch objective (P3) carries no cost term, so only the shape of α_u reaches the operational layer — scaling every α_u by a common factor rescales (P3) without moving its argmin — and the operational results are invariant to the level. Exp C sweeps that level."* |
| B7 | Parameter table row **"Clinical weight rho / H 3, M 2, L 1 / proposed [confirm]"** | Row: **"Value of life α_u"** / **"H \$1.5M, M \$1.0M, L \$0.5M"** / **"calibrated: α_u = Q_u × λ at λ = \$125K/QALY, Q = {12, 8, 4}"** |
| B8 | Parameter table row **"Survival weight alpha / value of a life (\$/life)"** | Delete the row — α_u above replaces it. If you keep a row for the sweep, call it *"Exp C sweep level"* / *"α_ref ∈ {0, 50K, 100K, 250K, 500K, 1M, 2M, 5M}"* / *"reference-tier value; operating point \$500K"*. |
| B9 | **"operational cost += ρ_leuk"** (failure-recourse bullet) and **"ρ_leuk (re-leukapheresis cost) ≈ \$5,000 per attempt"** | `c_releuk` in both places. It was never a tier weight — sharing the letter ρ with the clinical weights was the whole reason to rename it (`per_epoch.C_RELEUK`). |
| B10 | **Finalized experimental design**: *"Every survival-aware policy uses α = \$500K per life."* | *"Every survival-aware policy uses the same life-value vector α_u = {\$1.5M, \$1.0M, \$0.5M}. Because (P3) has no cost term, the policies' behaviour depends only on the vector's shape."* |
| B11 | Policies table, **α** column (`—`, `$500K/life` ×4) | Either delete the column — it no longer distinguishes the policies — or retitle it **"uses α_u"** with values `no` / `yes` ×4. |

---

## C. Drift that predates both changes

These are places the doc has never matched the code. Worth fixing in the same
pass, since they would propagate the same way.

| # | Doc says | Code does | Verified |
|---|---|---|---|
| C1 | S_p linearised by the **λ-method with SOS2**, (28)–(30), over breakpoints τ_k, σ_k | **Binary day-indicators** δ_{p,d}, one per integer day d ∈ 17…42, with `Σ_d δ = 1`, `TRT_p = Σ_d d·δ_{p,d}`, `S_p = Σ_d σ_{d,u} δ_{p,d}`. Exact, because TRT is integral — and it dominates SOS2 here, since no branching on convex-combination weights is needed. The doc's own note *"Integer TRT_p ⇒ (28)–(30) exact"* is the reason the simpler construction is available. | `ishipment_survival.py:398, 437-439` |
| C2 | (16) `Σ_m E1_m ≤ 2` | Demand-dependent: 2 facilities at N=100 and N=200, 3 at N=500 | `ishipment_survival.py:66, 73` |
| C3 | *"At most K_remake = 2 attempts, then cancel"* (recourse bullet) vs *"Max remakes K_remake / 2, then cancel"* (table) — the doc contradicts itself | K_REMAKE = 2 **remakes**, i.e. **3 manufacturing attempts** per patient | `per_epoch.py:60`, `simulation.py:88-90` |
| C4 | T_elig *"dropped (loose ~90-day backstop)"* | The backstop is **disabled entirely** — `BACKSTOP_WAIT = None`. The projected-survival gate S_min ≥ 0.75 is the only futility rule. The code path is retained but inert; `lost_backstop` is zero in all 114 recorded result rows. | `per_epoch.py:62`, `results/mpc/exp*.csv` |
| C5 | (P3) includes `+ Σ_{i,τ} c^op_i x_{i,τ}` | Dropped. On a frozen network c^op_i is a per-patient constant and every started patient is started exactly once, so it is inert in the argmin. | `per_epoch.py` `_epoch_data` docstring |
| C6 | (P7) failure recourse with subcontract to a partner facility m′ and explicit cancel z_i | Not implemented. A remake goes to the patient's own frozen facility; cancellation happens through the S_min gate or K_remake exhaustion, not a decision variable. | no match for `subcontract`/`partner` in any module |
| C7 | Outputs: *"the policy-vs-benchmark gap (index / MPC / **cost-only** / **AI**)"* | Five policies: fifo, survival_index, static_survival, adaptive_mpc, best_achievable. `cost_only` was removed; no AI/LLM policy exists. | `policies.POLICY_NAMES` |
| C8 | best_achievable: *"where it does not solve at scale, a failures-revealed MPC proxy is used and labelled"* | The proxy exists but **was never used** — the perfect-information MILP solved at both scales. Both aggregates record `bound_label = "perfect information"`. Keep the sentence only as a contingency, or drop it. | `results/mpc/expD_aggregates_N{100,200}.json` |

---

## D. Findings the doc could now state

| # | Finding | Numbers |
|---|---|---|
| D1 | **The network Exp C actually produces.** N=200: m1+m3, 14 slots, facility cost \$10.53M, offered load 1.14 — unchanged for α_ref = 0 … \$2M. It flips at α_ref = **\$5M** to m3+m6, 20 slots, \$15.04M, offered load 0.80. N=100: m1+m4, 8 slots, \$6.02M — **no flip anywhere in the swept range**. | `results/mpc/expC_value_of_life_N{100,200}.csv` |
| D2 | **What actually drives the design.** Survival is design-inert across the plausible range (α_ref = 0 and α_ref = \$500K give the identical network at both scales), and the ND deadline is irrelevant (ND=18 and ND=42 give the identical network). What moves the design is the **queue**. Already written up in `results/mpc/README.md` § "What actually drives the network". | four-way comparison in that README |
| D3 | **The emergence claim is now a bound, not an assertion.** Under (P8), i beats j iff α_u(i)·ΔS_i > α_u(j)·ΔS_j, and the one-day survival decrements are in the ratio **ΔS_H : ΔS_M : ΔS_L = 7.3 : 2.5 : 1** across the whole 17–42 d range. So high-risk patients are served first for *any* α_u with α_L/α_H < 7.3, and ahead of medium-risk for any with α_M/α_H < 2.9. **Do not reuse the doc's existing "8 : 2.5 : 1"** — that is the *hazard ratio* HR_u, a different quantity from the marginal survival decrement. | `cart_data.survival`; recomputed |
| D4 | **The \$5M flip in context.** At Q ≈ 4 QALYs, α_ref = \$5M is about \$1.25M/QALY — roughly ten times any accepted reimbursement threshold, though still below a regulatory value of a statistical life (~\$11–13M). So the honest reading is: across the entire policy-relevant range of life values the cost-minimising network is invariant, and all of the survival benefit therefore has to come from the operational layer. | — |

---

## One thing not to "fix"

The doc and `manuscript/methods.md` number the strategic equations differently:

* **doc** — survival extension (27)–(31), manufacturing queue (32)–(35)
* **manuscript** — manufacturing queue (27)–(30), survival linearisation (31)–(34), ND (35)

This is a deliberate reordering in the manuscript (queue first, since the queue
is what creates the hold that survival then prices), not an error in either.
Decide which order you want and align them on purpose; don't let one silently
overwrite the other. The manuscript's LaTeX numbering is automatic, so changing
the order there is free — the .docx is the one with hand-typed numbers.
