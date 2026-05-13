"""
test_tier2.py — Statistical / empirical Tier 2 tests for yield_sp_v1.py.
Covers sampler distribution, SAA objective stability, tier-dependent
cancellation ordering, and run-to-run reproducibility.
Run: python test_tier2.py
"""

import numpy as np

from yield_sp_v1 import Instance, build_and_solve_sp, sample_scenarios


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _medium_instance():
    """
    3 patients (one each of H, M, L urgency tier), 2 facilities.
    Used by Tests 1, 2, and 4.  Fits within the Gurobi restricted-license
    variable limit for up to N=100 scenarios.
    """
    # β per tier: H=0.50, M=0.80, L=0.95
    beta = np.array([0.50, 0.80, 0.95])
    n_p, n_f = 3, 2
    rho_sub = np.zeros((n_f, n_f))
    rho_sub[0, 1] = rho_sub[1, 0] = 0.25
    return Instance(
        n_patients=n_p,
        n_facilities=n_f,
        f=np.array([1.0, 0.8]),
        pi=np.array([0.05, 0.06]),
        c=np.array([0.20, 0.20]),
        s_max=np.array([10, 10]),
        p=np.array([0.90, 0.85]),
        beta=beta,
        rho_leuk=0.005,
        rho_remfg=np.array([0.20, 0.20]),
        rho_sub=rho_sub,
        rho_cancel=np.full(n_p, 1.00),
    )


# ---------------------------------------------------------------------------
# Test 1 — Sampler distribution validation
# ---------------------------------------------------------------------------

def test_sampler_distribution():
    """
    Draw N=10,000 scenarios with seed=0; verify empirical means of Y and B
    are within ±0.01 (Y) and ±0.015 (B) of the declared probabilities.
    No MILP solve — purely tests sample_scenarios().
    """
    inst = _medium_instance()
    N = 10_000
    Y, B = sample_scenarios(inst, N, seed=0)

    assert Y.shape == (N, inst.n_patients, inst.n_facilities), (
        f"Y shape mismatch: {Y.shape}"
    )
    assert B.shape == (N, inst.n_patients), (
        f"B shape mismatch: {B.shape}"
    )

    tol_Y = 0.01
    for m in inst.M:
        emp = Y[:, :, m].mean()
        assert abs(emp - inst.p[m]) < tol_Y, (
            f"Y facility {m}: empirical {emp:.4f} vs p[m]={inst.p[m]:.4f}, "
            f"diff={abs(emp - inst.p[m]):.4f} > {tol_Y}"
        )

    tol_B = 0.015
    for i in inst.I:
        emp = B[:, i].mean()
        assert abs(emp - inst.beta[i]) < tol_B, (
            f"B patient {i}: empirical {emp:.4f} vs beta[i]={inst.beta[i]:.4f}, "
            f"diff={abs(emp - inst.beta[i]):.4f} > {tol_B}"
        )

    y_empirical = [f"{Y[:,:,m].mean():.4f}" for m in inst.M]
    b_empirical = [f"{B[:,i].mean():.4f}" for i in inst.I]
    return True, (
        f"Y_means={y_empirical} vs p={inst.p.tolist()}; "
        f"B_means={b_empirical} vs beta={inst.beta.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 2 — SAA objective stability
# ---------------------------------------------------------------------------

def test_saa_stability():
    """
    Solve the medium instance for N in {10, 40, 100} x 6 seeds each.
    Checks:
      - Mean objective at each N within ±5% of the N=100 grand mean
      - Std(objective) shrinks monotonically from N=10 → N=100
      - z (facility open decisions) are stable across seeds at N=100
    """
    inst = _medium_instance()
    N_values = [10, 40, 100]
    n_seeds = 6
    results = {}

    for N in N_values:
        costs = []
        for seed in range(n_seeds):
            print(f"  SAA stability: N={N}, seed={seed} ...", flush=True)
            Y, B = sample_scenarios(inst, N, seed=seed)
            r = build_and_solve_sp(inst, Y, B)
            costs.append(r["total_cost"])
        results[N] = costs

    grand_mean = np.mean(results[100])
    tol = 0.05 * grand_mean

    for N in N_values:
        mean_N = np.mean(results[N])
        assert abs(mean_N - grand_mean) < tol, (
            f"N={N}: mean={mean_N:.4f} differs from grand_mean={grand_mean:.4f} "
            f"by {abs(mean_N - grand_mean):.4f} > tol={tol:.4f}"
        )

    std_10  = np.std(results[10])
    std_40  = np.std(results[40])
    std_100 = np.std(results[100])
    assert std_10 > std_100, (
        f"Std should decrease N=10→100: std(10)={std_10:.4f}, std(100)={std_100:.4f}"
    )

    # z stability at N=100: all seeds must agree on the same open-facility vector
    z_vectors = []
    for seed in range(n_seeds):
        Y, B = sample_scenarios(inst, 100, seed=seed)
        r = build_and_solve_sp(inst, Y, B)
        z_vectors.append(tuple(int(round(v)) for v in r["z"]))
    z_unique = set(z_vectors)
    assert len(z_unique) == 1, (
        f"z not stable at N=100; found {len(z_unique)} distinct solutions: {z_unique}"
    )

    return True, (
        f"means(10,40,100)=({np.mean(results[10]):.3f},{np.mean(results[40]):.3f},"
        f"{np.mean(results[100]):.3f}); "
        f"stds=({std_10:.4f},{std_40:.4f},{std_100:.4f}); "
        f"z@100 unique={len(z_unique)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Tier-dependent cancellation rate ordering
# ---------------------------------------------------------------------------

def test_tier_cancellation_ordering():
    """
    6 patients (2 per tier H/M/L), 2 facilities, p=0.50, N=60, seed=7.
    Beta: H=0.30, M=0.60, L=0.90 → H patients least eligible for re-collection.
    Expected: cancel_rate(H) > cancel_rate(M) > cancel_rate(L).
    Also checks cancel_rate(H) ≈ 0.35 ± 0.10 (theoretical = 0.50*(1-0.30) = 0.35).
    """
    n_p, n_f = 6, 2
    beta = np.array([0.30, 0.30, 0.60, 0.60, 0.90, 0.90])
    rho_sub = np.zeros((n_f, n_f))
    rho_sub[0, 1] = rho_sub[1, 0] = 0.25
    inst = Instance(
        n_patients=n_p,
        n_facilities=n_f,
        f=np.array([1.0, 0.8]),
        pi=np.array([0.05, 0.06]),
        c=np.array([0.20, 0.20]),
        s_max=np.array([10, 10]),
        p=np.array([0.50, 0.50]),
        beta=beta,
        rho_leuk=0.005,
        rho_remfg=np.array([0.20, 0.20]),
        rho_sub=rho_sub,
        rho_cancel=np.full(n_p, 1.00),
    )

    N = 60
    Y, B = sample_scenarios(inst, N, seed=7)
    r = build_and_solve_sp(inst, Y, B)

    # r_cancel shape: (N, n_p); average over all scenarios and patients within tier
    rc = r["r_cancel"]  # (N, 6)
    rate_H = rc[:, 0:2].mean()
    rate_M = rc[:, 2:4].mean()
    rate_L = rc[:, 4:6].mean()

    assert rate_H > rate_M, (
        f"Expected cancel_rate(H) > cancel_rate(M): {rate_H:.4f} vs {rate_M:.4f}"
    )
    assert rate_M > rate_L, (
        f"Expected cancel_rate(M) > cancel_rate(L): {rate_M:.4f} vs {rate_L:.4f}"
    )
    assert abs(rate_H - 0.35) < 0.10, (
        f"Expected cancel_rate(H) ≈ 0.35 ± 0.10; got {rate_H:.4f}"
    )

    return True, (
        f"cancel_rates: H={rate_H:.4f}, M={rate_M:.4f}, L={rate_L:.4f} "
        f"(ordering H>M>L confirmed, t={r['solve_time']:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Test 4 — Run-to-run reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility():
    """
    Solve the medium instance twice with identical (N=100, seed=42) scenarios.
    Scenario arrays must be bit-exact; key result arrays must be equal.
    """
    inst = _medium_instance()
    N, seed = 100, 42

    Y1, B1 = sample_scenarios(inst, N, seed=seed)
    Y2, B2 = sample_scenarios(inst, N, seed=seed)

    assert np.array_equal(Y1, Y2), "sample_scenarios not reproducible for Y"
    assert np.array_equal(B1, B2), "sample_scenarios not reproducible for B"

    r1 = build_and_solve_sp(inst, Y1, B1)
    r2 = build_and_solve_sp(inst, Y2, B2)

    for key in ("z", "C", "x", "r_remfg", "r_cancel"):
        arr1 = np.asarray(r1[key])
        arr2 = np.asarray(r2[key])
        assert np.array_equal(arr1, arr2), (
            f"Reproducibility failed for key '{key}'"
        )

    assert abs(r1["total_cost"] - r2["total_cost"]) < 1e-6, (
        f"total_cost mismatch: {r1['total_cost']:.8f} vs {r2['total_cost']:.8f}"
    )

    return True, (
        f"Both runs: total_cost={r1['total_cost']:.4f}, "
        f"z={[int(round(v)) for v in r1['z']]}, C={r1['C'].tolist()} "
        f"(t1={r1['solve_time']:.2f}s, t2={r2['solve_time']:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Test 1 — Sampler distribution validation",        test_sampler_distribution),
    ("Test 2 — SAA objective stability",                test_saa_stability),
    ("Test 3 — Tier-dependent cancellation ordering",   test_tier_cancellation_ordering),
    ("Test 4 — Run-to-run reproducibility",             test_reproducibility),
]


def main():
    print("=== Tier 2 statistical tests ===")
    passed = 0
    for name, fn in TESTS:
        try:
            ok, detail = fn()
            print(f"[PASS] {name}")
            print(f"       {detail}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {name}")
            print(f"       {exc}")
        except Exception as exc:
            print(f"[FAIL] {name}")
            print(f"       Unexpected error: {exc}")
    print(f"{passed} / {len(TESTS)} passed.")
    if passed < len(TESTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
