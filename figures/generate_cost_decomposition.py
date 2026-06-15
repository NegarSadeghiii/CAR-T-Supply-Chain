"""
figures/generate_cost_decomposition.py — cost decomposition stacked bar chart.

Reads per-scenario recourse arrays from results/out_of_sample_results.json
(populated by the augmented out_of_sample_evaluation.py) and decomposes mean
total cost for each of the three plans into six components:

  Stage 1 (committed):
    1. Facility opening  — f[m] * z[m]
    2. Capacity          — pi[m] * C[m]
    3. Primary mfg       — c[m] * sum_i x[i,m]

  Stage 2 (expected recourse, averaged over 2,000 OOS scenarios):
    4. Re-manufacture    — (rho_leuk + rho_remfg[m]) * r_remfg[i,m]
    5. Subcontract       — (rho_leuk + rho_sub[m,mp]) * r_sub[i,m,mp]
    6. Cancellation      — rho_cancel[i] * r_cancel[i]

Run: python figures/generate_cost_decomposition.py
Source: results/out_of_sample_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))

from figure_style import COLORS, setup_style, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

setup_style()

_RESULT = _REPO / "results" / "out_of_sample_results.json"
_NPZ    = _REPO / "results" / "out_of_sample_recourse.npz"

# Instance parameters (calibrated case-study, fixed across all plans)
_F   = np.array([0.5, 2.0, 3.0, 2.0])
_PI  = np.array([0.04, 0.06, 0.09, 0.06])
_C   = np.array([0.20, 0.18, 0.15, 0.18])
_RHO_LEUK   = 0.005
_RHO_REMFG  = _C.copy()       # rho_remfg[m] = c[m]
_RHO_SUB    = np.zeros((4, 4))
for _m in range(4):
    for _mp in range(4):
        if _m != _mp:
            _RHO_SUB[_m, _mp] = _C[_mp] * 1.15

# Tier-H patients: indices 0-9 (from case_study.py TIER_IDX)
_N_P = 50
_RHO_CANCEL = np.array(
    [6.00] * 10 + [2.00] * 25 + [0.75] * 15
)   # H: indices 0-9, M: 10-34, L: 35-49

PLAN_KEYS   = ["naive_deterministic", "expected_cost_deterministic", "stochastic_plan"]
PLAN_LABELS = ["Deterministic", "Expected-cost det.", "Stochastic plan"]
PLAN_COLORS = [COLORS["naive_det"], COLORS["expected_cost_det"], COLORS["sp"]]

COMP_LABELS = ["Facility opening", "Capacity", "Primary mfg",
               "Re-manufacture", "Subcontract", "Cancellation"]
COMP_COLORS = [
    "#5B9BD5",   # blue  — facility opening
    "#9EC5E8",   # light blue — capacity
    "#BDD7EE",   # pale blue — primary mfg
    COLORS["remfg"],        # green — re-mfg
    COLORS["subcontract"],  # amber — subcontract
    COLORS["cancel"],       # red   — cancellation
]


def _load():
    with open(_RESULT) as f:
        d = json.load(f)

    # Load per-scenario recourse arrays from compressed numpy file
    npz = np.load(str(_NPZ))

    results = {}
    for key in PLAN_KEYS:
        pr = d["plans"][key]

        # ── stage-1 decomposition ──────────────────────────────────────
        fs = pr["first_stage"]
        z  = np.array(fs["z"])
        C  = np.array(fs["C"])
        x  = np.array(fs["x"])            # (n_p, n_f)

        open_cost = float((_F  * z).sum())
        cap_cost  = float((_PI * C).sum())
        mfg_cost  = float((_C  * x.sum(axis=0)).sum())

        # ── stage-2 decomposition (mean over scenarios) ────────────────
        pfx      = pr["_recourse_npz_key"]       # e.g. "naive_deterministic"
        r_remfg  = npz[f"{pfx}__r_remfg"]        # (N, n_p, n_f)
        r_sub    = npz[f"{pfx}__r_sub"]           # (N, n_p, n_f, n_f)
        r_cancel = npz[f"{pfx}__r_cancel"]        # (N, n_p)

        rho_r = _RHO_LEUK + _RHO_REMFG           # (n_f,)
        rho_s = _RHO_LEUK + _RHO_SUB             # (n_f, n_f)

        s2_remfg  = (r_remfg  * rho_r[np.newaxis, np.newaxis, :]).sum(axis=(1, 2))
        s2_sub    = (r_sub    * rho_s[np.newaxis, np.newaxis, :, :]).sum(axis=(1, 2, 3))
        s2_cancel = (r_cancel * _RHO_CANCEL[np.newaxis, :]).sum(axis=1)

        mean_remfg  = float(s2_remfg.mean())
        mean_sub    = float(s2_sub.mean())
        mean_cancel = float(s2_cancel.mean())

        results[key] = [open_cost, cap_cost, mfg_cost,
                        mean_remfg, mean_sub, mean_cancel]

    return results


def generate_figure() -> None:
    data = _load()

    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos  = np.arange(len(PLAN_KEYS))
    bar_w  = 0.55

    bottoms = np.zeros(len(PLAN_KEYS))
    bars = []
    for k, (lbl, col) in enumerate(zip(COMP_LABELS, COMP_COLORS)):
        vals = np.array([data[key][k] for key in PLAN_KEYS])
        b = ax.bar(x_pos, vals, bar_w, bottom=bottoms,
                   color=col, label=lbl, zorder=3)
        bars.append(b)
        bottoms += vals

    # Total-cost annotation on top of each bar
    for j, key in enumerate(PLAN_KEYS):
        total = sum(data[key])
        ax.text(x_pos[j], total + 0.005, f"{total:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(PLAN_LABELS, fontsize=10)
    ax.set_ylabel("Mean total cost (M USD)", fontsize=11)
    ax.set_title("Cost decomposition across planning strategies\n"
                 "(Stage-1 committed + expected Stage-2 recourse, 2,000 OOS scenarios)",
                 fontsize=11)

    # Horizontal dashed divider between Stage-1 and Stage-2 (at end of mfg bar)
    # (visual guide only — position differs per plan, so we skip it)

    ax.legend(handles=[mpatches.Patch(color=c, label=l)
                       for l, c in zip(COMP_LABELS, COMP_COLORS)],
              fontsize=8, loc="upper right", ncol=1,
              title="Cost component", title_fontsize=8.5)

    ax.set_xlim(-0.5, len(PLAN_KEYS) - 0.5)
    ax.set_ylim(0, max(sum(v) for v in data.values()) * 1.12)

    save_figure(fig, "figure_cost_decomposition")


def update_readme() -> None:
    import re as _re
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    entry = (
        "| `figure_cost_decomposition.png` | "
        "Stacked-bar cost decomposition across the three planning strategies, "
        "breaking total mean cost (2,000 OOS scenarios) into six components: "
        "facility opening, capacity, primary manufacturing (Stage 1), and "
        "re-manufacture, subcontract, cancellation (expected Stage 2). "
        "| `results/out_of_sample_results.json` |\n"
    )

    cleaned = _re.sub(r"\| `figure_cost_decomposition[^|]+\|[^|]+\|\n?", "", existing)
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


def main() -> None:
    print("=== Cost decomposition figure ===\n")
    print("[1/2] Generating figure_cost_decomposition …")
    generate_figure()

    print()
    update_readme()

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p = _HERE / ext / f"figure_cost_decomposition.{ext}"
        if p.exists():
            print(f"  [OK]      {p.name}  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  [MISSING] {p.name}")
            ok = False

    if not ok:
        sys.exit(1)
    print("\nCost decomposition figure generated successfully.")


if __name__ == "__main__":
    main()
