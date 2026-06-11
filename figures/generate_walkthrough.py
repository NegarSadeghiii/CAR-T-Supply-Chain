"""
figures/generate_walkthrough.py — Figure 14 supply-chain walkthrough.

Tier-aggregate side-by-side comparison: 3×2 panel grid.
NO per-patient circles — all representations are tier-aggregated.

Panels (row × col):
  [0,*] Stage-1 network: tier nodes → facility nodes via trapezoidal flow bands
  [1,*] Scenario ω:      per-tier failure counts + drawn × marks
  [2,*] Recourse:        per-tier outcome stat cards

Run: python figures/generate_walkthrough.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

from figure_style import COLORS, setup_style, save_figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon

setup_style()

# ── palette / constants ────────────────────────────────────────────────────────
_TC         = {"H": COLORS["tier_H"], "M": COLORS["tier_M"], "L": COLORS["tier_L"]}
_BETA       = {"H": 0.55, "M": 0.78, "L": 0.92}
_TIER_SIZES = {"H": 10, "M": 25, "L": 15}
_TIER_ORDER = ["H", "M", "L"]
_TIER_IDXS  = {"H": range(0, 10), "M": range(10, 35), "L": range(35, 50)}
_DET_COL    = COLORS["naive_det"]
_SP_COL     = COLORS["sp"]

# Tier node layout (shared across panels, axes coords 0..1)
_TIER_CY = {"H": 0.82, "M": 0.50, "L": 0.18}
_TIER_H  = {"H": 0.09, "M": 0.20, "L": 0.12}

# Facility y-positions per column (position dominant open facility high)
# Det: m0(cap=40) at top, m1(cap=10) at bottom
# SP:  m2(cap=40) at top, m0(cap=10) at bottom
_FAC_CY = {
    "det": {0: 0.72, 1: 0.30, 2: 0.52, 3: 0.10},
    "sp":  {0: 0.30, 1: 0.52, 2: 0.72, 3: 0.10},
}
_FAC_LABELS = ["m₀", "m₁", "m₂", "m₃"]
_P_VALS     = [0.85, 0.92, 0.95, 0.92]


# ── data utilities ─────────────────────────────────────────────────────────────

def _load():
    with open(_REPO / "case_study_results.json") as f:
        d = json.load(f)
    inst = d["instance"]
    return dict(
        tiers      = inst["tiers"],
        p          = inst["p"],
        beta       = inst["beta"],
        rho_cancel = inst["rho_cancel"],
        rho_remfg  = inst["rho_remfg"],
        z_det = d["ev"]["z"],  C_det = d["ev"]["C"],
        x_det = np.array(d["ev"]["x"]),
        z_sp  = d["rp"]["z"],  C_sp  = d["rp"]["C"],
        x_sp  = np.array(d["rp"]["x"]),
    )


def _tier_routing(x_mat):
    """Returns {tier: {fac: count}} (only non-zero entries)."""
    result = {}
    for t, rng in _TIER_IDXS.items():
        row = {}
        for m in range(4):
            cnt = int(x_mat[list(rng), m].sum())
            if cnt > 0:
                row[m] = cnt
        result[t] = row
    return result


def _greedy_recourse(data, x, C, z, Y, B):
    n   = len(data["tiers"])
    out = ["ok"] * n
    spare = [
        max(0, int(C[m]) - int(sum(x[i, m] * Y[i, m] for i in range(n))))
        if z[m] else 0
        for m in range(4)
    ]
    for i in sorted(range(n), key=lambda i: -data["rho_cancel"][i]):
        m0 = int(np.argmax(x[i]))
        if Y[i, m0]:
            continue
        if not B[i]:
            out[i] = "cancel"
            continue
        placed = False
        for m in sorted(range(4), key=lambda m: data["rho_remfg"][m] if z[m] else 1e9):
            if z[m] and spare[m] >= 1:
                out[i] = "remfg"
                spare[m] -= 1
                placed = True
                break
        if not placed:
            out[i] = "cancel"
    return out


def _find_scenario(data):
    rng  = np.random.default_rng(4242)
    p    = np.array(data["p"])
    beta = np.array(data["beta"])
    n    = len(data["tiers"])
    for _ in range(2000):
        Y = rng.binomial(1, p, size=(n, 4)).astype(float)
        B = rng.binomial(1, beta).astype(float)
        do = _greedy_recourse(data, data["x_det"], data["C_det"], data["z_det"], Y, B)
        so = _greedy_recourse(data, data["x_sp"],  data["C_sp"],  data["z_sp"],  Y, B)
        hcd = sum(1 for i in range(10) if do[i] == "cancel")
        hcs = sum(1 for i in range(10) if so[i] == "cancel")
        if hcd - hcs >= 3 and 3 <= hcd <= 7:
            return Y, B, do, so
    # fallback
    rng2 = np.random.default_rng(4242)
    Y  = rng2.binomial(1, p, size=(n, 4)).astype(float)
    B  = rng2.binomial(1, beta).astype(float)
    do = _greedy_recourse(data, data["x_det"], data["C_det"], data["z_det"], Y, B)
    so = _greedy_recourse(data, data["x_sp"],  data["C_sp"],  data["z_sp"],  Y, B)
    return Y, B, do, so


def _fail_counts_by_tier(x_mat, Y):
    return {
        t: sum(1 for i in rng if Y[i, int(np.argmax(x_mat[i]))] == 0)
        for t, rng in _TIER_IDXS.items()
    }


def _outcomes_by_tier(outcomes):
    return {
        t: {k: sum(1 for i in rng if outcomes[i] == k)
            for k in ("ok", "remfg", "cancel")}
        for t, rng in _TIER_IDXS.items()
    }


# ── drawing primitives ─────────────────────────────────────────────────────────

def _fancy_rect(ax, cx, cy, w, h, fc, ec, lw=1.0, zorder=3, alpha=1.0, pad=0.012):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha,
    ))


def _flow_band(ax, x0, y0b, y0t, x1, y1b, y1t, color, alpha=0.30):
    pts = [(x0, y0b), (x0, y0t), (x1, y1t), (x1, y1b)]
    ax.add_patch(MplPolygon(pts, fc=color, ec=color, lw=0, zorder=1, alpha=alpha))


def _factory_icon(ax, cx, cy, h_box, fc, ec, alpha=1.0):
    w = 0.11
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h_box / 2), w, h_box,
        boxstyle="round,pad=0.004",
        fc=fc, ec=ec, lw=1.0, zorder=4, alpha=alpha,
    ))
    roof = [
        (cx - w / 2 - 0.012, cy + h_box / 2),
        (cx + w / 2 + 0.012, cy + h_box / 2),
        (cx, cy + h_box / 2 + 0.055),
    ]
    ax.add_patch(MplPolygon(roof, fc=fc, ec=ec, lw=0.8, zorder=4, alpha=alpha))
    # chimney
    cw, ch = 0.022, 0.035
    ax.add_patch(FancyBboxPatch(
        (cx + 0.018, cy + h_box / 2), cw, ch,
        boxstyle="round,pad=0.001",
        fc=fc, ec=ec, lw=0.5, zorder=4, alpha=alpha,
    ))


# ── Panel 0: Stage-1 network ──────────────────────────────────────────────────

def _panel_network(ax, routing, z, C, plan_col, fac_cy_map):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    tier_cx  = 0.17
    tier_xr  = tier_cx + 0.10   # right edge of tier node
    fac_cx   = 0.76
    fac_xl   = fac_cx - 0.08    # left edge of facility node

    # Capacity-proportional facility heights
    fac_h = {m: max(0.05, C[m] / 50 * 0.32) if z[m] else 0.05 for m in range(4)}

    # Tier-side segments (stacked per open facility, top→bottom)
    tier_seg = {}
    for t in _TIER_ORDER:
        cy_t, h_t = _TIER_CY[t], _TIER_H[t]
        cur = cy_t + h_t / 2
        for m in sorted(m for m in routing[t] if z[m]):
            cnt   = routing[t][m]
            seg_h = cnt / _TIER_SIZES[t] * h_t
            tier_seg[(t, m)] = (cur - seg_h, cur)
            cur -= seg_h

    # Facility-side segments (tiers stacked H→M→L, top→bottom)
    fac_seg = {}
    for m in range(4):
        if not z[m]:
            continue
        cy_f, h_f = fac_cy_map[m], fac_h[m]
        total_f   = sum(routing[t].get(m, 0) for t in _TIER_ORDER)
        if not total_f:
            continue
        cur = cy_f + h_f / 2
        for t in _TIER_ORDER:
            cnt = routing[t].get(m, 0)
            if not cnt:
                continue
            seg_h = cnt / total_f * h_f
            fac_seg[(m, t)] = (cur - seg_h, cur)
            cur -= seg_h

    # Flow bands
    for t in _TIER_ORDER:
        for m in sorted(routing[t]):
            if not z[m]:
                continue
            ts = tier_seg.get((t, m))
            fs = fac_seg.get((m, t))
            if ts and fs:
                _flow_band(ax, tier_xr, ts[0], ts[1], fac_xl, fs[0], fs[1],
                           color=_TC[t], alpha=0.35)

    # Tier nodes
    for t in _TIER_ORDER:
        cy, h = _TIER_CY[t], _TIER_H[t]
        _fancy_rect(ax, tier_cx, cy, 0.18, h, _TC[t], _TC[t], lw=1.5, zorder=3, pad=0.008)
        ax.text(tier_cx, cy + 0.017, f"Tier {t}",
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white", zorder=5)
        ax.text(tier_cx, cy - 0.022,
                f"n={_TIER_SIZES[t]}, β={_BETA[t]}",
                ha="center", va="center", fontsize=5.8, color="white", zorder=5)

    # Facility nodes
    for m in range(4):
        cy    = fac_cy_map[m]
        h_f   = fac_h[m]
        open_ = bool(z[m])
        fc    = plan_col if open_ else "#D4D4D4"
        ec    = plan_col if open_ else "#AAAAAA"
        alph  = 1.0 if open_ else 0.45
        h_box = max(h_f, 0.07)

        _factory_icon(ax, fac_cx, cy, h_box, fc, ec, alpha=alph)

        lbl_y = cy - h_box / 2 - 0.035
        ax.text(fac_cx, lbl_y, _FAC_LABELS[m],
                ha="center", va="top", fontsize=7, fontweight="bold",
                color=(plan_col if open_ else "#AAAAAA"), zorder=5)
        if open_:
            ax.text(fac_cx, lbl_y - 0.065,
                    f"C={C[m]}, p={_P_VALS[m]}",
                    ha="center", va="top", fontsize=5.5, color="#444444", zorder=5)
        else:
            ax.text(fac_cx, lbl_y - 0.065, "closed",
                    ha="center", va="top", fontsize=5.5, color="#AAAAAA", zorder=5)

    ax.text(0.50, 0.962, "FROZEN first-stage decisions",
            ha="center", va="center", fontsize=6.5, fontstyle="italic",
            color="#2255AA", zorder=6,
            bbox=dict(boxstyle="round,pad=0.14", fc="#EEF4FF", ec="#6688CC", lw=0.8))


# ── Panel 1: Scenario ω ───────────────────────────────────────────────────────

def _panel_scenario(ax, fail_dict, plan_col):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for t in _TIER_ORDER:
        cy    = _TIER_CY[t]
        nfail = fail_dict[t]
        n     = _TIER_SIZES[t]
        tc    = _TC[t]

        _fancy_rect(ax, 0.50, cy, 0.86, 0.22, "#FAFAFA", tc, lw=1.0, zorder=2, pad=0.008)

        ax.text(0.13, cy, f"Tier {t}\n({n} pat.)",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=tc, zorder=4)

        fail_col = "#CC0000" if nfail > 0 else COLORS["remfg"]
        ax.text(0.31, cy,
                f"{nfail}\nfailed",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=fail_col, zorder=4)

        if nfail > 0:
            n_show = min(nfail, 10)
            xs = np.linspace(0.43, 0.88, n_show) if n_show > 1 else [0.65]
            for xk in xs:
                d = 0.020
                ax.plot([xk - d, xk + d], [cy + d, cy - d],
                        color="#CC0000", lw=1.6, zorder=5, solid_capstyle="round")
                ax.plot([xk - d, xk + d], [cy - d, cy + d],
                        color="#CC0000", lw=1.6, zorder=5, solid_capstyle="round")
        else:
            ax.text(0.65, cy, "no failures",
                    ha="center", va="center", fontsize=7,
                    color=COLORS["remfg"], zorder=4)

    ax.text(0.50, 0.958,
            "Same Y,B scenario applied to both plans",
            ha="center", va="top", fontsize=6.5, fontstyle="italic", color="#666666")


# ── Panel 2: Recourse outcomes ────────────────────────────────────────────────

def _panel_recourse(ax, outc_dict, plan_col, total_cancel, h_cancel):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    COL_OK  = COLORS["remfg"]
    COL_RM  = COLORS["subcontract"]
    COL_CAN = COLORS["cancel"]

    for t in _TIER_ORDER:
        cy = _TIER_CY[t]
        d  = outc_dict[t]
        tc = _TC[t]

        _fancy_rect(ax, 0.50, cy, 0.86, 0.22, "#FAFAFA", tc, lw=1.0, zorder=2, pad=0.008)

        ax.text(0.13, cy, f"Tier {t}\n({_TIER_SIZES[t]} pat.)",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=tc, zorder=4)

        for k, (lbl, cnt, col) in enumerate([
            ("Treated",   d["ok"],     COL_OK),
            ("Re-mfg",    d["remfg"],  COL_RM),
            ("Cancelled", d["cancel"], COL_CAN),
        ]):
            xk = 0.39 + k * 0.215
            ax.text(xk, cy + 0.025, str(cnt),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=col, zorder=4)
            ax.text(xk, cy - 0.042, lbl,
                    ha="center", va="center", fontsize=5.5, color="#444444", zorder=4)

    sum_col = COL_CAN if h_cancel > 0 else COL_OK
    ax.text(0.50, 0.958,
            f"Cancelled: {total_cancel} total  (Tier-H: {h_cancel}/10)",
            ha="center", va="top", fontsize=7.5, fontweight="bold",
            color=sum_col, zorder=5,
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="#FFF0F0" if h_cancel > 0 else "#F0FFF0",
                      ec=sum_col, lw=0.9))


# ── Main assembly ──────────────────────────────────────────────────────────────

def generate_figure14(data, Y, B, det_out, sp_out):
    routing_det = _tier_routing(data["x_det"])
    routing_sp  = _tier_routing(data["x_sp"])

    fail_det = _fail_counts_by_tier(data["x_det"], Y)
    fail_sp  = _fail_counts_by_tier(data["x_sp"],  Y)

    outc_det = _outcomes_by_tier(det_out)
    outc_sp  = _outcomes_by_tier(sp_out)

    det_h_can     = outc_det["H"]["cancel"]
    sp_h_can      = outc_sp["H"]["cancel"]
    det_total_can = sum(v["cancel"] for v in outc_det.values())
    sp_total_can  = sum(v["cancel"] for v in outc_sp.values())

    fig, axes = plt.subplots(3, 2, figsize=(12, 9),
                              gridspec_kw={"hspace": 0.38, "wspace": 0.10})
    fig.patch.set_facecolor("white")

    axes[0, 0].set_title(
        "Deterministic plan\n"
        "Opens m₀ (cap=40, p=0.85) + m₁ (cap=10, p=0.92)",
        fontsize=9.5, fontweight="bold", color=_DET_COL, pad=6,
    )
    axes[0, 1].set_title(
        "Stochastic plan\n"
        "Opens m₀ (cap=10, p=0.85) + m₂ (cap=40, p=0.95)",
        fontsize=9.5, fontweight="bold", color=_SP_COL, pad=6,
    )

    row_labels = ["Stage-1\nnetwork", "Scenario ω\n(shared)", "Recourse\noutcomes"]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].text(
            -0.10, 0.50, lbl,
            transform=axes[row, 0].transAxes,
            ha="right", va="center", fontsize=8, fontstyle="italic",
            color="#555555",
        )

    _panel_network(axes[0, 0], routing_det, data["z_det"], data["C_det"],
                   _DET_COL, _FAC_CY["det"])
    _panel_network(axes[0, 1], routing_sp,  data["z_sp"],  data["C_sp"],
                   _SP_COL,  _FAC_CY["sp"])

    _panel_scenario(axes[1, 0], fail_det, _DET_COL)
    _panel_scenario(axes[1, 1], fail_sp,  _SP_COL)

    _panel_recourse(axes[2, 0], outc_det, _DET_COL, det_total_can, det_h_can)
    _panel_recourse(axes[2, 1], outc_sp,  _SP_COL,  sp_total_can,  sp_h_can)

    fig.suptitle(
        "Figure 14 — CAR-T Supply Chain: Deterministic vs. Stochastic Plan  "
        "(representative OOS scenario, seed 4242)",
        fontsize=11, fontweight="bold", y=1.005,
    )

    # Vertical divider between columns
    fig.add_artist(mlines.Line2D(
        [0.505, 0.505], [0.03, 0.97],
        transform=fig.transFigure,
        color="#CCCCCC", lw=0.9, zorder=0,
    ))

    plt.tight_layout(rect=[0.05, 0.03, 1.0, 1.0])

    fig.text(0.50, 0.008,
             "Scenario: OOS seed 4242 (N=2000).  "
             "Recourse: greedy proxy — remfg at cheapest open facility "
             "with spare capacity, else cancel.",
             ha="center", va="bottom", fontsize=6.5,
             fontstyle="italic", color="#888888")

    save_figure(fig, "figure14_supply_chain_walkthrough")
    return det_h_can, sp_h_can, det_total_can, sp_total_can


# ── README update ──────────────────────────────────────────────────────────────

def update_readme(det_h_can, sp_h_can, det_can, sp_can):
    import re as _re
    readme   = _HERE / "README.md"
    existing = readme.read_text()

    entry = (
        "| `figure14_supply_chain_walkthrough.png` | "
        "Side-by-side five-row supply-chain walkthrough comparing the "
        "Deterministic and Stochastic plans on a representative out-of-sample "
        "scenario (seed 4242). "
        "Row 1 (patient cohort, identical in both columns): 50 patients colour-coded "
        "by urgency tier (H red ×10, M amber ×25, L green ×15). "
        "Row 2 (first-stage plan): factory icons show which facilities are opened — "
        "Det opens m₀ (cap 40) + m₁ (cap 10); SP opens m₀ (cap 10) + m₂ (cap 40, "
        "p=0.95 highest-yield). "
        "Thin tier-coloured lines connect each patient to their assigned facility "
        "(frozen first-stage). "
        "Row 4 (scenario ω): same stochastic realization overlaid on both columns; "
        "red X marks patients who fail at their assigned facility. "
        "Row 5 (recourse): check = treated, R = re-manufactured (spare capacity used), "
        "X = cancelled. "
        f"In this scenario Det cancels {det_can} patients ({det_h_can} Tier-H); "
        f"SP cancels {sp_can} ({sp_h_can} Tier-H). "
        "| none (schematic, OOS seed 4242) |\n"
    )

    cleaned = _re.sub(
        r"\| `figure14_supply_chain[^|]+\| none[^|]+\|\n?", "", existing
    )
    readme.write_text(cleaned.rstrip("\n") + "\n" + entry)
    print(f"  Updated: {readme}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=== Figure 14 — supply-chain walkthrough (tier-aggregate) ===\n")

    print("[1/3] Loading data …")
    data = _load()

    print("[2/3] Finding illustrative OOS scenario …")
    Y, B, det_out, sp_out = _find_scenario(data)

    hcd = sum(1 for i in range(10) if det_out[i] == "cancel")
    hcs = sum(1 for i in range(10) if sp_out[i]  == "cancel")
    cd  = sum(o == "cancel" for o in det_out)
    cs  = sum(o == "cancel" for o in sp_out)
    print(f"  Scenario: Det H-cancel={hcd}, SP H-cancel={hcs}")
    print(f"            Det total-cancel={cd}, SP total-cancel={cs}")

    print("[3/3] Generating figure14_supply_chain_walkthrough …")
    generate_figure14(data, Y, B, det_out, sp_out)

    print()
    update_readme(hcd, hcs, cd, cs)

    print("\nOutput verification:")
    ok = True
    for ext in ("png", "pdf"):
        p      = _HERE / ext / f"figure14_supply_chain_walkthrough.{ext}"
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}")
        if status != "OK":
            ok = False

    if not ok:
        sys.exit(1)
    print("\nFigure 14 generated successfully.")


if __name__ == "__main__":
    main()
