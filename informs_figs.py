"""Deck-styled figures for the INFORMS runs.

The palette, canvas size and caption rule come from the slide template, so
every PNG here drops straight onto a slide without restyling.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BG, INK, MUTED, GRID = "#F5F0E6", "#1D2733", "#4B5563", "#ECE5D8"
RISK = {"High": "#B23A3A", "Low": "#2F5DA8"}
STAGE = {"collection": "#5F6E3A", "transport": "#B88A3E", "waiting": "#9A8C7A",
         "manufacturing": "#2F6B66", "qc": "#5E5080"}

# 1600 x 900 px at 200 dpi
FIGSIZE, DPI = (8.0, 4.5), 200


def font_family():
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    return "IBM Plex Sans" if "IBM Plex Sans" in names else "DejaVu Sans"


FONT = font_family()


def use_style():
    plt.rcParams.update({
        "font.family": FONT,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": DPI, "savefig.dpi": DPI,
    })


def caption(curves, n_seeds, extra=""):
    H, L = curves["High"], curves["Low"]
    s = (f"Illustrative curves: H eta={H['eta']:g}, k={H['k']:g}; "
         f"L eta={L['eta']:g}, k={L['k']:g}")
    if n_seeds:
        s += f" · {n_seeds} seed" + ("s" if n_seeds > 1 else "")
    s += " · base instance"
    return s + (" · " + extra if extra else "")


def finish(fig, path, cap):
    fig.text(0.01, 0.012, cap, fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Experiment 1
# -----------------------------------------------------------------------------
POLICY_LABEL = {
    "cost_speed": "cost + speed", "fifo": "FIFO",
    "high_risk_first": "high risk first", "elig_min": "worst same-cost plan",
    "elig_max": "eligibility-aware (ours)"}


def fig_exp1_dotplot(runs, curves, path, order=None):
    """One row per policy, one dot per seed, mean marked.

    A policy that could not be run at C_min sits at its own minimum cost, so it
    is NOT comparable with the rest of the chart.  Those rows are drawn hollow
    and labelled with the budget they needed -- otherwise the eye reads a higher
    dot as a better policy when it is really a more expensive one.
    """
    use_style()
    order = order or ["elig_min", "fifo", "high_risk_first", "cost_speed",
                      "elig_max"]
    order = [p for p in order if p in set(runs.policy)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(0)
    base_cost = runs[runs.budget == "C_min"].cost.mean()
    ticks, offbudget = [], False
    for y, pol in enumerate(order):
        sub = runs[runs.policy == pol]
        own = (sub.budget == "own_min_cost").all()
        col = RISK["High"] if pol == "elig_max" else MUTED
        face = BG if own else col
        offbudget = offbudget or own
        jitter = (rng.random(len(sub)) - 0.5) * 0.22
        ax.scatter(sub.ELIG, np.full(len(sub), y) + jitter, s=26,
                   facecolor=face, edgecolor=col if own else BG,
                   linewidth=1.0 if own else 0.8, alpha=.85, zorder=3)
        ax.scatter([sub.ELIG.mean()], [y], marker="D", s=58, facecolor=face,
                   edgecolor=col, linewidth=1.4, zorder=4)
        note = f"mean {sub.ELIG.mean():.3f}"
        if own:
            note += (f"  (needs +{100 * (sub.cost.mean() / base_cost - 1):.0f}%"
                     f" budget)")
        ax.text(sub.ELIG.mean(), y - 0.33, note, ha="center", va="bottom",
                fontsize=8, color=col)
        ticks.append(POLICY_LABEL.get(pol, pol)
                     + (" *" if own else ""))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ticks)
    ax.set_ylim(-0.7, len(order) - 0.4)
    ax.set_xlabel("expected eligible patients at delivery (of 10)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    extra = ("same cost, except * (hollow) at its own minimum" if offbudget
             else "all plans at the same cost")
    return finish(fig, path, caption(curves, runs.seed.nunique(), extra))


def draw_swimlane(ax, d, inst, title):
    """The notebook's stage geometry (stages()), drawn in the deck palette.

    fig_swimlane builds its own figure and cannot take an external axes, so the
    paired figure redraws from the same `stages` frame rather than forking the
    notebook's plotting code.
    """
    h = 0.30
    for y, r in d.reset_index(drop=True).iterrows():
        ax.plot([r.a_p, r.release], [y, y], color=STAGE["collection"], lw=2.0,
                solid_capstyle="butt", zorder=2)
        for (x0, x1) in ((r.out0, r.out1), (r.ret0, r.ret1)):
            ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=4,
                        arrowprops=dict(arrowstyle="-|>",
                                        color=STAGE["transport"], lw=1.6,
                                        shrinkA=0, shrinkB=0, mutation_scale=8))
        if r.W > 0:
            ax.add_patch(plt.Rectangle((r.wait0, y - h * .62),
                                       r.wait1 - r.wait0, h * 1.24,
                                       facecolor=STAGE["waiting"], alpha=.55,
                                       edgecolor="none", zorder=3))
        ax.add_patch(plt.Rectangle((r.mfg0, y - h), r.mfg1 - r.mfg0, 2 * h,
                                   facecolor=STAGE["manufacturing"],
                                   edgecolor=BG, linewidth=1.0, zorder=5))
        ax.add_patch(plt.Rectangle((r.qc0, y - h), r.qc1 - r.qc0, 2 * h,
                                   facecolor=STAGE["qc"], edgecolor=BG,
                                   linewidth=1.0, zorder=5))
        ax.plot([r.delivery], [y], marker="D", ms=4.6, color=INK, zorder=7)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{p} ({g[0]})" for p, g in zip(d.patient, d.group)],
                       fontsize=7.6)
    for lab, g in zip(ax.get_yticklabels(), d.group):
        lab.set_color(RISK[g])
    ax.set_ylim(len(d) - .4, -.8)
    ax.grid(axis="y", visible=False)
    ax.set_ylabel(title, fontsize=9.5)


def fig_exp1_swimlane_pair(m_ref, m_new, inst, curves, path, nb_stages,
                           seed=None):
    use_style()
    a, b = nb_stages(m_ref), nb_stages(m_new)
    n = len(a)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 0.30 * n + 3.0), sharex=True)
    draw_swimlane(axes[0], a, inst, "cost + speed")
    draw_swimlane(axes[1], b, inst, "eligibility-aware")
    axes[1].set_xlabel("day")
    hi = max(a.delivery.max(), b.delivery.max()) + 2
    axes[0].set_xlim(0, hi)
    handles = [
        Line2D([], [], color=STAGE["collection"], lw=2.0,
               label=f"collection to release ({inst['TLS']} d)"),
        Line2D([], [], color=STAGE["transport"], lw=1.6, marker=">",
               markersize=4, label="transport (length = days)"),
        Patch(facecolor=STAGE["waiting"], alpha=.55, label="waiting for a line"),
        Patch(facecolor=STAGE["manufacturing"],
              label=f"manufacturing ({inst['TMFE']} d)"),
        Patch(facecolor=STAGE["qc"], label=f"QC ({inst['TQC']} d)"),
        Line2D([], [], color=INK, marker="D", ls="none", markersize=4.6,
               label="delivery")]
    axes[1].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(.5, -.22), ncol=6, handlelength=1.4,
                   columnspacing=1.1, fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return finish(fig, path, caption(curves, None,
                                     f"seed {seed}" if seed else ""))


# -----------------------------------------------------------------------------
# Experiment 2
# -----------------------------------------------------------------------------
CONGESTION_ORDER = [("low", 25), ("base", 15), ("high", 7)]


def fig_exp2_gain(gains, curves, path, decline_labels):
    """Mean gain per congestion cell, whisker = min-max over seeds."""
    use_style()
    cells = [c for c, _ in CONGESTION_ORDER]
    declines = list(decline_labels)
    fig, axes = plt.subplots(1, len(declines), figsize=FIGSIZE, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dec in zip(axes, declines):
        for off, (kind, col, lab) in enumerate((
                ("same_cost_gain", RISK["High"], "same cost"),
                ("paid_gain", "#D98C8C", "+5% budget"))):
            xs, mean, lo, hi = [], [], [], []
            for i, cell in enumerate(cells):
                sub = gains[(gains.decline == dec) & (gains.congestion == cell)]
                if sub.empty:
                    continue
                xs.append(i + (off - 0.5) * 0.18)
                mean.append(sub[kind].mean())
                lo.append(sub[kind].min()); hi.append(sub[kind].max())
            mean, lo, hi = np.array(mean), np.array(lo), np.array(hi)
            ax.errorbar(xs, mean, yerr=[mean - lo, hi - mean], fmt="o",
                        color=col, ecolor=col, elinewidth=1.4, capsize=3,
                        markersize=6, markeredgecolor=BG, label=lab, zorder=3)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{c}\n({w} d window)" for c, w in CONGESTION_ORDER])
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.set_ylabel("gain in expected eligible patients")
        ax.text(0.02, 0.97, dec, transform=ax.transAxes, va="top", fontsize=9,
                color=MUTED)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.90))
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.supxlabel("congestion", fontsize=10, color=INK, y=0.075)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    return finish(fig, path, caption(curves, gains.seed.nunique()))


# -----------------------------------------------------------------------------
# B1 frontier
# -----------------------------------------------------------------------------
def fig_b1_frontier(front, knee_i, c_min, c_105, curves, path):
    use_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    f = front.sort_values("cost")
    ax.plot(f.cost, f.ELIG, color=MUTED, lw=1.4, zorder=2)
    sup, uns = f[~f.unsupported], f[f.unsupported]
    ax.scatter(sup.cost, sup.ELIG, s=44, facecolor=RISK["Low"], edgecolor=BG,
               linewidth=1.2, zorder=4, label="supported point")
    if len(uns):
        ax.scatter(uns.cost, uns.ELIG, s=44, facecolor=BG,
                   edgecolor=RISK["Low"], linewidth=1.4, zorder=4,
                   label="unsupported point")
    k = f.iloc[knee_i]
    ax.scatter([k.cost], [k.ELIG], s=150, marker="*", color=RISK["High"],
               edgecolor=BG, linewidth=0.8, zorder=5, label="knee")
    for x, lab in ((c_min, "min cost"), (c_105, "+5% budget")):
        ax.axvline(x, color=STAGE["transport"], lw=1.1, ls=(0, (4, 3)),
                   zorder=1)
        ax.text(x, ax.get_ylim()[0], f" {lab}", rotation=90, va="bottom",
                ha="left", fontsize=8, color=STAGE["transport"])
    ax.set_xlabel("total cost (US dollars)")
    ax.set_ylabel("expected eligible patients at delivery (of 10)")
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return finish(fig, path, caption(curves, 1, "base instance, seed 42"))


# -----------------------------------------------------------------------------
# B3 cost side
# -----------------------------------------------------------------------------
def fig_b3_sensitivity(points, base_front, curves, path, ncol=4):
    use_style()
    rows = sorted(points.row.unique())
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.0, 2.0 * nrow + 0.9),
                             sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        sub = points[points.row == row]
        labs = sorted(sub.label.unique(), key=lambda s: (len(s), s))
        cmap = plt.get_cmap("copper")
        for i, lab in enumerate(labs):
            s = sub[sub.label == lab].sort_values("cost")
            ax.plot(s.cost / 1000, s.ELIG, lw=1.5, marker="o", ms=3.0,
                    color=cmap(0.15 + 0.7 * i / max(len(labs) - 1, 1)),
                    label=lab)
        ax.plot(base_front.cost / 1000, base_front.ELIG, lw=1.2, ls=(0, (4, 3)),
                color=INK, zorder=1, label="base")
        ax.text(0.03, 0.96, row, transform=ax.transAxes, va="top", fontsize=8,
                color=INK)
        ax.legend(fontsize=6.4, loc="lower right", handlelength=1.2,
                  labelspacing=.25, borderpad=.1)
        ax.tick_params(labelsize=7.5)
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    fig.supxlabel("total cost (thousand US dollars)", fontsize=9.5, color=INK,
                  y=0.055)
    fig.supylabel("expected eligible patients (of 10)", fontsize=9.5,
                  color=INK, x=0.015)
    fig.tight_layout(rect=(0.03, 0.065, 1, 1))
    return finish(fig, path, caption(curves, 1, "one factor at a time, seed 42"))


def fig_b3_costmix(mix, curves, path):
    use_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    comps = [("CTM", "manufacturing", STAGE["manufacturing"]),
             ("TTC", "transport", STAGE["transport"]),
             ("HOLD", "holding", STAGE["waiting"])]
    labels = [f"{r.config}\n{r.point}" for _, r in mix.iterrows()]
    x = np.arange(len(mix))
    bottom = np.zeros(len(mix))
    for col, lab, c in comps:
        v = mix[col].values / 1000
        ax.bar(x, v, bottom=bottom, width=0.62, color=c, edgecolor=BG,
               linewidth=1.2, label=lab, zorder=3)
        bottom += v
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + 0.6, f"{tot:,.1f}k", ha="center", fontsize=7.6,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("cost (thousand US dollars)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return finish(fig, path, caption(curves, 1, "seed 42"))
