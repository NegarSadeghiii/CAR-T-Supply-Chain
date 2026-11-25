"""Trade-off figure of the base case, made from results_doe/all_runs.csv after run_doe.py has finished.

x = E_D (expected patients ineligible at delivery), y = total cost. The line is the mean over the replications of
the five PatientAware levels (labelled by their mean eps); Baseline-MinRisk and Baseline-FIFO are separate points.
Bars show the min-max over the replications.

Usage:  python figures.py                    (reads results_doe/)
        python figures.py --out results_doe_smoke
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_CASE = dict(ND=41, N=30, layout="C3", share=0.30)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PA_COLOR, MINRISK_COLOR, FIFO_COLOR = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results_doe"))
    args = ap.parse_args()
    out = Path(args.out)
    df = pd.read_csv(out / "all_runs.csv")
    bc = df[(df.ND == BASE_CASE["ND"]) & (df.N == BASE_CASE["N"]) & (df.layout == BASE_CASE["layout"])
            & np.isclose(df.urgent_share, BASE_CASE["share"]) & df.cost.notna()]
    if bc.empty:
        raise SystemExit(f"no base-case results in {out / 'all_runs.csv'}")

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    k = 1e3

    def stats(sub):
        return (sub.E_D.mean(), sub.E_D.min(), sub.E_D.max(), sub.cost.mean() / k, sub.cost.min() / k, sub.cost.max() / k)

    pa = bc[bc.eps_step.notna()].groupby("eps_step")
    pts = [(step, sub.eps.mean(), *stats(sub)) for step, sub in pa]
    xs, ys = [p[2] for p in pts], [p[5] for p in pts]
    for step, eps, x, xlo, xhi, y, ylo, yhi in pts:
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], fmt="none", ecolor=GRID,
                    elinewidth=1.3, capsize=3, zorder=1)
        ax.annotate(f"eps={eps:.3f}", (x, y), xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK2)
    ax.plot(xs, ys, color=PA_COLOR, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, zorder=3,
            label="PatientAware: min cost s.t. E_D <= eps (mean)")
    for model, color, marker in [("Baseline-MinRisk", MINRISK_COLOR, "D"), ("Baseline-FIFO", FIFO_COLOR, "s")]:
        sub = bc[bc.model == model]
        if sub.empty:
            continue
        x, xlo, xhi, y, ylo, yhi = stats(sub)
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]], fmt="none", ecolor=GRID,
                    elinewidth=1.3, capsize=3, zorder=1)
        ax.plot([x], [y], ls="none", marker=marker, ms=8, color=color, mec=SURF, mew=1.5, zorder=4, label=f"{model} (mean)")
    n = bc.rep.nunique()
    ax.set_xlabel("E_D: expected patients ineligible at delivery")
    ax.set_ylabel("total cost (thousand USD)")
    ax.set_title(f"Cost vs E_D, base case ND {BASE_CASE['ND']}, N {BASE_CASE['N']}, {BASE_CASE['layout']}, urgent 30% "
                 f"(mean of {n} replications; bars = min-max)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False)
    ax.margins(x=0.15, y=0.15)
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"tradeoff_base_case.{ext}", dpi=200)
    print(f"saved {out / 'figures' / 'tradeoff_base_case.png'} and .pdf")


if __name__ == "__main__":
    main()
