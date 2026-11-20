"""
Block 2 / Nov 20 + Nov 22: targeted vaccination experiment on the 26 contact networks.

For each method, vaccinate (remove) the top-k% ranked people, then simulate SIR from a random
unvaccinated index case (calibrated regime, 10% mean outbreak without vaccination) and measure:
  * final outbreak size  (fraction of the whole population)
  * peak size            (largest number infected in one generation, fraction of the population)
All methods are evaluated on the same random live-edge samples (common random numbers), so the
differences between methods are not Monte-Carlo noise from different samples.

References (not competitors): "Random" vaccination, and "True SIR ranking" (vaccinate the real top
spreaders, i.e. a perfect super-spreader detector).

Usage: python run_vaccination.py [--only NAME ...] [--runs 200]
"""
import os
import time
import argparse
import numpy as np
import pandas as pd

import common2 as c2
from common2 import cc

FRACTIONS = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20]
REFS = ["Random", "True SIR ranking"]
ALL = c2.METHODS + REFS


def run_network(name, runs):
    t0 = time.time()
    net = c2.block1_net(name)
    orders = {m: c2.ranking(net.scores[m]) for m in c2.METHODS}
    orders["Random"] = None
    orders["True SIR ranking"] = c2.ranking(net.y)
    res = c2.vaccination(net, orders, FRACTIONS, runs=runs)
    for r in res:
        r.update(dataset=name, group=cc.GROUP_OF[name], n=net.n)
    print(f"  {name:26s} n={net.n:4d}  no vaccination: outbreak={res[0]['outbreak']:.3f} "
          f"peak={res[0]['peak']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return res


def summarise(df):
    """Relative outbreak (vaccinated / unvaccinated) per network, method and fraction."""
    base = df[df.frac == 0].groupby("dataset")[["outbreak", "peak"]].first()
    df = df.join(base, on="dataset", rsuffix="_0")
    df["rel_outbreak"] = df.outbreak / df.outbreak_0
    df["rel_peak"] = df.peak / df.peak_0
    return df


def write_report(df, runs):
    S = c2.SHORT
    at = lambda f: df[np.isclose(df.frac, f)]
    piv = lambda d, col: d.pivot(index="dataset", columns="method", values=col)
    order = list(dict.fromkeys(df.dataset))
    groups = df.groupby("dataset").group.first()

    L = ["# Block 2: Targeted Vaccination on Human Contact Networks", "",
         "Each method vaccinates its top-ranked k% of people (they can neither catch nor pass on the infection). "
         f"SIR is then run from a random unvaccinated index case ({runs} live-edge samples, calibrated regime: "
         "10% mean outbreak without vaccination). The same random samples are used for every method.", "",
         "**Reduction** = 1 - (outbreak with vaccination) / (outbreak without vaccination). Higher is better. "
         "*Random* and *True SIR ranking* (vaccinating the real top spreaders) are references, not competitors. "
         "Best competing method per row in **bold**.", ""]

    def table(f, col, title):
        d = piv(at(f), col).reindex(order)
        red = 1 - d
        L.extend([f"## {title}", "", "| Dataset | Group | " + " | ".join(S[m] for m in ALL) + " |",
                  "|---|---|" + "---:|" * len(ALL)])
        for ds, r in red.iterrows():
            top = r[c2.METHODS].max()
            cells = [f"**{r[m]:.3f}**" if m in c2.METHODS and np.isclose(r[m], top) else f"{r[m]:.3f}" for m in ALL]
            L.append(f"| {ds} | {groups[ds]} | " + " | ".join(cells) + " |")
        means = red[ALL].mean()
        top = means[c2.METHODS].max()
        L.append("| **Mean** | | " + " | ".join(f"**{means[m]:.3f}**" if m in c2.METHODS and np.isclose(means[m], top)
                                               else f"{means[m]:.3f}" for m in ALL) + " |")
        L.append("")
        return red

    red10 = table(0.10, "rel_outbreak", "Outbreak reduction with 10% vaccinated")
    table(0.05, "rel_outbreak", "Outbreak reduction with 5% vaccinated")
    table(0.10, "rel_peak", "Peak reduction with 10% vaccinated")

    L += ["## Mean outbreak reduction vs. share vaccinated (mean over networks)", "",
          "| % vaccinated | " + " | ".join(S[m] for m in ALL) + " |", "|---:|" + "---:|" * len(ALL)]
    for f in FRACTIONS[1:]:
        red = 1 - piv(at(f), "rel_outbreak")
        L.append(f"| {f:.1%} | " + " | ".join(f"{red[m].mean():.3f}" for m in ALL) + " |")
    L.append("")

    L += ["## By group (outbreak reduction, 10% vaccinated)", "",
          "| Group | Networks | " + " | ".join(S[m] for m in ALL) + " |", "|---|---:|" + "---:|" * len(ALL)]
    for g in dict.fromkeys(groups):
        sub = red10[groups.reindex(red10.index) == g]
        L.append(f"| {g} | {len(sub)} | " + " | ".join(f"{sub[m].mean():.3f}" for m in ALL) + " |")
    L.append("")

    comp = red10[c2.METHODS]
    wins = comp.apply(lambda r: r.index[np.isclose(r, r.max())].tolist(), axis=1)
    ours = comp["Hybrid G-L (ours)"]
    L += ["## Head-to-head (outbreak reduction, 10% vaccinated)", "",
          f"- Ours is the best (or tied) method on **{sum('Hybrid G-L (ours)' in w for w in wins)} / {len(comp)}** networks",
          *[f"- Ours beats {m}: {(ours > comp[m]).sum()} / {len(comp)}" for m in c2.METHODS[1:]],
          f"- Ours beats random vaccination: {(ours > red10['Random']).sum()} / {len(comp)}",
          f"- Ours reaches {ours.mean() / red10['True SIR ranking'].mean():.0%} of the reduction obtained by "
          "vaccinating the true top spreaders", ""]
    L += ["Figures: `fig_vaccination_curves.png`, `fig_vaccination_by_group.png`, `fig_reduction10_by_network.png`."]
    with open(os.path.join(c2.RESULTS, "vaccination_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


SHOW = ["Hybrid G-L (ours)", "Strength (W-Deg)", "W-Eigenvector", "W-Closeness", "Prev. Deep 6-Ch",
        "Random", "True SIR ranking"]
COLORS = ["#2a6fdb", "#43a047", "#7a7a7a", "#b04ad6", "#e08a00", "#c62828", "#000000"]
STYLE = ["-", "-", "-", "-", "-", ":", "--"]


def make_figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, lab in zip(axes, ["rel_outbreak", "rel_peak"], ["Final outbreak size", "Peak size"]):
        for m, c, ls in zip(SHOW, COLORS, STYLE):
            mean = df[df.method == m].groupby("frac")[col].mean()
            ax.plot(mean.index * 100, mean.values, ls, color=c, marker="o", ms=3.5, label=c2.SHORT[m])
        ax.set_xlabel("% of people vaccinated")
        ax.set_ylabel(f"{lab} (relative to no vaccination)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=7, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Targeted vaccination: mean over 26 contact networks")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(c2.RESULTS, "fig_vaccination_curves.png"), dpi=150)
    plt.close(fig)

    groups = list(dict.fromkeys(df.group))
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.8), sharey=True)
    for ax, g in zip(axes, groups):
        sub = df[df.group == g]
        for m, c, ls in zip(SHOW, COLORS, STYLE):
            mean = sub[sub.method == m].groupby("frac")["rel_outbreak"].mean()
            ax.plot(mean.index * 100, mean.values, ls, color=c, marker="o", ms=3, label=c2.SHORT[m])
        ax.set_title(f"{g} ({sub.dataset.nunique()})")
        ax.set_xlabel("% vaccinated")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Outbreak size (relative)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=7, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Targeted vaccination by network group")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(c2.RESULTS, "fig_vaccination_by_group.png"), dpi=150)
    plt.close(fig)

    d10 = df[np.isclose(df.frac, 0.10)]
    red = 1 - d10.pivot(index="dataset", columns="method", values="rel_outbreak").reindex(list(dict.fromkeys(df.dataset)))
    fig, ax = plt.subplots(figsize=(15, 5.5))
    x = np.arange(len(red))
    bars = SHOW[:5]
    w = 0.16
    for k, m in enumerate(bars):
        ax.bar(x + (k - 2) * w, red[m], w, color=COLORS[k], label=c2.SHORT[m])
    ax.plot(x, red["True SIR ranking"], "k_", ms=14, mew=2, label="True SIR ranking")
    ax.plot(x, red["Random"], "_", color="#c62828", ms=14, mew=2, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(red.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Outbreak reduction, 10% vaccinated")
    ax.set_title("Outbreak reduction when the top 10% of each ranking is vaccinated", pad=28)
    ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.09), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(c2.RESULTS, "fig_reduction10_by_network.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()
    names = args.only or cc.DATASETS
    rows = [r for n in names for r in run_network(n, args.runs)]
    df = summarise(pd.DataFrame(rows))
    if args.only is None:
        df.to_csv(os.path.join(c2.RESULTS, "vaccination_results.csv"), index=False)
        write_report(df, args.runs)
        make_figures(df)
    d10 = df[np.isclose(df.frac, 0.10)]
    print((1 - d10.pivot(index="dataset", columns="method", values="rel_outbreak"))[ALL]
          .rename(columns=c2.SHORT).round(3).to_string())
    print("MEAN reduction @10%:", (1 - d10.groupby("method").rel_outbreak.mean())[ALL].round(3).to_dict())


if __name__ == "__main__":
    main()
