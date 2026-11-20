"""
Block 2 / Nov 21 + Nov 22: stability tests. Does a ranking computed from one view of a population still
find the super-spreaders when the contacts are measured differently, or on another day?

  (a) face-to-face vs co-location: the same people, two contact definitions (6 SocioPatterns sites)
  (b) Utah schools: contact within 2 m vs within 1 m (2 schools)
  (c) day 1 -> day 2: rank people from day-1 contacts, score against SIR spreading on day 2 (12 sites),
      and vaccinate the day-1 top 5% / 10% before the day-2 outbreak

For every pair (source view A -> target view B) and every method:
  tau20 / overlap20 / tau_all of the method's scores on A against the SIR labels on B (common people only)
  agreement = Kendall tau between the method's own scores on A and on B (how stable the ranking itself is)
Reference row "True SIR on A": the real spreading ranking from view A used to predict view B. It shows how
much the spreading itself changes between views, i.e. the best any view-A ranking can be expected to do.

Usage: python run_stability.py [--tests a b c] [--runs 200]
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import common2 as c2
from common2 import cc, hc

REF = "True SIR on A"
ALL = c2.METHODS + [REF]

F2F_COLOC = ["LH10_Hospital", "InVS13_Workplace", "InVS15_Workplace", "SFHH_Conference",
             "LyonSchool_Primary", "Thiers13_HighSchool"]
UTAH = ["Utah_MiddleSchool", "Utah_ElementarySchool"]
DAYS = ["LH10_Hospital", "LyonSchool_Primary", "Thiers11_HighSchool", "Thiers12_HighSchool", "Thiers13_HighSchool",
        "InVS13_Workplace", "InVS15_Workplace", "SFHH_Conference", "HT09_Conference", "Malawi_Village",
        "Utah_MiddleSchool", "Utah_ElementarySchool"]


def person_ids(name):
    """Processed node label -> person id (Netzschleuder labels are re-indexed per network). Duplicated ids dropped."""
    f = os.path.join(cc.PROCESSED, f"{name}_nodes.csv")
    meta = pd.read_csv(f)
    if "index" not in meta.columns:            # Utah: labels already are the person ids
        return None
    dup = meta["id"].duplicated(keep=False)
    return dict(zip(meta.loc[~dup, "index"], meta.loc[~dup, "id"]))


def cross_eval(A, B, idA=None, idB=None):
    """Scores on A -> SIR labels on B, over the people present in both LCCs."""
    keyA = {(idA[u] if idA else u): i for i, u in enumerate(A.nodes) if idA is None or u in idA}
    keyB = {(idB[u] if idB else u): i for i, u in enumerate(B.nodes) if idB is None or u in idB}
    common = sorted(set(keyA) & set(keyB))
    ia = np.array([keyA[p] for p in common])
    ib = np.array([keyB[p] for p in common])
    yB = B.y[ib]
    top = hc.get_subset_indices(yB, 20)
    out = {"common": len(common), "nA": A.n, "nB": B.n}
    for m in ALL:
        sA = A.y[ia] if m == REF else A.scores[m][ia]
        out[f"tau20|{m}"] = hc.safe_tau(sA, yB, top)
        out[f"overlap20|{m}"] = hc.top_overlap(sA, yB, 20)
        out[f"tau_all|{m}"] = kendalltau(sA, yB)[0]
        sB = B.y[ib] if m == REF else B.scores[m][ib]
        out[f"agree|{m}"] = kendalltau(sA, sB)[0]
    return out, ia, ib


def test_a():
    rows = []
    for name in F2F_COLOC:
        f2f, col = c2.block1_net(name), c2.block1_net(f"{name}_Coloc")
        i1, i2 = person_ids(name), person_ids(f"{name}_Coloc")
        for (A, B, ia, ib, d) in [(f2f, col, i1, i2, "face-to-face -> co-location"),
                                  (col, f2f, i2, i1, "co-location -> face-to-face")]:
            r, _, _ = cross_eval(A, B, ia, ib)
            rows.append(dict(test="a", site=name, direction=d, **r))
    return rows


def test_b():
    rows = []
    for name in UTAH:
        m2, m1 = c2.block1_net(name), c2.block1_net(f"{name}_1m")
        for (A, B, d) in [(m2, m1, "2 m -> 1 m"), (m1, m2, "1 m -> 2 m")]:
            r, _, _ = cross_eval(A, B)
            rows.append(dict(test="b", site=name, direction=d, **r))
    return rows


def day_nets(name):
    return [c2.path_net(f"{name}_day{d}", os.path.join(c2.DATA, f"{name}_day{d}.txt")) for d in (1, 2)]


def test_c(runs):
    rows, vacc = [], []
    for name in DAYS:
        A, B = day_nets(name)
        r, ia, ib = cross_eval(A, B)
        rows.append(dict(test="c", site=name, direction="day 1 -> day 2", **r))

        # vaccinate on day 2 using day-1 rankings; people not seen on day 1 cannot be chosen
        def order_from(scoreA):
            s = np.full(B.n, -np.inf)
            s[ib] = scoreA[ia]
            return c2.ranking(s)
        orders = {m: order_from(A.scores[m]) for m in c2.METHODS}
        orders["Random"] = None
        orders[REF] = order_from(A.y)
        orders["True SIR on day 2"] = c2.ranking(B.y)
        for x in c2.vaccination(B, orders, [0.0, 0.05, 0.10], runs=runs):
            vacc.append(dict(site=name, n_day2=B.n, seen_day1=len(ib), **x))
        print(f"  day test {name:24s} common={len(ib):4d}/{B.n:4d}  tau20 ours={r['tau20|Hybrid G-L (ours)']:.3f} "
              f"ref={r[f'tau20|{REF}']:.3f}", flush=True)
    return rows, vacc


def fmt_table(df, metric, L, label_cols):
    S = dict(c2.SHORT, **{REF: "SIR on A (ref.)"})
    L += ["| " + " | ".join(label_cols) + " | common | " + " | ".join(S[m] for m in ALL) + " |",
          "|" + "---|" * len(label_cols) + "---:|" + "---:|" * len(ALL)]
    for _, r in df.iterrows():
        top = max(r[f"{metric}|{m}"] for m in c2.METHODS)
        cells = [f"**{r[f'{metric}|{m}']:.3f}**" if m in c2.METHODS and np.isclose(r[f"{metric}|{m}"], top)
                 else f"{r[f'{metric}|{m}']:.3f}" for m in ALL]
        L.append("| " + " | ".join(str(r[c]) for c in label_cols) + f" | {r['common']} | " + " | ".join(cells) + " |")
    means = {m: df[f"{metric}|{m}"].mean() for m in ALL}
    top = max(means[m] for m in c2.METHODS)
    L.append("| **Mean** |" + " |" * (len(label_cols) - 1) + " | " +
             " | ".join(f"**{means[m]:.3f}**" if m in c2.METHODS and np.isclose(means[m], top) else f"{means[m]:.3f}"
                        for m in ALL) + " |")
    L.append("")


TITLES = {"a": "(a) Face-to-face vs co-location (same people, different contact definition)",
          "b": "(b) Utah schools: contacts within 2 m vs within 1 m",
          "c": "(c) Day 1 -> day 2: rank on day 1, score against spreading on day 2"}


def write_report(df, vacc):
    L = ["# Block 2: Stability Tests", "",
         "A ranking is computed from one view of a population (A) and checked against the SIR spreading on another "
         "view (B), over the people present in both. **SIR on A (ref.)** uses the true spreading ranking of view A: "
         "it shows how much the spreading itself changes between the two views. Calibrated SIR regime (10% mean "
         "outbreak) in every view. Best competing method in **bold**.", ""]
    for t in ["a", "b", "c"]:
        sub = df[df.test == t]
        if not len(sub):
            continue
        L += [f"## {TITLES[t]}", ""]
        for metric, desc in [("tau20", "Kendall tau inside the true top 20% of view B"),
                             ("overlap20", "Top-20% overlap with view B"),
                             ("agree", "Agreement: Kendall tau between each method's own scores on A and on B")]:
            L += [f"### {desc}", ""]
            fmt_table(sub, metric, L, ["site", "direction"])
    if vacc is not None and len(vacc):
        S = dict(c2.SHORT, **{REF: "SIR day 1 (ref.)", "True SIR on day 2": "SIR day 2 (oracle)"})
        meths = c2.METHODS + ["Random", REF, "True SIR on day 2"]
        base = vacc[vacc.frac == 0].groupby("site").outbreak.first()
        vacc = vacc.assign(red=1 - vacc.outbreak / vacc.site.map(base))
        for f in (0.05, 0.10):
            d = vacc[np.isclose(vacc.frac, f)].pivot(index="site", columns="method", values="red").reindex(DAYS)
            L += [f"## (c) Day-2 outbreak reduction when the day-1 top {f:.0%} are vaccinated", "",
                  "*SIR day 2 (oracle)* knows the day-2 contacts in advance, so it is an upper reference only.", "",
                  "| Site | " + " | ".join(S[m] for m in meths) + " |", "|---|" + "---:|" * len(meths)]
            for s, r in d.iterrows():
                top = r[c2.METHODS].max()
                L.append(f"| {s} | " + " | ".join(f"**{r[m]:.3f}**" if m in c2.METHODS and np.isclose(r[m], top)
                                                  else f"{r[m]:.3f}" for m in meths) + " |")
            mean = d[meths].mean()
            top = mean[c2.METHODS].max()
            L.append("| **Mean** | " + " | ".join(f"**{mean[m]:.3f}**" if m in c2.METHODS and np.isclose(mean[m], top)
                                                 else f"{mean[m]:.3f}" for m in meths) + " |")
            L.append("")
    L += ["Figure: `fig_stability.png`."]
    with open(os.path.join(c2.RESULTS, "stability_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["Hybrid G-L (ours)", "Strength (W-Deg)", "W-Eigenvector", "W-Closeness", "Prev. Deep 6-Ch", REF]
    colors = ["#2a6fdb", "#43a047", "#7a7a7a", "#b04ad6", "#e08a00", "#000000"]
    labels = dict(c2.SHORT, **{REF: "SIR on A (ref.)"})
    tests = [t for t in ["a", "b", "c"] if t in set(df.test)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = {"a": "F2F <-> co-location", "b": "2 m <-> 1 m", "c": "day 1 -> day 2"}
    w = 0.13
    for ax, metric, lab in zip(axes, ["tau20", "overlap20"], ["Kendall tau (top 20% of B)", "Top-20% overlap with B"]):
        x = np.arange(len(tests))
        for k, (m, c) in enumerate(zip(show, colors)):
            ax.bar(x + (k - 2.5) * w, [df[df.test == t][f"{metric}|{m}"].mean() for t in tests], w,
                   color=c, label=labels[m], hatch="//" if m == REF else None, edgecolor="white" if m == REF else None)
        ax.set_xticks(x)
        ax.set_xticklabels([names[t] for t in tests])
        ax.set_ylabel(lab)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Stability: rank on view A, evaluate against SIR spreading on view B")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(c2.RESULTS, "fig_stability.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="*", default=["a", "b", "c"])
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()
    rows, vacc = [], None
    if "a" in args.tests:
        rows += test_a()
    if "b" in args.tests:
        rows += test_b()
    if "c" in args.tests:
        r, v = test_c(args.runs)
        rows += r
        vacc = pd.DataFrame(v)
        vacc.to_csv(os.path.join(c2.RESULTS, "day2_vaccination_results.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(c2.RESULTS, "stability_results.csv"), index=False)
    write_report(df, vacc)
    make_figure(df)
    for t, sub in df.groupby("test"):
        print(f"test {t}: mean tau20", {c2.SHORT.get(m, m): round(sub[f'tau20|{m}'].mean(), 3) for m in ALL})


if __name__ == "__main__":
    main()
