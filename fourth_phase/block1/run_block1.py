"""
Fourth Phase / Block 1: SIR ground truth + zero-shot evaluation on human contact networks.

Steps per network (all cached in block1/cache/):
  1. LCC, SIR labels (Third_Phase recipe, fast live-edge estimator, 500 runs)
  2. Local 6-channel neighbourhood patches   (Third_Phase process_dataset)
  3. Global descriptors + SIGN aggregates    (Third_Phase hybrid_common)
  4. Classical weighted centralities

Then scores, with NO retraining:
  Hybrid G-L ensemble (3 seeds), previous Deep 6-Ch model, W-Eigenvector, Strength, W-Closeness,
  W-Betweenness.
Metrics (same as Third_Phase): Kendall tau inside the true top 20%, top-20% overlap; plus Kendall tau
over all nodes.

Outputs: block1/results/zero_shot_results.csv, network_sir_stats.csv, zero_shot_report.md, figures.
Usage:   python run_block1.py [--only NAME ...]
"""
import os
import sys
import time
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

import contact_common as cc
from contact_common import tw, hc

METHODS = ["Hybrid G-L (ours)", "Prev. Deep 6-Ch", "W-Eigenvector", "Strength (W-Deg)",
           "W-Closeness", "W-Betweenness"]


def cpath(name, kind):
    return os.path.join(cc.CACHE, f"{name}_{kind}.npy")


def prepare(name, regime):
    """Compute (or load) labels, local X, global features, baselines for one network."""
    t0 = time.time()
    sfx = "" if regime == "original" else "_cal"
    G, nodes, n_raw = cc.load_lcc(name)
    stats = {"dataset": name, "group": cc.GROUP_OF[name], "nodes_total": n_raw,
             "lcc_nodes": len(nodes), "lcc_edges": G.number_of_edges()}

    if not os.path.exists(cpath(name, "y" + sfx)):
        if regime == "original":
            P, beta, lam = cc.sir_setup(G, nodes)
        else:
            _, _, lam = cc.sir_setup(G, nodes)
            tau = cc.calibrate_tau(cc.norm_W(G, nodes))
            P, beta = cc.P_from_tau(cc.norm_W(G, nodes), tau), -np.expm1(-tau)
        y, y_raw = cc.sir_labels(P, return_raw=True)
        np.save(cpath(name, "y" + sfx), y)
        np.save(cpath(name, "y_raw" + sfx), y_raw)
        np.save(cpath(name, "sir_params" + sfx), np.array([beta, lam] + ([] if regime == "original" else [tau])))
    params = np.load(cpath(name, "sir_params" + sfx))
    beta, lam = params[0], params[1]
    y_raw = np.load(cpath(name, "y_raw" + sfx))
    y = np.load(cpath(name, "y" + sfx))
    if regime != "original":
        stats["tau"] = params[2]
    stats.update(beta=beta, lambda_max=lam, mean_outbreak_frac=float(y_raw.mean() / len(nodes)),
                 max_outbreak_frac=float(y_raw.max() / len(nodes)))

    if not os.path.exists(cpath(name, "X_local")):
        X, _ = tw.process_dataset(cc.path_of(name), name, cc.SEMANTICS, precalculated_y=y)
        np.save(cpath(name, "X_local"), X.astype(np.float32))

    if not (os.path.exists(cpath(name, "global_feats" + sfx)) and os.path.exists(cpath(name, "global_raw" + sfx))):
        if regime == "original":
            raw, W, _ = hc.compute_raw_features(G, nodes)
        else:
            raw, W = cc.compute_raw_features_tau(G, nodes, stats["tau"])
        np.save(cpath(name, "global_feats" + sfx), hc.transform_features(raw, W))
        np.save(cpath(name, "global_raw" + sfx), raw)

    if not os.path.exists(cpath(name, "baselines")):
        b = cc.baselines(G, nodes)
        np.save(cpath(name, "baselines"), np.stack([b[m] for m in METHODS[2:6]], axis=1))

    print(f"  prepared {name:26s} n={len(nodes):5d}  beta={beta:.3f}  mean outbreak="
          f"{stats['mean_outbreak_frac']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return stats


def load_models():
    spec = importlib.util.spec_from_file_location(
        "hybrid_eval", os.path.join(cc.THIRD, "weighted_model_hybrid_global_local", "evaluate_and_report.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    models, _ = ev.load_ensemble(torch.device("cpu"))
    return models, ev.load_previous_deep_model()


def score_network(name, models, prev, regime):
    sfx = "" if regime == "original" else "_cal"
    X = np.load(cpath(name, "X_local")).astype(np.float32)
    Gf = np.load(cpath(name, "global_feats" + sfx)).astype(np.float32)
    y = np.load(cpath(name, "y" + sfx))
    base = np.load(cpath(name, "baselines"))

    pred = np.mean([hc.zscore(hc.predict(m, X, Gf, torch.device("cpu"))) for m in models], axis=0)
    with torch.no_grad():
        prev_s = np.concatenate([prev(torch.from_numpy(X[i:i + 1024])).numpy().ravel()
                                 for i in range(0, len(X), 1024)]) if prev is not None else np.full(len(y), np.nan)
    scores = [pred, prev_s] + [base[:, j] for j in range(4)]
    np.save(cpath(name, "pred_hybrid" + sfx), pred)

    top = hc.get_subset_indices(y, 20)
    row = {"dataset": name, "group": cc.GROUP_OF[name], "nodes": len(y)}
    for m, s in zip(METHODS, scores):
        row[f"tau20|{m}"] = hc.safe_tau(s, y, top)
        row[f"overlap20|{m}"] = hc.top_overlap(s, y, 20)
        row[f"tau_all|{m}"] = kendalltau(s, y)[0]
    return row


def best(row, metric):
    vals = {m: row[f"{metric}|{m}"] for m in METHODS}
    top = np.nanmax(list(vals.values()))
    return [m for m, v in vals.items() if np.isclose(v, top, atol=1e-9)]


def write_report(df, stats, regime):
    short = {"Hybrid G-L (ours)": "Ours", "Prev. Deep 6-Ch": "Deep6", "W-Eigenvector": "W-Eig",
             "Strength (W-Deg)": "Strength", "W-Closeness": "W-Clos", "W-Betweenness": "W-Betw"}
    gt = ("Third-Phase recipe: beta = min(1.5 / lambda_max, 0.9), p = 1 - (1 - beta)^w_norm, 500 runs"
          if regime == "original" else
          f"calibrated: p = 1 - exp(-tau * w_norm) with tau chosen per network so that the mean outbreak is "
          f"{cc.TARGET_OUTBREAK:.0%} of the population, 500 runs; the model's beta-dependent global descriptors use the same tau")
    L = [f"# Block 1: Zero-Shot Results on Human Contact Networks ({regime} SIR regime)", "",
         "The Third-Phase Hybrid Global-Local WNLGCN (3-seed ensemble) was applied to the contact networks "
         f"**without any retraining**. Ground truth = expected SIR outbreak size from each node ({gt}).", ""]

    for metric, title, desc in [
            ("tau20", "Kendall tau inside the true top 20%", "How well each method orders the real top spreaders."),
            ("overlap20", "Top-20% overlap", "Share of the real top-20% spreaders that each method also puts in its top 20%."),
            ("tau_all", "Kendall tau over all nodes", "Agreement with the SIR ranking over the whole network.")]:
        L += [f"## {title}", "", desc + " Best value per row in **bold**.", "",
              "| Dataset | Group | n | " + " | ".join(short[m] for m in METHODS) + " |",
              "|---|---|---:|" + "---:|" * len(METHODS)]
        for _, r in df.iterrows():
            b = best(r, metric)
            cells = [(f"**{r[f'{metric}|{m}']:.3f}**" if m in b else f"{r[f'{metric}|{m}']:.3f}") for m in METHODS]
            L.append(f"| {r['dataset']} | {r['group']} | {r['nodes']} | " + " | ".join(cells) + " |")
        means = [df[f"{metric}|{m}"].mean() for m in METHODS]
        L.append("| **Mean** | | | " + " | ".join(f"**{v:.3f}**" if np.isclose(v, max(means)) else f"{v:.3f}"
                                                  for v in means) + " |")
        L.append("")

    L += ["## Summary by group (mean Kendall tau, top 20%)", "",
          "| Group | Networks | " + " | ".join(short[m] for m in METHODS) + " |",
          "|---|---:|" + "---:|" * len(METHODS)]
    for g, sub in df.groupby("group", sort=False):
        L.append(f"| {g} | {len(sub)} | " + " | ".join(f"{sub[f'tau20|{m}'].mean():.3f}" for m in METHODS) + " |")
    L.append("")

    wins = {m: sum(m in best(r, "tau20") for _, r in df.iterrows()) for m in METHODS}
    ours = df["tau20|Hybrid G-L (ours)"]
    L += ["## Head-to-head (Kendall tau, top 20%)", "",
          f"- Networks where ours is best or tied: **{wins['Hybrid G-L (ours)']} / {len(df)}**",
          *[f"- Ours beats {m}: {(ours > df[f'tau20|{m}']).sum()} / {len(df)} networks" for m in METHODS[1:]], ""]

    weak = df[ours < df[[f"tau20|{m}" for m in METHODS[1:]]].max(axis=1) - 0.02]
    L += ["## Where the model is weak (ours more than 0.02 below the best other method)", ""]
    if len(weak):
        L += ["| Dataset | Ours | Best other | Method | Mean outbreak (fraction) | beta |", "|---|---:|---:|---|---:|---:|"]
        st = stats.set_index("dataset")
        for _, r in weak.iterrows():
            others = {m: r[f"tau20|{m}"] for m in METHODS[1:]}
            m_best = max(others, key=others.get)
            L.append(f"| {r['dataset']} | {r['tau20|Hybrid G-L (ours)']:.3f} | {others[m_best]:.3f} | {m_best} | "
                     f"{st.loc[r['dataset'], 'mean_outbreak_frac']:.2f} | {st.loc[r['dataset'], 'beta']:.3f} |")
    else:
        L.append("None.")
    L += ["", "## SIR ground-truth settings per network", "",
          "| Dataset | LCC nodes / total | LCC edges | lambda_max | beta | Mean outbreak (fraction) | Max outbreak (fraction) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for _, s in stats.iterrows():
        L.append(f"| {s['dataset']} | {s['lcc_nodes']} / {s['nodes_total']} | {s['lcc_edges']} | {s['lambda_max']:.2f} | "
                 f"{s['beta']:.3f} | {s['mean_outbreak_frac']:.2f} | {s['max_outbreak_frac']:.2f} |")
    L += ["", f"Figures: `fig_tau20_by_network_{regime}.png`, `fig_mean_by_group_{regime}.png`."]
    with open(os.path.join(cc.RESULTS, f"zero_shot_report_{regime}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def make_figures(df, regime):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["Hybrid G-L (ours)", "Strength (W-Deg)", "W-Eigenvector", "W-Closeness", "Prev. Deep 6-Ch"]
    colors = ["#2a6fdb", "#43a047", "#7a7a7a", "#b04ad6", "#e08a00"]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    x = np.arange(len(df))
    w = 0.16
    for k, (m, c) in enumerate(zip(show, colors)):
        ax.bar(x + (k - 2) * w, df[f"tau20|{m}"], w, label=m, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Kendall tau (true top 20%)")
    ax.set_title(f"Zero-shot: ranking quality on human contact networks ({regime} SIR regime)", pad=28)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.09), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(cc.RESULTS, f"fig_tau20_by_network_{regime}.png"), dpi=150)
    plt.close(fig)

    groups = list(dict.fromkeys(df["group"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, lab in zip(axes, ["tau20", "overlap20"], ["Kendall tau (top 20%)", "Top-20% overlap"]):
        xg = np.arange(len(groups))
        for k, (m, c) in enumerate(zip(show, colors)):
            ax.bar(xg + (k - 2) * w, [df[df.group == g][f"{metric}|{m}"].mean() for g in groups], w, label=m, color=c)
        ax.set_xticks(xg)
        ax.set_xticklabels(groups)
        ax.set_ylabel(lab)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(f"Zero-shot: mean by network group ({regime} SIR regime)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(cc.RESULTS, f"fig_mean_by_group_{regime}.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--regime", choices=["original", "calibrated"], default="original")
    args = ap.parse_args()
    names = args.only or cc.DATASETS

    print("Step 1: SIR labels and features", flush=True)
    stats = pd.DataFrame([prepare(n, args.regime) for n in names])

    print("Step 2: zero-shot scoring", flush=True)
    models, prev = load_models()
    df = pd.DataFrame([score_network(n, models, prev, args.regime) for n in names])

    if args.only is None:
        df.to_csv(os.path.join(cc.RESULTS, f"zero_shot_results_{args.regime}.csv"), index=False)
        stats.to_csv(os.path.join(cc.RESULTS, f"network_sir_stats_{args.regime}.csv"), index=False)
        write_report(df, stats, args.regime)
        make_figures(df, args.regime)
    cols = ["dataset"] + [f"tau20|{m}" for m in METHODS]
    print(df[cols].rename(columns=lambda c: c.replace("tau20|", "")).round(3).to_string(index=False))
    print("MEAN tau20:", {m: round(df[f'tau20|{m}'].mean(), 3) for m in METHODS})


if __name__ == "__main__":
    main()
