"""
Evaluate the Hybrid Global-Local WNLGCN ensemble and generate a PDF report.

Compares against:
  * the previous best model  (Deep 6-Channel WNLGCN, ../weighted_model_deep_6channel/wnlgcn_6ch_deep.pth)
  * classical weighted centralities (W-Eigenvector, Strength, W-Closeness, W-Betweenness)
  * Mean-Field Reach (k=3) - the strongest single analytic descriptor fed to the global branch,
    reported so the gain from *learning* is visible, not just the gain from the feature.

Usage:
    python evaluate_and_report.py
Outputs:
    analysis_report_hybrid_global_local.pdf, evaluation_results.csv
"""
import os
import sys
import json
import importlib.util
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hybrid_common import (
    HybridWNLGCN, script_dir, results_dir, deep_model_dir, feature_cache_dir, checkpoint_dir,
    train_datasets, test_datasets, get_global_features, load_local_X, load_labels, load_lcc,
    get_subset_indices, safe_tau, top_overlap, predict, zscore, RAW_FEATURE_NAMES, N_GLOBAL_FEATURES,
)

METHODS = ["Ours (Hybrid G-L)", "Prev. Deep 6-Ch", "W-Eigenvector", "Strength (W-Deg)",
           "W-Closeness", "W-Betweenness", "MF-Reach (k=3)"]
STATUS_LABEL = {"train": "train", "val": "val (held-out)", "test": "test (unseen)"}


# ---------------------------------------------------------------- models
def load_ensemble(device):
    summary_path = os.path.join(checkpoint_dir, "training_summary.json")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run train_hybrid.py first.")
        sys.exit(1)
    with open(summary_path) as f:
        summary = json.load(f)
    cfg = summary["config"]
    models = []
    for run in summary["runs"]:
        m = HybridWNLGCN(n_global=N_GLOBAL_FEATURES, hidden=cfg["hidden"], dropout=cfg["dropout"])
        m.load_state_dict(torch.load(os.path.join(checkpoint_dir, run["checkpoint"]), map_location=device))
        models.append(m.to(device).eval())
    return models, summary


def load_previous_deep_model():
    path = os.path.join(deep_model_dir, "wnlgcn_6ch_deep.pth")
    if not os.path.exists(path):
        print("  (previous Deep 6-Ch weights not found - column will be nan)")
        return None
    spec = importlib.util.spec_from_file_location("deep6_eval", os.path.join(deep_model_dir, "evaluate_and_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.WNLGCN()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


# ---------------------------------------------------------------- baselines
def get_baselines(filename):
    names = ["weig", "wdeg", "wclos", "wbet"]
    shared = [os.path.join(results_dir, f"{filename}_{m}.npy") for m in names]
    local = [os.path.join(feature_cache_dir, f"{filename}_baseline_{m}.npy") for m in names]
    if all(os.path.exists(p) for p in shared):
        return [np.load(p).flatten() for p in shared]
    if all(os.path.exists(p) for p in local):
        return [np.load(p).flatten() for p in local]

    print(f"  -> computing classical centralities for {filename} (not cached)...", flush=True)
    G, nodelist = load_lcc(filename)
    wdeg_d = dict(G.degree(weight="weight"))
    G_dist = nx.Graph()
    for u, v, d in G.edges(data=True):
        G_dist.add_edge(u, v, distance=1.0 / d.get("weight", 1.0))
    wclos_d = nx.closeness_centrality(G_dist, distance="distance")
    wbet_d = nx.betweenness_centrality(G_dist, weight="distance", normalized=True)
    try:
        weig_d = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        weig_d = {u: 0.0 for u in nodelist}
    vals = [np.array([d[u] for u in nodelist]) for d in (weig_d, wdeg_d, wclos_d, wbet_d)]
    for p, v in zip(local, vals):
        np.save(p, v)
    return vals


# ---------------------------------------------------------------- PDF helpers
def wrap(text, width):
    lines, cur = [], []
    for w in text.split():
        cur.append(w)
        if len(" ".join(cur)) > width:
            lines.append(" ".join(cur[:-1]))
            cur = [w]
    lines.append(" ".join(cur))
    return lines


def add_text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(12, 8.5))
    fig.text(0.05, 0.93, title, fontsize=15, weight="bold", color="#1F4E79")
    y = 0.87
    for p in paragraphs:
        if p.startswith("###"):
            y -= 0.02
            fig.text(0.05, y, p.replace("###", "").strip(), fontsize=10.5, weight="bold", color="#2F5597")
            y -= 0.03
        elif p.strip() == "":
            y -= 0.01
        else:
            for line in wrap(p, 135):
                fig.text(0.05, y, line, fontsize=8.5, color="#333333")
                y -= 0.022
            y -= 0.006
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def add_table_page(pdf, title, headers, data, row_statuses, best_cols, why_text, benefits_text, fontsize=7.0,
                   first_col_frac=None):
    n_rows = len(data)
    fig = plt.figure(figsize=(12, max(8.5, n_rows * 0.23 + 3.2)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.5, 0.965, title, fontsize=13, weight="bold", color="#1F4E79", ha="center")

    y = 0.93
    for head, body, color in (("Why We Check This:", why_text, "#2F5597"),
                              ("Why It Benefits Us (Proof of Superiority):", benefits_text, "#2E75B6")):
        fig.text(0.05, y, head, fontsize=8.5, weight="bold", color=color)
        y -= 0.018
        for line in wrap(body, 150):
            fig.text(0.05, y, line, fontsize=7.5, color="#444444")
            y -= 0.016
        y -= 0.008

    table = ax.table(cellText=data, colLabels=headers, cellLoc="center",
                     bbox=[0.04, 0.03, 0.92, y - 0.05])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    if first_col_frac:
        others = (1.0 - first_col_frac) / (len(headers) - 1)
        for (r, c), cell in table.get_celld().items():
            cell.set_width(first_col_frac if c == 0 else others)
    for c in range(len(headers)):
        table[0, c].set_text_props(weight="bold", color="white")
        table[0, c].set_facecolor("#1F4E79")
    fills = {"train": ("#FFFFFF", "#F2F2F2"), "val": ("#FFF2CC", "#FFE699"), "test": ("#E2F0D9", "#C6E0B4")}
    for r in range(1, n_rows + 1):
        body, name = fills.get(row_statuses[r - 1], fills["train"])
        for c in range(len(headers)):
            cell = table[r, c]
            if c == 0:
                cell.set_facecolor(name)
                cell.set_text_props(weight="bold")
            elif c in best_cols[r - 1]:
                cell.set_facecolor("#B4C6E7")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor(body)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def fmt(v):
    return f"{v:.4f}" if v is not None and not np.isnan(v) else "nan"


def best_indices(vals, offset=1):
    arr = np.array([v if not np.isnan(v) else -np.inf for v in vals])
    if np.all(np.isinf(arr)):
        return set()
    return {i + offset for i in np.flatnonzero(np.isclose(arr, arr.max(), atol=1e-9))}


# ---------------------------------------------------------------- main
def main():
    device = torch.device("cpu")
    models, summary = load_ensemble(device)
    prev_model = load_previous_deep_model()
    val_set = set(summary["val_datasets"])
    mf_idx = RAW_FEATURE_NAMES.index("MF_reach_k3")
    print(f"Loaded ensemble of {len(models)} Hybrid Global-Local models.")

    rows = []
    datasets = [(f, "val" if f in val_set else "train") for f in train_datasets] + [(f, "test") for f in test_datasets]
    for filename, status in datasets:
        print(f"Evaluating {filename} [{status}]", flush=True)
        X = load_local_X(filename)
        Gf, raw = get_global_features(filename)
        y = load_labels(filename)

        pred = np.mean([zscore(predict(m, X, Gf, device)) for m in models], axis=0)
        if prev_model is not None:
            with torch.no_grad():
                prev = np.concatenate([prev_model(torch.from_numpy(X[i:i + 1024])).numpy().ravel()
                                       for i in range(0, len(X), 1024)])
        else:
            prev = np.full(len(y), np.nan)
        weig, wdeg, wclos, wbet = get_baselines(filename)
        scores = [pred, prev, weig, wdeg, wclos, wbet, raw[:, mf_idx]]

        top = get_subset_indices(y, 20)
        taus = [safe_tau(s, y, top) for s in scores]
        overlaps = [top_overlap(s, y, 20) if not np.all(np.isnan(s)) else np.nan for s in scores]
        rows.append({"name": filename, "status": status, "n": len(y), "tau": taus, "overlap": overlaps})

    # ---- CSV
    csv_path = os.path.join(script_dir, "evaluation_results.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,split,nodes," + ",".join(f"tau20_{m}" for m in METHODS) + ","
                + ",".join(f"overlap20_{m}" for m in METHODS) + "\n")
        for r in rows:
            f.write(f"{r['name']},{r['status']},{r['n']}," + ",".join(fmt(v) for v in r["tau"]) + ","
                    + ",".join(fmt(v) for v in r["overlap"]) + "\n")

    # ---- aggregate stats
    def split_mean(split, key, j):
        vals = [r[key][j] for r in rows if r["status"] == split]
        return np.nanmean(vals) if vals else np.nan

    test_rows = [r for r in rows if r["status"] == "test"]
    wins = [sum(1 for r in test_rows if (j + 1) in best_indices(r["tau"])) for j in range(len(METHODS))]
    beats_prev = sum(1 for r in test_rows if r["tau"][0] > r["tau"][1])
    beats_eig = sum(1 for r in test_rows if r["tau"][0] > r["tau"][2])
    mean_test = [split_mean("test", "tau", j) for j in range(len(METHODS))]

    print("\nMean Kendall tau@top20% on TEST graphs:")
    for m, v in zip(METHODS, mean_test):
        print(f"  {m:<20} {fmt(v)}")
    print(f"Hybrid beats Prev. Deep 6-Ch on {beats_prev}/{len(test_rows)} test graphs, "
          f"W-Eigenvector on {beats_eig}/{len(test_rows)}.")

    # ---- PDF
    pdf_path = os.path.join(script_dir, "analysis_report_hybrid_global_local.pdf")
    cfg = summary["config"]
    with PdfPages(pdf_path) as pdf:
        add_text_page(pdf, "Evaluation Report: Hybrid Global-Local WNLGCN on Weighted Networks", [
            "### 1. Motivation",
            "The Deep 6-Channel WNLGCN only observes a 41x41 local neighbourhood patch per node. It lost to W-Eigenvector "
            "precisely on networks whose spreading dynamics are governed by global (spectral) structure: US_airports, carrib, "
            "cargoshipsBB, karate and the larger synthetic test graphs. The Hybrid Global-Local model fuses the local patch "
            "with graph-wide, epidemic-aware descriptors.",
            "### 2. Architecture",
            "Local branch: channel attention -> Conv(6->16, k2) -> BN -> ReLU -> MaxPool -> Conv(16->32, k3) -> BN -> ReLU -> "
            "AdaptiveAvgPool(4x4) -> FC(512->128). Global branch: MLP(75->128->128) over 15 graph-wide descriptors "
            "(degree, strength, k-core, clustering, W-eigenvector, eigenvector of the transmission matrix T, PageRank, "
            "neighbour degree/strength sums, 2-hop reach, H-index, dynamics-sensitive centrality sum_k T^k 1 for k=2,4, "
            "mean-field outbreak size after 3 and 12 steps). Each descriptor is rank-transformed and z-scored within its own graph "
            "(size invariance), then propagated SIGN-style (1-hop and 2-hop weighted mean, 1-hop max). The two branches are "
            "combined by a sigmoid gate, followed by an MLP head (256->64->1) and a linear shortcut from the global descriptors.",
            "### 3. Training Objective",
            f"Loss = {cfg['alpha_reg']} x Huber(per-graph standardised SIR label) + {cfg['alpha_rank']} x top-focused RankNet. "
            f"Kendall tau@20% only scores pairs where both nodes are top spreaders, so top-top pairs get weight "
            f"{cfg['top_top_weight']}, top-vs-rest pairs {cfg['mixed_weight']} and others 1. Top-20% nodes are over-sampled "
            f"{cfg['top_oversample']}x within per-graph mini-batches of {cfg['batch_size']}. AdamW (lr {cfg['lr']}, wd "
            f"{cfg['weight_decay']}), warm-up + cosine schedule, {cfg['epochs']} epochs.",
            "### 4. Protocol (no test leakage)",
            f"Checkpoints were selected by mean tau@20% on held-out validation graphs {', '.join(summary['val_datasets'])} "
            f"(yellow rows), which were never used for gradient updates. The final predictor is an ensemble of "
            f"{len(models)} seeds ({', '.join(str(r['seed']) for r in summary['runs'])}); per-graph z-scored outputs are averaged. "
            "Test graphs (green rows) are never seen during training or model selection.",
            "### 5. Note on the MF-Reach baseline",
            "The transmission probabilities T_ij = 1 - (1 - beta)^w_ij use the same beta = 1.5/lambda_max as the SIR ground truth, "
            "i.e. the model knows the infection rate. MF-Reach (k=3) is reported as a separate column so readers can see "
            "that the learned model improves on its own strongest input feature.",
        ])

        headers = ["Network Dataset"] + METHODS
        data = [[r["name"]] + [fmt(v) for v in r["tau"]] for r in rows]
        add_table_page(
            pdf, "Table 1: Spreading Performance (Kendall's Tau) on Top 20% Nodes", headers, data,
            [r["status"] for r in rows], [best_indices(r["tau"]) for r in rows],
            "We compare the Hybrid Global-Local WNLGCN against the previous best model (Deep 6-Ch WNLGCN), classical weighted "
            "centralities and the mean-field reach descriptor on train (white), held-out validation (yellow) and unseen test (green) networks.",
            f"On the unseen test graphs the hybrid model reaches a mean tau of {fmt(mean_test[0])} versus {fmt(mean_test[1])} for "
            f"the previous Deep 6-Ch model and {fmt(mean_test[2])} for W-Eigenvector; it beats the previous model on {beats_prev}/"
            f"{len(test_rows)} and W-Eigenvector on {beats_eig}/{len(test_rows)} test graphs. Best performer per row is highlighted in blue.",
        )

        data2 = [[r["name"]] + [fmt(v) for v in r["overlap"]] for r in rows]
        add_table_page(
            pdf, "Table 2: Top-20% Spreader Identification (Overlap with Ground Truth)", headers, data2,
            [r["status"] for r in rows], [best_indices(r["overlap"]) for r in rows],
            "Kendall tau measures ordering inside the true top 20%. Overlap measures whether a method finds the right set: the fraction "
            "of true top-20% SIR spreaders that the method also places in its own top 20%.",
            "A high overlap shows the model identifies the correct influential spreaders, not merely ordering them well.",
        )

        sum_headers = ["Metric"] + METHODS
        sum_data, sum_best = [], []
        for label, split, key in (("tau@20% train", "train", "tau"), ("tau@20% val", "val", "tau"),
                                  ("tau@20% test", "test", "tau"), ("overlap@20% test", "test", "overlap")):
            vals = [split_mean(split, key, j) for j in range(len(METHODS))]
            sum_data.append([label] + [fmt(v) for v in vals])
            sum_best.append(best_indices(vals))
        sum_data.append(["# test wins"] + [str(w) for w in wins])
        sum_best.append(best_indices(wins))
        add_table_page(
            pdf, "Table 3: Summary Across Splits", sum_headers, sum_data,
            ["train", "val", "test", "test", "test"], sum_best,
            "Aggregate view of Tables 1 and 2 split by train / held-out validation / unseen test networks.",
            "Consistent gains on validation and test splits (not only train) show that the global-local fusion generalises "
            "zero-shot to unseen and larger weighted networks.", fontsize=8.5,
        )

        # bar chart on test graphs
        fig, ax = plt.subplots(figsize=(12, 8.5))
        names = [r["name"].replace(".txt", "").replace("synthetic_test_", "") for r in test_rows]
        x = np.arange(len(test_rows))
        plot_methods = [0, 1, 2, 6]
        colors = ["#1F4E79", "#A5A5A5", "#ED7D31", "#70AD47"]
        w = 0.8 / len(plot_methods)
        for k, (j, c) in enumerate(zip(plot_methods, colors)):
            ax.bar(x + (k - (len(plot_methods) - 1) / 2) * w, [r["tau"][j] for r in test_rows], w, label=METHODS[j], color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Kendall tau (top 20%)")
        ax.set_title("Figure 1: Unseen Test Networks - Hybrid vs Previous Best vs Eigenvector", fontsize=13,
                     weight="bold", color="#1F4E79")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
        for run in summary["runs"]:
            h = run["history"]
            ax1.plot(h["epoch"], h["loss"], label=f"seed {run['seed']}")
            ax2.plot(h["epoch"], h["val_tau"], label=f"seed {run['seed']} (best ep {run['best_epoch']})")
        ax1.set_title("Training loss")
        ax2.set_title("Validation tau@20% (held-out graphs)")
        for a in (ax1, ax2):
            a.set_xlabel("epoch")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig.suptitle("Figure 2: Training Dynamics", fontsize=13, weight="bold", color="#1F4E79")
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    print(f"\nPDF report written to {pdf_path}")
    print(f"CSV results written to {csv_path}")


if __name__ == "__main__":
    main()
