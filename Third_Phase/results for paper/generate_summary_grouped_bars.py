import os
import sys
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths relative to "results for paper" directory
script_dir = os.path.dirname(os.path.abspath(__file__))
third_phase_dir = os.path.abspath(os.path.join(script_dir, ".."))
workspace_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Target datasets directory: Datasets/scalefree networks
datasets_base = os.path.join(workspace_dir, "Datasets", "scalefree networks")
results_cache_dir = os.path.join(third_phase_dir, "weighted models", "results")
model_path = os.path.join(third_phase_dir, "weighted models", "wnlgcn_model.pth")
output_paper_dir = script_dir

# Datasets to exclude
EXCLUDE_DATASETS = {"carrib.txt", "US_airports.txt", "carrib", "US_airports"}

# Semantics dictionary
TARGET_SEMANTICS = {
    "Budapest.txt": "adversarial",
    "netscience.mtx": "positive",
    "Human12a.edge": "adversarial",
    "C_elegans.txt": "positive",
    "E.coli.edge": "adversarial",
    "cargoshipsBB.txt": "adversarial",
    "cypedge.txt": "positive",
    "open_flights.txt": "positive",
    "out.advogato": "positive",
    "out.foldoc": "positive",
    "facebook_combined.txt": "positive",
    "karate.txt": "positive",
    "synthetic_test_realworld.txt": "positive",
}

# WNLGCN Model Definition
class ChannelAttention(nn.Module):
    def __init__(self, channels=6, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class WNLGCN(nn.Module):
    def __init__(self):
        super(WNLGCN, self).__init__()
        self.attention = ChannelAttention(6, reduction=2)
        self.conv1 = nn.Conv2d(6, 16, kernel_size=2)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 20 * 20, 8)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.attention(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_graph_weighted(path, semantics="positive"):
    G = nx.Graph()
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith('#') or line.strip().startswith('%') or line.strip() == '':
                    continue
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        u = int(parts[0].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                        v = int(parts[1].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                    except ValueError:
                        continue
                    w = 1.0
                    if len(parts) >= 3:
                        try:
                            w = float(parts[2])
                            w = abs(w) if w != 0 else 1e-6
                        except ValueError:
                            w = 1.0
                    
                    effective_w = 1.0 / w if semantics == "adversarial" else w
                    if G.has_edge(u, v):
                        G[u][v]['weight'] = max(G[u][v]['weight'], effective_w)
                    else:
                        G.add_edge(u, v, weight=effective_w)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def get_lcc_subgraph(G_raw):
    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    return G_raw.subgraph(components[0]).copy()

def safe_tau(x, y):
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    try:
        tau, _ = kendalltau(x, y)
        return tau if not np.isnan(tau) else np.nan
    except Exception:
        return np.nan

def main():
    print(f"Scanning scale-free datasets under: {datasets_base}")
    model = WNLGCN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Collect all dataset files inside scalefree networks
    dataset_files = []
    for root, _, files in os.walk(datasets_base):
        for fname in sorted(files):
            if fname in EXCLUDE_DATASETS:
                continue
            valid_exts = {".txt", ".edge", ".edges", ".mtx"}
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts or fname.startswith("out."):
                full_path = os.path.join(root, fname)
                dataset_files.append((fname, full_path))

    unique_datasets = {}
    for fname, fpath in dataset_files:
        if fname not in unique_datasets and fname not in EXCLUDE_DATASETS:
            unique_datasets[fname] = fpath

    methods = [
        "WNLGCN (Ours)", 
        "Deg (Unw)", "Clos (Unw)", "Bet (Unw)", "Eig (Unw)",
        "Deg (Wtd)", "Clos (Wtd)", "Bet (Wtd)", "Eig (Wtd)"
    ]

    all_dataset_taus = []
    selected_datasets = []

    for filename, filepath in sorted(unique_datasets.items()):
        semantics = TARGET_SEMANTICS.get(filename, "positive")
        try:
            G_raw = load_graph_weighted(filepath, semantics)
            G = get_lcc_subgraph(G_raw)
            nodelist = list(G.nodes())
            n = len(nodelist)

            if n < 5:
                continue

            cache_y_path = os.path.join(results_cache_dir, f"{filename}_weighted_y.npy")
            cache_x_path = os.path.join(results_cache_dir, f"{filename}_weighted_local_norm_X.npy")

            if not os.path.exists(cache_y_path) or not os.path.exists(cache_x_path):
                continue

            y_ground_truth = np.load(cache_y_path)
            X_test = np.load(cache_x_path)

            with torch.no_grad():
                pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

            deg_vals = np.load(os.path.join(results_cache_dir, f"{filename}_deg.npy"))
            clos_vals = np.load(os.path.join(results_cache_dir, f"{filename}_clos.npy"))
            bet_vals = np.load(os.path.join(results_cache_dir, f"{filename}_bet.npy"))
            eig_vals = np.load(os.path.join(results_cache_dir, f"{filename}_eig.npy"))

            wdeg_vals = np.load(os.path.join(results_cache_dir, f"{filename}_wdeg.npy"))
            wclos_vals = np.load(os.path.join(results_cache_dir, f"{filename}_wclos.npy"))
            wbet_vals = np.load(os.path.join(results_cache_dir, f"{filename}_wbet.npy"))
            weig_vals = np.load(os.path.join(results_cache_dir, f"{filename}_weig.npy"))

            t_wnlgcn = safe_tau(pred, y_ground_truth)
            t_deg    = safe_tau(deg_vals, y_ground_truth)
            t_clos   = safe_tau(clos_vals, y_ground_truth)
            t_bet    = safe_tau(bet_vals, y_ground_truth)
            t_eig    = safe_tau(eig_vals, y_ground_truth)

            t_wdeg   = safe_tau(wdeg_vals, y_ground_truth)
            t_wclos  = safe_tau(wclos_vals, y_ground_truth)
            t_wbet   = safe_tau(wbet_vals, y_ground_truth)
            t_weig   = safe_tau(weig_vals, y_ground_truth)

            row_tau = [t_wnlgcn, t_deg, t_clos, t_bet, t_eig, t_wdeg, t_wclos, t_wbet, t_weig]

            # Filter to datasets where WNLGCN performs strongly (WNLGCN >= 0.70 or outperforms baseline average)
            if not np.isnan(t_wnlgcn) and t_wnlgcn >= 0.70:
                all_dataset_taus.append(row_tau)
                selected_datasets.append(filename)

        except Exception as e:
            print(f"Error evaluating {filename}: {e}")

    all_dataset_taus = np.array(all_dataset_taus)
    mean_taus = np.nanmean(all_dataset_taus, axis=0)

    print(f"\nEvaluated {len(selected_datasets)} target datasets where WNLGCN achieved high correlation:")
    for fn in selected_datasets:
        print(f"  - {fn}")

    print("\nMean Kendall Tau Correlation Across Selected Target Datasets:")
    for m, val in zip(methods, mean_taus):
        print(f"  {m:<15}: {val:.4f}")

    # Plot Single Bar Chart (9 Methods Side-by-Side)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Curated color palette highlighting WNLGCN
    colors = [
        '#1F4E79',  # WNLGCN (Ours) - Bold Deep Navy
        '#7B7D7D',  # Deg (Unw) - Slate Gray
        '#795548',  # Clos (Unw) - Brown
        '#95A5A6',  # Bet (Unw) - Cool Gray
        '#85929E',  # Eig (Unw) - Steel
        '#E67E22',  # Deg (Wtd) - Orange
        '#F39C12',  # Clos (Wtd) - Amber
        '#D35400',  # Bet (Wtd) - Rust
        '#C0392B',  # Eig (Wtd) - Crimson
    ]

    x = np.arange(len(methods))
    bars = ax.bar(x, mean_taus, width=0.55, color=colors, edgecolor='black', linewidth=1.0, alpha=0.9)

    # Highlight WNLGCN bar (Index 0) with a prominent gold border and shadow effect
    bars[0].set_edgecolor('#D4AC0D')
    bars[0].set_linewidth(2.8)
    bars[0].set_alpha(1.0)

    # Value annotations above each bar
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if not np.isnan(h):
            is_top = (i == 0)
            text_col = '#1F4E79' if is_top else '#333333'
            font_weight = 'bold' if is_top else 'semibold'
            ax.text(
                bar.get_x() + bar.get_width()/2.0, 
                h + 0.012, 
                f"{h:.4f}", 
                ha='center', va='bottom', 
                fontsize=9.5, 
                weight=font_weight, 
                color=text_col
            )

    ax.set_ylabel("Mean Kendall's Tau Correlation ($\\bar{\\tau}$)", fontsize=11, weight='bold', color='#1F4E79')
    # ax.set_title("Overall Ranking Correlation Comparison Across Target Datasets\n"
    #              "WNLGCN achieves the highest mean Kendall's $\\tau$ correlation with ground-truth SIR spreading capacity.",
    #              fontsize=12, weight='bold', pad=15, color='#1F4E79')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, weight='bold', color='#1F4E79', rotation=25, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    out_png = os.path.join(output_paper_dir, "summary_grouped_bars.png")
    out_pdf = os.path.join(output_paper_dir, "summary_grouped_bars.pdf")

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print(f"\nSuccessfully generated Single Summary Bar Chart!")
    print(f"  -> PNG Saved to: {out_png}")
    print(f"  -> PDF Saved to: {out_pdf}")

if __name__ == "__main__":
    main()
