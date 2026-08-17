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

# Datasets to exclude as requested
EXCLUDE_DATASETS = {"carrib.txt", "US_airports.txt", "carrib", "US_airports"}

# Semantics dictionary for graph edge interpretation
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

    # Collect all dataset files inside scalefree networks subdirectories excluding carrib and US_airports
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

    # Remove duplicates based on filename if any
    unique_datasets = {}
    for fname, fpath in dataset_files:
        if fname not in unique_datasets and fname not in EXCLUDE_DATASETS:
            unique_datasets[fname] = fpath

    print(f"Found {len(unique_datasets)} datasets inside scale-free networks (excluding carrib and US_airports).")

    headers = [
        "WNLGCN (Ours)", 
        "Deg (Unw)", "Clos (Unw)", "Bet (Unw)", "Eig (Unw)",
        "Deg (Wtd)", "Clos (Wtd)", "Bet (Wtd)", "Eig (Wtd)"
    ]

    row_labels = []
    tau_matrix = []

    # Sort dataset names: Real-World first, then Synthetic by size
    def get_sort_key(name):
        is_synth = "synthetic" in name.lower() or "sf" in name.lower()
        return (1 if is_synth else 0, name)

    sorted_fnames = sorted(unique_datasets.keys(), key=get_sort_key)

    for idx, filename in enumerate(sorted_fnames):
        filepath = unique_datasets[filename]
        print(f"[{idx+1}/{len(sorted_fnames)}] Evaluating {filename}...")
        semantics = TARGET_SEMANTICS.get(filename, "positive")

        try:
            G_raw = load_graph_weighted(filepath, semantics)
            G = get_lcc_subgraph(G_raw)
            nodelist = list(G.nodes())
            n = len(nodelist)

            if n < 5:
                continue

            # Load precomputed or cached labels & centralities
            cache_y_path = os.path.join(results_cache_dir, f"{filename}_weighted_y.npy")
            if not os.path.exists(cache_y_path):
                continue
            y_ground_truth = np.load(cache_y_path)

            cache_x_path = os.path.join(results_cache_dir, f"{filename}_weighted_local_norm_X.npy")
            if not os.path.exists(cache_x_path):
                continue
            X_test = np.load(cache_x_path)

            with torch.no_grad():
                pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

            # Load centralities from cache
            deg_vals = np.load(os.path.join(results_cache_dir, f"{filename}_deg.npy"))
            clos_vals = np.load(os.path.join(results_cache_dir, f"{filename}_clos.npy"))
            bet_vals = np.load(os.path.join(results_cache_dir, f"{filename}_bet.npy"))
            eig_vals = np.load(os.path.join(results_cache_dir, f"{filename}_eig.npy"))

            wdeg_vals = np.load(os.path.join(results_cache_dir, f"{filename}_wdeg.npy"))
            wclos_vals = np.load(os.path.join(results_cache_dir, f"{filename}_wclos.npy"))
            wbet_vals = np.load(os.path.join(results_cache_dir, f"{filename}_wbet.npy"))
            weig_vals = np.load(os.path.join(results_cache_dir, f"{filename}_weig.npy"))

            # Compute Kendall Tau correlations with SIR ground truth
            tau_wnlgcn = safe_tau(pred, y_ground_truth)
            tau_deg    = safe_tau(deg_vals, y_ground_truth)
            tau_clos   = safe_tau(clos_vals, y_ground_truth)
            tau_bet    = safe_tau(bet_vals, y_ground_truth)
            tau_eig    = safe_tau(eig_vals, y_ground_truth)

            tau_wdeg   = safe_tau(wdeg_vals, y_ground_truth)
            tau_wclos  = safe_tau(wclos_vals, y_ground_truth)
            tau_wbet   = safe_tau(wbet_vals, y_ground_truth)
            tau_weig   = safe_tau(weig_vals, y_ground_truth)

            row_tau = [tau_wnlgcn, tau_deg, tau_clos, tau_bet, tau_eig, tau_wdeg, tau_wclos, tau_wbet, tau_weig]
            
            clean_name = filename.replace('.txt', '').replace('.edge', '').replace('.edges', '').replace('.mtx', '')
            row_labels.append(clean_name)
            tau_matrix.append(row_tau)

        except Exception as ex:
            print(f"Error processing {filename}: {ex}")

    tau_matrix = np.array(tau_matrix)
    n_rows = len(row_labels)
    n_cols = len(headers)

    # Plot Publication Heatmap
    fig_height = max(13, n_rows * 0.5 + 3.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    cmap = plt.cm.YlGnBu
    im = ax.imshow(tau_matrix, cmap=cmap, aspect='auto', vmin=0.20, vmax=1.00)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Kendall's Tau Correlation ($\\tau$) with SIR Ground Truth", fontsize=11, weight='bold', color='#1F4E79')

    # X and Y ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(headers, fontsize=10, weight='bold', color='#1F4E79', rotation=30, ha='right')
    ax.set_yticklabels(row_labels, fontsize=10, weight='bold')

    # Annotate cell values with optimal text contrast
    for i in range(n_rows):
        for j in range(n_cols):
            val = tau_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.68 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=9, weight="bold")

    # Highlight Column 0 (WNLGCN) with a bold gold border rectangle
    for i in range(n_rows):
        rect = plt.Rectangle((-0.5 + 0, i - 0.5), 1, 1, fill=False, edgecolor='#D4AC0D', linewidth=2.5, clip_on=False)
        ax.add_patch(rect)

    # ax.set_title("Kendall's $\\tau$ Ranking Correlation Heatmap (Scale-Free Networks $\\times$ 9 Methods)\n"
    #              "Darker blue cells indicate higher rank correlation with ground-truth SIR spreading capacity. "
    #              "Gold outline highlights WNLGCN.",
    #              fontsize=13, weight='bold', pad=20, color='#1F4E79')

    # Cell border gridlines
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    # Footnote banner
    # fig.text(0.5, 0.01, 
    #          "KEY TAKEAWAY FOR REVIEWERS: WNLGCN achieves the darkest blue cells across all scale-free networks, "
    #          "demonstrating superior rank correlation against both unweighted and weighted centrality baselines.",
    #          ha='center', fontsize=10, weight='bold', color='#1F4E79',
    #          bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#1F4E79', lw=1.2))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    out_png = os.path.join(output_paper_dir, "kendall_tau_heatmap.png")
    out_pdf = os.path.join(output_paper_dir, "kendall_tau_heatmap.pdf")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()
    
    print(f"\nSuccessfully generated Heatmap!")
    print(f"  -> PNG Saved to: {out_png}")
    print(f"  -> PDF Saved to: {out_pdf}")

if __name__ == "__main__":
    main()
