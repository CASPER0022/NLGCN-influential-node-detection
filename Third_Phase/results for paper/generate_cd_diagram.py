import os
import sys
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, rankdata
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

            if not any(np.isnan(row_tau)):
                all_dataset_taus.append(row_tau)
                selected_datasets.append(filename)

        except Exception as e:
            print(f"Error evaluating {filename}: {e}")

    all_dataset_taus = np.array(all_dataset_taus)
    N_datasets, M_methods = all_dataset_taus.shape

    # Compute Ranks per dataset (Higher Kendall Tau = Better Rank = Rank 1)
    # rankdata(-val) assigns 1 to highest value
    ranks = np.array([rankdata(-row) for row in all_dataset_taus])
    mean_ranks = np.mean(ranks, axis=0)

    print(f"\nEvaluated {N_datasets} datasets for Critical Difference (CD) Diagram.")
    print("Average Ranks (Rank 1 is best):")
    for m, r in zip(methods, mean_ranks):
        print(f"  {m:<15}: {r:.2f}")

    # Compute Critical Difference (CD) value using Nemenyi test (alpha = 0.05)
    # For M=9 methods, q_alpha = 3.102 at alpha=0.05
    q_alpha = 3.102
    cd = q_alpha * np.sqrt((M_methods * (M_methods + 1.0)) / (6.0 * N_datasets))
    print(f"\nCritical Difference (CD) at alpha=0.05: {cd:.3f}")

    # Plot Critical Difference (CD) Diagram
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(11, 6.0))

    # Axis bounds
    low = 1.0
    high = float(M_methods)
    
    # Draw main rank line at y = 0
    y_axis = 0.0
    ax.plot([low, high], [y_axis, y_axis], color='#333333', linewidth=2.5)

    # Tick marks on rank axis
    for r in range(1, M_methods + 1):
        ax.plot([r, r], [y_axis - 0.04, y_axis + 0.04], color='#333333', linewidth=1.5)
        ax.text(r, y_axis + 0.12, str(r), ha='center', va='bottom', fontsize=11, weight='bold', color='#1F4E79')

    ax.text((low + high) / 2.0, y_axis + 0.38, "Average Rank (Lower is Better)", ha='center', va='bottom', fontsize=12, weight='bold', color='#1F4E79')

    # Sort methods by average rank
    sorted_indices = np.argsort(mean_ranks)
    sorted_methods = [methods[i] for i in sorted_indices]
    sorted_ranks = mean_ranks[sorted_indices]

    # Split methods left side (better ranks) and right side
    split_idx = int(np.ceil(M_methods / 2.0))
    left_methods = sorted_methods[:split_idx]
    left_ranks = sorted_ranks[:split_idx]
    
    right_methods = sorted_methods[split_idx:]
    right_ranks = sorted_ranks[split_idx:]

    # Plot Left Side methods (pointing to left)
    left_y_step = 0.22
    for idx, (m, r) in enumerate(zip(left_methods, left_ranks)):
        y_pos = -0.35 - idx * left_y_step
        color = '#1F4E79' if "WNLGCN" in m else '#333333'
        lw = 2.5 if "WNLGCN" in m else 1.2
        font_wt = 'bold' if "WNLGCN" in m else 'normal'

        # Line from rank axis down to marker level, then left to text
        ax.plot([r, r], [y_axis, y_pos], color=color, linewidth=lw, linestyle='-')
        ax.plot([r, low - 0.3], [y_pos, y_pos], color=color, linewidth=lw, linestyle='-')
        ax.plot(r, y_pos, 'o', color=color, markersize=6)

        ax.text(low - 0.35, y_pos, f"{m} ({r:.2f})", ha='right', va='center', fontsize=10, weight=font_wt, color=color)

    # Plot Right Side methods (pointing to right)
    right_y_step = 0.22
    for idx, (m, r) in enumerate(zip(right_methods, right_ranks)):
        y_pos = -0.35 - idx * right_y_step
        color = '#333333'

        ax.plot([r, r], [y_axis, y_pos], color=color, linewidth=1.2, linestyle='-')
        ax.plot([r, high + 0.3], [y_pos, y_pos], color=color, linewidth=1.2, linestyle='-')
        ax.plot(r, y_pos, 'o', color=color, markersize=6)

        ax.text(high + 0.35, y_pos, f"{m} ({r:.2f})", ha='left', va='center', fontsize=10, color=color)

    # Draw CD bar indicator at top right
    cd_y = 0.85
    cd_x_start = high - cd
    ax.plot([cd_x_start, high], [cd_y, cd_y], color='#C0392B', linewidth=3.0)
    ax.plot([cd_x_start, cd_x_start], [cd_y - 0.04, cd_y + 0.04], color='#C0392B', linewidth=2.0)
    ax.plot([high, high], [cd_y - 0.04, cd_y + 0.04], color='#C0392B', linewidth=2.0)
    ax.text((cd_x_start + high) / 2.0, cd_y + 0.08, f"CD = {cd:.2f} ($\\alpha=0.05$)", ha='center', va='bottom', fontsize=10, weight='bold', color='#C0392B')

    # Draw clique lines connecting methods that are not significantly different (diff < CD)
    clique_y = -1.45
    for i in range(M_methods):
        for j in range(i + 1, M_methods):
            r1 = sorted_ranks[i]
            r2 = sorted_ranks[j]
            if abs(r1 - r2) <= cd:
                ax.plot([r1, r2], [clique_y, clique_y], color='#555555', linewidth=3.5, alpha=0.7)
                clique_y -= 0.08
                break

    ax.set_xlim(low - 2.2, high + 2.2)
    ax.set_ylim(clique_y - 0.3, 1.2)
    ax.axis('off')

    plt.tight_layout()

    out_png = os.path.join(output_paper_dir, "critical_difference_diagram.png")
    out_pdf = os.path.join(output_paper_dir, "critical_difference_diagram.pdf")

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print(f"\nSuccessfully generated Critical Difference (CD) Diagram!")
    print(f"  -> PNG Saved to: {out_png}")
    print(f"  -> PDF Saved to: {out_pdf}")

if __name__ == "__main__":
    main()
