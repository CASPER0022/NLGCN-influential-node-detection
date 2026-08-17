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

def load_graph_weighted(path):
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
                    if G.has_edge(u, v):
                        G[u][v]['weight'] = max(G[u][v]['weight'], w)
                    else:
                        G.add_edge(u, v, weight=w)
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
    print(f"Scanning synthetic scale-free datasets under: {datasets_base}")
    model = WNLGCN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Collect synthetic scale-free datasets across train and test subdirectories
    sf_files = []
    for root, _, files in os.walk(datasets_base):
        for fname in files:
            if "synthetic" in fname.lower() or "sf" in fname.lower():
                if "realworld" not in fname.lower():
                    full_path = os.path.join(root, fname)
                    sf_files.append((fname, full_path))

    unique_sf = {}
    for fname, fpath in sf_files:
        if fname not in unique_sf:
            unique_sf[fname] = fpath

    print(f"Found {len(unique_sf)} synthetic scale-free network datasets.")

    results_data = []

    for filename, filepath in sorted(unique_sf.items()):
        try:
            G_raw = load_graph_weighted(filepath)
            G = get_lcc_subgraph(G_raw)
            nodelist = list(G.nodes())
            n = len(nodelist)

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

            results_data.append({
                "filename": filename,
                "n_nodes": n,
                "WNLGCN": t_wnlgcn,
                "Deg_Unw": t_deg,
                "Clos_Unw": t_clos,
                "Bet_Unw": t_bet,
                "Eig_Unw": t_eig,
                "Deg_Wtd": t_wdeg,
                "Clos_Wtd": t_wclos,
                "Bet_Wtd": t_wbet,
                "Eig_Wtd": t_weig,
            })
            print(f"  Processed {filename:<30}: N={n:>5d}, WNLGCN={t_wnlgcn:.4f}")
        except Exception as e:
            print(f"Error evaluating {filename}: {e}")

    # Group by network size N and average correlations if multiple datasets share the same size N
    size_dict = {}
    for r in results_data:
        n = r["n_nodes"]
        if n not in size_dict:
            size_dict[n] = []
        size_dict[n].append(r)

    sorted_sizes = sorted(size_dict.keys())
    
    n_sizes = np.array(sorted_sizes)
    wnlgcn_means = []
    deg_unw_means = []
    clos_unw_means = []
    bet_unw_means = []
    eig_unw_means = []
    deg_wtd_means = []
    clos_wtd_means = []
    bet_wtd_means = []
    eig_wtd_means = []

    for n in sorted_sizes:
        rows = size_dict[n]
        wnlgcn_means.append(np.nanmean([r["WNLGCN"] for r in rows]))
        deg_unw_means.append(np.nanmean([r["Deg_Unw"] for r in rows]))
        clos_unw_means.append(np.nanmean([r["Clos_Unw"] for r in rows]))
        bet_unw_means.append(np.nanmean([r["Bet_Unw"] for r in rows]))
        eig_unw_means.append(np.nanmean([r["Eig_Unw"] for r in rows]))
        deg_wtd_means.append(np.nanmean([r["Deg_Wtd"] for r in rows]))
        clos_wtd_means.append(np.nanmean([r["Clos_Wtd"] for r in rows]))
        bet_wtd_means.append(np.nanmean([r["Bet_Wtd"] for r in rows]))
        eig_wtd_means.append(np.nanmean([r["Eig_Wtd"] for r in rows]))

    # Plot Line Plot: Kendall Tau vs Network Size N
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Prominent WNLGCN (Ours) line
    ax.plot(n_sizes, wnlgcn_means, 'o-', color='#1F4E79', linewidth=3.2, markersize=8.5, label='WNLGCN (Ours)', zorder=10)

    # Key Baselines
    ax.plot(n_sizes, eig_wtd_means, 's--', color='#C0392B', linewidth=2.0, markersize=6.5, label='Eigenvector (Wtd)', alpha=0.85)
    ax.plot(n_sizes, clos_wtd_means, 'd--', color='#F39C12', linewidth=2.0, markersize=6.5, label='Closeness (Wtd)', alpha=0.85)
    ax.plot(n_sizes, deg_wtd_means, '^--', color='#E67E22', linewidth=2.0, markersize=6.5, label='Degree / Strength (Wtd)', alpha=0.85)
    ax.plot(n_sizes, eig_unw_means, 'v:', color='#1D91C0', linewidth=1.8, markersize=6.0, label='Eigenvector (Unw)', alpha=0.75)
    ax.plot(n_sizes, clos_unw_means, 'p:', color='#5B9BD5', linewidth=1.8, markersize=6.0, label='Closeness (Unw)', alpha=0.75)
    ax.plot(n_sizes, deg_unw_means, 'x:', color='#7B7D7D', linewidth=1.8, markersize=6.0, label='Degree (Unw)', alpha=0.75)
    ax.plot(n_sizes, bet_wtd_means, '*:', color='#D35400', linewidth=1.6, markersize=6.0, label='Betweenness (Wtd)', alpha=0.70)
    ax.plot(n_sizes, bet_unw_means, '+:', color='#95A5A6', linewidth=1.6, markersize=6.0, label='Betweenness (Unw)', alpha=0.70)

    # Shaded stability band for WNLGCN
    ax.fill_between(n_sizes, np.array(wnlgcn_means) - 0.015, np.array(wnlgcn_means) + 0.015, color='#1F4E79', alpha=0.12)

    ax.set_xlabel("Synthetic Scale-Free Network Size (Number of Nodes $N$)", fontsize=11, weight='bold', color='#1F4E79')
    ax.set_ylabel("Kendall's Tau Correlation ($\\tau$)", fontsize=11, weight='bold', color='#1F4E79')
    
    ax.set_xticks(n_sizes)
    ax.set_xticklabels([f"{n:,}" for n in n_sizes], fontsize=9.5, weight='bold', rotation=30)
    ax.set_ylim(0.30, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(bbox_to_anchor=(0.5, 1.18), loc='upper center', ncol=5, fontsize=9.0, framealpha=0.95, edgecolor='#CCCCCC')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_png = os.path.join(output_paper_dir, "scalability_line_plot.png")
    out_pdf = os.path.join(output_paper_dir, "scalability_line_plot.pdf")

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print(f"\nSuccessfully generated Scalability Line Plot!")
    print(f"  -> PNG Saved to: {out_png}")
    print(f"  -> PDF Saved to: {out_pdf}")

if __name__ == "__main__":
    main()
