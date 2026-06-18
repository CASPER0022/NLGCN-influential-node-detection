import os
import sys
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from joblib import Parallel, delayed

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Path resolution relative to script file
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "Datasets", "Budapest.txt"))
model_path = os.path.join(script_dir, "nlgcn_model.pth")

# ---- NLGCN Model Definition (Must match training script) ----
class ChannelAttention(nn.Module):
    def __init__(self, channels=9, reduction=3):
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

class NLGCN(nn.Module):
    def __init__(self):
        super(NLGCN, self).__init__()
        self.attention = ChannelAttention(9)
        self.conv1 = nn.Conv2d(9, 16, kernel_size=2)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 20 * 20, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.attention(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---- Graph Loader ----
def load_graph(path):
    print(f"Reading edges file from {path}...")
    G = nx.Graph()
    edges = np.loadtxt(path, dtype=int, usecols=(0, 1))
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    G.add_edges_from(edges)
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# ---- Optimized SIR Simulation ----
def SIR_simulation(G, seed, beta, mu, steps=1000):
    susceptible = set(G.nodes())
    infected = {seed}
    recovered = set()
    susceptible.remove(seed)

    for _ in range(steps):
        new_infected = set()
        new_recovered = set()

        for node in infected:
            for nbr in G.neighbors(node):
                if nbr in susceptible:
                    if random.random() < beta:
                        new_infected.add(nbr)

            if random.random() < mu:
                new_recovered.add(node)

        infected |= new_infected
        infected -= new_recovered
        recovered |= new_recovered
        susceptible -= new_infected

        if len(infected) == 0:
            break

    return len(recovered)

def single_node_sir(G, seed, beta, mu, runs=500):
    spread = 0
    for _ in range(runs):
        spread += SIR_simulation(G, seed, beta, mu)
    return spread / runs

# ---- Optimized Channel Feature Embedding ----
def embed_channel(mat, nodes, feature_dict):
    size = mat.shape[0]
    out = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            u = nodes[i]
            v = nodes[j]

            # diagonal (i == j) -> self feature
            if i == j:
                out[i, j] = feature_dict.get(u, 0)
            # first row (i == 0, j > 0) -> a_1j * W_j
            elif i == 0 and j > 0:
                if mat[i, j] == 1:
                    out[i, j] = feature_dict.get(v, 0)
            # first column (j == 0, i > 0) -> a_i1 * W_i
            elif j == 0 and i > 0:
                if mat[i, j] == 1:
                    out[i, j] = feature_dict.get(u, 0)
            # otherwise -> raw adjacency value a_ij
            else:
                out[i, j] = mat[i, j]
    return out

def main():
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}. Please complete training first.")
        sys.exit(1)

    print("Loading Budapest graph and calculating features...")
    G = load_graph(dataset_path)
    nodelist = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodelist)}
    n = len(nodelist)
    deg = np.array([G.degree(node) for node in nodelist])

    # 1. Compute Distance Matrix
    print("Calculating shortest path lengths...")
    dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.zeros((n, n))
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i != j and v in dist[u]:
                dist_matrix[i, j] = dist[u][v]

    # 2. Global Influence (NGI)
    print("Computing Global Influence (NGI)...")
    alpha = 0.5
    NGI = np.zeros(n)
    for i in range(n):
        dists = dist_matrix[i]
        mask = (dists > 0)
        if np.any(mask):
            NGI[i] = np.sum(np.sqrt(deg[mask] + alpha) / dists[mask])

    # 3. Local Influence (NLI)
    print("Computing Local Influence (NLI)...")
    K_hop = 3
    NLI = np.zeros(n)
    for i in range(n):
        hop_count = np.sum((dist_matrix[i] >= 1) & (dist_matrix[i] <= K_hop))
        if hop_count > 0:
            NLI[i] = (deg[i] * np.log10(hop_count)) / n

    # 4. Multi-scale weight updates
    print("Performing multi-scale updates...")
    A = nx.to_numpy_array(G, nodelist=nodelist)
    W_NLI1 = NLI.copy()
    W_NLI2 = W_NLI1 + A.dot(W_NLI1)
    W_NLI3 = W_NLI2 + A.dot(W_NLI2)

    W_NGI1 = NGI.copy()
    W_NGI2 = W_NGI1 + A.dot(W_NGI1)
    W_NGI3 = W_NGI2 + A.dot(W_NGI2)

    NLI_dict = {node: NLI[i] for i, node in enumerate(nodelist)}
    NGI_dict = {node: NGI[i] for i, node in enumerate(nodelist)}

    # Compute global Closeness Centrality and Core number (normalized)
    print("Computing global centralities (Closeness, Coreness)...")
    clos_cent = nx.closeness_centrality(G)
    core_cent = nx.core_number(G)
    max_core = max(core_cent.values()) if core_cent else 1

    # Pre-compute component sizes for all nodes
    comp_dict = {node: len(c) for c in nx.connected_components(G) for node in c}
    log_n_total = np.log10(n) if n > 1 else 1.0

    # 5. Extract neighborhood matrices (9 channels)
    print("Generating neighborhood channels...")
    L = 40
    channels = []
    for node in nodelist:
        nbrs = list(G.neighbors(node))
        nbrs_sorted = sorted(nbrs, key=lambda x: W_NLI3[node_index[x]], reverse=True)
        nbrs_selected = nbrs_sorted[:L]

        nodes = [node] + nbrs_selected
        if len(nodes) < L + 1:
            nodes += [None] * (L + 1 - len(nodes))

        size = L + 1
        mat = np.zeros((size, size))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u is not None and v is not None and G.has_edge(u, v):
                    mat[i, j] = 1

        f1 = {n: NLI_dict.get(n, 0) for n in nodes}
        f2 = {n: W_NLI2[node_index[n]] if n is not None else 0 for n in nodes}
        f3 = {n: W_NLI3[node_index[n]] if n is not None else 0 for n in nodes}
        f4 = {n: NGI_dict.get(n, 0) for n in nodes}
        f5 = {n: W_NGI2[node_index[n]] if n is not None else 0 for n in nodes}
        f6 = {n: W_NGI3[node_index[n]] if n is not None else 0 for n in nodes}
        f7 = {
            node_in_window: np.log10(comp_dict.get(node_in_window, 1)) / log_n_total
            if node_in_window is not None else 0
            for node_in_window in nodes
        }
        f8 = {n: clos_cent.get(n, 0) if n is not None else 0 for n in nodes}
        f9 = {n: core_cent.get(n, 0) / max_core if n is not None and max_core > 0 else 0 for n in nodes}

        c1 = embed_channel(mat, nodes, f1)
        c2 = embed_channel(mat, nodes, f2)
        c3 = embed_channel(mat, nodes, f3)
        c4 = embed_channel(mat, nodes, f4)
        c5 = embed_channel(mat, nodes, f5)
        c6 = embed_channel(mat, nodes, f6)
        c7 = embed_channel(mat, nodes, f7)
        c8 = embed_channel(mat, nodes, f8)
        c9 = embed_channel(mat, nodes, f9)

        tensor = np.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9])
        channels.append(tensor)

    X_test = np.array(channels)

    # Normalize input features per channel using global training statistics (leakage-free)
    mean_file = os.path.join(script_dir, "X_mean.npy")
    std_file = os.path.join(script_dir, "X_std.npy")
    if not os.path.exists(mean_file) or not os.path.exists(std_file):
        raise FileNotFoundError("Training statistics (X_mean.npy / X_std.npy) not found. Run train_nlgcn_multigraph.py first.")
    X_mean = np.load(mean_file)
    X_std = np.load(std_file)
    X_test = (X_test - X_mean) / (X_std + 1e-6)

    # 6. Load Trained NLGCN Model weights
    print(f"Loading NLGCN model weights from {model_path}...")
    model = NLGCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 7. Model Inference (Centrality Predictions)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_tensor).numpy().flatten()

    # 8. Run parallel SIR simulations for ground truth evaluation
    print("Running parallel SIR simulations for ground truth spreading capacity (SIR)...")
    k_avg = np.mean(deg)
    k2_avg = np.mean(deg**2)
    beta_c = k_avg / (k2_avg - k_avg) if (k2_avg - k_avg) != 0 else 0.1
    beta = 1.5 * beta_c
    mu = 1.0

    results = Parallel(n_jobs=-1)(
        delayed(single_node_sir)(G, node, beta, mu, runs=500) 
        for node in nodelist
    )
    labels = np.array(results)

    # 9. Compute traditional centrality measures
    print("\nCalculating traditional centrality measures...")
    G_clean = G.copy()
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))

    deg_cent = nx.degree_centrality(G_clean)
    deg_cent_vals = np.array([deg_cent[node] for node in nodelist])
    
    clos_cent_val = np.array([clos_cent[node] for node in nodelist])

    bet_cent = nx.betweenness_centrality(G_clean)
    bet_cent_vals = np.array([bet_cent[node] for node in nodelist])

    pr_cent = nx.pagerank(G_clean)
    pr_cent_vals = np.array([pr_cent[node] for node in nodelist])

    core_cent_vals = np.array([core_cent[node] for node in nodelist])

    try:
        eig_cent = nx.eigenvector_centrality(G_clean, max_iter=1000)
        eig_cent_vals = np.array([eig_cent[node] for node in nodelist])
        has_eig = True
    except Exception:
        eig_cent_vals = np.zeros_like(labels)
        has_eig = False

    # 10. Compute Kendall Tau correlations
    # A: Prediction vs SIR
    tau_pred_sir, _ = kendalltau(pred, labels)
    
    # B: Prediction vs Traditional Centrality Measures
    tau_pred_deg, _ = kendalltau(pred, deg_cent_vals)
    tau_pred_clos, _ = kendalltau(pred, clos_cent_val)
    tau_pred_bet, _ = kendalltau(pred, bet_cent_vals)
    tau_pred_pr, _ = kendalltau(pred, pr_cent_vals)
    tau_pred_core, _ = kendalltau(pred, core_cent_vals)
    if has_eig:
        tau_pred_eig, _ = kendalltau(pred, eig_cent_vals)
        eig_pred_str = f"{tau_pred_eig:.4f}"
    else:
        eig_pred_str = "Failed to converge"

    # C: Centrality Measures vs SIR
    tau_deg_sir, _ = kendalltau(deg_cent_vals, labels)
    tau_clos_sir, _ = kendalltau(clos_cent_val, labels)
    tau_bet_sir, _ = kendalltau(bet_cent_vals, labels)
    tau_pr_sir, _ = kendalltau(pr_cent_vals, labels)
    tau_core_sir, _ = kendalltau(core_cent_vals, labels)
    if has_eig:
        tau_eig_sir, _ = kendalltau(eig_cent_vals, labels)
        eig_sir_str = f"{tau_eig_sir:.4f}"
    else:
        eig_sir_str = "Failed to converge"

    # Display results
    print("\n" + "="*70)
    print("  BUDAPEST NETWORK MODEL PERFORMANCE EVALUATION")
    print("="*70)
    print(f"Nodes: {n} | Edges: {G.number_of_edges()}")
    print("-"*70)
    print(f"Model (NLGCN GNN) vs SIR ground truth Kendall Tau: {tau_pred_sir:.4f}")
    print("-"*70)
    
    print("\nTable 1: Correlation of NLGCN Predictions & Centrality Measures with SIR")
    print("-"*70)
    print(f"{'Method / Centrality Measure':<30} | {'Kendall Tau with SIR':<30}")
    print("-"*70)
    print(f"{'Ours (NLGCN GNN)':<30} | {tau_pred_sir:<30.4f}")
    print(f"{'Degree Centrality':<30} | {tau_deg_sir:<30.4f}")
    print(f"{'Closeness Centrality':<30} | {tau_clos_sir:<30.4f}")
    print(f"{'Betweenness Centrality':<30} | {tau_bet_sir:<30.4f}")
    print(f"{'PageRank':<30} | {tau_pr_sir:<30.4f}")
    print(f"{'Coreness (K-core)':<30} | {tau_core_sir:<30.4f}")
    print(f"{'Eigenvector Centrality':<30} | {eig_sir_str:<30}")
    print("-"*70)

    print("\nTable 2: Correlation of NLGCN Predictions with Traditional Centrality Measures")
    print("-"*70)
    print(f"{'Centrality Measure':<30} | {'Kendall Tau with NLGCN Prediction':<30}")
    print("-"*70)
    print(f"{'Degree Centrality':<30} | {tau_pred_deg:<30.4f}")
    print(f"{'Closeness Centrality':<30} | {tau_pred_clos:<30.4f}")
    print(f"{'Betweenness Centrality':<30} | {tau_pred_bet:<30.4f}")
    print(f"{'PageRank':<30} | {tau_pred_pr:<30.4f}")
    print(f"{'Coreness (K-core)':<30} | {tau_pred_core:<30.4f}")
    print(f"{'Eigenvector Centrality':<30} | {eig_pred_str:<30}")
    print("-"*70)

    # 11. Ranking evaluation (Top-15 predicted nodes vs Top-15 SIR nodes)
    ranking_pred = np.argsort(pred)[::-1]
    ranking_true = np.argsort(labels)[::-1]

    top_pred_nodes = [int(nodelist[idx]) for idx in ranking_pred[:15]]
    top_true_nodes = [int(nodelist[idx]) for idx in ranking_true[:15]]

    print(f"\nTop-15 Predicted key nodes: {top_pred_nodes}")
    print(f"Top-15 Ground Truth (SIR) nodes: {top_true_nodes}")

    overlap = set(top_pred_nodes).intersection(set(top_true_nodes))
    print(f"Number of overlapping top key nodes: {len(overlap)} / 15")

if __name__ == "__main__":
    main()
