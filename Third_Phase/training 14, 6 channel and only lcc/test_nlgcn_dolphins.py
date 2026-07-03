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
dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets", "Test", "dolphins.txt"))
model_path = os.path.join(script_dir, "nlgcn_model.pth")
results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))
os.makedirs(results_dir, exist_ok=True)

# ---- NLGCN Model Definition (6 Channels) ----
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

class NLGCN(nn.Module):
    def __init__(self):
        super(NLGCN, self).__init__()
        self.attention = ChannelAttention(6, reduction=2)
        self.conv1 = nn.Conv2d(6, 16, kernel_size=2)
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
    try:
        edges = np.loadtxt(path, dtype=int, usecols=(0, 1))
    except Exception:
        edges_str = np.loadtxt(path, dtype=str, usecols=(0, 1))
        cleaned = [[int(node.replace('V', '').replace('v', '')) for node in row] for row in edges_str]
        edges = np.array(cleaned)
    
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    G.add_edges_from(edges)
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

            if i == j:
                out[i, j] = feature_dict.get(u, 0)
            elif i == 0 and j > 0:
                if mat[i, j] == 1:
                    out[i, j] = feature_dict.get(v, 0)
            elif j == 0 and i > 0:
                if mat[i, j] == 1:
                    out[i, j] = feature_dict.get(u, 0)
            else:
                out[i, j] = mat[i, j]
    return out

def get_lcc_subgraph(G_raw):
    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    return G_raw.subgraph(components[0]).copy()

def main():
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}. Please complete training first.")
        sys.exit(1)

    filename = os.path.basename(dataset_path)
    cache_x_path = os.path.join(results_dir, f"{filename}_X.npy")
    cache_y_path = os.path.join(results_dir, f"{filename}_y.npy")

    # Load and filter graph to LCC
    G_raw = load_graph(dataset_path)
    G = get_lcc_subgraph(G_raw)
    nodelist = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodelist)}
    n = len(nodelist)
    deg = np.array([G.degree(node) for node in nodelist])
    print(f"LCC node count: {n} (out of {G_raw.number_of_nodes()}) | LCC edges: {G.number_of_edges()}")

    # Check cache for pre-saved X and y
    if os.path.exists(cache_x_path) and os.path.exists(cache_y_path):
        print("Loading precalculated features and labels from cache...")
        X_test = np.load(cache_x_path)
        labels = np.load(cache_y_path)
    else:
        print("Precalculated cache not fully found. Performing calculations...")
        # Load precalculated labels if they exist to skip SIR simulation
        labels = None
        if os.path.exists(cache_y_path):
            print("Found cached SIR simulation labels. Skipping simulation...")
            labels = np.load(cache_y_path)

        # 1. Compute Distance Matrix (LCC only)
        print("Calculating shortest path lengths within LCC...")
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

        NLI_dict = {node: np.log10(NLI[i] + 1) for i, node in enumerate(nodelist)}
        W_NLI2_dict = {node: np.log10(W_NLI2[i] + 1) for i, node in enumerate(nodelist)}
        W_NLI3_dict = {node: np.log10(W_NLI3[i] + 1) for i, node in enumerate(nodelist)}

        NGI_dict = {node: np.log10(NGI[i] + 1) for i, node in enumerate(nodelist)}
        W_NGI2_dict = {node: np.log10(W_NGI2[i] + 1) for i, node in enumerate(nodelist)}
        W_NGI3_dict = {node: np.log10(W_NGI3[i] + 1) for i, node in enumerate(nodelist)}

        # 5. Extract neighborhood matrices (6 channels)
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
            f2 = {n: W_NLI2_dict.get(n, 0) for n in nodes}
            f3 = {n: W_NLI3_dict.get(n, 0) for n in nodes}
            f4 = {n: NGI_dict.get(n, 0) for n in nodes}
            f5 = {n: W_NGI2_dict.get(n, 0) for n in nodes}
            f6 = {n: W_NGI3_dict.get(n, 0) for n in nodes}

            c1 = embed_channel(mat, nodes, f1)
            c2 = embed_channel(mat, nodes, f2)
            c3 = embed_channel(mat, nodes, f3)
            c4 = embed_channel(mat, nodes, f4)
            c5 = embed_channel(mat, nodes, f5)
            c6 = embed_channel(mat, nodes, f6)

            tensor = np.stack([c1, c2, c3, c4, c5, c6])
            channels.append(tensor)

        X_test = np.array(channels)

        # 6. Run parallel SIR simulations only if labels were not loaded
        if labels is None:
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
            np.save(cache_y_path, labels)

        # Save X_test to cache
        np.save(cache_x_path, X_test)
        print(f"Calculated results saved to cache inside {results_dir}")

    # Normalize input features per channel using global training statistics
    mean_file = os.path.join(script_dir, "X_mean.npy")
    std_file = os.path.join(script_dir, "X_std.npy")
    if not os.path.exists(mean_file) or not os.path.exists(std_file):
        raise FileNotFoundError("Training statistics (X_mean.npy / X_std.npy) not found. Run train_nlgcn_6channel_lcc.py first.")
    X_mean = np.load(mean_file)
    X_std = np.load(std_file)
    X_test = (X_test - X_mean) / (X_std + 1e-6)

    # Load Trained NLGCN Model weights
    print(f"Loading 6-channel NLGCN model weights from {model_path}...")
    model = NLGCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Model Inference (Centrality Predictions)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_tensor).numpy().flatten()

    # Compute traditional centrality measures on LCC
    print("\nCalculating traditional centrality measures on LCC...")
    deg_cent = nx.degree_centrality(G)
    deg_cent_vals = np.array([deg_cent[node] for node in nodelist])
    
    clos_cent = nx.closeness_centrality(G)
    clos_cent_val = np.array([clos_cent[node] for node in nodelist])

    bet_cent = nx.betweenness_centrality(G)
    bet_cent_vals = np.array([bet_cent[node] for node in nodelist])

    pr_cent = nx.pagerank(G)
    pr_cent_vals = np.array([pr_cent[node] for node in nodelist])

    core_cent = nx.core_number(G)
    core_cent_vals = np.array([core_cent[node] for node in nodelist])

    try:
        eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
        eig_cent_vals = np.array([eig_cent[node] for node in nodelist])
        has_eig = True
    except Exception:
        eig_cent_vals = np.zeros_like(labels)
        has_eig = False

    # Helper to compute Kendall's Tau for a given fraction
    def get_correlations(pct):
        if pct == 100:
            k = n
            indices = np.arange(n)
        else:
            k = max(1, int(pct / 100.0 * n))
            indices = np.argsort(labels)[::-1][:k]

        p = pred[indices]
        l = labels[indices]
        d = deg_cent_vals[indices]
        c = clos_cent_val[indices]
        b = bet_cent_vals[indices]
        pr = pr_cent_vals[indices]
        cor = core_cent_vals[indices]
        e = eig_cent_vals[indices]

        tau_pred, _ = kendalltau(p, l)
        tau_deg, _ = kendalltau(d, l)
        tau_clos, _ = kendalltau(c, l)
        tau_bet, _ = kendalltau(b, l)
        tau_pr, _ = kendalltau(pr, l)
        tau_core, _ = kendalltau(cor, l)
        tau_eig, _ = kendalltau(e, l) if has_eig else (np.nan, None)

        return {
            "k": k,
            "GNN": pointer_or_nan(tau_pred),
            "Degree": pointer_or_nan(tau_deg),
            "Closeness": pointer_or_nan(tau_clos),
            "Betweenness": pointer_or_nan(tau_bet),
            "PageRank": pointer_or_nan(tau_pr),
            "Coreness": pointer_or_nan(tau_core),
            "Eigenvector": pointer_or_nan(tau_eig)
        }

    def pointer_or_nan(v):
        return v if v is not None else np.nan

    res10 = get_correlations(10)
    res20 = get_correlations(20)
    res100 = get_correlations(100)

    # Format eigenvector display
    def fmt_val(val):
        return f"{val:.4f}" if not np.isnan(val) else "Failed"

    # Display results
    print("\n" + "="*95)
    print("  6-CHANNEL LCC MODEL PERFORMANCE EVALUATION (DOLPHINS)")
    print("="*95)
    print(f"LCC Nodes: {n} | LCC Edges: {G.number_of_edges()}")
    print("-"*95)
    
    print("\nTable 1: Kendall Tau Correlation of Predictions & Centrality Measures with SIR")
    print("-"*95)
    print(f"{'Method / Centrality Measure':<30} | {'Top 10% (k=' + str(res10['k']) + ')':<20} | {'Top 20% (k=' + str(res20['k']) + ')':<20} | {'Top 100% (All)':<15}")
    print("-"*95)
    print(f"{'Ours (6-ch LCC NLGCN GNN)':<30} | {res10['GNN']:<20.4f} | {res20['GNN']:<20.4f} | {res100['GNN']:<15.4f}")
    print(f"{'Degree Centrality':<30} | {res10['Degree']:<20.4f} | {res20['Degree']:<20.4f} | {res100['Degree']:<15.4f}")
    print(f"{'Closeness Centrality':<30} | {res10['Closeness']:<20.4f} | {res20['Closeness']:<20.4f} | {res100['Closeness']:<15.4f}")
    print(f"{'Betweenness Centrality':<30} | {res10['Betweenness']:<20.4f} | {res20['Betweenness']:<20.4f} | {res100['Betweenness']:<15.4f}")
    print(f"{'PageRank':<30} | {res10['PageRank']:<20.4f} | {res20['PageRank']:<20.4f} | {res100['PageRank']:<15.4f}")
    print(f"{'Coreness (K-core)':<30} | {res10['Coreness']:<20.4f} | {res20['Coreness']:<20.4f} | {res100['Coreness']:<15.4f}")
    print(f"{'Eigenvector Centrality':<30} | {fmt_val(res10['Eigenvector']):<20} | {fmt_val(res20['Eigenvector']):<20} | {fmt_val(res100['Eigenvector']):<15}")
    print("-"*95)

    # Ranking evaluation (Top-15 predicted nodes vs Top-15 SIR nodes)
    ranking_pred = np.argsort(pred)[::-1]
    ranking_true = np.argsort(labels)[::-1]

    top_k = min(15, n)
    top_pred_nodes = [int(nodelist[idx]) for idx in ranking_pred[:top_k]]
    top_true_nodes = [int(nodelist[idx]) for idx in ranking_true[:top_k]]

    print(f"\nTop-{top_k} Predicted key nodes: {top_pred_nodes}")
    print(f"Top-{top_k} Ground Truth (SIR) nodes: {top_true_nodes}")

    overlap = set(top_pred_nodes).intersection(set(top_true_nodes))
    print(f"Number of overlapping top key nodes: {len(overlap)} / {top_k}")

if __name__ == "__main__":
    main()
