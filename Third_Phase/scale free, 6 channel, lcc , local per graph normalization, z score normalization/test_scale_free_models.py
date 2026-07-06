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

# Set path targets relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_train_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets", "scalefree networks", "train"))
datasets_test_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets", "scalefree networks", "test"))
results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))

# Define list of training graphs to evaluate
train_networks = [
    {"name": "Budapest", "file": os.path.join(datasets_train_dir, "Budapest.txt")},
    {"name": "C_elegans", "file": os.path.join(datasets_train_dir, "C_elegans.txt")},
    {"name": "E.coli", "file": os.path.join(datasets_train_dir, "E.coli.edge")},
    {"name": "US_airports", "file": os.path.join(datasets_train_dir, "US_airports.txt")},
    {"name": "carrib", "file": os.path.join(datasets_train_dir, "carrib.txt")},
    {"name": "open_flights", "file": os.path.join(datasets_train_dir, "open_flights.txt")},
    {"name": "out.advogato", "file": os.path.join(datasets_train_dir, "out.advogato")},
    {"name": "out.foldoc", "file": os.path.join(datasets_train_dir, "out.foldoc")},
    {"name": "synthetic_sf_100", "file": os.path.join(datasets_train_dir, "synthetic_sf_100.txt")},
    {"name": "synthetic_sf_250", "file": os.path.join(datasets_train_dir, "synthetic_sf_250.txt")},
    {"name": "synthetic_sf_500", "file": os.path.join(datasets_train_dir, "synthetic_sf_500.txt")},
    {"name": "synthetic_sf_850", "file": os.path.join(datasets_train_dir, "synthetic_sf_850.txt")},
    {"name": "synthetic_sf_1000", "file": os.path.join(datasets_train_dir, "synthetic_sf_1000.txt")},
    {"name": "synthetic_sf_1500", "file": os.path.join(datasets_train_dir, "synthetic_sf_1500.txt")},
    {"name": "synthetic_sf_2000", "file": os.path.join(datasets_train_dir, "synthetic_sf_2000.txt")},
    {"name": "synthetic_sf_2500", "file": os.path.join(datasets_train_dir, "synthetic_sf_2500.txt")},
    {"name": "synthetic_sf_3000", "file": os.path.join(datasets_train_dir, "synthetic_sf_3000.txt")},
    {"name": "synthetic_sf_4000", "file": os.path.join(datasets_train_dir, "synthetic_sf_4000.txt")},
]

# Define list of test graphs to evaluate
test_networks = [
    {"name": "Cargoships", "file": os.path.join(datasets_test_dir, "cargoshipsBB.txt")},
    {"name": "Facebook", "file": os.path.join(datasets_test_dir, "facebook_combined.txt")},
    {"name": "Karate Club", "file": os.path.join(datasets_test_dir, "karate.txt")},
    {"name": "Synthetic Scale-Free (BA 800)", "file": os.path.join(datasets_test_dir, "synthetic_test_realworld.txt")},
    {"name": "synthetic_test_sf_250", "file": os.path.join(datasets_test_dir, "synthetic_test_sf_250.txt")},
    {"name": "synthetic_test_sf_500", "file": os.path.join(datasets_test_dir, "synthetic_test_sf_500.txt")},
    {"name": "synthetic_test_sf_1000", "file": os.path.join(datasets_test_dir, "synthetic_test_sf_1000.txt")},
    {"name": "synthetic_test_sf_2000", "file": os.path.join(datasets_test_dir, "synthetic_test_sf_2000.txt")},
    {"name": "synthetic_test_sf_5000", "file": os.path.join(datasets_test_dir, "synthetic_test_sf_5000.txt")},
]

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

def compute_raw_features(G, nodelist, node_index):
    n = len(nodelist)
    deg = np.array([G.degree(node) for node in nodelist])

    dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.zeros((n, n))
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i != j and v in dist[u]:
                dist_matrix[i, j] = dist[u][v]

    alpha = 0.5
    NGI = np.zeros(n)
    for i in range(n):
        dists = dist_matrix[i]
        mask = (dists > 0)
        if np.any(mask):
            NGI[i] = np.sum(np.sqrt(deg[mask] + alpha) / dists[mask])

    K_hop = 3
    NLI = np.zeros(n)
    for i in range(n):
        hop_count = np.sum((dist_matrix[i] >= 1) & (dist_matrix[i] <= K_hop))
        if hop_count > 0:
            NLI[i] = (deg[i] * np.log10(hop_count)) / n

    A = nx.to_numpy_array(G, nodelist=nodelist)
    W_NLI1 = NLI.copy()
    W_NLI2 = W_NLI1 + A.dot(W_NLI1)
    W_NLI3 = W_NLI2 + A.dot(W_NLI2)

    W_NGI1 = NGI.copy()
    W_NGI2 = W_NGI1 + A.dot(W_NGI1)
    W_NGI3 = W_NGI2 + A.dot(W_NGI2)

    # NO LOG transform
    NLI_dict = {node: NLI[i] for i, node in enumerate(nodelist)}
    W_NLI2_dict = {node: W_NLI2[i] for i, node in enumerate(nodelist)}
    W_NLI3_dict = {node: W_NLI3[i] for i, node in enumerate(nodelist)}

    NGI_dict = {node: NGI[i] for i, node in enumerate(nodelist)}
    W_NGI2_dict = {node: W_NGI2[i] for i, node in enumerate(nodelist)}
    W_NGI3_dict = {node: W_NGI3[i] for i, node in enumerate(nodelist)}

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

    return np.array(channels)

def get_kendall(pred, labels, pct, nodelist, deg_cent_vals, clos_cent_val, bet_cent_vals, pr_cent_vals, core_cent_vals, eig_cent_vals, has_eig):
    n = len(labels)
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

    def pointer_or_nan(v):
        return v if v is not None else np.nan

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

def evaluate_network_list(model, networks_list, title):
    print(f"\n==================== Evaluating {title} ====================")
    results = []

    for net in networks_list:
        name = net["name"]
        filepath = net["file"]
        filename = os.path.basename(filepath)

        if not os.path.exists(filepath):
            print(f"Warning: {name} not found at {filepath}, skipping.")
            continue

        print(f"Evaluating {name}...")

        # Load graph and LCC
        G_raw = load_graph(filepath)
        G = get_lcc_subgraph(G_raw)
        nodelist = list(G.nodes())
        node_index = {node: i for i, node in enumerate(nodelist)}
        n = len(nodelist)
        deg = np.array([G.degree(node) for node in nodelist])

        # Load cached labels or calculate them
        cache_y_path = os.path.join(results_dir, f"{filename}_y.npy")
        if os.path.exists(cache_y_path):
            labels = np.load(cache_y_path)
        else:
            k_avg = np.mean(deg)
            k2_avg = np.mean(deg**2)
            beta_c = k_avg / (k2_avg - k_avg) if (k2_avg - k_avg) != 0 else 0.1
            beta = 1.5 * beta_c
            mu = 1.0
            
            results_sir = Parallel(n_jobs=-1)(
                delayed(single_node_sir)(G, node, beta, mu, runs=500) 
                for node in nodelist
            )
            labels = np.array(results_sir)
            np.save(cache_y_path, labels)

        # Baseline traditional centrality algorithms
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

        # Load cached features or calculate them (Local Norm)
        cache_x_path = os.path.join(results_dir, f"{filename}_raw_local_norm_X.npy")
        if os.path.exists(cache_x_path):
            X_test = np.load(cache_x_path)
        else:
            X_raw = compute_raw_features(G, nodelist, node_index)
            X_mean = X_raw.mean(axis=(0, 2, 3), keepdims=True)
            X_std = X_raw.std(axis=(0, 2, 3), keepdims=True)
            X_test = (X_raw - X_mean) / (X_std + 1e-6)
            np.save(cache_x_path, X_test)

        # Run GNN model inference
        with torch.no_grad():
            pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

        # Compute rank correlations
        res_10 = get_kendall(pred, labels, 10, nodelist, deg_cent_vals, clos_cent_val, bet_cent_vals, pr_cent_vals, core_cent_vals, eig_cent_vals, has_eig)
        res_20 = get_kendall(pred, labels, 20, nodelist, deg_cent_vals, clos_cent_val, bet_cent_vals, pr_cent_vals, core_cent_vals, eig_cent_vals, has_eig)
        res_100 = get_kendall(pred, labels, 100, nodelist, deg_cent_vals, clos_cent_val, bet_cent_vals, pr_cent_vals, core_cent_vals, eig_cent_vals, has_eig)
        results.append((name, res_10, res_20, res_100))

    # Output markdown table
    print(f"\n### {title} Results Table")
    print("| Network | Metric / Centrality | Top 10% | Top 20% | Top 100% (All) |")
    print("| :--- | :--- | :---: | :---: | :---: |")
    for name, r10, r20, r100 in results:
        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "nan"
        print(f"| **{name}** | **Ours (NLGCN GNN)** | **{fmt(r10['GNN'])}** | **{fmt(r20['GNN'])}** | **{fmt(r100['GNN'])}** |")
        print(f"| | Degree | {fmt(r10['Degree'])} | {fmt(r20['Degree'])} | {fmt(r100['Degree'])} |")
        print(f"| | Closeness | {fmt(r10['Closeness'])} | {fmt(r20['Closeness'])} | {fmt(r100['Closeness'])} |")
        print(f"| | Betweenness | {fmt(r10['Betweenness'])} | {fmt(r20['Betweenness'])} | {fmt(r100['Betweenness'])} |")
        print(f"| | PageRank | {fmt(r10['PageRank'])} | {fmt(r20['PageRank'])} | {fmt(r100['PageRank'])} |")
        print(f"| | Coreness | {fmt(r10['Coreness'])} | {fmt(r20['Coreness'])} | {fmt(r100['Coreness'])} |")
        print(f"| | Eigenvector | {fmt(r10['Eigenvector'])} | {fmt(r20['Eigenvector'])} | {fmt(r100['Eigenvector'])} |")
        print("|---|---|---|---|---|")

def main():
    model_path = os.path.join(script_dir, "nlgcn_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Train the model first.")
        sys.exit(1)

    model = NLGCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Run evaluations on both training and test scale-free lists
    evaluate_network_list(model, train_networks, "Training Scale-Free Networks (Checking Fit)")
    evaluate_network_list(model, test_networks, "Test Scale-Free Networks (Generalization Test)")

if __name__ == "__main__":
    main()
