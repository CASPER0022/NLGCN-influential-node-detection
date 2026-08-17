import os
import sys
import time
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Path resolution relative to script file
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets", "weighted Datasets"))
# Cache results locally inside the weighted model folder
results_dir = os.path.abspath(os.path.join(script_dir, "results"))
os.makedirs(results_dir, exist_ok=True)

# ---- Classification of Target Datasets with Edge Semantics ----
target_datasets = {
    "Budapest.txt": "adversarial",
    "US_airports.txt": "positive",
    "netscience.mtx": "positive",
    "Human12a.edge": "adversarial",
    "C_elegans.txt": "positive",
    "E.coli.edge": "adversarial",
    "cargoshipsBB.txt": "adversarial",
    "NewSpain_18c_travelmap.txt": "adversarial",
    "carrib.txt": "positive",
    "cypedge.txt": "positive",
    "open_flights.txt": "positive",
    "out.advogato": "positive",
    "out.foldoc": "positive",
    "mammalia-voles-bhp-trapping.edges": "positive",
    "dolphins.txt": "positive",
    "football.net": "positive",
    "karate.txt": "positive"
}

# Training networks: 10 real-world weighted networks + 10 Scale-Free training networks
training_datasets = [
    "Budapest.txt",
    # "US_airports.txt",
    "netscience.mtx",
    "C_elegans.txt",
    "E.coli.edge",
    "carrib.txt",
    "cypedge.txt",
    "open_flights.txt",
    "out.advogato",
    # "out.foldoc",
    "synthetic_sf_100.txt",
    "synthetic_sf_250.txt",
    "synthetic_sf_500.txt",
    "synthetic_sf_850.txt",
    "synthetic_sf_1000.txt",
    "synthetic_sf_1500.txt",
    "synthetic_sf_2000.txt",
    "synthetic_sf_2500.txt",
    "synthetic_sf_3000.txt",
    "synthetic_sf_4000.txt"
]

# ---- WNLGCN Model Definition (6 Channels, weighted variant) ----
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

# ---- Weighted Graph Loader ----
def load_weighted_graph(path, semantics):
    """
    Loads an edge list as a WEIGHTED graph with direct or inverted semantics.
    For adversarial networks, weights represent costs or distances, so we invert
    them (effective_w = 1.0 / raw_w) so that higher values always represent stronger links.
    """
    G = nx.Graph()
    try:
        data = np.loadtxt(path, dtype=str)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] >= 3:
            for row in data:
                u = int(row[0].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                v = int(row[1].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                try:
                    w = float(row[2])
                    w = abs(w) if w != 0 else 1e-6
                except ValueError:
                    w = 1.0
                
                effective_w = 1.0 / w if semantics == "adversarial" else w
                
                if G.has_edge(u, v):
                    G[u][v]['weight'] = max(G[u][v]['weight'], effective_w)
                else:
                    G.add_edge(u, v, weight=effective_w)
        else:
            for row in data:
                u = int(row[0].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                v = int(row[1].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                G.add_edge(u, v, weight=1.0)
    except Exception:
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.strip().startswith('#') or line.strip().startswith('%') or line.strip() == '':
                        continue
                    parts = line.strip().split()
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
                                pass
                        
                        effective_w = 1.0 / w if semantics == "adversarial" else w
                        
                        if G.has_edge(u, v):
                            G[u][v]['weight'] = max(G[u][v]['weight'], effective_w)
                        else:
                            G.add_edge(u, v, weight=effective_w)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, True

# ---- Optimized Channel Feature Embedding ----
def embed_channel(mat_binary, mat_weighted, nodes, feature_dict, use_weighted_offdiag=False):
    size = mat_binary.shape[0]
    out = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            u = nodes[i]
            v = nodes[j]

            if i == j:
                out[i, j] = feature_dict.get(u, 0)
            elif i == 0 and j > 0:
                if mat_binary[i, j] == 1:
                    out[i, j] = feature_dict.get(v, 0)
            elif j == 0 and i > 0:
                if mat_binary[i, j] == 1:
                    out[i, j] = feature_dict.get(u, 0)
            else:
                out[i, j] = mat_weighted[i, j] if use_weighted_offdiag else mat_binary[i, j]
    return out

# ---- Optimized Chunked Weighted SIR Simulation ----
def worker_sir_weighted(adj_dict, nodes_subset, runs=500):
    results = []
    for seed in nodes_subset:
        spread = 0
        for _ in range(runs):
            visited = {seed}
            active = [seed]
            while active:
                next_active = []
                for node in active:
                    for nbr, p in adj_dict[node]:
                        if nbr not in visited:
                            if random.random() < p:
                                visited.add(nbr)
                                next_active.append(nbr)
                active = next_active
            spread += len(visited)
        results.append(spread / runs)
    return results

# ---- Process a Single Weighted Graph Dataset (Filtering to LCC only) ----
def process_dataset(filepath, filename, semantics, L=40, precalculated_y=None):
    print(f"\nProcessing {filename}...")
    G_raw, has_weight = load_weighted_graph(filepath, semantics)
    if not has_weight:
        print(f"  -> WARNING: no weight column detected in {filename}; treating as unweighted (w=1.0).")

    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    if not components:
        raise ValueError(f"No connected components found in {filename}")

    lcc_nodes = components[0]
    G = G_raw.subgraph(lcc_nodes).copy()

    nodelist = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodelist)}
    n = len(nodelist)

    # Per-graph weight normalization
    raw_weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    max_w = raw_weights.max() if len(raw_weights) > 0 else 1.0
    for u, v, d in G.edges(data=True):
        d['weight_norm'] = d['weight'] / max_w

    strength = np.array([G.degree(node, weight='weight_norm') for node in nodelist])
    print(f"  -> LCC Nodes: {n} (out of {G_raw.number_of_nodes()}) | LCC Edges: {G.number_of_edges()}")

    # Weighted Distance Matrix
    print("  -> Calculating weighted shortest paths within LCC...")
    G_dist = nx.Graph()
    G_dist.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        G_dist.add_edge(u, v, distance=1.0 / d['weight_norm'])

    dist = dict(nx.all_pairs_dijkstra_path_length(G_dist, weight='distance'))

    # Global Influence (NGI)
    print("  -> Computing Weighted Global Influence (W-NGI)...")
    alpha = 0.5
    NGI = np.zeros(n)
    for i, u in enumerate(nodelist):
        u_dists = dist.get(u, {})
        dists_list = []
        strengths_list = []
        for j, v in enumerate(nodelist):
            if u != v and v in u_dists:
                dists_list.append(u_dists[v])
                strengths_list.append(strength[j])
        if dists_list:
            NGI[i] = np.sum(np.sqrt(np.array(strengths_list) + alpha) / np.array(dists_list))

    # Topological hop matrix
    hop_dist = dict(nx.all_pairs_shortest_path_length(G))

    # Local Influence (NLI)
    print("  -> Computing Weighted Local Influence (W-NLI)...")
    K_hop = 3
    NLI = np.zeros(n)
    for i, u in enumerate(nodelist):
        u_hop_dists = hop_dist.get(u, {})
        hop_count = sum(1 for v in nodelist if u != v and v in u_hop_dists and 1 <= u_hop_dists[v] <= K_hop)
        if hop_count > 0:
            NLI[i] = (strength[i] * np.log10(hop_count)) / n

    # Vectorized Multi-scale Centrality Weight Generation
    print("  -> Performing vectorized weighted multi-scale aggregation...")
    A_binary = nx.to_numpy_array(G, nodelist=nodelist, weight=None)
    W_norm = nx.to_numpy_array(G, nodelist=nodelist, weight='weight_norm')

    W_NLI1 = NLI.copy()
    W_NLI2 = W_NLI1 + W_norm.dot(W_NLI1)
    W_NLI3 = W_NLI2 + W_norm.dot(W_NLI2)

    W_NGI1 = NGI.copy()
    W_NGI2 = W_NGI1 + W_norm.dot(W_NGI1)
    W_NGI3 = W_NGI2 + W_norm.dot(W_NGI2)

    NLI_dict = {node: NLI[i] for i, node in enumerate(nodelist)}
    W_NLI2_dict = {node: W_NLI2[i] for i, node in enumerate(nodelist)}
    W_NLI3_dict = {node: W_NLI3[i] for i, node in enumerate(nodelist)}

    NGI_dict = {node: NGI[i] for i, node in enumerate(nodelist)}
    W_NGI2_dict = {node: W_NGI2[i] for i, node in enumerate(nodelist)}
    W_NGI3_dict = {node: W_NGI3[i] for i, node in enumerate(nodelist)}

    # Parallel Weighted SIR Simulations
    if precalculated_y is not None:
        print("  -> Using cached weighted SIR Simulation labels...")
        labels = precalculated_y
    else:
        print("  -> Running parallel WEIGHTED SIR Simulations...")
        try:
            if n > 2:
                lambda_max = eigsh(W_norm, k=1, which='LA', return_eigenvectors=False)[0]
            else:
                lambda_max = np.max(np.linalg.eigvalsh(W_norm)) if n > 0 else 1.0
        except Exception:
            lambda_max = np.max(np.linalg.eigvalsh(W_norm)) if n > 0 else 1.0

        beta_c = 1.0 / lambda_max if lambda_max > 0 else 0.1
        beta = 1.5 * beta_c
        beta = min(beta, 0.9)

        # Precompute transmission probabilities for each edge to avoid NetworkX and math overhead in parallel workers
        adj_dict = {}
        for u in G.nodes():
            adj_dict[u] = []
            for v in G.neighbors(u):
                w = G[u][v].get('weight_norm', 1.0)
                p = 1.0 - (1.0 - beta) ** w
                adj_dict[u].append((v, p))

        num_cores = os.cpu_count() or 4
        chunks = np.array_split(nodelist, num_cores)

        results_chunks = Parallel(n_jobs=num_cores)(
            delayed(worker_sir_weighted)(adj_dict, chunk, runs=500)
            for chunk in chunks
        )
        labels = np.array([val for chunk in results_chunks for val in chunk])

        if np.max(labels) > 0:
            labels = labels / np.max(labels)

    # Neighborhood matrix extraction
    print("  -> Generating weighted neighborhood channels (6 channels)...")
    channels = []
    for node in nodelist:
        nbrs = list(G.neighbors(node))
        nbrs_sorted = sorted(nbrs, key=lambda x: W_NLI3[node_index[x]], reverse=True)
        nbrs_selected = nbrs_sorted[:L]

        nodes = [node] + nbrs_selected
        if len(nodes) < L + 1:
            nodes += [None] * (L + 1 - len(nodes))

        size = L + 1
        mat_binary = np.zeros((size, size))
        mat_weighted = np.zeros((size, size))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u is not None and v is not None and G.has_edge(u, v):
                    mat_binary[i, j] = 1
                    mat_weighted[i, j] = G[u][v]['weight_norm']

        f1 = {n: NLI_dict.get(n, 0) for n in nodes}
        f2 = {n: W_NLI2_dict.get(n, 0) for n in nodes}
        f3 = {n: W_NLI3_dict.get(n, 0) for n in nodes}

        f4 = {n: NGI_dict.get(n, 0) for n in nodes}
        f5 = {n: W_NGI2_dict.get(n, 0) for n in nodes}
        f6 = {n: W_NGI3_dict.get(n, 0) for n in nodes}

        c1 = embed_channel(mat_binary, mat_weighted, nodes, f1, use_weighted_offdiag=True)
        c2 = embed_channel(mat_binary, mat_weighted, nodes, f2, use_weighted_offdiag=True)
        c3 = embed_channel(mat_binary, mat_weighted, nodes, f3, use_weighted_offdiag=True)
        c4 = embed_channel(mat_binary, mat_weighted, nodes, f4, use_weighted_offdiag=True)
        c5 = embed_channel(mat_binary, mat_weighted, nodes, f5, use_weighted_offdiag=True)
        c6 = embed_channel(mat_binary, mat_weighted, nodes, f6, use_weighted_offdiag=True)

        tensor = np.stack([c1, c2, c3, c4, c5, c6])
        channels.append(tensor)

    channels = np.array(channels)

    # Local per-graph normalization
    X_mean = channels.mean(axis=(0, 2, 3), keepdims=True)
    X_std = channels.std(axis=(0, 2, 3), keepdims=True)
    channels = (channels - X_mean) / (X_std + 1e-6)

    return channels, labels

# ---- Helper to find nested dataset files recursively ----
def find_dataset_file(root_dir, filename):
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

# ---- Pairwise Margin Ranking Loss (within-batch, i.e. within-graph) ----
def pairwise_ranking_loss(pred, y, margin=0.3):
    """
    Penalizes any pair (i, j) in the batch whose predicted order disagrees
    with the ground-truth order, by at least `margin`.
    Only meaningful within a single graph's nodes.
    """
    pred = pred.view(-1)
    y = y.view(-1)

    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)   # (B, B)
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)             # (B, B)
    sign_y = torch.sign(diff_y)

    mask = diff_y.abs() > 1e-6   # ignore (near-)tied ground-truth pairs
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    losses = F.relu(margin - sign_y * diff_pred)
    return losses[mask].mean()

# ---- Main Execution Pipeline ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_X = []
    all_y = []

    start_time = time.time()
    for filename in training_datasets:
        filepath = find_dataset_file(datasets_dir, filename)
        if not filepath or not os.path.exists(filepath):
            print(f"Warning: {filename} not found recursively inside {datasets_dir}, skipping.")
            continue

        semantics = target_datasets.get(filename, "positive")
        # Separate cache namespace from the unweighted pipeline
        cache_x_path = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_weighted_y.npy")

        try:
            if os.path.exists(cache_x_path) and os.path.exists(cache_y_path):
                print(f"Loading cached weighted features and SIR labels for {filename}...")
                X = np.load(cache_x_path)
                y = np.load(cache_y_path)
            elif os.path.exists(cache_y_path):
                print(f"Loading cached weighted SIR labels for {filename} and regenerating features...")
                y = np.load(cache_y_path)
                X, _ = process_dataset(filepath, filename, semantics, precalculated_y=y)
                np.save(cache_x_path, X)
                print(f"Regenerated weighted features saved to {cache_x_path}.")
            else:
                X, y = process_dataset(filepath, filename, semantics)
                np.save(cache_x_path, X)
                np.save(cache_y_path, y)
                print(f"Cached weighted features and SIR labels for {filename} saved to {results_dir}.")
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_X:
        print("No datasets were successfully processed. Exiting.")
        sys.exit(1)

    # Compute global training statistics for standardizing target labels
    y_all_concat = np.concatenate(all_y, axis=0)
    y_mean = y_all_concat.mean()
    y_std = y_all_concat.std()

    # Save training stat files
    np.save(os.path.join(script_dir, "y_mean_weighted.npy"), y_mean)
    np.save(os.path.join(script_dir, "y_std_weighted.npy"), y_std)

    # ---- Build one dataset (and one DataLoader) PER graph, so that ----
    # ---- pairwise ranking comparisons never cross graph boundaries. ----
    per_graph_loaders = []
    for X_i, y_i in zip(all_X, all_y):
        y_i_norm = (y_i.reshape(-1, 1) - y_mean) / (y_std + 1e-6)
        X_t = torch.tensor(X_i, dtype=torch.float32)
        y_t = torch.tensor(y_i_norm, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
        bs = min(256, len(ds))
        per_graph_loaders.append(DataLoader(ds, batch_size=bs, shuffle=True))

    print(f"\nCompleted data preparation in {time.time() - start_time:.2f}s")
    print(f"Total training samples: {sum(len(l.dataset) for l in per_graph_loaders)}")
    print(f"Training over {len(per_graph_loaders)} graphs (grouped batching)")

    model = WNLGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    mse_criterion = nn.MSELoss()

    # Hybrid loss weight: how much to weight ranking vs. MSE
    ALPHA_RANK = 1.0
    MARGIN = 0.3

    epochs = 300
    print("\nStarting Weighted GNN training loop (Hybrid MSE + Ranking Loss)...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_rank = 0.0
        n_batches = 0

        # Shuffle graph order each epoch so the model doesn't see a fixed sequence
        loader_order = list(range(len(per_graph_loaders)))
        random.shuffle(loader_order)

        for gidx in loader_order:
            for batch_X, batch_y in per_graph_loaders[gidx]:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                outputs = model(batch_X)
                mse_loss = mse_criterion(outputs, batch_y)
                rank_loss = pairwise_ranking_loss(outputs, batch_y, margin=MARGIN)
                loss = mse_loss + ALPHA_RANK * rank_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                epoch_mse += mse_loss.item() * batch_X.size(0)
                epoch_rank += rank_loss.item() * batch_X.size(0)
                n_batches += batch_X.size(0)

        epoch_loss /= n_batches
        epoch_mse /= n_batches
        epoch_rank /= n_batches

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            eta = (elapsed / epoch) * (epochs - epoch) if epoch > 0 else 0
            print(f"Epoch {epoch}/{epochs} | Total: {epoch_loss:.6f} | "
                  f"MSE: {epoch_mse:.6f} | Rank: {epoch_rank:.6f} | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    model_save_path = os.path.join(script_dir, "wnlgcn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nSuccessfully saved trained model weights to: {model_save_path}")

if __name__ == "__main__":
    main()
