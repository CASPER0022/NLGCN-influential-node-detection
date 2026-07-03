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
from scipy.stats import kendalltau
from joblib import Parallel, delayed

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Path resolution relative to script file
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))

# 14 Training networks
training_datasets = [
    "cypedge.txt",
    "carrib.txt",
    "C_elegans.txt",
    "Budapest.txt",
    "US_airports.txt",
    "Human12a.edge",
    "synthetic_realworld_1.txt",
    "cargoshipsBB.txt",
    "synthetic_realworld_2.txt",
    "E.coli.edge",
    "netscience.mtx",
    "open_flights.txt",
    "out.advogato",
    "out.foldoc"
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

# ---- Process a Single Graph Dataset (Filtering to LCC only) ----
def process_dataset(filepath, filename, L=40, precalculated_y=None):
    print(f"\nProcessing {filename}...")
    G_raw = load_graph(filepath)
    
    # Extract Largest Connected Component (LCC)
    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    if not components:
        raise ValueError(f"No connected components found in {filename}")
        
    lcc_nodes = components[0]
    G = G_raw.subgraph(lcc_nodes).copy()
    
    nodelist = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodelist)}
    n = len(nodelist)
    deg = np.array([G.degree(node) for node in nodelist])
    print(f"  -> LCC Nodes: {n} (out of {G_raw.number_of_nodes()}) | LCC Edges: {G.number_of_edges()}")

    # 1. Distance Matrix (LCC only)
    print("  -> Calculating shortest paths within LCC...")
    dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.zeros((n, n))
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i != j and v in dist[u]:
                dist_matrix[i, j] = dist[u][v]

    # 2. Global Influence (NGI)
    print("  -> Computing Global Influence (NGI)...")
    alpha = 0.5
    NGI = np.zeros(n)
    for i in range(n):
        dists = dist_matrix[i]
        mask = (dists > 0)
        if np.any(mask):
            NGI[i] = np.sum(np.sqrt(deg[mask] + alpha) / dists[mask])

    # 3. Local Influence (NLI)
    print("  -> Computing Local Influence (NLI)...")
    K_hop = 3
    NLI = np.zeros(n)
    for i in range(n):
        hop_count = np.sum((dist_matrix[i] >= 1) & (dist_matrix[i] <= K_hop))
        if hop_count > 0:
            NLI[i] = (deg[i] * np.log10(hop_count)) / n

    # 4. Vectorized Multi-scale Centrality Weight Generation
    print("  -> Performing vectorized multi-scale aggregation...")
    A = nx.to_numpy_array(G, nodelist=nodelist)
    W_NLI1 = NLI.copy()
    W_NLI2 = W_NLI1 + A.dot(W_NLI1)
    W_NLI3 = W_NLI2 + A.dot(W_NLI2)

    W_NGI1 = NGI.copy()
    W_NGI2 = W_NGI1 + A.dot(W_NGI1)
    W_NGI3 = W_NGI2 + A.dot(W_NGI2)

    # Convert features to dictionaries for lookup (with log10(x + 1) scaling to control skewed distributions)
    NLI_dict = {node: np.log10(NLI[i] + 1) for i, node in enumerate(nodelist)}
    W_NLI2_dict = {node: np.log10(W_NLI2[i] + 1) for i, node in enumerate(nodelist)}
    W_NLI3_dict = {node: np.log10(W_NLI3[i] + 1) for i, node in enumerate(nodelist)}

    NGI_dict = {node: np.log10(NGI[i] + 1) for i, node in enumerate(nodelist)}
    W_NGI2_dict = {node: np.log10(W_NGI2[i] + 1) for i, node in enumerate(nodelist)}
    W_NGI3_dict = {node: np.log10(W_NGI3[i] + 1) for i, node in enumerate(nodelist)}

    # 5. Parallel SIR Simulations (Label Generation) - LCC only
    if precalculated_y is not None:
        print("  -> Using cached SIR Simulation labels...")
        labels = precalculated_y
    else:
        print("  -> Running parallel SIR Simulations...")
        k_avg = np.mean(deg)
        k2_avg = np.mean(deg**2)
        beta_c = k_avg / (k2_avg - k_avg) if (k2_avg - k_avg) != 0 else 0.1
        beta = 1.5 * beta_c
        mu = 1.0

        # Leverage joblib for parallel execution of node simulations
        results = Parallel(n_jobs=-1)(
            delayed(single_node_sir)(G, node, beta, mu, runs=500) 
            for node in nodelist
        )
        labels = np.array(results)
        
        # Normalize labels
        if np.max(labels) > 0:
            labels = labels / np.max(labels)

    # 6. Neighborhood matrix extraction (6 channels)
    print("  -> Generating neighborhood channels (6 channels)...")
    channels = []
    for node in nodelist:
        nbrs = list(G.neighbors(node))
        # sort neighbors using raw W_NLI3 importance score (ranking is identical using log-scale)
        nbrs_sorted = sorted(nbrs, key=lambda x: W_NLI3[node_index[x]], reverse=True)
        nbrs_selected = nbrs_sorted[:L]

        # Pad node lists if neighbors < L
        nodes = [node] + nbrs_selected
        if len(nodes) < L + 1:
            nodes += [None] * (L + 1 - len(nodes))

        # Adjacency matrix of selection
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

    channels = np.array(channels)
    
    # ---- LOCAL PER-GRAPH NORMALIZATION ----
    # Normalize features locally per graph (Instance Normalization)
    X_mean = channels.mean(axis=(0, 2, 3), keepdims=True)
    X_std = channels.std(axis=(0, 2, 3), keepdims=True)
    channels = (channels - X_mean) / (X_std + 1e-6)

    return channels, labels


# ---- Main Execution Pipeline ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    all_X = []
    all_y = []

    # Process each training dataset
    start_time = time.time()
    for filename in training_datasets:
        filepath = os.path.join(datasets_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found at {filepath}, skipping.")
            continue
        
        # Note: We now save the local-normalized features to local-normalized cache files
        cache_x_path = os.path.join(results_dir, f"{filename}_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_y.npy")
        
        try:
            if os.path.exists(cache_x_path) and os.path.exists(cache_y_path):
                print(f"Loading cached local-normalized features and SIR labels for {filename}...")
                X = np.load(cache_x_path)
                y = np.load(cache_y_path)
            elif os.path.exists(cache_y_path):
                print(f"Loading cached SIR labels for {filename} and regenerating local-normalized features...")
                y = np.load(cache_y_path)
                X, _ = process_dataset(filepath, filename, precalculated_y=y)
                np.save(cache_x_path, X)
                print(f"Regenerated local-normalized features saved to {cache_x_path}.")
            else:
                X, y = process_dataset(filepath, filename)
                np.save(cache_x_path, X)
                np.save(cache_y_path, y)
                print(f"Cached local-normalized features and SIR labels for {filename} saved to {results_dir}.")
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_X:
        print("No datasets were successfully processed. Exiting.")
        sys.exit(1)

    # Combine data from all datasets (all are already normalized locally!)
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # Normalize labels globally
    y_train = y_train.reshape(-1, 1)
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / (y_std + 1e-6)

    # Save label statistics
    np.save(os.path.join(script_dir, "y_mean.npy"), y_mean)
    np.save(os.path.join(script_dir, "y_std.npy"), y_std)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create PyTorch DataLoader for mini-batching
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"\nCompleted data preparation in {time.time() - start_time:.2f}s")
    print(f"Total training samples: {X_tensor.shape[0]}")

    # Initialize model, optimizer, loss
    model = NLGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # ---- Training GNN model ----
    epochs = 300
    print("\nStarting GNN training loop...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(loader.dataset)
        
        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            eta = (elapsed / epoch) * (epochs - epoch) if epoch > 0 else 0
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.6f} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    # ---- Save the trained model weights ----
    model_save_path = os.path.join(script_dir, "nlgcn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nSuccessfully saved trained model weights to: {model_save_path}")

if __name__ == "__main__":
    main()
