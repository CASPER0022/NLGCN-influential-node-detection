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

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Get current script path for robust relative execution
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "Datasets"))

training_datasets = [
    "cypedge.txt",      # 65 nodes
    "carrib.txt",       # 249 nodes
    "C_elegans.txt",    # 297 nodes
    "Budapest.txt",     # 480 nodes
    "US_airports.txt",  # 500 nodes
    "Human12a.edge",    # 501 nodes
    "synthetic_fragmented_1.txt", # 500 nodes (synthetic fragmented)
    "cargoshipsBB.txt", # 834 nodes
    "synthetic_fragmented_2.txt", # 1000 nodes (synthetic fragmented)
    "E.coli.edge",      # 1100 nodes
    "netscience.mtx",   # 1461 nodes
    "open_flights.txt", # 2939 nodes
    "out.advogato",     # 6539 nodes
    "out.foldoc"        # 13356 nodes
]


# ---- NLGCN Model definition (9 Channels) ----
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
        # Using L=40 as specified by the paper (shape: 16 * 20 * 20)
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
        cleaned = [[int(node.replace('V', '')) for node in row] for row in edges_str]
        edges = np.array(cleaned)
    
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

# ---- Process a Single Graph Dataset ----
def process_dataset(filepath, filename, L=40):
    print(f"\nProcessing {filename}...")
    G = load_graph(filepath)
    nodelist = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodelist)}
    n = len(nodelist)
    deg = np.array([G.degree(node) for node in nodelist])

    # 1. Compute Distance Matrix
    print("  -> Calculating shortest paths...")
    dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.zeros((n, n))
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i != j and v in dist[u]:
                dist_matrix[i, j] = dist[u][v]

    # 2. Vectorized NGI (Global Influence)
    print("  -> Computing Global Influence (NGI)...")
    alpha = 0.5
    NGI = np.zeros(n)
    for i in range(n):
        dists = dist_matrix[i]
        mask = (dists > 0)
        if np.any(mask):
            NGI[i] = np.sum(np.sqrt(deg[mask] + alpha) / dists[mask])

    # 3. Vectorized NLI (Local Influence)
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

    # Convert features to dictionaries for lookup
    NLI_dict = {node: NLI[i] for i, node in enumerate(nodelist)}
    NGI_dict = {node: NGI[i] for i, node in enumerate(nodelist)}

    # Compute global Closeness Centrality and Core number (normalized)
    print("  -> Computing global centralities (Closeness, Coreness)...")
    clos_cent = nx.closeness_centrality(G)
    core_cent = nx.core_number(G)
    max_core = max(core_cent.values()) if core_cent else 1

    # 5. Parallel SIR Simulations (Label Generation)
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

    # Pre-compute component sizes for all nodes
    comp_dict = {node: len(c) for c in nx.connected_components(G) for node in c}
    log_n_total = np.log10(n) if n > 1 else 1.0

    # 6. Neighborhood matrix extraction
    print("  -> Generating neighborhood channels...")
    channels = []
    for node in nodelist:
        nbrs = list(G.neighbors(node))
        # sort neighbors using W_NLI3 importance score
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

    channels = np.array(channels)
    return channels, labels


# ---- Main Execution Pipeline ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_X = []
    all_y = []

    # Process each training dataset
    start_time = time.time()
    for filename in training_datasets:
        filepath = os.path.join(datasets_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found at {filepath}, skipping.")
            continue
        
        try:
            X, y = process_dataset(filepath, filename)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_X:
        print("No datasets were successfully processed. Exiting.")
        sys.exit(1)

    # Combine data from all datasets
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # ---- Normalization per channel across all combined nodes ----
    X_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    X_std = X_train.std(axis=(0, 2, 3), keepdims=True)
    X_train = (X_train - X_mean) / (X_std + 1e-6)

    # Save training statistics for leakage-free evaluation
    np.save(os.path.join(script_dir, "X_mean.npy"), X_mean)
    np.save(os.path.join(script_dir, "X_std.npy"), X_std)

    # Normalize labels
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
            # Simple custom progress printing with ETA
            elapsed = time.time() - start_time
            eta = (elapsed / epoch) * (epochs - epoch) if epoch > 0 else 0
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.6f} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    # ---- Save the trained model weights ----
    model_save_path = os.path.join(script_dir, "nlgcn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nSuccessfully saved trained model weights to: {model_save_path}")

if __name__ == "__main__":
    main()
