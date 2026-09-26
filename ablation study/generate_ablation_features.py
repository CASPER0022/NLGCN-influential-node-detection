"""
generate_ablation_features.py

Generates the missing 6-channel feature tensors for the data-level ablations:
  (b) no_polarity        -> suffix '_nopolarity'
  (c) unweighted_smooth  -> suffix '_unweightedsmooth'
  (d) single_smooth      -> suffix '_singlesmooth'

Saves generated feature tensors to:
  e:\Honors Project\Implementation Codes\Third_Phase\weighted models\results
alongside copied ground-truth SIR labels (_y.npy) from existing '_weighted_y.npy' files.
"""

import os
import sys
import shutil
import random
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

# Set seeds
random.seed(42)
np.random.seed(42)

# Directory Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEIGHTED_MODEL_DIR = os.path.join(BASE_DIR, "Third_Phase", "weighted models")
RESULTS_DIR = os.path.join(WEIGHTED_MODEL_DIR, "results")
DATASETS_BASE_DIR = os.path.join(BASE_DIR, "Datasets")
TRAIN_FOLDER = os.path.join(DATASETS_BASE_DIR, "weighted Datasets", "train")
TEST_FOLDER = os.path.join(DATASETS_BASE_DIR, "weighted Datasets", "test")

# Target dataset semantics
TARGET_DATASETS = {
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

MODES = {
    "no_polarity": "_nopolarity",
    "unweighted_smooth": "_unweightedsmooth",
    "single_smooth": "_singlesmooth"
}


def load_weighted_graph(path, semantics):
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
    return G


def embed_channel(mat_binary, mat_weighted, nodes, feature_dict, use_weighted_offdiag=True):
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


def process_dataset_mode(filepath, filename, mode, L=40):
    """
    Extracts features for a specific ablation mode:
      - 'no_polarity': semantics="positive", normal smoothing with W_norm
      - 'unweighted_smooth': standard semantics, smoothing with A_binary
      - 'single_smooth': standard semantics, W_NLI3 = W_NLI2, W_NGI3 = W_NGI2
    """
    semantics = "positive" if mode == "no_polarity" else TARGET_DATASETS.get(filename, "positive")
    G_raw = load_weighted_graph(filepath, semantics)

    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    if not components:
        raise ValueError(f"No connected components found in {filename}")

    lcc_nodes = components[0]
    G = G_raw.subgraph(lcc_nodes).copy()

    nodelist = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodelist)}
    n = len(nodelist)

    # Weight normalization per graph
    raw_weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    max_w = raw_weights.max() if len(raw_weights) > 0 else 1.0
    for u, v, d in G.edges(data=True):
        d['weight_norm'] = d['weight'] / max_w

    strength = np.array([G.degree(node, weight='weight_norm') for node in nodelist])

    # Weighted Distance Matrix
    G_dist = nx.Graph()
    G_dist.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        G_dist.add_edge(u, v, distance=1.0 / d['weight_norm'])

    dist = dict(nx.all_pairs_dijkstra_path_length(G_dist, weight='distance'))

    # Global Influence (NGI)
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
    K_hop = 3
    NLI = np.zeros(n)
    for i, u in enumerate(nodelist):
        u_hop_dists = hop_dist.get(u, {})
        hop_count = sum(1 for v in nodelist if u != v and v in u_hop_dists and 1 <= u_hop_dists[v] <= K_hop)
        if hop_count > 0:
            NLI[i] = (strength[i] * np.log10(hop_count)) / n

    # Multi-scale Centrality Propagation
    A_binary = nx.to_numpy_array(G, nodelist=nodelist, weight=None)
    W_norm = nx.to_numpy_array(G, nodelist=nodelist, weight='weight_norm')

    W_NLI1 = NLI.copy()
    W_NGI1 = NGI.copy()

    if mode == "unweighted_smooth":
        W_NLI2 = W_NLI1 + A_binary.dot(W_NLI1)
        W_NLI3 = W_NLI2 + A_binary.dot(W_NLI2)
        W_NGI2 = W_NGI1 + A_binary.dot(W_NGI1)
        W_NGI3 = W_NGI2 + A_binary.dot(W_NGI2)
    elif mode == "single_smooth":
        W_NLI2 = W_NLI1 + W_norm.dot(W_NLI1)
        W_NLI3 = W_NLI2.copy()
        W_NGI2 = W_NGI1 + W_norm.dot(W_NGI1)
        W_NGI3 = W_NGI2.copy()
    else:  # no_polarity or baseline
        W_NLI2 = W_NLI1 + W_norm.dot(W_NLI1)
        W_NLI3 = W_NLI2 + W_norm.dot(W_NLI2)
        W_NGI2 = W_NGI1 + W_norm.dot(W_NGI1)
        W_NGI3 = W_NGI2 + W_norm.dot(W_NGI2)

    NLI_dict = {node: NLI[i] for i, node in enumerate(nodelist)}
    W_NLI2_dict = {node: W_NLI2[i] for i, node in enumerate(nodelist)}
    W_NLI3_dict = {node: W_NLI3[i] for i, node in enumerate(nodelist)}

    NGI_dict = {node: NGI[i] for i, node in enumerate(nodelist)}
    W_NGI2_dict = {node: W_NGI2[i] for i, node in enumerate(nodelist)}
    W_NGI3_dict = {node: W_NGI3[i] for i, node in enumerate(nodelist)}

    # Neighborhood tensor construction
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

    return channels


def discover_all_valid_datasets():
    datasets = []
    for folder in [TRAIN_FOLDER, TEST_FOLDER]:
        if os.path.exists(folder):
            for f in sorted(os.listdir(folder)):
                if f in ["karate.txt", "cargoshipsBB.txt"]:
                    continue
                filepath = os.path.join(folder, f)
                weighted_x = os.path.join(RESULTS_DIR, f"{f}_weighted_local_norm_X.npy")
                weighted_y = os.path.join(RESULTS_DIR, f"{f}_weighted_y.npy")
                if os.path.exists(filepath) and os.path.exists(weighted_x) and os.path.exists(weighted_y):
                    datasets.append((f, filepath))
    return datasets


def generate_single_dataset_features(filename, filepath, mode, suffix):
    cache_x = os.path.join(RESULTS_DIR, f"{filename}{suffix}_local_norm_X.npy")
    cache_y = os.path.join(RESULTS_DIR, f"{filename}{suffix}_y.npy")
    src_y = os.path.join(RESULTS_DIR, f"{filename}_weighted_y.npy")

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f"  [EXISTS] {filename:<32} mode='{mode}' ({suffix})")
        return

    print(f"  [COMPUTING] {filename:<32} mode='{mode}'...")
    X = process_dataset_mode(filepath, filename, mode)
    np.save(cache_x, X)
    if not os.path.exists(cache_y) and os.path.exists(src_y):
        shutil.copyfile(src_y, cache_y)
    print(f"  [SAVED] {filename:<32} mode='{mode}' -> {cache_x}")


def main():
    datasets = discover_all_valid_datasets()
    print(f"Found {len(datasets)} datasets with existing baseline features.")

    for mode, suffix in MODES.items():
        print(f"\n{'='*90}\nGenerating feature cache for mode: '{mode}' (suffix: '{suffix}')\n{'='*90}")
        for filename, filepath in datasets:
            generate_single_dataset_features(filename, filepath, mode, suffix)

    print("\nFeature generation for all data-level ablations completed successfully!")


if __name__ == "__main__":
    main()
