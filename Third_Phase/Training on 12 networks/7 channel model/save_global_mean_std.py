import os
import sys
import numpy as np
import networkx as nx

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))

training_datasets = [
    "cypedge.txt",      # 65 nodes
    "carrib.txt",       # 249 nodes
    "C_elegans.txt",    # 297 nodes
    "Budapest.txt",     # 480 nodes
    "US_airports.txt",  # 500 nodes
    "Human12a.edge",    # 501 nodes
    "cargoshipsBB.txt", # 834 nodes
    "E.coli.edge",      # 1100 nodes
    "netscience.mtx",   # 1461 nodes
    "open_flights.txt", # 2939 nodes
    "out.advogato",     # 6539 nodes
    "out.foldoc"        # 13356 nodes
]

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
    return G

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

def process_dataset(filepath, filename, L=40):
    print(f"Processing {filename} to extract features...")
    G = load_graph(filepath)
    nodelist = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodelist)}
    n = len(nodelist)
    deg = np.array([G.degree(node) for node in nodelist])

    # 1. Compute Distance Matrix
    dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.zeros((n, n))
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i != j and v in dist[u]:
                dist_matrix[i, j] = dist[u][v]

    # 2. Global Influence (NGI)
    alpha = 0.5
    NGI = np.zeros(n)
    for i in range(n):
        dists = dist_matrix[i]
        mask = (dists > 0)
        if np.any(mask):
            NGI[i] = np.sum(np.sqrt(deg[mask] + alpha) / dists[mask])

    # 3. Local Influence (NLI)
    K_hop = 3
    NLI = np.zeros(n)
    for i in range(n):
        hop_count = np.sum((dist_matrix[i] >= 1) & (dist_matrix[i] <= K_hop))
        if hop_count > 0:
            NLI[i] = (deg[i] * np.log10(hop_count)) / n

    # 4. Multi-scale weight updates
    A = nx.to_numpy_array(G, nodelist=nodelist)
    W_NLI1 = NLI.copy()
    W_NLI2 = W_NLI1 + A.dot(W_NLI1)
    W_NLI3 = W_NLI2 + A.dot(W_NLI2)

    W_NGI1 = NGI.copy()
    W_NGI2 = W_NGI1 + A.dot(W_NGI1)
    W_NGI3 = W_NGI2 + A.dot(W_NGI2)

    NLI_dict = {node: NLI[i] for i, node in enumerate(nodelist)}
    NGI_dict = {node: NGI[i] for i, node in enumerate(nodelist)}

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

        c1 = embed_channel(mat, nodes, f1)
        c2 = embed_channel(mat, nodes, f2)
        c3 = embed_channel(mat, nodes, f3)
        c4 = embed_channel(mat, nodes, f4)
        c5 = embed_channel(mat, nodes, f5)
        c6 = embed_channel(mat, nodes, f6)

        tensor = np.stack([c1, c2, c3, c4, c5, c6])
        channels.append(tensor)

    return np.array(channels)

def main():
    all_X = []
    for filename in training_datasets:
        filepath = os.path.join(datasets_dir, filename)
        if not os.path.exists(filepath):
            continue
        try:
            X = process_dataset(filepath, filename)
            all_X.append(X)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    X_train = np.concatenate(all_X, axis=0)
    X_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    X_std = X_train.std(axis=(0, 2, 3), keepdims=True)

    np.save(os.path.join(script_dir, "x_mean.npy"), X_mean)
    np.save(os.path.join(script_dir, "x_std.npy"), X_std)
    print("Successfully computed and saved global mean and std features!")

if __name__ == "__main__":
    main()
