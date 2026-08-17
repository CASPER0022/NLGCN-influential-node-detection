import os
import sys
import random
import numpy as np
import networkx as nx
import torch
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Directory Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Datasets folders
datasets_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
train_folder = os.path.join(datasets_base_dir, "weighted Datasets", "train")
test_folder = os.path.join(datasets_base_dir, "weighted Datasets", "test")

# Import the BBV generation algorithm
sys.path.append(script_dir)
from generate_weighted_synthetic_sf import generate_bbv_graph
# Import helper functions from train_wnlgcn_multigraph
from train_wnlgcn_multigraph import process_dataset, load_weighted_graph

def save_bbv_graph(G, path):
    with open(path, 'w') as f:
        for u, v, d in G.edges(data=True):
            f.write(f"{u} {v} {d.get('weight', 1.0):.6f}\n")

def compute_and_save_centralities(filepath, filename, results_dir):
    # Load LCC to match GNN features LCC filtering
    G_raw, _ = load_weighted_graph(filepath, "positive")
    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    G = G_raw.subgraph(components[0]).copy()
    nodelist = list(G.nodes())

    cache_wdeg = os.path.join(results_dir, f"{filename}_wdeg.npy")
    cache_wclos = os.path.join(results_dir, f"{filename}_wclos.npy")
    cache_wbet = os.path.join(results_dir, f"{filename}_wbet.npy")
    cache_weig = os.path.join(results_dir, f"{filename}_weig.npy")

    # 1. Degree Strength
    wdeg_dict = dict(G.degree(weight='weight'))
    wdeg_vals = np.array([wdeg_dict.get(n, 1.0) for n in nodelist])
    np.save(cache_wdeg, wdeg_vals)

    # 2. Closeness and Betweenness
    G_dist = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        G_dist.add_edge(u, v, distance=1.0 / w)

    wclos_dict = nx.closeness_centrality(G_dist, distance='distance')
    wclos_vals = np.array([wclos_dict[n] for n in nodelist])
    np.save(cache_wclos, wclos_vals)

    wbet_dict = nx.betweenness_centrality(G_dist, weight='distance', normalized=True)
    wbet_vals = np.array([wbet_dict[n] for n in nodelist])
    np.save(cache_wbet, wbet_vals)

    # 3. Weighted Eigenvector Centrality
    try:
        weig_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
        weig_vals = np.array([weig_dict[n] for n in nodelist])
    except Exception:
        weig_vals = np.zeros(len(nodelist))
    np.save(cache_weig, weig_vals)
    print(f"  -> Computed and saved centrality metrics for {filename}")

def main():
    # 7 train graph sizes (between 200 and 3000)
    train_sizes = [350, 650, 1100, 1600, 2100, 2600, 2900]
    # 10 test graph sizes (between 200 and 3000)
    test_sizes = [220, 450, 750, 950, 1300, 1750, 2250, 2450, 2750, 2950]

    print("Generating and processing 7 Train scale-free networks...")
    for size in train_sizes:
        filename = f"synthetic_sf_{size}.txt"
        filepath = os.path.join(train_folder, filename)
        
        # 1. Generate and save graph
        G = generate_bbv_graph(n=size, m=3, delta=1.5)
        save_bbv_graph(G, filepath)
        print(f"\nSaved train graph {filename} ({size} nodes)")

        # 2. Run simulation and precompute features
        cache_x_path = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_weighted_y.npy")
        
        X, y = process_dataset(filepath, filename, "positive")
        np.save(cache_x_path, X)
        np.save(cache_y_path, y)
        print(f"  -> Cached features and ground-truth SIR labels.")

    print("\nGenerating and processing 10 Test scale-free networks...")
    for size in test_sizes:
        filename = f"synthetic_test_sf_{size}.txt"
        filepath = os.path.join(test_folder, filename)
        
        # 1. Generate and save graph
        G = generate_bbv_graph(n=size, m=3, delta=1.5)
        save_bbv_graph(G, filepath)
        print(f"\nSaved test graph {filename} ({size} nodes)")

        # 2. Run simulation and precompute features
        cache_x_path = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_weighted_y.npy")
        
        X, y = process_dataset(filepath, filename, "positive")
        np.save(cache_x_path, X)
        np.save(cache_y_path, y)
        print(f"  -> Cached features and ground-truth SIR labels.")

        # 3. Precompute centralities
        compute_and_save_centralities(filepath, filename, results_dir)

    print("\nAll new datasets generated and cached successfully!")

if __name__ == "__main__":
    main()
