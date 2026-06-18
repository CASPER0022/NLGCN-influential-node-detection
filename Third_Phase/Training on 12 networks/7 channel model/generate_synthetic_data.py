import os
import random
import numpy as np
import networkx as nx

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))

def generate_skewed_fragmented_graph(num_nodes=500):
    sizes = []
    remaining = num_nodes
    while remaining > 0:
        # Sample size from geometric distribution capped at remaining nodes, minimum size 2
        size = int(np.random.geometric(p=0.08))
        size = max(2, min(size, remaining))
        sizes.append(size)
        remaining -= size
        if remaining <= 2:
            if remaining > 0:
                sizes[-1] += remaining
            break
            
    # Print statistics of generated components
    print(f"Generated {len(sizes)} components with sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.2f}")
    
    graphs = []
    node_offset = 0
    for size in sizes:
        # Erdős-Rényi random graph (p selected to keep it reasonably connected but sparse)
        # avg degree approx p * (size - 1)
        p = min(0.3, max(0.08, 4.0 / (size - 1))) if size > 1 else 0.1
        g = nx.fast_gnp_random_graph(size, p, seed=42)
        
        # Relabel nodes to have unique integer IDs
        mapping = {i: i + node_offset for i in range(size)}
        g = nx.relabel_nodes(g, mapping)
        graphs.append(g)
        node_offset += size
        
    G = nx.union_all(graphs)
    return G

def save_graph_edges(G, filepath):
    with open(filepath, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

def main():
    os.makedirs(datasets_dir, exist_ok=True)
    
    # 1. Generate synthetic fragmented graph 1 (N = 500)
    print("\nGenerating synthetic_fragmented_1...")
    G1 = generate_skewed_fragmented_graph(500)
    save_graph_edges(G1, os.path.join(datasets_dir, "synthetic_fragmented_1.txt"))
    print(f"Saved synthetic_fragmented_1.txt (Nodes: {G1.number_of_nodes()}, Edges: {G1.number_of_edges()})")
    
    # 2. Generate synthetic fragmented graph 2 (N = 1000)
    print("\nGenerating synthetic_fragmented_2...")
    G2 = generate_skewed_fragmented_graph(1000)
    save_graph_edges(G2, os.path.join(datasets_dir, "synthetic_fragmented_2.txt"))
    print(f"Saved synthetic_fragmented_2.txt (Nodes: {G2.number_of_nodes()}, Edges: {G2.number_of_edges()})")

if __name__ == "__main__":
    main()
