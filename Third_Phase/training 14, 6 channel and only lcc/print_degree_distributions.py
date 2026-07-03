import os
import numpy as np
import networkx as nx

# Paths setup
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
test_dir = os.path.join(datasets_dir, "Test")

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

# 2 Test networks (Mammalia and Facebook)
# (Budapest is evaluated in test_nlgcn_budapest.py too, but it's a training network)
test_datasets = [
    os.path.join("Test", "mammalia-voles-bhp-trapping.edges"),
    os.path.join("Test", "facebook_combined.txt")
]

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

def analyze_degree_distribution():
    print("="*110)
    print(f"{'Dataset Group & Name':<42} | {'Nodes (LCC)':<12} | {'Min Deg':<8} | {'Max Deg':<8} | {'Mean Deg':<10} | {'Std Deg':<10}")
    print("="*110)

    # Helper function to print stats of a path
    def process_and_print(filepath, name, group):
        if not os.path.exists(filepath):
            print(f"{f'[{group}] {name}':<42} | {'FILE NOT FOUND':<64}")
            return
            
        G_raw = load_graph(filepath)
        components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
        if not components:
            return
        G = G_raw.subgraph(components[0]).copy() # Filter to LCC only
        
        degrees = [G.degree(n) for n in G.nodes()]
        n = G.number_of_nodes()
        
        min_deg = np.min(degrees)
        max_deg = np.max(degrees)
        mean_deg = np.mean(degrees)
        std_deg = np.std(degrees)
        
        print(f"{f'[{group}] {name}':<42} | {n:<12} | {min_deg:<8} | {max_deg:<8} | {mean_deg:<10.2f} | {std_deg:<10.2f}")

    # Process training datasets
    for filename in training_datasets:
        filepath = os.path.join(datasets_dir, filename)
        process_and_print(filepath, filename, "TRAIN")

    print("-"*110)
    
    # Process test datasets
    for filename in test_datasets:
        filepath = os.path.join(datasets_dir, filename)
        name = os.path.basename(filename)
        process_and_print(filepath, name, "TEST")
        
    print("="*110)

if __name__ == "__main__":
    analyze_degree_distribution()
