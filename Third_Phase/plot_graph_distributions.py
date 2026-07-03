import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Path configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "Datasets"))
results_dir = os.path.abspath(os.path.join(script_dir, "results"))
os.makedirs(results_dir, exist_ok=True)

# Training networks
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

# Test networks
test_datasets = [
    "Test/mammalia-voles-bhp-trapping.edges",
    "Test/facebook_combined.txt",
    "Test/karate.txt",
    "Test/dolphins.txt",
    "Test/NewSpain_18c_travelmap.txt",
    "Test/synthetic_test_realworld.txt"
]

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

def get_lcc_subgraph(G_raw):
    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    if not components:
        return G_raw
    return G_raw.subgraph(components[0]).copy()

def main():
    all_networks = []
    
    print("Collecting network statistics...")
    
    # 1. Collect Train Networks
    for filename in training_datasets:
        filepath = os.path.join(datasets_dir, filename)
        if not os.path.exists(filepath):
            continue
        G_raw = load_graph(filepath)
        G = get_lcc_subgraph(G_raw)
        all_networks.append({
            "name": os.path.basename(filename),
            "type": "Train",
            "G": G
        })

    # 2. Collect Test Networks
    for filename in test_datasets:
        filepath = os.path.join(datasets_dir, filename)
        if not os.path.exists(filepath):
            continue
        G_raw = load_graph(filepath)
        G = get_lcc_subgraph(G_raw)
        all_networks.append({
            "name": os.path.basename(filename),
            "type": "Test",
            "G": G
        })

    # Print structural characteristics table
    print("\n" + "="*115)
    print(f"{'Network Name':<30} | {'Type':<6} | {'Nodes (N)':<10} | {'Edges (E)':<10} | {'Density':<8} | {'Avg Deg':<8} | {'Max Deg':<8} | {'Clustering':<10}")
    print("="*115)
    
    for net in all_networks:
        name = net["name"]
        G = net["G"]
        ntype = net["type"]
        
        n = G.number_of_nodes()
        e = G.number_of_edges()
        density = nx.density(G)
        degrees = [d for n, d in G.degree()]
        avg_deg = np.mean(degrees)
        max_deg = np.max(degrees)
        avg_clustering = nx.average_clustering(G)
        
        print(f"{name:<30} | {ntype:<6} | {n:<10} | {e:<10} | {density:<8.4f} | {avg_deg:<8.2f} | {max_deg:<8} | {avg_clustering:<10.4f}")
    print("="*115)

    # Plot degree distributions
    print("\nGenerating degree distribution plots...")
    num_nets = len(all_networks)
    cols = 4
    rows = (num_nets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()
    
    for i, net in enumerate(all_networks):
        name = net["name"]
        G = net["G"]
        ntype = net["type"]
        
        degrees = [G.degree(n) for n in G.nodes()]
        
        # Calculate degree histogram
        deg_counts = np.bincount(degrees)
        deg_vals = np.arange(len(deg_counts))
        # Filter out 0 counts for log plotting
        mask = deg_counts > 0
        deg_vals = deg_vals[mask]
        deg_counts = deg_counts[mask]
        
        # Determine color
        color = "teal" if ntype == "Train" else "salmon"
        
        ax = axes[i]
        ax.scatter(deg_vals, deg_counts / len(degrees), color=color, alpha=0.7, edgecolors="none", s=25)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{name} ({ntype})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Degree (k)", fontsize=8)
        ax.set_ylabel("P(k)", fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "network_degree_distributions.png")
    plt.savefig(plot_path, dpi=200)
    print(f"\nSuccessfully saved degree distribution grid plot to: {plot_path}")

if __name__ == "__main__":
    main()
