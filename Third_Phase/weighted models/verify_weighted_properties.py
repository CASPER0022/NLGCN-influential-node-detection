import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Set path targets relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
train_dir = os.path.join(datasets_dir, "weighted Datasets", "train")

# Define helper to find dataset path recursively
def find_dataset_file(root_dir, filename):
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def load_simple_graph(path):
    """
    Loads an edge list as a weighted graph, ignoring headers and parsing standard formats.
    """
    G = nx.Graph()
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
                    if G.has_edge(u, v):
                        G[u][v]['weight'] = max(G[u][v]['weight'], w)
                    else:
                        G.add_edge(u, v, weight=w)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def power_law_fit(x, y):
    # Fits log(y) = log(C) + beta * log(x)
    params = np.polyfit(np.log(x), np.log(y), 1)
    return params[0] # Exponent beta

def analyze_network(G, name, is_synthetic=False):
    nodes = list(G.nodes())
    degrees = np.array([G.degree(node) for node in nodes])
    strengths = np.array([G.degree(node, weight='weight') for node in nodes])
    
    # Calculate s(k) vs k
    unique_k = np.unique(degrees)
    avg_s_k = []
    for k in unique_k:
        avg_s_k.append(np.mean(strengths[degrees == k]))
    avg_s_k = np.array(avg_s_k)
    
    # Exponent beta fit
    beta = 1.0
    if len(unique_k) > 2:
        try:
            beta = power_law_fit(unique_k, avg_s_k)
        except Exception:
            pass
            
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    max_w = max(weights) if len(weights) > 0 else 1.0
    norm_weights = [w / max_w for w in weights]
    return {
        "name": name,
        "degrees": degrees,
        "strengths": strengths,
        "unique_k": unique_k,
        "avg_s_k": avg_s_k,
        "beta": beta,
        "weights": norm_weights,
        "is_synthetic": is_synthetic
    }

def main():
    # 1. Identify representative real-world networks (6 networks)
    real_files = [
        "US_airports.txt", 
        "C_elegans.txt", 
        "Budapest.txt", 
        "netscience.mtx", 
        "Human12a.edge", 
        "E.coli.edge"
    ]
    # 2. Identify representative synthetic BBV networks (3 networks)
    synthetic_files = [
        "synthetic_sf_weighted_train_250.txt",
        "synthetic_sf_weighted_train_1000.txt",
        "synthetic_sf_weighted_train_5000.txt"
    ]
    
    datasets_to_plot = []
    
    # Load real-world networks
    for filename in real_files:
        path = find_dataset_file(datasets_dir, filename)
        if path:
            print(f"Loading real-world network: {filename}...")
            G = load_simple_graph(path)
            datasets_to_plot.append(analyze_network(G, filename))
            
    # Load synthetic networks
    for filename in synthetic_files:
        syn_path = os.path.join(train_dir, filename)
        if os.path.exists(syn_path):
            print(f"Loading synthetic network: {filename}...")
            G = load_simple_graph(syn_path)
            datasets_to_plot.append(analyze_network(G, f"BBV Syn ({filename.split('_')[-1].split('.')[0]})", is_synthetic=True))
        else:
            print(f"Warning: {filename} not found at {syn_path}.")

    # Plot comparisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comparison of Real-World vs. BBV Synthetic Weighted Network Properties", fontsize=15, weight='bold', color='#1F4E79')
    
    # Subplot 1: Strength vs Degree s(k) ~ k^beta (Log-Log)
    ax = axes[0, 0]
    for data in datasets_to_plot:
        color = '#1F77B4' if not data["is_synthetic"] else '#FF7F0E'
        ax.loglog(data["unique_k"], data["avg_s_k"], 'o', label=f"{data['name']} ($\\beta$={data['beta']:.2f})", color=color, alpha=0.6)
        # Add fit line
        k_fit = np.linspace(min(data["unique_k"]), max(data["unique_k"]), 100)
        c_fit = data["avg_s_k"][0] / (data["unique_k"][0] ** data["beta"])
        s_fit = c_fit * (k_fit ** data["beta"])
        ax.loglog(k_fit, s_fit, '--', color=color, alpha=0.4)
        
    ax.set_title("Strength-Degree Coupling $s(k) \\sim k^\\beta$", fontsize=11, weight='bold', color='#2F5597')
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Average Strength $s(k)$")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Subplot 2: Degree Distribution P(k) (Log-Log)
    ax = axes[0, 1]
    for data in datasets_to_plot:
        color = '#1F77B4' if not data["is_synthetic"] else '#FF7F0E'
        k_counts = np.bincount(data["degrees"])
        k_val = np.arange(len(k_counts))
        mask = k_counts > 0
        pk = k_counts[mask] / len(data["degrees"])
        ax.loglog(k_val[mask], pk, 's', label=data["name"], color=color, alpha=0.6)
        
    ax.set_title("Degree Distribution $P(k)$", fontsize=11, weight='bold', color='#2F5597')
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Probability $P(k)$")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Subplot 3: Edge Weight Density/Distribution
    ax = axes[1, 0]
    for data in datasets_to_plot:
        color = '#1F77B4' if not data["is_synthetic"] else '#FF7F0E'
        ax.hist(data["weights"], bins=np.logspace(np.log10(min(data["weights"])+1e-5), np.log10(max(data["weights"])+1e-5), 30),
                label=data["name"], color=color, alpha=0.4, density=True, histtype='step', lw=2)
        
    ax.set_xscale('log')
    ax.set_title("Normalized Edge Weight Distribution", fontsize=11, weight='bold', color='#2F5597')
    ax.set_xlabel("Normalized Weight $w_{norm}$")
    ax.set_ylabel("Density")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Subplot 4: Cumulative Strength Distribution
    ax = axes[1, 1]
    for data in datasets_to_plot:
        color = '#1F77B4' if not data["is_synthetic"] else '#FF7F0E'
        sorted_s = np.sort(data["strengths"])
        ccdf = 1.0 - np.arange(len(sorted_s)) / len(sorted_s)
        ax.loglog(sorted_s, ccdf, label=data["name"], color=color, lw=2, alpha=0.6)
        
    ax.set_title("Cumulative Strength Distribution $P(S > s)$", fontsize=11, weight='bold', color='#2F5597')
    ax.set_xlabel("Strength $s$")
    ax.set_ylabel("CCDF")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(script_dir, "weighted_properties_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nVerification plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
