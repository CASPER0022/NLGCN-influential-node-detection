import os
import random
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def generate_bbv_graph(n, m, delta, w0=1.0):
    """
    Generates a realistic weighted scale-free network using the 
    Barrat-Barthélemy-Vespignani (BBV) model (PNAS 2004).
    
    Parameters:
    - n: Total number of nodes in the generated graph.
    - m: Number of edges a new node attaches to existing nodes (m <= m0).
    - delta: Reinforcement factor (weight increase allocated to existing edges of target hubs).
             delta = 0 returns a standard unweighted-like network structure with constant weights.
    - w0: Initial weight of new edges (default: 1.0).
    
    Returns:
    - G: NetworkX Graph with 'weight' attribute on every edge.
    """
    # Start with a complete graph of m + 1 nodes with initial edge weight w0
    m0 = m + 1
    G = nx.complete_graph(m0)
    for u, v in G.edges():
        G[u][v]['weight'] = w0
        
    # Helper to compute strength of all nodes
    def get_strengths():
        strengths = {}
        for u in G.nodes():
            strengths[u] = sum(G[u][v]['weight'] for v in G.neighbors(u))
        return strengths

    # Iteratively add nodes from m0 to n-1
    for new_node in range(m0, n):
        strengths = get_strengths()
        total_strength = sum(strengths.values())
        
        # Choose m target nodes to attach to using strength-based preferential attachment
        nodes = list(G.nodes())
        probs = [strengths[node] / total_strength for node in nodes]
        
        # Select m distinct targets
        targets = np.random.choice(nodes, size=m, replace=False, p=probs)
        
        # Add new node
        G.add_node(new_node)
        
        # For each target node i, connect new_node to i and reinforce i's neighbors
        for i in targets:
            # 1. Add new weighted edge (new_node, i)
            G.add_edge(new_node, i, weight=w0)
            
            # 2. Reinforce target i's existing connections
            # Increment weight of connection (i, j) by delta * (w_ij / s_i)
            neighbors_i = [nbr for nbr in G.neighbors(i) if nbr != new_node]
            s_i = strengths[i]
            
            if s_i > 0 and len(neighbors_i) > 0 and delta > 0:
                for j in neighbors_i:
                    w_ij = G[i][j]['weight']
                    G[i][j]['weight'] = w_ij + delta * (w_ij / s_i)
                    
    return G

def power_law_func(k, C, beta):
    return C * (k ** beta)

def diagnostic_report(G, name="Synthetic BBV Graph"):
    """
    Computes key structural and weighted properties of the network
    to verify scale-free and strength-degree coupling behavior.
    """
    nodes = list(G.nodes())
    degrees = np.array([G.degree(node) for node in nodes])
    strengths = np.array([G.degree(node, weight='weight') for node in nodes])
    
    # 1. Strength-Degree Coupling: fit s(k) ~ C * k^beta
    # Group strengths by degree to get average strength s(k) for each degree k
    unique_k = np.unique(degrees)
    avg_s_k = []
    for k in unique_k:
        avg_s_k.append(np.mean(strengths[degrees == k]))
    avg_s_k = np.array(avg_s_k)
    
    beta = 1.0
    try:
        if len(unique_k) > 2:
            popt, _ = curve_fit(power_law_func, unique_k, avg_s_k, p0=[1.0, 1.2])
            beta = popt[1]
    except Exception:
        # Fallback log-log linear fit
        try:
            params = np.polyfit(np.log(unique_k), np.log(avg_s_k), 1)
            beta = params[0]
        except Exception:
            pass
            
    # 2. Weight dispersion
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    weight_min = min(weights) if weights else 0
    weight_max = max(weights) if weights else 0
    weight_spread = weight_max / weight_min if weight_min > 0 else 1.0
    
    # 3. Average Clustering Coefficient
    clustering = nx.average_clustering(G, weight='weight')
    
    print(f"--- Diagnostic Report for {name} ---")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    print(f"Strength-Degree Coupling Exponent (beta): {beta:.3f}")
    print(f"Weight Range: [{weight_min:.4f}, {weight_max:.4f}] (Spread: {weight_spread:.1f}x)")
    print(f"Weighted Clustering Coefficient: {clustering:.4f}")
    print("-" * 50)
    
    return beta, weight_spread, clustering

if __name__ == "__main__":
    # Example usage / testing generating a BBV synthetic network
    print("Generating a test BBV weighted synthetic network...")
    G = generate_bbv_graph(n=1000, m=3, delta=1.5)
    diagnostic_report(G, "BBV Test (n=1000, m=3, delta=1.5)")
