import os
import numpy as np
import networkx as nx

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))


# List of all dataset files to check (including unseen test networks)
datasets_to_check = [
    ("Budapest.txt", False),
    ("C_elegans.txt", False),
    ("E.coli.edge", False),
    ("Human12a.edge", False),
    ("US_airports.txt", False),
    ("cargoshipsBB.txt", False),
    ("carrib.txt", False),
    ("cypedge.txt", False),
    ("netscience.mtx", False),
    ("open_flights.txt", False),
    ("out.advogato", False),
    ("out.foldoc", False),
    ("NewSpain_18c_travelmap.gml", True),         # Unseen GML test network
    ("mammalia-voles-bhp-trapping.edges", True)     # Unseen Edges test network
]

def load_graph(path):
    if path.endswith(".gml"):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return nx.parse_gml(lines, label='id')
        
    G = nx.Graph()
    try:
        # Try to load as integer values first
        edges = np.loadtxt(path, dtype=int, usecols=(0, 1))
    except Exception:
        # Handle files with string labels (like 'V1', 'V2') in carrib/cypedge
        edges_str = np.loadtxt(path, dtype=str, usecols=(0, 1))
        cleaned = []
        for row in edges_str:
            u_str = row[0].replace('V', '')
            v_str = row[1].replace('V', '')
            cleaned.append([int(u_str), int(v_str)])
        edges = np.array(cleaned)
    
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    G.add_edges_from(edges)
    return G

print(f"{'Dataset':<35} | {'Nodes (|N|)':<12} | {'Edges (|M|)':<12} | {'Avg Deg (<K>)':<13} | {'Max Deg':<8} | {'Avg CC':<8} | {'Density':<8} | {'Hetero (kappa)':<15} | {'Components':<10}")
print("-" * 145)

for filename, is_test in datasets_to_check:
    if is_test:
        filepath = os.path.join(datasets_dir, "Test", filename)
    else:
        filepath = os.path.join(datasets_dir, filename)
        
    if not os.path.exists(filepath):
        print(f"Error: {filename} not found at {filepath}")
        continue
    
    try:
        G = load_graph(filepath)
        
        nodes_count = G.number_of_nodes()
        edges_count = G.number_of_edges()
        
        # Calculate degrees
        degrees = np.array([d for n, d in G.degree()])
        avg_degree = np.mean(degrees) if len(degrees) > 0 else 0
        max_degree = np.max(degrees) if len(degrees) > 0 else 0
        
        # Calculate degree heterogeneity (kappa = <k^2> / <k>)
        avg_k2 = np.mean(degrees**2) if len(degrees) > 0 else 0
        kappa = avg_k2 / avg_degree if avg_degree > 0 else 0
        
        # Calculate clustering coefficient
        avg_cc = nx.average_clustering(G)
        
        # Calculate density
        density = nx.density(G)
        
        # Calculate connected components
        num_components = nx.number_connected_components(G)
        
        # Mark test sets with a star for visibility
        display_name = f"{filename} *" if is_test else filename
        print(f"{display_name:<35} | {nodes_count:<12} | {edges_count:<12} | {avg_degree:<13.4f} | {max_degree:<8} | {avg_cc:<8.4f} | {density:<8.4f} | {kappa:<15.4f} | {num_components:<10}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
