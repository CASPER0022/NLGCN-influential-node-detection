import os
import numpy as np
import networkx as nx

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))


# List of all 12 dataset files in the Datasets directory
dataset_files = [
    "Budapest.txt",
    "C_elegans.txt",
    "E.coli.edge",
    "Human12a.edge",
    "US_airports.txt",
    "cargoshipsBB.txt",
    "carrib.txt",
    "cypedge.txt",
    "netscience.mtx",
    "open_flights.txt",
    "out.advogato",
    "out.foldoc"
]

def load_graph(path):
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

print(f"{'Dataset':<20} | {'Nodes (|N|)':<12} | {'Edges (|M|)':<12} | {'Avg Degree (<K>)':<16} | {'Max Degree':<10} | {'Avg CC (<CC>)':<14} | {'Density':<10}")
print("-" * 105)

for filename in dataset_files:
    filepath = os.path.join(datasets_dir, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filename} not found at {filepath}")
        continue
    
    try:
        G = load_graph(filepath)
        
        nodes_count = G.number_of_nodes()
        edges_count = G.number_of_edges()
        
        # Calculate degrees
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = np.max(degrees) if degrees else 0
        
        # Calculate clustering coefficient
        avg_cc = nx.average_clustering(G)
        
        # Calculate density
        density = nx.density(G)
        
        print(f"{filename:<20} | {nodes_count:<12} | {edges_count:<12} | {avg_degree:<16.4f} | {max_degree:<10} | {avg_cc:<14.4f} | {density:<10.4f}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
