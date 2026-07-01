import os
import networkx as nx
import numpy as np

# Path configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
test_dir = os.path.join(datasets_dir, "Test")

training_datasets = [
    "cypedge.txt",
    "carrib.txt",
    "C_elegans.txt",
    "Budapest.txt",
    "US_airports.txt",
    "Human12a.edge",
    "synthetic_fragmented_1.txt",
    "cargoshipsBB.txt",
    "synthetic_fragmented_2.txt",
    "E.coli.edge",
    "netscience.mtx",
    "open_flights.txt",
    "out.advogato",
    "out.foldoc"
]

test_datasets = [
    "NewSpain_18c_travelmap.gml",
    "facebook_combined.txt",
    "mammalia-voles-bhp-trapping.edges"
]

def load_graph(path):
    if path.endswith('.gml'):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        G = nx.parse_gml(data, label='id')
        return G
    
    G = nx.Graph()
    try:
        edges = np.loadtxt(path, dtype=int, usecols=(0, 1))
    except Exception:
        try:
            edges_str = np.loadtxt(path, dtype=str, usecols=(0, 1))
            cleaned = [[int(node.replace('V', '')) for node in row] for row in edges_str]
            edges = np.array(cleaned)
        except Exception:
            edges = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            edges.append((int(parts[0]), int(parts[1])))
                        except ValueError:
                            edges.append((parts[0], parts[1]))
            edges = np.array(edges)
    
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def analyze_components(G, name):
    V = G.number_of_nodes()
    E = G.number_of_edges()
    
    # Calculate connected components
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    num_cc = len(components)
    
    lcc_size = len(components[0]) if num_cc > 0 else 0
    lcc_pct = (lcc_size / V * 100) if V > 0 else 0
    
    second_size = len(components[1]) if num_cc > 1 else 0
    second_pct = (second_size / V * 100) if V > 0 else 0
    
    print(f"{name:<35} | Nodes: {V:<5} | Edges: {E:<6} | CCs: {num_cc:<4} | LCC Size: {lcc_size:<5} ({lcc_pct:>6.2f}%) | 2nd CC Size: {second_size:<4} ({second_pct:>6.2f}%)")

print("="*125)
print(f"{'Dataset Name':<35} | {'V':<5} | {'E':<6} | {'CCs':<4} | {'LCC Size':<14} | {'2nd CC Size':<14}")
print("="*125)

print("\n--- TRAINING DATASETS ---")
for filename in training_datasets:
    filepath = os.path.join(datasets_dir, filename)
    if os.path.exists(filepath):
        try:
            G = load_graph(filepath)
            analyze_components(G, filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"{filename:<35} | NOT FOUND")

print("\n--- TEST DATASETS ---")
for filename in test_datasets:
    filepath = os.path.join(test_dir, filename)
    if os.path.exists(filepath):
        try:
            G = load_graph(filepath)
            analyze_components(G, filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"{filename:<35} | NOT FOUND")
print("="*125)
