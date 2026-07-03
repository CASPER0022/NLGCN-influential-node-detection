import os
import networkx as nx
import numpy as np

# Path configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets", "Test"))

gml_files = [
    "karate.gml",
    "dolphins.gml",
    "NewSpain_18c_travelmap.gml"
]

def convert_gml_to_txt():
    print("Starting conversion of GML files to standard edge lists...")
    for filename in gml_files:
        gml_path = os.path.join(test_dir, filename)
        txt_path = os.path.join(test_dir, filename.replace(".gml", ".txt"))
        
        if not os.path.exists(gml_path):
            print(f"Warning: {filename} not found at {gml_path}")
            continue
            
        print(f"Reading {filename}...")
        # Read the GML file using NetworkX with UTF-8 encoding
        try:
            with open(gml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            G = nx.parse_gml(content, label='id')
        except Exception as e:
            print(f"Failed to read with label='id', trying default read: {e}")
            with open(gml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            G = nx.parse_gml(content)
            
        # Convert all node labels to standard 0-indexed integers
        # to ensure np.loadtxt won't crash on string node names
        G = nx.convert_node_labels_to_integers(G, first_label=1)
        
        # Save as space-separated edge list
        edges = list(G.edges())
        edges_arr = np.array(edges)
        
        np.savetxt(txt_path, edges_arr, fmt='%d', delimiter=' ')
        print(f"Successfully converted and saved to {os.path.basename(txt_path)} (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")

if __name__ == "__main__":
    convert_gml_to_txt()
