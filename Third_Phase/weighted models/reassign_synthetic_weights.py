import os
import networkx as nx
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets", "weighted Datasets"))
train_dir = os.path.join(datasets_dir, "train")
test_dir = os.path.join(datasets_dir, "test")

def reassign_weights(directory):
    for filename in os.listdir(directory):
        if filename.startswith("synthetic_sf_") or filename.startswith("synthetic_test_sf_"):
            filepath = os.path.join(directory, filename)
            print(f"Reassigning realistic weights for: {filename}...")
            
            # Load the unweighted edge list (ignoring any existing weight column)
            G = nx.read_edgelist(filepath, nodetype=int, data=False)
            degrees = dict(G.degree())
            
            # Write out the weighted edges
            with open(filepath, "w") as f:
                for u, v in G.edges():
                    # BBV-like coupling: weight depends on degree product with realistic log-wide distribution
                    w = (degrees[u] * degrees[v])**0.4 * 10**(np.random.uniform(-3, 0))
                    f.write(f"{u} {v} {w:.6f}\n")
            print(f"  -> Finished {filename}")

if __name__ == "__main__":
    print("Starting weight assignment for synthetic datasets...")
    reassign_weights(train_dir)
    reassign_weights(test_dir)
    print("Weight assignment completed successfully!")
