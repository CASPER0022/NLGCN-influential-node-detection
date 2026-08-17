import os
import sys
import time
import random
import numpy as np
import networkx as nx

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
datasets_base = os.path.join(workspace_dir, "Datasets", "scalefree networks")

EXCLUDE_DATASETS = {"carrib.txt", "US_airports.txt", "carrib", "US_airports"}

# Set random seed for consistent sampling estimates
random.seed(42)
np.random.seed(42)

def load_graph_weighted(filepath):
    """
    Loads an edge list into a NetworkX undirected simple Graph with edge weights.
    Handles space/tab/comma delimitation, headers, comments, self-loops, and multi-edges.
    """
    G = nx.Graph()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('#') or line_str.startswith('%') or line_str.startswith('//'):
                    continue
                parts = line_str.replace(',', ' ').split()
                if len(parts) < 2:
                    continue
                
                try:
                    u_str = parts[0].replace('V', '').replace('v', '').replace('"', '').replace("'", '')
                    v_str = parts[1].replace('V', '').replace('v', '').replace('"', '').replace("'", '')
                    u = int(u_str) if u_str.lstrip('-').isdigit() else u_str
                    v = int(v_str) if v_str.lstrip('-').isdigit() else v_str
                except ValueError:
                    continue

                w = 1.0
                if len(parts) >= 3:
                    try:
                        w = float(parts[2])
                        w = abs(w) if w != 0 else 1e-6
                    except ValueError:
                        w = 1.0

                if u == v:
                    continue

                if G.has_edge(u, v):
                    G[u][v]['weight'] = max(G[u][v]['weight'], w)
                else:
                    G.add_edge(u, v, weight=w)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return G

def compute_path_metrics(G_lcc, sample_limit=500):
    """
    Computes Average Shortest Path Length (APL) and Diameter of the LCC.
    Uses exact calculation for N <= 1500, and fast sampled BFS for N > 1500.
    """
    lcc_nodes = list(G_lcc.nodes())
    n = len(lcc_nodes)
    if n <= 1:
        return 0.0, 0

    if n <= 1500:
        try:
            apl = float(nx.average_shortest_path_length(G_lcc))
            dia = int(nx.diameter(G_lcc))
            return apl, dia
        except Exception:
            pass

    # For larger graphs (N > 1500), perform sampled BFS
    sample_nodes = random.sample(lcc_nodes, min(n, sample_limit))
    total_dist_sum = 0
    total_pairs = 0
    max_dist = 0

    for source in sample_nodes:
        lengths = nx.single_source_shortest_path_length(G_lcc, source)
        for target, d in lengths.items():
            if source != target:
                total_dist_sum += d
                total_pairs += 1
                if d > max_dist:
                    max_dist = d

    apl_approx = total_dist_sum / total_pairs if total_pairs > 0 else 0.0
    return float(apl_approx), int(max_dist)

def compute_network_features(G, dataset_name):
    """
    Computes comprehensive structural and weighted network features for a graph.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0 or n_edges == 0:
        return None

    is_synth = "synthetic" in dataset_name.lower() or "sf" in dataset_name.lower()
    category = "Synthetic (BBV)" if is_synth else "Real-World"

    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees))
    max_degree = int(np.max(degrees))

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    avg_weight = float(np.mean(weights))
    min_weight = float(np.min(weights))
    max_weight = float(np.max(weights))

    strengths = [d for _, d in G.degree(weight='weight')]
    avg_strength = float(np.mean(strengths))
    max_strength = float(np.max(strengths))

    density = 2.0 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    c_unw = float(nx.average_clustering(G))
    
    try:
        c_w = float(nx.average_clustering(G, weight='weight'))
    except Exception:
        c_w = 0.0

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    num_cc = len(components)
    lcc_nodes_count = len(components[0])
    lcc_pct = (lcc_nodes_count / n_nodes) * 100.0

    G_lcc = G.subgraph(components[0]).copy()
    apl_lcc, diameter_lcc = compute_path_metrics(G_lcc)

    return {
        "Dataset": dataset_name,
        "Category": category,
        "Nodes": n_nodes,
        "Edges": n_edges,
        "Avg_Degree": avg_degree,
        "Max_Degree": max_degree,
        "Avg_Weight": avg_weight,
        "Min_Weight": min_weight,
        "Max_Weight": max_weight,
        "Avg_Strength": avg_strength,
        "Max_Strength": max_strength,
        "Density": density,
        "C_unw": c_unw,
        "C_w": c_w,
        "Num_CC": num_cc,
        "LCC_Nodes": lcc_nodes_count,
        "LCC_Pct": lcc_pct,
        "APL_LCC": apl_lcc,
        "Diameter_LCC": diameter_lcc
    }

def main():
    print(f"Scanning scale-free datasets under: {datasets_base} (excluding carrib and US_airports)")
    if not os.path.exists(datasets_base):
        print(f"Error: Directory {datasets_base} does not exist!")
        sys.exit(1)

    all_results = []
    unique_files = {}

    for root, _, files in os.walk(datasets_base):
        for fname in sorted(files):
            if fname in EXCLUDE_DATASETS:
                continue
            valid_exts = {".txt", ".edge", ".edges", ".mtx"}
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts or fname.startswith("out."):
                if fname not in unique_files:
                    unique_files[fname] = os.path.join(root, fname)

    for fname in sorted(unique_files.keys()):
        filepath = unique_files[fname]
        print(f"Processing {fname}...")
        t0 = time.time()
        G = load_graph_weighted(filepath)
        stats = compute_network_features(G, fname)
        t1 = time.time()
        
        if stats:
            print(f"  -> N={stats['Nodes']:,}, M={stats['Edges']:,}, AvgDeg={stats['Avg_Degree']:.2f}, C_unw={stats['C_unw']:.4f}, Elapsed: {t1-t0:.2f}s")
            all_results.append(stats)

    if not all_results:
        print("No valid dataset results computed.")
        return

    # Sort results: Category (Real-World then Synthetic), then Dataset Name
    all_results.sort(key=lambda x: (0 if x['Category'] == 'Real-World' else 1, x['Dataset']))

    # Print ASCII / Markdown Table to stdout
    print("\n" + "="*140)
    print("TOPOLOGICAL NETWORK FEATURES FOR SCALE-FREE DATASETS (RESEARCH PAPER FORMAT)")
    print("="*140 + "\n")

    headers = [
        "Dataset", "Category", "Nodes (N)", "Edges (M)", 
        "Avg Deg <k>", "Max Deg", "Avg Wt <w>", "Avg Str <s>", 
        "Density ρ", "C_unw", "C_w", "# CC", "LCC Nodes (%)", "APL (LCC)", "Diameter"
    ]

    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---'] * len(headers))}|")

    for r in all_results:
        row_str = (
            f"| {r['Dataset']:<25} "
            f"| {r['Category']:<15} "
            f"| {r['Nodes']:>9,d} "
            f"| {r['Edges']:>9,d} "
            f"| {r['Avg_Degree']:>11.2f} "
            f"| {r['Max_Degree']:>7d} "
            f"| {r['Avg_Weight']:>10.3f} "
            f"| {r['Avg_Strength']:>11.2f} "
            f"| {r['Density']:>9.4f} "
            f"| {r['C_unw']:>6.4f} "
            f"| {r['C_w']:>6.4f} "
            f"| {r['Num_CC']:>5d} "
            f"| {r['LCC_Nodes']:>5d} ({r['LCC_Pct']:>4.1f}%) "
            f"| {r['APL_LCC']:>9.3f} "
            f"| {r['Diameter_LCC']:>8d} |"
        )
        print(row_str)

    # Save to Markdown File
    md_path = os.path.join(script_dir, "network_features_scalefree.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Topological and Network Features of Scale-Free Datasets\n\n")
        f.write("This table summarizes the topological properties, clustering coefficients, and path metrics for scale-free datasets.\n\n")
        f.write(f"| {' | '.join(headers)} |\n")
        f.write(f"|{'|'.join(['---'] * len(headers))}|\n")
        for r in all_results:
            f.write(
                f"| {r['Dataset']} | {r['Category']} | {r['Nodes']:,} | {r['Edges']:,} "
                f"| {r['Avg_Degree']:.2f} | {r['Max_Degree']} | {r['Avg_Weight']:.3f} | {r['Avg_Strength']:.2f} "
                f"| {r['Density']:.4f} | {r['C_unw']:.4f} | {r['C_w']:.4f} | {r['Num_CC']} "
                f"| {r['LCC_Nodes']:,} ({r['LCC_Pct']:.1f}%) | {r['APL_LCC']:.3f} | {r['Diameter_LCC']} |\n"
            )
    print(f"\nSaved Markdown table to: {md_path}")

    # Save to CSV File
    csv_path = os.path.join(script_dir, "network_features_scalefree.csv")
    csv_headers = [
        "Dataset", "Category", "Nodes", "Edges", "Avg_Degree", "Max_Degree",
        "Avg_Weight", "Min_Weight", "Max_Weight", "Avg_Strength", "Max_Strength",
        "Density", "C_unw", "C_w", "Num_CC", "LCC_Nodes", "LCC_Pct", "APL_LCC", "Diameter_LCC"
    ]
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(",".join(csv_headers) + "\n")
        for r in all_results:
            row_vals = [
                r['Dataset'], r['Category'], str(r['Nodes']), str(r['Edges']),
                f"{r['Avg_Degree']:.4f}", str(r['Max_Degree']),
                f"{r['Avg_Weight']:.4f}", f"{r['Min_Weight']:.4f}", f"{r['Max_Weight']:.4f}",
                f"{r['Avg_Strength']:.4f}", f"{r['Max_Strength']:.4f}",
                f"{r['Density']:.6f}", f"{r['C_unw']:.4f}", f"{r['C_w']:.4f}",
                str(r['Num_CC']), str(r['LCC_Nodes']), f"{r['LCC_Pct']:.2f}",
                f"{r['APL_LCC']:.4f}", str(r['Diameter_LCC'])
            ]
            f.write(",".join(row_vals) + "\n")
    print(f"Saved CSV table to: {csv_path}")

    # Save to LaTeX File
    tex_path = os.path.join(script_dir, "network_features_scalefree.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% Research Paper Table: Network Features of Scale-Free Datasets\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Topological network features across benchmark scale-free datasets. $N$: Nodes, $M$: Edges, $\\langle k \\rangle$: Average Degree, $k_{max}$: Max Degree, $\\langle w \\rangle$: Average Edge Weight, $\\langle s \\rangle$: Average Node Strength, $\\rho$: Graph Density, $C_{unw}$: Unweighted Clustering, $C_{w}$: Weighted Clustering, $APL$: Average Shortest Path Length of LCC, $D$: Diameter.}\n")
        f.write("\\label{tab:network_features}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{lrrrrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & $N$ & $M$ & $\\langle k \\rangle$ & $k_{max}$ & $\\langle w \\rangle$ & $\\langle s \\rangle$ & $\\rho$ & $C_{unw}$ & $C_{w}$ & $APL$ & $D$ \\\\\n")
        f.write("\\midrule\n")
        
        for r in all_results:
            clean_name = r['Dataset'].replace('_', '\\_')
            f.write(
                f"{clean_name} & {r['Nodes']:,} & {r['Edges']:,} "
                f"& {r['Avg_Degree']:.2f} & {r['Max_Degree']} & {r['Avg_Weight']:.2f} "
                f"& {r['Avg_Strength']:.2f} & {r['Density']:.4f} & {r['C_unw']:.3f} "
                f"& {r['C_w']:.3f} & {r['APL_LCC']:.2f} & {r['Diameter_LCC']} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")
    print(f"Saved LaTeX paper table to: {tex_path}\n")

if __name__ == "__main__":
    main()
