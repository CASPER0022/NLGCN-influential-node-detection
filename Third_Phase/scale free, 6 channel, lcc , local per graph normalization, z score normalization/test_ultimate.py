import os
import sys
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set path targets relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))
os.makedirs(results_dir, exist_ok=True)

# ---- Datasets and their Semantics ----
target_datasets = {
    "facebook_combined.txt": "positive",
    "Budapest.txt": "adversarial",
    "US_airports.txt": "adversarial",
    "netscience.mtx": "adversarial",
    "Human12a.edge": "adversarial",
    "C_elegans.txt": "adversarial",
    "E.coli.edge": "adversarial",
    "cargoshipsBB.txt": "adversarial",
    "NewSpain_18c_travelmap.txt": "adversarial",
    "carrib.txt": "positive",
    "cypedge.txt": "positive",
    "open_flights.txt": "positive",
    "out.advogato": "positive",
    "out.foldoc": "positive",
    "mammalia-voles-bhp-trapping.edges": "positive",
    "dolphins.txt": "positive",
    "football.net": "positive",
    "karate.txt": "positive"
}

# Training datasets for the Scale-Free model
TRAIN_DATASETS = {
    "Budapest.txt", "C_elegans.txt", "E.coli.edge", "US_airports.txt",
    "carrib.txt", "open_flights.txt", "out.advogato", "out.foldoc"
}

# ---- NLGCN Model Definition (6 Channels) ----
class ChannelAttention(nn.Module):
    def __init__(self, channels=6, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class NLGCN(nn.Module):
    def __init__(self):
        super(NLGCN, self).__init__()
        self.attention = ChannelAttention(6, reduction=2)
        self.conv1 = nn.Conv2d(6, 16, kernel_size=2)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 20 * 20, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.attention(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---- Graph Loader (Weighted & Unweighted versions with Semantics) ----
def load_graph_weighted(path, semantics):
    G = nx.Graph()
    try:
        data = np.loadtxt(path, dtype=str)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] >= 3:
            for row in data:
                u = int(row[0].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                v = int(row[1].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                try:
                    w = float(row[2])
                    w = abs(w) if w != 0 else 1e-6
                except ValueError:
                    w = 1.0
                
                effective_w = 1.0 / w if semantics == "adversarial" else w
                
                if G.has_edge(u, v):
                    if semantics == "adversarial":
                        G[u][v]['weight'] = min(G[u][v]['weight'], effective_w)
                    else:
                        G[u][v]['weight'] = max(G[u][v]['weight'], effective_w)
                else:
                    G.add_edge(u, v, weight=effective_w)
        else:
            for row in data:
                u = int(row[0].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                v = int(row[1].replace('V', '').replace('v', '').replace('"', '').replace("'", ''))
                G.add_edge(u, v, weight=1.0)
    except Exception:
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
                        
                        effective_w = 1.0 / w if semantics == "adversarial" else w
                        
                        if G.has_edge(u, v):
                            if semantics == "adversarial":
                                G[u][v]['weight'] = min(G[u][v]['weight'], effective_w)
                            else:
                                G[u][v]['weight'] = max(G[u][v]['weight'], effective_w)
                        else:
                            G.add_edge(u, v, weight=effective_w)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# ---- Optimized SIR Simulation ----
def SIR_simulation(G, seed, beta, mu, steps=1000):
    susceptible = set(G.nodes())
    infected = {seed}
    recovered = set()
    susceptible.remove(seed)

    for _ in range(steps):
        new_infected = set()
        new_recovered = set()

        for node in infected:
            for nbr in G.neighbors(node):
                if nbr in susceptible:
                    if random.random() < beta:
                        new_infected.add(nbr)

            if random.random() < mu:
                new_recovered.add(node)

        infected |= new_infected
        infected -= new_recovered
        recovered |= new_recovered
        susceptible -= new_infected

        if len(infected) == 0:
            break

    return len(recovered)

def single_node_sir(G, seed, beta, mu, runs=500):
    spread = 0
    for _ in range(runs):
        spread += SIR_simulation(G, seed, beta, mu)
    return spread / runs

# ---- Feature Generation ----
def embed_channel(mat, nodes, feature_dict):
    size = mat.shape[0]
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            u = nodes[i]
            v = nodes[j]
            if i == j:
                out[i, j] = feature_dict.get(u, 0)
            elif i == 0 and j > 0:
                if mat[i, j] == 1:
                    out[i, j] = feature_dict.get(v, 0)
            elif j == 0 and i > 0:
                if mat[i, j] == 1:
                    out[i, j] = feature_dict.get(u, 0)
            else:
                out[i, j] = mat[i, j]
    return out

def get_lcc_subgraph(G_raw):
    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    return G_raw.subgraph(components[0]).copy()

def compute_raw_features(G, nodelist, node_index):
    n = len(nodelist)
    deg = np.array([G.degree(node) for node in nodelist])

    dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_matrix = np.zeros((n, n))
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i != j and v in dist[u]:
                dist_matrix[i, j] = dist[u][v]

    alpha = 0.5
    NGI = np.zeros(n)
    for i in range(n):
        dists = dist_matrix[i]
        mask = (dists > 0)
        if np.any(mask):
            NGI[i] = np.sum(np.sqrt(deg[mask] + alpha) / dists[mask])

    K_hop = 3
    NLI = np.zeros(n)
    for i in range(n):
        hop_count = np.sum((dist_matrix[i] >= 1) & (dist_matrix[i] <= K_hop))
        if hop_count > 0:
            NLI[i] = (deg[i] * np.log10(hop_count)) / n

    A = nx.to_numpy_array(G, nodelist=nodelist)
    W_NLI1 = NLI.copy()
    W_NLI2 = W_NLI1 + A.dot(W_NLI1)
    W_NLI3 = W_NLI2 + A.dot(W_NLI2)

    W_NGI1 = NGI.copy()
    W_NGI2 = W_NGI1 + A.dot(W_NGI1)
    W_NGI3 = W_NGI2 + A.dot(W_NGI2)

    NLI_dict = {node: NLI[i] for i, node in enumerate(nodelist)}
    W_NLI2_dict = {node: W_NLI2[i] for i, node in enumerate(nodelist)}
    W_NLI3_dict = {node: W_NLI3[i] for i, node in enumerate(nodelist)}

    NGI_dict = {node: NGI[i] for i, node in enumerate(nodelist)}
    W_NGI2_dict = {node: W_NGI2[i] for i, node in enumerate(nodelist)}
    W_NGI3_dict = {node: W_NGI3[i] for i, node in enumerate(nodelist)}

    L = 40
    channels = []
    for node in nodelist:
        nbrs = list(G.neighbors(node))
        nbrs_sorted = sorted(nbrs, key=lambda x: W_NLI3[node_index[x]], reverse=True)
        nbrs_selected = nbrs_sorted[:L]

        nodes = [node] + nbrs_selected
        if len(nodes) < L + 1:
            nodes += [None] * (L + 1 - len(nodes))

        size = L + 1
        mat = np.zeros((size, size))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u is not None and v is not None and G.has_edge(u, v):
                    mat[i, j] = 1

        f1 = {n: NLI_dict.get(n, 0) for n in nodes}
        f2 = {n: W_NLI2_dict.get(n, 0) for n in nodes}
        f3 = {n: W_NLI3_dict.get(n, 0) for n in nodes}
        f4 = {n: NGI_dict.get(n, 0) for n in nodes}
        f5 = {n: W_NGI2_dict.get(n, 0) for n in nodes}
        f6 = {n: W_NGI3_dict.get(n, 0) for n in nodes}

        c1 = embed_channel(mat, nodes, f1)
        c2 = embed_channel(mat, nodes, f2)
        c3 = embed_channel(mat, nodes, f3)
        c4 = embed_channel(mat, nodes, f4)
        c5 = embed_channel(mat, nodes, f5)
        c6 = embed_channel(mat, nodes, f6)

        tensor = np.stack([c1, c2, c3, c4, c5, c6])
        channels.append(tensor)

    return np.array(channels)

# ---- Helper for Safe Kendall-Tau at given fractions ----
def get_subset_indices(labels, pct):
    n = len(labels)
    if pct == 100:
        return np.arange(n)
    k = max(1, int(pct / 100.0 * n))
    return np.argsort(labels)[::-1][:k]

def safe_tau(x, y, indices):
    xs = x[indices]
    ys = y[indices]
    if len(xs) < 2 or np.all(xs == xs[0]) or np.all(ys == ys[0]):
        return np.nan
    try:
        tau, _ = kendalltau(xs, ys)
        return tau if not np.isnan(tau) else np.nan
    except Exception:
        return np.nan

# ---- Collect Dataset files ----
def get_all_dataset_files(root_dir):
    valid_exts = {".txt", ".edge", ".edges", ".mtx"}
    files_list = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                full_path = os.path.abspath(os.path.join(dirpath, f))
                files_list.append(full_path)
                
    # Deduplicate based on basename
    unique_basenames = {}
    for p in sorted(files_list):
        name = os.path.basename(p)
        if name not in unique_basenames:
            unique_basenames[name] = p
            
    return list(unique_basenames.values())

# ---- Render Matplotlib Table to PDF Page with Train/Test Coloring ----
def add_table_page(pdf, title, headers, data, row_statuses):
    row_count = len(data)
    fig_width = 12
    fig_height = max(6, row_count * 0.28 + 1.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    ax.set_title(title, fontsize=12, weight='bold', pad=15)
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)
    
    for col_idx in range(len(headers)):
        cell = table[0, col_idx]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1F4E79')
        
    # Apply coloring row by row based on status
    for row_idx in range(1, row_count + 1):
        status = row_statuses[row_idx - 1]
        
        # Test row gets light green highlighting
        face_color = "#E2F0D9" if status == "test" else "#FFFFFF"
        name_cell_color = "#C6E0B4" if status == "test" else "#F2F2F2"
        
        for col_idx in range(len(headers)):
            cell = table[row_idx, col_idx]
            if col_idx == 0:
                cell.set_facecolor(name_cell_color)
                if cell.get_text().get_text() != "":
                    cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor(face_color)
        
    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close()

def main():
    model_path = os.path.join(script_dir, "nlgcn_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Train the model first.")
        sys.exit(1)

    print("Loading NLGCN model...")
    model = NLGCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    all_files = get_all_dataset_files(datasets_dir)
    print(f"Found {len(all_files)} graph files to evaluate.")

    results_model_sir = []       # Table 1: Model vs SIR
    results_cent_model = []      # Table 2: Centralities vs Model
    results_sir_cent = []        # Table 3: SIR vs Centralities
    results_wcent_model = []     # Table 4: Weighted Centralities vs Model
    results_sir_wcent = []       # Table 5: SIR vs Weighted Centralities
    results_unw_vs_w = []        # Table 6: Unweighted vs Weighted

    for idx, filepath in enumerate(sorted(all_files)):
        filename = os.path.basename(filepath)
        
        # Determine semantics and train/test status
        semantics = target_datasets.get(filename, "adversarial") # default to adversarial for synthetic
        
        # Train dataset condition for the Scale-Free model
        is_train = (filename in TRAIN_DATASETS or filename.startswith("synthetic_sf_"))
        status = "train" if is_train else "test"
        
        print(f"\n[{idx+1}/{len(all_files)}] Evaluating {filename} (Status: {status.upper()}, Semantics: {semantics})...")

        try:
            # 1. Load Graph (both weighted and unweighted) and extract LCC
            G_raw = load_graph_weighted(filepath, semantics)
            G = get_lcc_subgraph(G_raw)
            nodelist = list(G.nodes())
            node_index = {node: i for i, node in enumerate(nodelist)}
            n = len(nodelist)
            
            if n < 5:
                print(f"  Skipping {filename} (graph too small: {n} nodes).")
                continue

            deg = np.array([G.degree(node) for node in nodelist])

            # 2. Get/Compute labels (SIR spreading capacity)
            cache_y_path = os.path.join(results_dir, f"{filename}_y.npy")
            if os.path.exists(cache_y_path):
                labels = np.load(cache_y_path)
            else:
                k_avg = np.mean(deg)
                k2_avg = np.mean(deg**2)
                beta_c = k_avg / (k2_avg - k_avg) if (k2_avg - k_avg) != 0 else 0.1
                beta = 1.5 * beta_c
                mu = 1.0
                print(f"  Calculating SIR spreading capacity (n={n}, beta={beta:.4f})...")
                results_sir = Parallel(n_jobs=-1)(
                    delayed(single_node_sir)(G, node, beta, mu, runs=500) 
                    for node in nodelist
                )
                labels = np.array(results_sir)
                np.save(cache_y_path, labels)

            # 3. Get/Compute neighborhood channel features
            cache_x_path = os.path.join(results_dir, f"{filename}_raw_local_norm_X.npy")
            if os.path.exists(cache_x_path):
                X_test = np.load(cache_x_path)
            else:
                print("  Calculating neighborhood features...")
                X_raw = compute_raw_features(G, nodelist, node_index)
                X_mean = X_raw.mean(axis=(0, 2, 3), keepdims=True)
                X_std = X_raw.std(axis=(0, 2, 3), keepdims=True)
                X_test = (X_raw - X_mean) / (X_std + 1e-6)
                np.save(cache_x_path, X_test)

            # 4. Model Inference
            with torch.no_grad():
                pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

            # 5. Compute Traditional Unweighted Centrality measures
            print("  Computing traditional unweighted centralities...")
            deg_cent = nx.degree_centrality(G)
            deg_vals = np.array([deg_cent[node] for node in nodelist])
            
            clos_cent = nx.closeness_centrality(G)
            clos_vals = np.array([clos_cent[node] for node in nodelist])

            bet_cent = nx.betweenness_centrality(G)
            bet_vals = np.array([bet_cent[node] for node in nodelist])

            pr_cent = nx.pagerank(G)
            pr_vals = np.array([pr_cent[node] for node in nodelist])

            core_cent = nx.core_number(G)
            core_vals = np.array([core_cent[node] for node in nodelist])

            try:
                eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
                eig_vals = np.array([eig_cent[node] for node in nodelist])
                has_eig = True
            except Exception:
                try:
                    eig_cent = nx.eigenvector_centrality_numpy(G)
                    eig_vals = np.array([eig_cent[node] for node in nodelist])
                    has_eig = True
                except Exception:
                    eig_vals = np.zeros(n)
                    has_eig = False

            # 6. Compute Weighted Centrality measures
            print("  Computing weighted centralities...")
            wdeg_dict = dict(G.degree(weight='weight'))
            wdeg_vals = np.array([wdeg_dict.get(n, 1.0) for n in nodelist])
            
            G_dist = nx.Graph()
            for u, v, d in G.edges(data=True):
                w = d.get('weight', 1.0)
                G_dist.add_edge(u, v, distance=1.0 / w)

            wclos_dict = nx.closeness_centrality(G_dist, distance='distance')
            wclos_vals = np.array([wclos_dict[n] for n in nodelist])

            wbet_dict = nx.betweenness_centrality(G_dist, weight='distance', normalized=True)
            wbet_vals = np.array([wbet_dict[n] for n in nodelist])

            wpr_dict = nx.pagerank(G, weight='weight')
            wpr_vals = np.array([wpr_dict[n] for n in nodelist])

            try:
                weig_dict = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                weig_vals = np.array([weig_dict[n] for n in nodelist])
                has_weig = True
            except Exception:
                try:
                    weig_dict = nx.eigenvector_centrality_numpy(G, weight='weight')
                    weig_vals = np.array([weig_dict[n] for n in nodelist])
                    has_weig = True
                except Exception:
                    weig_vals = np.zeros(n)
                    has_weig = False

            idx10 = get_subset_indices(labels, 10)
            idx20 = get_subset_indices(labels, 20)
            idx100 = get_subset_indices(labels, 100)

            # ---- Populate Results ----
            results_model_sir.append((
                filename,
                safe_tau(pred, labels, idx10),
                safe_tau(pred, labels, idx20),
                safe_tau(pred, labels, idx100),
                status
            ))

            t2_data = {}
            for name_metric, vals in [("Degree", deg_vals), ("Closeness", clos_vals), 
                                      ("Betweenness", bet_vals), ("PageRank", pr_vals), 
                                      ("Coreness", core_vals), ("Eigenvector", eig_vals)]:
                t2_data[name_metric] = (
                    safe_tau(vals, pred, idx10),
                    safe_tau(vals, pred, idx20),
                    safe_tau(vals, pred, idx100)
                )
            results_cent_model.append((filename, t2_data, status))

            t3_data = {}
            for name_metric, vals in [("Degree", deg_vals), ("Closeness", clos_vals), 
                                      ("Betweenness", bet_vals), ("PageRank", pr_vals), 
                                      ("Coreness", core_vals), ("Eigenvector", eig_vals)]:
                t3_data[name_metric] = (
                    safe_tau(labels, vals, idx10),
                    safe_tau(labels, vals, idx20),
                    safe_tau(labels, vals, idx100)
                )
            results_sir_cent.append((filename, t3_data, status))

            t4_data = {}
            for name_metric, vals in [("Strength (W-Deg)", wdeg_vals), ("Weighted Closeness", wclos_vals), 
                                      ("Weighted Betweenness", wbet_vals), ("Weighted PageRank", wpr_vals), 
                                      ("Weighted Eigenvector", weig_vals)]:
                t4_data[name_metric] = (
                    safe_tau(vals, pred, idx10),
                    safe_tau(vals, pred, idx20),
                    safe_tau(vals, pred, idx100)
                )
            results_wcent_model.append((filename, t4_data, status))

            t5_data = {}
            for name_metric, vals in [("Strength (W-Deg)", wdeg_vals), ("Weighted Closeness", wclos_vals), 
                                      ("Weighted Betweenness", wbet_vals), ("Weighted PageRank", wpr_vals), 
                                      ("Weighted Eigenvector", weig_vals)]:
                t5_data[name_metric] = (
                    safe_tau(labels, vals, idx10),
                    safe_tau(labels, vals, idx20),
                    safe_tau(labels, vals, idx100)
                )
            results_sir_wcent.append((filename, t5_data, status))

            t6_data = {}
            for name_metric, (unw, w) in [("Degree vs Strength", (deg_vals, wdeg_vals)), 
                                          ("Closeness (Unw vs W)", (clos_vals, wclos_vals)), 
                                          ("Betweenness (Unw vs W)", (bet_vals, wbet_vals)), 
                                          ("PageRank (Unw vs W)", (pr_vals, wpr_vals)), 
                                          ("Eigenvector (Unw vs W)", (eig_vals, weig_vals))]:
                t6_data[name_metric] = (
                    safe_tau(unw, w, idx10),
                    safe_tau(unw, w, idx20),
                    safe_tau(unw, w, idx100)
                )
            results_unw_vs_w.append((filename, t6_data, status))

        except Exception as ex:
            print(f"Error processing {filename}: {ex}")

    def fmt(v):
        return f"{v:.4f}" if (v is not None and not np.isnan(v)) else "nan"

    # ---- GENERATE PDF REPORT ----
    pdf_report_path = os.path.join(script_dir, "ultimate_evaluation_results.pdf")
    print(f"\nGenerating PDF Evaluation Report at {pdf_report_path}...")
    
    with PdfPages(pdf_report_path) as pdf:
        # Table 1: Model vs SIR
        t1_headers = ["Network Dataset", "Top 10% Correlation", "Top 20% Correlation", "Top 100% (All)"]
        t1_rows = [[name, fmt(r10), fmt(r20), fmt(r100)] for name, r10, r20, r100, _ in results_model_sir]
        t1_statuses = [status for _, _, _, _, status in results_model_sir]
        add_table_page(pdf, "Table 1: Correlation (Kendall's Tau) Between NLGCN Model and SIR Spreading Capacity\n(Light Green Highlight = Test Dataset)", t1_headers, t1_rows, t1_statuses)
        
        # Table 2: Centralities vs Model
        t2_headers = ["Network Dataset", "Centrality Metric", "Top 10%", "Top 20%", "Top 100% (All)"]
        t2_rows = []
        t2_statuses = []
        for name, t2, status in results_cent_model:
            first = True
            for metric, (r10, r20, r100) in t2.items():
                disp_name = name if first else ""
                t2_rows.append([disp_name, metric, fmt(r10), fmt(r20), fmt(r100)])
                t2_statuses.append(status)
                first = False
        add_table_page(pdf, "Table 2: Correlation (Kendall's Tau) Between Traditional Unweighted Centrality Measures and NLGCN Model\n(Light Green Highlight = Test Dataset)", t2_headers, t2_rows, t2_statuses)

        # Table 3: SIR vs Centralities
        t3_headers = ["Network Dataset", "Centrality Metric", "Top 10%", "Top 20%", "Top 100% (All)"]
        t3_rows = []
        t3_statuses = []
        for name, t3, status in results_sir_cent:
            first = True
            for metric, (r10, r20, r100) in t3.items():
                disp_name = name if first else ""
                t3_rows.append([disp_name, metric, fmt(r10), fmt(r20), fmt(r100)])
                t3_statuses.append(status)
                first = False
        add_table_page(pdf, "Table 3: Correlation (Kendall's Tau) Between SIR Spreading Capacity and Traditional Unweighted Centralities\n(Light Green Highlight = Test Dataset)", t3_headers, t3_rows, t3_statuses)

        # Table 4: Weighted Centralities vs Model
        t4_headers = ["Network Dataset", "Weighted Centrality Metric", "Top 10%", "Top 20%", "Top 100% (All)"]
        t4_rows = []
        t4_statuses = []
        for name, t4, status in results_wcent_model:
            first = True
            for metric, (r10, r20, r100) in t4.items():
                disp_name = name if first else ""
                t4_rows.append([disp_name, metric, fmt(r10), fmt(r20), fmt(r100)])
                t4_statuses.append(status)
                first = False
        add_table_page(pdf, "Table 4: Correlation (Kendall's Tau) Between NLGCN Model and Weighted Centrality Measures\n(Light Green Highlight = Test Dataset)", t4_headers, t4_rows, t4_statuses)

        # Table 5: SIR vs Weighted Centralities
        t5_headers = ["Network Dataset", "Weighted Centrality Metric", "Top 10%", "Top 20%", "Top 100% (All)"]
        t5_rows = []
        t5_statuses = []
        for name, t5, status in results_sir_wcent:
            first = True
            for metric, (r10, r20, r100) in t5.items():
                disp_name = name if first else ""
                t5_rows.append([disp_name, metric, fmt(r10), fmt(r20), fmt(r100)])
                t5_statuses.append(status)
                first = False
        add_table_page(pdf, "Table 5: Correlation (Kendall's Tau) Between SIR Spreading Capacity and Weighted Centrality Measures\n(Light Green Highlight = Test Dataset)", t5_headers, t5_rows, t5_statuses)

        # Table 6: Unweighted vs Weighted
        t6_headers = ["Network Dataset", "Centrality Comparison", "Top 10%", "Top 20%", "Top 100% (All)"]
        t6_rows = []
        t6_statuses = []
        for name, t6, status in results_unw_vs_w:
            first = True
            for metric, (r10, r20, r100) in t6.items():
                disp_name = name if first else ""
                t6_rows.append([disp_name, metric, fmt(r10), fmt(r20), fmt(r100)])
                t6_statuses.append(status)
                first = False
        add_table_page(pdf, "Table 6: Correlation (Kendall's Tau) Between Traditional Unweighted and Weighted Centrality Measures\n(Light Green Highlight = Test Dataset)", t6_headers, t6_rows, t6_statuses)

    print("\nPDF generated successfully!")

if __name__ == "__main__":
    main()
