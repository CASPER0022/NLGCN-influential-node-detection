"""
Shared code for the Hybrid Global-Local WNLGCN (weighted networks).

Idea
----
The Deep 6-Channel WNLGCN only sees a 41x41 local neighbourhood patch per node, so it
loses to W-Eigenvector exactly on graphs where spreading power is driven by *global*
structure. This model keeps the local CNN branch and adds a *global* branch built from
graph-wide, SIR-aware node descriptors:

  * spectral:        eigenvector of W, eigenvector of the transmission matrix T, PageRank
  * structural:      degree, strength, k-core, clustering, H-index, 2-hop reach
  * dynamics-aware:  dynamics-sensitive centrality  sum_k T^k 1   (Liu et al., 2016)
                     mean-field outbreak size       P <- 1 - exp(P @ log(1 - T))
    where T_ij = 1 - (1 - beta)^w_ij uses exactly the beta of the SIR ground truth
    (beta = 1.5 / lambda_max(W_norm), capped at 0.9).

Every descriptor is rank-transformed *within its own graph* (size invariant -> zero-shot
transfer to bigger graphs) and then propagated SIGN-style (1-hop / 2-hop weighted mean and
1-hop max), giving the global branch a GNN-like receptive field without full-graph training.
"""
import os
import sys
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata, kendalltau
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Directory Paths ----
script_dir = os.path.dirname(os.path.abspath(__file__))
weighted_model_dir = os.path.abspath(os.path.join(script_dir, "..", "weighted models"))
results_dir = os.path.join(weighted_model_dir, "results")
deep_model_dir = os.path.abspath(os.path.join(script_dir, "..", "weighted_model_deep_6channel"))
feature_cache_dir = os.path.join(script_dir, "feature_cache")
checkpoint_dir = os.path.join(script_dir, "checkpoints")

datasets_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
train_folder = os.path.join(datasets_base_dir, "weighted Datasets", "train")
test_folder = os.path.join(datasets_base_dir, "weighted Datasets", "test")

sys.path.append(weighted_model_dir)
from train_wnlgcn_multigraph import load_weighted_graph, target_datasets  # noqa: E402

# Training graphs held out for checkpoint selection (never used for gradient updates)
DEFAULT_VAL_DATASETS = ["carrib.txt", "synthetic_sf_1100.txt", "synthetic_sf_2100.txt"]


def _cached(f):
    return os.path.exists(os.path.join(results_dir, f"{f}_weighted_local_norm_X.npy"))


train_datasets = [f for f in sorted(os.listdir(train_folder)) if _cached(f)]
test_datasets = [f for f in sorted(os.listdir(test_folder)) if _cached(f)]


def graph_path(filename):
    for folder in (train_folder, test_folder):
        p = os.path.join(folder, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(filename)


def load_lcc(filename):
    """Same loading / LCC / node order as process_dataset() so rows align with the cached X."""
    semantics = target_datasets.get(filename, "positive")
    G_raw, _ = load_weighted_graph(graph_path(filename), semantics)
    components = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    G = G_raw.subgraph(components[0]).copy()
    return G, list(G.nodes())


# =====================================================================================
#                           Global (graph-wide) node descriptors
# =====================================================================================
RAW_FEATURE_NAMES = [
    "degree", "strength", "core", "clustering", "eig_W", "eig_T", "pagerank",
    "nbr_degree_sum", "nbr_strength_sum", "reach_2hop", "h_index",
    "DS_k2", "DS_k4", "MF_reach_k3", "MF_reach_k12",
]


def _leading_eigvec(M):
    n = M.shape[0]
    if n <= 3:
        vals, vecs = np.linalg.eigh(M.toarray())
        return vals[-1], np.abs(vecs[:, -1])
    vals, vecs = eigsh(M.astype(np.float64), k=1, which="LA")
    return vals[0], np.abs(vecs[:, 0])


def mean_field_reach(L, n, iters_record=(3, 12), block=512):
    """
    Expected outbreak size from every seed under an individual-based mean-field SIR:
        P_s <- 1 - exp(P_s @ L),  L_uv = log(1 - T_uv),  P_s[s] = 1.
    Returns {k: reach after k iterations}.
    """
    out = {k: np.zeros(n) for k in iters_record}
    K = max(iters_record)
    L = L.tocsr()
    for start in range(0, n, block):
        src = np.arange(start, min(n, start + block))
        P = np.zeros((len(src), n))
        P[np.arange(len(src)), src] = 1.0
        for it in range(1, K + 1):
            P = 1.0 - np.exp(np.asarray((L @ P.T).T))   # L symmetric
            P[np.arange(len(src)), src] = 1.0
            if it in out:
                out[it][src] = P.sum(axis=1)
    return out


def compute_raw_features(G, nodelist):
    n = len(nodelist)
    W = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr").astype(np.float64)
    W = W / W.max()
    W = sp.csr_matrix(W)
    A = W.copy()
    A.data = np.ones_like(A.data)

    deg = np.asarray(A.sum(axis=1)).ravel()
    strength = np.asarray(W.sum(axis=1)).ravel()

    # SIR transmission matrix with the ground-truth beta
    lam_W, eig_W = _leading_eigvec(W)
    beta = min(1.5 / lam_W if lam_W > 0 else 0.1, 0.9)
    T = W.copy()
    T.data = 1.0 - (1.0 - beta) ** W.data
    L = T.copy()
    L.data = np.log(1.0 - T.data)
    _, eig_T = _leading_eigvec(T)

    core_d = nx.core_number(G)
    core = np.array([core_d[u] for u in nodelist], dtype=float)
    clus_d = nx.clustering(G)
    clustering = np.array([clus_d[u] for u in nodelist], dtype=float)
    try:
        pr_d = nx.pagerank(G, weight="weight")
        pagerank = np.array([pr_d[u] for u in nodelist])
    except Exception:
        pagerank = strength / strength.sum()

    nbr_deg = A @ deg
    nbr_str = A @ strength
    A2 = (A @ A + A).tocsr()
    A2.setdiag(0)
    A2.eliminate_zeros()
    reach2 = np.diff(A2.indptr).astype(float)

    h_index = np.zeros(n)
    for i in range(n):
        d = np.sort(deg[A.indices[A.indptr[i]:A.indptr[i + 1]]])[::-1]
        h_index[i] = np.sum(d >= np.arange(1, len(d) + 1))

    v = np.ones(n)
    acc = np.zeros(n)
    ds = {}
    for k in range(1, 5):
        v = T @ v
        v = v / (np.abs(v).max() + 1e-12)   # avoid overflow; ranks unaffected per k
        acc = acc + v
        ds[k] = acc.copy()

    mf = mean_field_reach(L, n)

    feats = np.stack([
        deg, strength, core, clustering, eig_W, eig_T, pagerank,
        nbr_deg, nbr_str, reach2, h_index,
        ds[2], ds[4], mf[3], mf[12],
    ], axis=1)
    return feats, W, beta


def transform_features(raw, W):
    """Within-graph rank + scale-free z-score, then SIGN propagation (1-hop, 2-hop, max)."""
    n, d = raw.shape
    R = np.stack([(rankdata(raw[:, j]) - 1) / max(n - 1, 1) for j in range(d)], axis=1)
    Z = np.log1p(raw / (raw.mean(axis=0, keepdims=True) + 1e-12))
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-6)

    s = np.asarray(W.sum(axis=1)).ravel()
    P = sp.diags(1.0 / np.maximum(s, 1e-12)) @ W
    R1 = P @ R
    R2 = P @ R1
    Wc = sp.csr_matrix(W)
    if Wc.nnz > 0:
        Rmax = np.maximum.reduceat(R[Wc.indices], Wc.indptr[:-1], axis=0)
        Rmax[np.diff(Wc.indptr) == 0] = 0.0
    else:
        Rmax = np.zeros_like(R)
    return np.concatenate([R, Z, R1, R2, Rmax], axis=1).astype(np.float32)


N_GLOBAL_FEATURES = 5 * len(RAW_FEATURE_NAMES)


def get_global_features(filename, verbose=True):
    """Returns (features [n, N_GLOBAL_FEATURES], raw descriptors [n, 15]); cached on disk."""
    os.makedirs(feature_cache_dir, exist_ok=True)
    fpath = os.path.join(feature_cache_dir, f"{filename}_global_feats.npy")
    rpath = os.path.join(feature_cache_dir, f"{filename}_global_raw.npy")
    if os.path.exists(fpath) and os.path.exists(rpath):
        return np.load(fpath), np.load(rpath)

    if verbose:
        print(f"  -> Computing global descriptors for {filename} ...", flush=True)
    G, nodelist = load_lcc(filename)
    y = np.load(os.path.join(results_dir, f"{filename}_weighted_y.npy")).flatten()
    if len(nodelist) != len(y):
        raise ValueError(f"{filename}: LCC has {len(nodelist)} nodes but cached labels have {len(y)}")
    raw, W, _ = compute_raw_features(G, nodelist)
    feats = transform_features(raw, W)
    np.save(fpath, feats)
    np.save(rpath, raw)
    return feats, raw


def load_local_X(filename):
    return np.load(os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")).astype(np.float32)


def load_labels(filename):
    return np.load(os.path.join(results_dir, f"{filename}_weighted_y.npy")).flatten()


# =====================================================================================
#                                       Model
# =====================================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels=6, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(x.mean(dim=(2, 3))).view(b, c, 1, 1)
        return x * y


class HybridWNLGCN(nn.Module):
    def __init__(self, n_global=N_GLOBAL_FEATURES, hidden=128, dropout=0.2):
        super().__init__()
        # Local branch: 6-channel neighbourhood patch (41x41)
        self.attention = ChannelAttention(6, reduction=2)
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 20x20
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                                           # 4x4
        )
        self.local_fc = nn.Sequential(nn.Linear(32 * 16, hidden), nn.GELU(), nn.Dropout(dropout))

        # Global branch: graph-wide descriptors + SIGN aggregates
        self.global_mlp = nn.Sequential(
            nn.Linear(n_global, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )

        # Gated fusion
        self.gate = nn.Sequential(nn.Linear(2 * hidden, 2 * hidden), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        # Linear shortcut on global descriptors (lets the model start from a DS-like ranking)
        self.skip = nn.Linear(n_global, 1)

    def forward(self, x_local, x_global):
        h_l = self.local_fc(self.cnn(self.attention(x_local)).flatten(1))
        h_g = self.global_mlp(x_global)
        h = torch.cat([h_l, h_g], dim=1)
        h = h * self.gate(h)
        return self.head(h) + self.skip(x_global)


# =====================================================================================
#                                 Loss & metrics
# =====================================================================================
def top_focused_ranknet_loss(pred, y, is_top, top_top_weight=4.0, mixed_weight=2.0, sigma=2.0, tol=1e-4):
    """
    RankNet (logistic pairwise) loss. Kendall-tau@top20% only scores pairs where BOTH nodes are
    in the top 20%, so those pairs get the largest weight; top-vs-rest pairs are next
    (they decide who enters the top set).
    """
    pred = pred.view(-1)
    y = y.view(-1)
    is_top = is_top.view(-1)
    diff_p = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)
    mask = diff_y > tol   # each unordered pair once, i ranked above j
    if mask.sum() == 0:
        return pred.sum() * 0.0
    both = is_top.unsqueeze(1) * is_top.unsqueeze(0)
    either = torch.max(is_top.unsqueeze(1), is_top.unsqueeze(0))
    w = 1.0 + (mixed_weight - 1.0) * either + (top_top_weight - mixed_weight) * both
    losses = F.softplus(-sigma * diff_p)
    return (losses * w)[mask].sum() / w[mask].sum()


def get_subset_indices(labels, pct):
    k = max(1, int(pct / 100.0 * len(labels)))
    return np.argsort(labels)[::-1][:k]


def safe_tau(x, y, indices):
    xs, ys = x[indices], y[indices]
    if len(xs) < 2 or np.all(xs == xs[0]) or np.all(ys == ys[0]):
        return np.nan
    try:
        tau, _ = kendalltau(xs, ys)
        return tau if not np.isnan(tau) else np.nan
    except Exception:
        return np.nan


def top_overlap(pred, labels, pct=20):
    """Fraction of the true top-pct% spreaders that the method also places in its top-pct%."""
    k = max(1, int(pct / 100.0 * len(labels)))
    true_top = set(np.argsort(labels)[::-1][:k])
    pred_top = set(np.argsort(pred)[::-1][:k])
    return len(true_top & pred_top) / k


@torch.no_grad()
def predict(model, X_local, X_global, device, batch_size=1024):
    model.eval()
    out = []
    for i in range(0, len(X_local), batch_size):
        xl = torch.from_numpy(X_local[i:i + batch_size]).to(device)
        xg = torch.from_numpy(X_global[i:i + batch_size]).to(device)
        out.append(model(xl, xg).cpu().numpy().ravel())
    return np.concatenate(out)


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-9)
