"""
Shared code for Fourth Phase / Block 1: SIR ground truth, features, zero-shot evaluation of the
Third-Phase Hybrid Global-Local WNLGCN on weighted human contact networks.

Everything is reused from Third_Phase so the contact networks are treated exactly like the
networks the model was trained on:
  * graph loading, LCC extraction, node order, local 6-channel patches -> train_wnlgcn_multigraph.py
  * global descriptors and the hybrid model                            -> hybrid_common.py
  * SIR ground truth: same process and same parameters
        w_norm = w / max(w),  beta = min(1.5 / lambda_max(W_norm), 0.9),  p_uv = 1 - (1 - beta)^w_norm
    each newly infected node gets one chance to infect each susceptible neighbour (independent cascade),
    score(s) = expected final outbreak size from seed s, averaged over R runs, normalised by its max.

Only the SIR *estimator* differs: the original pure-Python loop is far too slow on these dense graphs, so
we use the equivalent live-edge formulation (Kempe et al., 2003): in each run every directed arc u->v is
"live" with probability p_uv, and the outbreak from s is exactly the set reachable from s over live arcs.
This gives the same expected outbreak size per seed; reachability is computed in C with scipy.
"""
import os
import sys
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
FOURTH = os.path.abspath(os.path.join(HERE, ".."))
PROJECT = os.path.abspath(os.path.join(FOURTH, ".."))
PROCESSED = os.path.join(FOURTH, "datasets", "processed")
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

THIRD = os.path.join(PROJECT, "Third_Phase")
sys.path.insert(0, os.path.join(THIRD, "weighted models"))
sys.path.insert(0, os.path.join(THIRD, "weighted_model_hybrid_global_local"))
import train_wnlgcn_multigraph as tw          # noqa: E402
import hybrid_common as hc                    # noqa: E402

SIR_RUNS = 500           # same number of runs as Third_Phase
SEMANTICS = "positive"   # contact time: larger weight = stronger link

# Networks used in Block 1 (SouthAfrica_Households is excluded: 88 separate households, LCC = 8 people)
GROUPS = {
    "Face-to-face": ["LH10_Hospital", "LyonSchool_Primary", "Thiers11_HighSchool", "Thiers12_HighSchool",
                     "Thiers13_HighSchool", "InVS13_Workplace", "InVS15_Workplace", "SFHH_Conference",
                     "HT09_Conference", "Infectious_Museum", "Malawi_Village", "Kenya_Households"],
    "Proximity": ["Utah_MiddleSchool", "Utah_ElementarySchool", "Utah_MiddleSchool_1m",
                  "Utah_ElementarySchool_1m", "Copenhagen_BT", "RealityMining_BT", "Haggle_Infocom05"],
    "Co-location": ["LH10_Hospital_Coloc", "InVS13_Workplace_Coloc", "InVS15_Workplace_Coloc",
                    "SFHH_Conference_Coloc", "LyonSchool_Primary_Coloc", "Thiers13_HighSchool_Coloc"],
    "Survey": ["Thiers13_Diaries"],
}
DATASETS = [d for g in GROUPS.values() for d in g]
GROUP_OF = {d: g for g, ds in GROUPS.items() for d in ds}


def path_of(name):
    return os.path.join(PROCESSED, f"{name}.txt")


def load_lcc(name):
    """Identical to Third_Phase process_dataset(): same loader, LCC and node order."""
    G_raw, _ = tw.load_weighted_graph(path_of(name), SEMANTICS)
    comps = sorted(nx.connected_components(G_raw), key=len, reverse=True)
    G = G_raw.subgraph(comps[0]).copy()
    return G, list(G.nodes()), G_raw.number_of_nodes()


def sir_setup(G, nodelist):
    """Return (P, beta, lambda_max): sparse matrix of per-edge transmission probabilities."""
    W = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr").astype(np.float64)
    W = W / W.max()                                              # weight_norm = w / max_w
    n = W.shape[0]
    if n > 2:
        lam = eigsh(W, k=1, which="LA", return_eigenvectors=False)[0]
    else:
        lam = np.max(np.linalg.eigvalsh(W.toarray()))
    beta = min(1.5 * (1.0 / lam if lam > 0 else 0.1), 0.9)
    P = W.copy()
    P.data = 1.0 - (1.0 - beta) ** P.data
    return sp.csr_matrix(P), beta, lam


def _reach_one_run(P, seed):
    """Outbreak size from every seed for one live-edge sample (each directed arc kept w.p. p)."""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    rows = np.repeat(np.arange(n), np.diff(P.indptr))
    keep = rng.random(P.nnz) < P.data
    live = sp.csr_matrix((np.ones(keep.sum()), (rows[keep], P.indices[keep])), shape=P.shape)
    # nodes in the same strongly connected component reach the same set -> one BFS per component
    n_comp, comp = connected_components(live, directed=True, connection="strong")
    rep = np.full(n_comp, -1)
    rep[comp[::-1]] = np.arange(n)[::-1]
    size_of_comp = np.empty(n_comp)
    for c in range(n_comp):
        size_of_comp[c] = len(breadth_first_order(live, rep[c], directed=True, return_predecessors=False))
    return size_of_comp[comp]


def _reach_chunk(P, seeds):
    return np.sum([_reach_one_run(P, s) for s in seeds], axis=0)


def sir_labels(P, runs=SIR_RUNS, base_seed=12345, n_jobs=-1, return_raw=False):
    """Expected outbreak size per seed node (normalised by max, as in Third_Phase)."""
    seeds = np.arange(runs) + base_seed
    chunks = np.array_split(seeds, min(runs, os.cpu_count() or 4))
    parts = Parallel(n_jobs=n_jobs)(delayed(_reach_chunk)(P, c) for c in chunks)
    y = np.sum(parts, axis=0) / runs
    return (y / y.max(), y) if return_raw else y / y.max()


# =====================================================================================
#   Calibrated regime: same transmission formula, rate chosen per network so the mean
#   outbreak is TARGET_OUTBREAK of the population.
#   p_uv = 1 - (1 - beta)^w_norm = 1 - exp(-tau * w_norm),  tau = -ln(1 - beta)  (no 0.9 cap)
# =====================================================================================
TARGET_OUTBREAK = 0.10


def norm_W(G, nodelist):
    W = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr").astype(np.float64)
    return sp.csr_matrix(W / W.max())


def P_from_tau(W, tau):
    P = W.copy()
    P.data = -np.expm1(-tau * W.data)
    return P


def calibrate_tau(W, target=TARGET_OUTBREAK, runs=100, iters=18):
    """Bisection on log(tau): mean outbreak fraction is increasing in tau."""
    n = W.shape[0]
    lo, hi = np.log(1e-3), np.log(1e7)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, raw = sir_labels(P_from_tau(W, np.exp(mid)), runs=runs, base_seed=777, return_raw=True)
        if raw.mean() / n < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def compute_raw_features_tau(G, nodelist, tau):
    """hybrid_common.compute_raw_features with the transmission rate set to tau (instead of 1.5/lambda_max)."""
    n = len(nodelist)
    W = norm_W(G, nodelist)
    A = W.copy()
    A.data = np.ones_like(A.data)
    deg = np.asarray(A.sum(axis=1)).ravel()
    strength = np.asarray(W.sum(axis=1)).ravel()

    _, eig_W = hc._leading_eigvec(W)
    T = P_from_tau(W, tau)
    L = W.copy()
    L.data = -tau * W.data                     # log(1 - T), exact and finite
    _, eig_T = hc._leading_eigvec(T)

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
        v = v / (np.abs(v).max() + 1e-12)
        acc = acc + v
        ds[k] = acc.copy()
    mf = hc.mean_field_reach(L, n)
    feats = np.stack([deg, strength, core, clustering, eig_W, eig_T, pagerank, nbr_deg, nbr_str, reach2,
                      h_index, ds[2], ds[4], mf[3], mf[12]], axis=1)
    return feats, W


def baselines(G, nodelist):
    """Classical weighted centralities, computed exactly as in Third_Phase evaluate_and_report.py."""
    wdeg_d = dict(G.degree(weight="weight"))
    G_dist = nx.Graph()
    for u, v, d in G.edges(data=True):
        G_dist.add_edge(u, v, distance=1.0 / d.get("weight", 1.0))
    wclos_d = nx.closeness_centrality(G_dist, distance="distance")
    wbet_d = nx.betweenness_centrality(G_dist, weight="distance", normalized=True)
    try:
        weig_d = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        weig_d = {u: 0.0 for u in nodelist}
    return {k: np.array([d[u] for u in nodelist]) for k, d in
            (("W-Eigenvector", weig_d), ("Strength (W-Deg)", wdeg_d),
             ("W-Closeness", wclos_d), ("W-Betweenness", wbet_d))}
