"""
Shared code for Fourth Phase / Block 2: application-level evaluation.

  * method scores for any contact network (Block 1 cache, or new networks such as single days)
  * targeted vaccination simulation (live-edge SIR with common random numbers)

All SIR quantities use the calibrated regime from Block 1 (p = 1 - exp(-tau * w_norm), tau chosen per
network so the mean outbreak is 10%). MF-Reach is a model input and is deliberately not a baseline.
"""
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components, shortest_path
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "block1"))
import contact_common as cc            # noqa: E402
from contact_common import tw, hc     # noqa: E402
import run_block1 as rb               # noqa: E402
import torch                          # noqa: E402

CACHE = os.path.join(HERE, "cache")
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
for d in (CACHE, DATA, RESULTS):
    os.makedirs(d, exist_ok=True)

# Ranking methods compared in Block 2 (same as Block 1, no MF-Reach)
METHODS = rb.METHODS
SHORT = {"Hybrid G-L (ours)": "Ours", "Prev. Deep 6-Ch": "Deep6", "W-Eigenvector": "W-Eig",
         "Strength (W-Deg)": "Strength", "W-Closeness": "W-Clos", "W-Betweenness": "W-Betw",
         "Random": "Random", "True SIR ranking": "SIR (ref.)"}


# ------------------------------------------------------------------ models
_MODELS = None


def models():
    global _MODELS
    if _MODELS is None:
        _MODELS = rb.load_models()
    return _MODELS


def model_scores(X, Gf):
    ens, prev = models()
    ours = np.mean([hc.zscore(hc.predict(m, X, Gf, torch.device("cpu"))) for m in ens], axis=0)
    with torch.no_grad():
        deep = np.concatenate([prev(torch.from_numpy(X[i:i + 1024])).numpy().ravel()
                               for i in range(0, len(X), 1024)])
    return ours, deep


# ------------------------------------------------------------------ network bundle
class Net:
    """Everything Block 2 needs about one network (calibrated regime)."""

    def __init__(self, key, nodes, W, tau, y, scores):
        self.key, self.nodes, self.W, self.tau, self.y, self.scores = key, nodes, W, tau, y, scores
        self.n = len(nodes)
        self.P = cc.P_from_tau(W, tau)
        self.index = {u: i for i, u in enumerate(nodes)}


def block1_net(name):
    """Network from the Block 1 cache (calibrated labels and features already computed)."""
    key = os.path.join(CACHE, f"{name}_scores.npz")
    G, nodes, _ = cc.load_lcc(name)
    tau = float(np.load(rb.cpath(name, "sir_params_cal"))[2])
    y = np.load(rb.cpath(name, "y_cal"))
    if os.path.exists(key):
        S = dict(np.load(key))
    else:
        X = np.load(rb.cpath(name, "X_local")).astype(np.float32)
        Gf = np.load(rb.cpath(name, "global_feats_cal")).astype(np.float32)
        ours, deep = model_scores(X, Gf)
        base = np.load(rb.cpath(name, "baselines"))
        S = {METHODS[0]: ours, METHODS[1]: deep, **{m: base[:, j] for j, m in enumerate(METHODS[2:6])}}
        np.savez(key, **S)
    return Net(name, nodes, cc.norm_W(G, nodes), tau, y, S)


def path_net(key, path):
    """Any edge-list file (e.g. one day of contacts): calibrate tau, SIR labels, features, all scores."""
    f = os.path.join(CACHE, f"{key}.npz")
    G_raw, _ = tw.load_weighted_graph(path, cc.SEMANTICS)
    import networkx as nx
    G = G_raw.subgraph(sorted(nx.connected_components(G_raw), key=len, reverse=True)[0]).copy()
    nodes = list(G.nodes())
    W = cc.norm_W(G, nodes)
    if not os.path.exists(f):
        t0 = time.time()
        tau = cc.calibrate_tau(W)
        y = cc.sir_labels(cc.P_from_tau(W, tau))
        X, _ = tw.process_dataset(path, key, cc.SEMANTICS, precalculated_y=y)
        raw, Wn = cc.compute_raw_features_tau(G, nodes, tau)
        Gf = hc.transform_features(raw, Wn).astype(np.float32)
        ours, deep = model_scores(X.astype(np.float32), Gf)
        b = cc.baselines(G, nodes)
        np.savez(f, tau=tau, y=y, nodes=np.array(nodes), **{METHODS[0]: ours, METHODS[1]: deep},
                 **{m: b[m] for m in METHODS[2:6]})
        print(f"  prepared {key:34s} n={len(nodes):4d} tau={tau:9.2f} ({time.time() - t0:.0f}s)", flush=True)
    d = np.load(f)
    assert list(d["nodes"]) == nodes
    return Net(key, nodes, W, float(d["tau"]), d["y"], {m: d[m] for m in METHODS})


# ------------------------------------------------------------------ rankings
def ranking(score, rng=None):
    """Node indices from highest to lowest score; ties broken at random (fixed seed)."""
    rng = rng or np.random.default_rng(0)
    jitter = rng.random(len(score)) * 1e-9 * (np.nanstd(score) + 1e-12)
    return np.argsort(-(np.nan_to_num(score, nan=-np.inf) + jitter), kind="stable")


# ------------------------------------------------------------------ vaccination
def _live_graph(P, rng):
    n = P.shape[0]
    rows = np.repeat(np.arange(n), np.diff(P.indptr))
    keep = rng.random(P.nnz) < P.data
    return rows[keep], P.indices[keep]


def _outbreak(r, c, n, vacc, peak_seeds):
    """Mean final size from a random unvaccinated index case + mean peak (largest generation) size."""
    ok = ~(vacc[r] | vacc[c])
    live = sp.csr_matrix((np.ones(ok.sum()), (r[ok], c[ok])), shape=(n, n))
    n_comp, comp = connected_components(live, directed=True, connection="strong")
    rep = np.full(n_comp, -1)
    rep[comp[::-1]] = np.arange(n)[::-1]
    size = np.empty(n_comp)
    for k in range(n_comp):
        size[k] = len(breadth_first_order(live, rep[k], directed=True, return_predecessors=False))
    reach = size[comp]
    final = reach[~vacc].mean() / n
    # In the independent cascade the infection generation of a node = its hop distance on the live graph
    d = shortest_path(live, unweighted=True, directed=True, indices=peak_seeds)
    d = np.atleast_2d(d)
    peaks = [np.bincount(row[np.isfinite(row)].astype(int)).max() for row in d]
    return final, float(np.mean(peaks)) / n


def _vacc_chunk(P, plans, seeds, n_peak_seeds):
    """plans: list of (method, frac, k, mask or None for random). Same live graph for every plan (CRN)."""
    n = P.shape[0]
    out = np.zeros((len(plans), 2))
    for s in seeds:
        rng = np.random.default_rng(10_000 + s)
        r, c = _live_graph(P, rng)
        perm = rng.permutation(n)
        for j, (_, _, k, mask) in enumerate(plans):
            if mask is None:                       # random vaccination: new random set each run
                mask = np.zeros(n, bool)
                mask[perm[:k]] = True
            free = perm[~mask[perm]][:n_peak_seeds]
            out[j] += _outbreak(r, c, n, mask, free)
    return out


def vaccination(net, orders, fractions, runs=200, n_peak_seeds=10, n_jobs=-1):
    """
    orders: {method: node indices ranked best-first} ("Random" -> None).
    Returns list of dicts (method, frac, k, outbreak, peak). Outbreak/peak = fraction of all n people.
    """
    n = net.n
    plans = [("none", 0.0, 0, np.zeros(n, bool))]
    for m, order in orders.items():
        for f in fractions:
            k = int(round(f * n))
            if f == 0 or k == 0:
                continue
            if order is None:
                plans.append((m, f, k, None))
            else:
                mask = np.zeros(n, bool)
                mask[order[:k]] = True
                plans.append((m, f, k, mask))
    seeds = np.arange(runs)
    chunks = np.array_split(seeds, min(runs, os.cpu_count() or 4))
    tot = np.sum(Parallel(n_jobs=n_jobs)(delayed(_vacc_chunk)(net.P, plans, ch, n_peak_seeds)
                                         for ch in chunks), axis=0) / runs
    res = []
    base_final, base_peak = tot[0]
    for m in orders:
        res.append(dict(method=m, frac=0.0, k=0, outbreak=base_final, peak=base_peak))
    for (m, f, k, _), (fin, pk) in zip(plans[1:], tot[1:]):
        res.append(dict(method=m, frac=f, k=k, outbreak=fin, peak=pk))
    return res
