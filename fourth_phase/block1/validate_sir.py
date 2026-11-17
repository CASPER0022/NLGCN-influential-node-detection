"""
Check that the fast live-edge SIR estimator reproduces the original Third_Phase SIR labels.
  1) karate.txt: compare with the cached Third_Phase labels (weighted models/results)
  2) LH10_Hospital: compare with the original pure-Python worker run now on the same graph
Two independent Monte-Carlo estimates of the same quantity should agree closely (tau near 1).
"""
import os
import random
import time
import numpy as np
from scipy.stats import kendalltau, pearsonr
import contact_common as cc
from contact_common import tw


def original_labels(G, nodelist, P):
    idx = {u: i for i, u in enumerate(nodelist)}
    adj = {u: [] for u in nodelist}
    Pc = P.tocoo()
    for i, j, p in zip(Pc.row, Pc.col, Pc.data):
        adj[nodelist[i]].append((nodelist[j], p))
    random.seed(0)
    y = np.array(tw.worker_sir_weighted(adj, nodelist, runs=cc.SIR_RUNS))
    return y / y.max()


def report(tag, a, b):
    top = np.argsort(b)[::-1][:max(2, len(b) // 5)]
    print(f"  {tag}: Kendall tau (all) = {kendalltau(a, b)[0]:.3f} | tau (true top 20%) = "
          f"{kendalltau(a[top], b[top])[0]:.3f} | Pearson r = {pearsonr(a, b)[0]:.4f}")


def main():
    # 1) karate vs cached Third_Phase labels
    path = os.path.join(cc.THIRD, "weighted models", "results", "karate.txt_weighted_y.npy")
    y_old = np.load(path).flatten()
    G_raw, _ = tw.load_weighted_graph(os.path.join(cc.PROJECT, "Datasets", "weighted Datasets", "test", "karate.txt"),
                                      "positive")
    import networkx as nx
    G = G_raw.subgraph(sorted(nx.connected_components(G_raw), key=len, reverse=True)[0]).copy()
    nodes = list(G.nodes())
    P, beta, _ = cc.sir_setup(G, nodes)
    y_new = cc.sir_labels(P)
    print(f"karate (n={len(nodes)}, beta={beta:.3f})")
    report("fast vs cached Third_Phase labels", y_new, y_old)

    # 2) LH10 vs original worker
    G, nodes, _ = cc.load_lcc("LH10_Hospital")
    P, beta, _ = cc.sir_setup(G, nodes)
    t = time.time(); y_new = cc.sir_labels(P); t_new = time.time() - t
    t = time.time(); y_orig = original_labels(G, nodes, P); t_orig = time.time() - t
    print(f"LH10_Hospital (n={len(nodes)}, beta={beta:.3f}) | fast {t_new:.1f}s vs original {t_orig:.1f}s")
    report("fast vs original worker", y_new, y_orig)


if __name__ == "__main__":
    main()
