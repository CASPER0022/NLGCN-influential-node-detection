"""
Ablation study for the Hybrid Global-Local WNLGCN (weighted networks).

Every variant is RETRAINED FROM SCRATCH with the same protocol as the full model
(same fit / validation split, same seeds, best-epoch selection on held-out validation
graphs, z-score ensemble over seeds). No input occlusion on a shared checkpoint.

Variants
--------
  Reference
    full                 complete model
  Architecture
    local_only           global branch removed (6-channel CNN only, new training recipe)
    global_only          local CNN branch removed (global descriptor MLP only)
    no_attention         channel attention removed from the CNN branch
    no_gate              gated fusion replaced by plain concatenation
    no_skip              linear shortcut from global descriptors removed
  Global descriptors
    no_dynamics_feats    drop SIR/beta-aware descriptors (eig_T, DS_k2, DS_k4, MF_k3, MF_k12)
    no_spectral_feats    drop spectral descriptors (eig_W, PageRank)
    no_structural_feats  drop structural descriptors (degree, strength, core, clustering, ...)
    no_sign_propagation  drop 1-hop / 2-hop mean and 1-hop max aggregation (node's own descriptors only)
    no_rank_norm         no within-graph rank / z normalisation: log descriptors standardised with
                         GLOBAL training-set statistics (tests size invariance)
  Training objective
    mse_only             ranking loss removed (Huber regression only)
    uniform_pair_weights RankNet with all pairs weighted equally (no top-top emphasis)
    no_top_oversampling  uniform node sampling inside each graph
    global_label_norm    labels standardised across all graphs instead of per graph
  Ensembling (no retraining needed)
    single_model         the full model's seeds evaluated individually (mean over seeds)

Outputs (in this folder):
    results/<variant>.json            raw per-graph results (resumable: finished variants are skipped)
    checkpoints/<variant>_seed<s>.pth best checkpoint per variant and seed
    ablation_report_hybrid.pdf        PDF report
    ablation_summary.csv, ablation_per_dataset.csv, ablation_table.tex (LaTeX table for the paper)

Usage (from this folder):
    python run_ablation_study.py                                  # all variants, 3 seeds x 24 epochs
    python run_ablation_study.py --seeds 42 --epochs 12           # quick pass
    python run_ablation_study.py --variants full no_gate          # selected variants only
    python run_ablation_study.py --report_only                    # rebuild the PDF from saved results
    python run_ablation_study.py --force --variants no_skip       # retrain a finished variant
"""
import os
import sys
import json
import time
import random
import argparse
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon

ablation_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(ablation_dir, ".."))
sys.path.insert(0, model_dir)

from hybrid_common import (  # noqa: E402
    ChannelAttention, train_datasets, test_datasets, DEFAULT_VAL_DATASETS, RAW_FEATURE_NAMES,
    get_global_features, load_local_X, load_labels, load_lcc, top_focused_ranknet_loss,
    get_subset_indices, safe_tau, top_overlap, zscore,
)
from evaluate_and_report import add_text_page, add_table_page, fmt, best_indices  # noqa: E402

results_out_dir = os.path.join(ablation_dir, "results")
ckpt_out_dir = os.path.join(ablation_dir, "checkpoints")
norank_cache_dir = os.path.join(ablation_dir, "feature_cache_norank")

NF = len(RAW_FEATURE_NAMES)
FEATURE_GROUPS = {
    "dynamics": ["eig_T", "DS_k2", "DS_k4", "MF_reach_k3", "MF_reach_k12"],
    "spectral": ["eig_W", "pagerank"],
    "structural": ["degree", "strength", "core", "clustering", "nbr_degree_sum",
                   "nbr_strength_sum", "reach_2hop", "h_index"],
}

# name -> (group, short description, overrides)
VARIANTS = {
    "full":                 ("Reference", "Complete Hybrid Global-Local model", {}),
    "local_only":           ("Architecture", "Global descriptor branch", {"use_global": False}),
    "global_only":          ("Architecture", "Local 6-channel CNN branch", {"use_local": False}),
    "no_attention":         ("Architecture", "Channel attention", {"use_attention": False}),
    "no_gate":              ("Architecture", "Gated fusion (-> concatenation)", {"use_gate": False}),
    "no_skip":              ("Architecture", "Linear global shortcut", {"use_skip": False}),
    "no_dynamics_feats":    ("Global descriptors", "SIR-aware descriptors (T, DS, MF)", {"drop_groups": ["dynamics"]}),
    "no_spectral_feats":    ("Global descriptors", "Spectral descriptors (eig_W, PageRank)", {"drop_groups": ["spectral"]}),
    "no_structural_feats":  ("Global descriptors", "Structural descriptors", {"drop_groups": ["structural"]}),
    "no_sign_propagation":  ("Global descriptors", "SIGN neighbour aggregation", {"sign": False}),
    "no_rank_norm":         ("Global descriptors", "Within-graph rank normalisation", {"feature_mode": "norank"}),
    "mse_only":             ("Training objective", "Ranking loss (Huber only)", {"alpha_rank": 0.0, "alpha_reg": 1.0}),
    "uniform_pair_weights": ("Training objective", "Top-focused pair weighting", {"top_top_weight": 1.0, "mixed_weight": 1.0}),
    "no_top_oversampling":  ("Training objective", "Top-20% over-sampling", {"top_oversample": 1.0}),
    "global_label_norm":    ("Training objective", "Per-graph label standardisation", {"label_norm": "global"}),
}
ORDER = list(VARIANTS) + ["single_model"]

BASE_CONFIG = {
    "use_local": True, "use_global": True, "use_attention": True, "use_gate": True, "use_skip": True,
    "drop_groups": [], "sign": True, "feature_mode": "rank",
    "alpha_reg": 0.5, "alpha_rank": 1.0, "top_top_weight": 4.0, "mixed_weight": 2.0, "sigma": 2.0,
    "top_oversample": 3.0, "label_norm": "per_graph",
    "hidden": 128, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 256,
}


# =====================================================================================
#                                    Model
# =====================================================================================
class AblationHybrid(nn.Module):
    """HybridWNLGCN with switchable components (all switches on == HybridWNLGCN)."""

    def __init__(self, n_global, cfg):
        super().__init__()
        h, p = cfg["hidden"], cfg["dropout"]
        self.use_local, self.use_global = cfg["use_local"], cfg["use_global"]
        self.use_gate = cfg["use_gate"]
        self.use_skip = cfg["use_skip"] and self.use_global
        fused = 0
        if self.use_local:
            self.attention = ChannelAttention(6, reduction=2) if cfg["use_attention"] else nn.Identity()
            self.cnn = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.local_fc = nn.Sequential(nn.Linear(32 * 16, h), nn.GELU(), nn.Dropout(p))
            fused += h
        if self.use_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(n_global, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p),
                nn.Linear(h, h), nn.GELU(),
            )
            fused += h
        if self.use_gate:
            self.gate = nn.Sequential(nn.Linear(fused, fused), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(fused, 64), nn.GELU(), nn.Dropout(p), nn.Linear(64, 1))
        if self.use_skip:
            self.skip = nn.Linear(n_global, 1)

    def forward(self, x_local, x_global):
        parts = []
        if self.use_local:
            parts.append(self.local_fc(self.cnn(self.attention(x_local)).flatten(1)))
        if self.use_global:
            parts.append(self.global_mlp(x_global))
        h = torch.cat(parts, dim=1)
        if self.use_gate:
            h = h * self.gate(h)
        out = self.head(h)
        if self.use_skip:
            out = out + self.skip(x_global)
        return out


# =====================================================================================
#                          Variant-specific global descriptors
# =====================================================================================
def feature_columns(cfg):
    """Column indices into the standard 75-d descriptor block [R, Z, R1, R2, Rmax]."""
    dropped = {RAW_FEATURE_NAMES.index(n) for g in cfg["drop_groups"] for n in FEATURE_GROUPS[g]}
    keep = [j for j in range(NF) if j not in dropped]
    blocks = range(5) if cfg["sign"] else range(2)
    return [b * NF + j for b in blocks for j in keep]


def norank_features(filename):
    """log1p(raw descriptors) + SIGN propagation, WITHOUT any per-graph normalisation (60-d)."""
    os.makedirs(norank_cache_dir, exist_ok=True)
    path = os.path.join(norank_cache_dir, f"{filename}_norank.npy")
    if os.path.exists(path):
        return np.load(path)
    G, nodelist = load_lcc(filename)
    W = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr").astype(np.float64))
    W = W / W.max()
    _, raw = get_global_features(filename, verbose=False)
    L0 = np.log1p(raw)
    P = sp.diags(1.0 / np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1e-12)) @ W
    L1 = P @ L0
    L2 = P @ L1
    Lmax = np.maximum.reduceat(L0[W.indices], W.indptr[:-1], axis=0)
    out = np.concatenate([L0, L1, L2, Lmax], axis=1).astype(np.float32)
    np.save(path, out)
    return out


class GlobalFeatureBuilder:
    def __init__(self, cfg, fit_names):
        self.cfg = cfg
        self.cols = feature_columns(cfg)
        self.mu = self.sd = None
        if cfg["feature_mode"] == "norank":
            stacked = np.concatenate([norank_features(f) for f in fit_names], axis=0)
            self.mu, self.sd = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6

    def __call__(self, filename):
        if self.cfg["feature_mode"] == "norank":
            return ((norank_features(filename) - self.mu) / self.sd).astype(np.float32)
        feats, _ = get_global_features(filename, verbose=False)
        return np.ascontiguousarray(feats[:, self.cols])

    @property
    def dim(self):
        return 4 * NF if self.cfg["feature_mode"] == "norank" else len(self.cols)


# =====================================================================================
#                                 Training / inference
# =====================================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_graph(f, cfg, gbuild, label_stats=None):
    y = load_labels(f)
    k = max(1, int(len(y) * 0.2))
    top_thr = np.partition(y, -k)[-k]
    if cfg["label_norm"] == "global" and label_stats is not None:
        y_norm = (y - label_stats[0]) / (label_stats[1] + 1e-9)
    else:
        y_norm = (y - y.mean()) / (y.std() + 1e-9)
    return {
        "name": f,
        "X": load_local_X(f) if cfg["use_local"] else None,
        "G": gbuild(f),
        "y": y,
        "y_norm": y_norm.astype(np.float32),
        "is_top": (y >= top_thr).astype(np.float32),
    }


@torch.no_grad()
def predict(model, X, G, device, batch_size=1024):
    model.eval()
    out = []
    for i in range(0, len(G), batch_size):
        xl = torch.from_numpy(X[i:i + batch_size]).to(device) if X is not None else None
        xg = torch.from_numpy(G[i:i + batch_size]).to(device)
        out.append(model(xl, xg).cpu().numpy().ravel())
    return np.concatenate(out)


def mean_val_tau(model, graphs, device):
    return float(np.nanmean([safe_tau(predict(model, g["X"], g["G"], device), g["y"],
                                      get_subset_indices(g["y"], 20)) for g in graphs]))


def train_seed(variant, seed, cfg, n_global, fit_graphs, val_graphs, args, device):
    set_seed(seed)
    model = AblationHybrid(n_global, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bsz = cfg["batch_size"]
    steps_per_epoch = sum(int(np.ceil(len(g["y"]) / bsz)) for g in fit_graphs)
    total, warm = steps_per_epoch * args.epochs, steps_per_epoch * 3
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))

    samplers = [torch.tensor(1.0 + (cfg["top_oversample"] - 1.0) * g["is_top"]) for g in fit_graphs]
    tensors = [(torch.from_numpy(g["X"]) if g["X"] is not None else None, torch.from_numpy(g["G"]),
                torch.from_numpy(g["y_norm"]), torch.from_numpy(g["is_top"])) for g in fit_graphs]
    ckpt = os.path.join(ckpt_out_dir, f"{variant}_seed{seed}.pth")
    best_val, best_epoch, t0 = -np.inf, 0, time.time()
    history = {"epoch": [], "val_tau": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = [gi for gi, g in enumerate(fit_graphs) for _ in range(int(np.ceil(len(g["y"]) / bsz)))]
        random.shuffle(order)
        for gi in order:
            Xl, Xg, yn, top = tensors[gi]
            idx = torch.multinomial(samplers[gi], min(bsz, len(yn)), replacement=False)
            xl = Xl[idx].to(device) if Xl is not None else None
            out = model(xl, Xg[idx].to(device)).view(-1)
            yb, tb = yn[idx].to(device), top[idx].to(device)
            loss = cfg["alpha_reg"] * F.smooth_l1_loss(out, yb)
            if cfg["alpha_rank"] > 0:
                loss = loss + cfg["alpha_rank"] * top_focused_ranknet_loss(
                    out, yb, tb, top_top_weight=cfg["top_top_weight"],
                    mixed_weight=cfg["mixed_weight"], sigma=cfg["sigma"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            v = mean_val_tau(model, val_graphs, device)
            history["epoch"].append(epoch)
            history["val_tau"].append(v)
            if v > best_val:
                best_val, best_epoch = v, epoch
                torch.save(model.state_dict(), ckpt)
    print(f"    [{variant} | seed {seed}] best val tau@20% {best_val:.4f} @ epoch {best_epoch} "
          f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval(), {"seed": seed, "best_val_tau": best_val, "best_epoch": best_epoch, "history": history}


def evaluate_variant(models, cfg, gbuild, names, label_stats, device):
    """Per-graph ensemble tau/overlap plus per-seed tau."""
    out = {}
    for f in names:
        g = build_graph(f, cfg, gbuild, label_stats)
        preds = [predict(m, g["X"], g["G"], device) for m in models]
        ens = np.mean([zscore(p) for p in preds], axis=0)
        top = get_subset_indices(g["y"], 20)
        out[f] = {
            "tau": safe_tau(ens, g["y"], top),
            "overlap": top_overlap(ens, g["y"], 20),
            "tau_seeds": [safe_tau(p, g["y"], top) for p in preds],
            "overlap_seeds": [top_overlap(p, g["y"], 20) for p in preds],
        }
    return out


def run_variant(variant, args, fit_names, val_names, device):
    group, desc, overrides = VARIANTS[variant]
    cfg = dict(BASE_CONFIG, **overrides)
    print(f"\n=== Variant '{variant}' ({group}): removes {desc} ===", flush=True)
    gbuild = GlobalFeatureBuilder(cfg, fit_names)
    label_stats = None
    if cfg["label_norm"] == "global":
        y_all = np.concatenate([load_labels(f) for f in fit_names])
        label_stats = (float(y_all.mean()), float(y_all.std()))
    fit_graphs = [build_graph(f, cfg, gbuild, label_stats) for f in fit_names]
    val_graphs = [build_graph(f, cfg, gbuild, label_stats) for f in val_names]
    print(f"    global descriptor dim = {gbuild.dim} | local branch = {cfg['use_local']}", flush=True)

    models, seed_info = [], []
    for s in args.seeds:
        m, info = train_seed(variant, s, cfg, gbuild.dim, fit_graphs, val_graphs, args, device)
        models.append(m)
        seed_info.append(info)
    del fit_graphs

    result = {
        "variant": variant, "group": group, "description": desc, "config": cfg,
        "epochs": args.epochs, "seeds": seed_info,
        "val": evaluate_variant(models, cfg, gbuild, val_names, label_stats, device),
        "test": evaluate_variant(models, cfg, gbuild, test_datasets, label_stats, device),
    }
    with open(os.path.join(results_out_dir, f"{variant}.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    t = np.nanmean([r["tau"] for r in result["test"].values()])
    print(f"    -> test tau@20% (ensemble) = {t:.4f}", flush=True)


# =====================================================================================
#                                     Reporting
# =====================================================================================
def load_results():
    res = {}
    for v in VARIANTS:
        p = os.path.join(results_out_dir, f"{v}.json")
        if os.path.exists(p):
            with open(p) as f:
                res[v] = json.load(f)
    if "full" in res and len(res["full"]["seeds"]) > 1:
        # single_model: the full model's seeds, each evaluated alone (no ensemble)
        full = res["full"]
        single = {k: full[k] for k in ("config", "epochs", "seeds")}
        single.update({"variant": "single_model", "group": "Ensembling",
                       "description": "Seed ensemble (single model, mean over seeds)"})
        for split in ("val", "test"):
            single[split] = {f: {"tau": float(np.nanmean(r["tau_seeds"])), "overlap": float(np.nanmean(r["overlap_seeds"])),
                                 "tau_seeds": r["tau_seeds"], "overlap_seeds": r["overlap_seeds"]}
                             for f, r in full[split].items()}
        res["single_model"] = single
    return res


def summarise(res):
    full_tau = np.array([res["full"]["test"][f]["tau"] for f in test_datasets])
    rows = []
    for v in ORDER:
        if v not in res:
            continue
        r = res[v]
        tau = np.array([r["test"][f]["tau"] for f in test_datasets])
        ov = np.array([r["test"][f]["overlap"] for f in test_datasets])
        seed_means = np.nanmean(np.array([r["test"][f]["tau_seeds"] for f in test_datasets]), axis=0)
        val_tau = np.nanmean([x["tau"] for x in r["val"].values()])
        mean_tau = np.nanmean(tau)
        delta = mean_tau - np.nanmean(full_tau)
        ok = ~np.isnan(tau) & ~np.isnan(full_tau)
        diff = tau[ok] - full_tau[ok]
        if v == "full" or np.allclose(diff, 0):
            p = np.nan
        else:
            try:
                p = float(wilcoxon(full_tau[ok], tau[ok]).pvalue)
            except ValueError:
                p = np.nan
        rows.append({
            "variant": v, "group": r["group"], "description": r["description"],
            "test_tau": mean_tau, "seed_mean": float(np.mean(seed_means)), "seed_std": float(np.std(seed_means)),
            "delta": delta, "pct": 100 * delta / np.nanmean(full_tau), "overlap": np.nanmean(ov),
            "val_tau": val_tau, "p": p, "wins": int(np.sum(diff > 1e-9)), "losses": int(np.sum(diff < -1e-9)),
            "per_graph": tau,
        })
    return rows


def write_report(res):
    if "full" not in res:
        print("No 'full' variant results yet - run it first.")
        return
    rows = summarise(res)
    n_test = len(test_datasets)
    full_row = rows[0]

    # ---- CSV / LaTeX
    with open(os.path.join(ablation_dir, "ablation_summary.csv"), "w") as f:
        f.write("variant,group,removed_component,test_tau20_ensemble,test_tau20_seed_mean,test_tau20_seed_std,"
                "delta_vs_full,pct_change,test_overlap20,val_tau20,wilcoxon_p,wins_vs_full,losses_vs_full\n")
        for r in rows:
            f.write(f"{r['variant']},{r['group']},\"{r['description']}\",{fmt(r['test_tau'])},{fmt(r['seed_mean'])},"
                    f"{fmt(r['seed_std'])},{r['delta']:+.4f},{r['pct']:+.2f},{fmt(r['overlap'])},{fmt(r['val_tau'])},"
                    f"{fmt(r['p'])},{r['wins']},{r['losses']}\n")
    with open(os.path.join(ablation_dir, "ablation_per_dataset.csv"), "w") as f:
        f.write("dataset," + ",".join(r["variant"] for r in rows) + "\n")
        for i, d in enumerate(test_datasets):
            f.write(d + "," + ",".join(fmt(r["per_graph"][i]) for r in rows) + "\n")
    with open(os.path.join(ablation_dir, "ablation_table.tex"), "w") as f:
        f.write("% Auto-generated by run_ablation_study.py\n\\begin{table}[t]\n\\centering\n"
                "\\caption{Ablation study of the Hybrid Global-Local WNLGCN. Kendall's $\\tau$ on the top-20\\% "
                f"nodes averaged over {n_test} unseen test networks (ensemble of {len(res['full']['seeds'])} seeds). "
                "$p$: two-sided Wilcoxon signed-rank test against the full model.}\n\\label{tab:ablation}\n"
                "\\begin{tabular}{llcccc}\n\\hline\nGroup & Variant (removed) & $\\tau_{20\\%}$ & $\\Delta$ & "
                "Overlap$_{20\\%}$ & $p$ \\\\\n\\hline\n")
        for r in rows:
            name = r["description"] if r["variant"] != "full" else "\\textbf{Full model}"
            tau = f"\\textbf{{{r['test_tau']:.4f}}}" if r["variant"] == "full" else f"{r['test_tau']:.4f}"
            delta = "--" if r["variant"] == "full" else f"{r['delta']:+.4f}"
            p = "--" if np.isnan(r["p"]) else (f"{r['p']:.3f}" if r["p"] >= 0.001 else "$<$0.001")
            f.write(f"{r['group']} & {name} & {tau} & {delta} & {r['overlap']:.4f} & {p} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # ---- PDF
    pdf_path = os.path.join(ablation_dir, "ablation_report_hybrid.pdf")
    full_res = res["full"]
    with PdfPages(pdf_path) as pdf:
        add_text_page(pdf, "Ablation Study: Hybrid Global-Local WNLGCN (Weighted Networks)", [
            "### 1. Protocol",
            f"Each variant removes exactly one component and is retrained from scratch with the identical protocol: "
            f"{len(full_res['seeds'])} seeds ({', '.join(str(s['seed']) for s in full_res['seeds'])}), {full_res['epochs']} epochs, "
            f"AdamW + warm-up/cosine schedule, best checkpoint selected on the held-out validation graphs "
            f"({', '.join(DEFAULT_VAL_DATASETS)}), and per-graph z-score ensembling over seeds. Retraining (rather than "
            "zeroing inputs of one trained network) measures what each component contributes when the network is free "
            "to adapt to its absence.",
            "### 2. Metrics",
            f"Kendall's tau over the true top-20% SIR spreaders and top-20% overlap, averaged over the {n_test} unseen test "
            "networks. 'Seed mean +/- std' is the test tau of individual (non-ensembled) models, showing run-to-run variance. "
            "The p-value is a two-sided Wilcoxon signed-rank test of per-network tau against the full model; W/L counts the "
            "test networks on which the variant beats / loses to the full model.",
            "### 3. Variant groups",
            "Architecture: removes a branch or module (global branch, local CNN branch, channel attention, gated fusion, linear shortcut).",
            "Global descriptors: removes a descriptor family (SIR-aware, spectral, structural), the SIGN neighbour aggregation, "
            "or the within-graph rank normalisation (replaced by global training-set standardisation).",
            "Training objective: removes the ranking loss, the top-focused pair weighting, the top-20% over-sampling, or the "
            "per-graph label standardisation.",
            "Ensembling: the full model's seeds evaluated individually, quantifying the gain from the seed ensemble.",
            "### 4. Note on SIR-aware descriptors",
            "The SIR-aware descriptors use the infection rate beta of the ground-truth simulation. The 'no_dynamics_feats' "
            "variant shows the performance the hybrid architecture reaches without any knowledge of beta.",
        ])

        headers = ["Variant (removed component)", "Test tau@20%", "Seed mean +/- std", "Delta vs Full",
                   "% change", "Overlap@20%", "Val tau@20%", "Wilcoxon p", "W / L"]
        data, statuses = [], []
        for r in rows:
            data.append([f"{r['variant']}\n({r['description']})", fmt(r["test_tau"]),
                         f"{r['seed_mean']:.4f} +/- {r['seed_std']:.4f}",
                         "--" if r["variant"] == "full" else f"{r['delta']:+.4f}",
                         "--" if r["variant"] == "full" else f"{r['pct']:+.2f}%",
                         fmt(r["overlap"]), fmt(r["val_tau"]),
                         "--" if np.isnan(r["p"]) else f"{r['p']:.4f}",
                         "--" if r["variant"] == "full" else f"{r['wins']} / {r['losses']}"])
            statuses.append("val" if r["variant"] == "full" else "train")
        biggest = min(rows[1:], key=lambda r: r["delta"]) if len(rows) > 1 else full_row
        add_table_page(
            pdf, "Table 1: Ablation Summary (Unseen Test Networks)", headers, data, statuses,
            [set() for _ in rows],
            "Each row retrains the model without one component. Negative deltas mean the component helps; a small "
            "Wilcoxon p (< 0.05) means the drop is consistent across test networks rather than seed noise.",
            f"The full model (yellow row) reaches tau = {fmt(full_row['test_tau'])}. The largest drop comes from removing "
            f"'{biggest['description']}' ({biggest['delta']:+.4f}, {biggest['pct']:+.2f}%), identifying it as the most "
            "important ingredient of the Hybrid Global-Local design.", fontsize=6.8, first_col_frac=0.24,
        )

        for title, groups in (("Table 2: Per-Network Kendall Tau@20% - Architecture & Ensembling",
                               ("Reference", "Architecture", "Ensembling")),
                              ("Table 3: Per-Network Kendall Tau@20% - Global Descriptors",
                               ("Reference", "Global descriptors")),
                              ("Table 4: Per-Network Kendall Tau@20% - Training Objective",
                               ("Reference", "Training objective"))):
            sel = [r for r in rows if r["group"] in groups]
            if len(sel) < 2:
                continue
            hdr = ["Test Network"] + [r["variant"] for r in sel]
            dat = [[d] + [fmt(r["per_graph"][i]) for r in sel] for i, d in enumerate(test_datasets)]
            dat.append(["MEAN"] + [fmt(r["test_tau"]) for r in sel])
            bests = [best_indices([r["per_graph"][i] for r in sel]) for i in range(len(test_datasets))]
            bests.append(best_indices([r["test_tau"] for r in sel]))
            add_table_page(pdf, title, hdr, dat, ["test"] * len(dat), bests,
                           "Per-network breakdown of the ensemble Kendall tau on the top-20% spreaders for this group of variants.",
                           "Best value per network highlighted in blue. A component is justified when the full model wins on "
                           "most networks, not only on average.", fontsize=6.8)

        # delta bar chart
        abl = rows[1:]
        if abl:
            fig, ax = plt.subplots(figsize=(12, 8.5))
            colors = {"Architecture": "#2E75B6", "Global descriptors": "#70AD47",
                      "Training objective": "#ED7D31", "Ensembling": "#7F7F7F"}
            y = np.arange(len(abl))
            ax.barh(y, [r["delta"] for r in abl], color=[colors.get(r["group"], "#999999") for r in abl],
                    xerr=[r["seed_std"] for r in abl], capsize=3)
            ax.set_yticks(y)
            ax.set_yticklabels([f"{r['variant']}{' *' if not np.isnan(r['p']) and r['p'] < 0.05 else ''}" for r in abl])
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel(f"Change in mean test Kendall tau@20% vs full model ({fmt(full_row['test_tau'])})")
            ax.set_title("Figure 1: Contribution of Each Component (negative = component helps; * p < 0.05)",
                         fontsize=13, weight="bold", color="#1F4E79")
            ax.grid(axis="x", alpha=0.3)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=c, label=g) for g, c in colors.items()], loc="lower right")
            fig.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        # validation curves of the full model
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for v, ls in (("full", "-"), ("local_only", "--"), ("global_only", ":"), ("no_dynamics_feats", "-.")):
            if v in res:
                h = res[v]["seeds"][0]["history"]
                ax.plot(h["epoch"], h["val_tau"], ls, label=f"{v} (seed {res[v]['seeds'][0]['seed']})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation tau@20%")
        ax.set_title("Figure 2: Validation Curves of Key Variants", fontsize=13, weight="bold", color="#1F4E79")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    print("\n" + "=" * 110)
    print(f"{'Variant':<22} {'Test tau':>9} {'Seed mean+/-std':>18} {'Delta':>9} {'%':>8} {'Overlap':>8} {'p':>8} {'W/L':>6}")
    print("-" * 110)
    for r in rows:
        print(f"{r['variant']:<22} {r['test_tau']:9.4f} {r['seed_mean']:9.4f}+/-{r['seed_std']:.4f} {r['delta']:+9.4f} "
              f"{r['pct']:+7.2f}% {r['overlap']:8.4f} {fmt(r['p']):>8} {r['wins']:>2}/{r['losses']:<2}")
    print("=" * 110)
    print(f"PDF:   {pdf_path}")
    print(f"CSV:   {os.path.join(ablation_dir, 'ablation_summary.csv')}, ablation_per_dataset.csv")
    print(f"LaTeX: {os.path.join(ablation_dir, 'ablation_table.tex')}")


# =====================================================================================
def main():
    p = argparse.ArgumentParser(description="Ablation study for the Hybrid Global-Local WNLGCN")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--force", action="store_true", help="retrain variants that already have results")
    p.add_argument("--report_only", action="store_true", help="only rebuild the report from saved results")
    args = p.parse_args()

    os.makedirs(results_out_dir, exist_ok=True)
    os.makedirs(ckpt_out_dir, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.report_only:
        val_names = [f for f in train_datasets if f in DEFAULT_VAL_DATASETS]
        fit_names = [f for f in train_datasets if f not in DEFAULT_VAL_DATASETS]
        print(f"Device: {device} | fit graphs: {len(fit_names)} | val graphs: {val_names} | test graphs: {len(test_datasets)}")
        variants = ["full"] + [v for v in args.variants if v != "full"]   # full first: it is the reference
        t0 = time.time()
        for v in variants:
            if v not in args.variants and os.path.exists(os.path.join(results_out_dir, "full.json")):
                continue
            done_path = os.path.join(results_out_dir, f"{v}.json")
            if os.path.exists(done_path) and not args.force:
                with open(done_path) as f:
                    prev = json.load(f)
                if prev["epochs"] == args.epochs and [s["seed"] for s in prev["seeds"]] == args.seeds:
                    print(f"Skipping '{v}' (results exist; use --force to retrain)")
                    continue
                print(f"Retraining '{v}': saved results used different seeds/epochs")
            run_variant(v, args, fit_names, val_names, device)
            print(f"    elapsed total: {(time.time() - t0) / 60:.1f} min", flush=True)

    write_report(load_results())


if __name__ == "__main__":
    main()
