"""
Train the Hybrid Global-Local WNLGCN on weighted networks.

  * local branch  : cached 6-channel 41x41 neighbourhood patches (same as Deep 6-Ch WNLGCN)
  * global branch : graph-wide, SIR-aware, rank-normalised descriptors (see hybrid_common.py)
  * loss          : Huber on per-graph standardised SIR labels + top-focused RankNet
  * sampling      : per-graph batches, top-20% spreaders over-sampled
  * selection     : best epoch by mean Kendall-tau@top20% on held-out VALIDATION train graphs
                    (test graphs are never looked at during training)
  * ensemble      : one model per seed; evaluate_and_report.py averages them

Usage:
    python train_hybrid.py                     # 3 seeds x 60 epochs
    python train_hybrid.py --seeds 42 --epochs 60   # quick run
"""
import os
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from hybrid_common import (
    HybridWNLGCN, train_datasets, DEFAULT_VAL_DATASETS, checkpoint_dir, get_global_features,
    load_local_X, load_labels, top_focused_ranknet_loss, get_subset_indices, safe_tau,
    predict, N_GLOBAL_FEATURES,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graphs(names):
    graphs = []
    for f in names:
        X = load_local_X(f)
        G, _ = get_global_features(f)
        y = load_labels(f)
        k = max(1, int(len(y) * 0.2))
        top_thr = np.partition(y, -k)[-k]
        graphs.append({
            "name": f,
            "X": X,
            "G": G,
            "y": y,
            "y_norm": ((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32),   # per-graph standardisation
            "is_top": (y >= top_thr).astype(np.float32),
        })
    return graphs


def evaluate_graphs(model, graphs, device):
    taus = {}
    for g in graphs:
        pred = predict(model, g["X"], g["G"], device)
        taus[g["name"]] = safe_tau(pred, g["y"], get_subset_indices(g["y"], 20))
    return taus


def train_one_seed(seed, train_graphs, val_graphs, args, device):
    set_seed(seed)
    model = HybridWNLGCN(n_global=N_GLOBAL_FEATURES, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = sum(int(np.ceil(len(g["y"]) / args.batch_size)) for g in train_graphs)
    total_steps = steps_per_epoch * args.epochs
    warmup = steps_per_epoch * 3
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    # sampling weights: top-20% spreaders drawn TOP_OVERSAMPLE x more often
    samplers = [torch.tensor(1.0 + (args.top_oversample - 1.0) * g["is_top"]) for g in train_graphs]
    tensors = [(torch.from_numpy(g["X"]), torch.from_numpy(g["G"]),
                torch.from_numpy(g["y_norm"]), torch.from_numpy(g["is_top"])) for g in train_graphs]

    history = {"epoch": [], "loss": [], "val_tau": []}
    best_val, best_epoch = -np.inf, 0
    ckpt_path = os.path.join(checkpoint_dir, f"hybrid_seed{seed}.pth")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = []
        for gi, g in enumerate(train_graphs):
            order += [gi] * int(np.ceil(len(g["y"]) / args.batch_size))
        random.shuffle(order)

        run_loss, run_n = 0.0, 0
        for gi in order:
            Xl, Xg, yn, top = tensors[gi]
            bs = min(args.batch_size, len(yn))
            idx = torch.multinomial(samplers[gi], bs, replacement=False)
            xl, xg = Xl[idx].to(device), Xg[idx].to(device)
            yb, tb = yn[idx].to(device), top[idx].to(device)

            out = model(xl, xg).view(-1)
            reg = F.smooth_l1_loss(out, yb)
            rank = top_focused_ranknet_loss(out, yb, tb, top_top_weight=args.top_top_weight,
                                            mixed_weight=args.mixed_weight, sigma=args.sigma)
            loss = args.alpha_reg * reg + args.alpha_rank * rank

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            run_loss += loss.item() * bs
            run_n += bs

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            taus = evaluate_graphs(model, val_graphs, device)
            val_tau = float(np.nanmean(list(taus.values())))
            history["epoch"].append(epoch)
            history["loss"].append(run_loss / run_n)
            history["val_tau"].append(val_tau)
            flag = ""
            if val_tau > best_val:
                best_val, best_epoch = val_tau, epoch
                torch.save(model.state_dict(), ckpt_path)
                flag = "  <- best"
            print(f"[seed {seed}] Epoch {epoch:3d}/{args.epochs} | loss {run_loss / run_n:.4f} | "
                  f"val tau@20% {val_tau:.4f} | lr {scheduler.get_last_lr()[0]:.5f} | "
                  f"{time.time() - t0:.0f}s{flag}", flush=True)

    print(f"[seed {seed}] best val tau@20% = {best_val:.4f} at epoch {best_epoch} -> {ckpt_path}")
    return {"seed": seed, "best_val_tau": best_val, "best_epoch": best_epoch,
            "checkpoint": os.path.basename(ckpt_path), "history": history}


def main():
    p = argparse.ArgumentParser(description="Train Hybrid Global-Local WNLGCN")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--alpha_reg", type=float, default=0.5)
    p.add_argument("--alpha_rank", type=float, default=1.0)
    p.add_argument("--top_top_weight", type=float, default=4.0)
    p.add_argument("--mixed_weight", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--top_oversample", type=float, default=3.0)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--val", nargs="+", default=DEFAULT_VAL_DATASETS,
                   help="training-folder graphs held out for checkpoint selection")
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    val_names = [f for f in train_datasets if f in args.val]
    fit_names = [f for f in train_datasets if f not in args.val]
    print(f"Device: {device}")
    print(f"Fitting on {len(fit_names)} graphs, validating on {len(val_names)}: {val_names}")

    print("Loading cached local patches + global descriptors (computed on first run)...")
    train_graphs = load_graphs(fit_names)
    val_graphs = load_graphs(val_names)
    print(f"Training nodes: {sum(len(g['y']) for g in train_graphs)} | "
          f"Validation nodes: {sum(len(g['y']) for g in val_graphs)}")

    runs = [train_one_seed(s, train_graphs, val_graphs, args, device) for s in args.seeds]

    summary = {"config": vars(args), "fit_datasets": fit_names, "val_datasets": val_names, "runs": runs}
    with open(os.path.join(checkpoint_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining complete. Ensemble members:")
    for r in runs:
        print(f"  seed {r['seed']}: val tau@20% {r['best_val_tau']:.4f} (epoch {r['best_epoch']})")
    print(f"Summary written to {os.path.join(checkpoint_dir, 'training_summary.json')}")
    print("Next: python evaluate_and_report.py")


if __name__ == "__main__":
    main()
