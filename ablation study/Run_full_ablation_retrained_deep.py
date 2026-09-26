"""
run_full_ablation_retrained_deep.py

PROPER ablation study for the REVISED DEEP 6-CHANNEL WNLGCN model: trains a
SEPARATE model from scratch for each configuration, rather than occluding
inputs on one shared trained checkpoint at inference time (which is what
run_full_ablation_train_and_test.py did, and which produces biased/inflated
deltas -- see explanation below).

Architecture and training setup here match `train_and_evaluate.py` exactly
(the actual executable script, used as ground truth over other documents
that gave inconsistent hyperparameters): 3-layer MLP head (6400 -> 256 ->
64 -> 1), GELU, BatchNorm1d, dropout=0.2, AdamW (lr=0.002, weight_decay=
1e-2), CosineAnnealingLR (300 epochs, eta_min=1e-5), hybrid MSE + top-heavy
pairwise ranking loss (alpha=1.5, margin=0.3, top_weight=3.0).

------------------------------------------------------------------------
CONFIGURATIONS
------------------------------------------------------------------------
  (a) full              - baseline: attention ON, hybrid loss, standard features
  (b) no_polarity        - cost-type networks NOT inverted (1/w skipped) at
                            feature-extraction time
  (c) unweighted_smooth  - smoothing step (Eqs. 3-4) uses BINARY adjacency
                            instead of weighted W (base NLI/NGI still weighted)
  (d) single_smooth      - one smoothing round instead of two (NLI^(3)/NGI^(3)
                            channels set equal to NLI^(2)/NGI^(2), i.e. the
                            second propagation step is skipped)
  (e) no_attention       - channel-attention module removed from the
                            architecture entirely (not just skipped at
                            inference -- it never exists, so the rest of the
                            network is trained without it from the start)
  (f) mse_only           - ranking-loss term removed; trained with plain MSE
  (g) shallow_mlp        - MLP capacity ablation: swaps the deep 3-layer head
                            for the original submitted paper's 2-layer,
                            8-unit head, everything else (AdamW, cosine LR,
                            hybrid loss, attention) held at the deep model's
                            settings -- isolates what the deeper head alone
                            contributes, separate from the optimizer/loss
                            changes

------------------------------------------------------------------------
IMPORTANT -- WHAT YOU MUST DO BEFORE RUNNING THIS SCRIPT
------------------------------------------------------------------------
(a), (e), (f), (g) are pure model/training-level changes and use your
EXISTING cached features (_weighted_local_norm_X.npy etc.) -- nothing to
prepare.

(b), (c), (d) are DATA-level ablations: the six-channel feature tensors
themselves must be regenerated with a modified feature-extraction pipeline,
because polarity correction, smoothing-adjacency weighting, and smoothing
depth are all properties of how the *features* are built, not the model.

I do not have your feature-extraction script (wherever NLI/NGI and the
six-channel tensors are actually constructed -- referenced elsewhere as
`train_wnlgcn_multigraph.py`) in front of me, so I have not reimplemented
it here; guessing at it risks silently producing wrong features and a
confidently-reported wrong number. Instead, generate these three cache
sets yourself with modified copies of your existing feature script:

  (b) no_polarity:
      Skip the `w = 1/w` inversion step for cost-type ("adversarial")
      networks entirely -- treat every network as "bigger = stronger."
      Save with suffix: _nopolarity_local_norm_X.npy (+ _y.npy)

  (c) unweighted_smooth:
      In Eqs. 3-4 (the NLI/NGI smoothing step only), replace the weighted
      adjacency matrix W with the BINARY adjacency matrix before computing
      NLI^(2)/NLI^(3)/NGI^(2)/NGI^(3). Leave base NLI/NGI untouched.
      Save with suffix: _unweightedsmooth_local_norm_X.npy (+ _y.npy)

  (d) single_smooth:
      Compute NLI^(2)/NGI^(2) normally, but set NLI^(3) = NLI^(2) and
      NGI^(3) = NGI^(2) instead of computing a second smoothing round
      (keeps the 6-channel tensor shape the architecture expects).
      Save with suffix: _singlesmooth_local_norm_X.npy (+ _y.npy)

FEATURE_SUFFIX below maps each config to its expected cache suffix. Point
RESULTS_DIR at wherever you saved these, or generate them directly into
your existing results_dir.

------------------------------------------------------------------------
WHY THIS IS DIFFERENT FROM THE OCCLUSION-BASED SCRIPT
------------------------------------------------------------------------
The previous script loaded ONE checkpoint (trained with attention active
and all six channels populated) and zeroed channels / skipped layers at
inference. That checkpoint's conv/fc weights were never trained to cope
with missing channels or a missing attention layer, so those numbers
measure "how badly does this specific model break under out-of-
distribution inputs," not "how much does this component actually
contribute." Every config here is trained from scratch under its own
condition, with the same seed, so the comparison is fair.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import kendalltau

# ============================================================================
# Paths -- EDIT THESE to match your directory layout
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WEIGHTED_MODEL_DIR = os.path.join(BASE_DIR, "Third_Phase", "weighted models")
RESULTS_DIR = os.path.join(WEIGHTED_MODEL_DIR, "results")
DATASETS_BASE_DIR = os.path.join(BASE_DIR, "Datasets")
TRAIN_FOLDER = os.path.join(DATASETS_BASE_DIR, "weighted Datasets", "train")
TEST_FOLDER = os.path.join(DATASETS_BASE_DIR, "weighted Datasets", "test")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "ablation_checkpoints_deep")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
EPOCHS = 300
LR = 0.002
WEIGHT_DECAY = 1e-2
ETA_MIN = 1e-5
ALPHA_RANK = 1.5
RANK_MARGIN = 0.3
TOP_WEIGHT = 3.0
TOP_PCT = 20  # evaluate top-20% spreaders, matches the paper's protocol

# Maps each config name to the cache-file suffix it reads features from.
# "_weighted" = your existing, already-generated caches.
# The three data-level configs you must generate yourself -- see docstring.
FEATURE_SUFFIX = {
    "full": "_weighted",
    "no_attention": "_weighted",
    "mse_only": "_weighted",
    "shallow_mlp": "_weighted",
    "no_polarity": "_nopolarity",
    "unweighted_smooth": "_unweightedsmooth",
    "single_smooth": "_singlesmooth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Model -- matches the REVISED DEEP model's architecture exactly
# (train_and_evaluate.py), with two structural toggles for ablation:
#   use_attention   -> for the (e) no_attention config
#   deep_head       -> for the (g) shallow_mlp config (2-layer, 8-unit head)
# ============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels=6, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class DeepWNLGCN(nn.Module):
    """
    use_attention=False -> attention submodule never constructed (fair
    ablation: the rest of the network is trained without it from the
    start, not just skipped at inference on a model that expects it).

    deep_head=False -> replaces the 3-layer (256 -> 64 -> 1) head with
    the original submitted paper's 2-layer (8 -> 1) head, everything
    else (AdamW, cosine LR, hybrid loss, attention) unchanged from the
    deep model's settings. Isolates the head-capacity contribution
    specifically, separate from the optimizer/schedule/loss changes.
    """

    def __init__(self, use_attention=True, deep_head=True):
        super().__init__()
        self.use_attention = use_attention
        self.deep_head = deep_head

        if use_attention:
            self.attention = ChannelAttention(6, reduction=2)
        self.conv1 = nn.Conv2d(6, 16, kernel_size=2)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)

        if deep_head:
            self.fc1 = nn.Linear(16 * 20 * 20, 256)
            self.bn1d = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 64)
            self.dropout = nn.Dropout(p=0.2)
            self.fc3 = nn.Linear(64, 1)
        else:
            self.fc1 = nn.Linear(16 * 20 * 20, 8)
            self.dropout = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        if self.use_attention:
            x = self.attention(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if self.deep_head:
            x = self.fc1(x)
            if x.size(0) > 1:
                x = self.bn1d(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = F.gelu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        else:
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        return x


# ============================================================================
# Loss -- hybrid MSE + top-heavy pairwise ranking loss, matching
# train_and_evaluate.py exactly, with a flag to drop the ranking term
# for the (f) mse_only config.
# ============================================================================
def top_heavy_pairwise_ranking_loss(pred, y, is_top, margin=RANK_MARGIN, top_weight=TOP_WEIGHT):
    pred = pred.view(-1)
    y = y.view(-1)
    is_top = is_top.view(-1)

    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)
    sign_y = torch.sign(diff_y)

    mask = diff_y.abs() > 1e-6
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pair_weight = 1.0 + (top_weight - 1.0) * torch.max(is_top.unsqueeze(1), is_top.unsqueeze(0))
    losses = F.relu(margin - sign_y * diff_pred)
    weighted_losses = losses * pair_weight
    return weighted_losses[mask].mean()


def compute_loss(outputs, batch_y, batch_is_top, use_rank_loss):
    mse = F.mse_loss(outputs, batch_y)
    if not use_rank_loss:
        return mse
    rank = top_heavy_pairwise_ranking_loss(outputs, batch_y, batch_is_top)
    return mse + ALPHA_RANK * rank


# ============================================================================
# Data loading
# ============================================================================
def discover_datasets(folder, suffix, results_dir=RESULTS_DIR):
    datasets = []
    if os.path.exists(folder):
        for f in sorted(os.listdir(folder)):
            if f in ["karate.txt", "cargoshipsBB.txt"]:
                continue
            cache_x = os.path.join(results_dir, f"{f}{suffix}_local_norm_X.npy")
            if os.path.exists(cache_x):
                datasets.append(f)
    return datasets


def load_graph_loaders(filenames, suffix, results_dir=RESULTS_DIR):
    """One DataLoader per graph, with global top-20% flags per graph,
    matching train_and_evaluate.py's per-graph batching so ranking pairs
    never cross networks."""
    all_y = []
    per_graph = []
    for fn in filenames:
        X = np.load(os.path.join(results_dir, f"{fn}{suffix}_local_norm_X.npy"))
        y = np.load(os.path.join(results_dir, f"{fn}{suffix}_y.npy")).flatten()
        k = max(1, int(len(y) * 0.2))
        top_threshold = np.partition(y, -k)[-k]
        is_top = (y >= top_threshold).astype(np.float32)
        all_y.append(y)
        per_graph.append((fn, X, y, is_top))

    y_concat = np.concatenate(all_y, axis=0)
    y_mean, y_std = y_concat.mean(), y_concat.std()

    loaders = []
    for fn, X, y, is_top in per_graph:
        y_norm = (y.reshape(-1, 1) - y_mean) / (y_std + 1e-6)
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(is_top, dtype=torch.float32).view(-1, 1),
        )
        bs = min(256, len(ds))
        loaders.append((fn, DataLoader(ds, batch_size=bs, shuffle=True)))
    return loaders, y_mean, y_std


def safe_tau(x, y, pct=TOP_PCT):
    n = len(y)
    k = max(1, int(pct / 100.0 * n))
    idx = np.argsort(y)[::-1][:k]
    xs, ys = x[idx], y[idx]
    if len(xs) < 2 or np.all(xs == xs[0]) or np.all(ys == ys[0]):
        return np.nan
    try:
        tau, _ = kendalltau(xs, ys)
        return tau if not np.isnan(tau) else np.nan
    except Exception:
        return np.nan


# ============================================================================
# Train one configuration from scratch
# ============================================================================
def train_one_config(config_name, use_attention, deep_head, use_rank_loss):
    suffix = FEATURE_SUFFIX[config_name]
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"wnlgcn_deep_ablation_{config_name}.pth")

    set_seed(SEED)  # same seed across configs so differences reflect the
                     # ablated component, not random initialization noise

    train_files = discover_datasets(TRAIN_FOLDER, suffix)
    if len(train_files) == 0:
        print(f"  !! No cached features found for suffix '{suffix}' in {RESULTS_DIR}. "
              f"Skipping '{config_name}' -- see docstring for what to generate.")
        return None

    train_loaders, y_mean, y_std = load_graph_loaders(train_files, suffix)

    model = DeepWNLGCN(use_attention=use_attention, deep_head=deep_head).to(DEVICE)

    # Check if checkpoint already exists to avoid unnecessary retraining
    if os.path.exists(ckpt_path):
        print(f"\n{'='*100}\nFound existing trained checkpoint for '{config_name}' at:\n  {ckpt_path}\nSkipping retraining and loading saved model weights!\n{'='*100}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        return {"model": model, "y_mean": y_mean, "y_std": y_std, "suffix": suffix}

    print(f"\n{'='*100}\nTraining config: {config_name}  "
          f"(attention={use_attention}, deep_head={deep_head}, "
          f"rank_loss={use_rank_loss}, feature_suffix='{suffix}')\n{'='*100}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(range(len(train_loaders)))
        random.shuffle(order)
        epoch_loss, n_seen = 0.0, 0
        for idx in order:
            _, loader = train_loaders[idx]
            for batch_X, batch_y, batch_is_top in loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                batch_is_top = batch_is_top.to(DEVICE)
                optimizer.zero_grad()
                out = model(batch_X)
                loss = compute_loss(out, batch_y, batch_is_top, use_rank_loss)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
                n_seen += batch_X.size(0)
        scheduler.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{EPOCHS} | loss {epoch_loss / max(n_seen,1):.6f} "
                  f"| lr {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"  saved -> {ckpt_path}")

    return {"model": model, "y_mean": y_mean, "y_std": y_std, "suffix": suffix}


# ============================================================================
# Evaluate one trained config on train + held-out test networks
# ============================================================================
def evaluate_config(trained):
    model, suffix = trained["model"], trained["suffix"]
    model.eval()

    def eval_group(folder):
        files = discover_datasets(folder, suffix)
        rows = []
        for fn in files:
            X = np.load(os.path.join(RESULTS_DIR, f"{fn}{suffix}_local_norm_X.npy"))
            y = np.load(os.path.join(RESULTS_DIR, f"{fn}{suffix}_y.npy")).flatten()
            with torch.no_grad():
                pred = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
            tau = safe_tau(pred, y)
            rows.append((fn, tau))
        return rows

    train_rows = eval_group(TRAIN_FOLDER)
    test_rows = eval_group(TEST_FOLDER)
    return train_rows, test_rows


# ============================================================================
# Main
# ============================================================================
CONFIGS = [
    # (name,               use_attention, deep_head, use_rank_loss)
    ("full",                True,  True,  True),
    ("no_attention",        False, True,  True),
    ("mse_only",            True,  True,  False),
    ("shallow_mlp",         True,  False, True),
    ("no_polarity",         True,  True,  True),
    ("unweighted_smooth",   True,  True,  True),
    ("single_smooth",       True,  True,  True),
]


def main():
    all_results = {}

    for name, use_attention, deep_head, use_rank_loss in CONFIGS:
        trained = train_one_config(name, use_attention, deep_head, use_rank_loss)
        if trained is None:
            continue
        train_rows, test_rows = evaluate_config(trained)
        all_results[name] = {"train": train_rows, "test": test_rows}

    if not all_results:
        print("\nNo configs produced results -- check that your feature caches exist.")
        return

    print("\n" + "=" * 110)
    print(" ABLATION SUMMARY -- HELD-OUT TEST NETWORKS (per-network)")
    print("=" * 110)
    for name, res in all_results.items():
        print(f"\n--- {name} ---")
        taus = [t for _, t in res["test"]]
        for fn, t in res["test"]:
            print(f"  {fn:<32} tau={t:.4f}" if not np.isnan(t) else f"  {fn:<32} tau=nan")
        print(f"  {'AVERAGE':<32} tau={np.nanmean(taus):.4f}")

    full_test_avg = np.nanmean([t for _, t in all_results.get("full", {}).get("test", [])])

    print("\n" + "=" * 110)
    print(" SUMMARY MATRIX -- ALL CONFIGS (RETRAINED FROM SCRATCH, DEEP MODEL BACKBONE)")
    print("=" * 110)
    print(f"{'Config':<22} | {'Train avg tau':<15} | {'Test avg tau':<15} | {'Delta vs full (test)':<22} | {'% change':<10}")
    print("-" * 110)
    for name, res in all_results.items():
        tr_avg = np.nanmean([t for _, t in res["train"]])
        te_avg = np.nanmean([t for _, t in res["test"]])
        delta = te_avg - full_test_avg
        pct = (delta / full_test_avg * 100.0) if full_test_avg else 0.0
        delta_str = "baseline" if name == "full" else f"{delta:+.4f}"
        pct_str = "" if name == "full" else f"{pct:+.2f}%"
        print(f"{name:<22} | {tr_avg:<15.4f} | {te_avg:<15.4f} | {delta_str:<22} | {pct_str:<10}")
    print("=" * 110)


if __name__ == "__main__":
    main()