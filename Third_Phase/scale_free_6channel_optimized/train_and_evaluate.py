import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import kendalltau

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Directory Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
weighted_model_dir = os.path.abspath(os.path.join(script_dir, "..", "weighted models"))
results_dir = os.path.join(weighted_model_dir, "results")

# Datasets folders
datasets_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
train_folder = os.path.join(datasets_base_dir, "scalefree networks", "train")
test_folder = os.path.join(datasets_base_dir, "scalefree networks", "test")

# Discover train/test files and verify they are cached in results_dir
train_datasets = []
for f in sorted(os.listdir(train_folder)):
    if os.path.exists(os.path.join(results_dir, f"{f}_weighted_local_norm_X.npy")):
        train_datasets.append(f)

test_datasets = []
for f in sorted(os.listdir(test_folder)):
    if os.path.exists(os.path.join(results_dir, f"{f}_weighted_local_norm_X.npy")):
        test_datasets.append(f)

print(f"Loaded {len(train_datasets)} training datasets and {len(test_datasets)} testing datasets.")

# ---- WNLGCN Model Definition (Optimized 6 Channels) ----
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

class WNLGCN(nn.Module):
    def __init__(self):
        super(WNLGCN, self).__init__()
        self.attention = ChannelAttention(6, reduction=2)
        self.conv1 = nn.Conv2d(6, 16, kernel_size=2)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 20 * 20, 32)  # Increased from 8 to 32
        self.dropout = nn.Dropout(p=0.2)        # Reduced from 0.5 to 0.2
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.attention(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---- Top-Heavy Pairwise Ranking Loss ----
def top_heavy_pairwise_ranking_loss(pred, y, is_top, margin=0.3, top_weight=3.0):
    pred = pred.view(-1)
    y = y.view(-1)
    is_top = is_top.view(-1)

    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)   # (B, B)
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)             # (B, B)
    sign_y = torch.sign(diff_y)

    mask = diff_y.abs() > 1e-6
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Weight pairs using globally-defined top spreaders status
    pair_weight = 1.0 + (top_weight - 1.0) * torch.max(is_top.unsqueeze(1), is_top.unsqueeze(0))
    losses = F.relu(margin - sign_y * diff_pred)
    weighted_losses = losses * pair_weight

    return weighted_losses[mask].mean()

# ---- Evaluation Helpers ----
def get_subset_indices(labels, pct):
    n = len(labels)
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_X = []
    all_y = []
    all_is_top = []

    # 1. Load cached training data directly
    for filename in train_datasets:
        cache_x_path = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_weighted_y.npy")

        X = np.load(cache_x_path)
        y = np.load(cache_y_path).flatten()
        
        # Calculate top 20% indices on the entire graph (globally)
        k = max(1, int(len(y) * 0.2))
        top_threshold = np.partition(y, -k)[-k]
        is_top = (y >= top_threshold).astype(np.float32)

        all_X.append(X)
        all_y.append(y)
        all_is_top.append(is_top)

    # Calculate training standardization metrics
    y_all_concat = np.concatenate(all_y, axis=0)
    y_mean = y_all_concat.mean()
    y_std = y_all_concat.std()

    # Loaders per graph
    per_graph_loaders = []
    for X_i, y_i, is_top_i in zip(all_X, all_y, all_is_top):
        y_i_norm = (y_i.reshape(-1, 1) - y_mean) / (y_std + 1e-6)
        X_t = torch.tensor(X_i, dtype=torch.float32)
        y_t = torch.tensor(y_i_norm, dtype=torch.float32)
        is_top_t = torch.tensor(is_top_i, dtype=torch.float32).view(-1, 1)
        
        ds = TensorDataset(X_t, y_t, is_top_t)
        bs = min(256, len(ds))
        per_graph_loaders.append(DataLoader(ds, batch_size=bs, shuffle=True))

    print(f"Data Preparation Complete. Total Training Samples: {sum(len(l.dataset) for l in per_graph_loaders)}")

    # 2. Train Model
    model = WNLGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    mse_criterion = nn.MSELoss()

    ALPHA_RANK = 1.0
    MARGIN = 0.3
    epochs = 300

    print("Starting WNLGCN Training with Top-Heavy Ranking Loss...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        loader_order = list(range(len(per_graph_loaders)))
        random.shuffle(loader_order)

        for gidx in loader_order:
            for batch_X, batch_y, batch_is_top in per_graph_loaders[gidx]:
                batch_X, batch_y, batch_is_top = batch_X.to(device), batch_y.to(device), batch_is_top.to(device)
                optimizer.zero_grad()

                outputs = model(batch_X)
                mse_loss = mse_criterion(outputs, batch_y)
                rank_loss = top_heavy_pairwise_ranking_loss(outputs, batch_y, batch_is_top, margin=MARGIN, top_weight=3.0)
                loss = mse_loss + ALPHA_RANK * rank_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                n_batches += batch_X.size(0)

        epoch_loss /= n_batches
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.6f}")

    # Save model weights
    model_save_path = os.path.join(script_dir, "wnlgcn_6ch_optimized.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved trained model weights to: {model_save_path}")

    print("Training finished. Evaluating on Test Datasets...")
    model.eval()

    # 3. Evaluation
    print("\n" + "="*100)
    print(f"{'Network Dataset':<30} | {'Ours (WNLGCN-6Ch)':<20} | {'W-Eigenvector':<15} | {'Strength (W-Deg)':<15} | {'W-Closeness':<15}")
    print("="*100)

    for filename in test_datasets:
        cache_x_path = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_weighted_y.npy")
        cache_weig = os.path.join(results_dir, f"{filename}_weig.npy")
        cache_wdeg = os.path.join(results_dir, f"{filename}_wdeg.npy")
        cache_wclos = os.path.join(results_dir, f"{filename}_wclos.npy")

        X_test = np.load(cache_x_path)
        labels = np.load(cache_y_path).flatten()
        weig_vals = np.load(cache_weig).flatten()
        wdeg_vals = np.load(cache_wdeg).flatten()
        wclos_vals = np.load(cache_wclos).flatten()

        with torch.no_grad():
            pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        top_indices = get_subset_indices(labels, 20)
        tau_ours = safe_tau(pred, labels, top_indices)
        tau_weig = safe_tau(weig_vals, labels, top_indices)
        tau_wdeg = safe_tau(wdeg_vals, labels, top_indices)
        tau_wclos = safe_tau(wclos_vals, labels, top_indices)

        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "nan"

        print(f"{filename:<30} | {fmt(tau_ours):<20} | {fmt(tau_weig):<15} | {fmt(tau_wdeg):<15} | {fmt(tau_wclos):<15}")
    print("="*100)

if __name__ == "__main__":
    main()
