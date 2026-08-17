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
train_folder = os.path.join(datasets_base_dir, "weighted Datasets", "train")
test_folder = os.path.join(datasets_base_dir, "weighted Datasets", "test")

# Discover train/test files and verify they are cached in results_dir
train_datasets = []
for f in sorted(os.listdir(train_folder)):
    if os.path.exists(os.path.join(results_dir, f"{f}_weighted_local_norm_X.npy")):
        train_datasets.append(f)

test_datasets = []
for f in sorted(os.listdir(test_folder)):
    if f in ["karate.txt", "cargoshipsBB.txt"]:
        continue
    if os.path.exists(os.path.join(results_dir, f"{f}_weighted_local_norm_X.npy")):
        test_datasets.append(f)

print(f"Loaded {len(train_datasets)} training datasets and {len(test_datasets)} testing datasets.")

# ---- MS-WNLGCN Model Definition (Multi-Scale 6 Channels) ----
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

class MSWNLGCN(nn.Module):
    def __init__(self):
        super(MSWNLGCN, self).__init__()
        self.attention = ChannelAttention(6, reduction=2)
        
        # Parallel multi-scale convolutional filters
        self.branch1 = nn.Conv2d(6, 8, kernel_size=2)
        self.branch2 = nn.Conv2d(6, 8, kernel_size=3)
        self.branch3 = nn.Conv2d(6, 8, kernel_size=4)
        
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        
        self.pool = nn.MaxPool2d(2)
        
        # MLP maps concatenated outputs (3200 + 2888 + 2888 = 8976 flat features)
        self.fc1 = nn.Linear(8976, 256)
        self.bn1d = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.attention(x)
        
        # Extract features at scale 2, 3, and 4
        out1 = self.pool(F.relu(self.bn1(self.branch1(x))))
        out2 = self.pool(F.relu(self.bn2(self.branch2(x))))
        out3 = self.pool(F.relu(self.bn3(self.branch3(x))))
        
        # Flatten
        flat1 = out1.view(out1.size(0), -1)
        flat2 = out2.view(out2.size(0), -1)
        flat3 = out3.view(out3.size(0), -1)
        
        # Concatenate scale descriptors
        flat = torch.cat([flat1, flat2, flat3], dim=1)
        
        flat = self.fc1(flat)
        if flat.size(0) > 1:
            flat = self.bn1d(flat)
        flat = F.gelu(flat)
        flat = self.dropout(flat)
        
        flat = F.gelu(self.fc2(flat))
        flat = self.dropout(flat)
        flat = self.fc3(flat)
        return flat

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

    # Load cached training data directly
    for filename in train_datasets:
        cache_x_path = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_weighted_y.npy")

        X = np.load(cache_x_path)
        y = np.load(cache_y_path).flatten()
        
        k = max(1, int(len(y) * 0.2))
        top_threshold = np.partition(y, -k)[-k]
        is_top = (y >= top_threshold).astype(np.float32)

        all_X.append(X)
        all_y.append(y)
        all_is_top.append(is_top)

    y_all_concat = np.concatenate(all_y, axis=0)
    y_mean = y_all_concat.mean()
    y_std = y_all_concat.std()

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

    # Train Model with AdamW and Cosine Annealing
    model = MSWNLGCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-2)
    
    epochs = 450
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_criterion = nn.MSELoss()

    ALPHA_RANK = 1.5
    MARGIN = 0.3
    TOP_WEIGHT = 3.0

    print("Starting Multi-Scale 6-Channel WNLGCN Training (AdamW + Cosine LR)...")
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
                rank_loss = top_heavy_pairwise_ranking_loss(outputs, batch_y, batch_is_top, margin=MARGIN, top_weight=TOP_WEIGHT)
                loss = mse_loss + ALPHA_RANK * rank_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                n_batches += batch_X.size(0)

        scheduler.step()
        epoch_loss /= n_batches
        if epoch % 50 == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.6f} | LR: {current_lr:.6f}")

    # Save model weights
    model_save_path = os.path.join(script_dir, "wnlgcn_6ch_multiscale.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved trained model weights to: {model_save_path}")

    print("Training finished. Evaluating on Test Datasets...")
    model.eval()

    # Evaluation
    print("\n" + "="*100)
    print(f"{'Network Dataset':<30} | {'Ours (MS-WNLGCN)':<20} | {'W-Eigenvector':<15} | {'Strength (W-Deg)':<15} | {'W-Closeness':<15}")
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
