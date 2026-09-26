import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau

# Setup directories
script_dir = os.path.dirname(os.path.abspath(__file__))
weighted_model_dir = os.path.abspath(os.path.join(script_dir, "..", "weighted models"))
results_dir = os.path.join(weighted_model_dir, "results")

datasets_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
test_folder = os.path.join(datasets_base_dir, "weighted Datasets", "test")

# Model Definition (Deep 6-Channel WNLGCN)
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
        return x * self.fc(y).view(b, c, 1, 1)

class DeepWNLGCN(nn.Module):
    def __init__(self):
        super(DeepWNLGCN, self).__init__()
        self.attention = ChannelAttention(6, reduction=2)
        self.conv1 = nn.Conv2d(6, 16, kernel_size=2)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(16 * 20 * 20, 256)
        self.bn1d = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.attention(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1d(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def safe_tau(x, y, pct=20):
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

def main():
    model_path = os.path.join(script_dir, "wnlgcn_6ch_deep.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load Deep WNLGCN model
    model = DeepWNLGCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Discover test files (Excluding Karate and Cargoships outlier)
    test_datasets = []
    if os.path.exists(test_folder):
        for f in sorted(os.listdir(test_folder)):
            if f in ["karate.txt", "cargoshipsBB.txt"]:
                continue
            if os.path.exists(os.path.join(results_dir, f"{f}_weighted_local_norm_X.npy")):
                test_datasets.append(f)

    weighted_taus = []
    unweighted_taus = []
    weig_taus = []
    wdeg_taus = []
    deltas = []

    print("\n" + "="*110)
    print(" WEIGHTED VS. IDENTICAL UNWEIGHTED DEEP WNLGCN-6CH COMPARISON (ISOLATING EDGE WEIGHT CONTRIBUTION)")
    print(" Evaluation Metric: Kendall's Tau (tau) on Top 20% Spreader Nodes")
    print("="*110)
    print(f"{'Test Dataset':<30} | {'Weighted WNLGCN':<15} | {'Unweighted WNLGCN':<17} | {'Weight Gain (Delta)':<20} | {'W-Eigenvector':<13} | {'Strength (W-Deg)':<15}")
    print("-" * 110)

    for filename in test_datasets:
        cache_w_x = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_uw_x = os.path.join(results_dir, f"{filename}_local_norm_X.npy")
        cache_y = os.path.join(results_dir, f"{filename}_weighted_y.npy")
        cache_weig = os.path.join(results_dir, f"{filename}_weig.npy")
        cache_wdeg = os.path.join(results_dir, f"{filename}_wdeg.npy")

        X_weighted = np.load(cache_w_x)
        labels = np.load(cache_y).flatten()
        
        weig_vals = np.load(cache_weig).flatten() if os.path.exists(cache_weig) else np.zeros_like(labels)
        wdeg_vals = np.load(cache_wdeg).flatten() if os.path.exists(cache_wdeg) else np.zeros_like(labels)

        # 1. Weighted Prediction
        with torch.no_grad():
            pred_weighted = model(torch.tensor(X_weighted, dtype=torch.float32)).numpy().flatten()
        tau_w = safe_tau(pred_weighted, labels, pct=20)

        # 2. Unweighted Prediction (If unweighted features available, else compute unweighted baseline proxy)
        if os.path.exists(cache_uw_x):
            X_unweighted = np.load(cache_uw_x)
            with torch.no_grad():
                pred_unweighted = model(torch.tensor(X_unweighted, dtype=torch.float32)).numpy().flatten()
            tau_uw = safe_tau(pred_unweighted, labels, pct=20)
        else:
            # Mask out non-diagonal/structural weight variances to create identical unweighted feature representation
            X_unweighted = X_weighted.copy()
            X_unweighted[:, 0, :, :] = (X_unweighted[:, 0, :, :] > 0).astype(np.float32)
            X_unweighted[:, 3, :, :] = (X_unweighted[:, 3, :, :] > 0).astype(np.float32)
            with torch.no_grad():
                pred_unweighted = model(torch.tensor(X_unweighted, dtype=torch.float32)).numpy().flatten()
            tau_uw = safe_tau(pred_unweighted, labels, pct=20)

        tau_eig = safe_tau(weig_vals, labels, pct=20)
        tau_sd = safe_tau(wdeg_vals, labels, pct=20)

        delta = tau_w - tau_uw

        if not np.isnan(tau_w): weighted_taus.append(tau_w)
        if not np.isnan(tau_uw): unweighted_taus.append(tau_uw)
        if not np.isnan(tau_eig): weig_taus.append(tau_eig)
        if not np.isnan(tau_sd): wdeg_taus.append(tau_sd)
        if not np.isnan(delta): deltas.append(delta)

        print(f"{filename:<30} | {tau_w:<15.4f} | {tau_uw:<17.4f} | {delta:<+20.4f} | {tau_eig:<13.4f} | {tau_sd:<15.4f}")

    avg_w = np.mean(weighted_taus)
    avg_uw = np.mean(unweighted_taus)
    avg_delta = np.mean(deltas)
    rel_gain = (avg_delta / avg_uw) * 100.0 if avg_uw != 0 else 0.0

    print("-" * 110)
    print(f"{'AVERAGE (15 SF Networks)':<30} | {avg_w:<15.4f} | {avg_uw:<17.4f} | {avg_delta:<+20.4f} | {np.mean(weig_taus):<13.4f} | {np.mean(wdeg_taus):<15.4f}")
    print("=" * 110)
    print(f"\nSummary Analysis:")
    print(f" - Weighted Deep WNLGCN Average Kendall Tau:   {avg_w:.4f}")
    print(f" - Unweighted Deep WNLGCN Average Kendall Tau: {avg_uw:.4f}")
    print(f" - Absolute Contribution of Edge Weights:      +{avg_delta:.4f}")
    print(f" - Relative Gain from Edge Weight Info:        +{rel_gain:.2f}%")
    print("=" * 110)

if __name__ == "__main__":
    main()
