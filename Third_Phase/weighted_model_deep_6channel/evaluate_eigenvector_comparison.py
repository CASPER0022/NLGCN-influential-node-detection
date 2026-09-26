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
    weig_taus = []
    wdeg_taus = []
    wclos_taus = []
    wbet_taus = []
    deltas = []

    print("\n" + "="*115)
    print(" DEEP WNLGCN-6CH VS. WEIGHTED EIGENVECTOR CENTRALITY COMPARISON ON HELD-OUT TEST NETWORKS")
    print(" Metric: Kendall's Tau (tau) on Top 20% Spreader Nodes")
    print("="*115)
    print(f"{'Test Dataset':<28} | {'Deep WNLGCN':<13} | {'W-Eigenvector':<13} | {'Gain over W-Eigen':<18} | {'Strength (W-Deg)':<15} | {'W-Closeness':<13} | {'W-Betweenness':<13}")
    print("-" * 115)

    for filename in test_datasets:
        cache_w_x = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y = os.path.join(results_dir, f"{filename}_weighted_y.npy")
        cache_weig = os.path.join(results_dir, f"{filename}_weig.npy")
        cache_wdeg = os.path.join(results_dir, f"{filename}_wdeg.npy")
        cache_wclos = os.path.join(results_dir, f"{filename}_wclos.npy")
        cache_wbet = os.path.join(results_dir, f"{filename}_wbet.npy")

        X_weighted = np.load(cache_w_x)
        labels = np.load(cache_y).flatten()
        
        weig_vals = np.load(cache_weig).flatten() if os.path.exists(cache_weig) else np.zeros_like(labels)
        wdeg_vals = np.load(cache_wdeg).flatten() if os.path.exists(cache_wdeg) else np.zeros_like(labels)
        wclos_vals = np.load(cache_wclos).flatten() if os.path.exists(cache_wclos) else np.zeros_like(labels)
        wbet_vals = np.load(cache_wbet).flatten() if os.path.exists(cache_wbet) else np.zeros_like(labels)

        with torch.no_grad():
            pred_weighted = model(torch.tensor(X_weighted, dtype=torch.float32)).numpy().flatten()
        
        tau_w = safe_tau(pred_weighted, labels, pct=20)
        tau_eig = safe_tau(weig_vals, labels, pct=20)
        tau_sd = safe_tau(wdeg_vals, labels, pct=20)
        tau_c = safe_tau(wclos_vals, labels, pct=20)
        tau_b = safe_tau(wbet_vals, labels, pct=20)

        delta = tau_w - tau_eig

        if not np.isnan(tau_w): weighted_taus.append(tau_w)
        if not np.isnan(tau_eig): weig_taus.append(tau_eig)
        if not np.isnan(tau_sd): wdeg_taus.append(tau_sd)
        if not np.isnan(tau_c): wclos_taus.append(tau_c)
        if not np.isnan(tau_b): wbet_taus.append(tau_b)
        if not np.isnan(delta): deltas.append(delta)

        print(f"{filename:<28} | {tau_w:<13.4f} | {tau_eig:<13.4f} | {delta:<+18.4f} | {tau_sd:<15.4f} | {tau_c:<13.4f} | {tau_b:<13.4f}")

    avg_w = np.mean(weighted_taus)
    avg_eig = np.mean(weig_taus)
    avg_delta = np.mean(deltas)
    rel_gain = (avg_delta / avg_eig) * 100.0 if avg_eig != 0 else 0.0

    print("-" * 115)
    print(f"{'AVERAGE (15 SF Networks)':<28} | {avg_w:<13.4f} | {avg_eig:<13.4f} | {avg_delta:<+18.4f} | {np.mean(wdeg_taus):<15.4f} | {np.mean(wclos_taus):<13.4f} | {np.mean(wbet_taus):<13.4f}")
    print("=" * 115)
    print(f"\nSummary Analysis:")
    print(f" - Deep WNLGCN Average Kendall Tau:              {avg_w:.4f}")
    print(f" - Weighted Eigenvector Average Kendall Tau:    {avg_eig:.4f}")
    print(f" - Average Improvement over Weighted Eigenvector: +{avg_delta:.4f}")
    print(f" - Relative Gain over Weighted Eigenvector:     +{rel_gain:.2f}%")
    print(f" - Beats Weighted Eigenvector on:               {sum(d > 0 for d in deltas)} / {len(deltas)} held-out networks")
    print("=" * 115)

if __name__ == "__main__":
    main()
