import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau

# Directory Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
weighted_model_dir = os.path.abspath(os.path.join(script_dir, "..", "weighted models"))
results_dir = os.path.join(weighted_model_dir, "results")

# Datasets folders
datasets_base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "Datasets"))
test_folder = os.path.join(datasets_base_dir, "weighted Datasets", "test")

# Discover test files
test_datasets = []
if os.path.exists(test_folder):
    for f in sorted(os.listdir(test_folder)):
        if f in ["karate.txt", "cargoshipsBB.txt"]:
            continue
        if os.path.exists(os.path.join(results_dir, f"{f}_weighted_local_norm_X.npy")):
            test_datasets.append(f)

# Model Definition
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
    model_path = os.path.join(script_dir, "wnlgcn_6ch_deep.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = WNLGCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    ours_list, weig_list, wdeg_list, wclos_list, wbet_list = [], [], [], [], []

    for filename in test_datasets:
        cache_x_path = os.path.join(results_dir, f"{filename}_weighted_local_norm_X.npy")
        cache_y_path = os.path.join(results_dir, f"{filename}_weighted_y.npy")
        cache_weig = os.path.join(results_dir, f"{filename}_weig.npy")
        cache_wdeg = os.path.join(results_dir, f"{filename}_wdeg.npy")
        cache_wclos = os.path.join(results_dir, f"{filename}_wclos.npy")
        cache_wbet = os.path.join(results_dir, f"{filename}_wbet.npy")

        X = np.load(cache_x_path)
        labels = np.load(cache_y_path).flatten()
        
        weig_vals = np.load(cache_weig).flatten()
        wdeg_vals = np.load(cache_wdeg).flatten()
        wclos_vals = np.load(cache_wclos).flatten()
        wbet_vals = np.load(cache_wbet).flatten()

        with torch.no_grad():
            pred = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

        indices = get_subset_indices(labels, 20)
        
        tau_ours = safe_tau(pred, labels, indices)
        tau_weig = safe_tau(weig_vals, labels, indices)
        tau_wdeg = safe_tau(wdeg_vals, labels, indices)
        tau_wclos = safe_tau(wclos_vals, labels, indices)
        tau_wbet = safe_tau(wbet_vals, labels, indices)

        if not np.isnan(tau_ours): ours_list.append(tau_ours)
        if not np.isnan(tau_weig): weig_list.append(tau_weig)
        if not np.isnan(tau_wdeg): wdeg_list.append(tau_wdeg)
        if not np.isnan(tau_wclos): wclos_list.append(tau_wclos)
        if not np.isnan(tau_wbet): wbet_list.append(tau_wbet)

    print("\n" + "="*50)
    print("AVERAGE PERFORMANCE ACROSS ALL TEST DATASETS (TOP 20%):")
    print("="*50)
    print(f"Ours (Deep WNLGCN-6Ch): {np.mean(ours_list):.4f}")
    print(f"Weighted Eigenvector:  {np.mean(weig_list):.4f}")
    print(f"Strength (W-Degree):   {np.mean(wdeg_list):.4f}")
    print(f"Weighted Closeness:    {np.mean(wclos_list):.4f}")
    print(f"Weighted Betweenness:  {np.mean(wbet_list):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
