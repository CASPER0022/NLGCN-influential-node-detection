import os
import numpy as np

# Path configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))

channels_info = {
    0: "Channel 1 (NLI 1 - Local Influence)",
    1: "Channel 2 (NLI 2 - Multi-scale)",
    2: "Channel 3 (NLI 3 - Multi-scale)",
    3: "Channel 4 (NGI 1 - Global Influence)",
    4: "Channel 5 (NGI 2 - Multi-scale)",
    5: "Channel 6 (NGI 6 - Multi-scale)"
}

def analyze_raw_ranges():
    if not os.path.exists(results_dir):
        print(f"Error: results directory not found at {results_dir}. Please run the training script first to populate cache.")
        return

    # Find all X.npy files in the results directory
    x_files = [f for f in os.listdir(results_dir) if f.endswith("_X.npy")]
    if not x_files:
        print("No cached feature files found. Run the training script first to cache graph features.")
        return

    print("="*110)
    print(f"{'Dataset Name':<30} | {'Channel':<38} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
    print("="*110)

    all_data = []

    for file in x_files:
        path = os.path.join(results_dir, file)
        dataset_name = file.replace("_X.npy", "")
        
        # Load raw features of shape (N, 6, 41, 41)
        X = np.load(path)
        all_data.append(X)
        
        print(f"\n--- {dataset_name} (Shape: {X.shape}) ---")
        for channel in range(6):
            channel_data = X[:, channel, :, :]
            # Exclude diagonal entries and first row/column padding if we want raw feature statistics,
            # or just take the full matrix statistics:
            # Here we take stats over the actual embedded matrix values:
            val_min = channel_data.min()
            val_max = channel_data.max()
            val_mean = channel_data.mean()
            val_std = channel_data.std()
            
            print(f"{'':<30} | {channels_info[channel]:<38} | {val_min:<10.4f} | {val_max:<10.4f} | {val_mean:<10.4f} | {val_std:<10.4f}")

    if all_data:
        # Combine all training nodes to see global statistics
        combined_X = np.concatenate(all_data, axis=0)
        print("\n" + "="*110)
        print(f"{'GLOBAL COMBINED STATISTICS (All Nodes)':<69} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
        print("="*110)
        for channel in range(6):
            channel_data = combined_X[:, channel, :, :]
            val_min = channel_data.min()
            val_max = channel_data.max()
            val_mean = channel_data.mean()
            val_std = channel_data.std()
            print(f"{channels_info[channel]:<69} | {val_min:<10.4f} | {val_max:<10.4f} | {val_mean:<10.4f} | {val_std:<10.4f}")
        print("="*110)

if __name__ == "__main__":
    analyze_raw_ranges()
