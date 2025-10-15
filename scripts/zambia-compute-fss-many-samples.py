import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
lead_time = sys.argv[1]
target_hour = sys.argv[2]

base_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/combined_nowcasts/ensemble/t{lead_time}"

windows = [3, 9, 25, 49, 81, 121]
PIXEL_SIZE_KM = 3

# ---------------------------------------------------------------------
# Fractions Skill Score
# ---------------------------------------------------------------------
def compute_fss(pred, obs, window):
    pred = np.clip(pred, 0, 1)
    obs = np.clip(obs, 0, 1)
    f_pred = uniform_filter(pred, size=window, mode="constant")
    f_obs = uniform_filter(obs, size=window, mode="constant")
    num = np.mean((f_pred - f_obs) ** 2)
    den = np.mean(f_pred ** 2 + f_obs ** 2)
    return 1 - num / (den + 1e-8)

# ---------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------
all_files = sorted(f for f in os.listdir(base_dir) if f.endswith(".pt"))
filtered_files = [f for f in all_files if f[9:11] == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
fss_results = {w: {"zcast": [], "nflics": [], "netncc": []} for w in windows}

for f in tqdm(filtered_files, desc="Computing FSS"):
    file_path = os.path.join(base_dir, f)
    try:
        data = torch.load(file_path, weights_only=False)
    except EOFError:
        print(f"Skipping corrupted file: {file_path}")
        continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

    # Extract data
    gt = np.nan_to_num(data["ground_truth"].astype(np.float32))
    zcast = np.nan_to_num(data["zcast"].astype(np.float32))
    nflics = np.nan_to_num(data["nflics"].astype(np.float32))
    netncc = np.nan_to_num(data["netncc"].astype(np.float32))

    # If stored as percentages, rescale
    if np.nanmax(nflics) > 1.5 or np.nanmax(netncc) > 1.5:
        nflics = nflics / 100.0
        netncc = netncc / 100.0

    nflics = np.clip(nflics, 0, 1)
    netncc = np.clip(netncc, 0, 1)

    # Compute FSS for all window sizes
    for w in windows:
        fss_results[w]["zcast"].append(compute_fss(zcast, gt, w))
        fss_results[w]["nflics"].append(compute_fss(nflics, gt, w))
        fss_results[w]["netncc"].append(compute_fss(netncc, gt, w))

# ---------------------------------------------------------------------
# Aggregate and display
# ---------------------------------------------------------------------
rows = []
print(f"\nAverage FSS for hour={target_hour} UTC (lead t+{lead_time})")
print(f"{'Window':>8} | {'Scale(km)':>10} | {'ZCAST':>10} | {'NFLICS':>10} | {'NetNCC':>10}")
print("-" * 62)

for w in windows:
    scale = w * PIXEL_SIZE_KM
    mean_z = np.nanmean(fss_results[w]["zcast"])
    mean_n = np.nanmean(fss_results[w]["nflics"])
    mean_c = np.nanmean(fss_results[w]["netncc"])
    rows.append({
        "window": w,
        "scale_km": scale,
        "zcast": mean_z,
        "nflics": mean_n,
        "netncc": mean_c
    })
    print(f"{w:>8} | {scale:>10.0f} | {mean_z:.4f} | {mean_n:.4f} | {mean_c:.4f}")

# ---------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------
output_csv = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/FSS/ensemble/fss_hour_{target_hour}_t{lead_time}.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(rows).to_csv(output_csv, index=False)
print(f"\nSaved FSS summary to {output_csv}")
