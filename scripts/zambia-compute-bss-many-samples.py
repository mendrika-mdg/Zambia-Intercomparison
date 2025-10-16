import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
lead_time = sys.argv[1]
target_hour = sys.argv[2]

base_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/combined_nowcasts/ensemble/t{lead_time}"
output_dir = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/BSS/ensemble"
os.makedirs(output_dir, exist_ok=True)

map_shape = (350, 370)  # spatial domain size

# ---------------------------------------------------------------------
# Pixelwise 2-D Brier Skill Score
# ---------------------------------------------------------------------
def compute_bss(pred_model, pred_ref, obs):
    pred_model = np.clip(pred_model, 0, 1)
    pred_ref   = np.clip(pred_ref, 0, 1)
    obs        = np.clip(obs, 0, 1)

    bs_model = (pred_model - obs) ** 2
    bs_ref   = (pred_ref - obs) ** 2

    # Return full 2-D field, no averaging
    return 1 - bs_model / (bs_ref + 1e-8)

# ---------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------
all_files = sorted(f for f in os.listdir(base_dir) if f.endswith(".pt"))
filtered_files = [f for f in all_files if f[9:11] == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
bss_z_list, bss_c_list = [], []

for f in tqdm(filtered_files, desc="Computing pixelwise BSS (2-D)"):
    file_path = os.path.join(base_dir, f)
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {f}: {e}")
        continue

    # Load arrays
    gt      = np.nan_to_num(data["ground_truth"].astype(np.float32))
    zcast   = np.nan_to_num(data["zcast"].astype(np.float32))
    nflics  = np.nan_to_num(data["nflics"].astype(np.float32))
    netncc  = np.nan_to_num(data["netncc"].astype(np.float32))

    # Rescale NFLICS and NetNCC from 0–100 to 0–1 if needed
    if np.nanmax(nflics) > 1.5:
        nflics = nflics / 100.0
    if np.nanmax(netncc) > 1.5:
        netncc = netncc / 100.0

    # Binarise ground truth if not already binary
    if np.nanmax(gt) > 1.5:
        gt = (gt > 0).astype(np.float32)

    # Validate shapes
    for name, arr in zip(["gt", "zcast", "nflics", "netncc"],
                         [gt, zcast, nflics, netncc]):
        if arr.shape != map_shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {map_shape}")

    # Compute per-pixel BSS (NFLICS = reference)
    bss_z = compute_bss(zcast, nflics, gt)
    bss_c = compute_bss(netncc, nflics, gt)

    bss_z_list.append(bss_z)
    bss_c_list.append(bss_c)

# ---------------------------------------------------------------------
# Aggregate across all samples (retain 2-D field)
# ---------------------------------------------------------------------
if not bss_z_list:
    raise RuntimeError("No valid files found for BSS computation.")

mean_bss_zcast  = np.nanmean(np.stack(bss_z_list, axis=0), axis=0)
mean_bss_netncc = np.nanmean(np.stack(bss_c_list, axis=0), axis=0)

assert mean_bss_zcast.shape == map_shape, f"Unexpected shape: {mean_bss_zcast.shape}"
assert mean_bss_netncc.shape == map_shape, f"Unexpected shape: {mean_bss_netncc.shape}"

# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
np.save(os.path.join(output_dir, f"bss_zcast_hour_{target_hour}_t{lead_time}.npy"), mean_bss_zcast)
np.save(os.path.join(output_dir, f"bss_netncc_hour_{target_hour}_t{lead_time}.npy"), mean_bss_netncc)

print(f"\nSaved mean 2-D pixelwise BSS maps for t+{lead_time}, hour={target_hour} UTC to {output_dir}")
print(f"ZCAST domain-mean BSS:  {np.nanmean(mean_bss_zcast):.4f}")
print(f"NetNCC domain-mean BSS: {np.nanmean(mean_bss_netncc):.4f}")
