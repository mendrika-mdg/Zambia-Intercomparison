import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

LEAD_TIME = "1" # it doesn't really matter since the input is the same for all lead times

BASE_DIR = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison"
SHARDS_DIR = f"{BASE_DIR}/shards/t{LEAD_TIME}/train_t{LEAD_TIME}"
SAVE_DIR = f"{BASE_DIR}/scaling"

os.makedirs(SAVE_DIR, exist_ok=True)

# Feature layout reminder:
#  0 lat, 1 lon, 2 lat_min, 3 lat_max,
#  4 lon_min, 5 lon_max, 6 tir, 7 size,
#  8 mask, 9 lag
MASK_COL_INDEX = 8
LAG_COL_INDEX = 9

# Only these columns should be scaled
COLS_TO_SCALE = [0, 1, 2, 3, 4, 5, 6, 7]

scaler = StandardScaler()


all_shards = sorted(
    [os.path.join(SHARDS_DIR, f) for f in os.listdir(SHARDS_DIR) if f.endswith(".pt")]
)

if not all_shards:
    raise FileNotFoundError(f"No shard files found in {SHARDS_DIR}")

print(f"Computing scaling parameters from {len(all_shards)} training shards (LT{LEAD_TIME})...")


n_real_total = 0

for shard_path in tqdm(all_shards, desc="Processing shards"):
    data = torch.load(shard_path, map_location="cpu")
    X = data["X"].numpy()  # shape (N, 288, 10)

    # Flatten to (N×288, F)
    flat = X.reshape(-1, X.shape[-1])

    # Identify real cores
    real = flat[flat[:, MASK_COL_INDEX] == 1]
    if real.shape[0] == 0:
        continue

    # Incremental update of scaler on real cores only
    scaler.partial_fit(real[:, COLS_TO_SCALE])
    n_real_total += real.shape[0]

print(f"Total real cores processed: {n_real_total:,}")

stats = {
    "mean": scaler.mean_,
    "scale": scaler.scale_,
    "cols": COLS_TO_SCALE,
    "n_real_cores": n_real_total
}

out_path = os.path.join(SAVE_DIR, f"scaler_realcores.pt")
torch.save(stats, out_path)

print(f"Scaling parameters saved to {out_path}")
print("Done.")
