import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Check command-line arguments
if len(sys.argv) != 3:
    print("Usage: python apply_scaler_to_shards.py <partition> <lead_time>")
    sys.exit(1)

# Read partition and lead time
PARTITION = sys.argv[1]
LEAD_TIME = sys.argv[2]

# Define directories
BASE_DIR = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison"
SCALER_PATH = f"{BASE_DIR}/scaling/scaler_realcores.pt"
SHARDS_DIR = f"{BASE_DIR}/shards/t{LEAD_TIME}/{PARTITION}_t{LEAD_TIME}"
SAVE_DIR = f"{BASE_DIR}/preprocessed/t{LEAD_TIME}/{PARTITION}_t{LEAD_TIME}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Column indices
MASK_COL_INDEX = 8
COLS_TO_SCALE = [0, 1, 2, 3, 4, 5, 6, 7]

# Load scaler
scaler = torch.load(SCALER_PATH, weights_only=False)
mean = np.asarray(scaler["mean"])       # shape (8,)
scale = np.asarray(scaler["scale"])     # shape (8,)

print(f"Loaded scaler from {SCALER_PATH}")
print(f"Applying to partition={PARTITION.upper()}, lead_time={LEAD_TIME}h")

# List shard files
all_shards = sorted(
    [os.path.join(SHARDS_DIR, f) for f in os.listdir(SHARDS_DIR) if f.endswith(".pt")]
)
if not all_shards:
    raise FileNotFoundError(f"No shards found in {SHARDS_DIR}")

print(f"Found {len(all_shards)} shard files to process...")

# Process each shard
for shard_path in tqdm(all_shards, desc=f"Scaling {PARTITION}_t{LEAD_TIME}"):
    data = torch.load(shard_path, map_location="cpu")
    X = data["X"].numpy()
    flat = X.reshape(-1, X.shape[-1])

    # Identify rows with real cores
    real_mask = flat[:, MASK_COL_INDEX] == 1
    rows = np.where(real_mask)[0]

    # Apply scaling to selected columns for real cores only
    flat[np.ix_(rows, COLS_TO_SCALE)] = (
        (flat[np.ix_(rows, COLS_TO_SCALE)] - mean) / scale
    )

    # Reshape back to original shape
    X_scaled = flat.reshape(X.shape)

    # Save scaled data
    scaled_data = {
        "X": torch.tensor(X_scaled, dtype=torch.float32),
        "G": data["G"],
        "Y": data["Y"],
        "ID": data["ID"],
    }
    torch.save(scaled_data, os.path.join(SAVE_DIR, os.path.basename(shard_path)))

print(f"Finished scaling {PARTITION}_t{LEAD_TIME}. Scaled files saved to {SAVE_DIR}")
