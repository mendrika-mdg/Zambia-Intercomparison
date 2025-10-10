import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Config
PARTITION = sys.argv[1]
LEAD_TIME = sys.argv[2]
BASE_DIR = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison"
RAW_INPUT_DIR = f"{BASE_DIR}/raw/inputs_t0"
RAW_TARGET_DIR = f"{BASE_DIR}/raw/targets_t{LEAD_TIME}"
SPLIT_FILE = f"{BASE_DIR}/splits/{PARTITION}_files.txt"
SHARDS_DIR = f"{BASE_DIR}/shards/t{LEAD_TIME}/{PARTITION}_t{LEAD_TIME}"
os.makedirs(SHARDS_DIR, exist_ok=True)

# Load list
with open(SPLIT_FILE) as f:
    files = [line.strip() for line in f if line.strip()]

FILES_PER_SHARD = 1000
print(f"Creating shards for partition={PARTITION.upper()}, lead time={LEAD_TIME}h")
print(f"Total input files: {len(files):,}")

# Buffers
shard_inputs, shard_globals, shard_targets, shard_ids = [], [], [], []
shard_index = 0

# Main loop
for i, fpath in enumerate(tqdm(files, desc="Sharding inputs and targets")):
    try:
        # Load input
        data_in = torch.load(fpath, map_location="cpu")
        if "input_tensor" not in data_in or "global_context" not in data_in:
            print(f"Missing keys in {fpath}, skipping file.")
            continue

        x = data_in["input_tensor"]
        g = data_in["global_context"]
        nowcast_id = data_in.get("nowcast_origin", os.path.basename(fpath))

        # Enforce shape
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        if g.ndim != 1:
            g = g.flatten()

        # Convert to NumPy
        x = x.detach().cpu().numpy()
        g = g.detach().cpu().numpy()

        # Load target
        fname = os.path.basename(fpath).replace("input-", "target-")
        target_path = os.path.join(RAW_TARGET_DIR, fname)
        if not os.path.exists(target_path):
            print(f"Missing target file: {target_path}")
            continue

        data_out = torch.load(target_path, map_location="cpu")
        y = data_out.get("data", None)
        if y is None:
            print(f"Missing 'data' key in {target_path}")
            continue
        if y.ndim != 2:
            y = y.squeeze()

        y = y.detach().cpu().numpy().astype(np.uint8)

        # Verify expected shapes (skip mismatched file)
        if x.shape[0] not in {288} or y.shape != (350, 370):
            print(f"Skipping {os.path.basename(fpath)} due to unexpected shape:")
            print(f"  Input:  {x.shape}")
            print(f"  Target: {y.shape}")
            continue

        # Append
        shard_inputs.append(x)
        shard_globals.append(g)
        shard_targets.append(y)
        shard_ids.append(nowcast_id)

        # Save shard
        if len(shard_inputs) >= FILES_PER_SHARD or (i + 1) == len(files):
            shard_path = os.path.join(SHARDS_DIR, f"shard_{shard_index:03d}.pt")
            torch.save({
                "X": torch.tensor(np.stack(shard_inputs), dtype=torch.float32),
                "G": torch.tensor(np.stack(shard_globals), dtype=torch.float32),
                "Y": torch.tensor(np.stack(shard_targets), dtype=torch.uint8),
                "ID": shard_ids
            }, shard_path)

            print(f"Saved shard_{shard_index:03d} ({len(shard_inputs)} samples) → {shard_path}")
            shard_index += 1
            shard_inputs, shard_globals, shard_targets, shard_ids = [], [], [], []

    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        continue

print(f"Finished creating {PARTITION.upper()} shards for LT{LEAD_TIME}.")
