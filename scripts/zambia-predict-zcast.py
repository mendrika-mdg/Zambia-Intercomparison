import numpy as np
import os
from tqdm import tqdm
import re
from datetime import datetime
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Import model
# ---------------------------------------------------------------------
sys.path.append("/home/users/mendrika/Zambia-Intercomparison/models/training")
from zambia_lag_corrected import Core2MapModel

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LEAD_TIME = sys.argv[1]
CHECKPOINT_PATH = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/checkpoints/t{LEAD_TIME}/best-core2map.ckpt"
SCALER_PATH = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/scaling/scaler_realcores.pt"
OUTPUT_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/zcast/nowcasts/t{LEAD_TIME}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load model and scaler
# ---------------------------------------------------------------------
model = Core2MapModel.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
model.eval()
model.to(DEVICE)

scaler = torch.load(SCALER_PATH, weights_only=False)
mean = np.asarray(scaler["mean"])   # shape (8,)
scale = np.asarray(scaler["scale"]) # shape (8,)

COLS_TO_SCALE = [0, 1, 2, 3, 4, 5, 6, 7]
MASK_COL_INDEX = 8
LAG_COL_INDEX = 9

# ---------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------
def load_zcast_input(year, month, day, hour, minute):
    base = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/raw/inputs_t0"
    path = f"{base}/input-{year}{month}{day}_{hour}{minute}.pt"
    return torch.load(path)

# ---------------------------------------------------------------------
# Collect all input files
# ---------------------------------------------------------------------
zcast_root = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/raw/inputs_t0"
pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")

input_files = []
for root, _, files in os.walk(zcast_root):
    for f in files:
        m = pattern.match(f)
        if m:
            year, month, day, hour, minute = m.groups()
            if int(year) >= 2020:
                nowcast_origin = datetime(int(year), int(month), int(day), int(hour), int(minute))
                full_path = os.path.join(root, f)
                input_files.append({
                    "year": year, "month": month, "day": day,
                    "hour": hour, "minute": minute,
                    "datetime": nowcast_origin, "path": full_path
                })

input_files = sorted(input_files, key=lambda x: x["datetime"])
print(f"Detected {len(input_files)} zcast input files.")

# ---------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------
for entry in tqdm(input_files, desc="Processing file"):
    year, month, day, hour, minute = (
        entry["year"], entry["month"], entry["day"], entry["hour"], entry["minute"]
    )

    try:
        zcast_input = load_zcast_input(year, month, day, hour, minute)
        input_tensor = zcast_input["input_tensor"].clone().unsqueeze(0)
        global_context = zcast_input["global_context"].clone().unsqueeze(0)

        if input_tensor.shape == (1, 288, 10):
            # Convert mean/scale to tensors on the right device
            mean_t = torch.tensor(mean, dtype=input_tensor.dtype, device=input_tensor.device)
            scale_t = torch.tensor(scale, dtype=input_tensor.dtype, device=input_tensor.device)

            # Scale only real cores (mask == 1)
            mask = input_tensor[:, :, MASK_COL_INDEX] > 0.5
            for col in COLS_TO_SCALE:
                x = input_tensor[:, :, col]
                x[mask] = (x[mask] - mean_t[col]) / scale_t[col]
                input_tensor[:, :, col] = x

            # Normalise lag column to [-1, 1]
            input_tensor[:, :, LAG_COL_INDEX] = 2 * (input_tensor[:, :, LAG_COL_INDEX] / 120) - 1

            # Forward pass
            zcast = torch.sigmoid(
                model(input_tensor.to(DEVICE), global_context.to(DEVICE))
            ).squeeze(0).squeeze(0).detach().cpu().numpy()

            # Save nowcast
            save_path = os.path.join(
                OUTPUT_DIR, f"zcast_{year}{month}{day}_{hour}{minute}.npy"
            )
            np.save(save_path, zcast)

        else:
            continue

    except Exception as e:
        print(f"Skipping {entry['path']}: {e}")
