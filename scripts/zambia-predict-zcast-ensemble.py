import numpy as np
import os
from tqdm import tqdm
import re
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Model import
# ---------------------------------------------------------------------
sys.path.append("/home/users/mendrika/Zambia-Intercomparison/models/training")
from zambia_ensemble_hybrid import Core2MapModel

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LEAD_TIME = sys.argv[1]
YEAR  = sys.argv[2]
MONTH = sys.argv[3]
HOUR  = sys.argv[4]

ENSEMBLE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/checkpoints/ensemble_hybrid/t{LEAD_TIME}"
SCALER_PATH  = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/scaling/scaler_realcores.pt"
OUTPUT_BASE  = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/zcast/ensemble_nowcasts/t{LEAD_TIME}"

# force CPU
DEVICE = torch.device("cpu")

# threads
num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(num_threads)
print(f"Running on CPU with {num_threads} threads")

MC_SAMPLES = int(os.environ.get("MC_SAMPLES", 10))  # default 10 for CPU

os.makedirs(OUTPUT_BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"{YEAR}{MONTH}", f"{HOUR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load ensemble models
# ---------------------------------------------------------------------
ckpts = []
for root, _, files in os.walk(ENSEMBLE_DIR):
    for f in files:
        if f.endswith(".ckpt"):
            ckpts.append(os.path.join(root, f))
ckpts = sorted(ckpts)
if not ckpts:
    raise RuntimeError(f"No checkpoints found in {ENSEMBLE_DIR}")

models = []
for path in ckpts:
    model = Core2MapModel.load_from_checkpoint(path, map_location=DEVICE)
    model.eval()
    models.append(model)
print(f"Loaded {len(models)} ensemble models on CPU")

# ---------------------------------------------------------------------
# Keep dropout active for MC sampling
# ---------------------------------------------------------------------
def enable_dropout(model, p=0.05):
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()
            m.p = p

# ---------------------------------------------------------------------
# Load scaler
# ---------------------------------------------------------------------
scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean  = np.asarray(scaler["mean"])
scale = np.asarray(scaler["scale"])
COLS_TO_SCALE = [0,1,2,3,4,5,6,7]
MASK_COL_INDEX = 8
LAG_COL_INDEX  = 9

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def load_zcast_input(year, month, day, hour, minute):
    path = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/raw/inputs_t0/input-{year}{month}{day}_{hour}{minute}.pt"
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        raise FileNotFoundError(f"Missing/corrupt input: {path}")
    return torch.load(path, map_location="cpu")

# ---------------------------------------------------------------------
# Collect inputs
# ---------------------------------------------------------------------
zcast_root = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/raw/inputs_t0"
pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")
input_files = []
for f in sorted(os.listdir(zcast_root)):
    m = pattern.match(f)
    if m:
        y, mo, d, h, mi = m.groups()
        if y == YEAR and mo == MONTH and h == HOUR:
            input_files.append((y, mo, d, h, mi))
print(f"Detected {len(input_files)} input files for {YEAR}-{MONTH} {HOUR} UTC")

# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
for year, month, day, hour, minute in tqdm(input_files, desc="Predicting"):
    try:
        zcast_input = load_zcast_input(year, month, day, hour, minute)
        x = zcast_input["input_tensor"].unsqueeze(0).float()
        g = zcast_input["global_context"].unsqueeze(0).float()

        if x.shape != (1, 288, 10):
            continue

        mean_t = torch.tensor(np.nan_to_num(mean), dtype=x.dtype)
        scale_t = torch.tensor(np.nan_to_num(scale, nan=1.0), dtype=x.dtype)

        mask = x[:, :, MASK_COL_INDEX] > 0.5
        for col in COLS_TO_SCALE:
            coldata = x[:, :, col]
            coldata[mask] = (coldata[mask] - mean_t[col]) / scale_t[col]
            x[:, :, col] = coldata

        x[:, :, LAG_COL_INDEX] = 2 * (x[:, :, LAG_COL_INDEX] / 120.0) - 1.0

        preds_all = []
        for model in models:
            enable_dropout(model, p=0.05)
            mc_preds = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    mc_pred = torch.sigmoid(model(x, g)).squeeze(0).squeeze(0)
                    mc_preds.append(mc_pred)
            preds_all.append(torch.stack(mc_preds))

        preds_all = torch.stack(preds_all)  # (E, M, H, W)
        mean_pred = preds_all.mean(dim=(0,1))
        var_pred  = preds_all.var(dim=(0,1))

        np.save(os.path.join(OUTPUT_DIR, f"zcast_mean_{year}{month}{day}_{hour}{minute}.npy"),
                mean_pred.numpy())
        np.save(os.path.join(OUTPUT_DIR, f"zcast_var_{year}{month}{day}_{hour}{minute}.npy"),
                var_pred.numpy())

    except Exception as e:
        print(f"Skipping {year}-{month}-{day} {hour}:{minute}: {e}")

print("All ensemble nowcasts completed on CPU.")
