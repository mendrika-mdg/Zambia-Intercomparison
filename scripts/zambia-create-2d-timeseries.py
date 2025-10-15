from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime, timedelta
import sys
import torch
from scipy.ndimage import zoom

# ---------------------------------------------------------------------
# Utility subroutines
# ---------------------------------------------------------------------

def update_hour(date_dict, hours_to_add, minutes_to_add):
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )
    updated = time_obj + timedelta(hours=hours_to_add, minutes=minutes_to_add)
    new_date_dict = {
        "year":   f"{updated.year:04d}",
        "month":  f"{updated.month:02d}",
        "day":    f"{updated.day:02d}",
        "hour":   f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }
    file_path = f"{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}.nc"
    return {'time': new_date_dict, 'path': file_path}


def load_wavelet_dataset(year, month, day, hour, minute, lead_time):
    nowcast_origin = {"year": year, "month": month, "day": day, "hour": hour, "minute": minute}
    nowcast_lt = update_hour(nowcast_origin, hours_to_add=lead_time, minutes_to_add=0)["time"]
    path_core = f"/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/{nowcast_lt['year']}/{nowcast_lt['month']}"
    file = f"{path_core}/{nowcast_lt['year']}{nowcast_lt['month']}{nowcast_lt['day']}{nowcast_lt['hour']}{nowcast_lt['minute']}.nc"
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing wavelet file: {file}")
    data = np.array(Dataset(file, mode='r')["cores"])
    return data


def load_ZCAST_nowcast(year, month, day, hour, minute, lead_time):
    base_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/zcast/nowcasts/t{lead_time}"
    file = f"{base_dir}/zcast_{year}{month}{day}_{hour}{minute}.npy"
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing ZCAST file: {file}")
    data = np.load(file)
    return data

def load_NFLICS_nowcast(year, month, day, hour, minute, lead_time):
    NFLICS_BaseDir = f"/gws/ssde/j25b/swift/nflics_nowcasts/{year}/{month}/{day}/{hour}{minute}"
    file = f"{NFLICS_BaseDir}/Nowcast_{year}{month}{day}{hour}{minute}_000_sadc.nc"
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing NFLICS file: {file}")
    data = Dataset(file, mode='r')["Probability"][lead_time, :, :]
    return data

def load_NetNCC_nowcast(year, month, day, hour, minute, lead_time):
    path_NetNCC = f"/gws/ssde/j25b/swift/NetNCC_preds/{year}/{month}"
    file = f"{path_NetNCC}/{year}{month}{day}{hour}{minute}_leadtime_{lead_time}hr.nc"
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing NetNCC file: {file}")
    data = Dataset(file, mode='r')["core_prob"][:]
    return data


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Ground Truth (MSG extent)
y_min, y_max = 580, 930
x_min, x_max = 1480, 1850

# NFLICS subdomain
x_min_nflics = 365
x_max_nflics = 758
y_min_nflics = 256
y_max_nflics = 590

# NetNCC subdomain
x_min_netncc = 33
x_max_netncc = 493
y_min_netncc = 41
y_max_netncc = 410

# Target resolution for all (same as ZCAST and GT)
TARGET_SHAPE = (350, 370)

lead_time = int(sys.argv[1])
save_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/combined_nowcasts/t{lead_time}"
os.makedirs(save_dir, exist_ok=True)


# ---------------------------------------------------------------------
# Detect all available ZCAST nowcasts
# ---------------------------------------------------------------------

zcast_root = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/zcast/nowcasts/t{lead_time}"
pattern = re.compile(r"zcast_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.npy$")

nowcast_files = []
for root, _, files in os.walk(zcast_root):
    for f in files:
        m = pattern.match(f)
        if m:
            year, month, day, hour, minute = m.groups()
            nowcast_origin = datetime(int(year), int(month), int(day), int(hour), int(minute))
            full_path = os.path.join(root, f)
            nowcast_files.append({
                "year": year, "month": month, "day": day,
                "hour": hour, "minute": minute,
                "datetime": nowcast_origin, "path": full_path
            })

nowcast_files = sorted(nowcast_files, key=lambda x: x["datetime"])
print(f"Detected {len(nowcast_files)} ZCAST nowcast files.")


# ---------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------

for entry in tqdm(nowcast_files, desc=f"Processing lead time {lead_time}"):
    year, month, day, hour, minute = (
        entry["year"], entry["month"], entry["day"], entry["hour"], entry["minute"]
    )
    try:
        # Load each model’s nowcast
        zcast = load_ZCAST_nowcast(year, month, day, hour, minute, lead_time)
        ground_truth = load_wavelet_dataset(year, month, day, hour, minute, lead_time)[0, y_min:y_max, x_min:x_max] != 0
        nflics = load_NFLICS_nowcast(year, month, day, hour, minute, lead_time)[y_min_nflics:y_max_nflics, x_min_nflics:x_max_nflics]
        netncc = load_NetNCC_nowcast(year, month, day, hour, minute, lead_time)[y_min_netncc:y_max_netncc, x_min_netncc:x_max_netncc]

        # Ensure no negative values in NFLICS
        nflics = np.clip(nflics, 0, None)

        # Bilinear interpolation (scipy.ndimage.zoom with order=1)
        zoom_nflics = zoom(nflics, (TARGET_SHAPE[0]/nflics.shape[0], TARGET_SHAPE[1]/nflics.shape[1]), order=1)
        zoom_netncc = zoom(netncc, (TARGET_SHAPE[0]/netncc.shape[0], TARGET_SHAPE[1]/netncc.shape[1]), order=1)

        # Combine and save as one .pt file
        nowcast_dict = {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "lead_time": lead_time,
            "zcast": zcast.astype(np.float32),
            "ground_truth": ground_truth.astype(np.float32),
            "nflics": zoom_nflics.astype(np.float32),
            "netncc": zoom_netncc.astype(np.float32)
        }

        save_path = os.path.join(save_dir, f"{year}{month}{day}_{hour}{minute}.pt")
        torch.save(nowcast_dict, save_path)

    except Exception as e:
        print(f"Skipping {entry['path']}: {e}")

print(f"\nFinished processing {len(nowcast_files)} files for lead time t+{lead_time}.")
