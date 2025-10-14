from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime, timedelta
import sys

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

def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    y_min = max(y - half, 0)
    y_max = min(y + half + 1, matrix.shape[0])
    x_min = max(x - half, 0)
    x_max = min(x + half + 1, matrix.shape[1])
    return matrix[y_min:y_max, x_min:x_max]


def save_city_nowcasts_by_lead(yx_locations, nowcast_origin, lead_time):
    base_save_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/time_series/zcast/t{lead_time}"
    os.makedirs(base_save_dir, exist_ok=True)
    save_path = f"{base_save_dir}/zambia_nowcasts_t{lead_time}.csv"

    if isinstance(nowcast_origin, datetime):
        nowcast_origin_str = nowcast_origin.strftime("%Y-%m-%d %H:%M")
    else:
        nowcast_origin_str = str(nowcast_origin)

    row = {"t0": nowcast_origin_str}
    for city, (_, _, zcast_val, gt_val) in yx_locations.items():
        row[f"{city}_t{lead_time}"] = round(zcast_val, 4)
        row[f"{city}_gt_lt{lead_time}"] = int(gt_val)

    df = pd.DataFrame([row])

    if os.path.exists(save_path):
        df.to_csv(save_path, mode="a", header=False, index=False)
    else:
        df.to_csv(save_path, mode="w", header=True, index=False)

    print(f"Appended results for {nowcast_origin_str} to {save_path}")


# ---------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------

# Zambia extent from MSG
y_min, y_max = 580, 930
x_min, x_max = 1480, 1850

lead_time = int(sys.argv[1])

# Predefined city locations with fixed indices
locations = {
    "Lusaka":      {"lat": -15.3875, "lon": 28.3228, "y_netncc": 150, "x_netncc": 265, "y_nflics": 493, "x_nflics": 564, "y_gt": 95,  "x_gt": 196},
    "Ndola":       {"lat": -12.9683, "lon": 28.6366, "y_netncc": 232, "x_netncc": 286, "y_nflics": 415, "x_nflics": 573, "y_gt": 177, "x_gt": 217},
    "Kasama":      {"lat": -10.2129, "lon": 31.1808, "y_netncc": 330, "x_netncc": 371, "y_nflics": 327, "x_nflics": 643, "y_gt": 275, "x_gt": 302},
    "Chinsali":    {"lat": -10.5500, "lon": 32.0667, "y_netncc": 319, "x_netncc": 394, "y_nflics": 338, "x_nflics": 668, "y_gt": 264, "x_gt": 325},
    "Kabwe":       {"lat": -14.4469, "lon": 28.4464, "y_netncc": 181, "x_netncc": 273, "y_nflics": 463, "x_nflics": 567, "y_gt": 126, "x_gt": 204},
    "Livingstone": {"lat": -17.8419, "lon": 25.8542, "y_netncc":  64, "x_netncc": 179, "y_nflics": 571, "x_nflics": 495, "y_gt":   9, "x_gt": 110},
    "Mongu":       {"lat": -15.2484, "lon": 23.1274, "y_netncc": 150, "x_netncc": 108, "y_nflics": 488, "x_nflics": 420, "y_gt":  95, "x_gt":  39},
    "Mansa":       {"lat": -11.1996, "lon": 28.8943, "y_netncc": 294, "x_netncc": 301, "y_nflics": 359, "x_nflics": 580, "y_gt": 239, "x_gt": 232},
    "Solwezi":     {"lat": -12.1722, "lon": 26.3981, "y_netncc": 258, "x_netncc": 222, "y_nflics": 390, "x_nflics": 511, "y_gt": 203, "x_gt": 153},
    "Chipata":     {"lat": -13.6339, "lon": 32.6508, "y_netncc": 213, "x_netncc": 397, "y_nflics": 437, "x_nflics": 684, "y_gt": 158, "x_gt": 328}
}


# ---------------------------------------------------------------------
# Automatically detect all available ZCAST nowcasts
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
    nowcast_origin = entry["datetime"]

    try:
        zcast = load_ZCAST_nowcast(year, month, day, hour, minute, lead_time)
        ground_truth = load_wavelet_dataset(year, month, day, hour, minute, lead_time)[0, y_min:y_max, x_min:x_max] != 0

        yx_locations = {}
        for name, info in locations.items():
            y_zcast = info["y_gt"]
            x_zcast = info["x_gt"]
            y_gt = info["y_gt"]
            x_gt = info["x_gt"]

            nflics_val = float(np.mean(extract_box(zcast, y_zcast, x_zcast)))
            gt_val     = int(np.max(extract_box(ground_truth, y_gt, x_gt)))

            yx_locations[name] = (y_zcast, x_zcast, round(nflics_val, 4), gt_val)

        save_city_nowcasts_by_lead(yx_locations, nowcast_origin, lead_time)

    except Exception as e:
        print(f"Skipping {entry['path']}: {e}")

print(f"\n Finished processing {len(nowcast_files)} files for lead time t+{lead_time}.")
