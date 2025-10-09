import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime

# Local module
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics

# --- Read command-line arguments ---
domain_lat_min = float(sys.argv[1])
domain_lat_max = float(sys.argv[2])
domain_lon_min = float(sys.argv[3])
domain_lon_max = float(sys.argv[4])
region_name = sys.argv[5]

# --- Paths ---
data_path = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
output_dir = f"/home/users/mendrika/Zambia-Intercomparison/outputs/nbx0"
os.makedirs(output_dir, exist_ok=True)

# --- Start logging ---
start_time = datetime.now()
print(f"\n[{start_time:%Y-%m-%d %H:%M:%S}] Starting storm count extraction for {region_name}")
print(f"Domain: lat {domain_lat_min}–{domain_lat_max}, lon {domain_lon_min}–{domain_lon_max}\n")

# --- File list ---
all_files = sorted(snflics.all_files_in(data_path))
print(f"Found {len(all_files)} files to process.\n")

storm_counts = []
processed = 0
skipped = 0

# --- Process files ---
for file_t0 in all_files:
    try:
        if not os.path.exists(file_t0):
            print(f"File not found: {file_t0}")
            skipped += 1
            continue

        time_t0 = snflics.get_time(file_t0)
        if time_t0["month"] in ["12", "01", "02"]:  # DJF season
            with Dataset(file_t0, "r") as data_t0:
                latitudes = data_t0["max_lat"][:].compressed()
                longitudes = data_t0["max_lon"][:].compressed()

                valid_indices = (
                    (longitudes >= domain_lon_min) & (longitudes <= domain_lon_max) &
                    (latitudes >= domain_lat_min) & (latitudes <= domain_lat_max)
                )
                count = np.count_nonzero(valid_indices)
                if count > 0:
                    storm_counts.append(count)
        processed += 1

        if processed % 500 == 0:
            print(f"Processed {processed}/{len(all_files)} files...")

    except OSError:
        print(f"Corrupted or unreadable file skipped: {file_t0}")
        skipped += 1
        continue
    except Exception as e:
        print(f"Error processing {file_t0}: {e}")
        skipped += 1
        continue

# --- Save results ---
if storm_counts:
    storm_counts = np.array(storm_counts)
    np.save(f"{output_dir}/nbx0-{region_name}.npy", storm_counts)
    print(f"\nSaved storm count array ({storm_counts.size} entries) to {output_dir}/nbx0-{region_name}.npy")
else:
    print("No valid storm data was found.")

# --- Final summary ---
end_time = datetime.now()
print(f"\n[{end_time:%Y-%m-%d %H:%M:%S}] Job completed successfully.")
print(f"Processed: {processed}, Skipped: {skipped}, Valid storm files: {len(storm_counts)}")
print(f"Total runtime: {end_time - start_time}\n")
