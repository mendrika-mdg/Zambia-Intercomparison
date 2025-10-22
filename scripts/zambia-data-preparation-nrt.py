import os
import sys
import torch
import numpy as np      
from netCDF4 import Dataset  
from scipy.ndimage import label
from skimage.transform import resize
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics # type: ignore


# For a given region, add yx bounds and context domain
y_min, y_max = 580, 930
x_min, x_max = 1476, 1846

# Import geodata and crop it accordingly
geodata = Dataset("/gws/ssde/j25b/swift/rt_cores/geoloc_grids/nxny2268_2080_nxnyds164580_blobdx0.04491576_arean41_n27_27_79.nc")
lons = geodata["lons_mid"][y_min:y_max, x_min:x_max]
lats = geodata["lats_mid"][y_min:y_max, x_min:x_max]

CONTEXT_LAT_MIN = np.min(lats)
CONTEXT_LAT_MAX = np.max(lats)
CONTEXT_LON_MIN = np.min(lons)
CONTEXT_LON_MAX = np.max(lons)

def get_time(file_path):
    """
    Extract zero-padded time components from a file path like:
    /.../2025/02/05/1345/Convective_struct_extended_202502051345_000.nc

    Returns:
        dict with keys: 'year', 'month', 'day', 'hour', 'minute'
    """
    basename = os.path.basename(file_path)

    # Extract the datetime string from the filename
    parts = basename.split("_")
    
    timestamp = parts[-2]
    if len(timestamp) != 12:
        raise ValueError(f"Invalid timestamp format in filename: {timestamp}")

    return {
        "year":   timestamp[0:4],
        "month":  timestamp[4:6],
        "day":    timestamp[6:8],
        "hour":   timestamp[8:10],
        "minute": timestamp[10:12]
    }

# Cropped core data
def prepare_core(file):

    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    try:
        # using a context manager to ensure proper file closure
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][y_min:y_max, x_min:x_max]
    except OSError as e:
        raise OSError(f"Error opening NetCDF file: {file}. {e}")
    
    return cores

def update_hour(date_dict, hours_to_add, minutes_to_add):
    """
    Add hours to a datetime dictionary and return the updated dict and a generated file path.

    Args:
        date_dict     (dict): Keys: 'year', 'month', 'day', 'hour', 'minute' as strings, e.g. "01", "23"
        hours_to_add   (int): Number of hours to add.
        minutes_to_add (int): Number of minutes to add.

    Returns:
        dict: {
            'time': updated time dictionary with zero-padded strings,
            'path': file path in format /.../YYYYMMDDHHMM_000.nc
        }
    """
    # Parse original time
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )

    # Add hours and minutes
    updated = time_obj + timedelta(hours=hours_to_add, minutes=minutes_to_add)

    # Create updated dictionary with padded strings
    new_date_dict = {
        "year":   f"{updated.year:04d}",
        "month":  f"{updated.month:02d}",
        "day":    f"{updated.day:02d}",
        "hour":   f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }

    # Build file path safely using single quotes inside f-strings
    path_core = f"/gws/ssde/j25b/swift/rt_cores/{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['day']}/{new_date_dict['hour']}{new_date_dict['minute']}"
    file_path = f"{path_core}/Convective_struct_extended_{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}_000.nc"

    return {'time': new_date_dict, 'path': file_path}


def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    y_min = max(y - half, 0)
    y_max = min(y + half + 1, matrix.shape[0])
    x_min = max(x - half, 0)
    x_max = min(x + half + 1, matrix.shape[1])
    return matrix[y_min:y_max, x_min:x_max]


def create_storm_database(data_t, lats, lons):
    """
    Identify storm cores and extract features for each core.

    Args:
        data_t (Dataset): Dataset containing 'cores' and 'tir' variables.
        lats, lons (np.ndarray): 2D lat/lon arrays of the domain.

    Returns:
        dict: Storm database indexed by core label.
    """

    # Crop domain
    tir_t   = data_t["cores"][y_min:y_max, x_min:x_max]

    if not np.any(tir_t):
        return {}
    
    # Max lat/lon of all detected power maxima
    Pmax_lat, Pmax_lon = data_t["Pmax_lat"][:], data_t["Pmax_lon"][:]

    # Restrict to context window
    valid = (
        (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
        (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
    )
    Pmax_lat, Pmax_lon = Pmax_lat[valid], Pmax_lon[valid]

    # Label connected components
    labeled_array, _ = label(tir_t < 0)
    core_labels = np.unique(labeled_array[labeled_array != 0])

    # Core-wise properties
    dict_storm_size      = {lab: np.sum(labeled_array == lab) * 9 for lab in core_labels}

    dict_storm_extent = {}
    for lab in core_labels:
        mask = labeled_array == lab
        dict_storm_extent[lab] = {
            "lat_min": float(np.nanmin(lats[mask])),
            "lat_max": float(np.nanmax(lats[mask])),
            "lon_min": float(np.nanmin(lons[mask])),
            "lon_max": float(np.nanmax(lons[mask]))
        }

    # Minimum TIR (3×3 mean around coldest pixel)
    dict_storm_temperature = {}
    for lab in core_labels:
        mask = labeled_array == lab
        tir_core = tir_t[mask]
        yx_indices = np.argwhere(mask)
        y, x = yx_indices[np.argmin(tir_core)]
        box = extract_box(tir_t, y, x)
        dict_storm_temperature[lab] = float(np.mean(box))

    # Assemble final database
    storm_database = {}
    for lat, lon in zip(Pmax_lat, Pmax_lon):
        try:
            y, x = snflics.to_yx(lat, lon, lats, lons)
            if y is None or x is None:
                continue
        except (IndexError, TypeError):
            continue
        lab = labeled_array[y, x]
        if lab == 0 or lab in storm_database:
            continue

        ext = dict_storm_extent[lab]
        storm_database[int(lab)] = {
            "lat": lat,
            "lon": lon,
            "lat_min": ext["lat_min"],
            "lat_max": ext["lat_max"],
            "lon_min": ext["lon_min"],
            "lon_max": ext["lon_max"],
            "tir": dict_storm_temperature[lab],
            "size": dict_storm_size[lab],
            "mask": 1,
        }
    return storm_database
    

def generate_fictional_storm(context_lat_min, context_lat_max,
                             context_lon_min, context_lon_max):
    """
    Generate a dummy (non-convective) storm entry with mask=0.
    Used when padding cores — values don't affect the model.

    Returns:
        tuple: (storm_id, storm_dict)
    """
    # Pick a random coordinate in or near the domain (anywhere is fine)
    lat = np.random.uniform(context_lat_min, context_lat_max)
    lon = np.random.uniform(context_lon_min, context_lon_max)

    # Create consistent placeholder values
    storm = {
        "lat": lat,
        "lon": lon,
        "lat_min": lat,
        "lat_max": lat,
        "lon_min": lon,
        "lon_max": lon,
        "tir": 30.0,     # warm non-convective background
        "size": 0.0,     
        "mask": 0        # ensures Transformer ignores it
    }

    return ("artificial", storm)


def pad_observed_storms(storm_db, nb_x0,
                        context_lat_min, context_lat_max,
                        context_lon_min, context_lon_max):
    """
    Ensure a fixed number of storm cores by either truncating or padding.

    Args:
        storm_db (dict): Dictionary of observed storms {id: storm_dict}.
        nb_x0 (int): Target number of cores expected by the model.
        context_lat_min, context_lat_max, context_lon_min, context_lon_max (float): 
            Domain boundaries.

    Returns:
        list: List of (storm_id, storm_dict) tuples of length nb_x0.
    """

    # Convert dict to list
    storm_list = list(storm_db.items())

    # --- CASE 1: too many cores, keep the strongest (coldest) ones ---
    if len(storm_list) >= nb_x0:
        # Sort by TIR ascending (colder = stronger convection)
        sorted_db = sorted(storm_list, key=lambda item: item[1]['tir'])
        return sorted_db[:nb_x0]

    # --- CASE 2: too few cores, pad with artificial storms ---
    needed = nb_x0 - len(storm_list)
    for _ in range(needed):
        storm_list.append(
            generate_fictional_storm(
                context_lat_min=context_lat_min,
                context_lat_max=context_lat_max,
                context_lon_min=context_lon_min,
                context_lon_max=context_lon_max
            )
        )

    return storm_list


def transform_to_array(data, time_lag):
    """
    Transform list of (id, dict) storm entries into a NumPy array.

    Each row corresponds to one storm core at a given time lag.

    Args:
        data (list): List of (id, storm_dict) tuples from pad_observed_storms().
        time_lag (int): Time lag index (e.g., 0, 1, 2).

    Returns:
        np.ndarray: Array of shape (N, F) with per-core features.
    """

    result = []
    for _, entry in data:
        lat      = float(entry["lat"])
        lon      = float(entry["lon"])
        lat_min  = float(entry.get("lat_min", lat))
        lat_max  = float(entry.get("lat_max", lat))
        lon_min  = float(entry.get("lon_min", lon))
        lon_max  = float(entry.get("lon_max", lon))
        tir      = float(entry["tir"])
        size     = float(entry["size"])
        mask     = int(entry["mask"])

        # Define the per-core feature vector (local only)
        # [lat, lon, lat_min, lat_max, lon_min, lon_max, tir, size, wp, mask, lag]
        result.append([
            lat, lon,
            lat_min, lat_max,
            lon_min, lon_max,
            tir, size,
            mask, time_lag
        ])

    return np.array(result, dtype=np.float32)


# Number of storms to consider (after analysing the whole dataset)
def process_file(file_t, nb_x0, time_lag,
                 lats, lons,
                 CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                 CONTEXT_LON_MIN, CONTEXT_LON_MAX):
    """
    Process one NetCDF file and return the per-core input tensor for the model.

    Args:
        file_t (str): Path to NetCDF file at time t.
        nb_x0 (int): Number of storm cores to keep/pad (e.g. 96).
        time_lag (int): Lag index (0, 1, 2).
        lats, lons (np.ndarray): 2D arrays for lat/lon of the domain grid.
        CONTEXT_LAT_MIN/MAX, CONTEXT_LON_MIN/MAX (float): Domain boundaries.

    Returns:
        torch.Tensor: Tensor of shape (nb_x0, F) containing per-core features.
    """
    try:
        with Dataset(file_t, "r") as data_t:
            # Read storm maxima
            x0_lat = data_t["Pmax_lat"][:]
            x0_lon = data_t["Pmax_lon"][:]
            if x0_lat.size == 0 or x0_lon.size == 0:
                return None  # nothing to process

            # Extract storm features for this file
            storm_database = create_storm_database(data_t, lats, lons)

            # Pad/truncate to nb_x0 cores
            X_features = pad_observed_storms(
                storm_database, nb_x0,
                CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                CONTEXT_LON_MIN, CONTEXT_LON_MAX
            )

            # Convert to numpy array
            input_features = transform_to_array(X_features, time_lag)

            # Convert to tensor
            input_tensor = torch.tensor(input_features, dtype=torch.float32)

        return input_tensor

    except Exception as e:
        print(f"Error processing {file_t}: {e}")
        return None

NB_X0 = 96

YEAR = sys.argv[1]

# Data where all the historical cores are located and the output folder
DATA_PATH = "/gws/ssde/j25b/swift/rt_cores/2025"
OUTPUT_FOLDER = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/raw/2025"

# year of interest
all_files = [file for file in snflics.all_files_in(DATA_PATH) if get_time(file)["year"] == YEAR and get_time(file)["month"] in ["02"]]
all_files.sort()

for file_t in all_files[:]:

    # Current nowcast origin time
    time_t = get_time(file_t)

    # Lag times (in minutes) before t
    lag_before_t = [0, 60, 120]  # t0, t-1h, t-2h
    file_before_t = [
        update_hour(time_t, hours_to_add=0, minutes_to_add=-m)["path"]
        for m in lag_before_t
    ]

    # Lead times (in hours) after t
    lead_times = [0, 1, 2, 4, 6]
    file_lead_times = [
        update_hour(time_t, hours_to_add=h, minutes_to_add=0)["path"]
        for h in lead_times
    ]

    # Time components
    year   = int(time_t["year"])
    month  = int(time_t["month"])
    day    = int(time_t["day"])
    hour   = int(time_t["hour"])
    minute = int(time_t["minute"])

    # Output paths
    NOWCAST_ORIGIN = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
    INPUT_LT0 = f"{OUTPUT_FOLDER}/inputs_t0/input-{NOWCAST_ORIGIN}.pt"
    OUTPUT_PATHS = {
        f"LT{i}": f"{OUTPUT_FOLDER}/targets_t{i}/target-{NOWCAST_ORIGIN}.pt"
        for i in lead_times
    }

    # Check that all required files exist
    if not (all(os.path.exists(f) for f in file_lead_times) and all(os.path.exists(f) for f in file_before_t)):
        print(f"Missing required files for {file_t}")
        continue

    try:
        # Prepare targets (future lead-time cores)
        core_series = [prepare_core(f) for f in file_lead_times]
    except OSError:
        print(f"Skipping {file_t}: unreadable core file.")
        continue

    # Skip if any of the files is invalid or empty
    if any((c is None) or (not np.any(c)) for c in core_series):
        print(f"Skipping {file_t}: one or more lead-time core files are empty.")
        continue

    # Check if there are valid cores at t0
    with Dataset(file_t, "r") as data_t:
        Pmax_lat = data_t["Pmax_lat"][:]
        Pmax_lon = data_t["Pmax_lon"][:]

        valid = (
            (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
            (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
        )
        Pmax_lat, Pmax_lon = Pmax_lat[valid], Pmax_lon[valid]

        if Pmax_lat.size == 0:
            print(f"No core in the domain for {NOWCAST_ORIGIN}")
            continue

    # Process inputs for each lag (t0, t-1h, t-2h)
    input_tensors = []
    for i, f in enumerate(file_before_t):
        t_tensor = process_file(
            f,
            nb_x0=NB_X0,
            time_lag=lag_before_t[i],
            lats=lats,
            lons=lons,
            CONTEXT_LAT_MIN=CONTEXT_LAT_MIN,
            CONTEXT_LAT_MAX=CONTEXT_LAT_MAX,
            CONTEXT_LON_MIN=CONTEXT_LON_MIN,
            CONTEXT_LON_MAX=CONTEXT_LON_MAX
        )
        if t_tensor is not None:
            input_tensors.append(t_tensor)

    if not input_tensors:
        print(f"No valid input tensor for {NOWCAST_ORIGIN}")
        continue

    # Concatenate all lag inputs into one tensor
    input_tensor = torch.cat(input_tensors, dim=0)  # (288, F)

    if input_tensor.shape != (288, 10):
        print(f"Missing one or more input lags for {NOWCAST_ORIGIN}")
        continue

    # Compute global context (month/time sin–cos)
    month_angle = 2 * np.pi * (month - 1) / 12
    tod_angle   = 2 * np.pi * (hour + minute / 60.0) / 24
    global_context = torch.tensor([
        np.sin(month_angle), np.cos(month_angle),
        np.sin(tod_angle), np.cos(tod_angle)
    ], dtype=torch.float32)

    # Save both input tensor and global context
    torch.save({
        "input_tensor": input_tensor,
        "global_context": global_context,
        "nowcast_origin": NOWCAST_ORIGIN
    }, INPUT_LT0)
    print(f"Saved input tensor + global context: {INPUT_LT0}")

    # Save binary targets with lead-time metadata
    for i, (h, core) in enumerate(zip(lead_times, core_series)):
        target_tensor = torch.tensor(core != 0, dtype=torch.uint8)
        output_file_path = OUTPUT_PATHS[f"LT{h}"]

        if target_tensor.shape != (350, 370):
            print(f"Unexpected target shape {target_tensor.shape} for lead {h}h at {NOWCAST_ORIGIN}")
            continue 

        torch.save({
            "data": target_tensor,           # (350, 370)
            "lead_time": h,                  # hours ahead
            "nowcast_origin": NOWCAST_ORIGIN # YYYYMMDD_HHMM
        }, output_file_path)

    print(f"Saved {len(lead_times)} targets for {NOWCAST_ORIGIN}")
