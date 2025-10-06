#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np      
from netCDF4 import Dataset  
from scipy.ndimage import label
from skimage.transform import resize
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics

def prepare_core(file):

    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    try:
        # using a context manager to ensure proper file closure
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][0, :, :]
    except OSError as e:
        raise OSError(f"Error opening NetCDF file: {file}. {e}")

    return cores


def update_hour(date_dict, hours_to_add, minutes_to_add):
    """
    Add hours and minutes to a datetime dictionary and return the updated dict and a generated file path.

    Args:
        date_dict     (dict): Keys: 'year', 'month', 'day', 'hour', 'minute' as strings, e.g. "01", "23"
        hours_to_add   (int): Number of hours to add.
        minutes_to_add (int): Number of minutes to add.

    Returns:
        tuple:
            - dict: Updated datetime dictionary with all fields as zero-padded strings.
            - str: File path in the format YYYY/MM/YYYYMMDDHHMM.nc
    """
    # Parse the original time
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )

    # Add hours
    updated = time_obj + timedelta(hours=hours_to_add, minutes=minutes_to_add)

    # Format updated dictionary
    new_date_dict = {
        "year":   f"{updated.year:04d}",
        "month":  f"{updated.month:02d}",
        "day":    f"{updated.day:02d}",
        "hour":   f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }

    # Generate file path
    file_path = f"{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}.nc"


    return {'time': new_date_dict, 'path': file_path}


def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Compute Haversine distance between two points or arrays of points.
        Inputs are in degrees. Output is in kilometers.

        Supports both scalar and array inputs (NumPy).
        """
        R = 6371.0  # Earth radius in kilometers

        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c


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
    cores_t = data_t["cores"][0, :, :]
    tir_t   = data_t['tir'][0, :, :]
    Pmax_lat, Pmax_lon = data_t["max_lat"][:], data_t["max_lon"][:]

    # label all cores
    labeled_array, _ = label(cores_t != 0)     
    core_labels = np.unique(labeled_array[labeled_array != 0])

    # creating database of sizes, intensities and ctts
    dict_storm_size = {lab: np.sum(labeled_array == lab) * 9 for lab in core_labels}

    # Compute min temperature of a core but based on 3x3 average around min TIR
    dict_storm_temperature = {}

    for lab in core_labels:

        mask = (labeled_array == lab)
        tir_core = tir_t[mask]      

        # tir_core is a 1D array                   
        min_index = np.argmin(tir_core)        

        # Get absolute indices of the min location
        yx_indices = np.argwhere(mask)[min_index]
        y, x = yx_indices

        # extract a 3x3 box around the location with min temperature
        box = extract_box(tir_t, y, x)
        avg_tir = float(np.mean(box))
        dict_storm_temperature[lab] = avg_tir

    storm_database = {}
    for lat, lon in zip(Pmax_lat, Pmax_lon):
        try:
            y, x = snflics.to_yx(lat, lon, lats, lons)
        except IndexError:
            continue
        lab = labeled_array[y, x]
        if lab == 0 or lab in storm_database:
            continue
        storm_database[int(lab)] = {
            "lat": lat, 
            "lon": lon, 
            "wp": dict_storm_intensity[lab], 
            "tir": dict_storm_temperature[lab],
            "size": dict_storm_size[lab], 
            "mask": 1
        }
    return storm_database


def crop_to_2048(image):
    """
    Crop a (2079, 2263) image to (2048, 2048) by:
    - Keeping the bottom 2048 rows (northernmost)
    - Keeping the leftmost 2048 columns (westernmost)
    - latitude meshgrid has the north in bottom and south in top
    """
    return image[-2048:, :2048]


def resize_image(data, target_shape=(512, 512)):
    """
    Resize a 2D image (regular or masked) to the given shape using bilinear interpolation.
    If the input is a masked array, the mask is ignored.

    Args:
        data (np.ndarray or np.ma.MaskedArray): Input 2D array.
        target_shape (tuple): Desired output shape (height, width).

    Returns:
        np.ndarray: Resized image (float32 or same dtype as input).
    """
    # Extract raw data (ignore mask if present)
    if np.ma.isMaskedArray(data):
        data = data.data

    resized_data = resize(
        data,
        output_shape=target_shape,
        order=1,               # bilinear interpolation
        mode="reflect",        # handles borders
        anti_aliasing=True,
        preserve_range=True
    ).astype(data.dtype)

    return resized_data


def generate_fictional_storm(context_lat_min, context_lat_max, context_lon_min, context_lon_max, min_km_buffer=10, max_deg_buffer=4.5):
    """
    Generate a synthetic storm outside context domain but near enough.

    Returns:
        tuple: (storm_id, storm_dict)
    """
    lat_range = (context_lat_min - max_deg_buffer, context_lat_max + max_deg_buffer)
    lon_range = (context_lon_min - max_deg_buffer, context_lon_max + max_deg_buffer)
    while True:
        lat, lon = np.random.uniform(*lat_range), np.random.uniform(*lon_range)
        if context_lat_min <= lat <= context_lat_max and context_lon_min <= lon <= context_lon_max:
            continue
        d_north = haversine_distance(lat, lon, context_lat_max, lon)
        d_south = haversine_distance(lat, lon, context_lat_min, lon)
        d_east  = haversine_distance(lat, lon, lat, context_lon_max)
        d_west  = haversine_distance(lat, lon, lat, context_lon_min)
        if min(d_north, d_south, d_east, d_west) < min_km_buffer:
            continue

        # lat lon in the buffer zone
        # warm enough to be non-convective, realistic for Africa, covers both day and night
        return ('artificial', {'lat': lat, 'lon': lon, 'wp': 0.0, 'tir': float(np.random.uniform(20.0, 35.0)), 'size': 0, 'mask': 0})


def pad_observed_storms(storm_db, nb_x0, context_lat_min, context_lat_max, context_lon_min, context_lon_max):

    # Convert dict to list of (key, value) tuples
    storm_list = list(storm_db.items())

    if len(storm_list) >= nb_x0:
        # taking the nb_x0 strongest cores if there are more than the max number of cores allowed by the model
        sorted_db = sorted(storm_list, key=lambda item: item[1]['tir'], reverse=False)
        return sorted_db[:nb_x0]
    else:
        # apply padding when there are less cores observed at time t0
        needed = nb_x0 - len(storm_list)
        storm_list.extend([
            generate_fictional_storm(
                context_lat_min=context_lat_min, 
                context_lat_max=context_lat_max, 
                context_lon_min=context_lon_min,
                context_lon_max=context_lon_max
            ) 
            for _ in range(needed)
        ])
        return storm_list


geodata = np.load("/gws/nopw/j04/cocoon/SSA_domain/lat_lon_2268_2080.npz")
lons = geodata["lon"][:]
lats = geodata["lat"][:]


def get_latlon_bounds(lat_grid, lon_grid):
    lat_grid = np.where(lat_grid == -999.999, np.nan, lat_grid)
    lon_grid = np.where(lon_grid == -999.999, np.nan, lon_grid)

    return {
        "lat_min": np.nanmin(lat_grid),
        "lat_max": np.nanmax(lat_grid),
        "lon_min": np.nanmin(lon_grid),
        "lon_max": np.nanmax(lon_grid)
    }

bounds = get_latlon_bounds(lats, lons)
print(bounds)


lats_centered  = crop_to_2048(lats)
lons_centered  = crop_to_2048(lons)


# Target domain
bounds = get_latlon_bounds(lats_centered, lons_centered)
TARGET_LAT_MIN = float(bounds['lat_min'])
TARGET_LAT_MAX = float(bounds['lat_max'])
TARGET_LON_MIN = float(bounds['lon_min'])
TARGET_LON_MAX = float(bounds['lon_max'])

CONTEXT_MARGIN = 4.5  # degrees
CONTEXT_LAT_MIN = TARGET_LAT_MIN - CONTEXT_MARGIN
CONTEXT_LAT_MAX = TARGET_LAT_MAX + CONTEXT_MARGIN
CONTEXT_LON_MIN = TARGET_LON_MIN - CONTEXT_MARGIN
CONTEXT_LON_MAX = TARGET_LON_MAX + CONTEXT_MARGIN


def transform_to_array(time_obs, data, time_lag):
    """
    Transform list of (id, dict) into numpy array.
    """

    year = int(time_obs['year'])
    month = int(time_obs['month'])
    day = int(time_obs['day'])
    hour = int(time_obs['hour'])
    minute = int(time_obs['minute'])
    result = []
    
    for _, entry in data:
        lat = float(entry['lat'])
        lon = float(entry['lon'])
        tir = float(entry['tir'])
        wp = float(entry['wp'])
        size = int(entry['size'])
        mask = int(entry['mask'])
        result.append([year, month, day, hour, minute, lat, lon, tir, wp, size, mask, time_lag])
    
    return np.array(result)


def process_file(file_t, time_t0, nb_x0, time_lag):
    try:        
        with Dataset(file_t, "r") as data_t:

            x0_lat, x0_lon = data_t["max_lat"][:], data_t["max_lon"][:]
            if x0_lat.size == 0 or x0_lon.size == 0:
                return

            storm_database = create_storm_database(data_t, lats, lons)

            X_features = pad_observed_storms(storm_database, nb_x0,
                                               CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                                               CONTEXT_LON_MIN, CONTEXT_LON_MAX)

            input_features = transform_to_array(time_t0, X_features, time_lag)
            input_tensor = torch.tensor(input_features, dtype=torch.float32)

        return input_tensor

    except Exception as e:
        print(f"Error on {file_t}: {e}")


YEAR = "2021"

# Data where all the historical cores are located and the output folder
DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
OUTPUT_FOLDER = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Pancast-v1-512"

# year of interest
all_files = [file for file in snflics.all_files_in(DATA_PATH) if snflics.get_time(file)["year"] == YEAR]
all_files.sort()

# Number of storms to consider (after analysing the whole dataset)
NB_X0 = 140


for file_t in all_files[10:11]:

    # nowcast origin
    time_t = snflics.get_time(file_t)

    # lag times considered in minute
    lag_before_t = [0, 30, 60, 90, 120]
    file_before_t = [DATA_PATH + update_hour(time_t, hours_to_add=0, minutes_to_add=-m)["path"] for m in lag_before_t]

    # lead times considered in hour
    lead_times = [0, 1, 3, 6]
    file_lead_times = [DATA_PATH + update_hour(time_t, hours_to_add=h, minutes_to_add=0)["path"] for h in lead_times]     

    # output folders
    NOWCAST_ORIGIN = f"{time_t['year']}{time_t['month']}{time_t['day']}_{time_t['hour']}{time_t['minute']}"

    INPUT_LT0 = f"{OUTPUT_FOLDER}/inputs_t0/input-{NOWCAST_ORIGIN}.pt"
    OUTPUT_PATHS = {
        f"LT{i}": f"{OUTPUT_FOLDER}/targets_t{i}/target-{NOWCAST_ORIGIN}.pt"
        for i in lead_times
    }

    # If all past and forecast files exist
    if all(os.path.exists(f) for f in file_lead_times) and all(os.path.exists(f) for f in file_before_t):
        try:
            core_series = [prepare_core(f) for f in file_lead_times]
        except OSError:
            continue

        with Dataset(file_t, "r") as data_t:
            Pmax_lat = data_t["max_lat"][:]
            Pmax_lon = data_t["max_lon"][:]

            if Pmax_lat.size != 0:

                input_tensor = []
                for i, f in enumerate(file_before_t):
                    input_tensor.append(process_file(f, time_t, NB_X0, lag_before_t[i]))
                input_tensor = torch.cat(input_tensor, dim=0)
                torch.save(input_tensor, INPUT_LT0)

                # Process targets for each lead time
                for i, core in enumerate(core_series):
                    core_centered = crop_to_2048(core)
                    core_resized  = resize_image(core_centered, target_shape=(512, 512))
                    target_tensor = torch.tensor(core_resized != 0, dtype=torch.uint8)
                    output_file_path = OUTPUT_PATHS[f"LT{lead_times[i]}"]
                    torch.save(target_tensor, output_file_path)



