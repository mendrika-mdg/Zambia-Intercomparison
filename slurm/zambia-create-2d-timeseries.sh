#!/bin/bash
#SBATCH --job-name=combined-nowcast
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/error/%j.err

set -e

# Load environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# Optional: tune for NetCDF performance
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# Parameters
lead_time=$1

script=/home/users/mendrika/Zambia-Intercomparison/scripts/zambia-create-2d-timeseries-ensemble.py

# Verify the script exists
if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

# Run
python "$script" "$lead_time"

echo "Job completed successfully."
