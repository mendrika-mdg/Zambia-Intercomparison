#!/bin/bash
#SBATCH --job-name=extract_nflics
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/error/%j.err

set -e

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------

module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# NetCDF / HDF5 configuration for multi-node safety
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

lead_time=$1
script=/home/users/mendrika/Zambia-Intercomparison/scripts/zambia-extract-nflics.py

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

if [ -z "$lead_time" ]; then
    echo "Usage: sbatch $0 <lead_time>"
    echo "Example: sbatch $0 3"
    exit 1
fi

if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

echo "------------------------------------------------------------"
echo "Running Zambia NFLICS extraction"
echo "Lead time: t+$lead_time"
echo "Python script: $script"
echo "Start time: $(date)"
echo "------------------------------------------------------------"

# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------

python "$script" "$lead_time"

echo "------------------------------------------------------------"
echo "Job completed successfully at $(date)"
echo "------------------------------------------------------------"
