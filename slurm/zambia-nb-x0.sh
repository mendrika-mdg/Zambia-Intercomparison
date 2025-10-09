#!/bin/bash

#SBATCH --job-name=nbx0-zambia
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/error/%j.err

# Exit immediately if any command fails
set -e

# Load Python environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# Define region parameters
domain_lat_min=-18.414806
domain_lat_max=-7.9918404
domain_lon_min=21.167515
domain_lon_max=35.316326
region_name="zambia"

# Run Python script with arguments
python /home/users/mendrika/Zambia-Intercomparison/scripts/zambia-nb-x0.py \
  "$domain_lat_min" "$domain_lat_max" "$domain_lon_min" "$domain_lon_max" "$region_name"

echo "Job completed successfully."