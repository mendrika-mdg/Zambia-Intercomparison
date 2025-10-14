#!/bin/bash
#SBATCH --job-name=predict-zcast
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
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

# Parameters
lead_time=$1

script=/home/users/mendrika/Zambia-Intercomparison/scripts/zambia-predict-zcast.py

# Verify the script exists
if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

# Run
python "$script" "$lead_time"

echo "Job completed successfully."
