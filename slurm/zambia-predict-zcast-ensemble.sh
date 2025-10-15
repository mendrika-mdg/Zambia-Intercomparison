#!/bin/bash
#SBATCH --job-name=predict-zcast-ensemble
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/error/%j.err

set -euo pipefail

echo "======================================================"
echo " Job started on $(hostname) at $(date)"
echo "======================================================"

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
module load jaspy/3.11
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
lead_time=$1
year=$2
month=$3
hour=$4

script=/home/users/mendrika/Zambia-Intercomparison/scripts/zambia-predict-zcast-ensemble.py

# Verify the script exists
if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

echo "Running ensemble nowcast inference:"
echo " Lead time : $lead_time"
echo " Year      : $year"
echo " Month     : $month"
echo " Hour      : $hour"
echo "======================================================"

# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
python "$script" "$lead_time" "$year" "$month" "$hour"

echo "======================================================"
echo " Job completed successfully at $(date)"
echo "======================================================"
