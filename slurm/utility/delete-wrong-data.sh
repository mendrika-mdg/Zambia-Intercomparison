#!/bin/bash
#SBATCH --job-name=delete
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/error/%j.err

set -euo pipefail
path="$1"

echo "==========================================="
echo " Cleaning contents of: $path"
echo "==========================================="

# Safety check
if [[ ! -d "$path" ]]; then
    echo "Directory not found: $path"
    exit 1
fi

# Delete contents (not folder)
echo "Deleting all files and subfolders in $path ..."
find "$path" -mindepth 1 -delete

echo "Finished cleaning $path"
