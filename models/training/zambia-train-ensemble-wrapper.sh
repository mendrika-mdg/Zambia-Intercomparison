#!/bin/bash
# ---------------------------------------------------------------------
# Wrapper to submit ensemble training jobs for all lead times and seeds
# ---------------------------------------------------------------------

JOB_SCRIPT="/home/users/mendrika/Zambia-Intercomparison/models/training/zambia-train-ensemble.sh"
LOG_DIR="/home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs"

# Lead times and seeds
LEAD_TIMES=("4" "6")
SEEDS=(42 1337 777 999 2025)

# Loop over all combinations
for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Submitting job for lead_time=${LEAD_TIME}, seed=${SEED}..."
        sbatch -J "ens_t${LEAD_TIME}_s${SEED}" "$JOB_SCRIPT" "$LEAD_TIME" "$SEED"
        sleep 2  # brief pause to avoid queue flooding
    done
done

echo "All ensemble jobs submitted successfully."
