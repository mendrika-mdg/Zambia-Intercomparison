#!/bin/bash
# Combine nowcast into a single file

JOB_SCRIPT="/home/users/mendrika/Zambia-Intercomparison/slurm/zambia-create-2d-timeseries.sh"
LOG_DIR="/home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs"

# Lead times to process
LEAD_TIMES=("1" "2" "4" "6")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    echo "Submitting job for lead_time=${LEAD_TIME}..."
    sbatch -J "combined${LEAD_TIME}" "$JOB_SCRIPT" "$LEAD_TIME"
    sleep 1  # small delay to avoid submission bursts
done

echo "All jobs submitted successfully."
