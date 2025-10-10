#!/bin/bash
# Launcher for Zambia data shard creation

JOB_SCRIPT="/home/users/mendrika/Zambia-Intercomparison/slurm/zambia-create-shards.sh"
LOG_DIR="/home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs"

# Make sure log directory exists
mkdir -p "${LOG_DIR}/output" "${LOG_DIR}/error"

# Lead times and partitions to process
LEAD_TIMES=("1" "2")
PARTITIONS=("test" "train" "val")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for PARTITION in "${PARTITIONS[@]}"; do
        echo "Submitting job for partition=${PARTITION}, lead_time=${LEAD_TIME}..."
        sbatch -J "$PARTITION$LEAD_TIME" "$JOB_SCRIPT" "$PARTITION" "$LEAD_TIME"
        sleep 1  # small delay to avoid submission bursts
    done
done

echo "All jobs submitted successfully."
