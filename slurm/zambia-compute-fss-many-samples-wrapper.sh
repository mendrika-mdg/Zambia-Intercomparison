#!/bin/bash
# Submit FSS computation jobs for all lead times and hours

JOB_SCRIPT="/home/users/mendrika/Zambia-Intercomparison/slurm/zambia-compute-fss-many-samples.sh"
LOG_DIR="/home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR/output" "$LOG_DIR/error"

# Lead times to process
LEAD_TIMES=("1" "2" "4" "6")

# Target hours (00–23)
TARGET_HOURS=($(seq -w 0 23))

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for HOUR in "${TARGET_HOURS[@]}"; do
        echo "Submitting job for lead_time=${LEAD_TIME}, hour=${HOUR}..."
        sbatch -J "fss_t${LEAD_TIME}_h${HOUR}" "$JOB_SCRIPT" "$LEAD_TIME" "$HOUR"
        sleep 1  # small delay to avoid submission bursts
    done
done

echo "All jobs submitted successfully."
