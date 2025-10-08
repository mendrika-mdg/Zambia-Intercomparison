#!/bin/bash
# Launcher for Zambia data preparation jobs across multiple years

JOB_SCRIPT="/home/users/mendrika/Zambia-Intercomparison/slurm/zambia-data-preparation.sh"
LOG_DIR="/home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs"

# Loop through years 2004–2024 inclusive
for year in $(seq 2004 2022); do
    echo "Submitting job for year ${year}..."
    sbatch -J ${year} "${JOB_SCRIPT}" "${year}"
    sleep 1  # small delay to avoid overwhelming the scheduler
done

echo "All jobs submitted successfully."
