#!/bin/bash
# ---------------------------------------------------------------------
# Launcher for Zambia Zcast Ensemble Prediction (DJF 2020–2024)
# ---------------------------------------------------------------------

JOB_SCRIPT="/home/users/mendrika/Zambia-Intercomparison/slurm/zambia-predict-zcast-ensemble.sh"
LOG_DIR="/home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs"

# Ensure log directories exist
mkdir -p "${LOG_DIR}/output" "${LOG_DIR}/error"

# Configuration
YEARS=("2020" "2021" "2022" "2023" "2024")
MONTHS=("12" "01" "02")
HOURS=($(seq -w 0 23))
LEAD_TIMES=("1" "2" "4" "6")

# ---------------------------------------------------------------------
# Submission loop
# ---------------------------------------------------------------------
for YEAR in "${YEARS[@]}"; do
  for MONTH in "${MONTHS[@]}"; do
    for HOUR in "${HOURS[@]}"; do
      for LEAD_TIME in "${LEAD_TIMES[@]}"; do
        JOB_NAME="zcast${LEAD_TIME}_${YEAR}${MONTH}_${HOUR}"
        echo "Submitting job: ${JOB_NAME}"
        sbatch -J "${JOB_NAME}" \
               -o "${LOG_DIR}/output/${JOB_NAME}-%j.out" \
               -e "${LOG_DIR}/error/${JOB_NAME}-%j.err" \
               "$JOB_SCRIPT" "$LEAD_TIME" "$YEAR" "$MONTH" "$HOUR"
        sleep 0.5  # small delay to avoid submission bursts
      done
    done
  done
done

echo "All DJF ensemble nowcast jobs submitted successfully."
