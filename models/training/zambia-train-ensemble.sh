#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpuhost006,gpuhost015
#SBATCH -o /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/error/%j.err

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

# Activate virtual environment
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

# ---------------------------------------------------------------------
# Torch DDP configuration
# ---------------------------------------------------------------------
export MASTER_ADDR="localhost"
export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=0

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
lead_time=$1
seed=$2

if [ -z "$lead_time" ] || [ -z "$seed" ]; then
    echo "Usage: sbatch zambia-train-ensemble.sh <lead_time> <seed>"
    exit 1
fi

# ---------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------
echo "Starting distributed training for lead_time=${lead_time}, seed=${seed}"
torchrun --standalone --nproc_per_node=4 /home/users/mendrika/Zambia-Intercomparison/models/training/zambia_ensemble_hybrid.py "$lead_time" "$seed"

echo "Training completed at $(date)"
