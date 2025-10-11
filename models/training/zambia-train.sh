#!/bin/bash
#SBATCH --job-name=pancast-t1                 # Job name
#SBATCH --partition=orchid                    # Partition to run on
#SBATCH --account=orchid                      # Account
#SBATCH --qos=orchid                          # QoS level
#SBATCH --nodes=1                             # Single node
#SBATCH --ntasks-per-node=4                   # torchrun spawns 4 processes (1 per GPU)
#SBATCH --gres=gpu:4                          # 4 GPUs per node
#SBATCH --cpus-per-task=4                     # 4 CPU threads per process
#SBATCH --mem=256G                            # Total memory allocation
#SBATCH --time=24:00:00                       # Wall time
#SBATCH --exclude=gpuhost006,gpuhost015       # Exclude bad GPU nodes
#SBATCH -o /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Zambia-Intercomparison/slurm/submission-logs/error/%j.err

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi                                       # Display GPU info

# Activate virtual environment
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

# ---------------------------------------------------------------------
# Torch DDP configuration
# ---------------------------------------------------------------------
export MASTER_ADDR="localhost"                  # Local master for torchrun
export MASTER_PORT=$((12000 + RANDOM % 20000))  # Random port to avoid conflicts
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK     # Set OpenMP threads per task

# Optional reproducibility setting
export PYTHONHASHSEED=0

# ---------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------
echo "Starting distributed training..."

torchrun --standalone --nproc_per_node=4 /home/users/mendrika/Zambia-Intercomparison/models/training/zambia-train.py

echo "Training completed at $(date)"
