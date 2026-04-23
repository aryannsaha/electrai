#!/bin/bash
#SBATCH --job-name=qm9_timestep_errors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/electrai/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/electrai/logs/%x-%j.err

cd /scratch/gpfs/ROSENGROUP/aryan/electrai
mkdir -p logs

module purge
module load anaconda3/2025.6
conda activate electrai
module load proxy/default

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export QM9_TIMESTEP_PROGRESS_EVERY="${QM9_TIMESTEP_PROGRESS_EVERY:-25}"


python -u scripts/eval_qm9_timestep_errors.py
