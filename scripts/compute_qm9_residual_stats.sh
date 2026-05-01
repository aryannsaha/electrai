#!/bin/bash
#SBATCH --job-name=qm9_residual_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/electrai/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/electrai/logs/%x-%j.err

set -eo pipefail

cd /scratch/gpfs/ROSENGROUP/aryan/electrai
mkdir -p logs

module purge
module load anaconda3/2025.6
conda activate electrai

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

EXTRA_ARGS=()
if [[ "${QM9_RESIDUAL_INCLUDE_SAMPLE_MEDIAN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--include-sample-median)
fi
if [[ "${QM9_RESIDUAL_SKIP_HISTOGRAM:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-histogram)
fi

python -u scripts/compute_qm9_residual_stats.py \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --progress-every "${QM9_RESIDUAL_PROGRESS_EVERY:-500}" \
  --histogram-bins "${QM9_RESIDUAL_HISTOGRAM_BINS:-4096}" \
  --output-dir "${QM9_RESIDUAL_OUTPUT_DIR:-scripts/qm9_residual_stats}" \
  "${EXTRA_ARGS[@]}"
