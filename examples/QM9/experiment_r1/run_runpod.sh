#!/usr/bin/env bash
set -euo pipefail

cd /workspace/electrai

if [ -f /workspace/miniforge3/etc/profile.d/conda.sh ]; then
    source /workspace/miniforge3/etc/profile.d/conda.sh
fi
conda activate electrai-runpod

export PYTHONPATH="/workspace/electrai/src:/workspace/electrai:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=300
export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

mkdir -p /workspace/electrai/examples/QM9/experiment_r1/checkpoints_runpod_8k_residual
mkdir -p /workspace/electrai/examples/QM9/experiment_r1/logs

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    examples/QM9/experiment_r1/train_runpod.py \
    --config examples/QM9/experiment_r1/config_runpod_8k.yml \
    2>&1 | tee -a /workspace/electrai/examples/QM9/experiment_r1/logs/runpod_8k_residual.log
