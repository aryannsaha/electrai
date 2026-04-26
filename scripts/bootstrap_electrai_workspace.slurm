#!/usr/bin/env bash
#SBATCH --job-name=electrai-bootstrap
#SBATCH --output=/workspace/electrai-bootstrap-%j.out
#SBATCH --error=/workspace/electrai-bootstrap-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Override these at submission time if needed:
#   sbatch --export=ALL,REPO_URL=https://github.com/YOU/electrai.git,BRANCH=my-branch,RUN_TESTS=1 bootstrap_electrai_workspace.slurm

export WORKSPACE="${WORKSPACE:-/workspace}"
export REPO_URL="${REPO_URL:-https://github.com/aryannsaha/electrai.git}"
export BRANCH="${BRANCH:-main}"
export ENV_NAME="${ENV_NAME:-electrai-runpod}"
export RUN_TESTS="${RUN_TESTS:-0}"

bash "${WORKSPACE}/bootstrap_electrai_workspace.sh"
