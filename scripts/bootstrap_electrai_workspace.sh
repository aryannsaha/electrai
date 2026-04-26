#!/usr/bin/env bash
set -euo pipefail

# Rebuild an ElectrAI workspace from scratch on RunPod, Slurm login nodes,
# or similar Linux machines.

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/aryannsaha/electrai.git}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/electrai}"
BRANCH="${BRANCH:-main}"
MINIFORGE_DIR="${MINIFORGE_DIR:-${WORKSPACE}/miniforge3}"
ENV_NAME="${ENV_NAME:-electrai-runpod}"
RUN_TESTS="${RUN_TESTS:-0}"
FORCE_RECLONE="${FORCE_RECLONE:-0}"

MINIFORGE_INSTALLER="${MINIFORGE_INSTALLER:-/tmp/Miniforge3-Linux-x86_64.sh}"
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"

log() {
  printf '\n[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

download() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Neither curl nor wget is installed; cannot download $url" >&2
    exit 1
  fi
}

ensure_conda_available() {
  if command -v conda >/dev/null 2>&1; then
    log "Using existing conda: $(command -v conda)"
    return
  fi

  if [ -x "${MINIFORGE_DIR}/bin/conda" ]; then
    log "Using existing Miniforge at ${MINIFORGE_DIR}"
    # shellcheck disable=SC1091
    source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
    return
  fi

  log "Installing Miniforge to ${MINIFORGE_DIR}"
  mkdir -p "$(dirname "$MINIFORGE_DIR")"
  download "$MINIFORGE_URL" "$MINIFORGE_INSTALLER"
  bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_DIR"
  # shellcheck disable=SC1091
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  conda config --set auto_activate_base false
}

clone_or_update_repo() {
  need_cmd git
  mkdir -p "$WORKSPACE"

  if [ "$FORCE_RECLONE" = "1" ] && [ -d "$REPO_DIR" ]; then
    log "Removing existing repo because FORCE_RECLONE=1: ${REPO_DIR}"
    rm -rf "$REPO_DIR"
  fi

  if [ -d "${REPO_DIR}/.git" ]; then
    log "Updating existing repo at ${REPO_DIR}"
    git -C "$REPO_DIR" fetch --all --prune
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only
  else
    log "Cloning ${REPO_URL} into ${REPO_DIR}"
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  fi
}

create_or_update_env() {
  # shellcheck disable=SC1091
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  eval "$(conda shell.bash hook)"

  local env_file="${REPO_DIR}/environment-runpod.yml"
  if [ ! -f "$env_file" ]; then
    log "No environment-runpod.yml found; creating ${ENV_NAME} with baseline dependencies"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      conda install -n "$ENV_NAME" -y python=3.11 pip uv git awscli
    else
      conda create -n "$ENV_NAME" -y -c conda-forge python=3.11 pip uv git awscli
    fi
  elif conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Updating Conda env ${ENV_NAME} from ${env_file}"
    conda env update -n "$ENV_NAME" -f "$env_file" --prune
  else
    log "Creating Conda env ${ENV_NAME} from ${env_file}"
    conda env create -n "$ENV_NAME" -f "$env_file"
  fi

  conda activate "$ENV_NAME"
  log "Installing ElectrAI package and development extras"
  cd "$REPO_DIR"
  python -m uv pip install --system -e ".[dev,zarr_conversion]"
}

verify_install() {
  # shellcheck disable=SC1091
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  cd "$REPO_DIR"

  log "Python and CUDA check"
  python - <<'PY'
import sys
import torch

print(f"python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__}")
print(f"cuda runtime: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

  if [ "$RUN_TESTS" = "1" ]; then
    log "Running pytest"
    python -m pytest
  else
    log "Skipping tests. Set RUN_TESTS=1 to run pytest."
  fi
}

main() {
  log "Bootstrapping ElectrAI workspace"
  clone_or_update_repo
  ensure_conda_available
  create_or_update_env
  verify_install

  cat <<EOF

Done.

Activate the environment:
  source ${MINIFORGE_DIR}/etc/profile.d/conda.sh
  conda activate ${ENV_NAME}
  cd ${REPO_DIR}

Useful commands:
  python -m pytest
  electrai train --config src/electrai/configs/MP/config_resnet.yaml

EOF
}

main "$@"
