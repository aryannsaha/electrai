#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/workspace/miniforge3}"
INSTALLER="${INSTALLER:-/tmp/Miniforge3-Linux-x86_64.sh}"
URL="${URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"

if command -v conda >/dev/null 2>&1; then
  echo "conda is already available at: $(command -v conda)"
  exit 0
fi

if [ -d "${INSTALL_DIR}" ]; then
  echo "${INSTALL_DIR} already exists."
  echo "Activate it with: source ${INSTALL_DIR}/etc/profile.d/conda.sh"
  exit 0
fi

if command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${INSTALLER}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${INSTALLER}" "${URL}"
else
  echo "Neither curl nor wget is installed, so I cannot download Miniforge."
  exit 1
fi

bash "${INSTALLER}" -b -p "${INSTALL_DIR}"

source "${INSTALL_DIR}/etc/profile.d/conda.sh"
conda config --set auto_activate_base false

echo
echo "Miniforge installed at ${INSTALL_DIR}"
echo "For this shell, run: source ${INSTALL_DIR}/etc/profile.d/conda.sh"
echo "Then run: bash scripts/setup_runpod_conda.sh"
