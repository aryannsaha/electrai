#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-electrai-runpod}"
ENV_FILE="${ENV_FILE:-environment-runpod.yml}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Use a RunPod image with conda/mamba, or install Miniforge first."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

python -m uv pip install --python "${CONDA_PREFIX}/bin/python" -e ".[dev,zarr_conversion]"

python - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"cuda runtime: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

echo
echo "Ready. Activate with: conda activate ${ENV_NAME}"
