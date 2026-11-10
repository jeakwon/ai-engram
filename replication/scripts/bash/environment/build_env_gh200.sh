#!/bin/bash
# Build the ai-engram-gh200 Conda environment on a GH200 (aarch64) node.
# Submit from the repository root:
#   sbatch replication/scripts/bash/environment/build_env_gh200.sh
#SBATCH --job-name=ai-engram-env
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: run this on a GH200/aarch64 node, not $(uname -m)." >&2
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/src/engram/__init__.py" ]] || { echo "ERROR: submit from the repository root (no ${REPO_ROOT}/src/engram)." >&2; exit 1; }

STORAGE=/net/storage/pr3/plgrid/plggdnnp
CONDA="${CONDA:-${STORAGE}/apps/miniforge3-gh200/bin/conda}"
PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
ENV_FILE="${REPO_ROOT}/replication/environment/ai_engram_gh200.yaml"
REQUIREMENTS="${REPO_ROOT}/replication/environment/requirements_gh200.txt"
TORCH_VERSION=2.6.0
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

export CONDA_PKGS_DIRS="${STORAGE}/cache/${USER}/ai-engram-gh200-conda-pkgs"
export PIP_CACHE_DIR="${STORAGE}/pip-cache/${USER}"
export TMPDIR="${STORAGE}/tmp/${USER}/ai-engram-gh200-build-${SLURM_JOB_ID:-local}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${TMPDIR}"

[[ -x "${CONDA}" ]] || { echo "ERROR: conda not found: ${CONDA}" >&2; exit 1; }
# AI_ENGRAM_ENV_REPAIR=1 brings an existing prefix in line with the spec
# (conda env update, then the same idempotent pip installs) instead of failing.
REPAIR="${AI_ENGRAM_ENV_REPAIR:-0}"
if [[ -e "${PREFIX}" && "${REPAIR}" != "1" ]]; then
    echo "ERROR: target prefix already exists, refusing to overwrite: ${PREFIX}" >&2
    echo "       set AI_ENGRAM_ENV_REPAIR=1 to update it to the spec instead." >&2
    exit 1
fi

echo "[build] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[build] prefix=${PREFIX}"
echo "[build] torch=${TORCH_VERSION} index=${TORCH_INDEX_URL}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "[build] nvidia-smi unavailable"

if [[ -e "${PREFIX}" ]]; then
    echo "[build] repair mode: updating existing prefix to ${ENV_FILE}"
    "${CONDA}" env update --prefix "${PREFIX}" --file "${ENV_FILE}"
else
    "${CONDA}" env create --yes --prefix "${PREFIX}" --file "${ENV_FILE}"
fi

PY="${PREFIX}/bin/python"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# torch first, from the CUDA index, so no other package pulls a PyPI build.
# Diagnostics: job 24947042 found no torch version at all on this index.
echo "[build] $("${PY}" -m pip --version)"
"${PY}" -m pip index versions torch --index-url "${TORCH_INDEX_URL}" || echo "[build] WARNING: pip index versions failed"
"${PY}" -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
echo "torch==${TORCH_VERSION}" > "${TMPDIR}/constraints.txt"
"${PY}" -m pip install -r "${REQUIREMENTS}" -c "${TMPDIR}/constraints.txt"
# The package itself: editable from this checkout, dependencies already pinned above.
"${PY}" -m pip install --no-deps -e "${REPO_ROOT}"

"${PY}" -m pip check || echo "[build] WARNING: pip check reported conflicts (see above)"
"${PY}" -m pip freeze > "${PREFIX}/ai-engram-gh200.freeze.txt"
echo "[build] pip freeze written to ${PREFIX}/ai-engram-gh200.freeze.txt"

echo "[smoke] CUDA"
"${PY}" - <<'PY'
import platform
import torch

assert platform.machine() == "aarch64", platform.machine()
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print(f"[smoke] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device={torch.cuda.get_device_name()} bf16_matmul_sum={y:.3e}")
PY

echo "[smoke] ai-engram TOFU import path"
(
    cd "${TMPDIR}"  # import the installed package, not a path-relative copy
    "${PY}" - <<'PY'
import importlib

import engram
from engram.benchmarks import tofu  # noqa: F401
from engram.benchmarks import _tofu_official  # noqa: F401  -- rouge_score, sklearn at import time

print(f"[smoke] engram=={engram.__version__} from {engram.__file__}")
for name in ["transformers", "datasets", "huggingface_hub", "scipy", "sklearn",
             "numpy", "rouge_score", "wandb", "pytest", "accelerate"]:
    module = importlib.import_module(name)
    print(f"[smoke] {name}=={getattr(module, '__version__', '?')}")

# The ReCAST phase two builds a Trainer; TrainingArguments alone checks for accelerate.
import tempfile
from transformers import TrainingArguments
TrainingArguments(output_dir=tempfile.mkdtemp(), report_to=[])
print("[smoke] transformers.TrainingArguments constructed")
PY
)

# The upstream CPU test suite, as its CI runs it; the TOFU tests are gated off and skip.
# Offline: these tests build models from configs, so any download attempt is a failure.
echo "[smoke] upstream tests (CPU, TOFU gated off, offline)"
(
    cd "${REPO_ROOT}"
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${PY}" -m pytest -q -rs -p no:cacheprovider tests/
)

rm -rf "${TMPDIR}"
echo "[build] done"
