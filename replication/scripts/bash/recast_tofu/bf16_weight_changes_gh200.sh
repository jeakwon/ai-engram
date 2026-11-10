#!/bin/bash
# Bitwise comparison of Init Fine. and Retain per binade of |w|: which weights a training with bf16 weights at lr 1e-5
# could move. Submit from the repository root:
#   sbatch replication/scripts/bash/recast_tofu/bf16_weight_changes_gh200.sh
#SBATCH --job-name=ai-engram-bf16-weight-changes
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/bf16_weight_changes.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1
OUT="${STORAGE}/runtime_artifacts/ai-engram/recast_tofu"
echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${PY}" -u replication/scripts/bf16_weight_changes.py \
    --model_a open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --model_b open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
    --lr 1e-5 \
    --out_json "${OUT}/bf16_weight_changes_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
