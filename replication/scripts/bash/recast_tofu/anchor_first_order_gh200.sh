#!/bin/bash
# ||Delta|| and the first-order loss changes <G, Delta> (Init Fine. mean gradients) of other models as updates of
# Init Fine., next to their actual forget / retain loss change, over the 48 down_up_o matrices and all parameters.
# Submit from the repository root with the models to compare:
#   sbatch --export=ALL,MODEL_IDS="<id or directory> ..." replication/scripts/bash/recast_tofu/anchor_first_order_gh200.sh
#SBATCH --job-name=ai-engram-anchor-first-order
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

: "${MODEL_IDS:?set MODEL_IDS with sbatch --export}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/anchor_first_order.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT="${STORAGE}/runtime_artifacts/ai-engram/recast_tofu"
echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
# shellcheck disable=SC2086  # MODEL_IDS is a space-separated list
"${PY}" -u replication/scripts/anchor_first_order.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --model_ids ${MODEL_IDS} \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --targets down_up_o \
    --attn_implementation sdpa \
    --grad_batch_size 8 \
    --out_json "${OUT}/anchor_first_order_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
