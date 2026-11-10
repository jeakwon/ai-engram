#!/bin/bash
# KL between the Retain model's next-token predictions and other models' on the answer tokens of retain90, forget10,
# holdout10, real authors and world facts. Submit from the repository root with the models to compare:
#   sbatch --export=ALL,MODEL_IDS="<id or directory> ..." replication/scripts/bash/recast_tofu/prediction_kl_gh200.sh
#SBATCH --job-name=ai-engram-prediction-kl
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

: "${MODEL_IDS:?set MODEL_IDS with sbatch --export}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/prediction_kl.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
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
"${PY}" -u replication/scripts/prediction_kl.py \
    --reference_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
    --tokenizer_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --model_ids ${MODEL_IDS} \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --holdout_split holdout10 \
    --attn_implementation sdpa \
    --batch_size 8 \
    --out_json "${OUT}/prediction_kl_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
