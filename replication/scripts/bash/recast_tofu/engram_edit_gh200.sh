#!/bin/bash
# The Engram edits of Table 3 (alpha = 0.6 and alpha_W-Norm, eigh) saved as checkpoints for phase-two runs, each
# reloaded and scored against the Table 3 anchors. Same statistics and arguments as table3_tofu_gh200.sh.
# Submit from the repository root:
#   sbatch replication/scripts/bash/recast_tofu/engram_edit_gh200.sh
#SBATCH --job-name=ai-engram-engram-edit
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/engram_edit.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT="${STORAGE}/runtime_artifacts/ai-engram"
mkdir -p "${OUT}/engram_models"
echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${PY}" -u replication/scripts/engram_edit.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --holdout_split holdout10 \
    --reference_config full \
    --n_total 4000 \
    --total_seed 0 \
    --model_dtype bfloat16 \
    --attn_implementation sdpa \
    --cov_batch_size 8 \
    --target_modules all-linear \
    --plain_alpha 0.6 \
    --wnorm_alpha 1.0 \
    --wnorm_power 1 \
    --eval_batch_size 16 \
    --table3_raw "${STORAGE}/experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/raw_metrics.json" \
    --out_root "${OUT}/engram_models" \
    --out_json "${OUT}/recast_tofu/engram_edit_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
