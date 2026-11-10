#!/bin/bash
# Model Utility sub-metrics of the Table 3 anchors (Init Fine., Retain), evaluation only, with the
# settings of the Table 3 run 24952059. Submit from the repository root:
#   sbatch replication/scripts/bash/table6/mu_submetrics_anchors_gh200.sh
#SBATCH --job-name=ai-engram-mu-anchors
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
[[ -f "${REPO_ROOT}/replication/scripts/mu_submetrics_anchors.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
RESULTS="${STORAGE}/experiment_results/ai-engram/table3"

echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[launch] nvidia-smi unavailable"
cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" -u replication/scripts/mu_submetrics_anchors.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --retain_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
    --dataset_id locuslab/TOFU \
    --model_dtype bfloat16 \
    --attn_implementation sdpa \
    --eval_batch_size 16 \
    --anchors_raw "${RESULTS}/table3_forget10_seed0_24952059/raw_metrics.json" \
    --out_json "${RESULTS}/table3_anchor_mu_submetrics_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
