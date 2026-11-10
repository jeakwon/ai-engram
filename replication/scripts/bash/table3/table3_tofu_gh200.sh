#!/bin/bash
# Table 3 of arXiv 2606.14997 on TOFU forget10: Init Fine., Retain, Engram (alpha = 0.6) and
# Engram (alpha_W-Norm) with the package's default solver (eigh), plus the float32 SVD path
# of the authors' notebook as a sensitivity check on the same statistics.
# Submit from the repository root:
#   sbatch replication/scripts/bash/table3/table3_tofu_gh200.sh
#SBATCH --job-name=ai-engram-table3
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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/table3_tofu.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # everything was prefetched by job 24947365
export WANDB_PROJECT=ai-engram-replication        # overrides the one ~/.bash_env may set
export WANDB_DIR="${STORAGE}/archive/${USER}/ai-engram_logs"
export TOKENIZERS_PARALLELISM=false

SEED=0
RUN_NAME="table3_forget10_seed${SEED}_${SLURM_JOB_ID:-local}"
OUT_DIR="${STORAGE}/experiment_results/ai-engram/table3/${RUN_NAME}"

echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[launch] env=${ENV_PREFIX} HF_HOME=${HF_HOME} WANDB_ENTITY=${WANDB_ENTITY:-unset} WANDB_PROJECT=${WANDB_PROJECT}"
echo "[launch] out_dir=${OUT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "[launch] nvidia-smi unavailable"

cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" -u replication/scripts/table3_tofu.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --retain_id open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --holdout_split holdout10 \
    --reference_config full \
    --n_total 4000 \
    --total_seed 0 \
    --model_dtype bfloat16 \
    --attn_implementation sdpa \
    --cov_batch_size 8 \
    --target_modules all-linear \
    --inverse_methods eigh svd \
    --plain_alpha 0.6 \
    --wnorm_alpha 1.0 \
    --wnorm_power 1 \
    --eval_batch_size 16 \
    --seed "${SEED}" \
    --out_dir "${OUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --wandb_mode online \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_tags engram_tofu_table3_faithful method_faithful \
        model_tofu_llama3p2_1b_instruct split_forget10 \
        init_fine retain alpha_uniform alpha_wnorm \
        alpha_0p6 wnorm_alpha_1p0_p1 layers_all_linear reference_full_4000 \
        inverse_eigh_main inverse_svd_check seed_0

echo "[launch] done"
