#!/bin/bash
# Signs of the first-order forget contributions of the ReCAST update components for every decoder projection type
# (plain and D at m 1.25, rho 30, s_min 0.3, EVR 0.97), and the forget / retain loss of Init Fine. and Retain.
# UPDATE_MODELS="<id> ..." in --export also decomposes each model's displacement from Init Fine. (Retain, Engram edits);
# LOSS_MODELS="" skips the loss measurements. Nothing is trained. Submit from the repository root:
#   sbatch replication/scripts/bash/recast_tofu/component_signs_gh200.sh
#SBATCH --job-name=ai-engram-comp-signs
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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/component_signs.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT="${STORAGE}/runtime_artifacts/ai-engram/recast_tofu"
mkdir -p "${OUT}"
echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${PY}" -u replication/scripts/component_signs.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --targets decoder_linear \
    --target_blocks all \
    --attn_implementation sdpa \
    --grad_batch_size 8 \
    --s_min 0.3 \
    --restriction_evr 0.97 \
    --rho 30 \
    --rank_multiplier 1.25 \
    --loss_models ${LOSS_MODELS-open-unlearning/tofu_Llama-3.2-1B-Instruct_full open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90} \
    --update_models ${UPDATE_MODELS:-} \
    --out_json "${OUT}/component_signs_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
