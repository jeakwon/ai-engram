#!/bin/bash
# Experiment 2 (A-C): response of the losses and the TOFU scores to the strength t of the fixed D update
# (DIRECTION=full) or of its signed version (DIRECTION=signed), D rho 30 gradient side, down_up_o, s_min 0.3,
# EVR 0.97, m 1.25; nothing is trained. Submit from the repository root:
#   sbatch --export=ALL,DIRECTION=full replication/scripts/bash/recast_tofu/contraction_response_gh200.sh
#SBATCH --job-name=ai-engram-contraction
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

: "${DIRECTION:?set DIRECTION=full|signed with sbatch --export}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/contraction_response.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT="${STORAGE}/runtime_artifacts/ai-engram/recast_tofu"
mkdir -p "${OUT}"
# Forget loss of the warm start in the runs (25244062_0 none / 25244062_2 signed), for the t = 1 control.
case "${DIRECTION}" in
    full) REF=11.1023 ;;
    signed) REF=1.6270 ;;
    *) echo "ERROR: DIRECTION must be full or signed" >&2; exit 1 ;;
esac
echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} direction=${DIRECTION} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${PY}" -u replication/scripts/contraction_response.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --holdout_split holdout10 \
    --targets down_up_o \
    --target_blocks all \
    --attn_implementation sdpa \
    --anchors_raw "${STORAGE}/experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/raw_metrics.json" \
    --direction "${DIRECTION}" \
    --grad_batch_size 8 \
    --eval_batch_size 16 \
    --hvp_batch_size 8 \
    --s_min 0.3 \
    --restriction_evr 0.97 \
    --rho 30 \
    --rank_multiplier 1.25 \
    --t_grid 0 0.01 0.025 0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0 \
    --fd_steps 0.001 0.003 0.01 0.03 \
    --eval_tofu yes \
    --reference_t1_forget_loss "${REF}" \
    --out_json "${OUT}/contraction_response_${DIRECTION}_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
