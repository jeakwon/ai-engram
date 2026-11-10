#!/bin/bash
# Forget / retain energy of the plain, C and D restriction bases and the warm-start output change on
# each set (no training). Settings of the m 1.25 runs (25016368 / 25016369); TARGETS (default down_up_o)
# picks the profile. Submit from the repository root:
#   sbatch --export=ALL,TARGETS=decoder_linear replication/scripts/bash/recast_tofu/basis_energy_gh200.sh
#SBATCH --job-name=ai-engram-basis-energy
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/basis_energy_analysis.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT="${STORAGE}/runtime_artifacts/ai-engram/recast_tofu"
TARGETS="${TARGETS:-down_up_o}"  # target profile; decoder_linear = every decoder-block projection, head = lm_head
UNTIE="${UNTIE:-no}"  # yes: the lm_head gets its own copy (no embedding edit)
mkdir -p "${OUT}"

echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" -u replication/scripts/basis_energy_analysis.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --targets "${TARGETS}" \
    --target_blocks all \
    --attn_implementation sdpa \
    --grad_batch_size 8 \
    --s_min 0.3 \
    --restriction_evr 0.97 \
    --rho 30 \
    --rank_multiplier 1.25 \
    --untie_lm_head "${UNTIE}" \
    --out_json "${OUT}/basis_energy_${TARGETS}_untie${UNTIE}_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
