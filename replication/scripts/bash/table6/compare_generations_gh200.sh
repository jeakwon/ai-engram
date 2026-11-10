#!/bin/bash
# Generations of ReCAST runs next to Init Fine. and Retain on the Model Utility splits (CPU work only;
# the partition needs a GPU request). Submit from the repository root:
#   sbatch replication/scripts/bash/table6/compare_generations_gh200.sh
#SBATCH --job-name=ai-engram-generations
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
RESULTS="${STORAGE}/experiment_results/ai-engram"
OUT="${STORAGE}/runtime_artifacts/ai-engram/generations"
mkdir -p "${OUT}"
RUN_DIR="${RESULTS}/recast_tofu/recast_restricted_shrink_%s_blocksall_tied_smin0p3_evr0p97_reference_lrwarm1_peak3e-5_to1e-5at5_e10_seed0_retain_float32_clip1p0_24989313_%s"

echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" -u replication/scripts/compare_generations.py \
    --anchors_json "${RESULTS}/table3/table3_anchor_mu_submetrics_24989315.json" \
    --run "restricted_shrink down_up_o=$(printf "${RUN_DIR}" down_up_o 0)/generations.json" \
    --run "restricted_shrink mlp_o=$(printf "${RUN_DIR}" mlp_o 3)/generations.json" \
    --splits real_authors world_facts retain \
    --examples 6 \
    --out_md "${OUT}/generations_${SLURM_JOB_ID:-local}.md"
