#!/bin/bash
# Table 6 sub-metric breakdown of the Table 3 replication rows and chosen ReCAST runs (CPU work only;
# the partition needs a GPU request). Submit from the repository root:
#   sbatch replication/scripts/bash/table6/table6_breakdown_gh200.sh
#SBATCH --job-name=ai-engram-table6
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
RESULTS="${STORAGE}/experiment_results/ai-engram"
OUT="${STORAGE}/runtime_artifacts/ai-engram/table6"
mkdir -p "${OUT}"

# ReCAST down + up + o_proj, seed 0 (restricted, s_min 0.3, EVR 0.97, tied head trained, 10 epochs peak 3e-5).
RECAST_DOWN_UP_O="${RESULTS}/recast_tofu/recast_restricted_semu_down_up_o_tied_smin0p3_evr0p97_lrwarm1_peak3e-5_to1e-5at5_e10_seed0_retain_float32_clip1p0_24959652_0/raw_metrics.json"

echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" -u replication/scripts/table6_breakdown.py \
    --anchors_raw "${RESULTS}/table3/table3_forget10_seed0_24952059/raw_metrics.json" \
    --anchor_rows "Engram (alpha=0.6)" "Engram (alpha_W-Norm)" \
    --run "ReCAST down+up+o_proj (seed 0)=${RECAST_DOWN_UP_O}" \
    --out_json "${OUT}/table6_breakdown_${SLURM_JOB_ID:-local}.json" \
    --out_md "${OUT}/table6_breakdown_${SLURM_JOB_ID:-local}.md"
