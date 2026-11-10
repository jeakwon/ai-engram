#!/bin/bash
# Screen published unlearning checkpoints on Memorization, Privacy, EM and FQ (no generation), one share of the model
# list per array task; the full evaluation follows for a few of them (user, 2026-10-29). Needs network access for the
# checkpoints, which are deleted from the cache after scoring. Submit from the repository root:
#   sbatch --array=0-<n-1> --export=ALL,MODELS=<file>,OUT_DIR=<dir> replication/scripts/bash/recast_tofu/checkpoint_screen_gh200.sh
#SBATCH --job-name=ai-engram-checkpoint-screen
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=replication/logs/slurm/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/tofu_checkpoint_screen.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
: "${MODELS:?set MODELS=<file with one model id per line>}" "${OUT_DIR:?set OUT_DIR=<output directory>}"
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
ANCHORS_RAW="${STORAGE}/experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/raw_metrics.json"
echo "[launch] host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${SLURM_ARRAY_TASK_ID:-0} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) models=${MODELS}"
cd "${REPO_ROOT}"
"${PY}" -u replication/scripts/tofu_checkpoint_screen.py \
    --models "${MODELS}" \
    --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --tasks "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --anchors_raw "${ANCHORS_RAW}" \
    --out_dir "${OUT_DIR}"
echo "[launch] done"
