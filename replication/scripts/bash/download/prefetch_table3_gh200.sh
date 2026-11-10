#!/bin/bash
# Download the models and TOFU configs of the Table 3 replication into HF_HOME,
# record their revisions, then load all of them again offline.
# Submit from the repository root, after the environment exists:
#   sbatch replication/scripts/bash/download/prefetch_table3_gh200.sh
#SBATCH --job-name=ai-engram-prefetch
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/prefetch_hf.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
ARTIFACTS="${STORAGE}/runtime_artifacts/ai-engram"

export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${ARTIFACTS}"

# Init Fine., Retain, and the forget-fluency classifier of engram.benchmarks._tofu_official.
MODEL_IDS=(
    open-unlearning/tofu_Llama-3.2-1B-Instruct_full
    open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
    madhurjindal/autonlp-Gibberish-Detector-492513457
)
ALLOW_PATTERNS=("*.json" "*.safetensors" "*.txt")
IGNORE_PATTERNS=("onnx/*")
# The configs engram.benchmarks.tofu.load_splits and tests/test_tofu_evaluate.py load for forget10.
DATASET_ID=locuslab/TOFU
DATASET_CONFIGS=(forget10_perturbed holdout10 retain_perturbed real_authors_perturbed world_facts_perturbed full)

echo "[prefetch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[prefetch] env=${ENV_PREFIX} HF_HOME=${HF_HOME}"
df -h "${HF_HOME}" | tail -1

cd "${REPO_ROOT}"
COMMON=(--model_ids "${MODEL_IDS[@]}"
        --allow_patterns "${ALLOW_PATTERNS[@]}"
        --ignore_patterns "${IGNORE_PATTERNS[@]}"
        --dataset_id "${DATASET_ID}"
        --dataset_configs "${DATASET_CONFIGS[@]}")

"${ENV_PREFIX}/bin/python" replication/scripts/prefetch_hf.py "${COMMON[@]}" \
    --max_workers 8 \
    --manifest "${ARTIFACTS}/hf_manifest_table3_${SLURM_JOB_ID:-local}.json"

# Positive control: everything the evaluation needs loads with the network disabled.
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    "${ENV_PREFIX}/bin/python" replication/scripts/prefetch_hf.py "${COMMON[@]}" --check_offline

echo "[prefetch] done"
