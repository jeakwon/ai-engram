#!/bin/bash
# Signed-shrink report (component_report.py) over the runs of the given recast array jobs. Submit from the
# repository root:
#   sbatch --export=ALL,JOBS="<array job> <array job> ...",TAG=<name> replication/scripts/bash/recast_tofu/component_report_gh200.sh
#SBATCH --job-name=ai-engram-comp-report
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

: "${JOBS:?set JOBS to the recast array job ids with sbatch --export}"
: "${TAG:?set TAG (report name) with sbatch --export}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/component_report.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
PY="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}/bin/python"
OUT="${STORAGE}/runtime_artifacts/ai-engram/recast_tofu"
RESULTS=()
for j in ${JOBS}; do
    for f in "${STORAGE}"/experiment_results/ai-engram/recast_tofu/*_"${j}"_*/results.json; do
        [[ -f "${f}" ]] && RESULTS+=("${f}")
    done
done
echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} runs=${#RESULTS[@]} jobs=${JOBS}"
cd "${REPO_ROOT}"
"${PY}" -u replication/scripts/component_report.py --results "${RESULTS[@]}" \
    --out_md "${OUT}/component_report_${TAG}.md" --out_json "${OUT}/component_report_${TAG}.json"
echo "[launch] done"
