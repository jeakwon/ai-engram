#!/bin/bash
# Selective input contrast with retain protection (recast_selective.py; extension, user 2026-11-03), driver
# recast_selective_tofu.py. MODE=diagnose: no training; checks of the massive-activation hypothesis, the selection grid
# (families x protection x lambda*) and every target's edit alone, all measured by inference (about 2-3 h).
# MODE=edit: one setting with the loss guard; saves the edited model (float32) for phase two, which runs from
# recast_tofu_gh200.sh with BASE_ID=<the printed MODEL_DIR>, BASE_TAG=<the printed TAG>, VARIANTS=(unrestricted) and
# S_MINS=(1.0) (phase one the identity, phase two the unchanged recipe on every parameter).
# Edit settings in --export: FAMILY=act|fisher|grad, LAMBDA_STAR, FLOOR (s in W - (1 - s) (W V) V^T), KAPPA (unset = no
# kappa protection), PREFIX=yes|no (default yes), RETAIN_TOKENS=nonprefix|answer (default nonprefix), GUARD_RATIO
# (default 2, 0 = no guard). FAMILY=plain (user 2026-11-04): the contract's plain restricted warm start (two-sided, SVD
# of the mean forget gradient at EVR, rows at max(cos, FLOOR)) with the forget gradient projected off the protected span;
# needs EVR, takes no LAMBDA_STAR and no guard (RETAIN_TOKENS plays no part). FAMILY=gcontrast and FAMILY=gmoment
# (user 2026-11-04): the same with the gradient-contrast basis at the plain basis's rank (RHO, default 1; SIDES=both|input,
# default both) or the per-example forget-gradient moment basis at its own EVR ranks, both gradients (gcontrast) or
# every per-example gradient (gmoment) projected off the protected span. FAMILY=grad (user 2026-11-04): the selective
# rule with the input-side Gram of the mean gradients, G_f^T G_f against G_r^T G_r. RETAIN_STAT=act (FAMILY=fisher or
# grad): the pencil's retain side is the act retain covariance over RETAIN_TOKENS instead of the family's own; tag
# _ract. LOO=yes measures every target's contribution within the joint edit and its edit alone (tag _looreport);
# LOO_GUARD_RATIO=<r> guards on the contribution (tag _loo<r>). REPORT_FLOORS=0,0.1,... measures the joint edit at
# further floors (no tag: the saved model is the same). RETAIN_LOSS_TOKENS=nonprefix (FAMILY=fisher or grad, own
# retain statistic): the retain loss behind the gradient statistics on question and answer tokens (tag _rlqa).
# RHO (FAMILY=act, fisher or grad; default 1, user 2026-11-05): the ridge rho / n of the selective pencil (tag _r<rho>
# when not 1). The CPU tests of recast_selective.py run first.
# Submit from the repository root: sbatch --export=ALL,MODE=diagnose replication/scripts/bash/recast_tofu/recast_selective_gh200.sh
#SBATCH --job-name=ai-engram-selective
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

: "${MODE:?set MODE=diagnose|edit with sbatch --export}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/recast_selective_tofu.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT_ROOT="${STORAGE}/experiment_results/ai-engram/recast_selective"
mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"

"${PY}" -m pytest -q -p no:cacheprovider replication/tests/test_recast_selective.py

case "${MODE}" in
    diagnose)
        "${PY}" replication/scripts/recast_selective_tofu.py --mode diagnose \
            --out_json "${OUT_ROOT}/diagnose_${SLURM_JOB_ID}.json" ;;
    edit)
        : "${FAMILY:?set FAMILY=act|fisher|grad|plain|gcontrast|gmoment}" "${FLOOR:?set FLOOR}"
        PREFIX="${PREFIX:-yes}"; RETAIN_TOKENS="${RETAIN_TOKENS:-nonprefix}"
        if [[ "${FAMILY}" == plain || "${FAMILY}" == gcontrast || "${FAMILY}" == gmoment ]]; then
            : "${EVR:?set EVR for FAMILY=${FAMILY}}"
            GUARD_RATIO=0
            TAG="selective_${FAMILY}_prefix${PREFIX}_k${KAPPA:-none}_e${EVR}_f${FLOOR}"
            SELECT_ARGS=(--evr "${EVR}")
            if [[ "${FAMILY}" == gcontrast ]]; then
                SIDES="${SIDES:-both}"; RHO="${RHO:-1}"
                TAG="selective_${FAMILY}_${SIDES}_r${RHO}_prefix${PREFIX}_k${KAPPA:-none}_e${EVR}_f${FLOOR}"
                SELECT_ARGS+=(--sides "${SIDES}" --rho "${RHO}")
            fi
        else
            : "${LAMBDA_STAR:?set LAMBDA_STAR}"
            GUARD_RATIO="${GUARD_RATIO:-2}"; RETAIN_STAT="${RETAIN_STAT:-own}"; LOO="${LOO:-no}"
            RS_TAG=""; [[ "${RETAIN_STAT}" == act ]] && RS_TAG="_ract"
            RETAIN_LOSS_TOKENS="${RETAIN_LOSS_TOKENS:-answer}"
            [[ "${RETAIN_LOSS_TOKENS}" == nonprefix ]] && RS_TAG="${RS_TAG}_rlqa"
            RHO="${RHO:-1}"; [[ "${RHO}" != 1 ]] && RS_TAG="${RS_TAG}_r${RHO}"
            LOO_TAG=""; [[ "${LOO}" == yes ]] && LOO_TAG="_looreport"
            [[ -n "${LOO_GUARD_RATIO:-}" ]] && LOO_TAG="_loo${LOO_GUARD_RATIO}"
            TAG="selective_${FAMILY}${RS_TAG}_${RETAIN_TOKENS}_prefix${PREFIX}_k${KAPPA:-none}_l${LAMBDA_STAR}_f${FLOOR}_g${GUARD_RATIO}${LOO_TAG}"
            SELECT_ARGS=(--lambda_star "${LAMBDA_STAR}" --retain_stat "${RETAIN_STAT}" --loo "${LOO}"
                         --retain_loss_tokens "${RETAIN_LOSS_TOKENS}" --rho "${RHO}"
                         ${LOO_GUARD_RATIO:+--loo_guard_ratio "${LOO_GUARD_RATIO}"})
        fi
        TAG="${TAG//./p}"
        MODEL_DIR="${STORAGE}/runtime_artifacts/ai-engram/recast_selective/models/${TAG}_${SLURM_JOB_ID}"
        echo "[launch] TAG=${TAG} MODEL_DIR=${MODEL_DIR}"
        "${PY}" replication/scripts/recast_selective_tofu.py --mode edit --family "${FAMILY}" \
            --retain_tokens "${RETAIN_TOKENS}" --prefix "${PREFIX}" ${KAPPA:+--kappa "${KAPPA}"} \
            "${SELECT_ARGS[@]}" --floor "${FLOOR}" --guard_ratio "${GUARD_RATIO}" \
            ${REPORT_FLOORS:+--report_floors "${REPORT_FLOORS}"} \
            --model_dir "${MODEL_DIR}" --out_json "${OUT_ROOT}/${TAG}_${SLURM_JOB_ID}.json" ;;
    *) echo "ERROR: unknown MODE ${MODE}" >&2; exit 1 ;;
esac
