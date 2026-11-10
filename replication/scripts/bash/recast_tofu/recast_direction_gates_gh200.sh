#!/bin/bash
# ReCAST with learned direction gates on TOFU forget10, scored like Table 3 (extension, user 2026-11-03;
# replication/scripts/recast_direction_gates.py). Phase one: a gate in [0, 1] on each of k fixed input directions of
# every down_proj and o_proj (input-contrast, its dual, forget-input or random basis), trained with the model frozen on
# J = L_R + alpha L_F + gamma O + beta R (retain NLL, forget unlikelihood -log(1 - p), open gates sum s, relative
# displacement) until convergence
# (user 2026-11-03; capped at GATE_MAX_EPOCHS), every epoch logging the probe J, the gates' change, path, drift and
# reversals; the gated weights are written into the native weights. Phase two: the campaign's retain-only recipe on every parameter (5 epochs,
# warm-up to 1e-5 over epoch 1, linear to 0, fp32 weights with bf16 autocast, clip 1), as restricted_shrink.
# One array task per entry of CONFIGS x SEEDS (seed innermost). Each task first runs the CPU tests of the ReCAST
# modules and of the gates and stops if they fail. Submit from the repository root, with the array matching:
#   sbatch --array=0-<n-1> replication/scripts/bash/recast_tofu/recast_direction_gates_gh200.sh
# --export overrides, printed in the [launch] line: EPOCHS=0 evaluates the gated model without phase two;
# GATE_INIT (the gates' start, default 1), GATE_LAMBDA_STAR (gate only lambda >= lambda*), GATE_CUT / GATE_CUT_S /
# GATE_ROTATE_EVR (with a binarization share, or with GATE_MAX_EPOCHS=0 on every candidate direction, user 2026-11-05:
# with GATE_LAMBDA_STAR=1 on protected_contrast the selective rule S2's directions; shrink the chosen directions to
# GATE_CUT_S, or rotate them, GATE_CUT_S then the share c each part keeps along its own direction, a partial turn);
# GATE_MAX_EPOCHS / GATE_CONV_EPOCHS / GATE_TOL / GATE_DRIFT_MAX set the gate schedule (e.g. 10 / 10 / 0 / 1e-9 runs
# exactly 10 epochs, the length of a calibration trial, reported as not converged); GATE_CONV_EPOCHS=0 is the fixed
# schedule (user 2026-11-05): exactly GATE_MAX_EPOCHS epochs, no convergence rule (GATE_TOL and GATE_DRIFT_MAX not
# passed), GATE_LR_SCHEDULE constant or linear (to 0 over the steps), the probe after every GATE_PROBE_EVERY-th epoch
# and the last.
#SBATCH --job-name=ai-engram-dgates
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=replication/logs/slurm/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/recast_tofu.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=ai-engram-replication
export WANDB_DIR="${STORAGE}/archive/${USER}/ai-engram_logs"
export TOKENIZERS_PARALLELISM=false

# Configurations, one per task (x SEEDS), as basis:alpha:gamma:optimizer:lr[:share].
#   basis: input_contrast (the lambda-leading directions of C_f v = lambda (C_r + rho/in I) v, orthonormalised by a QR
#     in lambda order), input_contrast_dual (the same eigenvectors, each gate removing exactly the coefficient of one),
#     protected_contrast / protected_contrast_dual (both with the constant prompt prefix's inputs protected and the
#     retain covariance over every non-prefix token; recast_selective.py, user 2026-11-04), forget_input (the leading
#     eigenvectors of C_f, no retain contrast) or random (a random orthonormal frame);
#   alpha: weight of the forget unlikelihood L_F; gamma: weight of the closing term O = sum_j s_j (user 2026-11-03:
#     every gate pushed shut at the price gamma, only L_R holds open what retain needs; 0:gamma is that objective alone);
#   optimizer:lr of the gates, constant (calibration 25593901: Adam 1e-2 monotone with few reversals; Adam 1e-5 2-3
#     orders of magnitude too slow; SGD oscillates at every lr);
#   share (optional): binarize after the gate phase (user 2026-11-04: the gates choose the directions, the cut removes
#     them whole): the most closed gates that remove this share of the forget output energy set to GATE_BINARY_S, the
#     rest to 1 (the global selection's budget: floors 0.35 / 0.3 / 0.25 removed 0.1259 / 0.1461 / 0.1676).
# Default bases: the prefix-protected ones (user 2026-11-04: "ochrona prefiksu powinna być domyślna do bramek"; the
# unprotected bases only as an explicit comparison). Current set: the closing objective alone (alpha 0, beta 0), gamma
# from the closing price of the census (25593901: 75% / 90% of the gates pushed shut at 2.4e-5 / 5.1e-5 on the QR basis,
# 2.5e-5 / 5.9e-5 on the dual), to convergence; continuous gates, the main target (user 2026-11-04).
# GATE_CONFIGS="<config> ..." in --export replaces this list.
CONFIGS=(protected_contrast_dual:0:1e-5:adam:1e-2 protected_contrast_dual:0:3e-5:adam:1e-2
         protected_contrast_dual:0:1e-4:adam:1e-2 protected_contrast:0:3e-5:adam:1e-2)
[[ -n "${GATE_CONFIGS:-}" ]] && read -ra CONFIGS <<< "${GATE_CONFIGS}"
SEEDS=(0)
[[ -n "${GATE_SEEDS:-}" ]] && read -ra SEEDS <<< "${GATE_SEEDS}"  # e.g. GATE_SEEDS="1 2" in --export

# Gate phase (fixed for this campaign unless the user changes them).
PROFILE=down_o
BLOCKS=all
GATE_RHO=1.0          # ridge of the input contrast (user rule since 2026-11-01: rho 1 for every contrast family)
GATE_K=1024           # directions gated per target (the best global selection, score order, seed 0 of 25455033, took 45-78 per o_proj, 159-481 per down_proj)
GATE_MAX_EPOCHS="${GATE_MAX_EPOCHS:-200}"  # cap on the forget-anchored epochs (13 steps each at 8 x 4 items)
GATE_CONV_EPOCHS="${GATE_CONV_EPOCHS:-20}" # trailing window of the convergence rule (260 steps)
GATE_TOL="${GATE_TOL:-1e-3}"               # converged: probe J fell by at most this (relative) over the window ...
GATE_DRIFT_MAX="${GATE_DRIFT_MAX:-0.5}"    # ... and ||s_end - s_start|| / path <= this (noise: ~0.06 SGD, ~0.27 Adam)
GATE_PROBE_RETAIN=400 # retain items of the fixed probe (with all 400 forget items), evaluated after every epoch
GATE_MICRO=8
GATE_ACCUM=4
GATE_BETA=0.0         # weight of R = sum e (1 - s)^2 / sum ||W0||^2
GATE_AUTOCAST=bfloat16
GATE_BINARY_S=0.04    # share a binarized direction keeps (the global selection's s); the choice is made at it
GATE_INIT="${GATE_INIT:-1.0}"                # every gate's start (1: the model is Init Fine. at the start)
GATE_LAMBDA_STAR="${GATE_LAMBDA_STAR:-}"     # empty: the top GATE_K directions per target
GATE_CUT="${GATE_CUT:-}"                     # empty: the chosen directions keep GATE_BINARY_S
GATE_CUT_S="${GATE_CUT_S:-}"
GATE_ROTATE_EVR="${GATE_ROTATE_EVR:-}"
GATE_LR_SCHEDULE="${GATE_LR_SCHEDULE:-constant}"  # linear: from the config's lr to 0, the fixed schedule only
GATE_PROBE_EVERY="${GATE_PROBE_EVERY:-1}"         # above 1: the fixed schedule only

# Phase two: the current recipe (user rules 2026-10-27 / 2026-10-28), as the restricted_shrink runs of the campaign.
EPOCHS="${EPOCHS:-5}"
LR_PEAK=1e-5
LR_POINTS=0:0,1:1e-5,5:0
LR_NAME=warm1_linear0_e5
EPOCH_LENGTH=retain
TRAIN_DTYPE=float32
MAX_GRAD_NORM=1.0
PHASE_TWO_RECIPE=fp32_autocast
OPTIM=adamw_torch

TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
SEED="${SEEDS[$((TASK % ${#SEEDS[@]}))]}"
CONFIG="${CONFIGS[$((TASK / ${#SEEDS[@]}))]:?task ${TASK} is beyond CONFIGS x SEEDS}"
IFS=: read -r BASIS ALPHA GAMMA GATE_OPTIM GATE_LR SHARE <<< "${CONFIG}"
[[ -n "${GATE_LR}" ]] || { echo "ERROR: config ${CONFIG} is not basis:alpha:gamma:optimizer:lr" >&2; exit 1; }

BIN_ARGS=()
BIN_TAG=""
if [[ -n "${SHARE}" ]]; then
    BIN_ARGS=(--direction_gates_binarize_share "${SHARE}" --direction_gates_binary_s "${GATE_BINARY_S}")
    BIN_TAG="_bin${SHARE/./p}s${GATE_BINARY_S/./p}"
fi
CUT_ARGS=()
CUT_TAG=""
if [[ -n "${GATE_CUT}" ]]; then
    [[ -n "${SHARE}" || "${GATE_MAX_EPOCHS}" == 0 ]] \
        || { echo "ERROR: GATE_CUT needs a binarization share or GATE_MAX_EPOCHS=0 (every candidate cut)" >&2; exit 1; }
    [[ -z "${SHARE}" ]] && CUT_TAG="_all"
    if [[ "${GATE_CUT}" == shrink ]]; then
        CUT_ARGS=(--direction_gates_cut shrink --direction_gates_cut_s "${GATE_CUT_S:?GATE_CUT=shrink needs GATE_CUT_S}")
        CUT_TAG="${CUT_TAG}_cut${GATE_CUT_S/./p}"
    elif [[ "${GATE_CUT}" == rotate ]]; then
        CUT_ARGS=(--direction_gates_cut rotate --direction_gates_rotate_evr "${GATE_ROTATE_EVR:?GATE_CUT=rotate needs GATE_ROTATE_EVR}")
        CUT_TAG="${CUT_TAG}_rot${GATE_ROTATE_EVR/./p}"
        if [[ -n "${GATE_CUT_S}" ]]; then
            CUT_ARGS+=(--direction_gates_cut_s "${GATE_CUT_S}")
            CUT_TAG="${CUT_TAG}c${GATE_CUT_S/./p}"
        fi
    else
        echo "ERROR: GATE_CUT must be empty, shrink (with GATE_CUT_S) or rotate" >&2; exit 1
    fi
elif [[ -n "${GATE_CUT_S}" || -n "${GATE_ROTATE_EVR}" ]]; then
    echo "ERROR: GATE_CUT_S and GATE_ROTATE_EVR need GATE_CUT" >&2; exit 1
fi
SCHED_ARGS=(--direction_gates_max_epochs "${GATE_MAX_EPOCHS}" --direction_gates_convergence_epochs "${GATE_CONV_EPOCHS}"
            --direction_gates_lr_schedule "${GATE_LR_SCHEDULE}" --direction_gates_probe_every "${GATE_PROBE_EVERY}")
if [[ "${GATE_CONV_EPOCHS}" == 0 ]]; then
    SCHED_TAG="_fixed${GATE_MAX_EPOCHS}_${GATE_LR_SCHEDULE}_pe${GATE_PROBE_EVERY}"
    TOL_SHOWN=none DRIFT_SHOWN=none
else
    SCHED_ARGS+=(--direction_gates_tol "${GATE_TOL}" --direction_gates_drift_max "${GATE_DRIFT_MAX}")
    SCHED_TAG="_max${GATE_MAX_EPOCHS}_w${GATE_CONV_EPOCHS}_tol${GATE_TOL//-/m}_drift${GATE_DRIFT_MAX/./p}"
    [[ "${GATE_LR_SCHEDULE}" != constant || "${GATE_PROBE_EVERY}" != 1 ]] && SCHED_TAG="${SCHED_TAG}_${GATE_LR_SCHEDULE}_pe${GATE_PROBE_EVERY}"
    TOL_SHOWN="${GATE_TOL}" DRIFT_SHOWN="${GATE_DRIFT_MAX}"
fi
LS_ARGS=()
LS_TAG=""
if [[ -n "${GATE_LAMBDA_STAR}" ]]; then
    LS_ARGS=(--direction_gates_lambda_star "${GATE_LAMBDA_STAR}")
    LS_TAG="_ls${GATE_LAMBDA_STAR/./p}"
fi
INIT_TAG=""
[[ "${GATE_INIT}" != "1.0" ]] && INIT_TAG="_init${GATE_INIT/./p}"
RHO_ARGS=()
B_TAG="${BASIS//_/}"
if [[ "${BASIS}" == input_contrast || "${BASIS}" == input_contrast_dual || "${BASIS}" == protected_contrast \
      || "${BASIS}" == protected_contrast_dual ]]; then
    RHO_ARGS=(--direction_gates_rho "${GATE_RHO}")
    B_TAG="${B_TAG}${GATE_RHO/./p}"
fi
EP_TAG="_traj"
[[ "${EPOCHS}" == "0" ]] && EP_TAG="_gatesonly"
TRAJ_ARGS=()
[[ "${EPOCHS}" != "0" ]] && TRAJ_ARGS=(--eval_trajectory nll)
LR_TAG="${GATE_LR/./p}"; LR_TAG="${LR_TAG//-/m}"
GM_TAG="${GAMMA/./p}"; GM_TAG="${GM_TAG//-/m}"
G_TAG="${B_TAG}_k${GATE_K}_a${ALPHA/./p}_g${GM_TAG}_b${GATE_BETA/./p}_${GATE_OPTIM}${LR_TAG}${SCHED_TAG}${LS_TAG}${INIT_TAG}${BIN_TAG}${CUT_TAG}_${GATE_AUTOCAST}"
RUN_NAME="recast_direction_gates_${PROFILE}_blocks${BLOCKS}_tied_${G_TAG}_lr${LR_NAME}_nll_seed${SEED}_${EPOCH_LENGTH}_${TRAIN_DTYPE}_clip${MAX_GRAD_NORM/./p}${EP_TAG}_${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT_DIR="${STORAGE}/experiment_results/ai-engram/recast_tofu/${RUN_NAME}"
# Init Fine. and Retain raw metrics: Table 3 run 24952059 (same evaluation code and environment).
ANCHORS_RAW="${STORAGE}/experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/raw_metrics.json"

echo "[launch] host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-none}_${TASK} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[launch] config ${CONFIG} of ${#CONFIGS[@]} (${CONFIGS[*]})"
echo "[launch] basis=${BASIS} alpha=${ALPHA} gamma=${GAMMA} binarize_share=${SHARE:-none} binary_s=${GATE_BINARY_S} cut=${GATE_CUT:-none} cut_s=${GATE_CUT_S:-none} rotate_evr=${GATE_ROTATE_EVR:-none} init=${GATE_INIT} lambda_star=${GATE_LAMBDA_STAR:-none} beta=${GATE_BETA} k=${GATE_K} gate_optim=${GATE_OPTIM} gate_lr=${GATE_LR} gate_max_epochs=${GATE_MAX_EPOCHS} convergence_epochs=${GATE_CONV_EPOCHS} tol=${TOL_SHOWN} drift_max=${DRIFT_SHOWN} lr_schedule=${GATE_LR_SCHEDULE} probe_every=${GATE_PROBE_EVERY} probe_retain=${GATE_PROBE_RETAIN} seed=${SEED} epochs=${EPOCHS} out_dir=${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[launch] nvidia-smi unavailable"

cd "${REPO_ROOT}"
echo "[gate] ReCAST module tests, direction-gate tests and selective-contrast tests (CPU)"
"${PY}" -m pytest -q -p no:cacheprovider replication/tests/test_recast_core.py replication/tests/test_recast_scores.py \
    replication/tests/test_recast_bases.py replication/tests/test_recast_contraction.py \
    replication/tests/test_recast_output_contrast.py replication/tests/test_recast_direction_families.py \
    replication/tests/test_recast_global_families.py replication/tests/test_recast_direction_gates.py \
    replication/tests/test_recast_selective.py

"${PY}" -u replication/scripts/recast_tofu.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --holdout_split holdout10 \
    --phase_one_dtype float32 \
    --train_dtype "${TRAIN_DTYPE}" \
    --eval_dtype bfloat16 \
    --attn_implementation sdpa \
    --variant direction_gates \
    --targets "${PROFILE}" \
    --target_blocks "${BLOCKS}" \
    --untie_lm_head no \
    --train_lm_head yes \
    --grad_batch_size 8 \
    --direction_gates_basis "${BASIS}" \
    "${RHO_ARGS[@]}" \
    --direction_gates_candidates "${GATE_K}" \
    "${SCHED_ARGS[@]}" \
    --direction_gates_probe_retain "${GATE_PROBE_RETAIN}" \
    --direction_gates_optim "${GATE_OPTIM}" \
    --direction_gates_micro_batch "${GATE_MICRO}" \
    --direction_gates_grad_accum "${GATE_ACCUM}" \
    --direction_gates_lr "${GATE_LR}" \
    --direction_gates_alpha "${ALPHA}" \
    --direction_gates_beta "${GATE_BETA}" \
    --direction_gates_gamma "${GAMMA}" \
    --direction_gates_init "${GATE_INIT}" \
    "${LS_ARGS[@]}" \
    "${BIN_ARGS[@]}" \
    "${CUT_ARGS[@]}" \
    --direction_gates_autocast "${GATE_AUTOCAST}" \
    --epochs "${EPOCHS}" \
    "${TRAJ_ARGS[@]}" \
    --epoch_length "${EPOCH_LENGTH}" \
    --per_device_batch_size 8 --grad_accum 4 --optim "${OPTIM}" --gradient_checkpointing no \
    --loss_normalization window_tokens \
    --lr "${LR_PEAK}" \
    --weight_decay 0.01 \
    --lr_scheduler piecewise_epochs \
    --lr_points "${LR_POINTS}" \
    --retain_loss nll \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --seed "${SEED}" \
    --eval_batch_size 16 \
    --anchors_raw "${ANCHORS_RAW}" \
    --out_dir "${OUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --wandb_mode online \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_tags recast_tofu_comparison matched_protocol method_recast_direction_gates \
        model_tofu_llama3p2_1b_instruct split_forget10 base_init_fine variant_direction_gates "gates_basis_${BASIS}" \
        "gates_k_${GATE_K}" "gates_alpha_${ALPHA/./p}" "gates_gamma_${GM_TAG}" "gates_binarize_${SHARE:-none}" "gates_init_${GATE_INIT/./p}" "gates_lambda_star_${GATE_LAMBDA_STAR:-none}" "gates_cut_${GATE_CUT:-none}${GATE_CUT_S}${GATE_ROTATE_EVR}${CUT_TAG:+_${CUT_TAG#_}}" "gates_beta_${GATE_BETA/./p}" "gates_optim_${GATE_OPTIM}" \
        "gates_lr_${LR_TAG}" "gates_max_epochs_${GATE_MAX_EPOCHS}" "gates_conv_${GATE_CONV_EPOCHS}" \
        "gates_tol_${TOL_SHOWN//-/m}" "gates_drift_${DRIFT_SHOWN/./p}" "gates_probe_${GATE_PROBE_RETAIN}" \
        "gates_lr_schedule_${GATE_LR_SCHEDULE}" "gates_probe_every_${GATE_PROBE_EVERY}" \
        "gates_autocast_${GATE_AUTOCAST}" phase2_retain_only retain_loss_nll \
        "optim_${OPTIM}" "phase2_recipe_${PHASE_TWO_RECIPE}" "lr_peak_${LR_PEAK}" "lr_${LR_NAME}" wd_0p01 \
        "epochs_${EPOCHS}" "epoch_len_${EPOCH_LENGTH}" phase1_dtype_float32 "dtype_${TRAIN_DTYPE}" \
        "targets_${PROFILE}" "blocks_${BLOCKS}" lm_head_tied embedding_trained "grad_clip_${MAX_GRAD_NORM/./p}" \
        "seed_${SEED}"

echo "[launch] done"
