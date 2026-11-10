#!/bin/bash
# Learning-rate calibration of the learned direction gates on TOFU forget10 (user 2026-11-03: "zmierzmy jak powinniśmy
# dobrać lr"; replication/scripts/recast_direction_gates.py, recast_tofu.py --direction_gates_calibration_trials).
# No phase two and no evaluation. With the model frozen at Init Fine. and every gate at 1:
#   1. gradient census: per window of CENSUS_EPOCHS forget-anchored epochs and per gate, the gradient of L_F and of
#      L_R alone; printed guides: |gradient| quantiles per term, the SGD lr whose 99th-percentile step is 1e-3 / 1e-2 /
#      1e-1, Adam's steady step per unit lr (|mean| / rms over the windows), the share of gates pushed shut per alpha,
#      and the closing price (the gamma that pushes a given share of the gates shut against L_R alone);
#   2. one short training per optimizer:lr of TRIALS, each from s = 1 on the same probe and windows (same seed):
#      probe J per epoch, drift, reversals, largest step, gate distribution.
# Outputs: results.json (census guides, trial summaries, column specificity), direction_gates_calibration_history.json
# (every trial's step and epoch records), direction_gates_calibration.pt (the census rows, the own ratios per column).
# One array task per basis (axis below). Each task first runs the CPU tests of the ReCAST modules and of the gates and
# stops if they fail. Submit from the repository root, with the array matching the axis:
#   sbatch --array=0-<n-1> replication/scripts/bash/recast_tofu/recast_gate_calibration_gh200.sh
#SBATCH --job-name=ai-engram-gcalib
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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

# Axis: basis. Default the prefix-protected bases (user 2026-11-04): protected_contrast (QR columns) and
# protected_contrast_dual (one eigen-coefficient per gate), both with the constant prompt prefix's inputs protected and
# the retain covariance over every non-prefix token; input_contrast[_dual], forget_input, random only as a comparison.
BASES=(protected_contrast protected_contrast_dual)

# Objective of the trials (the census is measured per term and does not depend on it): forget unlikelihood alone.
ALPHA=1.0
GAMMA=0
GATE_BETA=0.0
# Trials, optimizer:lr, each a short training from s = 1 (user 2026-11-03: Adam 1e-5 as for the network, SGD 1e-3;
# the grid brackets both by decades, since the gates' gradient scale is not known yet).
TRIALS=adam:1e-5,adam:1e-4,adam:1e-3,adam:1e-2,sgd:1e-3,sgd:1e-2,sgd:1e-1,sgd:1,sgd:10
TRIAL_EPOCHS=10       # 130 steps per trial at 8 x 4 items
CENSUS_EPOCHS=2       # 26 windows
SEED=0

PROFILE=down_o
BLOCKS=all
GATE_RHO=1.0          # ridge of the input contrast (user rule since 2026-11-01: rho 1 for every contrast family)
GATE_K=1024           # directions gated per target, as recast_direction_gates_gh200.sh
GATE_PROBE_RETAIN=400
GATE_MICRO=8
GATE_ACCUM=4
GATE_AUTOCAST=bfloat16

# Phase-two arguments: required by the parser, unused (the calibration stops before phase two); the current recipe.
EPOCHS=5
LR_PEAK=1e-5
LR_POINTS=0:0,1:1e-5,5:0
EPOCH_LENGTH=retain
TRAIN_DTYPE=float32
MAX_GRAD_NORM=1.0
OPTIM=adamw_torch

TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
BASIS="${BASES[${TASK}]:?task ${TASK} is beyond the configured axis}"
RHO_ARGS=()
B_TAG="${BASIS//_/}"
if [[ "${BASIS}" == input_contrast || "${BASIS}" == input_contrast_dual || "${BASIS}" == protected_contrast \
      || "${BASIS}" == protected_contrast_dual ]]; then
    RHO_ARGS=(--direction_gates_rho "${GATE_RHO}")
    B_TAG="${B_TAG}${GATE_RHO/./p}"
fi
RUN_NAME="recast_gate_calibration_${PROFILE}_blocks${BLOCKS}_${B_TAG}_k${GATE_K}_a${ALPHA/./p}_g${GAMMA/./p}_b${GATE_BETA/./p}_trials${TRIAL_EPOCHS}ep_census${CENSUS_EPOCHS}ep_${GATE_AUTOCAST}_seed${SEED}_${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT_DIR="${STORAGE}/experiment_results/ai-engram/recast_tofu/${RUN_NAME}"
ANCHORS_RAW="${STORAGE}/experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/raw_metrics.json"

echo "[launch] host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-none}_${TASK} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[launch] calibration basis=${BASIS} alpha=${ALPHA} gamma=${GAMMA} beta=${GATE_BETA} k=${GATE_K} trials=${TRIALS} trial_epochs=${TRIAL_EPOCHS} census_epochs=${CENSUS_EPOCHS} probe_retain=${GATE_PROBE_RETAIN} seed=${SEED} out_dir=${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[launch] nvidia-smi unavailable"

cd "${REPO_ROOT}"
echo "[gate] ReCAST module tests and direction-gate tests (CPU)"
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
    --direction_gates_calibration_trials "${TRIALS}" \
    --direction_gates_trial_epochs "${TRIAL_EPOCHS}" \
    --direction_gates_census_epochs "${CENSUS_EPOCHS}" \
    --direction_gates_probe_retain "${GATE_PROBE_RETAIN}" \
    --direction_gates_micro_batch "${GATE_MICRO}" \
    --direction_gates_grad_accum "${GATE_ACCUM}" \
    --direction_gates_alpha "${ALPHA}" \
    --direction_gates_beta "${GATE_BETA}" \
    --direction_gates_gamma "${GAMMA}" \
    --direction_gates_autocast "${GATE_AUTOCAST}" \
    --epochs "${EPOCHS}" \
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
    --wandb_tags recast_tofu_comparison gate_calibration method_recast_direction_gates \
        model_tofu_llama3p2_1b_instruct split_forget10 base_init_fine variant_direction_gates "gates_basis_${BASIS}" \
        "gates_k_${GATE_K}" "gates_alpha_${ALPHA/./p}" "gates_gamma_${GAMMA/./p}" "gates_beta_${GATE_BETA/./p}" \
        "gates_trial_epochs_${TRIAL_EPOCHS}" "gates_census_epochs_${CENSUS_EPOCHS}" "gates_probe_${GATE_PROBE_RETAIN}" \
        "gates_autocast_${GATE_AUTOCAST}" "targets_${PROFILE}" "blocks_${BLOCKS}" "seed_${SEED}"

echo "[launch] done"
