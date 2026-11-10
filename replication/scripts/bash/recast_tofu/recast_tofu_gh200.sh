#!/bin/bash
# ReCAST on TOFU forget10, scored like Table 3: one array task per combination of the axes below,
# decoded as a mixed-radix number with the ablation innermost and the variant outermost; EVR applies
# to the restricted variants only. The lm_head is not shrunk; head and input
# embedding stay one tied matrix, trained in phase two. Each task first runs the CPU parity
# tests of recast_core.py and stops if they fail.
# Submit from the repository root, stating the two phase-two choices explicitly:
#   sbatch --export=ALL,EPOCH_LENGTH=retain,TRAIN_DTYPE=float32,MAX_GRAD_NORM=1.0 replication/scripts/bash/recast_tofu/recast_tofu_gh200.sh
# MAX_GRAD_NORM=0 disables gradient clipping in the Hugging Face Trainer. EPOCHS=0 in --export evaluates the
# warm start alone (no phase two); the default is 5 (10 before 2026-10-27). TRAJECTORY=nll in --export also tracks
# the answer NLL and exact memorization of forget, retain_perturbed, real authors and world facts before phase two and
# after every epoch.
# COMPONENT_DIAGNOSTICS=yes in --export writes the per-target update components and the first-order loss check.
# SHRINK_SCORE in --export (Stage 2, restricted only): original (default), projected_row, projected_sensitivity or
# global_projected, the row score of S (recast_scores.py); SCORE_DIAGNOSTICS=yes reports every rule's scores, the
# first-order change of the applied update and the update norm on a floor grid. CORE_DIAGNOSTICS=yes (Stage 3) compares
# the weight core with the update core per target for the applied S, each rule's S and a random control S.
# BASE_ID=<model id or directory> (with BASE_TAG=<name for the run name>) in --export starts from another model than
# Init Fine., e.g. a saved Engram edit; with VARIANTS=(unrestricted) and S_MINS=(1.0) phase one is the identity and the
# run is phase two alone. SAVE_MODEL_SEED=<seed> in --export saves that seed's model after phase two.
# TRAIN_SPLIT=full in --export trains phase two on the whole TOFU set (forget and retain) instead of retain90: a
# relearning run that measures the recovery of a finished model (user, 2026-10-29); only with the identity phase one.
# PHASE_TWO_RECIPE in --export: fp32_autocast (default: float32 weights with bfloat16 autocast, micro-batch 8 x 4,
# adamw_torch) or deepspeed_zero3 (as OpenUnlearning trained Init Fine. and Retain, user 2026-10-28: DeepSpeed ZeRO 3
# with bf16, i.e. float32 master weights and bfloat16 forward / backward, here in one process with micro-batch 4 x 8,
# paged_adamw_32bit, gradient checkpointing, the NLL averaged per micro-batch; needs TRAIN_DTYPE=bfloat16 and
# MAX_GRAD_NORM=0, since their DeepSpeed config sets no gradient_clipping; runs in the ai-engram-gh200-ds environment).
#SBATCH --job-name=ai-engram-recast
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=replication/logs/slurm/%x-%A_%a.out

set -euo pipefail

: "${EPOCH_LENGTH:?set EPOCH_LENGTH=forget|retain with sbatch --export}"
: "${TRAIN_DTYPE:?set TRAIN_DTYPE=bfloat16|float32 with sbatch --export}"
: "${MAX_GRAD_NORM:?set MAX_GRAD_NORM (0 = no clipping) with sbatch --export}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/recast_tofu.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
PHASE_TWO_RECIPE="${PHASE_TWO_RECIPE:-fp32_autocast}"
case "${PHASE_TWO_RECIPE}" in
    fp32_autocast)
        RECIPE_ARGS=(--per_device_batch_size 8 --grad_accum 4 --optim adamw_torch --gradient_checkpointing no
                     --loss_normalization window_tokens)
        OPTIM=adamw_torch; RECIPE_TAG="" ;;
    deepspeed_zero3|deepspeed_zero3_clip)
        [[ "${TRAIN_DTYPE}" == bfloat16 ]] || { echo "ERROR: ${PHASE_TWO_RECIPE} needs TRAIN_DTYPE=bfloat16" >&2; exit 1; }
        # deepspeed_zero3: their config, no gradient_clipping (MAX_GRAD_NORM=0); deepspeed_zero3_clip (diagnostic,
        # 2026-10-28): the same config with gradient_clipping "auto", which the Trainer fills with MAX_GRAD_NORM.
        if [[ "${PHASE_TWO_RECIPE}" == deepspeed_zero3 ]]; then
            [[ "${MAX_GRAD_NORM}" == 0 ]] || { echo "ERROR: deepspeed_zero3 needs MAX_GRAD_NORM=0" >&2; exit 1; }
            DS_JSON=deepspeed_zero3_openunlearning.json
        else
            [[ "${MAX_GRAD_NORM}" != 0 ]] || { echo "ERROR: deepspeed_zero3_clip needs MAX_GRAD_NORM > 0" >&2; exit 1; }
            DS_JSON=deepspeed_zero3_clip_auto.json
        fi
        # A virtual environment over ai-engram-gh200 with deepspeed and bitsandbytes (build_env_ds_gh200.sh).
        PY="${AI_ENGRAM_DS_ENV:-${STORAGE}/conda_envs/ai-engram-gh200-ds}/bin/python"
        # deepspeed 0.15.4 checks its ops against nvcc under CUDA_HOME at import; CUDA_HOME alone, so torch keeps its
        # own CUDA libraries.
        export CUDA_HOME="${AI_ENGRAM_CUDA_HOME:-/net/software/aarch64/el9/CUDA/12.8.0}"
        [[ -x "${CUDA_HOME}/bin/nvcc" ]] || { echo "ERROR: no nvcc under CUDA_HOME=${CUDA_HOME}" >&2; exit 1; }
        RECIPE_ARGS=(--per_device_batch_size 4 --grad_accum 8 --optim paged_adamw_32bit --gradient_checkpointing yes
                     --loss_normalization micro_batch_mean
                     --deepspeed_config "${REPO_ROOT}/replication/scripts/bash/recast_tofu/${DS_JSON}")
        OPTIM=paged_adamw_32bit; RECIPE_TAG="_dszero3" ;;
    *) echo "ERROR: unknown PHASE_TWO_RECIPE ${PHASE_TWO_RECIPE}" >&2; exit 1 ;;
esac

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=ai-engram-replication
export WANDB_DIR="${STORAGE}/archive/${USER}/ai-engram_logs"
export TOKENIZERS_PARALLELISM=false

# Current configuration; the decisions behind each value and every earlier setting are in
# replication/docs/WORK_LOG.md. Axes, outermost first: variant, profile, floor, EVR, seed,
# learning rate, retain loss, retain input filter, ablation.
# restricted_shrink: the restricted warm start without adapters, phase two on every parameter.
VARIANTS=(restricted_shrink)
PROFILES=(down_up_o)
# Shrink floors: one value, or one per projection type of the profile (down_proj=0.2,up_proj=0.5,o_proj=0.5).
S_MINS=(0.2)
EVRS=(0.97)
SEEDS=(0 1 2)
# Learning-rate schedules, peak|points|name: warmup to the peak in epoch 1, the peak until epoch 5,
# then a step down to 1e-5 (user, 2026-10-22). Runs up to 24990942 went linearly from the peak at
# epoch 1 to the epoch-5 value instead.
# Since 2026-10-27 (user) phase two lasts 5 epochs with the schedule of the TOFU fine-tuning of Init Fine. and Retain:
# warm-up to 1e-5 over epoch 1, then linear to 0 at epoch 5. Earlier runs: "1e-5|0:0,1:1e-5,10:1e-5|warm1_const1e-5_e10"
# (constant after the warm-up, 10 epochs) and "1e-5|0:0,1:1e-5,5:1e-5|warm1_const1e-5_e5".
LR_CONFIGS=("1e-5|0:0,1:1e-5,5:0|warm1_linear0_e5")
# Phase-two retain loss, loss/KL weight: nll, or kl / nll_kl with the weight of KL to Init Fine.
RETAIN_LOSSES=(nll)
# Retain input covariance treatment (restricted only), kind/parameter: none, ridge/<kappa> (filter A of
# the warm-start delta), whiten/<rho> (restriction bases from the retain-whitened forget gradient, C) or
# repr/<rho>/<output side>[/<beta>] (bases D from the forget-vs-retain input covariance contrast; output side
# identity, gradient, whitened_gradient or representation; beta mixes the forget activations (1, default)
# with the forget gradient (0) on the input side; needs a RANK_MULTS budget), or gcontrast/<rho>/<sides> (user,
# 2026-11-01: the plain basis with the singular vectors of G_f replaced by the forget-vs-retain mean-gradient contrast,
# both sides, or the input side only with the output side of the plain basis), or gmoment/per_example (user,
# 2026-11-02: the plain basis with the singular vectors of G_f replaced by the leading eigenvectors of the per-example
# forget-gradient second moments E[g g^T], E[g^T g]). For gcontrast and gmoment, RANK_MULTS m gives both sides m x the
# plain-basis rank; RANK_MULTS none (user, 2026-11-02) gives each side its own-spectrum rank at the EVR: gmoment the
# EVR of its moment, gcontrast the directions holding that share of the mean forget gradient's energy on the side, or
# with gcontrast/<rho>/<sides>/kept_energy one rank for both sides whose two-sided restriction keeps that share.
# (24991502 ran ridge/0.1, ridge/1, ridge/10; 24993357 ridge/30-1000 and whiten/0.01-10.)
FILTERS=(repr/30/gradient)
# Edit budget (restricted only): none (rank from EVR), m (rank = round(m * plain-basis rank at EVR) per
# target, the same for the plain and the whitened basis) or qf/<q> (per target the smallest rank whose
# warm-start delta changes forget outputs by tr(D C_f D^T) / tr(W C_f W^T) >= q). Since 2026-10-29 (user) m stays 1:
# the rank comes from the EVR alone (runs up to 25345376 mostly used m 1.25 as the reference).
RANK_MULTS=(1.0)
# Independent ablations against the reference (full_space alignment, forget_only source, no
# layer scaling): each changes exactly one factor. Format: scope:source:scaling:phase2_free:tag.
# phase2_free (restricted_semu only, ignored otherwise): none, norms, attention, mlp or all (every
# parameter without an adapter; the default since 2026-10-22). (24979138 ran align_subspace,
# source_full_data, scaling_subspace_norm; 24982024 scaling_subspace_norm_mean; 24987066 reference,
# scaling_subspace_norm_mean, free_norms, free_attention, free_all with phase2_free none for the first two;
# 24987255, 24989313, 24989314 restricted_shrink with tag reference.)
ABLATIONS=(
    "full_space:forget_only:none:all:free_all"
)
# Components of the projected update (restricted only, innermost axis): none (the original update), full (its SVD
# reconstruction), signed (only components with c_j = <G_f, D_j> > 0), random (as many, drawn with the seed) or
# uniform_matched (the original update scaled to the signed update's norm).
COMPONENT_MODES=(none)
# Warm-start variants (restricted only, next-to-innermost axis): none, dirw/contrast (per-direction shrink weighted by
# the forget / retain input-energy contrast), and for Experiment 2 only scale/<t> (W_Init + t (W* - W_Init)) and
# scale/holdout (t chosen by forward passes so that the forget loss equals the Init Fine. holdout loss). Set the
# strength of a run with the floor (S_MINS), not with t (user, 2026-10-27). contract_top/<s> and contract_random/<s>
# (repr bases only) replace the update by contracting the weight-core components of P(W) by s, the largest first or
# random ones, until the displacement equals that of ReCAST at the floor, (1 - floor) ||P(W)|| (user, 2026-10-28);
# contract_top/<s>/<r> leaves every contracted component the singular value r instead of 0; contract_forget_retain/<s>
# takes the components by the first-order forget-minus-retain loss increase per unit of moved energy. CONTRACTION_DIAGNOSTICS=yes
# in --export reports every weight-core component's kappa, input contrast and gradient alignment (repr bases only).
# outcontrast/<rho> replaces the update by the output-contrast contraction (recast_output_contrast.py) with the ReCAST
# displacement. OUTPUT_DIAGNOSTICS=yes reports output ratios and removed output shares; WRITE_COMPENSATION=retain (with
# OUTPUT_DIAGNOSTICS=yes) scales the residual stream at every o_proj / down_proj write by its retain output ratio,
# stored in the weights (mechanism test, user 2026-10-28).
# global_output|global_gradient|global_input/<rho>/<floor>/<order>[/<alpha>] (user, 2026-10-29; repr bases, down_proj / o_proj
# targets only) replaces all updates together by the output-, gradient- or input-contrast directions of every target
# in one order (score: by the generalised eigenvalue; random: the control), each kept at <floor> of its part, until the
# forget output energy removed over all targets equals the total of the ReCAST updates at the floor (S_MINS); the
# order energy_ratio/<alpha> ranks by e_f / e_r^alpha, the absolute energies of each direction alone.
WARM_STARTS=(none dirw/contrast)
BLOCKS=all
EPOCHS="${EPOCHS:-5}"
# Phase one (gradients, covariances, SVDs, warm start) runs on float32 weights; TRAIN_DTYPE sets phase two
# (user, 2026-10-27). PHASE_ONE_DTYPE=bfloat16 in --export reproduces the earlier all-bfloat16 runs.
PHASE_ONE_DTYPE="${PHASE_ONE_DTYPE:-float32}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
i=${TASK}
COMP="${COMPONENT_MODES[$((i % ${#COMPONENT_MODES[@]}))]}"; i=$((i / ${#COMPONENT_MODES[@]}))
IFS=/ read -r WS_KIND WS_VALUE <<< "${WARM_STARTS[$((i % ${#WARM_STARTS[@]}))]}"; i=$((i / ${#WARM_STARTS[@]}))
IFS=: read -r SCOPE SOURCE SCALING FREE ABL_TAG <<< "${ABLATIONS[$((i % ${#ABLATIONS[@]}))]}"; i=$((i / ${#ABLATIONS[@]}))
IFS=/ read -r FILTER KAPPA FOUT FMIX <<< "${FILTERS[$((i % ${#FILTERS[@]}))]}"; i=$((i / ${#FILTERS[@]}))
RANK_MULT="${RANK_MULTS[$((i % ${#RANK_MULTS[@]}))]}"; i=$((i / ${#RANK_MULTS[@]}))
IFS=/ read -r RETAIN_LOSS KL_WEIGHT <<< "${RETAIN_LOSSES[$((i % ${#RETAIN_LOSSES[@]}))]}"; i=$((i / ${#RETAIN_LOSSES[@]}))
IFS='|' read -r LR_PEAK LR_POINTS LR_NAME <<< "${LR_CONFIGS[$((i % ${#LR_CONFIGS[@]}))]}"; i=$((i / ${#LR_CONFIGS[@]}))
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"; i=$((i / ${#SEEDS[@]}))
EVR="${EVRS[$((i % ${#EVRS[@]}))]}"; i=$((i / ${#EVRS[@]}))
S_MIN="${S_MINS[$((i % ${#S_MINS[@]}))]}"; i=$((i / ${#S_MINS[@]}))
PROFILE="${PROFILES[$((i % ${#PROFILES[@]}))]}"; i=$((i / ${#PROFILES[@]}))
VARIANT="${VARIANTS[${i}]:?task ${TASK} is beyond the configured axes}"
LOSS_ARGS=(--retain_loss "${RETAIN_LOSS}")
LOSS_TAG="${RETAIN_LOSS}"
if [[ "${RETAIN_LOSS}" == *kl* ]]; then
    LOSS_ARGS+=(--kl_weight "${KL_WEIGHT:?give the KL weight as loss/weight}")
    LOSS_TAG="${RETAIN_LOSS}${KL_WEIGHT/./p}"
    # KL_TEACHER=<model id or directory> with KL_TEACHER_TAG=<name> in --export: KL to that model instead of the base
    # (user, 2026-10-29).
    if [[ -n "${KL_TEACHER:-}" ]]; then
        LOSS_ARGS+=(--kl_teacher "${KL_TEACHER}")
        LOSS_TAG="${LOSS_TAG}_${KL_TEACHER_TAG:?set KL_TEACHER_TAG for KL_TEACHER}"
    fi
fi
FILTER_TAG="filter_${FILTER}${KAPPA:+${KAPPA/./p}}${FOUT:+_${FOUT}}"
if [[ "${FILTER}" == gcontrast ]]; then  # the fourth field of gcontrast is the own-spectrum rank rule
    FILTER_TAG="${FILTER_TAG}${FMIX:+_rule${FMIX}}"
else
    FILTER_TAG="${FILTER_TAG}${FMIX:+_mix${FMIX/./p}}"
fi
case "${RANK_MULT}" in
    none) ;;
    qf/*) QF="${RANK_MULT#qf/}"; FILTER_TAG="${FILTER_TAG}_qf${QF/./p}" ;;
    *) FILTER_TAG="${FILTER_TAG}_rankx${RANK_MULT/./p}" ;;
esac
EP_TAG=""
[[ "${EPOCHS}" == "0" ]] && EP_TAG="_warmstartonly"
[[ "${COMP}" != none ]] && FILTER_TAG="${FILTER_TAG}_comp${COMP}"
# The warm-start part goes into the run name and into a W&B tag of its own (W&B tags are at most 64 characters).
TAG_BASE="${FILTER_TAG}"
WS_TAG=""
[[ "${WS_KIND}" != none ]] && WS_TAG="${WS_KIND}${WS_VALUE//./p}" && WS_TAG="${WS_TAG//\//_}"
FILTER_TAG="${FILTER_TAG}${WS_TAG:+_${WS_TAG}}"
[[ "${WRITE_COMPENSATION:-}" == retain ]] && FILTER_TAG="${FILTER_TAG}_wcompretain"
# GLOBAL_BUDGET_UNIT=share in --export (global_* warm starts): the budget sums each target's removed share of its own
# forget output energy, which admits targets of any type (user, 2026-10-31); run tag _bshare.
[[ "${GLOBAL_BUDGET_UNIT:-energy}" == share ]] && FILTER_TAG="${FILTER_TAG}_bshare"
# GLOBAL_S_GAMMA=<gamma> in --export (global_* warm starts): each direction keeps clip(lambda^-gamma, floor, 1) of its part,
# the floor of the warm-start entry being the lower clip (user, 2026-10-31); run tag _sgamma<gamma>.
[[ -n "${GLOBAL_S_GAMMA:-}" ]] && FILTER_TAG="${FILTER_TAG}_sgamma${GLOBAL_S_GAMMA/./p}"
# GLOBAL_S_RULE=retain_share in --export: each direction keeps clip(e_r / (e_f + e_r), floor, 1) of its part (absolute
# output energies of the direction alone; user, 2026-10-31); run tag _sretainshare.
[[ -n "${GLOBAL_S_RULE:-}" ]] && FILTER_TAG="${FILTER_TAG}_s${GLOBAL_S_RULE//_/}"
# GLOBAL_S_THRESHOLD_GAMMA=<gamma> in --export: every direction keeps clip((lambda*/lambda)^gamma, floor, 1), one threshold
# lambda* meeting the budget (user, 2026-10-31); run tag _sthr<gamma>.
[[ -n "${GLOBAL_S_THRESHOLD_GAMMA:-}" ]] && FILTER_TAG="${FILTER_TAG}_sthr${GLOBAL_S_THRESHOLD_GAMMA/./p}"
BASE_ID="${BASE_ID:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
BASE_TAG="${BASE_TAG:-}"
[[ "${BASE_ID}" != open-unlearning/tofu_Llama-3.2-1B-Instruct_full && -z "${BASE_TAG}" ]] && { echo "ERROR: set BASE_TAG for BASE_ID=${BASE_ID}" >&2; exit 1; }
TRAIN_SPLIT="${TRAIN_SPLIT:-retain90}"
PHASE_TWO_TAG=phase2_retain_only
case "${TRAIN_SPLIT}" in
    retain90) ;;
    full) [[ "${VARIANT}" == unrestricted && "${S_MIN}" == 1.0 ]] || { echo "ERROR: TRAIN_SPLIT=full needs the identity phase one (VARIANTS=(unrestricted), S_MINS=(1.0))" >&2; exit 1; }
          EP_TAG="${EP_TAG}_trainfull"; PHASE_TWO_TAG=phase2_full ;;
    *) echo "ERROR: TRAIN_SPLIT=${TRAIN_SPLIT} is not retain90 or full" >&2; exit 1 ;;
esac
TRAJ_ARGS=()
if [[ -n "${TRAJECTORY:-}" ]]; then
    TRAJ_ARGS=(--eval_trajectory "${TRAJECTORY}")
    EP_TAG="${EP_TAG}_traj"
fi
S_TAG="$(sed 's/_proj//g; s/\./p/g; s/=//g; s/,/_/g' <<< "${S_MIN}")"  # 0p3, or down0p2_up0p5_o0p5
E_TAG="${EVR/./p}"
CLIP_TAG="${MAX_GRAD_NORM/./p}"
[[ "${MAX_GRAD_NORM}" == "0" ]] && CLIP_TAG=none
# Init Fine. and Retain raw metrics: Table 3 run 24952059 (same evaluation code and environment).
ANCHORS_RAW="${STORAGE}/experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/raw_metrics.json"
P1_TAG=""
[[ "${PHASE_ONE_DTYPE}" != "${TRAIN_DTYPE}" ]] && P1_TAG="_p1${PHASE_ONE_DTYPE}"
SHRINK_SCORE="${SHRINK_SCORE:-original}"
SCORE_TAG=""
[[ "${SHRINK_SCORE}" != original ]] && SCORE_TAG="_score_${SHRINK_SCORE}"
RUN_NAME="recast_${BASE_TAG:+${BASE_TAG}_}${VARIANT}_${PROFILE}_blocks${BLOCKS}_tied_smin${S_TAG}_evr${E_TAG}_${ABL_TAG}_${FILTER_TAG}${SCORE_TAG}_lr${LR_NAME}_${LOSS_TAG}_seed${SEED}_${EPOCH_LENGTH}${P1_TAG}_${TRAIN_DTYPE}${RECIPE_TAG}_clip${CLIP_TAG}${EP_TAG}_${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT_DIR="${STORAGE}/experiment_results/ai-engram/recast_tofu/${RUN_NAME}"
SAVE_ARGS=()
if [[ -n "${SAVE_MODEL_SEED:-}" && "${SEED}" == "${SAVE_MODEL_SEED}" ]]; then
    SAVE_ARGS=(--save_model_dir "${STORAGE}/runtime_artifacts/ai-engram/recast_tofu/models/${RUN_NAME}")
fi

VARIANT_ARGS=(--freeze_input_embedding no --shrink_layer_scaling "${SCALING}")
VARIANT_TAGS=(variant_unrestricted "scaling_${SCALING}")
if [[ "${VARIANT}" == restricted_* ]]; then
    VARIANT_ARGS=(--restriction_source "${SOURCE}" --restriction_evr "${EVR}"
                  --restriction_alignment_scope "${SCOPE}" --shrink_layer_scaling "${SCALING}")
    case "${FILTER}" in
        none) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening none --restriction_representation none) ;;
        ridge) VARIANT_ARGS+=(--retain_input_filter ridge --retain_filter_kappa "${KAPPA:?give kappa as ridge/kappa}"
                              --restriction_whitening none --restriction_representation none) ;;
        whiten) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening retain_input
                               --whitening_rho "${KAPPA:?give rho as whiten/rho}" --restriction_representation none) ;;
        repr) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening none
                             --restriction_representation "${FOUT:?give the output side as repr/rho/side}"
                             --whitening_rho "${KAPPA:?give rho as repr/rho/side}" ${FMIX:+--representation_mix "${FMIX}"}) ;;
        gcontrast) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening none --restriction_representation none
                                  --restriction_gradient_contrast "${FOUT:?give the sides as gcontrast/rho/sides}"
                                  --whitening_rho "${KAPPA:?give rho as gcontrast/rho/sides}"
                                  ${FMIX:+--gradient_contrast_rank_rule "${FMIX}"}) ;;
        gmoment) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening none --restriction_representation none
                                --restriction_gradient_moment "${KAPPA:?give the moment as gmoment/per_example}") ;;
        *) echo "ERROR: unknown retain input treatment ${FILTER}" >&2; exit 1 ;;
    esac
    case "${RANK_MULT}" in
        none) ;;
        qf/*) VARIANT_ARGS+=(--restriction_forget_output_target "${RANK_MULT#qf/}") ;;
        *) VARIANT_ARGS+=(--restriction_rank_multiplier "${RANK_MULT}") ;;
    esac
    [[ "${COMP}" != none ]] && VARIANT_ARGS+=(--component_selection "${COMP}")
    case "${WS_KIND}" in
        none) ;;
        scale) if [[ "${WS_VALUE}" == holdout ]]; then VARIANT_ARGS+=(--warm_start_scale_target holdout)
               else VARIANT_ARGS+=(--warm_start_scale "${WS_VALUE:?give t as scale/t}"); fi ;;
        dirw) VARIANT_ARGS+=(--direction_weighting "${WS_VALUE:?give dirw/contrast}") ;;
        outcontrast) VARIANT_ARGS+=(--output_contrast_rho "${WS_VALUE:?give the ridge as outcontrast/rho}") ;;
        family_input|family_output)
            [[ "${WS_KIND}" == family_input ]] && FAM=input_contrast || FAM=output_contrast
            VARIANT_ARGS+=(--family_contraction "${FAM}" --family_rho "${WS_VALUE:?give the ridge as ${WS_KIND}/rho}") ;;
        global_output|global_gradient|global_input)
            IFS=/ read -r G_RHO G_FLOOR G_ORDER G_ALPHA <<< "${WS_VALUE}"
            [[ -n "${G_RHO}" && -n "${G_FLOOR}" && -n "${G_ORDER}" ]] || { echo "ERROR: give ${WS_KIND}/rho/floor/order[/alpha]" >&2; exit 1; }
            VARIANT_ARGS+=(--global_family "${WS_KIND#global_}_contrast" --global_family_rho "${G_RHO}"
                           --global_family_floor "${G_FLOOR}" --global_family_order "${G_ORDER}"
                           ${G_ALPHA:+--global_family_alpha "${G_ALPHA}"}
                           --global_family_budget_unit "${GLOBAL_BUDGET_UNIT:-energy}"
                           ${GLOBAL_S_GAMMA:+--global_family_s_gamma "${GLOBAL_S_GAMMA}"}
                           ${GLOBAL_S_RULE:+--global_family_s_rule "${GLOBAL_S_RULE}"}
                           ${GLOBAL_S_THRESHOLD_GAMMA:+--global_family_s_threshold_gamma "${GLOBAL_S_THRESHOLD_GAMMA}"}) ;;
        contract_top|contract_random|contract_forget_retain|contract_contrast)
            VARIANT_ARGS+=(--weight_contraction "${WS_KIND#contract_}"
                           --contraction_strength "${WS_VALUE%%/*}")
            [[ -n "${WS_VALUE%%/*}" ]] || { echo "ERROR: give the strength as ${WS_KIND}/s[/residual]" >&2; exit 1; }
            [[ "${WS_VALUE}" == */* ]] && VARIANT_ARGS+=(--contraction_residual "${WS_VALUE#*/}") ;;
        *) echo "ERROR: unknown warm-start variant ${WS_KIND}" >&2; exit 1 ;;
    esac
    [[ "${COMPONENT_DIAGNOSTICS:-}" == yes ]] && VARIANT_ARGS+=(--component_diagnostics yes)
    [[ "${SCORE_DIAGNOSTICS:-}" == yes ]] && VARIANT_ARGS+=(--score_diagnostics yes)
    [[ "${CORE_DIAGNOSTICS:-}" == yes ]] && VARIANT_ARGS+=(--core_diagnostics yes)
    [[ "${CONTRACTION_DIAGNOSTICS:-}" == yes ]] && VARIANT_ARGS+=(--contraction_diagnostics yes)
    [[ "${OUTPUT_DIAGNOSTICS:-}" == yes ]] && VARIANT_ARGS+=(--output_diagnostics yes)
    [[ "${WRITE_COMPENSATION:-}" == retain ]] && VARIANT_ARGS+=(--write_compensation retain)
    [[ "${FAMILY_DIAGNOSTICS:-}" == yes ]] && VARIANT_ARGS+=(--family_diagnostics yes)
    VARIANT_TAGS=("variant_${VARIANT}" "source_${SOURCE}" "evr_${E_TAG}" "align_${SCOPE}" "scaling_${SCALING}" "${ABL_TAG}"
                  "${TAG_BASE}")
    [[ -n "${WS_TAG}" ]] && VARIANT_TAGS+=("ws_${WS_TAG}")
    if [[ "${VARIANT}" == "restricted_semu" ]]; then
        VARIANT_ARGS+=(--phase2_free "${FREE}")
        VARIANT_TAGS+=("phase2_free_${FREE}")
    fi
fi

echo "[launch] host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-none}_${TASK} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[launch] base=${BASE_ID} variant=${VARIANT} profile=${PROFILE} s_min=${S_MIN} epoch_length=${EPOCH_LENGTH} train_dtype=${TRAIN_DTYPE} max_grad_norm=${MAX_GRAD_NORM} out_dir=${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[launch] nvidia-smi unavailable"

cd "${REPO_ROOT}"
echo "[gate] recast_core parity tests, recast_scores, recast_bases and recast_contraction tests (CPU)"
"${PY}" -m pytest -q -p no:cacheprovider replication/tests/test_recast_core.py replication/tests/test_recast_scores.py \
    replication/tests/test_recast_bases.py replication/tests/test_recast_contraction.py \
    replication/tests/test_recast_output_contrast.py replication/tests/test_recast_direction_families.py \
    replication/tests/test_recast_global_families.py

"${PY}" -u replication/scripts/recast_tofu.py \
    --base_id "${BASE_ID}" \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split "${TRAIN_SPLIT}" \
    --holdout_split holdout10 \
    --phase_one_dtype "${PHASE_ONE_DTYPE}" \
    --train_dtype "${TRAIN_DTYPE}" \
    --eval_dtype bfloat16 \
    --attn_implementation sdpa \
    --variant "${VARIANT}" \
    --targets "${PROFILE}" \
    --target_blocks "${BLOCKS}" \
    --untie_lm_head no \
    --train_lm_head yes \
    --s_min "${S_MIN}" \
    --shrink_score "${SHRINK_SCORE}" \
    --grad_batch_size 8 \
    "${VARIANT_ARGS[@]}" \
    --epochs "${EPOCHS}" \
    "${TRAJ_ARGS[@]}" \
    --epoch_length "${EPOCH_LENGTH}" \
    "${RECIPE_ARGS[@]}" \
    --lr "${LR_PEAK}" \
    --weight_decay 0.01 \
    --lr_scheduler piecewise_epochs \
    --lr_points "${LR_POINTS}" \
    "${LOSS_ARGS[@]}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --seed "${SEED}" \
    --eval_batch_size 16 \
    --anchors_raw "${ANCHORS_RAW}" \
    --out_dir "${OUT_DIR}" \
    "${SAVE_ARGS[@]}" \
    --run_name "${RUN_NAME}" \
    --wandb_mode online \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_tags recast_tofu_comparison matched_protocol method_recast \
        model_tofu_llama3p2_1b_instruct split_forget10 "base_${BASE_TAG:-init_fine}" "${VARIANT_TAGS[@]}" "smin_${S_TAG}" \
        "${PHASE_TWO_TAG}" "retain_loss_${LOSS_TAG}" "optim_${OPTIM}" "phase2_recipe_${PHASE_TWO_RECIPE}" "shrink_score_${SHRINK_SCORE}" "lr_peak_${LR_PEAK}" "lr_${LR_NAME}" wd_0p01 "epochs_${EPOCHS}" "epoch_len_${EPOCH_LENGTH}" \
        "phase1_dtype_${PHASE_ONE_DTYPE}" "dtype_${TRAIN_DTYPE}" grad_views_1 "targets_${PROFILE}" "blocks_${BLOCKS}" head_no_shrink_trained lm_head_tied embedding_trained "grad_clip_${CLIP_TAG}" "seed_${SEED}"

echo "[launch] done"
