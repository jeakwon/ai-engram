#!/bin/bash
# Training-free proxies (warm-start output change on forget / retain answer tokens, and the NLL increase
# of the warm-started model on forget, retain and the pre-TOFU sets) for the configurations trained with
# phase two = restricted_semu free_all, constant lr 1e-5, NLL; then their Spearman correlation with the
# trained metrics. Profiles default to down_up_o; gradients and covariances cover down_up_o_v.
# CONFIG_SET=types in --export instead: every decoder projection type alone (q, k, v, o, gate, up, down), plain and
# D at m 1.25, at s_min 0.3 / 0.5 / 0.7 / 0.9 (selectivity by module type and shrink strength; nothing trained).
# CONFIG_SET=evr: D (rho 30, gradient side) with m 1 at EVR 0.98-0.998 on down_proj only and on down + up + o, to find
# the single EVR whose ranks match m 2 / m 1.25 at EVR 0.97 (one budget parameter instead of two).
# Submit from the repository root:
#   sbatch [--export=ALL,CONFIG_SET=types|evr] replication/scripts/bash/recast_tofu/warmstart_proxy_gh200.sh
#SBATCH --job-name=ai-engram-ws-proxy
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/warmstart_proxy.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
OUT="${STORAGE}/runtime_artifacts/ai-engram/recast_tofu"
mkdir -p "${OUT}"
SUMMARY="${OUT}/summary_for_proxy_${SLURM_JOB_ID:-local}.json"

# The first 37 configurations have trained runs (WORK_LOG, 2026-10-22 .. 2026-10-25); the rest are marked.
CONFIGS=(
    "plain_evr97:basis=plain" "plain_evr98:basis=plain,evr=0.98" "plain_smin02:basis=plain,s_min=0.2"
    "plain_smin02_evr98:basis=plain,s_min=0.2,evr=0.98"
    "plain_m1.25:basis=plain,m=1.25" "plain_m1.5:basis=plain,m=1.5" "plain_m2:basis=plain,m=2"
    "C_rho0.01:basis=whiten,rho=0.01" "C_rho0.1:basis=whiten,rho=0.1" "C_rho1:basis=whiten,rho=1"
    "C_rho3:basis=whiten,rho=3" "C_rho10:basis=whiten,rho=10" "C_rho30:basis=whiten,rho=30"
    "C_rho60:basis=whiten,rho=60" "C_rho100:basis=whiten,rho=100" "C_rho300:basis=whiten,rho=300"
    "C_rho30_m1:basis=whiten,rho=30,m=1" "C_rho30_m1.25:basis=whiten,rho=30,m=1.25"
    "C_rho30_m1.5:basis=whiten,rho=30,m=1.5" "C_rho30_m2:basis=whiten,rho=30,m=2"
    "C_rho30_evr98:basis=whiten,rho=30,evr=0.98" "C_rho30_smin02:basis=whiten,rho=30,s_min=0.2"
    "C_rho30_smin02_evr98:basis=whiten,rho=30,s_min=0.2,evr=0.98"
    "D_grad_m1:basis=repr,rho=30,out=gradient,m=1" "D_grad_m1.25:basis=repr,rho=30,out=gradient,m=1.25"
    "D_grad_m1.5:basis=repr,rho=30,out=gradient,m=1.5" "D_identity_m1:basis=repr,rho=30,out=identity,m=1"
    "D_repr_m1:basis=repr,rho=30,out=representation,m=1"
    "D_joint0.5_m1.25:basis=repr,rho=30,out=gradient,mix=0.5,m=1.25"
    "D_seq_m1.25:basis=repr,rho=30,out=whitened_gradient,m=1.25"
    "A_kappa0.1:basis=plain,kappa=0.1" "A_kappa1:basis=plain,kappa=1" "A_kappa10:basis=plain,kappa=10"
    "A_kappa30:basis=plain,kappa=30" "A_kappa100:basis=plain,kappa=100" "A_kappa300:basis=plain,kappa=300"
    "A_kappa1000:basis=plain,kappa=1000"
    # trained profiles with v_proj (25231801)
    "D_grad_m1.25_down_up_v:profile=down_up_v,basis=repr,rho=30,out=gradient,m=1.25"
    "D_grad_m1.25_down_up_o_v:profile=down_up_o_v,basis=repr,rho=30,out=gradient,m=1.25"
    # not trained yet: budget in forget-output units (each target q_f >= qf), down_up_o
    "D_grad_qf0.02:basis=repr,rho=30,out=gradient,qf=0.02" "D_grad_qf0.03:basis=repr,rho=30,out=gradient,qf=0.03"
    "D_grad_qf0.045:basis=repr,rho=30,out=gradient,qf=0.045" "D_grad_qf0.06:basis=repr,rho=30,out=gradient,qf=0.06"
    "D_grad_qf0.08:basis=repr,rho=30,out=gradient,qf=0.08"
    "plain_qf0.04:basis=plain,qf=0.04" "C_rho30_qf0.04:basis=whiten,rho=30,qf=0.04"
    # not trained yet: the budget on the most selective matrix only (down_proj), and down + up
    "D_grad_m1.25_down:profile=mlp_down,basis=repr,rho=30,out=gradient,m=1.25"
    "D_grad_m1.5_down:profile=mlp_down,basis=repr,rho=30,out=gradient,m=1.5"
    "D_grad_m2_down:profile=mlp_down,basis=repr,rho=30,out=gradient,m=2"
    "D_grad_m3_down:profile=mlp_down,basis=repr,rho=30,out=gradient,m=3"
    "D_grad_m4_down:profile=mlp_down,basis=repr,rho=30,out=gradient,m=4"
    "D_grad_qf0.03_down:profile=mlp_down,basis=repr,rho=30,out=gradient,qf=0.03"
    "D_grad_qf0.06_down:profile=mlp_down,basis=repr,rho=30,out=gradient,qf=0.06"
    "D_grad_qf0.1_down:profile=mlp_down,basis=repr,rho=30,out=gradient,qf=0.1"
    "D_grad_m1.25_down_up:profile=mlp_down_up,basis=repr,rho=30,out=gradient,m=1.25"
)
SET="${CONFIG_SET:-trained}"
PROFILE_ARGS=(--targets down_up_o --collect_targets down_up_o_v)
TRAINED_ARGS=(--trained_summary "${SUMMARY}")
if [[ "${SET}" == types ]]; then
    CONFIGS=()
    for t in attn_q attn_k attn_v attn_o mlp_gate mlp_up mlp_down; do
        for s in 0.3 0.5 0.7 0.9; do
            CONFIGS+=("plain_m1.25_${t}_smin${s}:profile=${t},basis=plain,m=1.25,s_min=${s}"
                      "D_grad_m1.25_${t}_smin${s}:profile=${t},basis=repr,rho=30,out=gradient,m=1.25,s_min=${s}")
        done
    done
    PROFILE_ARGS=(--targets decoder_linear --collect_targets decoder_linear)
    TRAINED_ARGS=()
elif [[ "${SET}" == evr ]]; then
    CONFIGS=("D_grad_m2_down:profile=mlp_down,basis=repr,rho=30,out=gradient,m=2"
             "D_grad_m1.25:profile=down_up_o,basis=repr,rho=30,out=gradient,m=1.25")
    for e in 0.98 0.985 0.99 0.993 0.995 0.998; do
        for t in mlp_down down_up_o; do
            CONFIGS+=("D_grad_m1_evr${e}_${t}:profile=${t},basis=repr,rho=30,out=gradient,m=1,evr=${e}")
        done
    done
    TRAINED_ARGS=()
elif [[ "${SET}" != trained ]]; then
    echo "ERROR: CONFIG_SET must be trained, types or evr" >&2; exit 1
fi
CONFIG_ARGS=()
for c in "${CONFIGS[@]}"; do CONFIG_ARGS+=(--config "${c}"); done

echo "[launch] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cd "${REPO_ROOT}"
"${PY}" -u replication/scripts/collect_recast_tofu.py --root "${STORAGE}/experiment_results/ai-engram/recast_tofu" \
    --out "${SUMMARY}" > /dev/null
echo "[launch] trained summary ${SUMMARY}"
"${PY}" -u replication/scripts/warmstart_proxy.py \
    --base_id open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    "${PROFILE_ARGS[@]}" \
    --target_blocks all \
    --attn_implementation sdpa \
    --grad_batch_size 8 \
    "${CONFIG_ARGS[@]}" \
    "${TRAINED_ARGS[@]}" \
    --out_json "${OUT}/warmstart_proxy_${SET}_${SLURM_JOB_ID:-local}.json"
echo "[launch] done"
