#!/bin/bash
# Build the ai-engram-gh200-ds environment on a GH200 (aarch64) node: a virtual environment over ai-engram-gh200 that
# adds deepspeed and bitsandbytes (replication/environment/requirements_ds_gh200.txt) for phase two as OpenUnlearning
# trained Init Fine. and Retain. ai-engram-gh200 itself is not changed; its packages are pinned as constraints.
# The smoke test trains a tiny Llama through the Trainer with replication's DeepSpeed config and paged_adamw_32bit,
# checks float32 master and bfloat16 module weights, no clipping, a moved model, and gathers the 16-bit weights into
# a plain model as recast_tofu.py does.
# Submit from the repository root:
#   sbatch replication/scripts/bash/environment/build_env_ds_gh200.sh
#SBATCH --job-name=ai-engram-env-ds
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=replication/logs/slurm/%x-%j.out

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: run this on a GH200/aarch64 node, not $(uname -m)." >&2
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/src/engram/__init__.py" ]] || { echo "ERROR: submit from the repository root (no ${REPO_ROOT}/src/engram)." >&2; exit 1; }

STORAGE=/net/storage/pr3/plgrid/plggdnnp
BASE="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PREFIX="${AI_ENGRAM_DS_ENV:-${STORAGE}/conda_envs/ai-engram-gh200-ds}"
REQUIREMENTS="${REPO_ROOT}/replication/environment/requirements_ds_gh200.txt"
DS_CONFIG="${REPO_ROOT}/replication/scripts/bash/recast_tofu/deepspeed_zero3_openunlearning.json"

export PIP_CACHE_DIR="${STORAGE}/pip-cache/${USER}"
export TMPDIR="${STORAGE}/tmp/${USER}/ai-engram-gh200-ds-build-${SLURM_JOB_ID:-local}"
export LD_LIBRARY_PATH="${BASE}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

[[ -x "${BASE}/bin/python" ]] || { echo "ERROR: base environment not found: ${BASE}" >&2; exit 1; }
# AI_ENGRAM_ENV_REPAIR=1 completes an existing prefix (the same idempotent steps) instead of failing.
REPAIR="${AI_ENGRAM_ENV_REPAIR:-0}"
if [[ -e "${PREFIX}" && "${REPAIR}" != "1" ]]; then
    echo "ERROR: target prefix already exists, refusing to overwrite: ${PREFIX}" >&2
    echo "       set AI_ENGRAM_ENV_REPAIR=1 to complete it instead." >&2
    exit 1
fi

echo "[build] host=$(hostname) job=${SLURM_JOB_ID:-none} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[build] base=${BASE} prefix=${PREFIX} repair=${REPAIR}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "[build] nvidia-smi unavailable"

"${BASE}/bin/python" -m venv --system-site-packages "${PREFIX}"
PY="${PREFIX}/bin/python"
# Every package of the base environment keeps its version (editable installs cannot be constraints).
"${BASE}/bin/python" -m pip list --format=freeze --exclude-editable > "${TMPDIR}/base_constraints.txt"
# deepspeed 0.15.4 reads the CUDA version from nvcc under CUDA_HOME when it installs (job 25303661) and, through its
# op compatibility checks, at every import (job 25303744), although it builds no op here. CUDA_HOME alone points it at
# the toolkit; PATH and LD_LIBRARY_PATH stay as they are, so torch keeps its own CUDA 12.6 libraries.
export CUDA_HOME="${AI_ENGRAM_CUDA_HOME:-/net/software/aarch64/el9/CUDA/12.8.0}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || { echo "ERROR: no nvcc under CUDA_HOME=${CUDA_HOME}" >&2; exit 1; }
echo "[build] CUDA_HOME=${CUDA_HOME} ($("${CUDA_HOME}/bin/nvcc" --version | tail -1))"
"${PY}" -m pip install --no-build-isolation -r "${REQUIREMENTS}" -c "${TMPDIR}/base_constraints.txt"
"${PY}" -m pip check || echo "[build] WARNING: pip check reported conflicts (see above)"
"${PY}" -m pip freeze > "${PREFIX}/ai-engram-gh200-ds.freeze.txt"
echo "[build] pip freeze written to ${PREFIX}/ai-engram-gh200-ds.freeze.txt"

echo "[smoke] bitsandbytes self-check"
"${PY}" -m bitsandbytes

echo "[smoke] DeepSpeed ZeRO 3 with bf16 through the Trainer, paged_adamw_32bit, gradient checkpointing"
(
    cd "${TMPDIR}"
    DS_CONFIG="${DS_CONFIG}" "${PY}" - <<'PY'
import copy
import os
import socket
import tempfile

import bitsandbytes
import deepspeed
import torch
import transformers
from deepspeed.runtime.constants import GRADIENT_CLIPPING_DEFAULT
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

print(f"[smoke] deepspeed=={deepspeed.__version__} bitsandbytes=={bitsandbytes.__version__} "
      f"torch=={torch.__version__} transformers=={transformers.__version__}")
assert GRADIENT_CLIPPING_DEFAULT == 0.0, GRADIENT_CLIPPING_DEFAULT
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": "0", "LOCAL_RANK": "0",
                   "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"})

torch.manual_seed(0)
config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
model = LlamaForCausalLM(config).to(torch.bfloat16).to("cuda")
before = {n: p.detach().clone() for n, p in model.named_parameters()}


class Tokens(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        ids = torch.randint(0, 128, (16,), generator=g)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


args = TrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4, gradient_accumulation_steps=2,
                         max_steps=4, learning_rate=1e-3, weight_decay=0.01, optim="paged_adamw_32bit", bf16=True,
                         max_grad_norm=0.0, gradient_checkpointing=True, deepspeed=os.environ["DS_CONFIG"],
                         report_to=[], save_strategy="no", logging_steps=1, seed=0)
assert args.hf_deepspeed_config.config.get("gradient_clipping", 0.0) == 0.0
trainer = Trainer(model=model, args=args, train_dataset=Tokens())
trainer.model_accepts_loss_kwargs = False
trainer.train()
engine = trainer.deepspeed
opt = engine.optimizer
report = {"zero_stage": engine.zero_optimization_stage(), "bf16": engine.bfloat16_enabled(),
          "gradient_clipping": engine.gradient_clipping(),
          "client_optimizer": f"{type(opt.optimizer).__module__}.{type(opt.optimizer).__name__}",
          "master_dtypes": sorted({str(t.dtype) for t in opt.fp32_partitioned_groups_flat}),
          "module_dtypes": sorted({str(p.ds_tensor.dtype) for p in model.parameters()}),
          "master_off_bf16_grid": sum(int((t != t.to(torch.bfloat16).float()).sum()) for t in opt.fp32_partitioned_groups_flat)
          / sum(t.numel() for t in opt.fp32_partitioned_groups_flat)}
print(f"[smoke] {report}")
assert report["zero_stage"] == 3 and report["bf16"] and report["gradient_clipping"] == 0.0, report
assert report["master_dtypes"] == ["torch.float32"] and report["module_dtypes"] == ["torch.bfloat16"], report
assert report["client_optimizer"].startswith("bitsandbytes."), report
assert report["master_off_bf16_grid"] > 0.5, report

state = trainer.accelerator.get_state_dict(engine)
trainer.accelerator.free_memory()
unset_hf_deepspeed_config()
plain = AutoModelForCausalLM.from_config(copy.deepcopy(model.config), torch_dtype=torch.bfloat16)
missing, unexpected = plain.load_state_dict(state, strict=False)
tied = plain.get_output_embeddings().weight.data_ptr() == plain.get_input_embeddings().weight.data_ptr()
assert not unexpected and set(missing) <= {"lm_head.weight"} and tied, (missing, unexpected, tied)
after = dict(plain.named_parameters())
moved = sum(int((after[n].cpu() != before[n].cpu()).sum()) for n in before)
total = sum(p.numel() for p in before.values())
print(f"[smoke] gathered into a plain model (tied head); entries changed {moved} / {total}")
assert moved > 0.5 * total
print("[smoke] DeepSpeed path passed")
PY
)

echo "[smoke] recast_core parity tests in the new environment"
(
    cd "${REPO_ROOT}"
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${PY}" -m pytest -q -p no:cacheprovider replication/tests/test_recast_core.py
)

rm -rf "${TMPDIR}"
echo "[build] done"
