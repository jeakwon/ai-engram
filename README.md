# ai-engram

[![PyPI](https://img.shields.io/pypi/v/ai-engram.svg)](https://pypi.org/project/ai-engram/)
[![Python](https://img.shields.io/pypi/pyversions/ai-engram.svg)](https://pypi.org/project/ai-engram/)
[![Docs](https://img.shields.io/badge/docs-jeakwon.github.io-7c4dff.svg)](https://jeakwon.github.io/ai-engram/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Minimal, efficient **covariance-based engram extraction** for editing neural
networks — built for HuggingFace causal LLMs.

An *engram* is the component of a layer's weights attributable to a target set
of inputs. `ai-engram` isolates it in **closed form** (no gradient descent),
from forward-only covariance statistics:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

- `Σ_target` — input covariance over the data you want to isolate (the "forget" set)
- `Σ_total`  — input covariance over the reference/total set
- one pseudo-inverse per layer; collection is forward-only (no backprop)
- **bias absorption** is automatic: layers with a bias are edited as the affine
  map `y = Wx + b` via homogeneous coordinates

Subtracting the engram (`W ← W − α·W_engram`) removes the target knowledge while
preserving the rest — the basis of fast, training-free unlearning / model editing.

> **Milestone 1 (this release):** statistics collection + engram-weight
> *extraction*. Applying the edit, a one-call `edit_llm` helper, adaptive
> scaling, model registries, and eval metrics come in later milestones.
> The extraction already reproduces TOFU unlearning — see [Validation](#validation).

## Install

```bash
pip install ai-engram
```

Installs `torch`, `tqdm`, and `transformers` — everything needed for HuggingFace
LLMs (and GPT-2 `Conv1D`) out of the box. The distribution is `ai-engram`; the
**import name is `engram`**.

📖 **Documentation: <https://jeakwon.github.io/ai-engram/>**

## Quickstart

Any `nn.Linear` model:

```python
import torch
from engram import EngramEditor, EditorConfig

editor = EngramEditor(model, EditorConfig(precision=torch.float64))

target_cov = editor.collect_statistics(forget_loader)   # Σ over data to isolate
total_cov  = editor.collect_statistics(total_loader)    # Σ over the reference set

weight_engrams, bias_engrams = editor.compute_engram_weights(target_cov, total_cov)
# weight_engrams[name] has the same shape as the layer's .weight
# bias_engrams[name]   is present only for bias-bearing layers
```

`collect_statistics` reads `batch[0]` by default; for models whose `forward`
takes keyword args, pass a `batch_fn`.

### HuggingFace causal LLMs

Works out of the box for the mainstream `nn.Linear`-based decoders
(Llama, Mistral, Qwen, Gemma, Phi, …) **and** the GPT-2 family, which uses
HuggingFace's `Conv1D` (a transposed linear) — registered automatically.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import EngramEditor, EditorConfig, MaskedLinearHandler

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()

editor = EngramEditor(model, EditorConfig(precision=torch.float32))

# accumulate covariance over answer tokens only (labels != -100)
masked = MaskedLinearHandler()
editor.registry[torch.nn.Linear] = masked

def batch_fn(batch):
    masked.current_mask = batch["labels"] != -100
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

target_cov = editor.collect_statistics(forget_loader, batch_fn=batch_fn)
total_cov  = editor.collect_statistics(total_loader,  batch_fn=batch_fn)
weight_engrams, bias_engrams = editor.compute_engram_weights(target_cov, total_cov)

# applying the edit (Milestone 2 will provide this as `editor.edit(...)`):
import copy
edited = copy.deepcopy(model)
mods = dict(edited.named_modules())
with torch.no_grad():
    for name, w in weight_engrams.items():
        mods[name].weight.data -= (0.6 * w).to(mods[name].weight.dtype)
```

Pass `target_layers=[...]` to `collect_statistics` to edit only specific modules
(e.g. decoder MLPs).

## How it works

| step | what | cost |
|---|---|---|
| 1. collect | forward pre-hooks accumulate `Σ = Σ xᵀx` per layer (no backward) | one forward pass over the data |
| 2. compute | `W_engram = W · Σ_target · pinv(Σ_total)` | one pseudo-inverse per layer |
| 3. apply *(M2)* | `W ← W − α·W_engram` | a single subtraction |

Efficiency is preserved end-to-end: forward-only hooks, in-place covariance
accumulation, CPU/GPU split (covariances on `storage_device`), `float64`
accumulation cast back to the model dtype, `inference_mode`, optional damping,
selective `target_layers`, and answer-token masking for LLMs.

### Layer coverage

| layer | handler | weight orientation |
|---|---|---|
| `nn.Linear` | `LinearHandler` | `[out, in]` |
| HF `Conv1D` (GPT-2) | `Conv1DHandler` | `[in, out]` → transposed internally |
| answer-token masking | `MaskedLinearHandler` | covariance over `labels != -100` |

## Configuration (`EditorConfig`)

| field | default | purpose |
|---|---|---|
| `device` | cuda if available | device for the matmul + pseudo-inverse |
| `storage_device` | cpu | where covariance matrices are accumulated/held |
| `precision` | `float64` | accumulation/solve precision (`float32` for big LLMs) |
| `damping_factor` | `0.0` | Tikhonov term `Σ_total + λI` for the pseudo-inverse |
| `absorb_bias` | `True` | absorb bias into the edit (affine `y=Wx+b`) when a layer has one; bias-free layers unaffected |
| `verbose` | `True` | progress bars |

## Validation

On the **TOFU forget10** unlearning benchmark with `tofu_Llama-3.2-1B-Instruct`
(answer-token-masked covariance, edit applied at α), the package's engram
extraction produces strong, *selective* forgetting — measured by answer-token
NLL (forget should rise, retain should be preserved):

| condition | forget NLL | retain NLL |
|---|---|---|
| base (memorised) | 0.13 | 0.14 |
| plain (α=0.6) | **2.10** (↑) | 0.70 |
| adaptive-norm (α=1.0, p=1) | **2.19** (↑) | **0.59** |

Adaptive-norm forgets *more* while preserving retain *better*.

The full **official 14-metric Overall** (rescaled against the finetuned base and
the `retain90` gold model) reproduces the paper almost exactly:

| condition | ai-engram | paper | diff |
|---|---|---|---|
| gold (retain90) | 0.998 | 0.998 | 0.000 |
| plain (α=0.6) | 0.706 | 0.698 | 0.008 |
| adaptive-norm (α=1.0, p=1) | 0.817 | 0.818 | 0.001 |

All within ~0.01 of the paper — see `tests/test_tofu_official.py` (~24 min on one A100).

See [`tests/`](tests/) and [`examples/`](examples/) for runnable end-to-end code.

## API

- `EngramEditor.collect_statistics(loader, target_layers=None, batch_fn=None) -> {name: Σ}`
- `EngramEditor.merge_statistics(*stats) -> {name: Σ}` *(static; sum to build totals)*
- `EngramEditor.compute_engram_weights(target_cov, total_cov) -> (weight_engrams, bias_engrams)`
- `EngramEditor.save_statistics(stats, path)` / `load_statistics(path)`

Full documentation — **<https://jeakwon.github.io/ai-engram/>** —
[API reference](https://jeakwon.github.io/ai-engram/api/) ·
[Guide](https://jeakwon.github.io/ai-engram/guide/) ·
[TOFU validation](https://jeakwon.github.io/ai-engram/tofu/).

## License

MIT — see [LICENSE](LICENSE).
