# ai-engram

[![tests](https://github.com/jeakwon/ai-engram/actions/workflows/tests.yml/badge.svg)](https://github.com/jeakwon/ai-engram/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-engram.svg)](https://pypi.org/project/ai-engram/)
[![Python](https://img.shields.io/pypi/pyversions/ai-engram.svg)](https://pypi.org/project/ai-engram/)
[![Docs](https://img.shields.io/badge/docs-jeakwon.github.io-7c4dff.svg)](https://jeakwon.github.io/ai-engram/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/jeakwon/ai-engram/blob/main/LICENSE)

**Closed-form, covariance-based engram extraction for editing HuggingFace LLMs** — forward-only, no gradient descent.

An *engram* is the slice of a layer's weights attributable to a target set of inputs. `ai-engram` isolates it analytically:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

`Σ_target` and `Σ_total` are input covariances over the **forget** set and the **reference** set. Subtracting it — `W ← W − α·W_engram` — removes that knowledge while keeping the rest: fast, training-free **unlearning / model editing**.

- **Closed-form** — one pseudo-inverse per layer; no optimization loop, no labels.
- **Forward-only** — covariances via forward pre-hooks; no backprop.
- **HF-native** — Llama, Mistral, Qwen, Gemma, Phi … and GPT-2 (`Conv1D`) out of the box.
- **Affine-correct** — automatic bias absorption for bias-bearing layers.

> **Milestone 1** (this release): statistics collection + engram **extraction**. Applying the edit, a one-call `edit_llm` helper, adaptive scaling, registries, and metrics come in later milestones — and the extraction already reproduces TOFU unlearning (see [Validation](#validation)).

## Install

```bash
pip install ai-engram
```

Pulls `torch`, `tqdm`, and `transformers` — HF LLMs and GPT-2 work out of the box. Distribution name `ai-engram`; **import name `engram`**.

📖 **Documentation: <https://jeakwon.github.io/ai-engram/>**

## Quickstart

Any `nn.Linear` (or GPT-2 `Conv1D`) model:

```python
import torch
from engram import EngramEditor, EditorConfig

editor = EngramEditor(model, EditorConfig())

target_cov = editor.collect_statistics(forget_loader)   # Σ over data to isolate
total_cov  = editor.collect_statistics(total_loader)    # Σ over the reference set

weight_engrams, bias_engrams = editor.compute_engram_weights(target_cov, total_cov)
# weight_engrams[name] matches the layer's .weight; bias_engrams[name] its .bias
```

### HuggingFace LLM (answer-token masked)

```python
from engram import EngramEditor, EditorConfig

editor = EngramEditor(model, EditorConfig())

batch_fn = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
mask_fn  = lambda b: b["labels"] != -100           # covariance over answer tokens only

g_forget = editor.collect_statistics(forget_loader, batch_fn=batch_fn, mask_fn=mask_fn)
g_total  = editor.collect_statistics(total_loader,  batch_fn=batch_fn, mask_fn=mask_fn)
weight_engrams, _ = editor.compute_engram_weights(g_forget, g_total)

# apply — Milestone 2 will expose this as editor.edit(...)
import copy
edited = copy.deepcopy(model)
mods = dict(edited.named_modules())
for name, w in weight_engrams.items():
    mods[name].weight.data -= (0.6 * w).to(mods[name].weight.dtype)
```

Restrict the edit to specific modules with `target_modules` — the same convention
as LoRA/PEFT (`["down_proj"]` by name suffix, or a regex string), plus
`layers_to_transform` for decoder-layer indices. See the
[Quickstart guide](https://jeakwon.github.io/ai-engram/quickstart/) for details.

**Mixture-of-experts.** Answer-token masking reaches the experts automatically on
transformers&nbsp;<5; on transformers&nbsp;≥5 (fused experts) opt in to the
detachable `engram.moe` adapter — `EngramEditor(model, adapters=[FusedExpertAdapter()])` —
covering ~35 fused MoE architectures (Mixtral, Qwen2/3/3.5-MoE, DeepSeek-V3,
GLM4-MoE, MiniMax, Mistral4, OLMoE, Phi-MoE, …).

## How it works

| step | what | cost |
|---|---|---|
| 1. collect | forward pre-hooks accumulate `Σ = Σ xᵀx` per layer | one forward pass, no backward |
| 2. compute | `W_engram = W · Σ_target · pinv(Σ_total)` | one pseudo-inverse per layer |
| 3. apply *(M2)* | `W ← W − α·W_engram` | a single subtraction |

Efficient by construction — forward-only hooks, in-place accumulation, CPU/GPU split (covariances on `storage_device`), a `float32` solve cast back to the model dtype, and answer-token masking. Handles `nn.Linear`, GPT-2 `Conv1D` (a transposed linear), and masked variants; full details in the [Guide](https://jeakwon.github.io/ai-engram/guide/).

### Configuration (`EditorConfig`)

| field | default | purpose |
|---|---|---|
| `storage_device` | model's device | where covariances are held; set `"cpu"` if the `D×D` matrices don't fit in VRAM (large models) |
| `absorb_bias` | `True` | absorb bias into the edit for bias-bearing layers |

## Validation

On **TOFU forget10** with `tofu_Llama-3.2-1B-Instruct`, the engram extraction reproduces the paper's official 14-metric **Overall** within **~0.01**:

| condition | ai-engram | paper |
|---|---|---|
| gold (retain90) | 0.998 | 0.998 |
| plain (α=0.6) | 0.706 | 0.698 |
| adaptive-norm (α=1.0, p=1) | 0.817 | 0.818 |

Answer-token NLL confirms strong, *selective* forgetting — the forget set's NLL jumps ~16× while retain is preserved, and adaptive-norm beats plain on both axes. Runnable end-to-end in
[`tests/`](https://github.com/jeakwon/ai-engram/tree/main/tests) and
[`examples/`](https://github.com/jeakwon/ai-engram/tree/main/examples); see the [TOFU page](https://jeakwon.github.io/ai-engram/tofu/).

## API

- `collect_statistics(loader, target_modules=None, batch_fn=None, mask_fn=None, layers_to_transform=None) -> {name: Σ}`
- `compute_engram_weights(target_cov, total_cov) -> (weight_engrams, bias_engrams)`
- `merge_statistics(*stats)` · `save_statistics(stats, path)` · `load_statistics(path)`

Full reference (auto-generated from docstrings): **[API docs](https://jeakwon.github.io/ai-engram/api/)**.

## License

[MIT](https://github.com/jeakwon/ai-engram/blob/main/LICENSE) © Jeakwon Kim
