# Guide

## The method

For a layer computing `y = W x` (`W` of shape `[out, in]`), `ai-engram` localizes
the part of `W` that responds to a *target* input distribution and isolates it as
the **engram weight**:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

where `Σ = Σ_i xᵢ xᵢᵀ` is the (uncentered) input covariance — `Σ_target` over the
data you want to forget, `Σ_total` over the full/reference set.

Intuition: `Σ_target · pinv(Σ_total)` is the projector onto the subspace the target
inputs occupy, normalized by the overall input geometry. `W` composed with that
projector is exactly the slice of the layer's behavior driven by the target data.
Subtracting `α·W_engram` removes that slice and leaves the rest intact.

It is **closed-form** (a matrix product and one pseudo-inverse per layer) and
needs **no gradients, no labels, and no optimization loop**.

## 1 — Collecting covariance

`collect_statistics` registers a `forward_pre_hook` on every supported layer,
flattens the layer input to `[N, D]`, and accumulates `xᵀx` in place:

```python
cov[name] += x.mT @ x      # D×D, on config.storage_device
```

- **Forward-only.** No backward pass is ever run; collection happens under
  `torch.inference_mode()`.
- **Streaming.** Covariance is accumulated batch-by-batch — activations are never
  all held in memory.
- **Covariance placement.** Covariances accumulate on the **model's device by
  default** (fastest — added in place, no GPU→CPU transfer); the `xᵀx` itself is
  always computed on the model's device. Set `storage_device="cpu"` when they
  don't fit in VRAM (see the tip below).
- **Precision (`float32`, fixed).** Accumulation and the closed-form solve run in
  `float32` — deliberately *not* `float64`. On ill-conditioned `Σ_total` (real LLM
  layers reach condition number ~1e13), `float64`'s finer `pinv` cutoff keeps the
  near-null directions and `1/σ`-amplifies them into a catastrophic edit (TOFU
  Overall ~0); `float32`'s coarser cutoff discards them — the implicit
  regularization that makes the edit work.

!!! tip "When to move covariances to CPU"
    Covariances default to the **model's device** and cost `Σₗ Dₗ²` *extra* memory
    (per-layer `D×D`, independent of batch size). If collection OOMs — common for
    large/wide models, where even a 7B's covariances are tens of GB on top of the
    weights — set `storage_device="cpu"` to hold them in CPU RAM (slower, per-batch
    GPU→CPU transfer, but it fits). `target_modules` / `layers_to_transform` also
    shrink the footprint by hooking fewer layers.

### Answer-token masking (LLMs)

For unlearning you usually want covariance over *answer* tokens only, not the
prompt. Pass a `mask_fn` — a `batch -> bool tensor`, one entry per token:

```python
editor.collect_statistics(
    loader,
    batch_fn=lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]},
    mask_fn=lambda b: b["labels"] != -100,          # answer tokens only
)
```

`mask_fn` is applied at the collector, so it works for **every** layer type —
`nn.Linear`, GPT-2 `Conv1D`, and any custom handler. It drops the non-selected
token rows before accumulation (and before the bias-absorption constant, so the
bias term's count equals the number of selected tokens). **MoE models work too**:
a routed expert layer recovers its tokens by matching them back to the router
input, so the mask reaches the experts automatically — no configuration.

!!! note "Legacy"
    `MaskedLinearHandler` (`editor.registry[nn.Linear] = MaskedLinearHandler()`)
    still works but is `nn.Linear`-only — prefer `mask_fn`.

### Selective layers (LoRA convention)

Pass `target_modules` to restrict collection, using the **same convention as
LoRA/PEFT**:

- **list** → match by module-name suffix: `target_modules=["down_proj", "q_proj"]`
  hits those projections in *every* layer.
- **string** → regex over the full module path:
  `target_modules=r".*layers\.5\..*down_proj"` (a single layer).
- **`None`** (default) → every supported layer ("all-linear").

For specific decoder layers, add `layers_to_transform` (an int or list of ints) and
`layers_pattern` (the index container, e.g. `"layers"` / `"h"`), exactly as in PEFT;
it combines with `target_modules` as an AND filter:

```python
editor.collect_statistics(loader, target_modules=["down_proj"],
                          layers_to_transform=[20, 21, 22], layers_pattern="layers")
```

`target_layers=` is kept as a deprecated alias (exact module names still match).

## 2 — Computing the engram

`compute_engram_weights(target_cov, total_cov)` returns
`(weight_engrams, bias_engrams)`. Per layer:

```python
W      = handler.weight_matrix(module)        # canonical [out, in]
engram = W @ Σ_target @ pinv(Σ_total)         # closed form, one pinv per layer
```

- **Pseudo-inverse.** `torch.linalg.pinv` (SVD with rcond thresholding) handles
  rank-deficient `Σ_total` directly — small singular values are cut, not inverted.
- The result is returned in `module.weight`'s shape, so applying the edit is a
  direct subtraction.
- A list of target dicts is summed first (`merge_statistics`), so you can pass
  per-class covariances.

## Bias absorption

A layer with a bias is affine: `y = Wx + b`. In homogeneous coordinates this is
exactly linear:

```
x̃ = [x ; 1]            (dim in+1)
W̃ = [W | b]            ([out, in+1])      ⇒   y = W̃ x̃
Σ̃ = Σ x̃ x̃ᵀ            ((in+1)×(in+1), captures the input mean and count)
W̃_engram = W̃ · Σ̃_target · pinv(Σ̃_total)        → split into  W_engram, b_engram
```

With `absorb_bias=True` (default, **automatic**), bias-bearing layers are handled
this way; the covariance for those layers is `(in+1)×(in+1)` and
`compute_engram_weights` returns a matching `bias_engrams[name]`. Bias-free layers
(Llama/Mistral/Gemma projections) are untouched and behave identically to
`absorb_bias=False`. Set `absorb_bias=False` to edit `W` only.

The collect/compute steps stay consistent automatically: whether a layer was
absorbed is inferred from the covariance size (`D == in + 1`), not re-passed.

## Layer coverage

| layer | handler | notes |
|---|---|---|
| `nn.Linear` | `LinearHandler` | weight stored `[out, in]` |
| HF `Conv1D` | `Conv1DHandler` | GPT-2 family; weight stored `[in, out]`, transposed internally and back |
| masked linear | `MaskedLinearHandler` | covariance over selected tokens |

`Conv1D` is registered automatically when `transformers` is importable. Modern
decoder LLMs use `nn.Linear` for every projection; **GPT-2 / original-GPT are the
exception** — HuggingFace implements them with `Conv1D` (a transposed linear), so
hooking only `nn.Linear` would miss them.

Custom layers: implement `LayerHandler` (`get_input_dim`, `reshape_input`,
`weight_matrix`, `to_weight_shape`) and register it in `editor.registry`.

!!! warning "Not yet supported"

    - **Quantized weights** (4/8-bit, GPTQ, AWQ) — the closed form needs a real
      float weight matrix; load in fp16/bf16/fp32.
    - **`Conv2d` / vision models** — planned.

## Efficiency summary

| technique | where | benefit |
|---|---|---|
| forward pre-hooks | `CovarianceCollector` | no backward pass |
| closed-form solve | `compute_engram_weights` | one `pinv` per layer, no training loop |
| CPU covariance storage | `storage_device` | keeps large `D×D` off the GPU |
| `float32` throughout | (fixed) | coarse `pinv` cutoff regularizes ill-conditioned `Σ` |
| `inference_mode` / `no_grad` | both stages | no autograd overhead |
| selective `target_modules` | collection | edit only what you need (LoRA convention) |
| answer-token masking | `mask_fn` | covariance over relevant tokens only (any layer, incl. MoE) |
