# API reference

```python
from engram import (
    EditorConfig, EngramEditor,
    CovarianceCollector,
    LayerHandler, LinearHandler, Conv1DHandler, MaskedLinearHandler,
)
```

## `EditorConfig`

Dataclass of editor settings.

| field | type | default | description |
|---|---|---|---|
| `device` | `torch.device` | cuda if available else cpu | device for the matmul + pseudo-inverse |
| `storage_device` | `torch.device` | cpu | where covariance matrices are accumulated/held |
| `precision` | `torch.dtype` | `torch.float64` | accumulation/solve precision (`float32` for big LLMs) |
| `damping_factor` | `float` | `0.0` | Tikhonov term `Σ_total + λI` for the pseudo-inverse |
| `absorb_bias` | `bool` | `True` | absorb bias into the edit for bias-bearing layers |
| `verbose` | `bool` | `True` | progress bars |

## `EngramEditor`

```python
EngramEditor(model: nn.Module, config: EditorConfig | None = None)
```

On construction, `editor.registry` maps layer types to handlers:
`{nn.Linear: LinearHandler()}`, plus `{Conv1D: Conv1DHandler()}` when
`transformers` is importable. Replace `registry[nn.Linear]` with a
`MaskedLinearHandler()` for answer-token covariance.

### `collect_statistics(dataloader, target_layers=None, batch_fn=None) -> dict[str, Tensor]`

Accumulate input covariance `Σ xᵀx` for each supported layer.

- **dataloader** — any iterable of batches.
- **target_layers** — optional list of module names to restrict to.
- **batch_fn** — maps a batch to model inputs: a tensor, a tuple of positional
  args, or a `dict` of keyword args. Defaults to `batch[0]`.

Returns `{layer_name: covariance[D, D]}` on `config.storage_device`. `D` is the
layer input dim, or `in + 1` when bias absorption applies.

### `merge_statistics(*stats_dicts) -> dict[str, Tensor]` *(staticmethod)*

Sum covariance dicts layer-wise (e.g. to build a total from per-class pieces).

### `compute_engram_weights(target_covariances, total_covariance) -> (dict, dict)`

Compute `W_engram = W · Σ_target · pinv(Σ_total)` per layer.

- **target_covariances** — a covariance dict, or a list of dicts (summed first).
- **total_covariance** — the reference covariance dict.

Returns `(weight_engrams, bias_engrams)`:

- `weight_engrams[name]` — same shape as the layer's `.weight`, in
  `config.precision` on `config.device`.
- `bias_engrams[name]` — same shape as `.bias`; present only for bias-absorbed
  layers (`bias_engrams` is empty for bias-free models).

### `save_statistics(stats, path)` / `load_statistics(path)`

`torch.save` / `torch.load` (with `weights_only=True`, onto
`config.storage_device`) for a statistics dict.

## Handlers

`LayerHandler` is the abstract interface; implement it for custom layer types.

```python
class LayerHandler(ABC):
    def get_input_dim(self, module, absorb_bias=False) -> int: ...
    def reshape_input(self, module, inputs, absorb_bias=False) -> Tensor: ...   # -> [N, D]
    def weight_matrix(self, module, absorb_bias=False) -> Tensor: ...           # -> [out, D]
    def to_weight_shape(self, w, module) -> Tensor: ...                         # [out, in] -> weight.shape
```

| handler | for | weight |
|---|---|---|
| `LinearHandler` | `nn.Linear` | `module.weight` (`[out, in]`) |
| `Conv1DHandler` | HF `Conv1D` (GPT-2) | `module.weight.t()`; result transposed back |
| `MaskedLinearHandler` | `nn.Linear` | as `LinearHandler`; covariance restricted to `current_mask` |

`MaskedLinearHandler.current_mask` — set to a boolean tensor (one entry per
flattened token row, e.g. `labels != -100`) before each forward pass, typically
inside `batch_fn`.

`handler_for(registry, module)` — returns the first registry handler whose type
matches `module`.

## `CovarianceCollector`

Context manager used internally by `collect_statistics`; registers/removes the
forward pre-hooks and exposes `.covariance_matrices`.

```python
with CovarianceCollector(model, config, registry, target_layers) as c:
    with torch.inference_mode():
        for batch in loader:
            model(**batch)
stats = c.covariance_matrices
```
