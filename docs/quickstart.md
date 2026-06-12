# Quickstart

The workflow is always the same three calls: **collect** covariance over the
target set, **collect** it over the total/reference set, **compute** the engram.

## Any `nn.Linear` model

```python
import torch
from engram import EngramEditor, EditorConfig

editor = EngramEditor(model, EditorConfig(precision=torch.float64))

target_cov = editor.collect_statistics(forget_loader)   # Σ over data to isolate
total_cov  = editor.collect_statistics(total_loader)    # Σ over the reference set

weight_engrams, bias_engrams = editor.compute_engram_weights(target_cov, total_cov)
```

- `weight_engrams[name]` has the **same shape** as the layer's `.weight`.
- `bias_engrams[name]` has the same shape as `.bias`, and is present **only** for
  bias-bearing layers (empty for bias-free models like Llama).

By default `collect_statistics` reads `batch[0]` from the loader (vision-style
`(x, y)` batches). For models whose `forward` takes keyword arguments, pass a
`batch_fn`.

## HuggingFace causal LLM

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
```

## Applying the edit (Milestone 2 preview)

`ai-engram` M1 stops at extraction; applying the edit is one subtraction:

```python
import copy
edited = copy.deepcopy(model)
mods = dict(edited.named_modules())
with torch.no_grad():
    for name, w in weight_engrams.items():
        mods[name].weight.data -= (alpha * w).to(mods[name].weight.dtype)
    for name, b in bias_engrams.items():
        mods[name].bias.data -= (alpha * b).to(mods[name].bias.dtype)
```

`alpha` (the *edit strength*) controls how much is removed; `1.0` is full
removal, smaller values are gentler. Milestone 2 will expose this as
`editor.edit(target_cov, total_cov, edit_strength=alpha)`.

## Editing only some layers

```python
target_layers = [n for n, m in model.named_modules() if n.endswith("mlp.down_proj")]
target_cov = editor.collect_statistics(forget_loader, target_layers=target_layers, batch_fn=batch_fn)
total_cov  = editor.collect_statistics(total_loader,  target_layers=target_layers, batch_fn=batch_fn)
```

## Reusing statistics

Covariances are plain tensors — collect once, save, reuse:

```python
editor.save_statistics(total_cov, "total_cov.pt")
total_cov = editor.load_statistics("total_cov.pt")

# build a total from per-target pieces
total_cov = EngramEditor.merge_statistics(cov_a, cov_b, cov_c)
```
