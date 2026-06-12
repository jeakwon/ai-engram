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
from engram import EngramEditor, EditorConfig

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()

editor = EngramEditor(model, EditorConfig(precision=torch.float32))

# accumulate covariance over answer tokens only (labels != -100)
batch_fn = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
mask_fn  = lambda b: b["labels"] != -100

target_cov = editor.collect_statistics(forget_loader, batch_fn=batch_fn, mask_fn=mask_fn)
total_cov  = editor.collect_statistics(total_loader,  batch_fn=batch_fn, mask_fn=mask_fn)
weight_engrams, bias_engrams = editor.compute_engram_weights(target_cov, total_cov)
```

## Applying the edit

!!! note "Milestone 2 preview"
    M1 stops at extraction. Applying the edit is the single subtraction below; M2
    will expose it as `editor.edit(target_cov, total_cov, edit_strength=alpha)`.

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
removal, smaller values are gentler.

## Editing only some layers

Pass `target_modules` — the **same convention as LoRA/PEFT**. A *list* matches by
module-name suffix (across every layer); a *string* is a regex over the full
module path:

```python
# every layer's MLP down_proj
target_cov = editor.collect_statistics(forget_loader, target_modules=["down_proj"], batch_fn=batch_fn)
total_cov  = editor.collect_statistics(total_loader,  target_modules=["down_proj"], batch_fn=batch_fn)
```

To restrict to specific **decoder layers**, use `layers_to_transform`
(+ `layers_pattern`), exactly like PEFT — or fold the index into a regex string:

```python
# layers 20-22 only
editor.collect_statistics(loader, target_modules=["down_proj"],
                          layers_to_transform=[20, 21, 22], layers_pattern="layers")
# equivalent single-layer selection via regex
editor.collect_statistics(loader, target_modules=r".*layers\.5\..*down_proj")
```

## Reusing statistics

Covariances are plain tensors — collect once, save, reuse:

```python
editor.save_statistics(total_cov, "total_cov.pt")
total_cov = editor.load_statistics("total_cov.pt")

# build a total from per-target pieces
total_cov = EngramEditor.merge_statistics(cov_a, cov_b, cov_c)
```
