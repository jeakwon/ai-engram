# ai-engram

Minimal, efficient **covariance-based engram extraction** for editing neural
networks — built for HuggingFace causal LLMs.

An *engram* is the component of a layer's weights attributable to a target set of
inputs. `ai-engram` isolates it in **closed form** (no gradient descent), from
forward-only covariance statistics:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

Subtracting it (`W ← W − α·W_engram`) removes the target knowledge while
preserving the rest — the basis of fast, training-free unlearning / model editing.

## Install

```bash
pip install ai-engram
```

Installs `torch`, `tqdm`, and `transformers` — HuggingFace LLMs and GPT-2
(`Conv1D`) work out of the box. Import name: `engram`.

## Quickstart

```python
from engram import EngramEditor, EditorConfig

editor = EngramEditor(model, EditorConfig())

target_cov = editor.collect_statistics(forget_loader)   # Σ over data to isolate
total_cov  = editor.collect_statistics(total_loader)    # Σ over the reference set

weight_engrams, bias_engrams = editor.compute_engram_weights(target_cov, total_cov)
```

See the [Quickstart](quickstart.md) for HuggingFace LLMs and answer-token masking.

## Why

- **Closed-form.** One pseudo-inverse per layer. No optimization loop, no labels.
- **Forward-only.** Covariances are gathered with forward pre-hooks — no backprop,
  half the memory and compute of a training step.
- **HF-native.** Works on the mainstream `nn.Linear` decoders (Llama, Mistral,
  Qwen, Gemma, Phi, …) and the GPT-2 family (`Conv1D`) out of the box.
- **MoE-ready.** Answer-token masking reaches mixture-of-experts; transformers&nbsp;≥5
  fused experts are supported via the optional, detachable `engram.moe` adapter.
- **Selective.** Pick layers with `target_modules` — the LoRA/PEFT convention
  (name suffix or regex) — plus `layers_to_transform`.
- **Affine-correct.** Bias absorption is automatic for bias-bearing layers.
- **Tiny.** A few hundred lines; import name `engram`; covariance + solve in `float32`.

## Documentation

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Guide](guide.md) — the method, configuration, efficiency, and design in depth
- [API reference](api.md)
- [TOFU validation](tofu.md) — reproducing unlearning results
