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

## Why

- **Closed-form.** One pseudo-inverse per layer. No optimization loop, no labels.
- **Forward-only.** Covariances are gathered with forward pre-hooks — no backprop,
  half the memory and compute of a training step.
- **HF-native.** Works on the mainstream `nn.Linear` decoders (Llama, Mistral,
  Qwen, Gemma, Phi, …) and the GPT-2 family (`Conv1D`) out of the box.
- **Affine-correct.** Bias absorption is automatic for bias-bearing layers.
- **Tiny.** A few hundred lines; import name `engram`.

## Documentation

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Guide](guide.md) — the method, configuration, efficiency, and design in depth
- [API reference](api.md)
- [TOFU validation](tofu.md) — reproducing unlearning results

## Scope

**Milestone 1 (current):** statistics collection + engram-weight *extraction*.
Applying the edit, a one-call `edit_llm` helper, adaptive scaling, registries,
and eval metrics are on the roadmap (see the README). Extraction already
reproduces TOFU unlearning — see [TOFU validation](tofu.md).

## Install

```bash
pip install "ai-engram[llm]"
```

```python
from engram import EngramEditor, EditorConfig
```
