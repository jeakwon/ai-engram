# Changelog

All notable changes to **ai-engram** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0, so minor
(`0.x`) releases may include breaking changes.

## [Unreleased]

### Added
- **`EngramEditor.apply` / `EngramEditor.edit`** — apply the extracted engram to the
  model (`W <- W - scale * W_engram`): `scaling="uniform"` or `"adaptive"`
  (`s_l = alpha*(‖W_e‖/‖W‖)^p`), bias support, and `inplace`. Fused-expert keys are
  written to their 3D-Parameter slices via the adapter. `edit(target, total, …)`
  does compute + apply in one call.

### Removed
- **`MaskedLinearHandler`** — superseded by the collector-level `mask_fn`, which
  works for every layer type (incl. GPT-2 `Conv1D` and fused MoE experts).

### Changed
- Renamed the TOFU **"official"** evaluation to **"evaluate"**
  (`tests/test_tofu_evaluate.py`, gate `ENGRAM_RUN_TOFU_EVALUATE`) — "official"
  misleadingly implied it was the endorsed benchmark eval rather than a reproduction.

## [0.5.0] — 2026-06-13

A correctness + ergonomics release. `EditorConfig` is slimmed from six fields to
two, the numerically dangerous `float64` default is removed, layer selection
follows the LoRA/PEFT convention, and mixture-of-experts — including the
transformers ≥5 fused-expert layout — is supported.

### Fixed

- **`float64` was catastrophic on ill-conditioned covariance.** Real LLM layers
  reach `Σ_total` condition numbers ~`1e13`; `float64`'s fine `pinv` cutoff keeps
  near-null directions and `1/σ`-amplifies them, destroying the edit (TOFU forget10
  Overall ~0 vs `float32`'s 0.706 / 0.817 = paper). The covariance accumulation and
  the closed-form solve now always run in **`float32`**, and `precision` is no
  longer a configurable option.

### Added

- **`target_modules`** layer selection — the **LoRA/PEFT convention**: a list
  matches by module-name suffix (`["down_proj"]`), a string is a regex over the
  full module path. Plus `layers_to_transform` / `layers_pattern` for
  decoder-layer-index selection. (`target_layers` kept as a deprecated alias.)
- **Answer-token masking via `mask_fn`** applied at the collector, so it works for
  every layer type (`nn.Linear`, GPT-2 `Conv1D`, …) — including MoE routed experts.
- **`engram.moe`** — an optional, detachable adapter for transformers ≥5
  **fused-expert** MoE layers: `FusedExpertAdapter` (collect per-expert covariance)
  and `apply_engram_weights` (edit the 3D-Parameter slices). Covers ~35 fused
  architectures (Mixtral, Qwen2/3/3.5-MoE, DeepSeek-V3, GLM4-MoE, MiniMax, Mistral4,
  OLMoE, Phi-MoE, …). The core stays MoE-unaware; without the adapter nothing
  changes.
- **GitHub Actions CI** running `pytest tests/` on every push and PR (against the
  latest `transformers`, which is how the fused-expert layout was caught).

### Changed

- **`EditorConfig` slimmed to `{storage_device, absorb_bias}`.** `storage_device`
  now defaults to the **model's device** (was CPU) — fastest, no per-batch transfer;
  set `"cpu"` for models whose `D×D` covariances don't fit in VRAM.
- **`device` removed** — the compute device is derived from the model
  (`next(model.parameters()).device`), fixing a cuda/cpu mismatch footgun.
- **`verbose` removed** — progress bars auto-detect a TTY (`tqdm(disable=None)`).
- **`damping_factor` removed** — `pinv`'s SVD thresholding is the regularizer.

## [0.4.0] — 2026-06-12

Initial public release: closed-form, forward-only covariance-based engram
extraction (`collect_statistics` → `compute_engram_weights`) with automatic bias
absorption and GPT-2 `Conv1D` support. Reproduces the TOFU forget10 Overall
within ~0.01 of the paper.
