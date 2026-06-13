# Changelog

All notable changes to **ai-engram** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0, so minor
(`0.x`) releases may include breaking changes.

## [0.6.0] — 2026-06-13

Editing arrives, and statistics become count-aware with a pluggable scaling family.
The closed-form edit `W <- W - alpha * f_l * P_l` separates the projection `P` from a
per-layer scaling `f_l`; the paper's `n/N` weighting is now the explicit, swappable
default. **Breaking** (pre-1.0): the statistics and engram types changed.

### Added
- **`EngramEditor.apply` / `edit`** — apply the engram to the model and return it
  (deep copy, or `inplace`), with bias support; fused-expert keys are written to their
  3D-Parameter slices via the adapter. `edit(target, total, …)` does compute + apply.
- **Pluggable per-layer scaling** (`engram.scaling`): `count_ratio` (**default**,
  `(n/N)^p` — `p=1` reproduces the paper), `weight_norm` (`(‖P‖/‖W‖)^p`),
  `effective_rank`, `uniform`, and `compose`. Or write your own
  `{name: LayerScaleInfo} -> {name: float}`.
- **`Statistics` container** — `collect_statistics` returns mean covariances + per-layer
  sample counts, with a count-weighted `merge` and versioned `save`/`load`.
- **Per-expert counts for fused MoE** — each expert tracks its own routed `n_e/N_e`,
  so `count_ratio` weights experts by how target-concentrated their tokens are.

### Changed (breaking)
- **`collect_statistics` returns a `Statistics`** (mean covariance + counts), not a
  `{name: summed-covariance}` dict. Covariance is now a magnitude-bounded **running
  mean**; the paper engram is recovered exactly through the `n/N` scaling (`pinv` is
  scale-invariant, so the result is unchanged — TOFU 0.998 / 0.706 / 0.817 hold).
- **`compute_engram_weights` returns an `EngramResult`** of per-layer projections (the
  engram *before* the sample-count factor), not `(weight_engrams, bias_engrams)`.
- **`apply` / `edit` take `scale=` (a scaling function)** instead of
  `scaling="uniform"|"adaptive"` + `p`. `count_ratio(1.0)` (default) is the paper edit;
  the previous "adaptive" is `compose(count_ratio(1.0), weight_norm(p))`.
- **Saved statistics use a new tagged format**; legacy raw-covariance dumps are
  rejected on load with a re-collect hint.
- Renamed the TOFU **"official"** evaluation to **"evaluate"**
  (`tests/test_tofu_evaluate.py`, gate `ENGRAM_RUN_TOFU_EVALUATE`).

### Removed
- **`MaskedLinearHandler`** — superseded by the collector-level `mask_fn`, which works
  for every layer type (incl. GPT-2 `Conv1D` and fused MoE experts).

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
