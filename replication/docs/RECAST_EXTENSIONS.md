# ReCAST port on TOFU: extensions beyond the porting contract

This document describes what the ReCAST comparison campaign (`recast_tofu_comparison`) added to the port after the
porting contract: what each extension does, its interface (command-line flags, launcher knobs, modules), its tests
and where its results are recorded. It describes code; the chronology, the user decisions, the deviations and every
cited number are in `replication/docs/WORK_LOG.md`. Search the log for the keyword given as **Record** in each
section. The porting contract and its parity tests (`replication/tests/test_recast_core.py`) are unchanged: every
extension is opt-in, and a run without its flags is the contract's ReCAST.

Model and data throughout: Llama-3.2-1B-Instruct fine-tuned on the whole TOFU set (Init Fine.,
`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`), unlearning `forget10` with `retain90` as the retain set, scored
on the Table 3 columns: Memorization (Mem), Model Utility (Util), Privacy (Priv), Exact Memorization (EM),
log10 Forget Quality (FQ) and their Overall.

## Notation

For a target weight W (out x in), on answer tokens unless stated otherwise:

- `G_f`, `G_r`: the mean gradients of the forget and retain answer loss with respect to W.
- `C_f`, `C_r`: the uncentred input covariances `E[x x^T]` of W's input over forget and retain tokens.
- `g_e`: the per-example (per-sequence) gradient of example e; `E[g g^T]` and `E[g^T g]` are the output and input
  partial traces of the sequence-level empirical Fisher.
- Trace normalisation: `A~ = A / tr A`, so `v^T A~ v` is the share of A's energy along a unit direction v. An
  average direction has share 1 / n, n the dimension.
- Contrast pencil with ridge rho: `A~_f v = lambda (A~_r + rho / n I) v`. lambda is the ratio of the forget share to
  the ridged retain share along v. The eigenvectors are orthonormalised by a QR decomposition in lambda order, so the
  leading k columns span the k leading eigenvectors.
- Restricted warm start (the contract): `W* = W + P((S - I) W)`, with `P(X) = U U^T X V V^T`, U and V the
  restriction bases, and `S = diag(s_i)`, `s_i = max(score_i, floor)`. In practice every row sits at the floor, so the
  warm start removes the share `1 - floor` of the block `U U^T W V V^T`.
- Input-side removal (extensions): `W <- W - (1 - s) (W V) V^T`, which removes the share `1 - s` of W's response to
  the input directions V for every output.

## 1. Recipe changes that apply to every run

| Change | Interface | Since |
|---|---|---|
| `restricted_shrink` variant: the restriction applies to the phase-one warm start only, and phase two trains every parameter (no adapter). | `--variant restricted_shrink` | 2026-10-22 |
| Phase two of 5 epochs with the schedule of the TOFU fine-tuning: warm-up to 1e-5 over epoch 1, then linear to 0 at epoch 5. Earlier runs used 10 epochs at a constant rate. | launcher `LR_CONFIGS` | 2026-10-27 |
| Phase one in float32. The weights are float32 while gradients, covariances and the warm start are computed; phase two trains float32 weights with bfloat16 autocast (`fp32_autocast`). | `--phase_one_dtype`, `PHASE_ONE_DTYPE`, `TRAIN_DTYPE` | 2026-10-27 |
| Gradient clipping at norm 1. | `--max_grad_norm`, `MAX_GRAD_NORM` | - |
| Retain-loss normalisation: over the answer tokens of the whole accumulation window, or per micro-batch as OpenUnlearning trained Init Fine. and Retain. | `--loss_normalization` | - |
| DeepSpeed ZeRO 3 phase two (OpenUnlearning's recipe), in its own environment. Implemented but not the default, since it is too slow; returning to it is an open item. | `--deepspeed_config`, `PHASE_TWO_RECIPE=deepspeed_zero3[_clip]`, `replication/environment/requirements_ds_gh200.txt`, `replication/scripts/bash/environment/build_env_ds_gh200.sh`, `deepspeed_zero3_*.json` | 2026-10-28 |
| Rank multiplier fixed at 1: the restriction rank comes from the explained-variance ratio (EVR) of the plain basis alone. | `--restriction_rank_multiplier`, `RANK_MULTS` | 2026-10-29 |
| KL to another teacher: the retain loss `nll_kl` with KL to a given model, e.g. Init Fine. after a plain 5-epoch fine-tuning on retain. | `--kl_teacher`, `KL_TEACHER`, `KL_TEACHER_TAG` | 2026-10-29 |
| Trajectory: the answer NLL and exact memorization of forget, retain_perturbed, real authors and world facts before phase two and after every epoch. | `--eval_trajectory`, `TRAJECTORY=nll` | - |
| Start from a saved model (an Engram edit, a selective edit of section 6, a teacher); with `VARIANTS=(unrestricted)` and `S_MINS=(1.0)` phase one is the identity and the run is phase two alone. | `--base_id`, `BASE_ID`, `BASE_TAG` | - |
| Save the trained model of one seed. | `SAVE_MODEL_SEED` | - |
| Relearning: phase two on the whole TOFU set (forget and retain), with the identity phase one, to measure how fast a finished model recovers. | `TRAIN_SPLIT=full` | 2026-10-29 |

A dash in the last column means the date is not stated here; the log has it. The collector
`replication/scripts/collect_recast_tofu.py` shows the new arguments (phase-one dtype, recipe, shrink score,
contraction, direction gates) with their defaults.

## 2. Stage 1-3: per-type floors, row scores, weight-core contraction

**Record:** `Stage 1`, `Stage 2`, `Stage 3`, `contract_top`, `forget_retain`.

- **Per-type floors (Stage 1).**
  - What it does: `--s_min down_proj=0.2,up_proj=0.5,o_proj=0.5` gives one floor per projection type.
  - Budget matching: `recast_core.update_budget` and `uniform_floor_for_budget` compute the update norm of per-type
    floors and the uniform floor with the same norm.
  - Layer scaling: `--shrink_layer_scaling subspace_norm` scales the floor per target.
  - Tables: `replication/scripts/recast_stage1_tables.py`.
- **Row scores of S (Stage 2, `replication/scripts/recast_scores.py`).**
  - `--shrink_score` selects the score:
    - `original`: `cos(W_i, -G_r,i)`;
    - `projected_row`: the same cosine inside the restriction;
    - `projected_sensitivity`: the cosine form of the first-order retain-loss change of shrinking row i;
    - `global_projected`: one factor per matrix.
  - `--score_diagnostics yes` reports every rule's scores, the first-order change of the applied update and the
    update norm on a floor grid.
- **Weight core against update core (Stage 3).**
  - Analysis: `replication/scripts/recast_bases.py` compares the SVD of `U^T W V` with the SVD of `U^T (S - I) W V`;
    `--core_diagnostics yes` writes it.
  - Contraction (`replication/scripts/recast_contraction.py`): contracts weight-core components until the
    displacement equals ReCAST's at the floor, `(1 - floor) ||P(W)||`.
    - Orders: `top` (largest singular value first), `random`, `forget_retain` (first-order forget-minus-retain loss
      increase per unit of moved energy) and `contrast` (largest input contrast first).
    - Flags: `--weight_contraction`, `--contraction_strength`, `--contraction_residual`.
    - Launcher entries: `contract_top/<s>[/<r>]`, `contract_random/<s>`, `contract_forget_retain/<s>`.
  - Component diagnostics: `--contraction_diagnostics yes` reports kappa, the input contrast and the gradient
    alignment of every component; `replication/scripts/recast_component_diagnostics.py` relates them.
  - Tables: `replication/scripts/recast_contraction_tables.py`.
- **Tests:** `test_recast_scores.py`, `test_recast_bases.py`, `test_recast_contraction.py`, and the per-type floor
  parsing in `test_recast_core.py` (`test_parse_type_values`).

## 3. Output-side replacements and mechanism tests

**Record:** `outcontrast`, `write compensation`, `family_contraction`.

- **Output contrast (`replication/scripts/recast_output_contrast.py`).**
  - Directions: `(W C_f W^T)~ u = lambda ((W C_r W^T)~ + rho / out I) u`.
  - Flag `--output_contrast_rho` (launcher `outcontrast/<rho>`) replaces each update by removing W's output along the
    leading directions, at ReCAST's displacement.
  - `--output_diagnostics yes` reports, for the applied update, the output ratios `||W' a|| / ||W a||` and the
    removed output shares on forget and retain inputs.
- **Write compensation.**
  - What it does: a mechanism test that rescales the residual stream at every o_proj / down_proj write by its retain
    output ratio, stored in the weights.
  - Interface: `--write_compensation retain`, `WRITE_COMPENSATION=retain`.
  - Code: `recast_core.residual_write_divisors`, `apply_write_divisors`.
- **Direction families (`replication/scripts/recast_direction_families.py`).** Three families of forget-specific
  directions per target, compared at the same removed forget output:
  - output contrast;
  - gradient contrast (`G G^T` on the output side);
  - input contrast (`C_f` against `C_r`).

  Interface: `--family_diagnostics yes` compares them; `--family_contraction output_contrast|input_contrast` with
  `--family_rho` replaces the update by one family.
- **Tests:** `test_recast_output_contrast.py`, `test_recast_direction_families.py`.

## 4. Global selection over all targets

**Record:** `global families`, `global_order_summary`, `s_gamma`, `retain_share`.

`replication/scripts/recast_global_families.py`, flags `--global_family*`, launcher
`global_output|global_gradient|global_input/<rho>/<floor>/<order>[/<alpha>]`:

- **Ordering.** The directions of every down_proj and o_proj target enter one global order. Available orders:
  - `score`: lambda, highest first;
  - `random`: the control;
  - `energy_ratio`: the absolute output energies of each direction alone, `e_f / e_r^alpha`.
- **Removal.** Each chosen direction keeps the share `floor` of its part.
- **Stopping.** Selection continues until the forget output energy removed over all targets equals that of the
  ReCAST updates at the floor.
- **Budget units** (`GLOBAL_BUDGET_UNIT`):
  - `energy`: absolute energy; down_proj and o_proj write into the same residual stream, so their energies share
    one unit;
  - `share`: each target's removed share of its own forget output energy, which admits any projection type.
- **Per-direction strengths** (replace the single floor):
  - `GLOBAL_S_GAMMA`: `clip(lambda^-gamma, floor, 1)`;
  - `GLOBAL_S_RULE=retain_share`: `clip(e_r / (e_f + e_r), floor, 1)`;
  - `GLOBAL_S_THRESHOLD_GAMMA`: `clip((lambda* / lambda)^gamma, floor, 1)`, with one threshold lambda* found by
    bisection to meet the budget.
- **Order summary:** `replication/scripts/global_order_summary.py` (standard library) summarises the runs by order.
- **Tests:** `test_recast_global_families.py`.

The best run of the campaign so far uses this machinery: the input contrast (rho 1) over down_proj and o_proj, `score`
order, budget floor 0.3 (section 8).

## 5. Gradient-based restriction bases

**Record:** `gradient contrast`, `gcontrast`, `per-example`, `own-spectrum`, `kept_energy`.

### 5.1 Forget-versus-retain gradient contrast (2026-11-01)

`recast_core.gradient_contrast_basis`, flag `--restriction_gradient_contrast both|input` (ridge rho =
`--whitening_rho`), launcher `FILTERS=(gcontrast/<rho>/<sides>[/<rule>])`.

- **Bases.**
  - Input side: V from the pencil `(G_f^T G_f)~ v = lambda ((G_r^T G_r)~ + rho / in I) v`.
  - Output side (`both`): U from the same pencil for `G G^T`.
  - `input`: only the input side changes, with the plain basis's output side as the control.
- **Rank rules.**
  - **Matched rank:** `--restriction_rank_multiplier 1` (`RANK_MULTS=(1.0)`) gives both sides the rank of the plain
    basis at `--restriction_evr`.
  - **Own-spectrum ranks** (2026-11-02), without a rank multiplier and for `restricted_shrink` only; selected with
    `--gradient_contrast_rank_rule`, the fourth launcher field:
    - `side_energy` (default): each side takes the lambda-ordered directions until they hold `--restriction_evr` of
      `||G_f||^2` on that side;
    - `kept_energy`: one rank r for both sides, the smallest with
      `||U_r U_r^T G_f V_r V_r^T||^2 / ||G_f||^2 >= --restriction_evr`.
  - For the plain basis all three rules reduce to its EVR. `recast_core.share_rank` computes the cumulative-share
    rank.
- **Report:** `[gcontrast]` and `results["gradient_contrast"]` give the median ranks per side, the kept forget-gradient
  energy per type, and the rank rule.

### 5.2 Per-example gradient moments (2026-11-02)

`recast_core.collect_example_gradient_moments`, `gradient_moment_basis`; flag
`--restriction_gradient_moment per_example`; launcher `FILTERS=(gmoment/per_example)`.

- **What it computes.** U and V are the leading eigenvectors of the forget per-example moments `E[g g^T]` and
  `E[g^T g]`: the partial traces of the sequence-level empirical Fisher of the forget set, with no retain contrast.
- **How.** The moments are collected with forward hooks and `autograd.grad` at the target outputs. g_e is B times the
  example's share of its batch's mean-loss gradient, so the mean of the g_e equals `G_f` exactly.
- **Rank.** Matched rank with a rank multiplier, else each side's own EVR rank of its moment (the input rank may
  exceed min(out, in)).
- **Report:** `[gmoment]`, `results["gradient_moment"]`.

### 5.3 Tests

`test_recast_core.py`:
- the per-example moments match per-example gradients on a sequence toy model;
- the moment basis keeps directions that cancel in the mean gradient;
- `share_rank`;
- both bases' own-spectrum ranks, including the `kept_energy` curve on a diagonal toy (ranks 1 / 2 / 3 at shares
  0.05 / 0.5 / 0.9995);
- the warm start reproduces `W + P((S - I) W)` with each basis;
- invalid flag combinations.

## 6. Selective input contrast with retain protection (2026-11-03)

**Record:** `selective`, `recast_selective`, `massive-activation hypothesis`.

### 6.1 Motivation

Three findings led to this rule.

1. **The loss yardstick disagrees with the energy yardstick on down_proj in block 1.**
   - Measurement: removing input directions and measuring the real change of the answer negative log-likelihood (NLL)
     by inference on 400 forget and 400 retain examples.
   - Energy yardstick (activation energy over answer tokens): down_proj in block 1 is the most specific block.
   - Loss yardstick: it is the least specific (forget and retain NLL rise almost equally).
2. **Bases from forget statistics alone have no loss specificity.** The plain basis and the per-sequence Fisher of
   the forget set raise forget and retain NLL equally.
3. **The contrast ratio and a budget both misrank directions.**
   - lambda is a ratio, so a direction strong in both sets (forget share 100, retain share 25) ranks like a weak one
     (4, 1).
   - A budget-driven rank keeps removing shared directions in a target whose selectivity has ended. This is how
     `kept_energy` pushed several blocks to full rank.

**Hypothesis, confirmed by job 25591062:**
- The answer-token statistics cannot see the constant prompt prefix.
- An edit of the weights acts on every position, including that prefix.
- In block 1, down_proj produces the massive activation of the BOS token. Its input neuron 1417 reaches 1162 at BOS,
  against a median of 0.006, and writes residual dimension 400 to about 410. That value stays through block 15,
  while prompt and answer tokens reach at most 1-9.
- Such massive activations make BOS the model's attention sink (Sun et al. 2024; Xiao et al. 2023): the token on
  which heads park attention they do not need.
- Removing the old activation-contrast directions of down_proj block 1:
  - at the prefix positions only: forget and retain NLL rise alike (+1.13 / +1.15);
  - everywhere else only: +0.24 / +0.01, as specific as any target.
- Down_proj block 0 shows the same split on a smaller scale. Other blocks are unaffected by the prefix.

### 6.2 The rule (`replication/scripts/recast_selective.py`)

Per target, the removed input directions V are chosen in three steps.

1. **Retain protection.** V stays orthogonal to a protected span B, made of two parts.
   - **Prefix span (`prefix`).**
     - The prefix: the chat template's leading tokens are identical in every example. These are 36 tokens: BOS, the
       system header with the dates and the system prompt, `<|eot_id|>`, and the user header.
     - Why they form one fixed set: attention is causal, so their inputs to every target are one fixed set of
       vectors, the same for forget and retain (equal across sequences within 1e-5).
     - Effect: with their span protected, the warm start leaves the whole prefix computation exactly unchanged in
       every layer, including the BOS massive activations. By induction: prefix positions only attend to prefix
       positions, and every edited weight acts unchanged on their inputs.
     - At most 36 dimensions per target.
   - **Kappa protection (`kappa`).**
     - Protected directions: the eigenvectors of `C_r / tr C_r` with eigenvalue above `kappa / n`. Here `C_r` is
       taken over every non-prefix token, question and answer, i.e. every token the edit acts on.
     - Guarantee: by the Rayleigh bound, every direction left has a retain share of at most kappa times the average
       share 1 / n.
     - kappa is a hyperparameter of this rule; leaving it out disables this part.
     - Scale: `C_r` averages over the 195 713 non-prefix tokens of retain90, and its trace is the mean input energy
       `||x||^2` per token. The average share 1 / n is 1 / 2048 for o_proj and 1 / 8192 for down_proj.
     - At kappa 4 (job 25591665), the protected directions are:
       - o_proj: 47-93 directions (2-5% of the dimension), holding 59-74% of the retain energy;
       - down_proj: 127-324 directions (1.5-4%), holding 49-76%.
2. **Selectivity stop.**
   - The pencil: in the complement, `(P A~_f P) v = lambda (P A~_r P + rho / n I) v` with `P = I - B B^T`.
   - The traces are those of the full statistics, so lambda keeps the units of the unprotected contrast.
   - Selected: every direction with `lambda >= lambda*`, one threshold for all targets and no budget. A target
     without selective directions loses none.
   - The QR columns need no per-column eigenvalue. They span the selected eigenvectors, the removal depends only on
     that span, and each column's Rayleigh quotient lies in `[lambda*, lambda_1]`.
3. **Loss guard** (driver, optional), per target:
   - Edit that target alone at the training floor.
   - Measure the forget and retain answer-NLL changes `dF`, `dR` by inference.
   - While `dF / dR < --guard_ratio`, keep the most selective half of the directions and measure again, at most
     `--guard_steps` times; if it still fails, the target gets no edit.

The edit is `W <- W - (1 - s) (W V) V^T`, input side, floor s.

Statistics:
- `collect_position_statistics` collects the covariances per position class: `prefix`, `prompt`, `answer`,
  `nonprefix`.
- Classes are positional. The OpenUnlearning collator pads with the EOS id (`<|eot_id|>`), so its attention mask
  also masks the template's own `<|eot_id|>` tokens.
- The families are `act` (C_f against C_r) and `fisher` (the per-sequence Fisher input moments, forget against
  retain).

Which tokens each stage uses (S2):

| Stage | Forget | Retain | Prefix |
|---|---|---|---|
| Phase one, the directions | answer tokens | question and answer tokens | left out of the statistics, its span protected |
| Phase one, the measured NLL change | answer tokens | answer tokens | - |
| Phase two (5 epochs on retain90) | not used | loss on answer tokens only | - |

- In phase two the question and the prefix carry label -100. The forward pass covers the whole sequence, so the
  gradient still reaches the weights through the question and prefix representations.
- The gradient families (plain, gradient contrast, per-example, Fisher) take the loss on answer tokens. Their
  gradients still sum contributions from every position, the prefix included.

### 6.3 Driver and launcher

`replication/scripts/recast_selective_tofu.py`, launcher `replication/scripts/bash/recast_tofu/recast_selective_gh200.sh`.
Each job first runs the CPU tests of the rule.

**`MODE=diagnose`** (no training, about 2 h). It runs three parts.
- **Hypothesis checks:**
  - the largest |activation| of the residual stream and of the down_proj input at each prefix position, on prompt
    tokens and on answer tokens;
  - the prefix inputs' energy along the old activation-contrast directions;
  - removal of those directions at prefix positions only, elsewhere only, or everywhere, for down_proj in blocks 0-3
    and o_proj in blocks 0-1.
- **Selection grid:**
  - 11 pencils:
    - `act` with answer-token retain statistics;
    - `act` with non-prefix retain statistics;
    - the latter plus the prefix span;
    - the latter plus kappa 1, 4 or 16;
    - `fisher` without protection, with the prefix span, and with the prefix span plus kappa 1, 4 or 16;
  - lambda* in {1, 1.5, 2, 4, 8};
  - the joint edit of every target at floors 0 and 0.3, measured by inference.
- **Every target's edit alone**, at one setting per family.

Output: `experiment_results/ai-engram/recast_selective/diagnose_<job>.json`.

**`MODE=edit`** builds one setting, applies the loss guard, measures the joint edit, and saves the edited model in
float32, its tokenizer and `selective_directions.pt`.
- Knobs: `FAMILY`, `LAMBDA_STAR`, `FLOOR`, `KAPPA` (unset = off), `PREFIX` (default `yes`), `RETAIN_TOKENS` (default
  `nonprefix`), `GUARD_RATIO` (default 2, 0 = off). The restricted families of section 6.5 take `EVR` (and, for
  `gcontrast`, `RHO` and `SIDES`) in place of `LAMBDA_STAR` and run without the guard.
- Model directory: `runtime_artifacts/ai-engram/recast_selective/models/<TAG>_<job>`.

**Phase two** runs from that model with `recast_tofu_gh200.sh`:
- `BASE_ID=<model dir>`, `BASE_TAG=<TAG>`;
- `VARIANTS=(unrestricted)`, `S_MINS=(1.0)`, `FILTERS=(none)`, so phase one is the identity;
- the current recipe, every parameter trained;
- submitted with `--dependency=afterok:<edit job>`.

**Tests:** `test_recast_selective.py` (14 tests) covers:
- prefix length and position masks (including masked in-template EOS tokens);
- the collected covariances against a manual computation;
- the kappa bound and the prefix span;
- equality with `ordered_directions` without protection;
- the (100, 25) direction against the (4, 1) direction;
- orthonormality and orthogonality to B, and every column's Rayleigh quotient at least lambda*;
- the input shrink;
- the energy report;
- the restricted warm starts of section 6.5 (5 tests).

### 6.4 Results (seed 0)

**Warm-start grid** (diagnose; joint edit at floor 0.3; dF / dR in nats, ratio in brackets):

| Pencil | lambda* 1 | lambda* 2 |
|---|---|---|
| act, non-prefix retain statistics | 3.51 / 2.04 (1.7) | 1.00 / 0.27 (3.7) |
| + prefix span | 2.45 / 0.73 (3.4) | 0.76 / 0.05 (14.4) |
| + prefix span + kappa 4 | 1.39 / 0.23 (6.0) | 0.48 / 0.02 (29) |
| Fisher contrast (with or without the prefix span) | 1.27 / 0.33 (3.9) | 0.69 / 0.08 (8.3) |

- The prefix span matters for the activation contrast.
- It does not matter for the Fisher contrast: the prefix sits equally in the forget and retain Fisher.
- At equal dF the protected activation contrast is the more specific one.

**After phase two** (activation contrast, current recipe). Format: Mem / Util / Priv / EM / FQ / Overall.

The variants are named S0-S4. The log used the letters D, E, F, C, A for them until 2026-11-04; those collided
with the campaign's bases C and D. Floor 0.3, lambda* 1, one change per step:

| Variant | Warm start dF / dR | After phase two |
|---|---|---|
| S0: lambda* stop, answer-token statistics | +2.83 / +1.40 | 0.9290 / 0.9582 / 0.7748 / 0.1332 / -0.01 / 0.8796 |
| S1: S0 + prefix span | +2.07 / +0.44 | 0.9156 / 0.9581 / 0.7485 / 0.1433 / -0.06 / 0.8641 |
| S2: S1 + retain over non-prefix tokens | +2.45 / +0.73 | 0.9599 / 0.9747 / 0.8166 / 0.0992 / -0.03 / 0.9112 |
| S3: S2 + guard 2 | +2.28 / +0.62 | 0.9335 / 0.9757 / 0.7812 / 0.1160 / -0.04 / 0.8886 |
| S4: S3 + kappa 4 | +1.37 / +0.23 | 0.8927 / 0.9771 / 0.5852 / 0.2011 / -0.61 / 0.7787 |

Stronger edits:

| Variant | Warm start dF / dR | After phase two |
|---|---|---|
| S2, floor 0.2 | +2.98 / +1.02 | 0.9770 / 0.9547 / 0.9270 / 0.0267 / -0.81 / 0.9524 |
| S2, floor 0.25 | +2.71 / +0.87 | 0.9692 / 0.9597 / 0.8719 / 0.0615 / -0.12 / 0.9315 |
| S3, floor 0.2 | +2.79 / +0.88 | 0.9679 / 0.9597 / 0.9046 / 0.0530 / -0.44 / 0.9432 |
| S4, lambda* 0.75, floor 0.2 | +2.23 / +0.63 | 0.9659 / 0.9636 / 0.9019 / 0.0491 / -0.33 / 0.9429 |
| S3, floor 0.1 | +3.34 / +1.21 | 0.9717 / 0.9331 / 0.8496 / 0.0296 / -0.81 / 0.9152 |
| S3, lambda* 0.75, floor 0.2 | +3.40 / +1.37 | 0.9742 / 0.9345 / 0.8269 / 0.0346 / -0.61 / 0.9075 |
| S4, floor 0.1 | +1.97 / +0.44 | Overall 0.9028 (Priv 0.8088) |
| S4, floor 0.2 | +1.67 / +0.32 | Overall 0.8556 (Priv 0.7133) |

Reference: the best global selection on the same seed has 0.9881 / 0.9231 / 0.9333 / 0.0097 / -1.03 / 0.9473.

Seeds 0-2 (phase two repeated from the saved edit; the edit does not depend on the seed), means:

| Variant | Scores | sd of Overall |
|---|---|---|
| S2, floor 0.2 | 0.9809 / 0.9552 / 0.9189 / 0.0293 / -0.49 / 0.9510 | 0.0019 |
| S3, floor 0.2 | 0.9715 / 0.9620 / 0.8962 / 0.0516 / -0.18 / 0.9420 | 0.0011 |
| S4, lambda* 0.75, floor 0.2 | 0.9650 / 0.9624 / 0.8935 / 0.0524 / -0.24 / 0.9391 | 0.0054 |
| Global selection (6 seeds, section 4) | 0.9824 / 0.9247 / 0.9330 / 0.0109 / -0.64 / 0.9460 | 0.0048 |

Reading (single seeds, except the table above):
- The retain statistics over question and answer tokens carry the gain (S1 -> S2).
- The guard and kappa weaken the edit at a fixed floor, so they need a lower floor or more directions.
- **Why the guard does not help after phase two.** It cut the same 8 of 32 targets at floors 0.3 and 0.2: o_proj in
  blocks 2, 6, 7, 8, 10, 12 and 15, and down_proj in block 15.
  - Each of them, edited alone, had a ratio `dF / dR` of 1.7-1.97: below the threshold 2, but still forget-favouring.
  - At floor 0.2 the guard removed 497 of 10974 directions. The warm-start ratio rose from 2.9 to 3.2 (dF 2.98 -> 2.79,
    dR 1.02 -> 0.88).
  - After phase two, Utility is equal (0.955 / 0.960) and Privacy is lower (0.927 -> 0.905). Phase two retrains the
    retain answers and repairs the retain damage the guard prevented, while the forgetting it gave up stays lost.
  - The guard was built for targets that harm both sets, like unprotected down_proj block 1 (ratio 1.2, +1.2 nats on
    retain). With the prefix span protected no such target is left.
- S2 at floor 0.2, with no guard, is the best rule of the campaign: 0.9510 over seeds 0-2, +0.005 over the global
  selection (n 6), with Utility +0.031 and Privacy -0.014 (means by hand; no test run).
- Privacy falls again when the edit is too strong (S3 at floor 0.1; S3 at lambda* 0.75): the forget NLL overshoots the
  Retain model's.
- Over three seeds, S3 at floor 0.2 trails the global selection by 0.004 in Overall, with Utility +0.037 and Privacy
  -0.037.

### 6.5 Protection for the restricted warm starts (2026-11-04)

`recast_selective.plain_restricted_update`, `contrast_restricted_update` and `moment_restricted_update`; driver and
launcher `FAMILY=plain`, `gcontrast` or `gmoment` with `EVR` (no lambda*, no guard).

- **Bases**, each computed as `recast_tofu.py` computes it, so that without protection the warm start is that of the
  corresponding `recast_tofu.py` run:
  - `plain`: the SVD of the mean forget gradient (the contract);
  - `gcontrast`: the forget-vs-retain gradient contrast (section 5.1) at the plain basis's rank (`--restriction_rank_multiplier
    1`), with `RHO` (default 1) and `SIDES` (default `both`);
  - `gmoment`: the per-example forget-gradient moments (section 5.2), each side at its own EVR rank.
- **What is projected.** `plain` projects the mean forget gradient. `gcontrast` projects both mean gradients, so the
  pencil's directions with lambda > 0 lie outside B. `gmoment` projects every per-example gradient: the output moment
  becomes `E[g P g^T]` (collected by `collect_example_moments`) and the input moment `P E[g^T g] P`.
- **Control in the same job.** With protection every target also reports the unprotected update: ranks, kept
  forget-gradient energy, `|dW| / |W|` and the prefix output change. These are compared with the per-target report of the
  old `recast_tofu.py` run.

The plain family in detail:

- **Base computation.** The contract's `restricted_shrink` warm start with the plain basis, computed as in
  `recast_core.warm_start`:
  - `svd_basis` of the mean forget gradient at the EVR;
  - every row at `max(cos(W_i, -G_r,i), floor)`;
  - the two-sided projection.
- **Protection.** The forget gradient's input side is first projected off the protected span: `G_f (I - B B^T)`.
  - Vh is then orthogonal to B, so the update leaves `W x` unchanged for every x in B.
  - B is the prefix inputs' span, plus, with `KAPPA`, the retain input directions above kappa / n. Here the retain
    statistics are taken over the question and answer tokens.
- **Rank.** It comes from the EVR of the projected gradient.
- **Control.** Without protection the edit must repeat the warm start of a `recast_tofu.py` run: the per-target ranks
  and relative weight changes.
- **Tests:** equality with the `recast_core` formula, and unchanged outputs on the prefix inputs, with the unprotected
  update as the positive control.

**Results** (EVR 0.99, floor 0.2, seed 0; warm start dF / dR, then Mem / Util / Priv / EM / FQ / Overall):

| Variant | Warm start | After phase two |
|---|---|---|
| P0: no protection (control) | +10.0 / +10.0 | 0.9356 / 0.8338 / 0.8732 / 0.0661 / -2.00 / 0.8789 |
| P1: prefix span | +13.7 / +13.3 | 0.9804 / 0.8401 / 0.9151 / 0.0304 / -0.33 / 0.9082 |
| P2: prefix span + kappa 16 | +3.5 / +2.7 | 0.9818 / 0.8304 / 0.9098 / 0.0401 / -0.55 / 0.9031 |
| P3: prefix span + kappa 4 | +2.1 / +1.4 | 0.9457 / 0.8938 / 0.8047 / 0.1103 / -1.19 / 0.8775 |

P1 over seeds 0-2 (phase two repeated from the saved edit): 0.9786 / 0.8056 / 0.9156 / 0.0403 / -0.41 / 0.8939, sd of
Overall 0.0131. Plain without protection over the same seeds: 0.8756, so +0.018, with Privacy +0.046 and Utility -0.017.

- P0 repeats the original `recast_tofu.py` run exactly: every target's rank and relative change, and all six scores.
- The mean forget gradient lies largely in the span of the 36 prefix inputs. After the projection only 16-64% of its
  energy remains in most targets, and 0.4% in down_proj block 1, whose plain basis is rank 1, the BOS direction.
- Unprotected, plain ReCAST changes the prefix outputs by 30-60% in most targets.

**All three families with the prefix span** (seed 0; warm start dF / dR, then Mem / Util / Priv / EM / FQ / Overall, and
the unprotected cell for comparison):

| Family, floor | Warm start | After phase two | Unprotected |
|---|---|---|---|
| plain, EVR 0.99, floor 0.1 | +14.6 / +14.4 | 0.9637 / 0.7029 / 0.9306 / 0.0315 / -0.09 / 0.8487 | - |
| plain, floor 0.2 (P1) | +13.7 / +13.3 | 0.9804 / 0.8401 / 0.9151 / 0.0304 / -0.33 / 0.9082 | 0.8789 |
| plain, floor 0.3 | +11.9 / +11.5 | 0.9327 / 0.8775 / 0.8035 / 0.1045 / -1.80 / 0.8680 | 0.8358 |
| gradient contrast, matched rank at EVR 0.995, floor 0.1 | +3.3 / +2.5 | 0.9799 / 0.8068 / 0.9363 / 0.0273 / -2.00 / 0.9014 | 0.9160 |
| gradient contrast, floor 0.2 | +2.6 / +1.8 | 0.9556 / 0.8856 / 0.8524 / 0.0774 / -1.80 / 0.8958 | 0.8997 |
| per-example, own EVR 0.99, floor 0.4 | +11.5 / +11.2 | 0.9752 / 0.7960 / 0.8895 / 0.0531 / -0.19 / 0.8808 | 0.8897 |
| per-example, floor 0.5 | +7.2 / +6.8 | 0.8795 / 0.9062 / 0.7064 / 0.1713 / -2.00 / 0.8205 | 0.811 |

- The control holds: the unprotected updates computed in the same jobs repeat the old runs. Ranks are equal in all 32
  targets, and `|dW| / |W|` agrees within 6e-8 for the gradient contrast and 2e-9 for the per-example basis.
- With protection, the prefix outputs move by at most 5e-7. Unprotected, they moved by up to 37% (gradient contrast),
  56% (per-example) and 69% (plain).
- The prefix span helps plain. Its best floor stays at 0.2.
- It does not help the gradient contrast (-0.015 at its best floor) or the per-example basis (-0.009). Two things
  changed at once in those runs: the projected source raised the smallest per-target ranks from 2 / 437 to 582 / 898.
  Like the Fisher contrast in the diagnose grid, the gradient contrast already discounts what forget and retain share.

### 6.6 Open items (user 2026-11-04)

- **The gradient contrast in the new rule.** S2's rule (lambda* stop, prefix span, no guard, input-side removal) with V
  from the pencil of `G_f^T G_f` against `G_r^T G_r`, both projected off the prefix span.
- **The Fisher contrast in the new rule with phase two.** The driver supports it (`FAMILY=fisher`, `PREFIX=yes`,
  `GUARD_RATIO=0`). It was only measured in the diagnose grid so far.
- **Family specificity in the new rule.** Compare the activation, Fisher and gradient contrasts, with plain and
  per-example as references without a contrast, under one protection and one stop:
  - per target and jointly, at matched warm-start dF: `dF / dR` by inference, the lambda spectra, the overlap of the
    selected spans;
  - then after phase two.
- The gradient and Fisher families see every position through the backward pass. S2's choice of retain tokens
  (question and answer) has its counterpart in the retain loss on those tokens (`--retain_loss_tokens nonprefix`,
  2026-11-05); the first two items run in that form and with the answer-only retain loss (WORK_LOG 2026-11-05 00:38,
  01:20).
- **The ridge rho in S2 (TODO, user 2026-11-05).** S2 so far ran only at rho 1. Check how S2 (act, lambda* 1, floor
  0.2, no guard) depends on rho, below 1 (0.1, 0.01) and above (e.g. 3, 10):
  - In share units `lambda = F / (R + rho / n)`, so rho changes only the directions whose retain share is below about
    rho / n. A smaller rho promotes them, and admits retain-free directions with a forget share down to
    `lambda* rho / n`. For those, only the product `lambda* rho` matters.
  - The earlier evidence against a small rho comes from rules with a budget or an EVR: basis C, Overall 0.557 at rho
    0.01 and 0.893 at 30; the Rayleigh minimum of the retain cost at rho 0.3-1. The lambda* rule has no budget.
  - Report per setting the direction counts, `dF / dR` at floors 0-0.3, and the six scores after phase two.
  - The launcher passes `RHO` only for `gcontrast`; the act / fisher / grad branch needs it too, with a run tag.

Status (2026-11-09): the first four items and the rho TODO have run; results in section 6.7. Family specificity at
matched warm-start dF was compared only through the floors, not per target.

### 6.7 Results after 2026-11-05

**Record:** WORK_LOG 2026-11-05 06:05 to 22:25 and 2026-11-09 07:05. Seed 0 unless noted. Every cell is the activation
contrast (`act`) with the prefix span, retain statistics over question and answer tokens, rho 1 and no guard unless
stated. Scores: Mem / Util / Priv / EM / FQ / Overall after phase two; warm start: dF / dR in nats.

**Code.** The launcher's `RHO` for act / fisher / grad is in the repository. The following options exist only in
scratch copies of the code so far and still need to be merged: rho 0 (the pencil on the protected span's complement,
statistics in float64), the two-sided selective families `grad_both` / `fisher_both`, `--gradient_positions`
(`tokens`, `answer`, `prefix_answer`, `nonprefix`), the retain-loss and position options for the EVR families, the
rotation (`--rotation pairs|haar`, `--rotation_free`, `--rotation_kappa`) and the sampled Fisher (`--fisher_labels`).

#### Kappa without the guard

S4 is S3 plus kappa 4. To separate kappa from the guard, S2 plus kappa ran with no guard.

| Variant | Directions | Warm start | Floor 0.2 | Floor 0.1 |
|---|---|---|---|---|
| S2 | 10974 | 2.98 / 1.02 (f0.1 3.52 / 1.34) | 0.9770 / 0.9547 / 0.9270 / 0.0267 / -0.81 / 0.9524 | 0.9725 / 0.9182 / 0.8524 / 0.0432 / -0.74 / 0.9117 |
| + kappa 16 | 10567 | 2.58 / 0.78 (3.04 / 1.03) | 0.9770 / 0.9555 / 0.8972 / 0.0426 / -0.06 / 0.9420 | 0.9782 / 0.9389 / 0.8741 / 0.0204 / -0.61 / 0.9284 |
| + kappa 12 | 10368 | 2.46 / 0.69 (2.90 / 0.92) | 0.9731 / 0.9675 / 0.8757 / 0.0525 / -0.12 / 0.9366 | 0.9839 / 0.9548 / 0.8954 / 0.0054 / -0.44 / 0.9433 |
| + kappa 8 | 9952 | 2.22 / 0.57 (2.62 / 0.77) | 0.9612 / 0.9654 / 0.8464 / 0.0765 / -0.15 / 0.9209 | 0.9915 / 0.9507 / 0.9336 / 0.0095 / -0.68 / 0.9580 |
| + kappa 4 | 8643 | 1.70 / 0.33 (2.01 / 0.45) | 0.9290 / 0.9681 / 0.7241 / 0.1374 / -0.24 / 0.8595 | 0.9658 / 0.9598 / 0.8205 / 0.0833 / -0.01 / 0.9102 |

- The drop from S2 to S4 comes from kappa: kappa 4 alone gives 0.8595 at floor 0.2, S4 0.8556.
- Overall peaks where the warm start reaches about dF 2.6-3.1 with dR at most 1: S2 at floor 0.2, kappa 8 at 0.1.

#### The threshold lambda*

| lambda* (directions) | Floor 0 | Floor 0.1 | Floor 0.2 |
|---|---|---|---|
| 1.25 (7036) | 3.09 / 0.96; 0.9826 / 0.9535 / 0.9436 / 0.0264 / -0.95 / 0.9596 | 2.67 / 0.75; 0.9721 / 0.9622 / 0.8877 / 0.0416 / -0.15 / 0.9391 | 2.25 / 0.56; 0.9313 / 0.9750 / 0.7833 / 0.1223 / -0.19 / 0.8886 |
| 1.5 (4555) | 2.25 / 0.49; 0.9437 / 0.9785 / 0.7905 / 0.0941 / -0.09 / 0.8964 | 1.94 / 0.37; 0.8984 / 0.9809 / 0.6829 / 0.1569 / -0.33 / 0.8341 | 1.63 / 0.28; 0.8754 / 0.9934 / 0.5680 / 0.2199 / -0.55 / 0.7673 |
| 1.75 (3025) | 1.64 / 0.23; 0.8871 / 0.9878 / 0.5983 / 0.1891 / -0.12 / 0.7872 | 1.42 / 0.18; 0.8657 / 0.9893 / 0.4858 / 0.2425 / -0.24 / 0.7101 | 1.21 / 0.13; 0.8156 / 0.9894 / 0.4444 / 0.2823 / -1.11 / 0.6686 |
| 2 (2118) | 1.25 / 0.12; 0.8263 / 0.9872 / 0.4691 / 0.2816 / -0.88 / 0.6889 | 1.09 / 0.09; 0.7990 / 0.9881 / 0.3902 / 0.3265 / -1.27 / 0.6216 | 0.92 / 0.07; 0.7524 / 0.9874 / 0.3403 / 0.3662 / -2.00 / 0.5682 |

- Along the grid Overall follows the warm-start dF. A higher lambda* buys a better dF / dR ratio but too little
  strength, and a lower floor does not make up for it beyond lambda* 1.25.
- **The best rule of the campaign: lambda* 1.25 at floor 0** (complete removal of 7036 directions). Seeds 0-2:
  0.9844 / 0.9531 / 0.9433 / 0.0160 / -0.48 / 0.9599, sd of Overall 0.0025 (0.9596 / 0.9626 / 0.9576).
- S2 plus kappa 8 at floor 0.1, seeds 0-2: 0.9827 / 0.9468 / 0.9352 / 0.0150 / -0.57 / 0.9545, sd 0.0033. S2 at floor
  0.2: 0.9510.

#### The leave-one-out guard

The guard of section 6.2 measures each target's edit alone. The leave-one-out guard instead measures its contribution
within the joint edit (the joint change minus the change without that target) and halves the target's directions while
that contribution's dF / dR is below the threshold. S2 at floor 0.2:

| Guard | Directions | Warm start | After phase two |
|---|---|---|---|
| none | 10974 | 2.98 / 1.02 | 0.9770 / 0.9547 / 0.9270 / 0.0267 / -0.81 / 0.9524 |
| leave-one-out 1.5 (26 targets kept, 4 halved, 2 dropped) | 10057 | 2.76 / 0.84 | 0.9736 / 0.9591 / 0.8892 / 0.0472 / -0.28 / 0.9391 |
| leave-one-out 2 (15 / 15 / 2) | 6892 | 2.07 / 0.45 | 0.9307 / 0.9720 / 0.7269 / 0.1346 / -0.04 / 0.8623 |

The contributions are not additive. Their sum is 2.78 / 1.37 against the joint 2.98 / 1.02, while the targets alone
sum to 1.46 / 0.41. Neither guard helps.

#### The ridge rho

Overall at rho 0 / 0.01 / 0.1 / 1 / 3 / 10 (rho 0: the pencil solved on the complement of the protected span):

| Setting | Directions | Overall |
|---|---|---|
| S2, floor 0.1 | 66670 / 62150 / 43340 / 10974 / 3433 / 659 | 0.8635 / 0.8790 / 0.8443 / **0.9117** / 0.8857 / 0.5099 |
| S2, floor 0.2 | as above | 0.9057 / 0.9183 / 0.9016 / **0.9524** / 0.8014 / 0.4330 |
| lambda* 1.25, floor 0 | 55684 / 51308 / 33731 / 7036 / 1974 / 305 | 0.8407 / 0.8502 / 0.8385 / **0.9596** / 0.7886 / 0.4308 |
| lambda* 1.25, floor 0.1 | as above | 0.9016 / 0.9035 / 0.8903 / **0.9391** / 0.7258 / 0.3535 |
| kappa 8, floor 0.1 | 66104 / 61562 / 42639 / 9952 / 2355 / 216 | 0.8815 / 0.8746 / 0.8652 / **0.9580** / 0.6876 / 0.2777 |
| kappa 8, floor 0.2 | as above | 0.9001 / 0.9177 / 0.9159 / **0.9209** / 0.6089 / 0.2551 |

- **rho 1 is best in all six settings.**
- Below 1 the rule admits tens of thousands of directions that are weak in both sets, since without the ridge the
  ratio of two small shares easily exceeds lambda*. dR falls (S2 floor 0.2: 1.02 -> 0.64 at rho 0.01), yet Utility
  falls to 0.77-0.91 (0.92-0.97 at rho 1). The warm-start dR does not see that cost.
- Above 1 few directions remain and the edit under-forgets: at rho 10 Privacy is 0.11-0.29 and FQ -5.5 to -8.0.
- The six scores of every cell are in the log.

#### Gradient statistics in S2's rule (one-sided)

The selective rule with gradient statistics: `fisher` (the per-sequence Fisher input moments), `grad` (the Gram of
the mean gradients), `fisher` with the act retain side (`ract`). lambda* 1, no guard. Retain loss on answers, or on
question and answer (`rlqa`). The gradient positions are every attended position, or only S2's token sets (`gtok`:
answer tokens for forget, question and answer for retain).

| Family | Directions | Floor 0.1 | Floor 0.2 |
|---|---|---|---|
| fisher | 9032 | 1.84 / 0.61; 0.8667 / 0.9407 / 0.7039 / 0.2014 / -2.31 / 0.8247 | 1.55 / 0.45; 0.7990 / 0.9572 / 0.6123 / 0.2560 / -4.59 / 0.7635 |
| fisher, ract | 8411 | 1.77 / 0.71; 0.8285 / 0.9326 / 0.6149 / 0.2369 / -3.36 / 0.7682 | 1.54 / 0.55; 0.7616 / 0.9637 / 0.5340 / 0.2896 / -5.04 / 0.7103 |
| grad | 9702 | 4.50 / 3.38; 0.9568 / 0.8760 / 0.8640 / 0.0870 / -0.49 / 0.8971 | 3.47 / 2.43; 0.9243 / 0.9244 / 0.7712 / 0.1446 / -0.88 / 0.8670 |
| fisher, rlqa | 11675 | 4.73 / 3.70; 0.9706 / 0.9306 / 0.9025 / 0.0764 / -0.68 / **0.9337** | 3.91 / 2.93; 0.9244 / 0.9436 / 0.8125 / 0.1273 / -1.27 / 0.8896 |
| grad, rlqa | 10466 | 8.49 / 7.66; 0.9644 / 0.8196 / 0.9336 / 0.0159 / -0.06 / 0.9014 | 7.08 / 6.37; 0.9706 / 0.8907 / 0.8878 / 0.0815 / -0.44 / 0.9148 |
| fisher, rlqa, gtok | 13313 | 4.22 / 3.43; 0.9732 / 0.9207 / 0.8902 / 0.0715 / -0.49 / 0.9268 | 3.56 / 2.73; 0.9355 / 0.9478 / 0.8128 / 0.1265 / -1.35 / 0.8943 |
| grad, rlqa, gtok | 12689 | 6.74 / 6.09; 0.9735 / 0.8564 / 0.9270 / 0.0312 / -0.33 / 0.9164 | 5.57 / 4.91; 0.9630 / 0.9009 / 0.8902 / 0.0930 / -0.95 / 0.9170 |

- The retain loss on question and answer helps both families (Fisher +0.11 at floor 0.1, grad +0.05 at floor 0.2).
- Restricting the gradient positions to S2's token sets changes Overall by -0.007 to +0.015.
- The best gradient cell (Fisher, rlqa, floor 0.1: 0.9337) stays below the activation contrast.
- With the prefix span protected, the projection already removes the prefix positions' gradient terms
  `delta_t x_t^T` exactly, since `x_t` lies in B. The position choice therefore changes only the question
  positions.

#### Two-sided selective bases (`grad_both`, `fisher_both`)

Plain's restricted form `U U^T ((S - I) W) V^T V`, rows at `max(cos(W_i, -G_r,i), floor)`. U and V are chosen by the
selective rule on each side (rho / out and rho / in), from the mean gradients (`G P G^T`, `G^T G`) or the per-example
gradient moments, forget against retain. The input side keeps the prefix span. Overall at floor 0.1 / 0.2:

| Family | lambda* | Retain loss | Directions out / in | Warm start f0.1 | Overall |
|---|---|---|---|---|---|
| grad_both | 1.25 | answer | 5558 / 8368 | 0.39 / 0.16 | 0.2567 / 0.2277 |
| grad_both | 1.25 | rlqa | 6155 / 9150 | 1.28 / 0.96 | 0.3646 / 0.3004 |
| grad_both | 1.25 | rlqa, gtok | 6478 / 11216 | 1.12 / 0.83 | 0.3616 / 0.2982 |
| grad_both | 1 | answer | 6617 / 9702 | 0.51 / 0.23 | 0.3207 / 0.2764 |
| grad_both | 1 | rlqa | 7239 / 10466 | 1.54 / 1.18 | 0.4493 / 0.3676 |
| fisher_both | 1.25 | answer | 5653 / 7029 | 0.15 / 0.02 | 0.1768 / 0.1658 |
| fisher_both | 1.25 | rlqa | 6892 / 9672 | 0.61 / 0.35 | 0.2694 / 0.2288 |
| fisher_both | 1.25 | rlqa, gtok | 6902 / 11218 | 0.59 / 0.35 | 0.2572 / 0.2272 |
| fisher_both | 1 | answer | 7430 / 9032 | 0.25 / 0.04 | 0.2272 / 0.2088 |
| fisher_both | 1 | rlqa | 8725 / 11675 | 0.85 / 0.51 | 0.3583 / 0.2998 |

- The two-sided selective basis keeps almost none of the forget gradient. The warm start reaches dF 0.15-1.54,
  against 13.65 for plain restricted at floor 0.2 (0.9082).
- After phase two the forget set is barely touched: Mem 0.33-0.56, Priv 0.07-0.26, FQ -10 to -14.
- The high-energy forget-gradient directions carry retain energy too, so they fail the threshold on either side.
  Lowering lambda* to 1 adds about 1000 directions per side and little strength.

#### Plain restricted: retain loss and gradient positions

Plain restricted takes its loss on the answer tokens and sums `delta_t x_t^T` over every position. Its basis comes
from the forget gradient alone, and the retain gradient enters only the row shrink. In every target 99.6-100% of the
rows sit at the floor, so the retain loss barely acts. EVR 0.99, Overall at floors 0.1 / 0.2 / 0.3.

| Variant | Prefix span | Directions | Warm start f0.2 | Overall |
|---|---|---|---|---|
| reference (all positions, answer loss) | yes | 29001 | 13.65 / 13.29 | 0.8487 / **0.9082** / 0.8680 |
| rlqa | yes | 29001 | 13.65 / 13.29 | 0.8249 / 0.9053 / 0.8680 |
| gtok | yes | 28360 | 13.20 / 12.72 | 0.8758 / 0.9030 / 0.8639 |
| gtok + rlqa | yes | 28360 | 13.20 / 12.72 | 0.8735 / 0.9056 / 0.8639 |
| all positions | no | 24666 | 10.02 / 9.98 | 0.7359 / 0.8789 / 0.8358 |
| answer positions | no | 27322 | 11.40 / 11.21 | 0.8365 / 0.8653 / 0.8705 |
| prefix + answer positions | no | 23638 | 9.15 / 8.90 | 0.8711 / 0.8774 / 0.8235 |
| question + answer positions | no | 27531 | 12.57 / 12.41 | 0.7503 / 0.8686 / 0.8535 |

- **Phase two is deterministic and sensitive.** Identical edits give identical phase twos: rlqa at floor 0.3 repeats
  the reference in every score. Yet the few rows that rlqa changes at floor 0.1 move Overall by -0.024. Differences of
  ±0.03 in this table are of that size.
- **Without the prefix span the position set matters.** At floor 0.1, leaving the question positions out of the
  forget gradient raises Utility from 0.49-0.52 to 0.66-0.74 (Overall 0.84-0.87 against 0.74-0.75).
- **No position set reaches the protected reference** (0.9082). With the prefix span, prefix + answer equals answer,
  and question + answer equals all positions.

#### Rotation instead of contraction

At lambda* 1.25 with floor 0, the contraction removes W's response along V. The rotation instead keeps it and turns
it within V:

`W' = W (I - V V^T + V Q V^T)`

- With Q orthogonal, W keeps its singular values, and inputs orthogonal to V (the prefix among them) are unchanged.
- `pairs`: rotations by an angle in the planes of consecutive directions (lambda order). Each direction keeps `cos a`
  of its response and passes `sin a` to its partner. 90 degrees is the counterpart of floor 0; 180 degrees is the
  floor -1.
- `haar`: a random orthogonal Q.

Two guards:
- **(A) Free output space.** The removal as at floor 0, with the rotated response injected only through
  `P_free = I - O O^T`. O spans the directions holding 99% of the residual stream's second moment where the target
  writes: o_proj at the post-attention norm's input, down_proj at the block output. The moment is taken over retain
  question and answer tokens, or with `all` also over the forget ones.
- **(B) Low-retain directions.** Rotate only the directions whose retain share is at most 1 / n, and shrink the rest
  to 0.

| Edit | Warm start | After phase two |
|---|---|---|
| contraction, floor 0 | 3.09 / 0.96 | 0.9826 / 0.9535 / 0.9436 / 0.0264 / -0.95 / **0.9596** |
| pairs 90 | 4.72 / 2.06 | 0.9476 / 0.8985 / 0.7660 / 0.1326 / -1.27 / 0.8636 |
| pairs 180 | 7.95 / 4.75 | 0.8326 / 0.6901 / 0.3459 / 0.3815 / -2.31 / 0.5414 |
| haar | 4.55 / 1.97 | 0.9333 / 0.8876 / 0.7448 / 0.1433 / -1.35 / 0.8473 |
| pairs 90, A retain | 3.19 / 1.03 | 0.9751 / 0.9460 / 0.8870 / 0.0525 / -0.44 / 0.9346 |
| pairs 90, A all | 3.17 / 1.02 | 0.9837 / 0.9413 / 0.9165 / 0.0354 / -0.28 / 0.9464 |
| pairs 90, B (943 of 7036 rotated) | 3.25 / 1.00 | 0.9771 / 0.9514 / 0.8843 / 0.0408 / -0.55 / 0.9359 |
| pairs 90, A retain + B | 3.11 / 0.96 | 0.9807 / 0.9487 / 0.9245 / 0.0338 / -0.55 / 0.9507 |

- **Unguarded rotation over-forgets.** After phase two, its forget NLL ends above the Retain level: 3.14 / 4.21 / 3.17,
  against 2.74 for the contraction. The loss MIA AUC is 0.437 / 0.214 / 0.424 against 0.569, where 0.5 means
  indistinguishable from holdout. Privacy penalises this.
- **It also costs Utility:** model utility 0.489 / 0.316 / 0.478 against 0.546.
- **The guards reduce the rotation to about the contraction.** The residual stream's occupied space holds 1700-1900 of
  2048 directions, and only 7-9% of the rotated response's energy lies in the free part.
- **None beats the contraction.** The best guarded cell (0.9507) is below the contraction's lowest seed (0.9576).
- **The selected directions are not retain-free.** At lambda* 1.25 each carries 1.3-2.3 times an average direction's
  retain energy (largest 3.7). Their forget share is simply higher still.

#### Short phase two as a cheaper yardstick

Spearman correlation with the full Overall over 23 runs (the share of concordant pairs in brackets):
- phase two of 0.1 epoch: its Overall 0.64 (0.76), its Privacy 0.70 (0.79);
- phase two of 1 epoch: Mem 0.77 (0.84), the forget NLL at its end 0.73 (0.85), its Overall 0.70 (0.78);
- the warm start's forget minus retain NLL: 0.63; its retain NLL alone: -0.12.

Neither short run orders the best runs correctly. Loading and evaluation (about 260 s) dominate either way, so the
short phase two saves little time.

#### Open

- **The sampled Fisher** (user 2026-11-09). The Fisher input moments with labels drawn from the model's predictive
  distribution at the loss positions: teacher forcing, 4 draws per sequence. One-sided (`fisher`, S2's rule), seed 0.
  - Warm start, sampled against empirical (dF / dR at floor 0.1, 0.2): retain loss on answers 2.14 / 0.65, 1.82 / 0.48
    (1.84 / 0.61, 1.55 / 0.45); rlqa 4.08 / 2.72, 3.34 / 2.05 (4.73 / 3.70, 3.91 / 2.93). The sampled Fisher has the
    better ratio in all four cells.
  - After phase two, retain loss on answers, floor 0.1: 0.9244 / 0.9407 / 0.8102 / 0.1421 / -0.95 / 0.8878 (empirical
    0.8247).
  - **TODO (user 2026-11-10):** the three missing phase twos, cancelled before they started. Their edits are saved, so
    phase two alone runs from them (`BASE_ID` = the model directory):
    - retain loss on answers, floor 0.2 (edit 25938417, `selective_fisher_fs4_nonprefix_prefixyes_knone_l1_f0p2_g0`);
    - rlqa, floor 0.1 (edit 25938419, `selective_fisher_rlqa_fs4_nonprefix_prefixyes_knone_l1_f0p1_g0`). This one is
      the comparison with the best Fisher cell (empirical rlqa f0.1, 0.9337);
    - rlqa, floor 0.2 (edit 25938421, `selective_fisher_rlqa_fs4_nonprefix_prefixyes_knone_l1_f0p2_g0`).
  - The mean-gradient families have no sampled counterpart, since that gradient's expectation under the model's labels
    is zero.
  - The trace of these moments is the target's block of `Tr(F) = E_x E_{y~p(y|x)} ||d l(x, y) / d theta||^2`.
  - Contexts: the draws keep the data's preceding tokens instead of a generated answer.
- **Merging the scratch code into the repository** (see **Code** above).

## 7. Learned direction gates (2026-11-03, separate line of work)

**Record:** `direction gates`, `gate calibration`.

`replication/scripts/recast_direction_gates.py`, `--variant direction_gates`, launchers `recast_direction_gates_gh200.sh`
and `recast_gate_calibration_gh200.sh`, summaries `replication/scripts/gate_calibration_summary.py` and
`replication/scripts/gate_runs_summary.py`, tests `test_recast_direction_gates.py`.

- **What it does.** Instead of removing whole input directions, it learns a gate `s_j` in [0, 1] on each of k fixed
  input directions per target: `W_eff = W0 - sum_j (1 - s_j) (W0 v_j) v_j^T`, applied as a hook, with every gate
  starting at exactly 1.
- **Bases:**
  - `input_contrast`: the QR columns;
  - `input_contrast_dual`: the generalised eigenvectors without the QR, with oblique parts so that each gate carries
    exactly its lambda;
  - `protected_contrast`, `protected_contrast_dual` (2026-11-04): the same two with the constant prompt prefix's
    inputs protected (the pencil in the complement of their span, as section 6) and the retain covariance over every
    non-prefix token; the gates then leave the prefix's computation unchanged. **The default for the gates** (user
    2026-11-04); the unprotected bases only as an explicit comparison;
  - `forget_input`;
  - `random`: the control.
- **Objective.** With the model frozen: `J = L_R + alpha L_F + gamma O + beta R`.
  - `L_R`: the retain NLL;
  - `L_F`: the forget token unlikelihood `-log(1 - p)`;
  - `O`: the sum of the gates;
  - `R`: the relative squared displacement.
- **Training and after.** The gates are trained by Adam or SGD to convergence or an epoch cap. `W_eff` is then
  written into the weights, and phase two runs as for `restricted_shrink`.
- **Fixed schedule (2026-11-05).** `--direction_gates_convergence_epochs 0` trains for exactly
  `--direction_gates_max_epochs` epochs, without the convergence rule. `--direction_gates_lr_schedule linear` lowers
  the lr linearly to 0 over the steps, and `--direction_gates_probe_every` evaluates the probe less often. The aim is a
  short gate phase of known length (user: the method should be practical).
- **Binarization (2026-11-04).** Optional (`--direction_gates_binarize_share`, `--direction_gates_binary_s`): the gates
  choose the directions and the cut removes them whole. The most closed gates, as few as remove the given share of the
  forget output energy over all targets (the global selection's budget), keep `s`; every other gate returns to 1.
- **Candidates, start and cuts (2026-11-04).** `--direction_gates_lambda_star` gates only the directions with
  `lambda >= lambda*` (with `protected_contrast` and lambda* 1 these are the directions of S2, section 6);
  `--direction_gates_init` sets the gates' start (0.2 on that set is S2 at floor 0.2); `--direction_gates_max_epochs 0`
  skips the gate training. After a binarization, or with no gate training on every candidate (S2's directions with
  lambda* 1, 2026-11-05), `--direction_gates_cut shrink` with `--direction_gates_cut_s` shrinks the chosen directions
  to another value (0, or -1 to reflect them), and `--direction_gates_cut rotate` rotates their parts onto outputs
  orthogonal to their own span and to the forget output subspace. With `--direction_gates_cut_s c` the rotation is a
  partial turn: each part keeps `c` of itself along its own direction, at its full norm.
- **Diagnostics (2026-11-04).** Every gated run reports how much the gates change the outputs at the prefix positions
  (BOS included) and, after phase two, whether the cut directions regrew (`regrowth`, `back` per group of gates).
- **Calibration mode.** `--direction_gates_calibration_trials` measures the gradient scale at `s = 1` and runs a short
  training per optimizer and learning rate, without phase two.
- **Status (2026-11-10).** Seed 0 unless n is given; Overall here, every score and job ID in the log.
  - Calibration (25593901): Adam 1e-2 is monotone with few reversals. Adam 1e-5 is 2-3 orders of magnitude too slow.
    SGD oscillates at every learning rate. Adam above 1e-2 was not tried.
  - `L_F` gates: phase two undoes them. Even converged (forget NLL 11.8 after phase one), Overall 0.563.
  - The closing objective alone (alpha 0, beta 0), 1024 candidates per target, gamma 1e-4, to convergence: 0.9132 +-
    0.0040 on the dual basis without the prefix protection, 0.8779 +- 0.0101 with it (n 3). A larger gamma is worse
    (2e-4 / 3e-4 / 1e-3: 0.83 / 0.74 / 0.53 protected, 0.73 / 0.62 / 0.48 unprotected).
  - On the lambda* 1 set (S2's directions), gamma 3e-4: from s = 1 (`protected_contrast_dual`) 0.9164 +- 0.0035 (n 2);
    from S2's point (`protected_contrast`, every gate at 0.2) 0.9483. These are the best continuous gates.
  - S2 written as gates (every gate at 0.2, no gate training) repeats S2: 0.9552 against 0.9524.
  - Binarization (a step toward continuous gates): the best run, the protected QR basis at the share 0.1461 with the
    chosen directions shrunk to 0, 0.9496.
  - Length of the gate phase: 43-120 epochs to convergence, about 13 s each (9-26 min), against about 4 min of phase
    two. Shorter phases lose. The stop after 27 epochs (window 5, tol 1e-2) gives 0.8280 / 0.8477 against 0.8870 /
    0.9161 converged (protected / unprotected). A fixed schedule with the lr linear to 0 over 10 / 20 / 40 epochs gives
    0.39 / 0.63 / 0.79 protected and 0.28 / 0.60 / 0.80 unprotected.
- **Findings.**
  - The parametrization is not the limit; the objective is. The gates hold S2 exactly. From S2's point the retain loss
    reopens the gates under a weak gamma (1e-5 / 3e-5: 0.28 / 0.49), `L_F` drifts to 0.918, and gamma 3e-4 stays near
    S2 (0.9483).
  - Cut directions never regrow in phase two: `back` stays at the cut's value after every shrink, both rotations and
    the reflection. The forget answers return through other weights.
  - The candidate set's selectivity matters more than the objective. At the same forget NLL after phase one (3.03)
    the gates on the lambda* set leave the retain NLL at 0.79, those on 1024 candidates at 1.58.
  - The result follows the state after phase one. The best runs end it at a forget NLL of about 2.9-3.3 and a retain
    NLL of about 0.8-1.2 (S2 as gates 3.27 / 1.22; the lambda* set from S2's point 2.88 / 0.96, from s = 1 3.03 /
    0.79). Below about 2 the forgetting is too weak. Above it, or with a higher retain NLL, the forget NLL ends past
    Retain's 2.70 after phase two and Privacy falls. One seed per point: an observation, not a rule.
  - The closing speed is bound by the Adam step (about lr per step). A decaying lr halves the gates' travel, so at lr
    1e-2 the fixed schedule does not shorten the phase.
  - Prefix protection helped the `L_F` and the binarized gates (all three pairs of 25643611) but costs 0.035 for the
    continuous closing gates on 1024 candidates (n 3).
- **Did not hold up:**
  - the `L_F` objective;
  - gamma above 1e-4 on 1024 candidates;
  - the short stop rule and the fixed schedule with a decaying lr;
  - turning or reflecting the cut parts. On S2's directions the quarter turn gives 0.7284, the partial turn keeping 0.2
    gives 0.8204 and the reflection 0.4287, and even the shrink to 0 gives 0.8685, against S2's shrink to 0.2 (0.9524);
    each costs Utility and forgets past Retain. On binarized directions: rotation 0.86-0.87, reflection 0.54.
- **Directions (2026-11-10), most promising first.**
  1. Gates on the lambda* 1.25 set, the campaign's best selective rule (0.9599, n 3, section 6.7): from its point (needs
     `--direction_gates_init 0`, refused today) or from s = 1 with the closing objective. The shortest way for
     continuous gates to reach the best result.
  2. A target state instead of the price gamma: close the gates until the forget NLL after phase one reaches a target
     (about 3) at the least retain cost, as a constraint or with a Lagrange multiplier. The stop then follows from the
     target, and the phase ends once the target is reached.
  3. Speed: Adam above 1e-2 at a constant lr (3e-2, 1e-1), and a start near a solution (to convergence in 47 epochs
     from S2's point against 76 from s = 1).
  4. The recovery-aware objective (open item below): the regrowth reports show that the recovery goes around the cut
     directions, which is what this objective targets.
  5. Prefix protection for continuous gates: settle the default once the lambda* set is measured without it.
  - Not run (cancelled while pending on 2026-11-09, see the log): seeds 1-2 from S2's point and seed 2 from s = 1 at
    gamma 3e-4; gamma 5e-4 / 1e-3 on the lambda* set; the lambda* set without the prefix protection.
- **Open item (user 2026-11-04).** The Classification cycles, where gate epochs alternate with model epochs and the
  gates keep acting while the model trains. They work directly against the recovery in phase two, but they break the
  rule that the restriction belongs to phase one only. Not implemented.
- **Main target (user 2026-11-04).** Continuous gates, without binarization; binarization and budgets are steps
  toward it.
- **Open item (user 2026-11-04).** An objective that accounts for the recovery in phase two, e.g. `L_F` measured after
  a simulated phase-two step on retain. Not implemented.

## 8. Standalone analyses and tools

| Script (launcher) | What it measures |
|---|---|
| `prediction_kl.py` (`prediction_kl_gh200.sh`) | KL between a reference model's and other models' next-token distributions on answer tokens, per split. |
| `bf16_weight_changes.py` (`bf16_weight_changes_gh200.sh`) | Which weights a bf16-weight training at lr 1e-5 could move: a bitwise comparison of two checkpoints from one base. |
| `tofu_checkpoint_screen.py` (`checkpoint_screen_gh200.sh`) | Published unlearning checkpoints screened on Mem / Priv / EM / FQ (no generation) against Table 3. |
| `global_order_summary.py` | Global input-contrast runs by order, with the size of the removed directions. |
| `recast_stage1_tables.py`, `recast_contraction_tables.py`, `recast_component_diagnostics.py` | Tables of sections 2 and 3. |

## 9. Results so far

All runs: down_proj + o_proj, current recipe, `restricted_shrink` unless noted. Mem / Util / Priv / EM / FQ / Overall;
n is the number of seeds. The log holds the job IDs and the comparisons with their tests.

| Configuration | n | Scores |
|---|---|---|
| Selective input contrast, lambda* 1.25, floor 0, no guard (6.7) | 3 | 0.9844 / 0.9531 / 0.9433 / 0.0160 / -0.48 / 0.9599 |
| Selective input contrast S2 + kappa 8, floor 0.1, no guard (6.7) | 3 | 0.9827 / 0.9468 / 0.9352 / 0.0150 / -0.57 / 0.9545 |
| Selective input contrast S2, lambda* 1, floor 0.2, no guard (6) | 3 | 0.9809 / 0.9552 / 0.9189 / 0.0293 / -0.49 / 0.9510 |
| Direction gates, closing gamma 3e-4 on the lambda* 1 set, from S2's point (7) | 1 | 0.9831 / 0.9406 / 0.9233 / 0.0330 / -0.68 / 0.9483 |
| Global input contrast, rho 1, `score` order, budget floor 0.3 (section 4) | 6 | 0.9824 / 0.9247 / 0.9330 / 0.0109 / -0.64 / 0.9460 |
| Selective input contrast S3, lambda* 1, floor 0.2 (6) | 3 | 0.9715 / 0.9620 / 0.8962 / 0.0516 / -0.18 / 0.9420 |
| Selective input contrast S4, lambda* 0.75, floor 0.2 (6) | 3 | 0.9650 / 0.9624 / 0.8935 / 0.0524 / -0.24 / 0.9391 |
| Selective Fisher contrast, lambda* 1, retain loss on question + answer, floor 0.1, no guard (6.7) | 1 | 0.9706 / 0.9306 / 0.9025 / 0.0764 / -0.68 / 0.9337 |
| Direction gates, closing gamma 3e-4 on the lambda* 1 set, from s = 1, protected dual basis (7) | 2 | 0.9569 / 0.9482 / 0.8518 / 0.0884 / -0.10 / 0.9164 |
| Gradient contrast, both sides, matched rank at EVR 0.995, floor 0.1 (5.1) | 1 | 0.9864 / 0.8377 / 0.9367 / 0.0108 / -0.55 / 0.9160 |
| Direction gates, closing gamma 1e-4, 1024 candidates per target, dual basis without protection (7) | 3 | 0.9479 / 0.9118 / 0.8823 / 0.0269 / -0.18 / 0.9132 |
| Plain basis with the prefix span protected, EVR 0.99, floor 0.2 (6.5) | 3 | 0.9786 / 0.8056 / 0.9156 / 0.0403 / -0.41 / 0.8939 |
| Per-example forget moments, own EVR 0.99, floor 0.4 (5.2) | 1 | 0.9841 / 0.8063 / 0.8963 / 0.0420 / -0.33 / 0.8897 |
| Direction gates, closing gamma 1e-4, 1024 candidates per target, protected dual basis (7) | 3 | 0.9192 / 0.9260 / 0.8007 / 0.0643 / -1.06 / 0.8779 |
| Plain basis, EVR 0.99, floor 0.2 | 3 | 0.9430 / 0.8221 / 0.8700 / 0.0736 / -1.57 / 0.8756 |
| Gradient contrast, `kept_energy` share 0.45, floor 0.3 (best of both grids) | 1 | 0.9762 / 0.7815 / 0.8605 / 0.0547 / -1.27 / 0.8655 |

Rules that did not hold up, with the reason:
- **Gradient contrast with `side_energy` ranks** (best Overall 0.770): the input side needs almost every direction to
  hold a share of `||G_f||^2`.
- **Gradient contrast with `kept_energy` ranks:**
  - What happens: forcing one share in every block raises the rank to full in blocks whose forget gradient lies in
    directions shared with retain (down_proj in blocks 0, 1, 15; o_proj in blocks 5-8).
  - Larger floors at larger shares (up to share 0.5, floor 0.5) do not close the gap to the matched rank.
- **KL 0.1 / 0.3 to Init Fine. after retain fine-tuning** on the gradient-contrast, per-example and plain cells:
  - Utility moves by -0.05 to +0.02, within the seed spread.
  - Privacy falls by 0.01-0.03 in every cell.
- **The selective rule at rho other than 1** (6.7). Below 1, many weak directions cost Utility. Above 1, the edit
  under-forgets.
- **Loss guards on the selective rule** (6.2, 6.7), alone or leave-one-out: they give up forgetting that phase two does
  not restore.
- **Two-sided selective bases** (6.7): too little of the forget gradient survives the threshold on both sides (Overall
  0.17-0.45).
- **Rotation instead of contraction** (6.7): it over-forgets and costs Utility. Guarded, it falls back to about the
  contraction.
- **Learned gates: the `L_F` objective, shorter gate phases, turned or reflected cuts** (7). Phase two undoes the
  `L_F` gates. Stopping before convergence, or decaying the lr, leaves the gates too open. Turning or reflecting the
  cut parts (S2's directions or binarized ones) costs Utility and forgets past Retain.
