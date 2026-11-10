"""ReCAST on TOFU forget10 (Llama-3.2-1B-Instruct), scored like Table 3 of arXiv 2606.14997.

One run = one variant at one shrink floor:

- ``unrestricted``: retain-aligned row shrink ``W* = s * W`` of every decoder-block linear
  weight, then phase two trains the whole model;
- ``restricted_semu``: the same proposal projected into the forget-gradient SVD subspace
  (``W* = W + P(s * W - W)``), each target replaced by a SEMU adapter ``anchor + U A Vh``, and
  phase two trains the cores ``A`` plus the groups without an adapter named by ``--phase2_free``
  (``none`` = only the cores, as Classification);
- ``restricted_shrink`` (extension, user request 2026-10-22): the restricted warm start written into
  the native weights, no adapter, and phase two trains every parameter as ``unrestricted``;
- ``direction_gates`` (extension, user 2026-11-03, ``recast_direction_gates.py``): no ReCAST update; a gate in
  [0, 1] on each of k fixed input directions of every target (input-contrast, forget-input or random basis) is
  trained with the model frozen on the retain NLL plus ``--direction_gates_alpha`` x the forget unlikelihood plus
  ``--direction_gates_gamma`` x the open gates plus ``--direction_gates_beta`` x the displacement until convergence
  (capped by ``--direction_gates_max_epochs``) or, with ``--direction_gates_convergence_epochs 0``, for a fixed number
  of epochs (lr constant or linear to 0, user 2026-11-05), the gated weights are written into the native weights, and
  phase two trains every parameter as ``restricted_shrink``. The chosen directions (binarized, or with no gate
  training every candidate, as the selective rule S2 with ``--direction_gates_lambda_star``) can be shrunk or rotated
  (``--direction_gates_cut``). With ``--direction_gates_calibration_trials`` the run only measures
  how to set the gates' learning rate: the gradient census at s = 1 and a short training per (optimizer, lr) from the
  same start, then it stops (no phase two, no evaluation).

Restricted variants can soften the warm start along the input directions retain uses
(``--retain_input_filter ridge``, extension): the projected delta is right-multiplied by
``mu (C_r + mu I)^-1``, C_r the retain input covariance over answer tokens at Init Fine. (A), or
build each restriction basis from the forget gradient whitened by C_r (``--restriction_whitening
retain_input``, ``recast_core.whitened_svd_basis``) (C). The edit budget is the EVR rank, an explicit
rank multiple of it, or a forget-output target per target (``--restriction_forget_output_target``).
``--component_selection`` (extension, signed shrink) keeps, removes or rescales the Frobenius-orthogonal
components ``D_j`` of each projected update by the sign of their first-order forget contribution
``c_j = <G_f, D_j>``, with a random and a norm-matched control; ``--component_diagnostics yes`` reports the
decomposition and the actual against the predicted forget / retain loss change of the warm start.
``--epochs 0`` evaluates the warm-started model without phase two, and ``--eval_trajectory nll`` tracks
the answer-token NLL and exact memorization of forget, retain (the 400 Model Utility items), real authors
and world facts before phase two and after every epoch (extensions).

The evaluation also keeps what ``compute_full_tofu`` computes and discards (``tofu_capture.py``):
the nine sub-metrics behind Model Utility (``_model_utility_subs``) and the generations behind the
four ROUGE scores (``generations.json``); no aggregate changes.

Phase two uses retain data only (no forget loss; user decision 2026-10-22) with the
OpenUnlearning TOFU trainer recipe, through the Hugging Face Trainer. ``--retain_loss`` picks the
retain NLL, KL(Init Fine. || model) per non-padding position against a frozen copy of the starting
model, or NLL + ``--kl_weight`` x KL (extension, user request 2026-10-22). The unlearned model is
scored with engram.benchmarks._tofu_official against the Init Fine. and Retain raw metrics of
the Table 3 run, so its row is directly comparable with Table 3.

Usage (GH200 node, ai-engram-gh200 environment, offline cache): every argument is required.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import socket
import time

from recast_core import (COMPONENT_SELECTIONS, CONTRACTION_MODES, CONTRAST_RANK_RULES, GRADIENT_CONTRAST_SIDES,
                         SHRINK_SCORES, TARGET_PROFILES,
                         WRITE_PROJECTIONS,
                         SemuAdapterLinear, apply_semu_adapters, apply_write_divisors, residual_write_divisors,
                         change_report,
                         collect_example_gradient_moments,
                         collect_input_covariances, collect_mean_gradients, drift_report, free_parameters,
                         frozen_checksums, mix_sources, ridge_input_filter, svd_basis,
                         parse_blocks,
                         parse_points,
                         parse_type_values,
                         piecewise_linear,
                         scale_update, select_targets, snapshot_trainable, untie_output_embedding, warm_start)
from recast_direction_families import FAMILIES
from recast_direction_gates import CONTRAST_BASES, GATE_BASES, GATE_LR_SCHEDULES, GATE_OPTIMIZERS, PROTECTED_BASES
from recast_global_families import BUDGET_UNITS, GLOBAL_ORDERS
from table3_tofu import COLUMNS, PAPER_TABLE3, Stage, git_info, to_jsonable
from tofu_capture import TofuCapture

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")
RESTRICTED_VARIANTS = ("restricted_semu", "restricted_shrink")


def floor_arg(value):
    """--s_min: one floor, or ``projection=floor`` pairs (one per projection type)."""
    try:
        return float(value)
    except ValueError:
        return parse_type_values(value)


def floor_label(s_min):
    return f"{s_min:g}" if not isinstance(s_min, dict) else ",".join(f"{k}={v:g}" for k, v in s_min.items())


def parse_gate_trials(text: str):
    """``adam:1e-5,sgd:1e-3`` -> [("adam", 1e-5), ("sgd", 1e-3)]; ValueError on anything else."""
    trials = []
    for item in text.split(","):
        name, sep, lr = item.partition(":")
        if not sep or name not in GATE_OPTIMIZERS or float(lr) <= 0:
            raise ValueError(f"need optimizer:lr pairs with an optimizer from {GATE_OPTIMIZERS} and a positive lr, "
                             f"got {item!r}")
        trials.append((name, float(lr)))
    return trials


def parse_args():
    p = argparse.ArgumentParser(description="ReCAST on TOFU, scored like Table 3")
    # data and model
    p.add_argument("--base_id", required=True)
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--forget_split", required=True, help="e.g. forget10 (forget gradient)")
    p.add_argument("--retain_split", required=True, help="e.g. retain90 (retain gradient and phase two)")
    p.add_argument("--holdout_split", required=True, help="e.g. holdout10 (evaluation only)")
    p.add_argument("--phase_one_dtype", required=True, choices=["bfloat16", "float32"],
                   help="weights during gradient and covariance collection and the warm start (the SVDs and the update "
                        "are computed in float32 / float64 either way)")
    p.add_argument("--train_dtype", required=True, choices=["bfloat16", "float32"],
                   help="weights during phase two; the model is cast after phase one when it differs")
    p.add_argument("--eval_dtype", required=True, choices=["bfloat16"], help="weights during evaluation (as the anchors)")
    p.add_argument("--attn_implementation", required=True)
    # ReCAST phase one
    p.add_argument("--variant", required=True,
                   choices=["unrestricted", "restricted_semu", "restricted_shrink", "direction_gates"],
                   help="restricted_shrink: restricted warm start, then phase two on every parameter (no adapter); "
                        "direction_gates: learned gates on fixed input directions instead of a ReCAST update, then "
                        "phase two on every parameter")
    p.add_argument("--targets", required=True, choices=sorted(TARGET_PROFILES))
    p.add_argument("--target_blocks", required=True, help="'all' or decoder-block indices, e.g. 8-15 or 0,2,5-7")
    p.add_argument("--untie_lm_head", choices=["yes", "no"],
                   help="when the head is a target or trained: give it its own weight copy")
    p.add_argument("--train_lm_head", required=True, choices=["yes", "no"],
                   help="phase two trains the head outside any restriction (Classification fc rule)")
    p.add_argument("--s_min", type=floor_arg,
                   help="shrink floor: s = max(cos(W_row, -G_r,row), s_min); one value, or one per projection type of "
                        "--targets (extension), e.g. down_proj=0.2,up_proj=0.5,o_proj=0.5; every variant but "
                        "direction_gates")
    p.add_argument("--grad_batch_size", type=int, required=True)
    p.add_argument("--freeze_input_embedding", choices=["yes", "no"], help="unrestricted only")
    p.add_argument("--restriction_source", choices=["forget_only", "full_data"],
                   help="restricted only; full_data = example-weighted mean of forget and retain gradients")
    p.add_argument("--restriction_alignment_scope", choices=["full_space", "restriction_subspace"],
                   help="restricted only: align W with -G_r, or P(W) with -P(G_r) (Classification option)")
    p.add_argument("--shrink_layer_scaling", choices=["none", "subspace_norm", "subspace_norm_mean"],
                   help="subspace_norm (restricted, extension): per-target floor 1-(1-s_min)*r/max r, r=||P(W)||/||W||; "
                        "every variant but direction_gates")
    p.add_argument("--restriction_evr", type=float, help="restricted only: EVR threshold of the forget-gradient SVD")
    p.add_argument("--restricted_head", choices=["unrestricted", "adapter"],
                   help="restricted with the head: 'unrestricted' = Classification --recast_train_fc_unrestricted")
    p.add_argument("--retain_input_filter", choices=["none", "ridge"],
                   help="restricted only (extension): right-multiply the warm-start delta by mu (C_r + mu I)^-1, "
                        "C_r the retain input covariance over answer tokens, mu = kappa tr(C_r) / d")
    p.add_argument("--retain_filter_kappa", type=float, help="ridge filter only: kappa")
    p.add_argument("--restriction_whitening", choices=["none", "retain_input"],
                   help="restricted only (extension): restriction bases from the SVD of G_f (C_r + eps I)^-1/2, "
                        "C_r the retain input covariance over answer tokens, eps = rho tr(C_r) / d")
    p.add_argument("--whitening_rho", type=float, help="retain_input whitening only: rho")
    p.add_argument("--restriction_rank_multiplier", type=float,
                   help="restricted only (extension): explicit edit budget, rank = round(m * r0) per target, r0 the "
                        "plain-basis rank at --restriction_evr; the same ranks for the plain and the whitened basis")
    p.add_argument("--restriction_representation",
                   choices=["none", "identity", "gradient", "whitened_gradient", "representation", "output_only"],
                   help="restricted only (extension, basis D): input covectors from C_f v = gamma (C_r + eps I) v, "
                        "eps = rho tr(C_r) / d (--whitening_rho); output side identity (one-sided), gradient (G_f V), "
                        "whitened_gradient (the output side of C) or representation (W C W^T contrast); needs "
                        "--restriction_rank_multiplier")
    p.add_argument("--restriction_gradient_contrast", choices=["none", *GRADIENT_CONTRAST_SIDES],
                   help="restricted only (extension, user 2026-11-01): bases from the forget-vs-retain mean-gradient "
                        "contrast, input side (G_f^T G_f / tr) v = lambda (G_r^T G_r / tr + rho / in I) v, output side "
                        "the same for G G^T (both) or the left singular vectors of G_f (input), rho = --whitening_rho; "
                        "rank m x the plain-basis rank with --restriction_rank_multiplier, else per side the directions "
                        "holding --restriction_evr of the mean forget gradient's energy on that side (user 2026-11-02)")
    p.add_argument("--gradient_contrast_rank_rule", choices=list(CONTRAST_RANK_RULES),
                   help="gradient contrast without --restriction_rank_multiplier (user 2026-11-02): side_energy (default) = "
                        "per side the directions holding --restriction_evr of the mean forget gradient's energy on that "
                        "side; kept_energy = one rank for both sides, the smallest whose two-sided restriction keeps "
                        "--restriction_evr of it, ||U U^T G_f V V^T||^2 / ||G_f||^2")
    p.add_argument("--restriction_gradient_moment", choices=["mean", "per_example"],
                   help="restricted only (extension, user 2026-11-02): mean (default) = the plain basis, the SVD of the "
                        "mean forget gradient; per_example = U and V the leading eigenvectors of the per-example "
                        "forget-gradient second moments E[g g^T] and E[g^T g]; rank m x the plain-basis rank with "
                        "--restriction_rank_multiplier, else per side the EVR rank of its own moment at --restriction_evr")
    p.add_argument("--restriction_forget_output_target", type=float,
                   help="restricted only (extension): per target the smallest rank of its basis family whose "
                        "warm-start delta D changes forget outputs by tr(D C_f D^T) / tr(W C_f W^T) >= this value, "
                        "C_f the forget input covariance over answer tokens; replaces --restriction_rank_multiplier")
    p.add_argument("--component_selection", choices=list(COMPONENT_SELECTIONS),
                   help="restricted only (extension): apply sum_j alpha_j D_j of the components D_j of each projected "
                        "update, c_j = <G_f, D_j>: full (alpha 1, the SVD reconstruction), signed (1 if c_j > 0, else "
                        "0), random (as many ones as signed, drawn with --seed), uniform_matched (the original update "
                        "scaled to the signed update's Frobenius norm, per target); absent = the original update")
    p.add_argument("--component_diagnostics", choices=["yes"],
                   help="restricted only (extension): per-target components (sigma, c_j) to components.json, and the "
                        "forget / retain loss before and after the warm start against <G, Delta W>; with phase two "
                        "on native weights also after phase two: ||theta - theta_Init|| over the targets and over all "
                        "parameters, and <G, theta - theta_Init> with every parameter's Init Fine. mean gradient")
    p.add_argument("--warm_start_scale", type=float,
                   help="restricted only (extension): apply W_Init + t (W* - W_Init) with this t instead of W*, every "
                        "target from its Init weights; t in [0, 1 / (1 - s_min)] (s_min 0.3: t = 1 is W*)")
    p.add_argument("--warm_start_scale_target", choices=["holdout"],
                   help="restricted only (extension, gradient-free): choose t by bisection, with forward passes "
                        "only, so that the forget loss of W_Init + t (W* - W_Init) equals the Init Fine. loss on the "
                        "holdout split (authors the model never saw)")
    p.add_argument("--direction_weighting", choices=["contrast"],
                   help="restricted only (extension, gradient-free): shrink each input direction v_k of the basis by "
                        "s_k = 1 - (1 - s_min) w_k, w_k = forget/retain input-energy share ratio along v_k over its "
                        "mean (clipped so that s_k >= 0)")
    p.add_argument("--representation_mix", type=float,
                   help="basis D only: input-side forget signal (1 - beta) G_f^T G_f / tr + beta C_f / tr; "
                        "1 (default) = activations (D), 0 = gradient (the input side of C)")
    p.add_argument("--phase2_free",
                   help="restricted only: parameters without an adapter that phase two trains freely: none, all, "
                        "or a comma list of norms, attention, mlp (extension; Classification freezes them)")
    # phase two (OpenUnlearning TOFU recipe)
    p.add_argument("--epochs", type=float, required=True, help="0: evaluate the warm start, no phase two")
    p.add_argument("--eval_trajectory", choices=["nll"],
                   help="extension: before phase two and after every epoch, the answer-token NLL and exact "
                        "memorization (teacher-forced, as TOFU's metric_em) of forget, retain_perturbed, real "
                        "authors and world facts, plus the mean training loss of the epoch; random state restored, so "
                        "training is unchanged")
    p.add_argument("--epoch_length", required=True, choices=["forget", "retain"],
                   help="forget: an epoch is len(forget) random retain items (OpenUnlearning anchor=forget); "
                        "retain: an epoch is one pass over the retain split")
    p.add_argument("--per_device_batch_size", type=int, required=True)
    p.add_argument("--grad_accum", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--warmup_epochs", type=float, help="Hugging Face schedulers only")
    p.add_argument("--lr_scheduler", required=True,
                   help="a Hugging Face scheduler name, or piecewise_epochs (linear through --lr_points)")
    p.add_argument("--lr_points", help="piecewise_epochs only: epoch:lr pairs, e.g. 0:0,1:3e-5,5:1e-5,10:1e-5")
    p.add_argument("--optim", required=True)
    p.add_argument("--retain_loss", required=True, choices=["nll", "kl", "nll_kl"],
                   help="kl: KL(Init Fine. || model) per non-padding position of the retain batch")
    p.add_argument("--kl_weight", type=float, help="weight of the KL term (retain_loss kl or nll_kl)")
    p.add_argument("--kl_teacher", help="model id or directory of the KL teacher instead of the base model (user, "
                                        "2026-10-29: e.g. Init Fine. after a plain fine-tuning on retain); kl losses only")
    p.add_argument("--shrink_score", choices=list(SHRINK_SCORES),
                   help="the row score of S (Stage 2, recast_scores.py): original = cos(W_i, -G_r,i); projected_row = "
                        "cos(P(W)_i, -P(G_r)_i); projected_sensitivity = the cosine form of <G_r, P(e_i W_i)>; "
                        "global_projected = cos(P(W), -P(G_r)), one factor per matrix; restricted only; every variant "
                        "but direction_gates")
    p.add_argument("--score_diagnostics", choices=["yes"],
                   help="extension: every rule's score summary, the first-order change of the applied update and the "
                        "update norm on a floor grid, per target (restricted, forget_only source)")
    p.add_argument("--core_diagnostics", choices=["yes"],
                   help="extension (Stage 3, recast_bases.py): per target, the weight core U^T W Vh^T against the update "
                        "core U^T (S - I) W Vh^T for the applied S, each rule's S and a random control S (restricted, "
                        "forget_only source)")
    p.add_argument("--weight_contraction", choices=list(CONTRACTION_MODES),
                   help="extension (recast_contraction.py): each restricted target's update contracts the weight-core "
                        "components of P(W) by --contraction_strength, the largest first (top) or random ones (random, "
                        "drawn with the seed), until the displacement is (1 - floor) ||P(W)||, that of ReCAST at the "
                        "floor (restricted_shrink with a representation basis)")
    p.add_argument("--contraction_strength", type=float,
                   help="beta of every contracted component, in (0, 1]; goes with --weight_contraction")
    p.add_argument("--contraction_residual", type=float,
                   help="the singular value every contracted component keeps instead of 0 (user, 2026-10-28); needs "
                        "--weight_contraction")
    p.add_argument("--contraction_diagnostics", choices=["yes"],
                   help="per target, every weight-core component's kappa, input contrast and forget / retain gradient "
                        "alignment, without changing the update (restricted_shrink with a representation basis)")
    p.add_argument("--output_diagnostics", choices=["yes"],
                   help="per target, the output ratios ||W' a|| / ||W a|| and removed output shares on forget and retain "
                        "inputs of the applied update, of every SVD ordering (with --contraction_diagnostics) and of the "
                        "output-contrast contraction of down_proj and o_proj (recast_output_contrast.py), without changing "
                        "the update (restricted_shrink with a representation basis)")
    p.add_argument("--output_contrast_rho", type=float,
                   help="extension (user, 2026-10-28): replace each restricted target's update by the output-contrast "
                        "contraction at this ridge with the displacement of the ReCAST update (recast_output_contrast.py; "
                        "restricted_shrink with a representation basis)")
    p.add_argument("--family_diagnostics", choices=["yes"],
                   help="for down_proj and o_proj, the retain output share each family of forget-specific directions "
                        "(output, gradient and input contrast; recast_direction_families.py) removes at the forget output "
                        "share of the applied update, without changing the update (restricted_shrink with a "
                        "representation basis)")
    p.add_argument("--family_contraction", choices=["output_contrast", "input_contrast"],
                   help="extension (user, 2026-10-29): replace each restricted target's update by removing the family's "
                        "directions (beta 1, in the order of the contrast) until the forget output share of the ReCAST "
                        "update is removed (recast_direction_families.py); with --family_rho")
    p.add_argument("--family_rho", type=float, help="ridge of --family_contraction")
    p.add_argument("--global_family", choices=list(FAMILIES),
                   help="extension (user, 2026-10-29): replace the updates of all targets (down_proj and o_proj only) "
                        "together: the family's directions of every target in one order (--global_family_order), each "
                        "chosen one kept at --global_family_floor of its part, until the forget output energy removed "
                        "over all targets equals the total the ReCAST updates remove (recast_global_families.py); with "
                        "--global_family_rho, --global_family_floor and --global_family_order")
    p.add_argument("--global_family_rho", type=float, help="ridge of --global_family")
    p.add_argument("--global_family_floor", type=float,
                   help="share of its part each chosen direction keeps (beta = 1 - floor), in [0, 1)")
    p.add_argument("--global_family_order", choices=list(GLOBAL_ORDERS),
                   help="score: by the generalised eigenvalue, highest first; random: the same directions in an order "
                        "drawn with --seed (the control); energy_ratio: by e_f / e_r^alpha, the absolute output "
                        "energies of each direction alone (with --global_family_alpha)")
    p.add_argument("--global_family_budget_unit", choices=list(BUDGET_UNITS),
                   help="energy (default): the budget sums absolute forget output energies (down_proj / o_proj only); "
                        "share (user, 2026-10-31): each target's removed energy as a share of its own forget output "
                        "energy, summed, which admits any target type")
    p.add_argument("--global_family_s_gamma", type=float,
                   help="extension (user, 2026-10-31): each chosen direction keeps clip(lambda^-gamma, "
                        "--global_family_floor, 1) of its part instead of the floor itself (lambda <= 1: untouched)")
    p.add_argument("--global_family_s_rule", choices=["retain_share"],
                   help="extension (user, 2026-10-31): each chosen direction keeps clip(e_r / (e_f + e_r), "
                        "--global_family_floor, 1) of its part, from the absolute output energies of removing it alone")
    p.add_argument("--global_family_s_threshold_gamma", type=float,
                   help="extension (user, 2026-10-31): contract every direction by clip((lambda* / lambda)^gamma, "
                        "--global_family_floor, 1) with one threshold lambda* that meets the budget (score order only)")
    p.add_argument("--global_family_alpha", type=float,
                   help="energy_ratio order only, >= 0: 1 ranks by the forget / retain energy ratio, 0 by the forget "
                        "energy alone (user, 2026-10-29: the trade-off between specificity and absolute forget energy)")
    # direction gates (extension, user 2026-11-03; recast_direction_gates.py)
    p.add_argument("--direction_gates_basis", choices=list(GATE_BASES),
                   help="direction_gates only: the input directions gated per target: input_contrast (the lambda-leading "
                        "directions of C_f v = lambda (C_r + rho / in I) v, orthonormalised by a QR in lambda order), "
                        "input_contrast_dual (the same eigenvectors without the QR, each gate removing exactly the "
                        "coefficient of one of them), protected_contrast / protected_contrast_dual (both with the "
                        "constant prompt prefix's inputs protected and the retain covariance over every non-prefix token, "
                        "user 2026-11-04), forget_input (the leading eigenvectors of C_f, no retain contrast) or random "
                        "(a random orthonormal frame drawn with --seed, the control)")
    p.add_argument("--direction_gates_rho", type=float,
                   help="the contrast bases only (input_contrast[_dual], protected_contrast[_dual]): the ridge rho")
    p.add_argument("--direction_gates_candidates", type=int,
                   help="direction_gates only: k, the directions gated per target (clipped to its input width)")
    p.add_argument("--direction_gates_max_epochs", type=int,
                   help="direction_gates only: cap on the gate-training epochs, each one pass over the forget split "
                        "with a random retain partner per forget item (OpenUnlearning anchor=forget)")
    p.add_argument("--direction_gates_convergence_epochs", type=int,
                   help="direction_gates only, >= 0: the trailing epochs over which convergence is judged (user "
                        "2026-11-03: phase one to convergence); the phase stops at the first epoch whose window meets "
                        "--direction_gates_tol and --direction_gates_drift_max. 0: the fixed schedule (user 2026-11-05), "
                        "exactly --direction_gates_max_epochs epochs, no convergence rule, no _tol and _drift_max")
    p.add_argument("--direction_gates_tol", type=float,
                   help="with the convergence rule, >= 0: largest relative decrease of the probe J over the window")
    p.add_argument("--direction_gates_drift_max", type=float,
                   help="with the convergence rule, in (0, 1]: largest drift ratio ||s_end - s_start|| / sum_t "
                        "||s_t - s_{t-1}|| of the gates over the window (1: every update one way)")
    p.add_argument("--direction_gates_lr_schedule", choices=list(GATE_LR_SCHEDULES),
                   help="direction_gates only (not the calibration): constant, or linear from --direction_gates_lr to 0 "
                        "over the steps of --direction_gates_max_epochs (the fixed schedule only, "
                        "--direction_gates_convergence_epochs 0)")
    p.add_argument("--direction_gates_probe_every", type=int,
                   help="direction_gates only (not the calibration), >= 1: the probe J after every n-th epoch and after "
                        "the last (above 1 only with the fixed schedule; the convergence rule needs it every epoch)")
    p.add_argument("--direction_gates_probe_retain", type=int,
                   help="direction_gates only: retain items (seeded) of the fixed probe on which J is evaluated after "
                        "every epoch, with every forget item")
    p.add_argument("--direction_gates_optim", choices=list(GATE_OPTIMIZERS),
                   help="direction_gates only: adam (betas 0.9 / 0.999, eps 1e-8) or sgd (momentum 0); the lr per "
                        "--direction_gates_lr_schedule")
    p.add_argument("--direction_gates_micro_batch", type=int,
                   help="direction_gates only: forget (and as many retain) items per forward")
    p.add_argument("--direction_gates_grad_accum", type=int,
                   help="direction_gates only: micro-batches per gate step (losses averaged over the window's answer tokens)")
    p.add_argument("--direction_gates_lr", type=float,
                   help="direction_gates only: learning rate of the gates (constant, or the start of the linear "
                        "schedule)")
    p.add_argument("--direction_gates_alpha", type=float,
                   help="direction_gates only: weight of the forget unlikelihood -log(1 - p(y_t)) in J")
    p.add_argument("--direction_gates_beta", type=float,
                   help="direction_gates only: weight of R = ||W_eff - W0||_F^2 / sum ||W0||_F^2 in J")
    p.add_argument("--direction_gates_gamma", type=float,
                   help="direction_gates only, >= 0: weight of the closing term O = sum_j s_j in J (user 2026-11-03: every "
                        "gate pushed shut at the price gamma, held open only by the retain loss); alpha or gamma > 0")
    p.add_argument("--direction_gates_lambda_star", type=float,
                   help="contrast bases only, > 0: gate only the directions with lambda >= lambda* (at most "
                        "--direction_gates_candidates per target; a target without any gets no gates), as the "
                        "selective rule's stop (user 2026-11-04)")
    p.add_argument("--direction_gates_init", type=float,
                   help="direction_gates only (not the calibration), in (0, 1]: the value every gate starts the gate "
                        "phase from, after the identity check at 1 (0.2 with lambda* 1 on protected_contrast is the "
                        "selective rule S2 at floor 0.2)")
    p.add_argument("--direction_gates_cut", choices=["shrink", "rotate"],
                   help="with --direction_gates_binarize_share (the binarized directions), or with "
                        "--direction_gates_max_epochs 0 (every candidate direction: with --direction_gates_lambda_star "
                        "the selective rule's choice, user 2026-11-05): what is done to the chosen directions, shrink "
                        "(to --direction_gates_cut_s, by default --direction_gates_binary_s) or rotate (their parts "
                        "carried by an orthogonal map onto outputs orthogonal to their own span and the forget output "
                        "subspace, user 2026-11-04)")
    p.add_argument("--direction_gates_cut_s", type=float,
                   help="with --direction_gates_cut shrink, in [-1, 1): the share a chosen direction keeps (-1: the part "
                        "reflected; with binarization the choice itself is made at --direction_gates_binary_s); with "
                        "--direction_gates_cut rotate, in (-1, 1): the share c of itself a part keeps along its own "
                        "direction, the rest turned away at the same norm (a partial turn by arccos c; without it the "
                        "quarter turn, c 0)")
    p.add_argument("--direction_gates_rotate_evr", type=float,
                   help="with --direction_gates_cut rotate, in (0, 1): the share of tr(W0 C_f W0^T) the avoided forget "
                        "output subspace holds")
    p.add_argument("--direction_gates_binarize_share", type=float,
                   help="direction_gates only, in (0, 1): after the gate phase the gates choose the directions and the "
                        "cut removes them whole (user 2026-11-04): the most closed gates, as few as remove this share of "
                        "the forget output energy over all targets, set to --direction_gates_binary_s, every other gate "
                        "to 1 (the global selection's budget with the learned order)")
    p.add_argument("--direction_gates_binary_s", type=float,
                   help="with --direction_gates_binarize_share, in [0, 1): the share a removed direction keeps")
    p.add_argument("--direction_gates_calibration_trials",
                   help="direction_gates only: comma-separated optimizer:lr pairs (e.g. adam:1e-5,sgd:1e-3); the run then "
                        "only calibrates the learning rate (gradient census, one short training per pair from s = 1) and "
                        "stops before phase two")
    p.add_argument("--direction_gates_trial_epochs", type=int,
                   help="calibration only: forget-anchored epochs per trial")
    p.add_argument("--direction_gates_census_epochs", type=int,
                   help="calibration only: forget-anchored epochs of windows in the gradient census at s = 1")
    p.add_argument("--direction_gates_autocast", choices=["bfloat16", "none"],
                   help="direction_gates only: autocast of the gate-training forward and backward (the weights stay "
                        "float32)")
    p.add_argument("--write_compensation", choices=["retain"],
                   help="extension (user, 2026-10-28, mechanism test): after the warm start, scale the residual stream at "
                        "every down_proj / o_proj write by that write's retain output ratio c_retain, stored in the "
                        "weights as the write and every later write divided by the running product (RMSNorm reads are "
                        "scale invariant); needs --output_diagnostics")
    p.add_argument("--max_grad_norm", type=float, required=True,
                   help="with --deepspeed_config DeepSpeed clips (its config's gradient_clipping, 0 when absent); the "
                        "run stops unless the two agree")
    p.add_argument("--gradient_checkpointing", required=True, choices=["yes", "no"])
    p.add_argument("--loss_normalization", required=True, choices=["window_tokens", "micro_batch_mean"],
                   help="window_tokens: the retain NLL averaged over the answer tokens of the whole accumulation window "
                        "(transformers >= 4.46); micro_batch_mean: averaged per micro-batch, then over the micro-batches "
                        "(transformers 4.45.1, with which OpenUnlearning trained Init Fine. and Retain)")
    p.add_argument("--deepspeed_config",
                   help="extension: phase two through DeepSpeed in one process with this config (OpenUnlearning's ZeRO "
                        "stage 3 with bf16: float32 master weights, bfloat16 forward and backward); needs --train_dtype "
                        "bfloat16, --retain_loss nll, a variant that trains every parameter and a tied head")
    p.add_argument("--seed", type=int, required=True)
    # evaluation and bookkeeping
    p.add_argument("--eval_batch_size", type=int, required=True)
    p.add_argument("--anchors_raw", required=True, help="raw_metrics.json holding 'Init Fine.' and 'Retain'")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_model_dir", help="extension: save the model after phase two (train dtype) to this directory")
    p.add_argument("--run_name", required=True)
    p.add_argument("--wandb_mode", required=True, choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", required=True)
    p.add_argument("--wandb_tags", nargs="+", required=True)
    args = p.parse_args()
    with_head = TARGET_PROFILES[args.targets][1]
    restricted = args.variant in RESTRICTED_VARIANTS
    gated = args.variant == "direction_gates"
    calibrating = gated and args.direction_gates_calibration_trials is not None
    piecewise = args.lr_scheduler == "piecewise_epochs"
    needed = {  # argument -> whether this configuration uses it
        "untie_lm_head": with_head or args.train_lm_head == "yes",
        "freeze_input_embedding": args.variant == "unrestricted",
        "s_min": not gated,
        "shrink_score": not gated,
        "shrink_layer_scaling": not gated,
        **{f"direction_gates_{k}": gated for k in ("basis", "candidates", "probe_retain", "micro_batch", "grad_accum",
                                                    "alpha", "beta", "gamma", "autocast")},
        **{f"direction_gates_{k}": gated and not calibrating for k in ("max_epochs", "convergence_epochs", "optim",
                                                                        "lr", "init", "lr_schedule", "probe_every")},
        **{f"direction_gates_{k}": gated and not calibrating and args.direction_gates_convergence_epochs != 0
           for k in ("tol", "drift_max")},
        **{f"direction_gates_{k}": calibrating for k in ("trial_epochs", "census_epochs")},
        "direction_gates_rho": args.direction_gates_basis in CONTRAST_BASES,
        "restriction_source": restricted,
        "restriction_evr": restricted,
        "restriction_alignment_scope": restricted,
        "restricted_head": restricted and with_head,
        "phase2_free": args.variant == "restricted_semu",
        "retain_input_filter": restricted,
        "retain_filter_kappa": args.retain_input_filter == "ridge",
        "restriction_whitening": restricted,
        "whitening_rho": args.restriction_whitening == "retain_input" or args.restriction_representation not in (None, "none")
                         or args.restriction_gradient_contrast not in (None, "none"),
        "warmup_epochs": not piecewise,
        "lr_points": piecewise,
        "kl_weight": "kl" in args.retain_loss,
    }
    if args.restriction_rank_multiplier is not None and (not restricted or args.restriction_rank_multiplier <= 0):
        p.error("--restriction_rank_multiplier needs a restricted variant and a positive value")
    if args.restriction_gradient_contrast not in (None, "none") and (
            not restricted or args.restriction_whitening != "none"
            or args.restriction_representation not in (None, "none") or args.retain_input_filter != "none"
            or args.restriction_forget_output_target is not None or args.restriction_source != "forget_only"):
        p.error("--restriction_gradient_contrast needs a restricted variant, --restriction_source forget_only, "
                "--restriction_whitening none, --restriction_representation none, --retain_input_filter none and no "
                "--restriction_forget_output_target")
    if args.restriction_gradient_moment == "per_example" and (
            not restricted or args.restriction_whitening != "none"
            or args.restriction_representation not in (None, "none") or args.retain_input_filter != "none"
            or args.restriction_gradient_contrast not in (None, "none")
            or args.restriction_forget_output_target is not None or args.restriction_source != "forget_only"):
        p.error("--restriction_gradient_moment per_example needs a restricted variant, --restriction_source forget_only, "
                "--restriction_whitening none, --restriction_representation none, --retain_input_filter none and no "
                "gradient contrast or --restriction_forget_output_target")
    own_spectrum = (args.restriction_gradient_contrast not in (None, "none")
                    or args.restriction_gradient_moment == "per_example") and args.restriction_rank_multiplier is None
    if own_spectrum and (args.variant != "restricted_shrink" or args.component_selection or args.component_diagnostics
                         or args.direction_weighting is not None or args.score_diagnostics or args.core_diagnostics):
        p.error("own-spectrum ranks (a gradient-contrast or per-example basis without --restriction_rank_multiplier) need "
                "restricted_shrink and no component selection or diagnostics, direction weighting or score or core "
                "diagnostics (the two sides may get different ranks)")
    if args.gradient_contrast_rank_rule is not None and (
            args.restriction_gradient_contrast in (None, "none") or args.restriction_rank_multiplier is not None):
        p.error("--gradient_contrast_rank_rule needs --restriction_gradient_contrast and no --restriction_rank_multiplier")
    target = args.restriction_forget_output_target
    if target is not None and (not restricted or not 0.0 < target < 1.0 or args.restriction_rank_multiplier is not None
                               or args.retain_input_filter != "none" or args.shrink_layer_scaling != "none"
                               or args.restriction_alignment_scope != "full_space"):
        p.error("--restriction_forget_output_target needs a restricted variant, a value in (0, 1), no "
                "--restriction_rank_multiplier, --retain_input_filter none, --shrink_layer_scaling none and "
                "--restriction_alignment_scope full_space")
    if args.restriction_representation not in (None, "none") and (
            (args.restriction_rank_multiplier is None and target is None) or args.restriction_whitening != "none"):
        p.error("--restriction_representation needs --restriction_rank_multiplier or "
                "--restriction_forget_output_target, and --restriction_whitening none")
    if args.epochs < 0:
        p.error("--epochs must be non-negative (0 = no phase two)")
    if isinstance(args.s_min, dict) and (
            set(args.s_min) != set(TARGET_PROFILES[args.targets][0]) or not all(0.0 <= v <= 1.0 for v in args.s_min.values())
            or args.warm_start_scale is not None or args.warm_start_scale_target is not None
            or args.shrink_layer_scaling != "none"):
        p.error(f"per-type --s_min needs one floor in [0, 1] for every projection type of --targets {args.targets}, no "
                "--warm_start_scale / --warm_start_scale_target and --shrink_layer_scaling none")
    if args.warm_start_scale is not None and (args.warm_start_scale_target is not None or not restricted
                                              or not 0.0 <= args.warm_start_scale
                                              <= (1.0 / (1.0 - args.s_min) if args.s_min < 1.0 else 0.0)):
        p.error("--warm_start_scale needs a restricted variant, no --warm_start_scale_target and a value in "
                "[0, 1 / (1 - s_min)]")
    if args.warm_start_scale_target is not None and (not restricted or args.s_min >= 1.0):
        p.error("--warm_start_scale_target needs a restricted variant and s_min < 1")
    if args.direction_weighting is not None and (not restricted or args.retain_input_filter != "none"
                                                 or args.component_selection or args.component_diagnostics):
        p.error("--direction_weighting needs a restricted variant, --retain_input_filter none and no component "
                "selection or diagnostics")
    if (args.component_selection or args.component_diagnostics) and (
            not restricted or args.restriction_source != "forget_only" or args.retain_input_filter != "none"):
        p.error("--component_selection / --component_diagnostics need a restricted variant, --restriction_source "
                "forget_only (c_j uses the forget gradient) and --retain_input_filter none")
    if (args.shrink_score not in (None, "original") or args.score_diagnostics or args.core_diagnostics) and (
            not restricted or args.restriction_alignment_scope != "full_space" or args.shrink_layer_scaling != "none"
            or args.retain_input_filter != "none" or args.direction_weighting is not None or args.component_selection
            or args.restriction_forget_output_target is not None or args.restriction_source != "forget_only"):
        p.error("--shrink_score other than original and --score_diagnostics need a restricted variant, full_space "
                "alignment, --shrink_layer_scaling none, --retain_input_filter none, --restriction_source forget_only "
                "and no direction weighting, component selection or forget-output target")
    if (args.weight_contraction is None) != (args.contraction_strength is None):
        p.error("--weight_contraction and --contraction_strength go together")
    if args.weight_contraction is not None and (
            args.variant != "restricted_shrink" or args.restriction_representation in (None, "none")
            or not 0.0 < args.contraction_strength <= 1.0 or args.restriction_alignment_scope != "full_space"
            or args.shrink_layer_scaling != "none" or args.retain_input_filter != "none"
            or args.direction_weighting is not None or args.component_selection or args.component_diagnostics
            or args.restriction_forget_output_target is not None or args.shrink_score != "original"
            or args.score_diagnostics or args.core_diagnostics):
        p.error("--weight_contraction needs restricted_shrink with a representation basis, a strength in (0, 1], "
                "full_space alignment, --shrink_layer_scaling none, --retain_input_filter none, the original shrink "
                "score and no direction weighting, component selection, forget-output target or diagnostics")
    if args.contraction_residual is not None and (args.weight_contraction is None or args.contraction_residual < 0):
        p.error("--contraction_residual needs --weight_contraction and a value >= 0")
    if args.contraction_diagnostics and (
            args.variant != "restricted_shrink" or args.restriction_representation in (None, "none")
            or args.restriction_alignment_scope != "full_space" or args.shrink_layer_scaling != "none"
            or args.retain_input_filter != "none" or args.direction_weighting is not None or args.component_selection
            or args.restriction_forget_output_target is not None):
        p.error("--contraction_diagnostics needs restricted_shrink with a representation basis, full_space alignment, "
                "--shrink_layer_scaling none, --retain_input_filter none and no direction weighting, component "
                "selection or forget-output target")
    if args.output_diagnostics and (args.variant != "restricted_shrink"
                                    or args.restriction_representation in (None, "none")):
        p.error("--output_diagnostics needs restricted_shrink with a representation basis (both input covariances)")
    if args.output_contrast_rho is not None and (
            args.variant != "restricted_shrink" or args.restriction_representation in (None, "none")
            or args.output_contrast_rho <= 0 or args.weight_contraction or args.direction_weighting is not None
            or args.component_selection or args.retain_input_filter != "none"
            or args.restriction_forget_output_target is not None or args.shrink_score != "original"):
        p.error("--output_contrast_rho needs restricted_shrink with a representation basis, a positive value, the "
                "original shrink score and no weight contraction, direction weighting, component selection, input "
                "filter or forget-output target")
    if args.family_diagnostics and (args.variant != "restricted_shrink"
                                    or args.restriction_representation in (None, "none")):
        p.error("--family_diagnostics needs restricted_shrink with a representation basis (both input covariances)")
    if (args.family_contraction is None) != (args.family_rho is None) or (args.family_contraction is not None and (
            args.variant != "restricted_shrink" or args.restriction_representation in (None, "none")
            or args.family_rho <= 0 or args.weight_contraction or args.output_contrast_rho is not None
            or args.direction_weighting is not None or args.component_selection or args.retain_input_filter != "none"
            or args.restriction_forget_output_target is not None or args.shrink_score != "original")):
        p.error("--family_contraction and --family_rho go together and need restricted_shrink with a representation "
                "basis, a positive ridge, the original shrink score and no other update replacement, direction "
                "weighting, component selection, input filter or forget-output target")
    global_args = (args.global_family, args.global_family_rho, args.global_family_floor, args.global_family_order)
    if (any(a is not None for a in global_args) or args.global_family_alpha is not None
            or args.global_family_budget_unit is not None or args.global_family_s_gamma is not None
            or args.global_family_s_rule is not None or args.global_family_s_threshold_gamma is not None) and (
            (args.global_family_s_threshold_gamma is not None and (
                args.global_family_s_threshold_gamma <= 0 or args.global_family_s_gamma is not None
                or args.global_family_s_rule is not None or args.global_family_order != "score")) or
            (args.global_family_s_gamma is not None and args.global_family_s_gamma <= 0)
            or (args.global_family_s_rule is not None and args.global_family_s_gamma is not None) or
            any(a is None for a in global_args) or args.variant != "restricted_shrink"
            or args.restriction_representation in (None, "none") or args.global_family_rho <= 0
            or not 0.0 <= args.global_family_floor < 1.0 or TARGET_PROFILES[args.targets][1]
            or (args.global_family_order == "energy_ratio") != (args.global_family_alpha is not None)
            or (args.global_family_alpha is not None and args.global_family_alpha < 0)
            or ((args.global_family_budget_unit or "energy") == "energy"
                and not set(TARGET_PROFILES[args.targets][0]) <= {"down_proj", "o_proj"})
            or args.family_contraction or args.weight_contraction or args.output_contrast_rho is not None
            or args.direction_weighting is not None or args.component_selection or args.retain_input_filter != "none"
            or args.restriction_forget_output_target is not None or args.shrink_score != "original"):
        p.error("--global_family, --global_family_rho, --global_family_floor and --global_family_order go together and "
                "need restricted_shrink with a representation basis, a positive ridge, a floor in [0, 1), "
                "--global_family_alpha >= 0 with, and only with, the energy_ratio order, down_proj / "
                "o_proj targets only unless --global_family_budget_unit share, the original shrink score and no "
                "other update replacement, direction "
                "weighting, component selection, input filter or forget-output target")
    if args.direction_gates_calibration_trials is not None:
        if not gated:
            p.error("--direction_gates_calibration_trials needs --variant direction_gates")
        try:
            parse_gate_trials(args.direction_gates_calibration_trials)
        except ValueError as err:
            p.error(f"--direction_gates_calibration_trials: {err}")
    positive = [getattr(args, f"direction_gates_{k}") for k in ("candidates", "probe_every",
                                                                 "probe_retain", "micro_batch", "grad_accum", "lr",
                                                                 "rho", "trial_epochs", "census_epochs", "lambda_star")]
    nonnegative = [args.direction_gates_alpha, args.direction_gates_beta, args.direction_gates_tol,
                   args.direction_gates_gamma, args.direction_gates_max_epochs, args.direction_gates_convergence_epochs]
    if args.direction_gates_convergence_epochs != 0 and (
            args.direction_gates_lr_schedule not in (None, "constant") or args.direction_gates_probe_every not in (None, 1)):
        p.error("--direction_gates_lr_schedule linear and --direction_gates_probe_every above 1 need the fixed schedule "
                "(--direction_gates_convergence_epochs 0): the convergence rule judges the probe J of every epoch at "
                "a constant lr")
    if args.direction_gates_lambda_star is not None and args.direction_gates_basis not in CONTRAST_BASES:
        p.error(f"--direction_gates_lambda_star goes only with the bases {CONTRAST_BASES}")
    if args.direction_gates_init is not None and not 0 < args.direction_gates_init <= 1:
        p.error("--direction_gates_init must lie in (0, 1]")
    cut, cut_s = args.direction_gates_cut, args.direction_gates_cut_s
    if ((cut is not None and args.direction_gates_binarize_share is None
         and (not gated or calibrating or args.direction_gates_max_epochs != 0))
            or (cut_s is not None and not ((cut == "shrink" and -1 <= cut_s < 1) or (cut == "rotate" and -1 < cut_s < 1)))
            or ((cut == "rotate") != (args.direction_gates_rotate_evr is not None))
            or (args.direction_gates_rotate_evr is not None and not 0 < args.direction_gates_rotate_evr < 1)
            or (cut == "shrink" and args.direction_gates_binarize_share is None and cut_s is None)):
        p.error("--direction_gates_cut needs --direction_gates_binarize_share or --direction_gates_max_epochs 0 (every "
                "candidate cut; a shrink then needs --direction_gates_cut_s); --direction_gates_cut_s goes with "
                "--direction_gates_cut shrink (in [-1, 1)) or rotate (in (-1, 1)); --direction_gates_rotate_evr (in "
                "(0, 1)) with, and only with, --direction_gates_cut rotate")
    binarize = (args.direction_gates_binarize_share, args.direction_gates_binary_s)
    if (binarize.count(None) == 1 or (binarize[0] is not None and (not gated or calibrating
                                                                     or not 0 < binarize[0] < 1
                                                                     or not 0 <= binarize[1] < 1))):
        p.error("--direction_gates_binarize_share (in (0, 1)) and --direction_gates_binary_s (in [0, 1)) go together, "
                "with --variant direction_gates and without the calibration")
    if gated and None not in (args.direction_gates_alpha, args.direction_gates_gamma) and (
            args.direction_gates_alpha + args.direction_gates_gamma <= 0):
        p.error("direction_gates needs --direction_gates_alpha or --direction_gates_gamma positive (something must "
                "push the gates shut)")
    if gated and (args.phase_one_dtype != "float32" or with_head or any(v is not None and v <= 0 for v in positive)
                  or any(v is not None and v < 0 for v in nonnegative)
                  or (args.direction_gates_drift_max is not None and not 0 < args.direction_gates_drift_max <= 1)
                  or (None not in (args.direction_gates_max_epochs, args.direction_gates_convergence_epochs)
                      and args.direction_gates_max_epochs > 0
                      and args.direction_gates_convergence_epochs > args.direction_gates_max_epochs)):
        p.error("direction_gates needs --phase_one_dtype float32 (the gated weights are written once, in float32), "
                "targets without the head, positive --direction_gates_candidates, _probe_every, _probe_retain, "
                "_micro_batch, _grad_accum, _lr, _rho and _lambda_star, --direction_gates_max_epochs >= 0, "
                "_convergence_epochs >= 0 (at most _max_epochs unless that is 0), "
                "--direction_gates_alpha, _beta and _tol >= 0 and --direction_gates_drift_max in (0, 1]")
    if args.kl_teacher is not None and "kl" not in args.retain_loss:
        p.error("--kl_teacher needs --retain_loss kl or nll_kl")
    if args.write_compensation and (not args.output_diagnostics or args.warm_start_scale is not None
                                    or args.warm_start_scale_target is not None or args.epochs == 0):
        p.error("--write_compensation needs --output_diagnostics (the retain output ratios), no warm-start scale and a "
                "phase two")
    if args.deepspeed_config is not None and (args.train_dtype != "bfloat16" or args.retain_loss != "nll"
                                              or args.variant not in ("unrestricted", "restricted_shrink")
                                              or args.untie_lm_head == "yes" or args.epochs == 0):
        p.error("--deepspeed_config needs --train_dtype bfloat16, --retain_loss nll, the unrestricted or restricted_shrink "
                "variant, a tied head (--untie_lm_head no) and a phase two (--epochs > 0)")
    if args.eval_trajectory and (args.epochs == 0 or args.epochs != int(args.epochs)):
        p.error("--eval_trajectory needs a positive whole number of --epochs")
    if args.representation_mix is not None and (args.restriction_representation in (None, "none")
                                                or not 0.0 <= args.representation_mix <= 1.0):
        p.error("--representation_mix needs --restriction_representation and a value in [0, 1]")
    missing = [k for k, use in needed.items() if use and getattr(args, k) is None]
    unused = [k for k, use in needed.items() if not use and getattr(args, k) is not None]
    if missing:
        p.error(f"{args.variant} with --targets {args.targets} needs --{', --'.join(missing)}")
    if unused:
        p.error(f"{args.variant} with --targets {args.targets} does not use --{', --'.join(unused)}; do not pass them")
    if (args.untie_lm_head == "no" and args.train_lm_head == "yes" and args.freeze_input_embedding == "yes"):
        p.error("a tied head is the input embedding: --train_lm_head yes contradicts --freeze_input_embedding yes")
    if args.shrink_layer_scaling not in (None, "none") and not restricted:
        p.error("--shrink_layer_scaling other than none needs the restricted variant")
    if piecewise:
        peak = max(y for _, y in parse_points(args.lr_points))
        if abs(peak - args.lr) > 1e-12 * max(1.0, args.lr):
            p.error(f"--lr must equal the peak of --lr_points ({peak:g}), got {args.lr:g}")
    return args


def summarize_by_projection(report):
    """Aggregate the per-target warm-start report over the seven projection types."""
    out = {}
    for proj in PROJECTIONS:
        rows = [v for k, v in report.items() if k.endswith(proj)]
        if not rows:
            continue
        agg = {"targets": len(rows),
               "shrink_mean": sum(r["shrink_mean"] for r in rows) / len(rows),
               "shrink_at_floor": sum(r["shrink_at_floor"] for r in rows) / len(rows),
               "alignment_mean": sum(r["alignment_mean"] for r in rows) / len(rows),
               "weight_rel_change_mean": sum(r["weight_rel_change"] for r in rows) / len(rows),
               "floor_min": min(r["floor"] for r in rows), "floor_max": max(r["floor"] for r in rows)}
        kept = [r["retain_filter_kept"] for r in rows if "retain_filter_kept" in r]
        if kept:
            agg.update({"retain_filter_kept_mean": sum(kept) / len(kept), "retain_filter_kept_min": min(kept)})
        if "rank" in rows[0]:
            ranks = sorted(r["rank"] for r in rows)
            agg.update({"rank_min": ranks[0], "rank_median": ranks[len(ranks) // 2], "rank_max": ranks[-1],
                        "total_rank": rows[0]["total_rank"], "inactive": sum(not r["active"] for r in rows),
                        "kept_fraction_of_proposal": sum(r["applied_norm"] for r in rows)
                        / max(sum(r["proposal_norm"] for r in rows), 1e-12)})
            if "reference_rank" in rows[0]:
                refs = sorted(r["reference_rank"] for r in rows)
                agg["reference_rank_median"] = refs[len(refs) // 2]
        out[proj] = agg
    return out


def main():
    args = parse_args()
    import torch
    import transformers
    from datasets import load_dataset
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    import engram
    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    timings = {}
    runtime = {
        "git": git_info(repo), "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "engram": f"{engram.__version__} from {engram.__file__}", "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        # the git commit alone does not identify uncommitted script files
        "sha256": {f: hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), "rb").read()).hexdigest()
                   for f in ("recast_core.py", "recast_scores.py", "recast_bases.py", "recast_contraction.py",
                             "recast_output_contrast.py", "recast_direction_families.py", "recast_global_families.py",
                             "recast_direction_gates.py",
                             "recast_tofu.py", "table3_tofu.py", "tofu_capture.py")},
    }
    print(f"[run] args={json.dumps(vars(args))}")
    print(f"[run] runtime={json.dumps(runtime)}")

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.run_name, tags=args.wandb_tags,
                     mode=args.wandb_mode, config={**vars(args), "runtime": runtime})
    print(f"[wandb] mode={args.wandb_mode} url={getattr(run, 'url', None)}")
    results = {"args": vars(args), "runtime": runtime, "timings_s": timings}

    def save_results():
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(to_jsonable(results), f, indent=2)

    # ---------------- data ----------------
    with Stage("load datasets", timings):
        forget = load_dataset(args.dataset_id, args.forget_split)["train"]
        retain = load_dataset(args.dataset_id, args.retain_split)["train"]
        D = {"fp": load_dataset(args.dataset_id, f"{args.forget_split}_perturbed")["train"],
             "ho": load_dataset(args.dataset_id, args.holdout_split)["train"],
             "rp": load_dataset(args.dataset_id, "retain_perturbed")["train"],
             "ra": load_dataset(args.dataset_id, "real_authors_perturbed")["train"],
             "wf": load_dataset(args.dataset_id, "world_facts_perturbed")["train"]}
        print(f"[data] forget={len(forget)} retain={len(retain)} eval={({k: len(v) for k, v in D.items()})}")
    with open(args.anchors_raw) as f:
        anchors = json.load(f)
    ft, rt = anchors["Init Fine."], anchors["Retain"]
    anchor_check = {k: T.evaluate_scores(v, rt, ft) for k, v in (("Init Fine.", ft), ("Retain", rt))}
    print(f"[anchors] {args.anchors_raw}: Init Fine. Overall={anchor_check['Init Fine.']['Overall']:.4f} "
          f"Retain Overall={anchor_check['Retain']['Overall']:.4f}")

    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"[override] tokenizer pad_token was None -> eos_token {tok.eos_token!r} (as the Table 3 run)")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    model = (AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=dtype[args.phase_one_dtype],
                                                  attn_implementation=args.attn_implementation).to(device))
    print(f"[model] {args.base_id} dtype={args.phase_one_dtype} (phase one), {args.train_dtype} (phase two) "
          f"tie_word_embeddings={model.config.tie_word_embeddings}")
    if args.untie_lm_head == "yes":
        untie = untie_output_embedding(model)
        print(f"[untie] lm_head gets its own copy of the input embedding: {json.dumps(untie)}")
        results["untie"] = untie
    head = model.get_output_embeddings().weight
    print(f"[model] lm_head shares the input embedding tensor: "
          f"{head.data_ptr() == model.get_input_embeddings().weight.data_ptr()}")

    # ---------------- phase one ----------------
    targets = select_targets(model, args.targets, blocks=parse_blocks(args.target_blocks))
    restricted = args.variant in RESTRICTED_VARIANTS
    gated = args.variant == "direction_gates"
    calibrating = gated and args.direction_gates_calibration_trials is not None
    head_unrestricted = restricted and args.restricted_head == "unrestricted"
    unrestricted_targets = {"lm_head"} if head_unrestricted else set()
    print(f"[targets] profile={args.targets} blocks={args.target_blocks}: {len(targets)} weights "
          f"({({p: sum(k.endswith(p) for k in targets) for p in PROJECTIONS if any(k.endswith(p) for k in targets)})}); "
          f"outside the restriction: {sorted(unrestricted_targets) or 'none'}")

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def gradient_loader(ds):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.grad_batch_size,
                          collate_fn=T.Collator(tok, "right"))

    grad_retain, n_retain = None, None
    if not gated:  # the gates need no ReCAST update, hence no source gradients
        with Stage("retain gradient", timings):
            grad_retain, n_retain = collect_mean_gradients(model, gradient_loader(retain), targets, feats)
        print(f"[grad] retain: examples={n_retain} global_norm={sum(float(g.square().sum()) for g in grad_retain.values()) ** 0.5:.4e}")
    grad_forget = None
    if restricted:
        restricted_targets = {n: m for n, m in targets.items() if n not in unrestricted_targets}
        with Stage("forget gradient", timings):
            grad_forget, n_forget = collect_mean_gradients(model, gradient_loader(forget), restricted_targets, feats)
        print(f"[grad] forget ({len(restricted_targets)} restricted targets): examples={n_forget} "
              f"global_norm={sum(float(g.square().sum()) for g in grad_forget.values()) ** 0.5:.4e}")
    moments = None
    if restricted and args.restriction_gradient_moment == "per_example":
        with Stage("forget per-example gradient moments", timings):
            moments, n_moment = collect_example_gradient_moments(model, gradient_loader(forget), restricted_targets, feats)
        print(f"[gmoment] per-example forget-gradient moments: examples={n_moment}")

    input_filters, whitening, covs, covs_forget = None, None, None, None
    representation = restricted and args.restriction_representation not in (None, "none")
    contrast = restricted and (args.direction_weighting == "contrast" or args.component_selection == "contrast")
    if restricted and (args.retain_input_filter == "ridge" or args.restriction_whitening == "retain_input" or representation
                       or contrast):
        with Stage("retain input covariance", timings):
            covs, n_tokens = collect_input_covariances(model, gradient_loader(retain), restricted_targets, feats)
        results["retain_input_covariance"] = {"answer_tokens": n_tokens}
    target = args.restriction_forget_output_target if restricted else None
    if representation or target is not None or contrast:
        with Stage("forget input covariance", timings):
            covs_forget, n_tokens_forget = collect_input_covariances(model, gradient_loader(forget), restricted_targets, feats)
        results["forget_input_covariance"] = {"answer_tokens": n_tokens_forget}
    if gated:
        with Stage("retain input covariance", timings):
            covs, n_tokens = collect_input_covariances(model, gradient_loader(retain), targets, feats)
        with Stage("forget input covariance", timings):
            covs_forget, n_tokens_forget = collect_input_covariances(model, gradient_loader(forget), targets, feats)
        results["retain_input_covariance"] = {"answer_tokens": n_tokens}
        results["forget_input_covariance"] = {"answer_tokens": n_tokens_forget}
        print(f"[gates] input covariances over answer tokens: retain {n_tokens}, forget {n_tokens_forget}")
        # The constant prompt prefix (recast_selective, user 2026-11-04): its inputs for every gated run (the prefix
        # change of the gates is reported), the protected span and the non-prefix retain covariance for the protected
        # bases.
        import recast_selective as S
        items = [T.QADataset(ds, tok, akey="answer")[i]["input_ids"].tolist() for ds in (forget, retain)
                 for i in range(len(ds))]
        prefix_len = S.common_prefix_length(items)
        del items
        with Stage("prefix inputs", timings):
            _, _, prefix_inputs, prefix_dev = S.collect_position_statistics(
                model, gradient_loader(forget.select(range(min(len(forget), 64)))), targets, feats, prefix_len,
                ("answer",))
        prefix_dev = max(prefix_dev.values())
        if prefix_dev > 1e-3:
            raise RuntimeError(f"the prefix inputs differ across sequences (relative {prefix_dev:.2e})")
        print(f"[control] constant prefix: {prefix_len} tokens, its inputs equal across sequences within "
              f"{prefix_dev:.2e} (64 forget items)")
        covs_nonprefix, protected = None, {}
        if args.direction_gates_basis in PROTECTED_BASES:
            with Stage("retain non-prefix covariance", timings):
                covs_np, cnt_np, pre_r, dev_r = S.collect_position_statistics(
                    model, gradient_loader(retain), targets, feats, prefix_len, ("nonprefix",))
            same = max(float((pre_r[n] - prefix_inputs[n]).abs().max() / prefix_inputs[n].abs().max()) for n in targets)
            if max(max(dev_r.values()), same) > 1e-3:
                raise RuntimeError(f"the prefix inputs differ: within retain {max(dev_r.values()):.2e}, forget vs retain "
                                   f"{same:.2e}")
            covs_nonprefix = {n: c["nonprefix"].to(device) for n, c in covs_np.items()}
            del covs_np
            ranks = {}
            for n, t in targets.items():
                protected[n], info = S.protected_span(t.in_features, prefix_inputs=prefix_inputs[n].to(device))
                ranks[n] = info["prefix_rank"]
            results["retain_nonprefix_covariance"] = {"tokens": cnt_np["nonprefix"], "prefix_len": prefix_len,
                                                      "prefix_rank": ranks}
            print(f"[gates] protected bases: retain covariance over {cnt_np['nonprefix']} non-prefix tokens (question "
                  f"and answer); protected span = the prefix inputs, rank {min(ranks.values())}-{max(ranks.values())} "
                  f"per target; prefix inputs equal forget vs retain within {same:.2e}")
    if representation:
        mix = 1.0 if args.representation_mix is None else args.representation_mix
        print(f"[repr] restriction bases from N v = gamma (C_r + eps I) v, N = {1 - mix:g} G_f^T G_f / tr + {mix:g} "
              f"C_f / tr, eps = rho tr(C_r) / d, rho={args.whitening_rho:g}, output side "
              f"{args.restriction_representation}; answer tokens retain {n_tokens}, forget {n_tokens_forget}")
    if restricted and args.restriction_whitening == "retain_input":
        whitening = covs
        print(f"[whiten] restriction bases from G_f (C_r + eps I)^-1/2, eps = rho tr(C_r) / d, "
              f"rho={args.whitening_rho:g}, retain answer tokens={n_tokens}")
    if restricted and args.retain_input_filter == "ridge":
        input_filters, filter_report = {}, {}
        for n, c in covs.items():
            input_filters[n], mu = ridge_input_filter(c, args.retain_filter_kappa)
            # kept weight on average over retain's input energy vs over a uniform direction
            filter_report[n] = {"mu": mu, "kept_retain_weighted": float((input_filters[n] * c).sum() / torch.trace(c)),
                                "kept_uniform": float(torch.trace(input_filters[n]) / c.shape[0])}
        # Positive control: the filter keeps less along the directions retain uses than on average.
        bad = [n for n, r in filter_report.items() if not r["kept_retain_weighted"] < r["kept_uniform"]]
        if bad:
            raise RuntimeError(f"ridge filter does not favour retain directions for {bad[:3]}")
        by_proj = {p: [r for n, r in filter_report.items() if n.endswith(p)] for p in PROJECTIONS}
        print(f"[filter] ridge kappa={args.retain_filter_kappa:g}, retain answer tokens={n_tokens}: " + ", ".join(
            f"{p} kept retain-weighted {sum(r['kept_retain_weighted'] for r in rs) / len(rs):.3f} "
            f"uniform {sum(r['kept_uniform'] for r in rs) / len(rs):.3f}" for p, rs in by_proj.items() if rs))
        results["retain_filter"] = {"answer_tokens": n_tokens, "per_target": filter_report}

    original = {n: m.weight.detach().to("cpu", copy=True) for n, m in targets.items()}  # drift reference

    @torch.no_grad()
    def mean_loss(ds):
        """The loss the source gradients differentiate: the example-weighted mean of each batch's mean NLL."""
        was_training = model.training
        model.eval()
        total, count = 0.0, 0
        for b in gradient_loader(ds):
            x = feats(b)
            size = int(x["input_ids"].shape[0])
            total += float(model(**x).loss) * size
            count += size
        model.train(was_training)
        return total / count

    losses_before = None
    if args.component_diagnostics:
        with Stage("losses before the warm start", timings):
            losses_before = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
    holdout_init = None
    if args.warm_start_scale_target == "holdout":
        with Stage("holdout loss at Init Fine.", timings):
            holdout_init = mean_loss(D["ho"])
        print(f"[scale] target: Init Fine. loss on the holdout split {holdout_init:.4f}")
    full_grads, init_state = None, None
    if args.component_diagnostics and args.epochs > 0 and args.variant != "restricted_semu":
        def full_gradient(ds):
            """Every parameter's mean gradient, defined as the source gradients (eval mode, example-weighted mean
            of batch-mean losses); accumulated on the GPU, returned on the CPU."""
            was_training = model.training
            model.eval()
            params = dict(model.named_parameters())
            flags = {n: t.requires_grad for n, t in params.items()}
            for t in params.values():
                t.requires_grad_(True)
            sums = {n: torch.zeros_like(t, dtype=torch.float32) for n, t in params.items()}
            count = 0
            for b in gradient_loader(ds):
                x = feats(b)
                size = int(x["input_ids"].shape[0])
                model.zero_grad(set_to_none=True)
                model(**x).loss.backward()
                for n, t in params.items():
                    if t.grad is not None:
                        sums[n].add_(t.grad.float(), alpha=float(size))
                count += size
            model.zero_grad(set_to_none=True)
            for n, t in params.items():
                t.requires_grad_(flags[n])
            model.train(was_training)
            return {n: (v / count).cpu() for n, v in sums.items()}

        with Stage("full gradients at Init Fine.", timings):
            full_grads = {"forget": full_gradient(forget), "retain": full_gradient(retain)}
        init_state = {n: t.detach().to("cpu", copy=True) for n, t in model.named_parameters()}
        torch.cuda.empty_cache()
    source = grad_forget
    if restricted and args.restriction_source == "full_data":
        source = {n: mix_sources(grad_forget[n], grad_retain[n], n_forget, n_retain) for n in grad_forget}
        print(f"[source] full_data: (n_f G_f + n_r G_r) / (n_f + n_r) with n_f={n_forget}, n_r={n_retain}")
    gate_summary = None
    if gated:
        import recast_direction_gates as RG
        restrictions = {}
        gen = torch.Generator().manual_seed(args.seed) if args.direction_gates_basis == "random" else None
        with Stage("direction gates: bases", timings):
            directions, scores, writes, specificity = {}, {}, {}, {}
            pencil_retain = covs_nonprefix if covs_nonprefix is not None else covs
            no_gates = []
            for n in targets:
                v, sc, write = RG.gate_directions(covs_forget[n], pencil_retain[n], args.direction_gates_basis,
                                                  args.direction_gates_candidates, rho=args.direction_gates_rho,
                                                  generator=gen, protected=protected.get(n),
                                                  lambda_star=args.direction_gates_lambda_star)
                if v.shape[1] == 0:
                    no_gates.append(n)
                    continue
                directions[n], scores[n] = v, sc
                if write is not None:
                    writes[n] = write
                if args.direction_gates_basis in CONTRAST_BASES:
                    specificity[n] = RG.column_specificity(directions[n], covs_forget[n], pencil_retain[n],
                                                           args.direction_gates_rho, scores[n])
            gates = RG.DirectionGates(targets, directions, writes)
            # the write directions (u = v for an orthonormal basis), kept for the regrowth report after phase two
            write_dirs = {n: (writes[n] if n in writes else directions[n]).float().cpu() for n in directions}
            del directions, writes, pencil_retain, covs_nonprefix
        n_gates = sum(g.numel() for g in gates.gates)
        k_by_type = {}
        for n, g in zip(gates.names, gates.gates):
            k_by_type.setdefault(n.rsplit(".", 1)[-1], []).append(g.numel())
        k_by_type = {t: f"{sum(v)} ({min(v)}-{max(v)} per target)" for t, v in k_by_type.items()}
        if args.direction_gates_lambda_star is not None:
            print(f"[gates] lambda* {args.direction_gates_lambda_star:g}: gates on the directions with lambda >= lambda* "
                  f"(at most {args.direction_gates_candidates} per target): {json.dumps(k_by_type)}; targets without "
                  f"any: {no_gates or 'none'}")
        if calibrating:
            schedule = (f"calibration: trials {args.direction_gates_calibration_trials}, "
                        f"{args.direction_gates_trial_epochs} epochs each from s = 1, gradient census over "
                        f"{args.direction_gates_census_epochs} epochs")
        elif args.direction_gates_convergence_epochs == 0:
            schedule = (f"{args.direction_gates_optim} lr {args.direction_gates_lr:g} ("
                        + ("constant" if args.direction_gates_lr_schedule == "constant" else "linear to 0")
                        + f"), fixed schedule of {args.direction_gates_max_epochs} epochs (no convergence rule), probe "
                        f"after every {args.direction_gates_probe_every}. and the last epoch")
        else:
            schedule = (f"{args.direction_gates_optim} lr {args.direction_gates_lr:g} (constant), to convergence over "
                        f"{args.direction_gates_convergence_epochs} epochs (probe J decrease <= "
                        f"{args.direction_gates_tol:g} and drift <= {args.direction_gates_drift_max:g}), at most "
                        f"{args.direction_gates_max_epochs} epochs")
        print(f"[gates] basis {args.direction_gates_basis}"
              + (f" (rho {args.direction_gates_rho:g})" if args.direction_gates_rho is not None else "")
              + f": {n_gates} gates over {len(gates.names)} targets, k per type {json.dumps(k_by_type)}; "
              f"J = L_R + {args.direction_gates_alpha:g} L_F + {args.direction_gates_gamma:g} O + "
              f"{args.direction_gates_beta:g} R; {schedule}; windows of {args.direction_gates_micro_batch} x "
              f"{args.direction_gates_grad_accum} forget items, autocast {args.direction_gates_autocast}; probe: every "
              f"forget item and {args.direction_gates_probe_retain} retain items")
        spec_summary = {n: {k: v for k, v in r.items() if k != "own_ratio"} for n, r in specificity.items()}
        for typ in sorted({n.rsplit(".", 1)[-1] for n in specificity}):
            rows = [r for n, r in spec_summary.items() if n.endswith(typ)]
            rank = [r["spearman_with_lambda"] for r in rows if r.get("spearman_with_lambda") is not None]
            print(f"[gates] {typ} column specificity: own ratio r_j = v^T C_f v / v^T B v against lambda_j, Spearman "
                  + (f"{min(rank):.3f}-{max(rank):.3f}" if rank else "-")
                  + f"; columns with lambda > 1: {sum(r['lambda_above_1'] for r in rows)}, of them r <= 1: "
                  f"{sum(r['lambda_above_1_own_at_most_1'] for r in rows)}; max |r - lambda| / lambda "
                  f"{max(r['max_rel_gap_to_lambda'] for r in rows):.3e}; off-diagonal share of V^T C_f V "
                  f"{min(r['offdiag_share'] for r in rows):.3f}-{max(r['offdiag_share'] for r in rows):.3f}")
        forget_items, retain_items = T.QADataset(forget, tok, akey="answer"), T.QADataset(retain, tok, akey="answer")
        gate_collator = T.Collator(tok, "right")
        probe = feats(gate_collator([forget_items[i] for i in range(args.direction_gates_micro_batch)]))
        autocast = torch.bfloat16 if args.direction_gates_autocast == "bfloat16" else None
        for ac in {autocast, None}:
            RG.check_identity(model, gates, probe, ac)
        print("[control] gates at 1: logits bitwise equal to Init Fine. on a forget micro-batch "
              f"(autocast {args.direction_gates_autocast} and none)")
        if not calibrating and args.direction_gates_init != 1.0:
            with torch.no_grad():
                for g in gates.gates:
                    g.fill_(args.direction_gates_init)
            print(f"[gates] every gate starts at {args.direction_gates_init:g}")
        checksums_before = RG.weight_checksums(model)
        steps_per_epoch = math.ceil(len(forget) / (args.direction_gates_micro_batch * args.direction_gates_grad_accum))
        gate_config = {k: v for k, v in vars(args).items() if k.startswith("direction_gates")}

        def fmt(v, spec):
            return "-" if v is None else format(v, spec)

        def loggers(label, prefix, max_epochs):
            clock = time.time()

            def log_step(rec):
                if rec["step"] <= 3:  # the scale of each term's gradient against the learning rate
                    print(f"[{label}] step {rec['step']}: window L_F {rec['forget_unlikelihood']:.4f} L_R "
                          f"{rec['retain_nll']:.4f} R {rec['regulariser']:.3e} O {rec['openness']:.1f}; ||grad|| "
                          f"alpha L_F {rec['grad_forget_l2']:.3e} / L_R {rec['grad_retain_l2']:.3e} / beta R "
                          f"{rec['grad_regulariser_l2']:.3e} / gamma O {rec['grad_closing_l2']:.3e} / J "
                          f"{rec['grad_l2']:.3e} (max {rec['grad_max']:.3e}), {rec['gates_pushed_shut']} gates pushed "
                          f"shut; update ||ds|| {rec['update_l2']:.3e} (max {rec['update_max']:.3e}, "
                          f"{rec['gates_moving']} gates), cos to the previous {fmt(rec['update_cosine'], '.3f')}, "
                          f"reversals {fmt(rec['reversal_share'], '.3f')}")
                run.log({f"{prefix}/step/{k}": v for k, v in rec.items() if v is not None})

            def log_epoch(rec):
                window = ("" if "window_drift" not in rec else
                          f"; window: J decrease {rec['window_objective_decrease']:.2e}, drift "
                          f"{rec['window_drift']:.3f}" + (" -> converged" if rec["window_converged"] else ""))
                moves = ("" if rec["epoch"] == 0 else
                         f"; epoch ||net ds|| {rec['epoch_net_l2']:.3e} (max {rec['epoch_net_max']:.3e}), path "
                         f"{rec['epoch_path_l2']:.3e}, drift {rec['epoch_drift']:.3f}, reversals "
                         f"{fmt(rec['epoch_reversal_share'], '.3f')}")
                probe_text = ("probe not evaluated" if "probe_objective" not in rec else
                              f"probe J {rec['probe_objective']:.4f} (L_R {rec['probe_retain_nll']:.4f}, L_F "
                              f"{rec['probe_forget_unlikelihood']:.4f}, forget NLL {rec['probe_forget_nll']:.4f}, O "
                              f"{rec['probe_openness']:.1f}, R {rec['probe_regulariser']:.3e})")
                print(f"[{label}] epoch {rec['epoch']}/{max_epochs} (step {rec['steps']}, {time.time() - clock:.0f} s): "
                      f"{probe_text}; gates mean "
                      f"{rec['gate_mean']:.6f}, min {rec['gate_min']:.4f}, below 0.5 {rec['gates_below_0p5']}, at 0 "
                      f"{rec['gates_at_0']}, at 1 {rec['gates_at_1']}" + moves + window)
                run.log({f"{prefix}/epoch/{k}": v for k, v in rec.items() if v is not None})

            return log_step, log_epoch

        common = dict(micro_batch=args.direction_gates_micro_batch, grad_accum=args.direction_gates_grad_accum,
                      alpha=args.direction_gates_alpha, beta=args.direction_gates_beta,
                      gamma=args.direction_gates_gamma, probe_retain=args.direction_gates_probe_retain, seed=args.seed,
                      autocast_dtype=autocast)
        if calibrating:
            # Learning-rate calibration (user 2026-11-03): the gradient scale at s = 1, then a short training per
            # (optimizer, lr) from s = 1 on the same probe and windows (same seed); no phase two, no evaluation.
            with Stage("direction gates: gradient census", timings):
                census = RG.gradient_census(model, gates, forget_items, retain_items, gate_collator, feats,
                                            epochs=args.direction_gates_census_epochs,
                                            micro_batch=args.direction_gates_micro_batch,
                                            grad_accum=args.direction_gates_grad_accum, seed=args.seed,
                                            autocast_dtype=autocast)
            guide = RG.census_summary(census, args.direction_gates_alpha, args.direction_gates_gamma)
            ag, mg = guide["abs_gradient"], guide["abs_mean_gradient"]
            print(f"[census] {guide['windows']} windows x {guide['gates']} gates at s = 1; |gradient| per window "
                  f"median / q90 / q99: alpha L_F {ag['alpha_forget']['q50']:.3e} / {ag['alpha_forget']['q90']:.3e} / "
                  f"{ag['alpha_forget']['q99']:.3e}, L_R {ag['retain']['q50']:.3e} / {ag['retain']['q90']:.3e} / "
                  f"{ag['retain']['q99']:.3e}, J {ag['objective']['q50']:.3e} / {ag['objective']['q90']:.3e} / "
                  f"{ag['objective']['q99']:.3e}; |mean over windows| J {mg['objective']['q50']:.3e} / "
                  f"{mg['objective']['q90']:.3e} / {mg['objective']['q99']:.3e}")
            print(f"[census] SGD lr for a 99th-percentile step of 1e-3 / 1e-2 / 1e-1: "
                  + " / ".join(fmt(v, ".3e") for v in guide["sgd_lr_for_step"].values())
                  + f"; Adam step per unit lr (|mean| / rms over windows) median {guide['adam_step_per_lr']['q50']:.3f}, "
                  f"q90 {guide['adam_step_per_lr']['q90']:.3f}: a median gate needs "
                  f"{fmt(guide['adam_steps_to_close_times_lr'], '.3e')} / lr steps to close")
            print(f"[census] gates pushed shut at this alpha / gamma: {guide['pushed_shut_share']:.3f}; at gamma 0 by "
                  f"alpha: {json.dumps(guide['pushed_shut_share_by_alpha'])}; closing price (gamma pushing that share "
                  f"shut against L_R alone): {json.dumps({k: float(f'{v:.3e}') for k, v in guide['closing_price'].items()})}, "
                  f"share free to close at gamma 0: {guide['closing_free_share']:.3f}")
            trials, trial_records = [], []
            for optim, lr in parse_gate_trials(args.direction_gates_calibration_trials):
                gates.reset_()
                tag = f"{optim}_{lr:g}"
                log_step, log_epoch = loggers(f"trial {optim} lr {lr:g}", f"direction_gates/calibration/{tag}",
                                              args.direction_gates_trial_epochs)
                started = time.time()
                with Stage(f"direction gates: trial {optim} lr {lr:g}", timings):
                    out = RG.train_gates(model, gates, forget_items, retain_items, gate_collator, feats,
                                         max_epochs=args.direction_gates_trial_epochs,
                                         convergence_epochs=args.direction_gates_trial_epochs, tol=0.0, drift_max=1.0,
                                         optim=optim, lr=lr, log_step=log_step, log_epoch=log_epoch, **common)
                ep, st = out["epochs"], out["steps"]
                objs = [e["probe_objective"] for e in ep]
                shares = [e["epoch_reversal_share"] for e in ep[1:] if e["epoch_reversal_share"] is not None]
                row = {"optim": optim, "lr": lr, "seconds": time.time() - started, "epochs": out["epochs_run"],
                       "steps": len(st), "probe_objective_start": objs[0], "probe_objective_end": objs[-1],
                       "probe_objective_min": min(objs), "probe_increases": sum(b > a for a, b in zip(objs, objs[1:])),
                       "relative_decrease": ep[-1]["window_objective_decrease"], "drift": ep[-1]["window_drift"],
                       "reversal_share_mean": sum(shares) / len(shares) if shares else None,
                       "update_max": max(r["update_max"] for r in st),
                       **{f"{k}_end": ep[-1][k] for k in ("probe_retain_nll", "probe_forget_unlikelihood",
                                                          "probe_forget_nll", "probe_openness", "gate_mean",
                                                          "gate_min", "gates_below_0p5", "gates_at_0")}}
                trials.append(row)
                trial_records.append({"optim": optim, "lr": lr, "steps": st, "epochs": ep})
                print(f"[calibration] {optim} lr {lr:g}: probe J {row['probe_objective_start']:.4f} -> "
                      f"{row['probe_objective_end']:.4f} (min {row['probe_objective_min']:.4f}, "
                      f"{row['probe_increases']} epoch increases, relative decrease {row['relative_decrease']:.3e}); "
                      f"L_R {row['probe_retain_nll_end']:.4f}, forget NLL {row['probe_forget_nll_end']:.4f}; drift "
                      f"{row['drift']:.3f}, reversals {fmt(row['reversal_share_mean'], '.3f')}, max step "
                      f"{row['update_max']:.3e}; gates mean {row['gate_mean_end']:.6f}, below 0.5 "
                      f"{row['gates_below_0p5_end']}, at 0 {row['gates_at_0_end']} ({row['seconds']:.0f} s)")
                run.log({f"direction_gates/calibration/{tag}/{k}": v for k, v in row.items()
                         if isinstance(v, (int, float))})
            moved = [n for n, v in RG.weight_checksums(model).items() if checksums_before[n] != v]
            if moved:
                raise RuntimeError(f"calibration changed model tensors: {moved[:3]}")
            print(f"[control] calibration: {len(trials)} trials, every model parameter and buffer unchanged (checksums)")
            torch.save({"names": gates.names, "sizes": [g.numel() for g in gates.gates], "census": census,
                        "scores": [scores[n].cpu() for n in gates.names],
                        "own_ratio": {n: r["own_ratio"] for n, r in specificity.items()}, "config": gate_config},
                       os.path.join(args.out_dir, "direction_gates_calibration.pt"))
            with open(os.path.join(args.out_dir, "direction_gates_calibration_history.json"), "w") as f:
                json.dump({"trials": trial_records}, f)
            results["direction_gates_calibration"] = {
                "config": gate_config, "gates": n_gates, "steps_per_epoch": steps_per_epoch, "census": guide,
                "trials": trials, "column_specificity": spec_summary,
                "history_file": "direction_gates_calibration_history.json"}
            save_results()
            print("[calibration] done: phase two and evaluation skipped")
            run.finish()
            return
        log_step, log_epoch = loggers("gates", "direction_gates", args.direction_gates_max_epochs)
        with Stage("direction gates: training", timings):
            trained = RG.train_gates(
                model, gates, forget_items, retain_items, gate_collator, feats,
                max_epochs=args.direction_gates_max_epochs, convergence_epochs=args.direction_gates_convergence_epochs,
                tol=args.direction_gates_tol, drift_max=args.direction_gates_drift_max,
                optim=args.direction_gates_optim, lr=args.direction_gates_lr, log_step=log_step, log_epoch=log_epoch,
                lr_schedule=args.direction_gates_lr_schedule, probe_every=args.direction_gates_probe_every, **common)
        history, epoch_records = trained["steps"], trained["epochs"]
        # Positive controls: the gate phase moved gates and nothing else.
        moved = [n for n, v in RG.weight_checksums(model).items() if checksums_before[n] != v]
        if moved or len(history) != trained["epochs_run"] * steps_per_epoch:
            raise RuntimeError(f"gate phase: {len(history)} steps for {trained['epochs_run']} epochs of "
                               f"{steps_per_epoch}, model tensors changed: {moved[:3]}")
        summary = gates.summary()
        moved_gates = sum(int((g.detach() != args.direction_gates_init).sum()) for g in gates.gates)
        if args.direction_gates_max_epochs > 0 and moved_gates == 0:
            raise RuntimeError("no gate moved from its start: the gate phase changed nothing")
        print(f"[control] gate phase: {len(history)} steps, every model parameter and buffer unchanged (checksums), "
              f"{moved_gates} of {n_gates} gates moved from their start {args.direction_gates_init:g}")
        if trained["stop"] == "no_training":
            print(f"[gates] stop: no gate training (--direction_gates_max_epochs 0), every gate at "
                  f"{args.direction_gates_init:g}")
        else:
            print(f"[gates] stop: {trained['stop']} after {trained['epochs_run']} of {args.direction_gates_max_epochs} "
                  f"epochs" + ("" if trained["converged"] else
                               " (fixed schedule, no convergence rule)" if trained["stop"] == "fixed_schedule" else
                               " (NOT converged: the cap was reached first)"))
        soft_gates = [g.detach().cpu().clone() for g in gates.gates]
        binarized, chosen, cut_info = None, None, None
        if args.direction_gates_binarize_share is not None:
            with Stage("direction gates: binarize", timings):
                binarized = RG.binarize_by_share(gates, covs_forget, covs, args.direction_gates_binarize_share,
                                                 args.direction_gates_binary_s)
            print(f"[gates] binarized (the gates choose, the cut removes whole): the {binarized['gates_removed']} most "
                  f"closed of {binarized['gates_below_1']} gates below 1 (threshold s <= {binarized['threshold']:.4f}; "
                  f"per type {json.dumps(binarized['per_type'])}) set to {args.direction_gates_binary_s:g}, every other "
                  f"to 1: removed output energy forget share {binarized['forget_share']:.4f} (budget "
                  f"{args.direction_gates_binarize_share:g}), retain share {binarized['retain_share']:.4f}")
            chosen = {n: (g.detach() == args.direction_gates_binary_s) for n, g in zip(gates.names, gates.gates)}
            n_chosen, chosen_at = binarized["gates_removed"], f"chosen at {args.direction_gates_binary_s:g}"
        elif args.direction_gates_cut is not None:  # no gate training: every candidate is cut (S2's choice with lambda*)
            chosen = {n: torch.ones_like(g, dtype=torch.bool) for n, g in zip(gates.names, gates.gates)}
            n_chosen = n_gates
            chosen_at = "every candidate direction" + ("" if args.direction_gates_lambda_star is None else
                                                       f", lambda >= {args.direction_gates_lambda_star:g}")
            print(f"[gates] cut without binarization: all {n_chosen} candidate directions chosen ({chosen_at})")
        cut_s = args.direction_gates_cut_s
        if args.direction_gates_cut == "shrink" and cut_s is not None:
            with torch.no_grad():
                for g, n in zip(gates.gates, gates.names):
                    g[chosen[n]] = cut_s
            print(f"[gates] cut shrink: the {n_chosen} chosen directions keep {cut_s:g} ({chosen_at})")
            cut_info = {"mode": "shrink", "s": cut_s, "chosen": n_chosen}
        elif args.direction_gates_cut == "rotate":
            keep = 0.0 if cut_s is None else cut_s
            rot_gen = torch.Generator().manual_seed(args.seed)
            rotations = {}
            with Stage("direction gates: rotate", timings):
                for i, n in enumerate(gates.names):
                    idx = chosen[n].nonzero().flatten()
                    if idx.numel() == 0:
                        continue
                    rep = RG.rotate_cut(gates.parts(i)[1][:, idx], targets[n].weight, covs_forget[n],
                                        args.direction_gates_rotate_evr, rot_gen, keep=keep)
                    rotations[n] = {"index": idx.cpu(), **rep}
            with torch.no_grad():
                for g, n in zip(gates.gates, gates.names):
                    g[chosen[n]] = 1.0  # the chosen parts are rotated, not shrunk
            print(f"[gates] cut rotate ({chosen_at}): {sum(r['rank'] for r in rotations.values())} chosen parts over "
                  f"{len(rotations)} targets carried onto outputs orthogonal to their own span and to the forget "
                  f"output subspace (share {args.direction_gates_rotate_evr:g}: "
                  f"{min(r['forget_dims'] for r in rotations.values())}-{max(r['forget_dims'] for r in rotations.values())} "
                  f"dims), "
                  + ("a quarter turn" if keep == 0.0 else
                     f"a partial turn by arccos {keep:g} (each part keeps {keep:g} of itself along its own direction)")
                  + f"; the moved component's overlap with their span max "
                  f"{max(r['own_overlap'] for r in rotations.values()):.2e}, with the forget subspace max "
                  f"{max(r['forget_overlap'] for r in rotations.values()):.2e}; Gram error max "
                  f"{max(r['gram_error'] for r in rotations.values()):.2e}")
            cut_info = {"mode": "rotate", "evr": args.direction_gates_rotate_evr, "keep": keep, "chosen": n_chosen,
                        "per_target": {n: {k: v for k, v in r.items() if k not in ("rotated", "index")}
                                       for n, r in rotations.items()}}
        if binarized is not None and cut_info is not None:
            binarized["cut"] = cut_info
        if chosen is not None:
            summary = gates.summary()
        pchange = RG.prefix_change(gates, prefix_inputs)
        for typ in sorted({n.rsplit(".", 1)[-1] for n in pchange}):
            rows = [r for n, r in pchange.items() if n.endswith(typ)]
            print(f"[gates] {typ} change at the prefix positions (||dW z||^2 / ||W0 z||^2): BOS max "
                  f"{max(r['bos_share'] for r in rows):.3e}, prefix mean {sum(r['prefix_share_mean'] for r in rows) / len(rows):.3e} "
                  f"(largest {max(r['prefix_share_max'] for r in rows):.3e}); read leakage max "
                  f"{max(r['read_leak_max'] for r in rows):.3e}")
        with Stage("direction gates: removed energies", timings):
            energy = RG.removed_energy(gates, covs_forget, covs)
        if args.direction_gates_cut == "rotate":
            for i, n in enumerate(gates.names):
                if n not in rotations:
                    continue
                idx = rotations[n]["index"].to(gates.parts(i)[0].device)
                v = gates.parts(i)[0][:, idx]
                da = rotations[n]["rotated"] - gates.parts(i)[1][:, idx].double()
                for side, cov in (("forget", covs_forget[n]), ("retain", covs[n])):
                    e = RG.edit_energy(da, v, targets[n].weight, cov)
                    energy[n].update({f"{side}_energy_removed": energy[n][f"{side}_energy_removed"] + e["energy"]})
                    energy[n][f"{side}_share"] = energy[n][f"{side}_energy_removed"] / energy[n][f"{side}_energy"]
                z = prefix_inputs[n].to(v.device).double()
                dz = (z @ v.double()) @ da.transpose(0, 1)
                base = (z @ targets[n].weight.detach().double().transpose(0, 1)).square().sum(1).clamp_min(1e-300)
                share = dz.square().sum(1) / base
                pchange[n].update({"rotate_bos_share": float(share[0]), "rotate_prefix_share_max": float(share.max())})
            fe = sum(e["forget_energy_removed"] for e in energy.values()) / sum(e["forget_energy"] for e in energy.values())
            re_ = sum(e["retain_energy_removed"] for e in energy.values()) / sum(e["retain_energy"] for e in energy.values())
            print(f"[gates] rotation edit: output energy changed, forget share {fe:.4f}, retain share {re_:.4f} (a turn "
                  f"keeping c = {keep:g} moves each part by sqrt(2 (1 - c)) = {math.sqrt(2.0 * (1.0 - keep)):.4f} of "
                  f"its norm, the shrink to c by 1 - c = {1.0 - keep:.4f}); prefix change max "
                  f"{max(r.get('rotate_prefix_share_max', 0.0) for r in pchange.values()):.3e}")
        torch.save({"names": gates.names, "gates": [g.detach().cpu() for g in gates.gates], "soft_gates": soft_gates,
                    "scores": [scores[n].cpu() for n in gates.names], "config": gate_config,
                    "own_ratio": {n: r["own_ratio"] for n, r in specificity.items()},
                    "contraction_snapshots": trained["contraction_snapshots"], "gate_path": trained["gate_path"],
                    "gate_reversals": trained["gate_reversals"],
                    "probe_retain_indices": trained["probe_retain_indices"]},
                   os.path.join(args.out_dir, "direction_gates.pt"))
        with open(os.path.join(args.out_dir, "direction_gates_history.json"), "w") as f:
            json.dump({"steps": history, "epochs": epoch_records}, f)
        with Stage("direction gates: materialize", timings):
            mat = RG.materialize_checked(model, gates, probe)
        c = mat["check"]
        if args.direction_gates_cut == "rotate":
            worst = 0.0
            with torch.no_grad():
                for i, n in enumerate(gates.names):
                    if n not in rotations:
                        continue
                    idx_cpu = rotations[n]["index"]
                    idx = idx_cpu.to(gates.parts(i)[0].device)
                    v = gates.parts(i)[0][:, idx].double()
                    a_cut = gates.parts(i)[1][:, idx].double()
                    w = targets[n].weight
                    w64 = w.detach().double() + (rotations[n]["rotated"] - a_cut) @ v.transpose(0, 1)
                    w.copy_(w64.to(w.dtype))
                    u = write_dirs[n][:, idx_cpu].to(w.device).double()
                    got = w.detach().double() @ u
                    worst = max(worst, float((got - rotations[n]["rotated"]).abs().max()
                                             / rotations[n]["rotated"].abs().max().clamp_min(1e-300)))
            if worst > 1e-3:
                raise RuntimeError(f"the rotated parts are not in the weights: relative error {worst:.2e}")
            print(f"[control] rotation written: W u_j = Q a_j for every rotated part within {worst:.2e} (relative)")
        print(f"[control] materialized weights reproduce the gated logits: max |diff| {c['max_abs_logit_diff']:.2e} of "
              f"max |logit| {c['max_abs_logit']:.2f} (relative {c['relative']:.2e} <= {RG.MATERIALIZE_RTOL:g}), argmax "
              f"agreement {c['argmax_agreement']:.4f}")
        ws_report = {}
        for n in gates.names:
            finite = scores[n][torch.isfinite(scores[n])]
            ws_report[n] = {**summary[n], **energy[n], **mat["per_target"][n],
                            "score_max": float(finite.max()) if finite.numel() else None,
                            "score_min": float(finite.min()) if finite.numel() else None}
        gate_summary = RG.summarize_gates(ws_report)
        for typ, agg in gate_summary.items():
            print(f"[gates] {typ}: {agg['targets']} targets, gate mean {agg['gate_mean']:.4f}, below 0.5 "
                  f"{agg['below_0p5']} / below 0.1 {agg['below_0p1']} / at 0 {agg['at_0']} of {agg['k']}, sum(1 - s) "
                  f"{agg['contraction_sum']:.1f}; removed output energy: forget share {agg['forget_share']:.4f}, retain "
                  f"share {agg['retain_share']:.4f}; ||dW|| / ||W|| mean {agg['weight_rel_change_mean']:.4f}")
        gate_path, rev = trained["gate_path"], trained["gate_reversals"]
        print(f"[gates] per gate over the phase: path sum_t |ds| median {float(gate_path.median()):.3e}, max "
              f"{float(gate_path.max()):.3e}; reversals median {float(rev.double().median()):.0f}, max {int(rev.max())} of "
              f"{len(history) - 1} step pairs")
        results["direction_gates"] = {
            "config": gate_config, "gates": n_gates, "steps": len(history), "steps_per_epoch": steps_per_epoch,
            "epochs_run": trained["epochs_run"], "stop": trained["stop"], "converged": trained["converged"],
            "epochs": epoch_records, "per_type": gate_summary, "materialization": c,
            "regulariser_norm2": gates.norm2, "column_specificity": spec_summary, "binarized": binarized,
            "cut": cut_info, "history_file": "direction_gates_history.json"}
        regrowth_saved = {n: {"u": write_dirs[n], "a": gates.parts(i)[1].detach().float().cpu(),
                              "s": gates.gates[i].detach().double().cpu()} for i, n in enumerate(gates.names)}
        if args.direction_gates_cut == "rotate":
            for n, r in rotations.items():
                regrowth_saved[n].update({"rotated": r["rotated"].float().cpu(), "rotated_index": r["index"]})
            del rotations
        results["direction_gates"]["prefix_change"] = pchange
        del gates, scores, probe, energy, mat, trained, specificity, write_dirs
        torch.cuda.empty_cache()
    else:
        with Stage("warm start", timings):
            restrictions, ws_report = warm_start(
                targets, grad_retain, args.s_min, source, args.restriction_evr if source is not None else None,
                unrestricted=unrestricted_targets, alignment_scope=args.restriction_alignment_scope or "full_space",
                layer_scaling=args.shrink_layer_scaling, input_filters=input_filters, whitening=whitening,
                whitening_rho=args.whitening_rho, rank_multiplier=args.restriction_rank_multiplier,
                representation_output=args.restriction_representation if representation else None,
                gradient_contrast=(args.restriction_gradient_contrast
                                   if restricted and args.restriction_gradient_contrast not in (None, "none") else None),
                gradient_moments=moments, contrast_rank_rule=args.gradient_contrast_rank_rule or "side_energy",
                forget_covariances=covs_forget, retain_covariances=covs if (representation or contrast) else None,
                direction_weighting=args.direction_weighting,
                representation_mix=1.0 if args.representation_mix is None else args.representation_mix,
                forget_output_target=target, component_selection=args.component_selection,
                component_diagnostics=bool(args.component_diagnostics), component_seed=args.seed,
                shrink_score=args.shrink_score, score_diagnostics=bool(args.score_diagnostics),
                core_diagnostics=bool(args.core_diagnostics), weight_contraction=args.weight_contraction,
                contraction_strength=1.0 if args.contraction_strength is None else args.contraction_strength,
                contraction_residual=args.contraction_residual or 0.0,
                contraction_diagnostics=bool(args.contraction_diagnostics), output_diagnostics=bool(args.output_diagnostics),
                output_contrast_rho=args.output_contrast_rho, family_diagnostics=bool(args.family_diagnostics),
                family_contraction=args.family_contraction, family_rho=args.family_rho,
                global_family=args.global_family, global_family_rho=args.global_family_rho,
                global_family_floor=args.global_family_floor, global_family_order=args.global_family_order or "score",
                global_family_alpha=args.global_family_alpha,
                global_family_budget_unit=args.global_family_budget_unit or "energy",
                global_family_s_gamma=args.global_family_s_gamma, global_family_s_rule=args.global_family_s_rule,
                global_family_s_threshold_gamma=args.global_family_s_threshold_gamma)
    if restricted and args.restriction_gradient_contrast not in (None, "none"):
        kept = {p: [r["evr"] for n, r in ws_report.items() if n.endswith(p) and r.get("active")] for p in PROJECTIONS}
        kept = {p: sum(v) / len(v) for p, v in kept.items() if v}
        ranks = {p: {side: sorted(r.get(key, r["rank"]) for n, r in ws_report.items() if n.endswith(p) and r.get("active"))
                     for side, key in (("out", "rank"), ("in", "rank_in"))} for p in kept}
        ranks = {p: {side: v[len(v) // 2] for side, v in d.items()} for p, d in ranks.items()}
        print(f"[gcontrast] sides {args.restriction_gradient_contrast}, rho {args.whitening_rho:g}; forget-gradient "
              f"energy kept, mean per type: " + ", ".join(f"{p} {v:.3f}" for p, v in kept.items())
              + "; median ranks out / in: " + ", ".join(f"{p} {d['out']} / {d['in']}" for p, d in ranks.items()))
        results["gradient_contrast"] = {"sides": args.restriction_gradient_contrast, "rho": args.whitening_rho,
                                        "kept_forget_gradient_energy": kept, "median_rank": ranks,
                                        "rank_rule": ("rank_multiplier" if args.restriction_rank_multiplier
                                                      else args.gradient_contrast_rank_rule or "side_energy")}
    if moments is not None:
        # Per target: the mean-gradient energy kept (evr), each side's share of the per-example energy at the applied rank,
        # and the rank each side would need for --restriction_evr of its own per-example energy.
        rows = {}
        for n, (m_out, m_in) in moments.items():
            r = ws_report[n]
            if not r.get("active"):
                continue
            row = {"kept_forget_gradient_energy": r["evr"], "rank_out": r["rank"], "rank_in": r.get("rank_in", r["rank"])}
            for side, m in (("out", m_out), ("in", m_in)):
                cum = torch.cumsum(torch.linalg.eigvalsh(m).flip(0).clamp_min(0.0), 0)
                cum = cum / cum[-1]
                row[f"moment_share_{side}"] = float(cum[row[f"rank_{side}"] - 1])
                row[f"own_evr_rank_{side}"] = min(int((cum < args.restriction_evr).sum()) + 1, cum.numel())
            rows.setdefault(n.rsplit(".", 1)[-1], []).append(row)
        per_type = {p: {k: (sorted(x[k] for x in v)[len(v) // 2] if "rank" in k else sum(x[k] for x in v) / len(v))
                        for k in v[0]} for p, v in rows.items()}
        print("[gmoment] per type (means; ranks: medians): " + "; ".join(
            f"{p} kept G_f {d['kept_forget_gradient_energy']:.3f}, rank out {d['rank_out']} in {d['rank_in']}, "
            f"per-example share out {d['moment_share_out']:.3f} in {d['moment_share_in']:.3f}, own-EVR rank out "
            f"{d['own_evr_rank_out']} in {d['own_evr_rank_in']}" for p, d in per_type.items()))
        results["gradient_moment"] = {"kind": "per_example", "examples": n_moment, "per_type": per_type}
    if args.global_family:
        g = next(r["global_family"]["global"] for r in ws_report.values() if "global_family" in r)
        by_type = {}
        for n, r in ws_report.items():
            if "global_family" in r:
                by_type.setdefault(n.rsplit(".", 1)[-1], []).append(r["global_family"])
        results["global_family"] = {"global": g, "per_type": {
            typ: {"directions": sum(x["directions"] for x in rs), "of": sum(x["of"] for x in rs),
                  "forget_energy_removed": sum(x["forget_energy_removed"] for x in rs),
                  "retain_energy_removed": sum(x["retain_energy_removed"] for x in rs),
                  "recast_forget_energy": sum(x["recast_forget_energy"] for x in rs),
                  "recast_retain_energy": sum(x["recast_retain_energy"] for x in rs)}
            for typ, rs in by_type.items()}}
        print(f"[global family] {g['family']} rho {g['rho']:g} order {g['order']}"
              f"{'' if g['alpha'] is None else ' alpha ' + format(g['alpha'], 'g')} floor {g['floor']:g} "
              f"budget unit {g['unit']} (sum of target shares {g['forget_share_sum']:.4f})"
              f"{'' if g['s_gamma'] is None else ', s_j = clip(lambda^-' + format(g['s_gamma'], 'g') + ')'}"
              f"{'' if g['s_rule'] is None else ', s_j = clip(e_r / (e_f + e_r))'}"
              f"{'' if g.get('s_threshold_gamma') is None else ', s_j = clip((lambda*/lambda)^' + format(g['s_threshold_gamma'], 'g') + '), lambda* ' + format(g['threshold'], '.4g')}: "
              f"{g['directions']} of {g['of']} directions, reached {g['reached']}; removed forget energy "
              f"{g['forget_energy_removed']:.4e} (budget {g['budget']:.4e}, share {g['forget_share']:.4f}), retain "
              f"{g['retain_energy_removed']:.4e} (ReCAST {g['recast_retain_energy']:.4e}, share {g['retain_share']:.4f}); "
              f"||dW|| {g['displacement']:.3f} (ReCAST {g['recast_update_norm']:.3f})")
        for typ, a in sorted(results["global_family"]["per_type"].items()):
            print(f"[global family] {typ}: {a['directions']} of {a['of']} directions; forget energy "
                  f"{a['forget_energy_removed']:.4e} (ReCAST {a['recast_forget_energy']:.4e}), retain "
                  f"{a['retain_energy_removed']:.4e} (ReCAST {a['recast_retain_energy']:.4e})")
    if args.family_contraction:
        by_type = {}
        for n, r in ws_report.items():
            if "family_contraction" in r:
                by_type.setdefault(n.rsplit(".", 1)[-1], []).append(r["family_contraction"])
        results["family_contraction"] = {
            typ: {k: sum(x[k] for x in rs) / len(rs) for k in ("directions", "forget_share", "retain_share",
                                                              "recast_forget_share", "recast_retain_share",
                                                              "displacement", "recast_update_norm")}
            for typ, rs in by_type.items()}
        for typ, a in sorted(results["family_contraction"].items()):
            print(f"[family contraction] {args.family_contraction} rho {args.family_rho} {typ}: {a['directions']:.0f} "
                  f"directions, removed output forget {a['forget_share']:.4f} (ReCAST {a['recast_forget_share']:.4f}), "
                  f"retain {a['retain_share']:.4f} (ReCAST {a['recast_retain_share']:.4f}); ||dW|| per target "
                  f"{a['displacement']:.3f} (ReCAST {a['recast_update_norm']:.3f})")
    if args.family_diagnostics:
        for typ in ("down_proj", "o_proj"):
            fr = [r["families"] for n, r in ws_report.items() if n.endswith(typ) and "families" in r]
            if not fr:
                continue
            m = lambda xs: sum(xs) / len(xs)  # noqa: E731
            parts = [f"ReCAST forget {m([x['recast_forget_share'] for x in fr]):.4f} retain "
                     f"{m([x['recast_retain_share'] for x in fr]):.4f}"]
            for fam in ("output_contrast", "gradient_contrast", "input_contrast"):
                for rho in fr[0][fam]:
                    parts.append(f"{fam} rho {rho}: retain {m([x[fam][rho]['retain_share'] for x in fr]):.4f}, "
                                 f"{m([x[fam][rho]['directions'] for x in fr]):.0f} directions")
            print(f"[families] {typ} ({len(fr)} targets, means at ReCAST's forget share) " + "; ".join(parts))
    if args.output_diagnostics:
        by_type = {}
        for n, r in ws_report.items():
            if "output_energy" in r:
                by_type.setdefault(n.rsplit(".", 1)[-1], []).append(r)

        def od_mean(reports, key):
            xs = [x[key] for x in reports if x.get(key) is not None]
            return f"{sum(xs) / len(xs):.4f}" if xs else "–"

        for typ, rs in sorted(by_type.items()):
            parts = [f"applied: c forget {od_mean([r['output_energy'] for r in rs], 'forget_ratio')} retain "
                     f"{od_mean([r['output_energy'] for r in rs], 'retain_ratio')}, removed forget "
                     f"{od_mean([r['output_energy'] for r in rs], 'forget_removed_share')} retain "
                     f"{od_mean([r['output_energy'] for r in rs], 'retain_removed_share')}"]
            for group in ("output_energy_svd", "output_contrast"):
                for k in sorted({k for r in rs for k in r.get(group, {})}):
                    sub = [r[group][k] for r in rs if k in r.get(group, {})]
                    label = k if group == "output_energy_svd" else f"output contrast rho {k}"
                    parts.append(f"{label}: removed forget {od_mean(sub, 'forget_removed_share')} retain "
                                 f"{od_mean(sub, 'retain_removed_share')}")
            print(f"[output] {typ} ({len(rs)} targets, means) " + "; ".join(parts))
    if args.contraction_diagnostics:
        prof = [r["contraction_components"] for r in ws_report.values() if "contraction_components" in r]
        contrasts = [c for x in prof for c in x["input_contrast"]]
        print(f"[contraction components] {len(prof)} targets, {len(contrasts)} components; input contrast mean "
              f"{sum(contrasts) / max(len(contrasts), 1):.4f}, min {min(contrasts, default=float('nan')):.4f}, max "
              f"{max(contrasts, default=float('nan')):.4f} (per-component arrays in warm_start.per_target)")
    if args.weight_contraction:
        ct_rows = [r for r in ws_report.values() if "contraction" in r]
        ct = [r["contraction"] for r in ct_rows]

        def ct_mean(key, sub=None):
            xs = [x[key] if sub is None else x[key][sub] for x in ct]
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else None

        results["contraction"] = {
            "mode": args.weight_contraction, "strength": args.contraction_strength,
            "residual": args.contraction_residual or 0.0, "targets": len(ct),
            "targets_short_of_budget": [n for n, r in ws_report.items() if "contraction" in r
                                        and not r["contraction"]["budget_reached"]],
            "components_mean": ct_mean("components"), "contracted_mean": ct_mean("contracted"),
            "full_mean": ct_mean("full"), "energy_fraction_mean": ct_mean("energy_fraction"),
            "kappa_rank_mean": ct_mean("kappa_rank_mean"), "kappa_rank_weighted": ct_mean("kappa_rank_weighted"),
            "input_contrast_moved": ct_mean("input_contrast", "moved"),
            "input_contrast_all": ct_mean("input_contrast", "all"),
            "update_norm": sum(r["applied_norm"] ** 2 for r in ct_rows) ** 0.5,
            "recast_update_norm": sum(x["recast_update_norm"] ** 2 for x in ct) ** 0.5}
        ct_fmt = {k: ("–" if v is None else f"{v:.4f}") for k, v in results["contraction"].items()
                  if isinstance(v, float) or v is None}
        print(f"[contraction] {args.weight_contraction} strength {args.contraction_strength} on {len(ct)} targets: "
              f"{ct_fmt['contracted_mean']} of {ct_fmt['components_mean']} components ({ct_fmt['full_mean']} fully), "
              f"energy {ct_fmt['energy_fraction_mean']}, kappa rank {ct_fmt['kappa_rank_mean']} (weighted "
              f"{ct_fmt['kappa_rank_weighted']}), input contrast moved {ct_fmt['input_contrast_moved']} vs all "
              f"{ct_fmt['input_contrast_all']}; ||dW|| {ct_fmt['update_norm']} (ReCAST {ct_fmt['recast_update_norm']})")
    if args.output_contrast_rho is not None:
        oc = [r["output_contrast_applied"] for r in ws_report.values() if "output_contrast_applied" in r]
        results["output_contrast"] = {
            "rho": args.output_contrast_rho, "targets": len(oc),
            **{k: sum(x[k] for x in oc) / len(oc) for k in ("forget_removed_share", "retain_removed_share",
                                                            "forget_ratio", "retain_ratio", "directions")},
            "update_norm": sum(x["displacement"] ** 2 for x in oc) ** 0.5,
            "recast_update_norm": sum(x["recast_update_norm"] ** 2 for x in oc) ** 0.5}
        a = results["output_contrast"]
        print(f"[output contrast] rho {a['rho']} on {a['targets']} targets: {a['directions']:.1f} directions, removed "
              f"output forget {a['forget_removed_share']:.4f} retain {a['retain_removed_share']:.4f}; ||dW|| "
              f"{a['update_norm']:.4f} (ReCAST {a['recast_update_norm']:.4f})")
    if args.write_compensation:
        # c_k = the retain output ratio of write k; h <- c_k h + write_k at every write is the same function as write k
        # and every later write divided by c_1 ... c_k (every read is an RMSNorm), so it goes into the weights.
        factors = {n: r["output_energy"]["retain_ratio"] for n, r in ws_report.items()
                   if n.endswith(WRITE_PROJECTIONS) and "output_energy" in r}
        divisors = residual_write_divisors(model.config.num_hidden_layers, factors)
        named = dict(model.named_parameters())
        before = {n: p.detach().clone() for n, p in named.items()}
        apply_write_divisors(model, divisors)
        # Positive control: each write is its old weight divided by its divisor, every other parameter is untouched.
        bad = [n for n, p in named.items()
               if not torch.allclose(p.detach() * divisors.get(n[:-len(".weight")], 1.0) if n.endswith(".weight") else p.detach(),
                                     before[n], rtol=1e-5, atol=1e-8)]
        if bad:
            raise RuntimeError(f"write compensation changed more than the writes: {bad[:3]}")
        del before
        for n in restrictions:
            restrictions[n] = type(restrictions[n])(baseline=targets[n].weight.detach().clone(),
                                                    basis=restrictions[n].basis)
        results["write_compensation"] = {"mode": args.write_compensation, "factors": factors, "divisors": divisors,
                                         "min_divisor": min(divisors.values()), "final_divisor": list(divisors.values())[-1]}
        print(f"[write compensation] {len(factors)} writes with a factor (mean {sum(factors.values()) / len(factors):.4f}); "
              f"running product down to {min(divisors.values()):.4f}, so the last writes grow "
              f"{1.0 / min(divisors.values()):.1f}x")
    if args.core_diagnostics:
        def core_aggregate(rows, source):
            """Means over targets of one S source's weight-core / update-core comparison."""
            c = [r["cores"][source] for r in rows]
            mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
            out = {"targets": len(c), "rank_mean": mean([x["rank"] for x in c]),
                   "kappa_effective_rank": mean([x["kappa_effective_rank"] for x in c]),
                   "sigma_effective_rank": mean([x["sigma_effective_rank"] for x in c]),
                   "spectrum_l2_distance": mean([x["spectrum_l2_distance"] or 0.0 for x in c]),
                   "beta_offdiag_energy_fraction": mean([x["beta"]["offdiag_energy_fraction"] for x in c]),
                   "beta_outside_unit_interval": mean([x["beta"]["outside_unit_interval"] for x in c]),
                   "angles": {k: {f: mean([x["angles"][k][f] for x in c if k in x["angles"]])
                                  for f in ("left_mean_cos", "left_min_cos", "right_mean_cos", "right_min_cos")}
                              for k in c[0]["angles"]}}
            for name in c[0].get("alignment", {}):
                out[f"{name}_alignment_abs_mean"] = {
                    "weight": mean([x["alignment"][name]["weight_abs_mean"] for x in c]),
                    "update": mean([x["alignment"][name]["update_abs_mean"] for x in c])}
            if "input_contrast" in c[0]:
                out["input_contrast_leading"] = {"weight": mean([x["input_contrast"]["weight_mean"] for x in c]),
                                                 "update": mean([x["input_contrast"]["update_mean"] for x in c])}
            return out

        cored = {n: r for n, r in ws_report.items() if "cores" in r}
        sources = list(next(iter(cored.values()))["cores"])
        core_report = {p_: {src: core_aggregate([r for n, r in cored.items() if n.endswith(p_)], src) for src in sources}
                       for p_ in PROJECTIONS if any(n.endswith(p_) for n in cored)}
        core_report["all"] = {src: core_aggregate(list(cored.values()), src) for src in sources}
        for p_, by_src in core_report.items():
            for src, agg in by_src.items():
                a16 = agg["angles"].get("16") or agg["angles"][max(agg["angles"], key=int)]
                print(f"[cores] {p_:9s} {src:22s} k=16 cos left {a16['left_mean_cos']:.6f} (min {a16['left_min_cos']:.6f}) "
                      f"right {a16['right_mean_cos']:.6f} (min {a16['right_min_cos']:.6f}); spectrum distance "
                      f"{agg['spectrum_l2_distance']:.3e}; off-diagonal energy {agg['beta_offdiag_energy_fraction']:.3e}; "
                      f"effective rank K {agg['kappa_effective_rank']:.1f} B {agg['sigma_effective_rank']:.1f}"
                      + (f"; input contrast K {agg['input_contrast_leading']['weight']:.3f} "
                         f"B {agg['input_contrast_leading']['update']:.3f}" if "input_contrast_leading" in agg else ""))
        results["core_diagnostics"] = core_report
    if args.score_diagnostics:
        def score_aggregate(rows):
            """Per rule: means over targets of the share above floor 0.2, the median and the 99th percentile and the
            correlations with the original; the summed first-order changes and the joint update norm per floor."""
            mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
            out = {rule: {"above_floor_0.2": mean([r["scores"][rule]["above_floor"]["0.2"] for r in rows]),
                          "q50": mean([r["scores"][rule]["quantiles"]["q50"] for r in rows]),
                          "q99": mean([r["scores"][rule]["quantiles"]["q99"] for r in rows]),
                          "pearson_vs_original": mean([r["scores"][rule].get("pearson_vs_original") or 0.0 for r in rows]),
                          "spearman_vs_original": mean([r["scores"][rule].get("spearman_vs_original") or 0.0
                                                        for r in rows])}
                   for rule in ("original", "projected_row", "projected_sensitivity", "core_row")}
            gl = [r["scores"]["global_projected"] for r in rows]
            out["global_projected"] = {"mean": mean(gl), "min": min(gl), "max": max(gl),
                                       "above_floor_0.2": mean([float(x > 0.2) for x in gl])}
            out["zero_norm_rows"] = {k: sum(r["scores"]["zero_norm_rows"][k] for r in rows)
                                     for k in rows[0]["scores"]["zero_norm_rows"]}
            first = [r["applied_first_order"] for r in rows]
            out["applied"] = {"update_norm": sum(f["update_norm"] ** 2 for f in first) ** 0.5,
                              "proposal_norm": sum(f["proposal_norm"] ** 2 for f in first) ** 0.5,
                              "predicted_retain_change": sum(f["predicted_retain_change"] for f in first),
                              "predicted_forget_change": sum(f["predicted_forget_change"] for f in first)}
            out["applied"]["kept_energy_fraction"] = (out["applied"]["update_norm"] / out["applied"]["proposal_norm"]) ** 2
            out["update_norm_by_floor"] = {rule: {f: sum(r["update_norm_by_floor"][rule][f] ** 2 for r in rows) ** 0.5
                                                  for f in rows[0]["update_norm_by_floor"][rule]}
                                           for rule in rows[0]["update_norm_by_floor"]}
            return out

        scored = {n: r for n, r in ws_report.items() if "scores" in r}
        score_report = {p_: score_aggregate([r for n, r in scored.items() if n.endswith(p_)]) for p_ in PROJECTIONS
                        if any(n.endswith(p_) for n in scored)}
        score_report["all"] = score_aggregate(list(scored.values()))
        for p_, agg in score_report.items():
            print(f"[scores] {p_:9s} above floor 0.2 / median / q99 / Spearman vs original: " + "; ".join(
                f"{rule} {agg[rule]['above_floor_0.2']:.4f} / {agg[rule]['q50']:+.4f} / {agg[rule]['q99']:+.4f} / "
                f"{agg[rule]['spearman_vs_original']:+.3f}" for rule in ("original", "projected_row",
                                                                         "projected_sensitivity", "core_row"))
                  + f"; global {agg['global_projected']['mean']:+.4f} [{agg['global_projected']['min']:+.4f}, "
                    f"{agg['global_projected']['max']:+.4f}]; applied ({args.shrink_score}) norm "
                    f"{agg['applied']['update_norm']:.3f}, kept energy {agg['applied']['kept_energy_fraction']:.4f}, "
                    f"predicted dL_r {agg['applied']['predicted_retain_change']:+.4e} dL_f "
                    f"{agg['applied']['predicted_forget_change']:+.4e}")
        results["score_diagnostics"] = score_report
    if args.component_selection or args.component_diagnostics:
        comp = {n: r["components"] for n, r in ws_report.items() if "components" in r}
        arrays = {n: {k: c.pop(k) for k in ("sigma", "contribution") if k in c} for n, c in comp.items()}
        # Positive controls: the fp64 components rebuild the fp32 original update to within fp32 rounding of the
        # proposal (||P(X) - sum D_j|| <= 1e-5 ||X||; the fp32 core of a near-zero update can lose most relative
        # digits), their contributions sum to <G_f, P(X)> up to ||G_f|| times that difference (Cauchy-Schwarz), and
        # the selection kept what it should.
        def control_failed(n, c):
            if c["reconstruction_abs_error"] > 1e-5 * ws_report[n]["proposal_norm"]:
                return True
            gap = abs(c["contribution_sum"] - c["first_order_forget_direct"])
            return gap > c["grad_forget_norm"] * c["reconstruction_abs_error"] + 1e-6 * abs(c["first_order_forget_direct"]) + 1e-12

        bad = [n for n, c in comp.items() if control_failed(n, c)]
        if not comp or bad:
            raise RuntimeError(f"component decomposition control failed at {bad[:3]} ({len(comp)} targets): " + "; ".join(
                f"{n}: reconstruction {comp[n]['reconstruction_abs_error']:.3e} vs proposal "
                f"{ws_report[n]['proposal_norm']:.3e}, sum c_j {comp[n]['contribution_sum']:.6e} vs "
                f"{comp[n]['first_order_forget_direct']:.6e}" for n in bad[:3]))
        worst = max(comp, key=lambda n: comp[n]["reconstruction_rel_error"])
        if args.component_selection in ("signed", "random") and any(c["kept"] != c["positive"] for c in comp.values()):
            raise RuntimeError("the selection kept a different number of components than have c_j > 0")
        if args.component_selection in (None, "full", "uniform_matched") and any(
                c["kept"] != c["components"] and not (c["kept"] == 0 and c["positive"] == 0) for c in comp.values()):
            raise RuntimeError("a full or uniform update dropped components")  # gamma = 0 only without c_j > 0

        def summed(rows):
            keys = ("components", "positive", "negative", "kept", "energy", "contribution_sum", "contribution_positive",
                    "contribution_negative", "predicted_forget_change", "predicted_retain_change")
            out = {k: sum(r[k] for r in rows) for k in keys}
            energy = sum(r["energy"] for r in rows)
            for k in ("energy_positive_fraction", "energy_negative_fraction", "applied_energy_fraction"):
                out[k] = sum(r[k] * r["energy"] for r in rows) / energy if energy > 0 else 0.0
            out["original_norm"] = sum(r["original_norm"] ** 2 for r in rows) ** 0.5
            out["applied_norm"] = sum(r["applied_norm"] ** 2 for r in rows) ** 0.5
            out["effective_rank_original_mean"] = sum(r["effective_rank_original"] for r in rows) / len(rows)
            out["effective_rank_applied_mean"] = sum(r["effective_rank_applied"] for r in rows) / len(rows)
            out["max_reconstruction_rel_error"] = max(r["reconstruction_rel_error"] for r in rows)
            out["targets"] = len(rows)
            return out

        comp_summary = {p_: summed([c for n, c in comp.items() if n.endswith(p_)]) for p_ in PROJECTIONS
                        if any(n.endswith(p_) for n in comp)}
        comp_summary["all"] = summed(list(comp.values()))
        for p_, c in comp_summary.items():
            print(f"[components] {p_:9s} selection={args.component_selection or 'none'}: components {c['components']} "
                  f"c_j>0 {c['positive']} kept {c['kept']}; energy with c_j>0 {c['energy_positive_fraction']:.4f}, "
                  f"c_j<0 {c['energy_negative_fraction']:.4f}; applied energy {c['applied_energy_fraction']:.4f}, "
                  f"norm {c['applied_norm']:.4e} / {c['original_norm']:.4e}; effective rank "
                  f"{c['effective_rank_applied_mean']:.1f} / {c['effective_rank_original_mean']:.1f}; sum c_j "
                  f"{c['contribution_sum']:.4e} (+{c['contribution_positive']:.4e} {c['contribution_negative']:.4e}); "
                  f"predicted dL_f {c['predicted_forget_change']:.4e} dL_r {c['predicted_retain_change']:.4e}")
        print(f"[control] components: rebuild the original update (max rel. error "
              f"{comp_summary['all']['max_reconstruction_rel_error']:.2e} at {worst}, abs "
              f"{comp[worst]['reconstruction_abs_error']:.2e}, update norm {comp[worst]['original_norm']:.2e}), "
              f"sum c_j = <G_f, P(X)>, selection counts")
        results["components"] = {"selection": args.component_selection, "per_projection": comp_summary}
        if args.component_diagnostics:
            with open(os.path.join(args.out_dir, "components.json"), "w") as f:
                json.dump(to_jsonable({n: {**comp[n], **arrays[n]} for n in comp}), f)
            with Stage("losses after the warm start", timings):
                losses_after = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
            check = {}
            for side, key in (("forget", "predicted_forget_change"), ("retain", "predicted_retain_change")):
                actual = losses_after[side] - losses_before[side]
                predicted = comp_summary["all"][key]
                check[side] = {"before": losses_before[side], "after": losses_after[side], "actual_change": actual,
                               "predicted_change": predicted, "abs_error": abs(actual - predicted),
                               "rel_error": abs(actual - predicted) / abs(predicted) if abs(predicted) > 1e-8 else None}
                print(f"[first-order] {side}: loss {losses_before[side]:.4f} -> {losses_after[side]:.4f}, actual change "
                      f"{actual:.4e}, predicted <G, Delta W> {predicted:.4e}, abs error {check[side]['abs_error']:.4e}, "
                      f"rel error {'-' if check[side]['rel_error'] is None else format(check[side]['rel_error'], '.3g')}")
            results["components"]["first_order_check"] = check
    if target is not None:
        # Positive control: every active target reaches the forget-output target, or uses its whole family.
        reached = {n: r for n, r in ws_report.items() if "forget_output_change" in r}
        short = [n for n, r in reached.items() if r["forget_output_change"] < target and r["rank"] < r["total_rank"]]
        if not reached or short:
            raise RuntimeError(f"forget-output budget missed the target {target:g} at {short[:3]} "
                               f"({len(reached)} targets reported)")
        full = [n for n, r in reached.items() if r["forget_output_change"] < target]
        print(f"[budget] forget-output target {target:g}, forget answer tokens {n_tokens_forget}: " + ", ".join(
            f"{p} q_f {sum(r['forget_output_change'] for r in rs) / len(rs):.4f} rank "
            f"{sum(r['rank'] for r in rs) / len(rs):.0f} (EVR rank {sum(r['reference_rank'] for r in rs) / len(rs):.0f})"
            for p in PROJECTIONS for rs in [[r for n, r in reached.items() if n.endswith(p)]] if rs)
            + f"; at full rank below the target: {len(full)}")
        results["forget_output_budget"] = {"target": target, "below_target_at_full_rank": full}
    if representation and args.restriction_representation == "output_only":
        print("[repr] output_only: no input restriction (Vh = I), so the input-contrast control does not apply")
    elif representation:
        # Positive control: the representation covectors have a higher forget / retain input-energy ratio than
        # the plain covectors of the same rank.
        def contrast(vh, cf, cr):
            vh = vh.to(device=cf.device, dtype=cf.dtype)
            return float(torch.trace(vh @ cf @ vh.transpose(0, 1)) / torch.trace(vh @ cr @ vh.transpose(0, 1)))
        ratio = {}
        for n, r in restrictions.items():
            if r.basis.active:
                plain = svd_basis(source[n], args.restriction_evr, rank=r.basis.rank)
                ratio[n] = {"representation": contrast(r.basis.vh, covs_forget[n], covs[n]),
                            "plain": contrast(plain.vh, covs_forget[n], covs[n]),
                            "kept_forget_gradient_energy": r.basis.evr, "plain_kept_forget_gradient_energy": plain.evr}
        d_mean = sum(e["representation"] for e in ratio.values()) / len(ratio)
        p_mean = sum(e["plain"] for e in ratio.values()) / len(ratio)
        if not d_mean > p_mean:
            raise RuntimeError(f"representation bases show no higher forget/retain contrast than plain ones "
                               f"({d_mean} vs {p_mean})")
        print(f"[repr] forget/retain input energy ratio per basis: representation {d_mean:.3f}, plain {p_mean:.3f}; "
              f"forget-gradient energy kept: representation "
              f"{sum(e['kept_forget_gradient_energy'] for e in ratio.values()) / len(ratio):.3f}, plain "
              f"{sum(e['plain_kept_forget_gradient_energy'] for e in ratio.values()) / len(ratio):.3f}")
        results["representation"] = {"rho": args.whitening_rho, "output_side": args.restriction_representation,
                                     "input_mix": mix, "per_target": ratio}
    if whitening is not None:
        # Positive control: the whitened covectors carry less retain input energy than the plain ones.
        def relative_energy(vh, c):  # mean retain energy per covector / mean energy of a direction
            vh = vh.to(device=c.device, dtype=c.dtype)
            return float(torch.trace(vh @ c @ vh.transpose(0, 1)) / vh.shape[0] / (torch.trace(c) / c.shape[0]))
        energy = {}
        for n, r in restrictions.items():
            if r.basis.active:
                # with an explicit budget the plain basis is compared at the same rank
                plain = svd_basis(source[n], args.restriction_evr, rank=r.basis.rank if (
                    args.restriction_rank_multiplier is not None or target is not None) else None)
                energy[n] = {"whitened": relative_energy(r.basis.vh, whitening[n]),
                             "plain": relative_energy(plain.vh, whitening[n]),
                             "rank_whitened": r.basis.rank, "rank_plain": plain.rank}
        w_mean = sum(e["whitened"] for e in energy.values()) / len(energy)
        p_mean = sum(e["plain"] for e in energy.values()) / len(energy)
        if not w_mean < p_mean:
            raise RuntimeError(f"whitened bases carry no less retain energy than plain ones ({w_mean} vs {p_mean})")
        print(f"[whiten] retain input energy per covector / average direction: whitened {w_mean:.3f}, plain "
              f"{p_mean:.3f}; " + ", ".join(
                  f"{p} rank whitened/plain {sum(e['rank_whitened'] for e in es) / len(es):.0f}/"
                  f"{sum(e['rank_plain'] for e in es) / len(es):.0f}"
                  for p in PROJECTIONS for es in [[e for k, e in energy.items() if k.endswith(p)]] if es))
        results["whitening"] = {"rho": args.whitening_rho, "per_target": energy}
    if args.warm_start_scale is not None or args.warm_start_scale_target is not None:
        # W(t) = W_Init + t (W* - W_Init), each target rebuilt from its Init weights for every t
        starred = {n: m.weight.detach().clone() for n, m in targets.items()}
        init_w = {n: original[n].to(device) for n in targets}

        def set_scale(t):
            with torch.no_grad():
                for n, m in targets.items():
                    m.weight.copy_(scale_update(init_w[n], starred[n], t).to(m.weight.dtype))

        search = []
        scale = args.warm_start_scale
        if args.warm_start_scale_target == "holdout":
            with Stage("scale search", timings):
                lo, hi = 0.0, 1.0 / (1.0 - args.s_min)
                set_scale(hi)
                f_hi = mean_loss(forget)
                search.append((hi, f_hi))
                if f_hi <= holdout_init:
                    scale = hi  # even s = 0 does not reach the target
                else:
                    for _ in range(16):  # forget loss grows with t; bracket [lo, hi] to ~2e-5 in t
                        mid = (lo + hi) / 2
                        set_scale(mid)
                        f_mid = mean_loss(forget)
                        search.append((mid, f_mid))
                        lo, hi = (mid, hi) if f_mid < holdout_init else (lo, mid)
                    scale = hi
        set_scale(scale)
        # Positive control: every target moved by exactly t times its warm-start update.
        moved = {n: float((m.weight.detach().float() - init_w[n].float()).norm()) for n, m in targets.items()}
        full = {n: float((starred[n].float() - init_w[n].float()).norm()) for n in targets}
        bad = [n for n in targets if abs(moved[n] - scale * full[n]) > 1e-4 * max(full[n], 1e-12) + 1e-6]
        if bad:
            raise RuntimeError(f"scaled update is not t times the warm start at {bad[:3]}")
        for n in restrictions:
            restrictions[n] = type(restrictions[n])(baseline=targets[n].weight.detach().clone(),
                                                    basis=restrictions[n].basis)
        scale_report = {"t": scale, "s_equivalent": 1.0 - (1.0 - args.s_min) * scale,
                        "displacement": sum(v * v for v in moved.values()) ** 0.5,
                        "displacement_t1": sum(v * v for v in full.values()) ** 0.5, "search": search}
        if holdout_init is not None:
            scale_report.update({"target_holdout_loss": holdout_init, "forget_loss": mean_loss(forget)})
        print(f"[scale] t = {scale:.5f} (uniform floor equivalent s = {scale_report['s_equivalent']:.4f}); displacement "
              f"{scale_report['displacement']:.3f} of {scale_report['displacement_t1']:.3f} at t = 1"
              + (f"; forget loss {scale_report['forget_loss']:.4f} for the holdout target {holdout_init:.4f} after "
                 f"{len(search)} evaluations" if holdout_init is not None else ""))
        print("[control] scaled update: every target moved by t times its warm-start update")
        results["warm_start_scale"] = scale_report
        del starred, init_w
    if args.variant == "unrestricted" and args.s_min == 1.0:
        # Positive control: s = max(cos, 1) = 1 for every row, so phase one leaves the start model as it is.
        moved = [n for n, r in ws_report.items() if r["weight_rel_change"] != 0.0]
        if moved:
            raise RuntimeError(f"s_min = 1 changed {len(moved)} targets, e.g. {moved[:3]}")
        print("[control] s_min = 1: phase one left every target unchanged; phase two starts from --base_id")
    del grad_retain, grad_forget, source, input_filters, whitening, covs, covs_forget, moments
    torch.cuda.empty_cache()
    ws_summary = summarize_by_projection(ws_report) if gate_summary is None else gate_summary
    for proj, agg in ws_summary.items():
        print(f"[warm-start] {proj}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})}")
    results["warm_start"] = {"per_projection": ws_summary, "per_target": ws_report}
    if args.phase_one_dtype != args.train_dtype:
        tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        model.to(dtype[args.train_dtype])
        # Positive control: every parameter now has the phase-two dtype and the head is tied as before.
        wrong = sorted({str(t.dtype) for t in model.parameters()} - {str(dtype[args.train_dtype])})
        still = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        if wrong or still != tied:
            raise RuntimeError(f"cast to {args.train_dtype} left dtypes {wrong}, head tied {tied} -> {still}")
        print(f"[dtype] phase one in {args.phase_one_dtype}; weights cast to {args.train_dtype} for phase two "
              f"(head tied: {still})")

    # ---------------- restriction backend and trainable set ----------------
    if args.variant == "restricted_semu":
        adapter_report = apply_semu_adapters(model, restrictions)
        print(f"[adapter] {json.dumps(adapter_report)}")
        results["adapter"] = adapter_report
        for p in model.parameters():
            p.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, SemuAdapterLinear):
                m.weight.requires_grad_(True)
        for name in unrestricted_targets:  # shrunk without projection, trained freely (fc rule)
            dict(model.named_modules())[name].weight.requires_grad_(True)
        free = free_parameters(model, args.phase2_free)
        for params in free.values():
            for p in params.values():
                p.requires_grad_(True)
        free_summary = {g: {"tensors": len(ps), "parameters": sum(p.numel() for p in ps.values())} for g, ps in free.items()}
        print(f"[train] trained without an adapter (--phase2_free {args.phase2_free}): {json.dumps(free_summary)}")
        results["phase2_free"] = free_summary
    else:  # unrestricted, restricted_shrink: native weights, every parameter trains
        free = {}
        for p in model.parameters():
            p.requires_grad_(True)
        if args.freeze_input_embedding == "yes":
            model.get_input_embeddings().weight.requires_grad_(False)
        if args.variant == "restricted_shrink":
            print("[train] restricted_shrink: warm start restricted, no adapter; phase two trains every parameter")
        if gated:
            print("[train] direction_gates: gated weights written into the native weights; phase two trains every "
                  "parameter")
    # The head outside the phase-one targets (Classification --no-recast_shrink_fc): trained freely or frozen.
    model.get_output_embeddings().weight.requires_grad_(
        args.train_lm_head == "yes" or "lm_head" in unrestricted_targets or args.variant != "restricted_semu"
        and "lm_head" in targets)
    if args.train_lm_head == "no" and "lm_head" in targets:
        raise ValueError("--train_lm_head no conflicts with a head that is a phase-one target")
    if model.get_input_embeddings().weight.requires_grad and args.untie_lm_head == "yes":
        raise ValueError("an untied head with a trainable input embedding is not a configuration of this campaign")
    print(f"[train] input embedding trainable: {model.get_input_embeddings().weight.requires_grad}; "
          f"lm_head trainable: {model.get_output_embeddings().weight.requires_grad}; shared tensor: "
          f"{model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable parameters {n_train:,} / {n_all:,} ({100.0 * n_train / n_all:.3f}%)")

    # ---------------- phase two: retain NLL only ----------------
    change = {}
    if args.epochs == 0:
        print("[train] --epochs 0: no phase two; the warm-started model is evaluated as it is")
        results["phase_two"] = {"skipped": True}
    else:
        retain_qa = T.QADataset(retain, tok, akey="answer")

        class AnchoredRetain(Dataset):
            """OpenUnlearning ForgetRetainDataset(anchor='forget'), retain side only: len(forget) items,
            each a uniformly random retain example (torch.randint, seeded by the Trainer)."""

            def __len__(self):
                return len(forget)

            def __getitem__(self, idx):
                return retain_qa[int(torch.randint(0, len(retain_qa), (1,)).item())]

        train_ds = AnchoredRetain() if args.epoch_length == "forget" else retain_qa
        collator = T.Collator(tok, "right")

        def collate(items):
            batch = collator(items)
            batch.pop("index")
            return batch

        effective_batch = args.per_device_batch_size * args.grad_accum * max(1, torch.cuda.device_count())
        piecewise = args.lr_scheduler == "piecewise_epochs"
        lr_points = parse_points(args.lr_points) if piecewise else None
        # OpenUnlearning formula for Hugging Face schedulers; the piecewise schedule carries its own warmup.
        warmup_steps = 0 if piecewise else int(args.warmup_epochs * len(train_ds) // effective_batch)

        class PiecewiseTrainer(Trainer):
            """Learning rate linear through (epoch, lr) points; replaces the Hugging Face scheduler."""

            def create_scheduler(self, num_training_steps, optimizer=None):
                if self.lr_scheduler is None:
                    steps_per_epoch = num_training_steps / float(self.args.num_train_epochs)
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer if optimizer is not None else self.optimizer,
                        lambda step: piecewise_linear(lr_points, step / steps_per_epoch) / args.lr)
                    self._created_lr_scheduler = True
                    print(f"[override] piecewise schedule replaces lr_scheduler_type=constant: points={lr_points}, "
                          f"steps/epoch={steps_per_epoch:g}, total steps={num_training_steps}")
                return self.lr_scheduler

        if args.deepspeed_config:
            # DeepSpeed expects a distributed environment: one process, on a free local port.
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": "0", "LOCAL_RANK": "0",
                               "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"})
        targs = TrainingArguments(
            output_dir=os.path.join(args.out_dir, "trainer"), per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, weight_decay=args.weight_decay,
            num_train_epochs=args.epochs, warmup_steps=warmup_steps,
            lr_scheduler_type="constant" if piecewise else args.lr_scheduler,
            optim=args.optim, max_grad_norm=args.max_grad_norm, bf16=True, logging_steps=5, save_strategy="no",
            report_to=[], seed=args.seed, remove_unused_columns=False,
            gradient_checkpointing=args.gradient_checkpointing == "yes", deepspeed=args.deepspeed_config,
        )
        if args.deepspeed_config:
            # The Trainer leaves clipping to DeepSpeed and fills gradient_clipping only where the config says "auto";
            # without the key DeepSpeed does not clip (OpenUnlearning's config has none).
            ds_clip = float(targs.hf_deepspeed_config.config.get("gradient_clipping", 0.0))
            if ds_clip != args.max_grad_norm:
                raise RuntimeError(f"DeepSpeed would clip at {ds_clip:g}, --max_grad_norm is {args.max_grad_norm:g}")
        teacher, teacher_checksums = None, None
        if "kl" in args.retain_loss:  # the model before phase one (or --kl_teacher), frozen
            teacher_id = args.kl_teacher or args.base_id
            teacher = (AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=dtype[args.train_dtype],
                                                            attn_implementation=args.attn_implementation)
                       .to(device).eval())
            teacher.requires_grad_(False)
            teacher_checksums = frozen_checksums(teacher)
            print(f"[train] retain_loss={args.retain_loss} kl_weight={args.kl_weight}: teacher {teacher_id}"
                  + (" (Init Fine.)" if args.kl_teacher is None else " (--kl_teacher)"))

        class RetainLoss:
            """NLL from the Trainer; with a teacher, KL(teacher || model) averaged over non-padding positions."""
            kl_log = []

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                nll, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                    num_items_in_batch=num_items_in_batch)
                if teacher is None:
                    return (nll, outputs) if return_outputs else nll
                with torch.no_grad():
                    ref = teacher(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
                mask = inputs["attention_mask"].bool()
                kl = F.kl_div(F.log_softmax(outputs.logits[mask].float(), dim=-1),
                              F.log_softmax(ref[mask].float(), dim=-1), reduction="batchmean", log_target=True)
                self.kl_log.append(float(kl.detach()))
                # The Trainer leaves the loss unscaled when the model normalises the NLL over the whole
                # accumulation window (num_items_in_batch); the per-micro-batch KL mean then needs the 1/GA.
                if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                    kl = kl / self.current_gradient_accumulation_steps
                loss = args.kl_weight * kl + (nll if "nll" in args.retain_loss else 0.0)
                return (loss, outputs) if return_outputs else loss

        trajectory = []
        # the evaluation sets, all small: retain = retain_perturbed (the 400 retain items of Model Utility)
        tracked = {"forget": forget, "retain": D["rp"], "real_authors": D["ra"], "world_facts": D["wf"]}

        @torch.no_grad()
        def trajectory_point(epoch, log_history=()):
            """Answer-token NLL and exact memorization (teacher-forced, as metric_em, in the training dtype) per
            set, and the epoch's mean training loss; every random state is restored (DataLoader iterators draw a
            seed)."""
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate(), np.random.get_state())
            was_training = model.training
            model.eval()
            losses_ = [h["loss"] for h in log_history if "loss" in h and epoch - 1 < h["epoch"] <= epoch]
            point = {"epoch": epoch, "train_loss_mean": sum(losses_) / len(losses_) if losses_ else None}
            with Stage(f"trajectory epoch {epoch:g}", timings):
                for name, ds in tracked.items():
                    nll, count, em = 0.0, 0, []
                    for b in DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.eval_batch_size,
                                        collate_fn=collator):
                        x = feats(b)
                        logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1]
                        labels = x["labels"][:, 1:]
                        mask = labels != -100
                        nll += float(F.cross_entropy(logits[mask].float(), labels[mask], reduction="sum"))
                        count += int(mask.sum())
                        # per item: the fraction of answer tokens the argmax predicts (metric_em), then the mean
                        hit = ((logits.argmax(-1) == labels) & mask).sum(1)
                        em += [float(h) / int(n) for h, n in zip(hit, mask.sum(1)) if int(n) > 0]
                    point[name] = {"nll": nll / count, "em": sum(em) / len(em)}
            model.train(was_training)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
            random.setstate(rng[2])
            np.random.set_state(rng[3])
            trajectory.append(point)
            results["trajectory"] = trajectory
            print(f"[trajectory] epoch {epoch:g}: train loss "
                  f"{'-' if point['train_loss_mean'] is None else format(point['train_loss_mean'], '.4f')}; "
                  + "; ".join(f"{n} nll {point[n]['nll']:.4f} em {point[n]['em']:.4f}" for n in tracked))
            run.log({"trajectory/epoch": epoch, **{f"trajectory/{n}_{k}": v for n in tracked for k, v in point[n].items()}})

        class Trajectory(TrainerCallback):
            def on_epoch_end(self, train_args, state, control, **kwargs):
                trajectory_point(round(state.epoch), state.log_history)

        trainer_class = type("RetainTrainer", (RetainLoss, PiecewiseTrainer if piecewise else Trainer), {})
        trainer = trainer_class(model=model, args=targs, train_dataset=train_ds, data_collator=collate,
                                callbacks=[Trajectory()] if args.eval_trajectory else None)
        if args.loss_normalization == "micro_batch_mean":
            # As transformers 4.45.1: no num_items_in_batch, so each micro-batch's NLL is its own token mean and the
            # Trainer divides it by the accumulation steps.
            trainer.model_accepts_loss_kwargs = False
        print(f"[train] loss normalization {args.loss_normalization} (model_accepts_loss_kwargs="
              f"{trainer.model_accepts_loss_kwargs}); gradient checkpointing {args.gradient_checkpointing}; "
              f"deepspeed {args.deepspeed_config}")
        print(f"[train] epoch_length={args.epoch_length} items/epoch={len(train_ds)} effective_batch={effective_batch} "
              f"warmup_steps={warmup_steps} optimizer={targs.optim} scheduler={args.lr_scheduler} "
              f"max_grad_norm={targs.max_grad_norm} ({'clipping off' if not targs.max_grad_norm else 'clipping on'})")
        snapshot, checksums = snapshot_trainable(model), frozen_checksums(model)
        if args.eval_trajectory:
            trajectory_point(0)
        with Stage("phase two", timings):
            trainer.train()
        ds_report = None
        if args.deepspeed_config:
            engine = trainer.deepspeed
            ds_report = {"zero_stage": engine.zero_optimization_stage(), "bf16": engine.bfloat16_enabled(),
                         "gradient_clipping": float(engine.gradient_clipping()),
                         "client_optimizer": f"{type(engine.optimizer.optimizer).__module__}."
                                             f"{type(engine.optimizer.optimizer).__name__}",
                         "master_dtypes": sorted({str(t.dtype) for t in engine.optimizer.fp32_partitioned_groups_flat}),
                         "module_dtypes": sorted({str(p.ds_tensor.dtype) for p in model.parameters()})}
            flats = engine.optimizer.fp32_partitioned_groups_flat
            ds_report["master_off_bf16_grid"] = sum(int((c != c.to(torch.bfloat16).float()).sum())
                                                    for t in flats for c in t.split(1 << 26)) / sum(t.numel() for t in flats)
            print(f"[deepspeed] {json.dumps(ds_report)}")
            # Positive controls: float32 master weights behind bfloat16 module weights, most of them off the bfloat16
            # grid after training (small steps kept), and the clipping recorded.
            if (ds_report["master_dtypes"] != ["torch.float32"] or ds_report["module_dtypes"] != ["torch.bfloat16"]
                    or ds_report["master_off_bf16_grid"] <= 0.5):
                raise RuntimeError(f"expected float32 master weights off the bfloat16 grid behind bfloat16 module "
                                   f"weights, got {ds_report}")
            if ds_report["gradient_clipping"] != args.max_grad_norm:
                raise RuntimeError(f"DeepSpeed clipped at {ds_report['gradient_clipping']:g}, --max_grad_norm is "
                                   f"{args.max_grad_norm:g}")
            # ZeRO 3 left the parameters partitioned: gather the 16-bit weights (what OpenUnlearning saves) into a
            # plain model, built without zero.Init, for the reports and the evaluation below.
            from transformers.integrations.deepspeed import unset_hf_deepspeed_config
            state = trainer.accelerator.get_state_dict(engine)
            trainer.accelerator.free_memory()
            unset_hf_deepspeed_config()
            plain = AutoModelForCausalLM.from_config(copy.deepcopy(model.config), torch_dtype=dtype[args.train_dtype],
                                                     attn_implementation=args.attn_implementation)
            missing_keys, unexpected_keys = plain.load_state_dict(state, strict=False)
            tied = plain.get_output_embeddings().weight.data_ptr() == plain.get_input_embeddings().weight.data_ptr()
            if unexpected_keys or set(missing_keys) - ({"lm_head.weight"} if tied else set()) or not tied:
                raise RuntimeError(f"gathered weights: missing {missing_keys}, unexpected {unexpected_keys}, tied {tied}")
            del state, engine
            model = plain.to(device)
            trainer.model = trainer.model_wrapped = trainer.deepspeed = trainer.optimizer = trainer.lr_scheduler = None
            torch.cuda.empty_cache()
            print(f"[deepspeed] 16-bit weights gathered into a plain model ({args.train_dtype}, head tied)")
        if args.eval_trajectory and [p_["epoch"] for p_ in trajectory] != list(range(int(args.epochs) + 1)):
            raise RuntimeError(f"trajectory epochs {[p_['epoch'] for p_ in trajectory]}, expected 0..{args.epochs:g}")
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        change = change_report(model, snapshot, checksums)
        post_warm_start = {n: (restrictions[n].baseline if n in restrictions else snapshot[n + ".weight"]) for n in targets}
        groups = [g for g in PROJECTIONS if any(n.endswith(g) for n in targets)]
        drift = {"vs_original": drift_report(model, original, groups),
                 "vs_post_warm_start": drift_report(model, post_warm_start, groups)}
        for ref, rep in drift.items():
            print(f"[drift] {ref}: " + ", ".join(f"{g} norm_ratio={v['norm_ratio']:.4f} cos={v['cosine']:.4f}" for g, v in rep.items()))
        # Positive control for --phase2_free: every freed group must have moved.
        free_change = {g: {k: change_report(model, {n: snapshot[n] for n in ps}, {})[k]
                           for k in ("trainable_entries", "trainable_entries_changed_fraction", "trainable_rel_change")}
                       for g, ps in free.items()}
        if free_change:
            print(f"[train] free-group changes: {json.dumps(free_change)}")
        del snapshot, original, post_warm_start
        print(f"[train] optimizer steps={trainer.state.global_step} loss first={losses[0] if losses else None} "
              f"last={losses[-1] if losses else None}")
        # Positive control for the schedule: the learning rate the optimizer actually logged.
        lr_log = [(h["step"], h["learning_rate"]) for h in trainer.state.log_history if "learning_rate" in h]
        if lr_log:
            spe = trainer.state.global_step / args.epochs
            picks = [min(lr_log, key=lambda e: abs(e[0] - x * spe)) for x in (0.5, 1, 3, 5, 7.5, args.epochs)]
            print("[train] logged lr at epoch 0.5/1/3/5/7.5/end: " + ", ".join(f"step {s}: {v:.3g}" for s, v in picks))
        print(f"[train] changes: {json.dumps(change)}")
        kl_log = trainer.kl_log
        if teacher is not None:
            # Positive controls: the KL term was computed on every micro-batch, and the teacher never moved.
            expected = math.ceil(len(train_ds) / args.per_device_batch_size) * int(args.epochs)
            if args.epochs == int(args.epochs) and len(kl_log) != expected:
                raise RuntimeError(f"KL computed on {len(kl_log)} micro-batches, expected {expected}")
            if frozen_checksums(teacher) != teacher_checksums:
                raise RuntimeError("the KL teacher changed during phase two")
            k = max(1, len(kl_log) // 20)
            print(f"[train] KL(Init Fine. || model) per position: micro-batches={len(kl_log)} "
                  f"first {k} mean={sum(kl_log[:k]) / k:.4f} last {k} mean={sum(kl_log[-k:]) / k:.4f}; teacher unchanged")
            del teacher
        if gated and not args.deepspeed_config:
            regrowth = RG.regrowth_report({n: targets[n].weight for n in regrowth_saved}, regrowth_saved)
            for scope, groups in regrowth.items():
                print(f"[regrowth] {scope}: " + "; ".join(
                    f"{g} {d['count']} gates, start {d['start']:.3f} -> regrowth {d['regrowth']:.3f}, back {d['back']:.3f}"
                    for g, d in groups.items() if d["count"] and d["energy"] > 0))
            results["direction_gates_regrowth"] = regrowth
            del regrowth_saved
        results["phase_two"] = {"steps": trainer.state.global_step, "losses": losses, "log_history": trainer.state.log_history,
                                "changes": change, "drift": drift,
                                "free_groups": free_change, "warmup_steps": warmup_steps, "items_per_epoch": len(train_ds),
                                "kl_per_microbatch": kl_log, "deepspeed": ds_report}
        del trainer
        torch.cuda.empty_cache()
    if args.save_model_dir:
        with Stage("save model", timings):
            model.save_pretrained(args.save_model_dir)
            tok.save_pretrained(args.save_model_dir)
        print(f"[save] model ({args.train_dtype}) and tokenizer -> {args.save_model_dir}")
        results["saved_model"] = args.save_model_dir
    save_results()

    if full_grads is not None:
        with Stage("losses after phase two", timings):
            after = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
        params = dict(model.named_parameters())
        update = {}
        for scope, names in (("targets", sorted(n + ".weight" for n in targets)), ("all", sorted(init_state))):
            sq, dot_f, dot_r = 0.0, 0.0, 0.0
            for n in names:
                d = params[n].detach().to("cpu", torch.float64) - init_state[n].double()
                sq += float(d.square().sum())
                dot_f += float((full_grads["forget"][n].double() * d).sum())
                dot_r += float((full_grads["retain"][n].double() * d).sum())
            update[scope] = {"tensors": len(names), "update_norm": sq ** 0.5, "predicted_forget_change": dot_f,
                             "predicted_retain_change": dot_r}
        actual = {side: after[side] - losses_before[side] for side in after}
        print(f"[phase-two check] loss forget {losses_before['forget']:.4f} -> {after['forget']:.4f} ({actual['forget']:+.4f}), "
              f"retain {losses_before['retain']:.4f} -> {after['retain']:.4f} ({actual['retain']:+.4f}); " + "; ".join(
                  f"{scope} ({u['tensors']} tensors): ||theta - theta_Init|| {u['update_norm']:.3f}, predicted forget "
                  f"{u['predicted_forget_change']:+.4f} retain {u['predicted_retain_change']:+.4f}"
                  for scope, u in update.items()))
        results["components"]["phase_two_check"] = {"loss_init": losses_before, "loss_after": after,
                                                     "actual_change": actual, "update": update}
        del full_grads, init_state, params
        save_results()

    # ---------------- evaluation ----------------
    model.eval()
    if args.eval_dtype != args.train_dtype:
        model.to(dtype[args.eval_dtype])
        print(f"[eval] cast {args.train_dtype} -> {args.eval_dtype} for evaluation (anchors were scored in bfloat16)")
    strength = (f"{args.direction_gates_basis}, alpha={args.direction_gates_alpha:g}, gamma={args.direction_gates_gamma:g}, "
                f"{args.direction_gates_optim} lr={args.direction_gates_lr:g}"
                + ("" if args.direction_gates_lambda_star is None else f", lambda*={args.direction_gates_lambda_star:g}")
                + (f", init={args.direction_gates_init:g}" if args.direction_gates_init not in (None, 1.0) else "")
                + ("" if args.direction_gates_binarize_share is None else
                   f", binarized share={args.direction_gates_binarize_share:g} s={args.direction_gates_binary_s:g}"
                   + ("" if args.direction_gates_cut is None else
                      f" cut {args.direction_gates_cut}"
                      + (f" {args.direction_gates_cut_s:g}" if args.direction_gates_cut_s is not None else "")))
                + ("" if args.direction_gates_cut is None or args.direction_gates_binarize_share is not None else
                   f", every candidate cut {args.direction_gates_cut}"
                   + (f" {args.direction_gates_cut_s:g}" if args.direction_gates_cut_s is not None else ""))
                + ("" if args.direction_gates_convergence_epochs != 0 or not args.direction_gates_max_epochs else
                   f", fixed {args.direction_gates_max_epochs} epochs lr {args.direction_gates_lr_schedule}")
                if gated
                else f"s_min={floor_label(args.s_min)}")
    row = f"ReCAST {args.variant} ({strength}){' warm start only' if args.epochs == 0 else ''}"
    # compute_full_tofu discards the nine Model Utility sub-metrics and the generations; keep them.
    with TofuCapture(T, ("forget", "retain", "real_authors", "world_facts")) as capture:
        with Stage(f"evaluate {row}", timings):
            raw = T.compute_full_tofu(model, tok, D, bs=args.eval_batch_size)
    subs, generations = capture.check(raw["model_utility"], forget_rouge=raw["forget_Q_A_ROUGE"])
    print("[control] capture: sub-metrics give the reported Model Utility; per-split ROUGE means match")
    raw["_model_utility_subs"] = subs
    print(f"[result] model utility sub-metrics: {json.dumps({k: round(v, 4) for k, v in subs.items()})}")
    with open(os.path.join(args.out_dir, "raw_metrics.json"), "w") as f:
        json.dump(to_jsonable({row: raw}), f)
    with open(os.path.join(args.out_dir, "generations.json"), "w") as f:
        json.dump(generations, f)
    scores = T.evaluate_scores(raw, rt, ft)
    results.update({"row": row, "scores": scores, "anchor_check": anchor_check})
    save_results()
    print(f"[result] {row}: " + ", ".join(f"{c}={scores[c]:.4f}" for c in COLUMNS))
    for ref in ("Engram (alpha=0.6)", "Engram (alpha_W-Norm)"):
        print(f"[result] paper {ref}: " + ", ".join(f"{c}={PAPER_TABLE3[ref][c]:.4f}" for c in COLUMNS))

    run.log({**{f"scores/{c}": scores[c] for c in COLUMNS}, **{f"phase_two/{k}": v for k, v in change.items()
                                                               if isinstance(v, (int, float))}})
    results["cuda_max_memory_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    save_results()
    run.summary.update({"timings_s": timings, "warm_start": to_jsonable(ws_summary), "row": row,
                        "cuda_max_memory_gb": results["cuda_max_memory_gb"],
                        **({"components": to_jsonable(results["components"])} if "components" in results else {})})
    run.finish()


if __name__ == "__main__":
    main()
