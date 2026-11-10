"""ReCAST for causal language models: the TOFU port of the Classification implementation.

Specification: unlearning-svd SEMU/Classification/docs/RECAST_PORTING_CONTRACT.md, with the
Classification code (unlearn/own/dash/) as the behavioral reference:

- phase one (warm start, no optimizer step): example-weighted mean source gradients; per output
  row ``a = cos(W_row, -G_r,row)`` (``plasticity_common.cosine_against_negative_gradient``,
  ``per_filter``); shrink ``s = max(a, s_min)`` (``shrink_floor_mode=fixed``);
- unrestricted: ``W* = s * W``;
- restricted: forget-gradient SVD with the smallest rank reaching the EVR threshold
  (``compute_dash_svd_subspace``), two-sided projection ``P(D) = U U^T D Vh^T Vh``
  (``_project_matrix_with_subspace``), ``W* = W + P(s * W - W)`` (alignment scope
  ``full_space``), then the SEMU adapter ``W = anchor + U A Vh`` with only ``A`` trainable
  (``recast_adapters.py`` / ``CustomLinear``).

The language-model adapters (targets, native loss, data) live here and in table3-style
scripts; none of this touches the engram package.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from recast_bases import compare_cores, weight_components
from recast_contraction import (MODES as CONTRACTION_MODES, component_profile, contraction_beta, contraction_delta,
                                contraction_report, forget_retain_score)
from recast_output_contrast import (output_contrast_contraction, output_contrast_simulation, output_energy,
                                    output_energy_report)

from recast_direction_families import FAMILIES, family_contraction_delta, family_report, ordered_directions
from recast_global_families import (BUDGET_UNITS, GLOBAL_ORDERS, calibrated_selection, delta_from_coef, global_delta,
                                    global_selection, target_directions)

OUTPUT_CONTRAST_RHOS = (0.01, 0.1, 1.0, 30.0)  # ridges of the simulated output-contrast contractions
FAMILY_RHOS = (0.1, 1.0, 30.0)  # ridges of the direction-family comparison
from recast_scores import RULES as SHRINK_SCORES, row_factors, score_summary, shrink_scores, update_stats

COS_EPS = 1.0e-12      # cosine_against_negative_gradient
ENERGY_EPS = 1.0e-12   # compute_dash_svd_subspace


# ---------------- target adapter ----------------

TARGET_PROFILES = {
    "mlp_head": (("gate_proj", "up_proj", "down_proj"), True),
    "mlp": (("gate_proj", "up_proj", "down_proj"), False),
    "mlp_gate": (("gate_proj",), False),
    "mlp_up": (("up_proj",), False),
    "mlp_down": (("down_proj",), False),
    "mlp_down_up": (("down_proj", "up_proj"), False),
    "mlp_up_gate": (("up_proj", "gate_proj"), False),
    "mlp_gate_down": (("gate_proj", "down_proj"), False),
    "down_up_o": (("down_proj", "up_proj", "o_proj"), False),
    "down_o": (("down_proj", "o_proj"), False),
    "up_o": (("up_proj", "o_proj"), False),
    "down_up_q": (("down_proj", "up_proj", "q_proj"), False),
    "mlp_o": (("gate_proj", "up_proj", "down_proj", "o_proj"), False),
    "down_up_v": (("down_proj", "up_proj", "v_proj"), False),
    "down_up_o_v": (("down_proj", "up_proj", "o_proj", "v_proj"), False),
    "down_o_v": (("down_proj", "o_proj", "v_proj"), False),
    "head": ((), True),
    "attn_q": (("q_proj",), False),
    "attn_k": (("k_proj",), False),
    "attn_v": (("v_proj",), False),
    "attn_o": (("o_proj",), False),
    "decoder_linear": (("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), False),
}


def parse_blocks(spec: str) -> Optional[set]:
    """``"all"`` -> None (every block); ``"8-15"`` or ``"0,2,5-7"`` -> the set of block indices."""
    if spec.strip().lower() == "all":
        return None
    blocks = set()
    for part in spec.split(","):
        a, sep, b = part.strip().partition("-")
        blocks.update(range(int(a), int(b) + 1) if sep else {int(a)})
    if not blocks:
        raise ValueError(f"no blocks in {spec!r}")
    return blocks


def select_targets(model: nn.Module, profile: str, prefix: str = "model.layers.",
                   head_name: str = "lm_head", blocks: Optional[set] = None) -> Dict[str, nn.Linear]:
    """Exact nn.Linear targets of a profile: block projections by suffix (optionally only in the given
    block indices), plus the LM head if listed."""
    suffixes, with_head = TARGET_PROFILES[profile]
    targets = {name: m for name, m in model.named_modules()
               if type(m) is nn.Linear and name.startswith(prefix) and name.rsplit(".", 1)[-1] in suffixes
               and (blocks is None or int(name[len(prefix):].split(".", 1)[0]) in blocks)}
    if with_head:
        head = dict(model.named_modules()).get(head_name)
        if type(head) is not nn.Linear:
            raise ValueError(f"profile {profile!r} needs an nn.Linear {head_name!r}")
        targets[head_name] = head
    if not targets:
        raise ValueError(f"no nn.Linear targets for profile {profile!r}")
    biased = [n for n, m in targets.items() if m.bias is not None]
    if biased:
        raise ValueError(f"targets with bias are not supported by this port: {biased[:3]}")
    return targets


@torch.no_grad()
def untie_output_embedding(model: nn.Module) -> dict:
    """Give the output head its own copy of a tied input-embedding matrix.

    The input embedding keeps the original tensor; the head gets an equal, separate Parameter
    and the config stops declaring the tie, so nothing re-ties them later.
    """
    head, emb = model.get_output_embeddings(), model.get_input_embeddings()
    was_tied = head.weight.data_ptr() == emb.weight.data_ptr()
    if was_tied:
        head.weight = nn.Parameter(emb.weight.detach().clone())
    model.config.tie_word_embeddings = False
    return {"was_tied": was_tied, "tied_after": head.weight.data_ptr() == emb.weight.data_ptr(),
            "equal_values": bool(torch.equal(head.weight, emb.weight))}


# ---------------- source gradients (contract section 1, K = 1 view) ----------------

def collect_mean_gradients(
    model: nn.Module,
    loader: Iterable,
    targets: Dict[str, nn.Linear],
    batch_fn: Callable[[dict], dict],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Example-weighted mean over batches of the gradient of each batch's mean loss.

    Only target weights require gradients during collection; training mode and every
    ``requires_grad`` flag are restored afterwards, and no optimizer is involved.
    """
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    names = list(targets)
    weights = [targets[n].weight for n in names]
    for w in weights:
        w.requires_grad_(True)
    sums = {n: torch.zeros_like(w, dtype=torch.float32) for n, w in zip(names, weights)}
    count = 0
    try:
        for batch in loader:
            inputs = batch_fn(batch)
            size = int(inputs["input_ids"].shape[0])
            loss = model(**inputs).loss
            grads = torch.autograd.grad(loss, weights)
            for n, g in zip(names, grads):
                sums[n].add_(g.float(), alpha=float(size))
            count += size
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modes:
            m.train(mode)
    if count == 0:
        raise ValueError("empty source loader")
    return {n: s / float(count) for n, s in sums.items()}, count


def collect_example_gradient_moments(
    model: nn.Module,
    loader: Iterable,
    targets: Dict[str, nn.Linear],
    batch_fn: Callable[[dict], dict],
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], int]:
    """Second moments of the per-example gradients (extension, user 2026-11-02), float64.

    Example e of a batch of size B gets ``g_e = B sum_t delta_t x_t^T``: its share of the gradient of the batch's mean
    loss (``delta_t`` the loss gradient at the target's output, ``x_t`` its input, over the example's positions with
    ``attention_mask`` 1), times B, so that the mean of the g_e over the dataset is exactly the mean gradient of
    ``collect_mean_gradients``. Returns per target ``(E[g g^T], E[g^T g])`` (out x out, in x in), computed as
    ``D_e^T (X_e X_e^T) D_e`` and ``X_e^T (D_e D_e^T) X_e`` without forming any g_e; their mean part is
    ``G G^T`` and ``G^T G``. Modes and ``requires_grad`` flags are restored as in ``collect_mean_gradients``.
    """
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    names = list(targets)
    for n in names:
        targets[n].weight.requires_grad_(True)
    saved: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = [targets[n].register_forward_hook(lambda mod, inp, out, n=n: saved.__setitem__(n, (inp[0].detach(), out)))
             for n in names]
    sums = {n: (torch.zeros(targets[n].out_features, targets[n].out_features, dtype=torch.float64,
                            device=targets[n].weight.device),
                torch.zeros(targets[n].in_features, targets[n].in_features, dtype=torch.float64,
                            device=targets[n].weight.device)) for n in names}
    count = 0
    try:
        for batch in loader:
            inputs = batch_fn(batch)
            size = int(inputs["input_ids"].shape[0])
            loss = model(**inputs).loss
            deltas = torch.autograd.grad(loss, [saved[n][1] for n in names], allow_unused=True)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            for n, d in zip(names, deltas):
                if d is None:
                    continue
                x = saved[n][0].float() * mask
                d = d.float() * mask * float(size)
                out_sum, in_sum = sums[n]
                out_sum.add_(d.flatten(0, -2).transpose(0, 1) @ (x @ x.transpose(-1, -2) @ d).flatten(0, -2))
                in_sum.add_(x.flatten(0, -2).transpose(0, 1) @ (d @ d.transpose(-1, -2) @ x).flatten(0, -2))
            saved.clear()
            count += size
    finally:
        for h in hooks:
            h.remove()
        saved.clear()
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modes:
            m.train(mode)
    if count == 0:
        raise ValueError("empty source loader")
    return {n: (o / float(count), i / float(count)) for n, (o, i) in sums.items()}, count


# ---------------- retain input filter (extension of this port) ----------------

@torch.no_grad()
def collect_input_covariances(
    model: nn.Module,
    loader: Iterable,
    targets: Dict[str, nn.Linear],
    batch_fn: Callable[[dict], dict],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Uncentred covariance ``E[x x^T]`` of each target's input over answer tokens (``labels != -100``,
    Engram's answer-token ``mask_fn``), summed in fp32; training mode is restored afterwards."""
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    sums = {n: torch.zeros(t.in_features, t.in_features, dtype=torch.float32, device=t.weight.device)
            for n, t in targets.items()}
    state = {"mask": None}

    def hook(name):
        def fn(module, args):
            x = args[0][state["mask"]].float()  # (answer tokens, in_features)
            sums[name].addmm_(x.transpose(0, 1), x)
        return fn

    handles = [t.register_forward_pre_hook(hook(n)) for n, t in targets.items()]
    count = 0
    try:
        for batch in loader:
            inputs = batch_fn(batch)
            state["mask"] = inputs["labels"] != -100
            count += int(state["mask"].sum())
            model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    finally:
        for h in handles:
            h.remove()
        for m, mode in modes:
            m.train(mode)
    if count == 0:
        raise ValueError("no answer tokens in the loader")
    return {n: s / float(count) for n, s in sums.items()}, count


@torch.no_grad()
def ridge_input_filter(cov: torch.Tensor, kappa: float) -> Tuple[torch.Tensor, float]:
    """``M = mu (C + mu I)^-1`` with ``mu = kappa * tr(C) / d``.

    Right-multiplying a delta by M keeps each eigen-direction of the retain input covariance C
    (eigenvalue lambda) with weight ``mu / (lambda + mu)``: the share ``lambda / (lambda + mu)``
    that retain uses is removed. ``kappa -> inf`` is no filter, ``kappa -> 0`` removes every
    direction retain touches. Computed in float64, symmetrised, returned in float32.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    c = cov.double()
    d = c.shape[0]
    mu = kappa * float(torch.trace(c)) / d
    m = mu * torch.linalg.inv(c + mu * torch.eye(d, dtype=c.dtype, device=c.device))
    return ((m + m.transpose(0, 1)) / 2).float(), mu


# ---------------- shrink proposal (contract section 2) ----------------

def row_alignment(weight: torch.Tensor, grad: torch.Tensor, eps: float = COS_EPS) -> torch.Tensor:
    """``cos(W_row, -G_row)`` per output row, as cosine_against_negative_gradient(per_filter)."""
    w = weight.reshape(weight.shape[0], -1).float()
    g = grad.reshape(grad.shape[0], -1).float()
    dot = (w * (-g)).sum(dim=1)
    return dot / (w.norm(dim=1) * g.norm(dim=1) + eps)


def shrink_factors(alignment: torch.Tensor, s_min: float) -> torch.Tensor:
    """``shrink_floor_mode=fixed``: ``max(a, s_min)``; values stay in [s_min, 1] since cos <= 1."""
    if not 0.0 <= s_min <= 1.0:
        raise ValueError(f"s_min must be in [0, 1], got {s_min}")
    return alignment.clamp(min=float(s_min))


# ---------------- restriction subspace (contract section 3) ----------------

@dataclass
class Basis:
    active: bool
    reason: str
    u: Optional[torch.Tensor]
    vh: Optional[torch.Tensor]
    rank: int
    total_rank: int
    evr: float
    rank_in: Optional[int] = None  # rows of vh when they differ from ``rank`` (the columns of u); None: the same


def share_rank(energies: torch.Tensor, threshold: float) -> int:
    """Smallest k whose leading ``energies`` (in their given order) hold at least ``threshold`` of their sum."""
    cum = torch.cumsum(energies.double().clamp_min(0.0), 0)
    return min(int((cum / cum[-1] < threshold).sum()) + 1, int(energies.numel()))


def svd_basis(source: torch.Tensor, evr_threshold: float, eps: float = ENERGY_EPS,
              rank: Optional[int] = None) -> Basis:
    """fp32 SVD; rank = smallest k with cumulative energy fraction >= threshold, or ``rank`` when given
    (extension: an explicit budget; ``evr`` then reports the energy fraction it keeps)."""
    if not 0.0 < float(evr_threshold) <= 1.0:
        raise ValueError(f"evr_threshold must be in (0, 1], got {evr_threshold}")
    if rank is not None and rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    matrix = source.reshape(source.shape[0], -1).detach().float()
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    energy = s.square()
    total = energy.sum()
    total_rank = int(s.numel())
    if not torch.isfinite(total) or float(total) <= eps:
        return Basis(False, "zero_energy", None, None, 0, total_rank, 0.0)
    evr = torch.cumsum(energy, dim=0) / total
    if rank is None:
        rank = int(torch.searchsorted(evr, torch.tensor(float(evr_threshold), device=evr.device, dtype=evr.dtype)).item()) + 1
    rank = max(1, min(rank, total_rank))
    return Basis(True, "", u[:, :rank].contiguous(), vh[:rank, :].contiguous(), rank, total_rank, float(evr[rank - 1]))


def whitened_svd_basis(source: torch.Tensor, cov: torch.Tensor, evr_threshold: float, rho: float,
                       eps: float = ENERGY_EPS, rank: Optional[int] = None) -> Basis:
    """Restriction basis C (extension of this port): the forget gradient measured against the retain
    input covariance.

    ``A = C_r + epsilon I`` with ``epsilon = rho tr(C_r) / d``. The SVD of ``G A^-1/2`` (EVR rank as
    ``svd_basis``) gives the output directions U; its right singular vectors w, mapped back as
    ``A^-1/2 w`` and orthonormalised, are the input covectors with the most forget-gradient energy
    per unit of retain input energy (generalised eigenproblem ``G^T G v = gamma A v``). A delta built
    on them changes retain outputs ``E||D x||^2 = tr(D C_r D^T)`` least for its forget alignment.
    ``rho -> inf`` gives ``svd_basis``.
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    a = cov.double()
    d = a.shape[0]
    lam, q = torch.linalg.eigh(a + rho * float(torch.trace(a)) / d * torch.eye(d, dtype=a.dtype, device=a.device))
    a_inv_sqrt = (q * lam.rsqrt()) @ q.transpose(0, 1)
    g = source.reshape(source.shape[0], -1).double()
    white = svd_basis((g @ a_inv_sqrt).float(), evr_threshold, eps, rank=rank)
    if not white.active:
        return white
    v, _ = torch.linalg.qr(a_inv_sqrt @ white.vh.double().transpose(0, 1))
    return Basis(True, "", white.u, v.transpose(0, 1).float().contiguous(), white.rank, white.total_rank, white.evr)


def _generalised_top(c_num: torch.Tensor, c_den: torch.Tensor, rho: float, rank: int) -> torch.Tensor:
    """Orthonormal columns spanning the top-``rank`` solutions of ``c_num v = gamma (c_den + eps I) v``,
    eps = rho tr(c_den) / d (float64)."""
    a = c_den.double()
    d = a.shape[0]
    lam, q = torch.linalg.eigh(a + rho * float(torch.trace(a)) / d * torch.eye(d, dtype=a.dtype, device=a.device))
    a_inv_sqrt = (q * lam.rsqrt()) @ q.transpose(0, 1)
    m = a_inv_sqrt @ c_num.double() @ a_inv_sqrt
    gamma, w = torch.linalg.eigh((m + m.transpose(0, 1)) / 2)
    top = w[:, torch.argsort(gamma, descending=True)[:rank]]
    v, _ = torch.linalg.qr(a_inv_sqrt @ top)
    return v


def representation_basis(source: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor,
                         weight: torch.Tensor, rank: int, rho: float, output_side: str,
                         eps: float = ENERGY_EPS, input_mix: float = 1.0) -> Basis:
    """Restriction basis D (extension of this port): the input covectors come from the forget and
    retain *representations*, not the gradient.

    Input side: the top-``rank`` solutions of ``N v = gamma (C_r + eps I) v`` (input covariances
    over answer tokens, eps = rho tr(C_r) / d). ``input_mix`` beta sets the forget signal
    ``N = (1 - beta) G_f^T G_f / tr + beta C_f / tr``: 1 = the forget activations (D), 0 = the forget
    gradient (the input side of C); in between = the joint contrast. Output side: ``identity`` (no
    output restriction, ``u=None``), ``gradient`` (left singular vectors of ``G_f V``: the outputs the
    forget loss moves along those inputs), ``whitened_gradient`` (the top-``rank`` left singular
    vectors of ``G_f (C_r + eps I)^-1/2``, the output side of C) or ``representation`` (the same
    contrast for the output covariances ``W C W^T``). ``output_only`` (user, 2026-10-29) restricts only the output side
    (the ``representation`` contrast) and leaves every input direction (``Vh = I``). The rank is an explicit budget;
    ``evr`` reports the forget-gradient energy fraction the restriction keeps, ``||P(G_f)||^2 / ||G_f||^2``.
    """
    if output_side not in ("identity", "gradient", "whitened_gradient", "representation", "output_only"):
        raise ValueError(f"output_side must be identity, gradient, whitened_gradient, representation or output_only, "
                         f"got {output_side!r}")
    if rho <= 0 or rank < 1:
        raise ValueError(f"rho and rank must be positive, got {rho}, {rank}")
    if not 0.0 <= input_mix <= 1.0:
        raise ValueError(f"input_mix must be in [0, 1], got {input_mix}")
    g = source.reshape(source.shape[0], -1).double()
    total_rank = min(g.shape)
    if float(torch.trace(cov_forget)) <= eps or float(g.square().sum()) <= eps:
        return Basis(False, "zero_energy", None, None, 0, total_rank, 0.0)
    if output_side == "output_only":
        w = weight.reshape(weight.shape[0], -1).double()
        rank = min(rank, g.shape[0])
        u = _generalised_top(w @ cov_forget.double() @ w.transpose(0, 1), w @ cov_retain.double() @ w.transpose(0, 1),
                             rho, rank)
        kept = u @ (u.transpose(0, 1) @ g)
        eye = torch.eye(g.shape[1], dtype=torch.float32, device=g.device)
        return Basis(True, "", u.float().contiguous(), eye, rank, total_rank, float(kept.square().sum() / g.square().sum()))
    rank = min(rank, g.shape[1])
    numerator = cov_forget.double() / float(torch.trace(cov_forget))
    if input_mix < 1.0:
        gtg = g.transpose(0, 1) @ g
        numerator = input_mix * numerator + (1.0 - input_mix) * gtg / float(torch.trace(gtg))
    v = _generalised_top(numerator, cov_retain, rho, rank)  # (in, r)
    if output_side == "identity":
        u = None
    elif output_side == "gradient":
        u = torch.linalg.svd(g @ v, full_matrices=False)[0][:, :min(rank, g.shape[0])]
    elif output_side == "whitened_gradient":
        a = cov_retain.double()
        lam, q = torch.linalg.eigh(a + rho * float(torch.trace(a)) / a.shape[0]
                                   * torch.eye(a.shape[0], dtype=a.dtype, device=a.device))
        u = torch.linalg.svd(g @ ((q * lam.rsqrt()) @ q.transpose(0, 1)), full_matrices=False)[0][:, :min(rank, g.shape[0])]
    else:
        w = weight.reshape(weight.shape[0], -1).double()
        u = _generalised_top(w @ cov_forget.double() @ w.transpose(0, 1), w @ cov_retain.double() @ w.transpose(0, 1),
                             rho, min(rank, g.shape[0]))
    kept = g @ v @ v.transpose(0, 1) if u is None else u @ (u.transpose(0, 1) @ g @ v) @ v.transpose(0, 1)
    return Basis(True, "", None if u is None else u.float().contiguous(), v.transpose(0, 1).float().contiguous(),
                 rank, total_rank, float(kept.square().sum() / g.square().sum()))


GRADIENT_CONTRAST_SIDES = ("both", "input")
CONTRAST_RANK_RULES = ("side_energy", "kept_energy")  # own-spectrum ranks of the contrast basis (gradient_contrast_basis)


def _rank_arguments(rank: Optional[int], energy_threshold: Optional[float],
                    kept_threshold: Optional[float] = None) -> None:
    if sum(x is not None for x in (rank, energy_threshold, kept_threshold)) != 1:
        raise ValueError("give exactly one of rank, energy_threshold and kept_threshold")
    if rank is not None and rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    for name, value in (("energy_threshold", energy_threshold), ("kept_threshold", kept_threshold)):
        if value is not None and not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {value}")


def gradient_contrast_basis(source: torch.Tensor, grad_retain: torch.Tensor, rank: Optional[int], rho: float,
                            sides: str, eps: float = ENERGY_EPS, energy_threshold: Optional[float] = None,
                            kept_threshold: Optional[float] = None) -> Basis:
    """Restriction basis from the forget-vs-retain contrast of the mean gradients (extension, user 2026-11-01): the
    plain basis with the singular vectors of G_f replaced by contrast directions.

    Input side: the leading directions of ``(G_f^T G_f / tr) v = lambda (G_r^T G_r / tr + rho / in I) v``.
    Output side, ``sides="both"``: the leading directions of
    ``(G_f G_f^T / tr) u = lambda (G_r G_r^T / tr + rho / out I) u``; ``sides="input"`` (control): the leading
    left singular vectors of G_f, the plain basis's output side. Each family is orthonormalised in the order
    of lambda (``recast_direction_families.ordered_directions``). ``rank`` is an explicit budget, the same on both
    sides; ``energy_threshold`` instead (user, 2026-11-02) gives each side the smallest number of leading directions
    that holds at least that share of the mean forget gradient's energy on that side, ``sum_i ||G_f v_i||^2 /
    ||G_f||^2`` on the input side and ``sum_i ||G_f^T u_i||^2 / ||G_f||^2`` on the output side (lambda is a ratio, not
    an energy; for the plain basis both shares are its EVR). ``kept_threshold`` instead (user, 2026-11-02) gives both
    sides one rank, the smallest r whose two-sided restriction keeps at least that share of the mean forget gradient's
    energy, ``||U_r U_r^T G_f V_r V_r^T||^2 / ||G_f||^2`` (at most min(out, in); for the plain basis again its EVR).
    ``evr`` reports the forget-gradient energy fraction the two-sided restriction keeps, ``||P(G_f)||^2 / ||G_f||^2``.
    """
    if sides not in GRADIENT_CONTRAST_SIDES:
        raise ValueError(f"sides must be one of {GRADIENT_CONTRAST_SIDES}, got {sides!r}")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    _rank_arguments(rank, energy_threshold, kept_threshold)
    g = source.reshape(source.shape[0], -1).double()
    gr = grad_retain.reshape(grad_retain.shape[0], -1).to(device=g.device, dtype=g.dtype)
    total_rank = min(g.shape)
    if float(g.square().sum()) <= eps or float(gr.square().sum()) <= eps:
        return Basis(False, "zero_energy", None, None, 0, total_rank, 0.0)
    v = ordered_directions(g.transpose(0, 1) @ g, gr.transpose(0, 1) @ gr, rho)
    if sides == "both":
        u = ordered_directions(g @ g.transpose(0, 1), gr @ gr.transpose(0, 1), rho)
        out_energy = (g.transpose(0, 1) @ u).square().sum(0) if energy_threshold is not None else None
    else:
        u, s, _ = torch.linalg.svd(g, full_matrices=False)
        out_energy = s.square()
    if rank is not None:
        r_out = r_in = min(rank, total_rank)
    elif kept_threshold is not None:  # kept(r) = sum_{i, j <= r} (u_i^T G_f v_j)^2 / ||G_f||^2
        core = (u[:, :total_rank].transpose(0, 1) @ g @ v[:, :total_rank]).square()
        curve = torch.cumsum(torch.cumsum(core, 0), 1).diagonal() / g.square().sum()
        r_out = r_in = min(int((curve < kept_threshold).sum()) + 1, total_rank)
    else:
        r_out, r_in = share_rank(out_energy, energy_threshold), share_rank((g @ v).square().sum(0), energy_threshold)
    u, v = u[:, :r_out], v[:, :r_in]
    kept = u @ (u.transpose(0, 1) @ g @ v) @ v.transpose(0, 1)
    return Basis(True, "", u.float().contiguous(), v.transpose(0, 1).float().contiguous(), r_out, total_rank,
                 float(kept.square().sum() / g.square().sum()), r_in if energy_threshold is not None else None)


def gradient_moment_basis(source: torch.Tensor, moment_out: torch.Tensor, moment_in: torch.Tensor,
                          rank: Optional[int] = None, eps: float = ENERGY_EPS,
                          energy_threshold: Optional[float] = None) -> Basis:
    """Restriction basis from the per-example forget gradients (extension, user 2026-11-02): the plain basis with the
    singular vectors of the mean gradient G_f replaced by the leading eigenvectors of the per-example second moments
    (``collect_example_gradient_moments``), U from ``E[g g^T]`` and V from ``E[g^T g]``.

    With one gradient shared by every example the moments are ``G_f G_f^T`` and ``G_f^T G_f`` and the basis spans the
    plain basis's subspaces; directions along which examples pull in different ways (cancelling in the mean) enter
    here with their full per-example energy. ``rank`` is an explicit budget, the same on both sides;
    ``energy_threshold`` instead gives each side the EVR rank of its own moment's spectrum (the smallest k whose
    leading eigenvalues hold that share of the trace), so the input side of a wide weight may exceed min(out, in).
    ``evr`` reports the mean-gradient energy fraction the restriction keeps, ``||P(G_f)||^2 / ||G_f||^2``, as
    ``gradient_contrast_basis`` does.
    """
    _rank_arguments(rank, energy_threshold)
    g = source.reshape(source.shape[0], -1).double()
    total_rank = min(g.shape)
    if not torch.isfinite(g).all() or float(g.square().sum()) <= eps:
        return Basis(False, "zero_energy", None, None, 0, total_rank, 0.0)
    lam_out, u = torch.linalg.eigh(moment_out.to(device=g.device, dtype=torch.float64))
    lam_in, v = torch.linalg.eigh(moment_in.to(device=g.device, dtype=torch.float64))
    lam_out, u, lam_in, v = lam_out.flip(0), u.flip(1), lam_in.flip(0), v.flip(1)
    if rank is not None:
        r_out = r_in = min(rank, total_rank)
    else:
        r_out, r_in = share_rank(lam_out, energy_threshold), share_rank(lam_in, energy_threshold)
    u, v = u[:, :r_out], v[:, :r_in]
    kept = u @ (u.transpose(0, 1) @ g @ v) @ v.transpose(0, 1)
    return Basis(True, "", u.float().contiguous(), v.transpose(0, 1).float().contiguous(), r_out, total_rank,
                 float(kept.square().sum() / g.square().sum()), None if rank is not None else r_in)


def project(delta: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor) -> torch.Tensor:
    """Two-sided projection ``U U^T D Vh^T Vh`` in fp32; ``u=None`` leaves the output side free
    (one-sided ``D Vh^T Vh``)."""
    m = delta.reshape(delta.shape[0], -1).float()
    vh = vh.float()
    if u is None:
        return ((m @ vh.transpose(0, 1)) @ vh).reshape_as(delta)
    u = u.float()
    return (u @ (u.transpose(0, 1) @ m @ vh.transpose(0, 1)) @ vh).reshape_as(delta)


def forget_output_change(delta: torch.Tensor, weight: torch.Tensor, cov: torch.Tensor) -> float:
    """``q = tr(D C D^T) / tr(W C W^T) = E||D x||^2 / E||W x||^2`` for inputs with uncentred covariance
    ``C``, as sums of elementwise products (no out x out matrix)."""
    c = cov.to(device=delta.device, dtype=torch.float32)
    d, w = delta.reshape(delta.shape[0], -1).float(), weight.reshape(weight.shape[0], -1).float()
    return float(((d @ c) * d).sum() / ((w @ c) * w).sum())


def basis_prefix(full: Basis, rank: int, source: torch.Tensor, gradient_output: bool) -> Basis:
    """The rank-``rank`` member of a basis family built at a larger rank. Covectors are nested (leading
    singular vectors, or the QR of leading (generalised) eigenvectors) and so are output directions,
    except D's gradient output side, recomputed from ``G_f V`` (by QR: the same span as its left singular
    vectors when ``G_f V`` has full column rank). ``evr`` = ``||P(G_f)||^2 / ||G_f||^2``."""
    vh = full.vh[:rank].contiguous()
    g = source.reshape(source.shape[0], -1).float()
    if full.u is None:
        u = None
    elif gradient_output:
        u = torch.linalg.qr(g.double() @ vh.double().transpose(0, 1))[0].float().contiguous()
    else:
        u = full.u[:, :rank].contiguous()
    kept = float(project(g, u, vh).square().sum() / g.square().sum())
    return Basis(True, "", u, vh, rank, full.total_rank, kept)


def prefix_for_forget_output(full: Basis, source: torch.Tensor, weight: torch.Tensor, proposal: torch.Tensor,
                             cov_forget: torch.Tensor, target: float, gradient_output: bool) -> Tuple[Basis, float]:
    """Smallest prefix rank whose delta ``P(proposal)`` changes the forget outputs by ``q >= target``
    (bisection: q grows with the rank of nested bases), or the full rank when even it falls short."""
    def change(rank):
        b = basis_prefix(full, rank, source, gradient_output)
        return b, forget_output_change(project(proposal, b.u, b.vh), weight, cov_forget)

    lo, hi = 1, full.rank
    best = change(hi)
    if best[1] < target:
        return best
    while lo < hi:
        mid = (lo + hi) // 2
        cand = change(mid)
        if cand[1] >= target:
            hi, best = mid, cand
        else:
            lo = mid + 1
    return best


# ---------------- components of a projected update (extension: signed shrink) ----------------

COMPONENT_SELECTIONS = ("full", "signed", "random", "uniform_matched", "contrast")


@dataclass
class UpdateComponents:
    """``P(X) = U B Vh`` with the core ``B = U^T X Vh^T = P diag(sigma) Q^T`` (thin SVD): the update is the sum of the
    Frobenius-orthogonal components ``D_j = sigma_j (U p_j)(Vh^T q_j)^T``, with first-order forget contributions
    ``c_j = <G_f, D_j>_F = sigma_j p_j^T (U^T G_f Vh^T) q_j``. float64; ``U = I`` when the output side is free."""
    left: torch.Tensor          # U P, (out, k)
    sigma: torch.Tensor         # (k,)
    right: torch.Tensor         # Q^T Vh, (k, in)
    contribution: torch.Tensor  # c_j, (k,)


def decompose_update(proposal: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor,
                     grad: torch.Tensor) -> UpdateComponents:
    """Components of ``project(proposal, u, vh)``; the SVD runs on the small core, not the full matrix."""
    x = proposal.reshape(proposal.shape[0], -1).double()
    g = grad.reshape(grad.shape[0], -1).double()
    v = vh.double()
    if u is None:
        core, k = x @ v.transpose(0, 1), g @ v.transpose(0, 1)
    else:
        ud = u.double()
        core, k = ud.transpose(0, 1) @ x @ v.transpose(0, 1), ud.transpose(0, 1) @ g @ v.transpose(0, 1)
    p, sigma, qh = torch.linalg.svd(core, full_matrices=False)
    contribution = sigma * ((p.transpose(0, 1) @ k) * qh).sum(1)
    return UpdateComponents(p if u is None else ud @ p, sigma, qh @ v, contribution)


def compose_update(c: UpdateComponents, alpha: torch.Tensor) -> torch.Tensor:
    """``sum_j alpha_j D_j`` (float64)."""
    return (c.left * (alpha * c.sigma)) @ c.right


def input_contrast(rows: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor) -> torch.Tensor:
    """Per input covector b (a row): the forget / retain share of input energy along it,
    (b C_f b^T / tr C_f) / (b C_r b^T / tr C_r), computed without any gradient (fp64)."""
    r = rows.double()
    cf, cr = cov_forget.to(r.device).double(), cov_retain.to(r.device).double()
    num = ((r @ cf) * r).sum(1) / float(torch.trace(cf))
    den = ((r @ cr) * r).sum(1) / float(torch.trace(cr))
    return num / den.clamp_min(1e-300)


def direction_contrast_weights(vh: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor,
                               s_min: float) -> torch.Tensor:
    """Gradient-free per-direction shrink weights (extension): w_k = contrast_k / mean(contrast) for the input
    covectors of ``vh``, clipped to [0, 1 / (1 - s_min)] so that every direction keeps a shrink factor
    s_k = 1 - (1 - s_min) w_k in [0, 1]; the mean weight is 1 before clipping (the same average shrink as the
    uniform update, moved towards the directions forget uses more than retain)."""
    contrast = input_contrast(vh, cov_forget, cov_retain)
    w = contrast / contrast.mean()
    cap = 1.0 / (1.0 - float(s_min)) if s_min < 1 else 1.0
    return w.clamp(0.0, cap)


def component_weights(c: UpdateComponents, selection: str, generator: Optional[torch.Generator] = None,
                      eps: float = ENERGY_EPS, contrast: Optional[torch.Tensor] = None) -> torch.Tensor:
    """alpha per component: ``full`` 1; ``signed`` 1 where c_j > 0, else 0; ``random`` as many ones as ``signed`` at
    positions drawn by ``generator``; ``uniform_matched`` the constant gamma = ||signed update|| / ||full update||
    (0 for a zero update), i.e. the full update scaled to the Frobenius norm of the signed one; ``contrast``
    (gradient-free) 1 where the component's input covector carries a larger share of forget than of retain input
    energy (``contrast`` > 1, from ``input_contrast``), else 0."""
    n = c.sigma.numel()
    ones = torch.ones(n, dtype=c.sigma.dtype, device=c.sigma.device)
    positive = (c.contribution > 0).to(c.sigma.dtype)
    if selection == "full":
        return ones
    if selection == "signed":
        return positive
    if selection == "contrast":
        if contrast is None:
            raise ValueError("the contrast selection needs the input contrast of every component")
        return (contrast > 1.0).to(c.sigma.dtype)
    if selection == "random":
        alpha = torch.zeros(n, dtype=c.sigma.dtype)
        alpha[torch.randperm(n, generator=generator)[:int(positive.sum())]] = 1.0
        return alpha.to(c.sigma.device)
    if selection == "uniform_matched":
        energy = c.sigma.square()
        total = float(energy.sum())
        return ones * ((float((positive * energy).sum()) / total) ** 0.5 if total > eps else 0.0)
    raise ValueError(f"component selection must be one of {COMPONENT_SELECTIONS}, got {selection!r}")


def effective_rank(values: torch.Tensor) -> float:
    """Roy-Vetterli effective rank exp(H(p)), p = values / sum(values), for non-negative singular values."""
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values[values > 0] / total
    return float(torch.exp(-(p * p.log()).sum()))


def scale_update(original: torch.Tensor, baseline: torch.Tensor, t: float) -> torch.Tensor:
    """``W(t) = W_Init + t (W* - W_Init)`` from the Init weights, never from an already modified one (fp32)."""
    w0 = original.float()
    return w0 + float(t) * (baseline.float().to(w0.device) - w0)


# The two writes of a decoder layer into the residual stream, in the order the layer adds them.
WRITE_PROJECTIONS = ("self_attn.o_proj", "mlp.down_proj")


def residual_write_divisors(num_layers: int, factors: Dict[str, float], prefix: str = "model.layers.") -> Dict[str, float]:
    """Scaling the residual stream by c_k at write k (``h <- c_k h + write_k``, extension: write compensation) leaves
    every RMSNorm read unchanged if instead write k and every later write are divided by ``c_1 ... c_k``; returns that
    divisor for every o_proj and down_proj in layer order (factor 1 for a write without one)."""
    out, prod = {}, 1.0
    for i in range(num_layers):
        for p in WRITE_PROJECTIONS:
            name = f"{prefix}{i}.{p}"
            prod *= float(factors.get(name, 1.0))
            out[name] = prod
    return out


@torch.no_grad()
def apply_write_divisors(model: nn.Module, divisors: Dict[str, float]) -> None:
    """Divide each named write's weight by its divisor in place (the writes have no bias in Llama)."""
    modules = dict(model.named_modules())
    for name, d in divisors.items():
        if getattr(modules[name], "bias", None) is not None:
            raise ValueError(f"{name} has a bias; the write compensation assumes none")
        modules[name].weight.div_(d)


def update_budget(floors: Dict[str, float], energies: Dict[str, float]) -> float:
    """``B = sqrt(sum_k (1 - s_k)^2 E_k)``: the Frobenius norm of the whole restricted update when every row of a type
    k sits at its floor s_k, so that each target's update is ``(s_k - 1) P(W_l)``; ``E_k = sum_{l in k} ||P(W_l)||^2``
    (the ``projected_weight_norm`` of the warm-start report, squared and summed per type)."""
    if set(floors) != set(energies):
        raise ValueError(f"floors and energies need the same types, got {sorted(floors)} and {sorted(energies)}")
    return math.sqrt(sum((1.0 - floors[k]) ** 2 * energies[k] for k in floors))


def uniform_floor_for_budget(budget: float, energies: Dict[str, float]) -> Optional[float]:
    """The one floor s with ``update_budget({k: s}, energies) = budget``, ``1 - budget / sqrt(sum_k E_k)``; None when
    no s in [0, 1] reaches it (the budget exceeds the full projected norm, i.e. it would need s < 0)."""
    total = math.sqrt(sum(energies.values()))
    s = 1.0 - budget / total
    return s if 0.0 <= s <= 1.0 else None


def parse_type_values(spec: str) -> Dict[str, float]:
    """``down_proj=0.2,up_proj=0.5`` -> {projection: value}, e.g. a shrink floor per projection type (extension)."""
    out: Dict[str, float] = {}
    for item in spec.split(","):
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key or key in out:
            raise ValueError(f"expected projection=t pairs without repeats, got {spec!r}")
        out[key] = float(value)
    return out


def component_report(c: UpdateComponents, alpha: torch.Tensor, original: torch.Tensor, applied: torch.Tensor,
                     grad_forget: torch.Tensor, grad_retain: torch.Tensor, keep_arrays: bool) -> dict:
    """Per-target diagnostics of the decomposition and of the update actually applied."""
    energy = c.sigma.square()
    total = float(energy.sum())
    pos, neg = c.contribution > 0, c.contribution < 0
    d = original.reshape(original.shape[0], -1).double()
    full = compose_update(c, torch.ones_like(alpha))
    rep = {
        "components": int(c.sigma.numel()), "positive": int(pos.sum()), "negative": int(neg.sum()),
        "kept": int((alpha > 0).sum()), "alpha_max": float(alpha.max()) if alpha.numel() else 0.0,
        "energy": total,
        "energy_positive_fraction": float(energy[pos].sum()) / total if total > 0 else 0.0,
        "energy_negative_fraction": float(energy[neg].sum()) / total if total > 0 else 0.0,
        "contribution_sum": float(c.contribution.sum()), "contribution_positive": float(c.contribution[pos].sum()),
        "contribution_negative": float(c.contribution[neg].sum()),
        "first_order_forget_direct": float((grad_forget.reshape_as(d).double() * d).sum()),  # <G_f, P(X)>
        "reconstruction_rel_error": float((d - full).norm() / d.norm()) if float(d.norm()) > 0 else 0.0,
        "reconstruction_abs_error": float((d - full).norm()), "grad_forget_norm": float(grad_forget.norm()),
        "original_norm": float(d.norm()), "applied_norm": float(applied.norm()),
        "applied_energy_fraction": float((alpha.square() * energy).sum()) / total if total > 0 else 0.0,
        "effective_rank_original": effective_rank(c.sigma), "effective_rank_applied": effective_rank(alpha * c.sigma),
        "predicted_forget_change": float((alpha * c.contribution).sum()),
        "predicted_retain_change": float((grad_retain.reshape_as(d).double() * applied.reshape_as(d)).sum()),
    }
    if keep_arrays:
        rep["sigma"] = c.sigma.tolist()
        rep["contribution"] = c.contribution.tolist()
    return rep


# ---------------- warm start ----------------

@dataclass
class Restriction:
    baseline: torch.Tensor  # W*, the post-warm-start weight
    basis: Basis


@torch.no_grad()
def mix_sources(grad_forget: torch.Tensor, grad_retain: torch.Tensor, n_forget: int, n_retain: int) -> torch.Tensor:
    """``full_data`` restriction source: the exact example-weighted mean gradient over forget + retain."""
    if n_forget <= 0 or n_retain <= 0:
        raise ValueError("full_data needs positive example counts")
    return (n_forget * grad_forget.float() + n_retain * grad_retain.float()) / float(n_forget + n_retain)


@torch.no_grad()
def warm_start(
    targets: Dict[str, nn.Linear],
    grad_retain: Dict[str, torch.Tensor],
    s_min: Union[float, Dict[str, float]],
    restriction_source: Optional[Dict[str, torch.Tensor]] = None,
    evr_threshold: Optional[float] = None,
    unrestricted: Iterable[str] = (),
    alignment_scope: str = "full_space",
    layer_scaling: str = "none",
    input_filters: Optional[Dict[str, torch.Tensor]] = None,
    whitening: Optional[Dict[str, torch.Tensor]] = None,
    whitening_rho: Optional[float] = None,
    rank_multiplier: Optional[float] = None,
    representation_output: Optional[str] = None,
    gradient_contrast: Optional[str] = None,
    gradient_moments: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    contrast_rank_rule: str = "side_energy",
    forget_covariances: Optional[Dict[str, torch.Tensor]] = None,
    retain_covariances: Optional[Dict[str, torch.Tensor]] = None,
    representation_mix: float = 1.0,
    forget_output_target: Optional[float] = None,
    component_selection: Optional[str] = None,
    component_diagnostics: bool = False,
    component_seed: int = 0,
    direction_weighting: Optional[str] = None,
    shrink_score: str = "original",
    score_diagnostics: bool = False,
    core_diagnostics: bool = False,
    weight_contraction: Optional[str] = None,
    contraction_strength: float = 1.0,
    contraction_residual: float = 0.0,
    contraction_diagnostics: bool = False,
    output_diagnostics: bool = False,
    output_contrast_rho: Optional[float] = None,
    family_diagnostics: bool = False,
    family_contraction: Optional[str] = None,
    family_rho: Optional[float] = None,
    global_family: Optional[str] = None,
    global_family_rho: Optional[float] = None,
    global_family_floor: Optional[float] = None,
    global_family_order: str = "score",
    global_family_alpha: Optional[float] = None,
    global_family_budget_unit: str = "energy",
    global_family_s_gamma: Optional[float] = None,
    global_family_s_rule: Optional[str] = None,
    global_family_s_threshold_gamma: Optional[float] = None,
) -> Tuple[Dict[str, Restriction], Dict[str, dict]]:
    """Apply the ReCAST warm start in place. Restricted when ``restriction_source`` is given.

    ``s_min`` is the shrink floor, one value for every target or (extension) a dict with one floor per projection
    type, the last component of the target name (``{"down_proj": 0.2, "up_proj": 0.5}``).

    Restricted targets whose source has no energy follow ``zero_energy_behavior=skip_projection``:
    the raw shrink proposal is applied and the target gets no basis (it stays a native,
    unrestricted parameter). Targets listed in ``unrestricted`` get the same treatment by design,
    as ``fc`` under the Classification ``--recast_train_fc_unrestricted``.

    ``alignment_scope="restriction_subspace"`` (Classification option) aligns ``P(W)`` with
    ``-P(G_r)`` and shrinks ``P(W)``: ``W* = W + P((s - 1) * P(W))``. ``layer_scaling="subspace_norm"``
    (extension of this port, after Engram's alpha_W-Norm) gives each restricted target the floor
    ``1 - (1 - s_min) * r_l / max r`` with ``r_l = ||P(W_l)|| / ||W_l||`` (it can only weaken the
    shrink); ``"subspace_norm_mean"`` divides by the mean of r instead and clips to [0, 1], so the
    average shrink amount stays ``1 - s_min`` (before clipping). ``input_filters`` (extension, e.g.
    ``ridge_input_filter`` of the retain input covariance) right-multiply the projected delta of
    every active restricted target, ``W* = W + P(s W - W) M``, so the delta leaves the input
    directions retain uses; the adapter basis stays the forget subspace. ``whitening`` (retain input
    covariances, extension) builds each restriction basis with ``whitened_svd_basis`` at
    ``whitening_rho`` instead of ``svd_basis``. ``rank_multiplier`` m (extension: an explicit edit
    budget) gives every restricted target the rank ``round(m * r0)``, r0 the rank of the plain basis
    at ``evr_threshold``, for the plain and the whitened basis alike, so the two compare at equal rank.
    ``representation_output`` (extension, basis D) builds each basis with ``representation_basis``
    from ``forget_covariances`` and ``retain_covariances`` at ``whitening_rho`` and ``representation_mix``
    (the input-side mix of forget activations and forget gradient); it needs a budget.
    ``gradient_contrast`` (extension, user 2026-11-01: ``both`` or ``input``) builds each basis with
    ``gradient_contrast_basis`` from the forget gradient ``restriction_source`` and ``grad_retain`` at ``whitening_rho``;
    with ``rank_multiplier`` both sides get its rank, without it (user 2026-11-02) ``contrast_rank_rule``
    ``side_energy`` gives each side the rank holding ``evr_threshold`` of the mean forget gradient's energy on that side
    and ``kept_energy`` both sides the smallest rank whose two-sided restriction keeps ``evr_threshold`` of it.
    ``gradient_moments`` (extension, user 2026-11-02: per target the per-example forget-gradient moments
    ``(E[g g^T], E[g^T g])`` of ``collect_example_gradient_moments``) builds each basis with ``gradient_moment_basis``;
    with ``rank_multiplier`` both sides get its rank, without it each side the EVR rank of its own moment's spectrum at
    ``evr_threshold``. These own-spectrum ranks differ between the sides, so they exclude component selection and
    diagnostics, direction weighting and score and core diagnostics.
    ``forget_output_target`` q* (extension: a budget in forget-output units, instead of
    ``rank_multiplier``) gives every restricted target the smallest rank of its basis family whose delta
    changes the forget outputs by ``q_f = tr(D C_f D^T) / tr(W C_f W^T) >= q*`` (``C_f`` from
    ``forget_covariances``), so the energy-weighted total ``sum tr(D C_f D^T) / sum tr(W C_f W^T)``
    reaches q* too; the report gives each target's ``forget_output_change``.
    ``component_selection`` (extension: signed shrink) splits each active target's projected update ``P(X)``,
    ``X = S W - W``, into the Frobenius-orthogonal components of ``decompose_update`` with the forget gradient
    ``restriction_source`` and applies ``sum_j alpha_j D_j`` with alpha from ``component_weights`` (``random`` draws
    from a generator seeded by ``component_seed`` and the target name); ``uniform_matched`` applies gamma P(X) itself.
    ``component_diagnostics`` reports the decomposition (``components`` per target, with sigma and c_j) without
    changing the update unless a selection is given; the ``contrast`` selection (gradient-free) keeps the components
    whose input covector carries a larger share of forget than of retain input energy. ``direction_weighting
    ="contrast"`` (gradient-free) shrinks each input direction of the basis by s_k = 1 - (1 - s) w_k with w_k from
    ``direction_contrast_weights`` (``forget_covariances`` / ``retain_covariances``), i.e. the update
    ``U U^T X Vh^T diag(w) Vh``. ``shrink_score`` (extension, Stage 2) picks the row score of S on each active
    restricted target from ``recast_scores``: ``original`` (the cosine above, unchanged), ``projected_row``,
    ``projected_sensitivity`` or ``global_projected`` (one factor per matrix); the floor and the proposal
    ``P((S - I) W)`` stay. ``score_diagnostics`` reports, per active restricted target, every rule's score summary,
    the first-order retain / forget change ``<G, Delta W>`` of the applied update and every rule's update norm on a
    floor grid (for norm matching). ``core_diagnostics`` (Stage 3, ``recast_bases.compare_cores``) compares, per active
    restricted target, the weight core ``U^T W Vh^T`` with the update core ``U^T (S - I) W Vh^T`` for the applied S,
    for each rule's S at the same floor and for a control S with factors drawn uniformly from [floor, 1] (seeded by
    the target name), with the gradients' alignment and, given both covariances, the input contrast of the leading
    directions; it changes no update. ``weight_contraction`` (extension, ``recast_contraction``) replaces the update
    of each active restricted target by ``-sum_j beta_j E_j`` over the weight-core components ``E_j = kappa_j u_j
    v_j^T`` of ``P(W)``: beta_j = ``contraction_strength`` on the components with the largest kappa (``top``), on
    random ones (``random``, seeded by ``component_seed`` and the target name) or on those with the largest first-order
    forget-minus-retain loss increase per unit of moved energy (``forget_retain``, from the forget gradient
    ``restriction_source`` and ``grad_retain``) until the displacement equals
    ``(1 - floor) ||P(W)||``, that of ReCAST with every row at the floor; ``contraction`` per target reports it with
    the input contrast of the moved energy; ``contraction_residual`` leaves every contracted component that singular
    value instead of 0. ``contraction_diagnostics`` reports, per active restricted target, every weight-core
    component's kappa, input contrast and forget / retain gradient alignment (``contraction_components``) without
    changing the update. ``output_diagnostics`` (``recast_output_contrast``) reports, per active restricted target, the
    output ratios and removed output shares on forget and retain inputs of the applied update (``output_energy``), of
    every SVD ordering at the same displacement when the components are computed (``output_energy_svd``) and, for
    down_proj and o_proj, of the output-contrast contraction at each ridge of ``OUTPUT_CONTRAST_RHOS`` with the
    applied displacement (``output_contrast``); it changes no update. ``output_contrast_rho`` (extension) replaces the
    update of each active restricted target by the output-contrast contraction at that ridge with the displacement of
    the ReCAST update (``output_contrast_applied``). ``family_diagnostics`` (``recast_direction_families``) reports, for
    down_proj and o_proj, the retain output share each family of forget-specific directions (output, gradient and
    input contrast at each ridge of ``FAMILY_RHOS``) removes when it removes the forget output share of the applied
    update (``families``); it changes no update. ``family_contraction`` (``output_contrast`` or ``input_contrast``,
    with ``family_rho``) replaces the update of each active restricted target by that family's removal of the same
    forget output share as the ReCAST update (``family_contraction`` per target). ``global_family`` (one of
    ``recast_direction_families.FAMILIES``, with ``global_family_rho``, ``global_family_floor`` and
    ``global_family_order``; ``recast_global_families``) replaces the updates of all targets (down_proj and o_proj
    only) together: the family's directions of every target enter one order (``score``: by lambda; ``random``: keys
    drawn per target from a generator seeded by ``component_seed`` and the target name; ``energy_ratio``: by
    ``e_f / e_r^global_family_alpha``, the absolute output energies of each direction alone), each chosen direction keeps
    ``global_family_floor`` of its part, until the forget output energy removed over all targets equals the total the
    ReCAST updates at ``s_min`` remove (``global_family`` per target, the totals under ``global``).
    ``global_family_budget_unit="share"`` counts each target's removed forget energy as a share of its own forget output
    energy and matches the sum of the ReCAST updates' shares instead; it admits targets of any type (``energy`` admits
    down_proj and o_proj only, whose outputs share the residual stream). ``global_family_s_gamma`` gives each direction
    its own kept share ``clip(lambda^-gamma, global_family_floor, 1)`` instead of the floor itself;
    ``global_family_s_rule="retain_share"`` gives it ``clip(e_r / (e_f + e_r), global_family_floor, 1)`` from the
    absolute output energies of removing the direction alone; ``global_family_s_threshold_gamma`` contracts every
    direction by ``clip((lambda* / lambda)^gamma, global_family_floor, 1)`` with one threshold lambda* chosen so that
    the budget is met (``recast_global_families.calibrated_selection``). Targets are updated
    independently from the Init Fine. weights and gradients, each exactly once.
    """
    if alignment_scope not in ("full_space", "restriction_subspace"):
        raise ValueError(f"alignment_scope must be full_space or restriction_subspace, got {alignment_scope!r}")
    if layer_scaling not in ("none", "subspace_norm", "subspace_norm_mean"):
        raise ValueError(f"layer_scaling must be none, subspace_norm or subspace_norm_mean, got {layer_scaling!r}")
    unrestricted = set(unrestricted)
    restricted = restriction_source is not None
    if isinstance(s_min, dict):
        missing = sorted({n.rsplit(".", 1)[-1] for n in targets} - set(s_min))
        if missing or layer_scaling != "none":
            raise ValueError(f"per-type floors need a floor for every target type (missing {missing}) and "
                             "layer_scaling none")

    def floor_of(name):
        return float(s_min[name.rsplit(".", 1)[-1]] if isinstance(s_min, dict) else s_min)

    if restricted and evr_threshold is None:
        raise ValueError("restricted warm start needs evr_threshold")
    if not restricted and (alignment_scope != "full_space" or layer_scaling != "none" or input_filters is not None
                           or whitening is not None):
        raise ValueError("restriction_subspace alignment, subspace_norm scaling, input filters and whitening need "
                         "a restriction")
    if whitening is not None and whitening_rho is None:
        raise ValueError("whitening needs whitening_rho")
    if rank_multiplier is not None and (not restricted or rank_multiplier <= 0):
        raise ValueError("rank_multiplier needs a restriction and a positive value")
    if (component_selection is not None or component_diagnostics) and (
            not restricted or input_filters is not None
            or (component_selection is not None and component_selection not in COMPONENT_SELECTIONS)):
        raise ValueError(f"component selection / diagnostics need a restriction and no input filter; selections: "
                         f"{COMPONENT_SELECTIONS}")
    if component_selection == "contrast" and (forget_covariances is None or retain_covariances is None):
        raise ValueError("the contrast selection needs forget_covariances and retain_covariances")
    if direction_weighting is not None and (
            direction_weighting != "contrast" or not restricted or input_filters is not None
            or component_selection is not None or component_diagnostics
            or forget_covariances is None or retain_covariances is None):
        raise ValueError("direction_weighting='contrast' needs a restriction, both covariances, no input filter and "
                         "no component selection or diagnostics (they decompose the unweighted update)")
    if shrink_score not in SHRINK_SCORES:
        raise ValueError(f"shrink_score must be one of {SHRINK_SCORES}, got {shrink_score!r}")
    if (shrink_score != "original" or score_diagnostics or core_diagnostics) and (
            not restricted or alignment_scope != "full_space" or layer_scaling != "none" or input_filters is not None
            or direction_weighting is not None or component_selection is not None
            or forget_output_target is not None):
        raise ValueError("shrink_score other than original and score_diagnostics need a restriction, full_space "
                         "alignment, no layer scaling, input filter, direction weighting, component selection or "
                         "forget-output target (Stage 2 changes only the row scores)")
    if forget_output_target is not None and (
            not restricted or not 0.0 < forget_output_target < 1.0 or rank_multiplier is not None
            or forget_covariances is None or alignment_scope != "full_space" or layer_scaling != "none"
            or input_filters is not None):
        raise ValueError("forget_output_target needs a restriction, a value in (0, 1), forget_covariances, "
                         "full_space alignment, no layer scaling, no input filter and no rank_multiplier")
    if gradient_contrast is not None and (
            gradient_contrast not in GRADIENT_CONTRAST_SIDES or not restricted
            or whitening_rho is None or whitening_rho <= 0 or whitening is not None or representation_output is not None
            or forget_output_target is not None):
        raise ValueError(f"a gradient-contrast basis ({GRADIENT_CONTRAST_SIDES}) needs a restriction and a positive "
                         "whitening_rho, and excludes whitening, representation bases and forget_output_target")
    if gradient_moments is not None and (
            not restricted or gradient_contrast is not None or whitening is not None
            or representation_output is not None or forget_output_target is not None):
        raise ValueError("per-example gradient moments need a restriction and exclude gradient-contrast, whitened and "
                         "representation bases and forget_output_target")
    own_spectrum = (gradient_contrast is not None or gradient_moments is not None) and rank_multiplier is None
    if own_spectrum and (component_selection is not None or component_diagnostics or direction_weighting is not None
                         or score_diagnostics or core_diagnostics):
        raise ValueError("own-spectrum ranks (a gradient-contrast or per-example basis without rank_multiplier) exclude "
                         "component selection and diagnostics, direction weighting and score and core diagnostics")
    if contrast_rank_rule not in CONTRAST_RANK_RULES or (
            contrast_rank_rule != "side_energy" and (gradient_contrast is None or not own_spectrum)):
        raise ValueError(f"contrast_rank_rule must be one of {CONTRAST_RANK_RULES}; other than side_energy it needs a "
                         "gradient-contrast basis without rank_multiplier")
    contrast_thresholds = ({} if not own_spectrum else {"kept_threshold": evr_threshold}
                           if contrast_rank_rule == "kept_energy" else {"energy_threshold": evr_threshold})
    if representation_output is not None and (
            (rank_multiplier is None and forget_output_target is None) or whitening_rho is None
            or whitening is not None or forget_covariances is None or retain_covariances is None):
        raise ValueError("a representation basis needs rank_multiplier or forget_output_target, whitening_rho and "
                         "both covariances, and excludes whitening")
    if weight_contraction is not None and (
            weight_contraction not in CONTRACTION_MODES or not 0.0 < contraction_strength <= 1.0
            or representation_output is None or alignment_scope != "full_space" or layer_scaling != "none"
            or input_filters is not None or direction_weighting is not None or component_selection is not None
            or component_diagnostics or forget_output_target is not None or shrink_score != "original"
            or score_diagnostics or core_diagnostics
            or any(1.0 - floor_of(n) > contraction_strength for n in targets)):
        raise ValueError(f"weight_contraction ({CONTRACTION_MODES}) needs a strength in (0, 1], a representation "
                         "basis, full_space alignment, the original shrink score, floors with 1 - floor <= strength "
                         "and no layer scaling, input filter, direction weighting, component selection or "
                         "diagnostics, forget-output target, score or core diagnostics")
    if contraction_residual < 0.0 or (contraction_residual > 0.0 and weight_contraction is None):
        raise ValueError("contraction_residual needs a weight_contraction and a value >= 0")
    if contraction_diagnostics and (
            representation_output is None or alignment_scope != "full_space" or layer_scaling != "none"
            or input_filters is not None or direction_weighting is not None or component_selection is not None
            or forget_output_target is not None):
        raise ValueError("contraction_diagnostics needs a representation basis, full_space alignment and no layer "
                         "scaling, input filter, direction weighting, component selection or forget-output target")
    if output_diagnostics and (not restricted or forget_covariances is None or retain_covariances is None):
        raise ValueError("output_diagnostics needs a restriction and both input covariances")
    if output_contrast_rho is not None and (
            output_contrast_rho <= 0 or not restricted or forget_covariances is None or retain_covariances is None
            or weight_contraction is not None or direction_weighting is not None or component_selection is not None
            or input_filters is not None or forget_output_target is not None or shrink_score != "original"):
        raise ValueError("output_contrast_rho needs a positive value, a restriction, both input covariances, the "
                         "original shrink score and no weight contraction, direction weighting, component selection, "
                         "input filter or forget-output target")
    if family_diagnostics and (not restricted or forget_covariances is None or retain_covariances is None):
        raise ValueError("family_diagnostics needs a restriction and both input covariances")
    if (family_contraction is None) != (family_rho is None) or (family_contraction is not None and (
            family_contraction not in ("output_contrast", "input_contrast") or family_rho <= 0 or not restricted
            or forget_covariances is None or retain_covariances is None or weight_contraction is not None
            or output_contrast_rho is not None or direction_weighting is not None or component_selection is not None
            or input_filters is not None or forget_output_target is not None or shrink_score != "original")):
        raise ValueError("family_contraction (output_contrast or input_contrast) needs a positive family_rho, a "
                         "restriction, both input covariances, the original shrink score and no other update "
                         "replacement, direction weighting, component selection, input filter or forget-output target")
    if (global_family is None) != (global_family_rho is None) or (global_family is None) != (
            global_family_floor is None) or (global_family is not None and (
            global_family not in FAMILIES or global_family_rho <= 0 or not 0.0 <= global_family_floor < 1.0
            or global_family_order not in GLOBAL_ORDERS
            or (global_family_order == "energy_ratio") != (global_family_alpha is not None)
            or (global_family_alpha is not None and global_family_alpha < 0)
            or not restricted or forget_covariances is None
            or retain_covariances is None or unrestricted
            or global_family_budget_unit not in BUDGET_UNITS
            or (global_family_s_gamma is not None and global_family_s_gamma <= 0)
            or global_family_s_rule not in (None, "retain_share")
            or (global_family_s_rule is not None and global_family_s_gamma is not None)
            or (global_family_s_threshold_gamma is not None and (
                global_family_s_threshold_gamma <= 0 or global_family_s_gamma is not None
                or global_family_s_rule is not None or global_family_order != "score"))
            or (global_family_budget_unit == "energy"
                and any(n.rsplit(".", 1)[-1] not in ("down_proj", "o_proj") for n in targets))
            or weight_contraction is not None or output_contrast_rho is not None or family_contraction is not None
            or direction_weighting is not None or component_selection is not None or input_filters is not None
            or forget_output_target is not None or shrink_score != "original")):
        raise ValueError(f"global_family (one of {FAMILIES}) needs global_family_rho > 0, global_family_floor in "
                         f"[0, 1), global_family_order in {GLOBAL_ORDERS} (energy_ratio with, and only it, a "
                         "global_family_alpha >= 0), a restriction of every target, both input "
                         "covariances, only down_proj / o_proj targets with the energy budget unit, the original "
                         "shrink score and no other update "
                         "replacement, direction weighting, component selection, input filter or forget-output target")

    # Pass 1: bases and per-target floors.
    bases: Dict[str, Basis] = {}
    reference_rank: Dict[str, int] = {}
    if restricted:
        for name, module in targets.items():
            if name in unrestricted:
                bases[name] = Basis(False, "parameter_unrestricted", None, None, 0, min(module.weight.shape), 0.0)
                continue
            rank = None
            if rank_multiplier is not None or forget_output_target is not None:
                reference_rank[name] = svd_basis(restriction_source[name], evr_threshold).rank
            if rank_multiplier is not None:
                rank = max(1, round(rank_multiplier * reference_rank[name]))
            if forget_output_target is not None:
                rank = min(module.weight.shape)  # the whole family; the target picks a prefix below
            if gradient_contrast is not None:
                bases[name] = gradient_contrast_basis(restriction_source[name], grad_retain[name], rank, whitening_rho,
                                                      gradient_contrast, **contrast_thresholds)
            elif gradient_moments is not None:
                bases[name] = gradient_moment_basis(restriction_source[name], *gradient_moments[name], rank,
                                                    energy_threshold=evr_threshold if own_spectrum else None)
            elif representation_output is not None:
                bases[name] = representation_basis(restriction_source[name], forget_covariances[name],
                                                   retain_covariances[name], module.weight.detach(), rank,
                                                   whitening_rho, representation_output,
                                                   input_mix=representation_mix)
            elif whitening is not None:
                bases[name] = whitened_svd_basis(restriction_source[name], whitening[name], evr_threshold,
                                                 whitening_rho, rank=rank)
            else:
                bases[name] = svd_basis(restriction_source[name], evr_threshold, rank=rank)
            if forget_output_target is not None and bases[name].active:
                w32 = module.weight.detach().float()
                proposal = (shrink_factors(row_alignment(w32, grad_retain[name]), floor_of(name)) - 1.0).unsqueeze(1) * w32
                bases[name], _ = prefix_for_forget_output(bases[name], restriction_source[name], w32, proposal,
                                                          forget_covariances[name], forget_output_target,
                                                          representation_output == "gradient")
    floors = {name: floor_of(name) for name in targets}
    rel_norm: Dict[str, float] = {}
    if layer_scaling != "none":
        for name, b in bases.items():
            if b.active:
                w32 = targets[name].weight.detach().float()
                rel_norm[name] = float(project(w32, b.u, b.vh).norm() / (w32.norm() + COS_EPS))
        values = list(rel_norm.values())
        ref = (max(values) if layer_scaling == "subspace_norm" else sum(values) / len(values)) if values else 0.0
        for name, r in rel_norm.items():
            floors[name] = min(1.0, max(0.0, 1.0 - (1.0 - float(s_min)) * (r / ref if ref > 0 else 0.0)))

    # Pass 2: alignment, shrink, projection.
    restrictions: Dict[str, Restriction] = {}
    report: Dict[str, dict] = {}
    family_parts: Dict[str, dict] = {}
    for name, module in targets.items():
        w = module.weight
        w32 = w.detach().float().clone()  # .float() of a float32 weight aliases it; the report needs the original
        basis = bases.get(name)
        in_subspace = alignment_scope == "restriction_subspace" and basis is not None and basis.active
        shrink_source = project(w32, basis.u, basis.vh) if in_subspace else w32
        g = project(grad_retain[name].float(), basis.u, basis.vh) if in_subspace else grad_retain[name]
        a = row_alignment(shrink_source, g)
        s = shrink_factors(a, floors[name])
        scores = None
        if (shrink_score != "original" or score_diagnostics or core_diagnostics) and basis is not None and basis.active:
            scores = shrink_scores(w32, grad_retain[name], basis.u, basis.vh, eps=COS_EPS)
            if shrink_score != "original":
                s = row_factors(scores, shrink_score, floors[name]).float().to(w32.device)
                a = (torch.full_like(s, scores["global_projected"]) if shrink_score == "global_projected"
                     else scores[shrink_score].float().to(w32.device))
        proposal = (s - 1.0).unsqueeze(1) * shrink_source   # s * X - X, X = W or P(W)
        info = {"rows": int(s.numel()), "floor": floors[name], "shrink_mean": float(s.mean()),
                "shrink_at_floor": float((a <= floors[name]).float().mean()), "alignment_mean": float(a.mean()),
                "proposal_norm": float(proposal.norm()), "shrink_score": shrink_score}
        if scores is not None and score_diagnostics:
            # core_row scores the r output directions of the basis, not the rows: no correlation with the original
            info["scores"] = {rule: score_summary(scores[rule], reference=None if rule == "core_row" else scores["original"])
                              for rule in ("original", "projected_row", "projected_sensitivity", "core_row")}
            info["scores"]["global_projected"] = scores["global_projected"]
            info["scores"]["zero_norm_rows"] = scores["zero_norm_rows"]
            info["applied_first_order"] = update_stats(w32, s, basis.u, basis.vh, {
                "retain": grad_retain[name], "forget": restriction_source[name]})
            info["update_norm_by_floor"] = {
                rule: {f"{f:g}": update_stats(w32, row_factors(scores, rule, f), basis.u, basis.vh, {})["update_norm"]
                       for f in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)} for rule in SHRINK_SCORES}
        if scores is not None and core_diagnostics:
            gen = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            control = floors[name] + (1.0 - floors[name]) * torch.rand(s.numel(), generator=gen, dtype=torch.float64)
            sources = {"applied": s, **{rule: row_factors(scores, rule, floors[name]) for rule in SHRINK_SCORES},
                       "control_uniform_random": control.to(w32.device)}
            contrast_fn = None
            if forget_covariances is not None and retain_covariances is not None:
                contrast_fn = lambda rows, n=name: input_contrast(rows, forget_covariances[n], retain_covariances[n])  # noqa: E731
            info["cores"] = {k: compare_cores(w32, f, basis.u, basis.vh, grads={
                "retain": grad_retain[name], "forget": restriction_source[name]}, contrast_fn=contrast_fn)
                for k, f in sources.items()}
        if name in rel_norm:
            info["subspace_rel_norm"] = rel_norm[name]
        if name in reference_rank:
            info["reference_rank"] = reference_rank[name]
        if restricted:
            delta = project(proposal, basis.u, basis.vh) if basis.active else proposal
            if (weight_contraction is not None or contraction_diagnostics) and basis.active:
                comps = weight_components(w32, basis.u, basis.vh)
                contrast = input_contrast(comps["right"], forget_covariances[name], retain_covariances[name])
                if contraction_diagnostics:
                    info["contraction_components"] = component_profile(
                        comps, {"forget": restriction_source[name], "retain": grad_retain[name]}, contrast)
                if weight_contraction is not None:
                    generator = torch.Generator().manual_seed(int(component_seed) * 1000003 + zlib.crc32(name.encode()))
                    score = (forget_retain_score(comps, restriction_source[name], grad_retain[name])
                             if weight_contraction == "forget_retain"
                             else contrast if weight_contraction == "contrast" else None)
                    beta = contraction_beta(comps["kappa"], weight_contraction, 1.0 - floors[name], contraction_strength,
                                            generator, residual=contraction_residual, score=score)
                    info["contraction"] = contraction_report(comps["kappa"], beta, contraction_strength, contrast,
                                                             residual=contraction_residual)
                    info["contraction"].update({"mode": weight_contraction, "strength": contraction_strength,
                                                "residual": contraction_residual, "budget": 1.0 - floors[name],
                                                "budget_reached": info["contraction"]["energy_fraction"]
                                                >= (1.0 - floors[name]) ** 2 * (1.0 - 1e-9),
                                                "recast_update_norm": float(delta.norm())})
                    delta = contraction_delta(comps, beta).to(delta.dtype)
                if output_diagnostics:  # every SVD ordering at the ReCAST displacement, beta 1, without applying it
                    sims = {}
                    for mode in CONTRACTION_MODES:
                        gen = torch.Generator().manual_seed(int(component_seed) * 1000003 + zlib.crc32(name.encode()))
                        sc = (forget_retain_score(comps, restriction_source[name], grad_retain[name])
                              if mode == "forget_retain" else contrast if mode == "contrast" else None)
                        b = contraction_beta(comps["kappa"], mode, 1.0 - floors[name], 1.0, gen, score=sc)
                        sims[mode] = output_energy_report(w32, contraction_delta(comps, b), forget_covariances[name],
                                                          retain_covariances[name])
                    info["output_energy_svd"] = sims
                del comps
            if output_contrast_rho is not None and basis.active:  # same displacement as the ReCAST update it replaces
                oc_delta, oc_report = output_contrast_contraction(w32, forget_covariances[name], retain_covariances[name],
                                                                  output_contrast_rho, float(delta.norm()))
                info["output_contrast_applied"] = dict(oc_report, rho=output_contrast_rho,
                                                       recast_update_norm=float(delta.norm()))
                delta = oc_delta.to(delta.dtype)
            if family_contraction is not None and basis.active:  # the forget output share of the ReCAST update
                recast_out = output_energy_report(w32, delta, forget_covariances[name], retain_covariances[name])
                fam_delta, fam_report = family_contraction_delta(w32, forget_covariances[name], retain_covariances[name],
                                                                 family_contraction, family_rho,
                                                                 recast_out["forget_removed_share"])
                info["family_contraction"] = dict(fam_report, family=family_contraction, rho=family_rho,
                                                  recast_forget_share=recast_out["forget_removed_share"],
                                                  recast_retain_share=recast_out["retain_removed_share"],
                                                  recast_update_norm=float(delta.norm()))
                delta = fam_delta.to(delta.dtype)
            if direction_weighting is not None and basis.active:
                vh32 = basis.vh.float()
                w_dir = direction_contrast_weights(basis.vh, forget_covariances[name], retain_covariances[name],
                                                   floors[name]).float().to(delta.device)
                delta = ((delta @ vh32.transpose(0, 1)) * w_dir) @ vh32
                cap = 1.0 / (1.0 - floors[name]) if floors[name] < 1 else 1.0
                info["direction_weights"] = {"mean": float(w_dir.mean()), "min": float(w_dir.min()),
                                             "max": float(w_dir.max()),
                                             "at_cap": float((w_dir >= cap - 1e-6).float().mean())}
            if (component_selection is not None or component_diagnostics) and basis.active:
                parts = decompose_update(proposal, basis.u, basis.vh, restriction_source[name])
                generator = torch.Generator().manual_seed(int(component_seed) * 1000003 + zlib.crc32(name.encode()))
                contrast = (input_contrast(parts.right, forget_covariances[name], retain_covariances[name])
                            if component_selection == "contrast" else None)
                alpha = component_weights(parts, component_selection or "full", generator, contrast=contrast)
                applied = compose_update(parts, alpha)
                if component_selection == "uniform_matched":  # the original update itself, scaled
                    applied = float(alpha[0]) * delta.reshape_as(applied).double() if alpha.numel() else 0.0 * applied
                info["components"] = component_report(parts, alpha, delta, applied, restriction_source[name],
                                                      grad_retain[name], keep_arrays=component_diagnostics)
                if component_selection is not None:
                    delta = applied.reshape_as(delta).to(delta.dtype)
                del parts, applied
            if input_filters is not None and basis.active:
                filtered = delta @ input_filters[name].to(delta.device)
                info["retain_filter_kept"] = float(filtered.norm() / (delta.norm() + COS_EPS))
                delta = filtered
            info.update({"rank": basis.rank, "total_rank": basis.total_rank, "evr": basis.evr, "active": basis.active,
                         "reason": basis.reason, "applied_norm": float(delta.norm())})
            if basis.rank_in is not None:  # own-spectrum ranks: rank is the output side's
                info["rank_in"] = basis.rank_in
            if basis.active:  # ||P(W)||: with every row at the floor s, applied_norm = (1 - s) ||P(W)||
                info["projected_weight_norm"] = float(project(w32, basis.u, basis.vh).norm())
            if output_diagnostics and basis.active:
                info["output_energy"] = output_energy_report(w32, delta, forget_covariances[name],
                                                             retain_covariances[name])
                if name.rsplit(".", 1)[-1] in ("down_proj", "o_proj"):  # the writes into the residual stream
                    info["output_contrast"] = {
                        f"{rho:g}": output_contrast_simulation(w32, forget_covariances[name], retain_covariances[name],
                                                               rho, float(delta.norm()))
                        for rho in OUTPUT_CONTRAST_RHOS}
            if family_diagnostics and basis.active and name.rsplit(".", 1)[-1] in ("down_proj", "o_proj"):
                applied = output_energy_report(w32, delta, forget_covariances[name], retain_covariances[name])
                info["families"] = {"recast_forget_share": applied["forget_removed_share"],
                                    "recast_retain_share": applied["retain_removed_share"],
                                    **family_report(w32, forget_covariances[name], retain_covariances[name],
                                                    restriction_source[name], grad_retain[name],
                                                    applied["forget_removed_share"], rhos=FAMILY_RHOS)}
            if forget_output_target is not None and basis.active:
                info["forget_output_change"] = forget_output_change(delta, w32, forget_covariances[name])
        else:
            delta = proposal
            info["applied_norm"] = float(delta.norm())
        if global_family is not None:  # the ReCAST update only sets the budget; the family's update follows the loop
            gen = (torch.Generator().manual_seed(int(component_seed) * 1000003 + zlib.crc32(name.encode()))
                   if global_family_order == "random" else None)
            family_parts[name] = target_directions(w32, forget_covariances[name], retain_covariances[name],
                                                   restriction_source[name], grad_retain[name], global_family,
                                                   global_family_rho, keys_generator=gen, key_alpha=global_family_alpha,
                                                   s_gamma=global_family_s_gamma, s_min=global_family_floor,
                                                   s_rule=global_family_s_rule,
                                                   keep_h=global_family_s_threshold_gamma is not None)
            info["global_family"] = {"recast_forget_energy": output_energy(delta, forget_covariances[name]),
                                     "recast_retain_energy": output_energy(delta, retain_covariances[name]),
                                     "recast_update_norm": float(delta.norm())}
            report[name] = info
            continue
        w.copy_((w32 + delta).to(w.dtype))
        info["weight_rel_change"] = float((w.detach().float() - w32).norm() / (w32.norm() + COS_EPS))
        if restricted:
            restrictions[name] = Restriction(baseline=w.detach().clone(), basis=basis)
        report[name] = info
    if global_family is not None:
        recast = [report[n]["global_family"] for n in family_parts]
        if global_family_budget_unit == "share":
            budget = sum(report[n]["global_family"]["recast_forget_energy"] / family_parts[n]["energy_f"]
                         for n in family_parts if family_parts[n]["energy_f"] > 0)
        else:
            budget = sum(r["recast_forget_energy"] for r in recast)
        per_direction = global_family_s_gamma is not None or global_family_s_rule is not None
        sel_floor = 0.0 if per_direction else global_family_floor  # folded coefficients: beta 1
        if global_family_s_threshold_gamma is not None:
            selection = calibrated_selection(family_parts, budget, global_family_s_threshold_gamma,
                                             s_min=global_family_floor, unit=global_family_budget_unit)
        else:
            selection = global_selection(family_parts, sel_floor, budget, unit=global_family_budget_unit)
        selection["global"].update({
            "family": global_family, "rho": global_family_rho, "order": global_family_order, "alpha": global_family_alpha,
            "s_gamma": global_family_s_gamma, "s_rule": global_family_s_rule, "s_min": global_family_floor,
            "s_threshold_gamma": global_family_s_threshold_gamma,
            "recast_retain_energy": sum(r["recast_retain_energy"] for r in recast),
            "recast_update_norm": math.sqrt(sum(r["recast_update_norm"] ** 2 for r in recast))})
        for name, module in targets.items():
            w = module.weight
            w32 = w.detach().float().clone()
            info, chosen = report[name], selection["per_target"][name]
            part = family_parts.pop(name)
            if global_family_s_threshold_gamma is not None:
                delta = delta_from_coef(part, w32, chosen.pop("coef")).to(w32.dtype)
            else:
                delta = global_delta(part, w32, chosen["directions"], 1.0 - sel_floor, chosen["beta_last"]).to(w32.dtype)
            del part
            info["global_family"].update(chosen)
            info["global_family"]["global"] = selection["global"]
            info["applied_norm"] = float(delta.norm())
            w.copy_((w32 + delta).to(w.dtype))
            info["weight_rel_change"] = float((w.detach().float() - w32).norm() / (w32.norm() + COS_EPS))
            restrictions[name] = Restriction(baseline=w.detach().clone(), basis=bases[name])
    return restrictions, report


# ---------------- SEMU adapter backend ----------------

class SemuAdapterLinear(nn.Module):
    """``W = a + u @ weight @ vh`` (Classification ``CustomLinear`` without gates); only ``weight`` trains.
    ``u=None`` (extension, one-sided restriction): ``W = a + weight @ vh``, the core spanning every output."""

    def __init__(self, in_features: int, out_features: int, core: torch.Tensor, anchor: torch.Tensor,
                 u: Optional[torch.Tensor], vh: torch.Tensor):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(core)
        self.register_buffer("a", anchor)
        self.register_buffer("u", u)
        self.register_buffer("vh", vh)

    def effective_weight(self) -> torch.Tensor:
        if self.u is None:
            return self.a + self.weight @ self.vh
        return self.a + torch.einsum("ij,jk,kl->il", self.u, self.weight, self.vh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight())


def adapter_core_and_anchor(baseline: torch.Tensor, basis: Basis) -> Tuple[torch.Tensor, torch.Tensor]:
    """``A0 = U^T W* Vh^T`` and ``anchor = W* - U A0 Vh`` in fp32, as ``_adapter_core_and_anchor``."""
    m = baseline.float()
    vh = basis.vh.float()
    if basis.u is None:  # one-sided restriction
        core = m @ vh.transpose(0, 1)
        return core, m - core @ vh
    u = basis.u.float()
    core = u.transpose(0, 1) @ m @ vh.transpose(0, 1)
    anchor = m - u @ core @ vh
    return core, anchor


@torch.no_grad()
def apply_semu_adapters(model: nn.Module, restrictions: Dict[str, Restriction]) -> dict:
    """Replace every active restricted target by a SemuAdapterLinear starting at W*."""
    effective, trainable, replaced, max_rel_err = 0, 0, 0, 0.0
    modules = dict(model.named_modules())
    for name, r in restrictions.items():
        if not r.basis.active:
            continue
        linear = modules[name]
        dtype, device = linear.weight.dtype, linear.weight.device
        core, anchor = adapter_core_and_anchor(r.baseline, r.basis)
        adapter = SemuAdapterLinear(
            linear.in_features, linear.out_features,
            core.to(device=device, dtype=dtype), anchor.to(device=device, dtype=dtype),
            None if r.basis.u is None else r.basis.u.to(device=device, dtype=dtype),
            r.basis.vh.to(device=device, dtype=dtype),
        )
        adapter.train(linear.training)
        parent_name, _, child = name.rpartition(".")
        setattr(modules[parent_name], child, adapter)
        err = (adapter.effective_weight().float() - r.baseline.float()).norm() / (r.baseline.float().norm() + COS_EPS)
        max_rel_err = max(max_rel_err, float(err))
        effective += r.baseline.numel()
        trainable += adapter.weight.numel()
        replaced += 1
    return {"adapters": replaced, "effective_parameters": effective, "trainable_core_parameters": trainable,
            "trainable_fraction": trainable / effective if effective else 0.0,
            "max_rel_error_initial_weight": max_rel_err}


FREE_GROUPS = {"norms": (), "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
               "mlp": ("gate_proj", "up_proj", "down_proj")}


def free_parameters(model: nn.Module, spec: str, prefix: str = "model.layers.") -> Dict[str, Dict[str, nn.Parameter]]:
    """Parameters without a SEMU adapter that phase two trains freely, by group.

    A deviation from Classification, whose ``set_requires_grad`` freezes everything outside the
    adapters. ``spec`` is ``none``, ``all`` or a comma list of ``norms`` (every LayerNorm/RMSNorm),
    ``attention`` and ``mlp`` (block projections still plain nn.Linear). A named group that selects
    nothing is an error; ``all`` only needs one group to select something.
    """
    if spec == "none":
        return {}
    groups = list(FREE_GROUPS) if spec == "all" else spec.split(",")
    unknown = [g for g in groups if g not in FREE_GROUPS]
    if unknown:
        raise ValueError(f"unknown free groups {unknown}; choose from {sorted(FREE_GROUPS)}")
    selected = {g: {} for g in groups}
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm) or type(m).__name__.endswith("RMSNorm"):
            group = "norms"
        elif type(m) is nn.Linear and name.startswith(prefix):
            group = next((g for g, suffixes in FREE_GROUPS.items() if name.rsplit(".", 1)[-1] in suffixes), None)
        else:
            continue
        if group in selected:
            selected[group].update({f"{name}.{n}": p for n, p in m.named_parameters(recurse=False)})
    empty = [g for g in groups if not selected[g]]
    if (spec != "all" and empty) or not any(selected.values()):
        raise ValueError(f"free group(s) {empty} select no parameter")
    return {g: ps for g, ps in selected.items() if ps}


@torch.no_grad()
def drift_report(model: nn.Module, reference: Dict[str, torch.Tensor],
                 group_suffixes: Iterable[str]) -> Dict[str, dict]:
    """Per projection type: norm ratio ||W|| / ||W_ref|| and cosine(W, W_ref), aggregated over layers.

    ``W`` is the current effective weight (``a + U A Vh`` for adapters); ``reference`` maps
    target names to reference weights (for example the original or the post-warm-start ones).
    """
    modules = dict(model.named_modules())
    sums: Dict[str, list] = {}
    for name, ref in reference.items():
        group = next((g for g in group_suffixes if name.endswith(g)), None)
        if group is None:
            continue
        m = modules[name]
        w = (m.effective_weight() if isinstance(m, SemuAdapterLinear) else m.weight).float()
        r = ref.to(device=w.device, dtype=torch.float32)
        acc = sums.setdefault(group, [0.0, 0.0, 0.0])
        acc[0] += float((w * r).sum())
        acc[1] += float(w.square().sum())
        acc[2] += float(r.square().sum())
    return {g: {"norm_ratio": (ww / rr) ** 0.5 if rr > 0 else 0.0,
                "cosine": dot / ((ww * rr) ** 0.5) if ww > 0 and rr > 0 else 0.0}
            for g, (dot, ww, rr) in sums.items()}


# ---------------- phase-two learning-rate schedule ----------------

def parse_points(spec: str) -> list:
    """``"0:0,1:3e-5,5:1e-5,10:1e-5"`` -> [(0.0, 0.0), (1.0, 3e-5), ...], non-decreasing in x.

    An x given twice is a step (``5:3e-5,5:1e-5`` drops to 1e-5 at 5); no x may appear three times,
    and the last x must exceed the first.
    """
    points = [tuple(float(v) for v in item.split(":")) for item in spec.split(",") if item.strip()]
    if len(points) < 2 or any(len(p) != 2 for p in points):
        raise ValueError(f"need at least two x:y points, got {spec!r}")
    if any(b[0] < a[0] for a, b in zip(points, points[1:])) or points[-1][0] <= points[0][0]:
        raise ValueError(f"points must be non-decreasing in x and span a positive range: {spec!r}")
    if any(a[0] == c[0] for a, c in zip(points, points[2:])):
        raise ValueError(f"an x may appear at most twice (one step): {spec!r}")
    return points


def piecewise_linear(points: list, x: float) -> float:
    """Linear interpolation through the points; constant before the first and after the last. At a
    step (repeated x) the later value holds from that x on."""
    if x < points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x < x1:  # zero-width step segments never match: x >= x0 == x1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


# ---------------- phase-two controls ----------------

@torch.no_grad()
def snapshot_trainable(model: nn.Module) -> Dict[str, torch.Tensor]:
    """CPU copy of every trainable parameter, to measure what phase two actually changed."""
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def frozen_checksums(model: nn.Module) -> Dict[str, Tuple[float, float]]:
    """(sum, sum of squares) in float64 of every frozen parameter and every buffer."""
    tensors = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    tensors += list(model.named_buffers())
    return {n: (float(t.double().sum()), float(t.double().square().sum())) for n, t in tensors}


@torch.no_grad()
def change_report(model: nn.Module, snapshot: Dict[str, torch.Tensor],
                  checksums: Dict[str, Tuple[float, float]]) -> dict:
    """Fraction of trainable entries that moved, their relative change, and frozen tensors that moved."""
    params = dict(model.named_parameters())
    changed, total, diff_sq, base_sq = 0, 0, 0.0, 0.0
    for n, before in snapshot.items():
        after = params[n].detach().to("cpu")
        d = after.float() - before.float()
        changed += int((after != before).sum())
        total += before.numel()
        diff_sq += float(d.square().sum())
        base_sq += float(before.float().square().sum())
    now = frozen_checksums(model)
    moved = [n for n, v in checksums.items() if now.get(n) != v]
    return {"trainable_entries": total, "trainable_entries_changed_fraction": changed / total if total else 0.0,
            "trainable_rel_change": (diff_sq / base_sq) ** 0.5 if base_sq > 0 else 0.0,
            "frozen_tensors_checked": len(checksums), "frozen_tensors_moved": moved[:10],
            "frozen_tensors_moved_count": len(moved)}
