"""Contraction of chosen weight-core components at a fixed displacement (extension).

With the weight components ``E_j = kappa_j u_j v_j^T`` of ``recast_bases.weight_components`` (``P(W) = sum_j E_j``),
``W(beta) = W - sum_j beta_j E_j`` moves the weights by ``||W(beta) - W||^2 = sum_j beta_j^2 kappa_j^2``, because the
``E_j`` are mutually Frobenius-orthogonal. A budget ``b`` asks for ``||W(beta) - W|| = b ||P(W)||``. ReCAST with every
row at the floor s contracts every component by ``beta_j = 1 - s`` (Stage 3), so ``b = 1 - s`` is its displacement and
the floor keeps setting the strength of a run.

``contraction_beta`` spends the budget on components taken in a fixed order, ``beta_j = strength`` each, the last one
contracted partly: ``top`` takes the largest kappa first, ``random`` a permutation drawn by the generator and
``forget_retain`` (user, 2026-10-28) the largest ``forget_retain_score`` first, the first-order forget-loss increase
minus the retain-loss increase per unit of moved energy, and ``contrast`` (user, 2026-10-29) the largest input contrast
of the component's input direction v_j first (``recast_core.input_contrast``; with a one-sided basis the components
are those of ``W Vh^T Vh``). Components without energy carry no displacement and are never
chosen. A ``residual`` (user, 2026-10-28) leaves every contracted component the singular value ``residual`` instead of
0. ``component_profile`` lists every component's kappa, input contrast and gradient alignments.
"""

import math
from typing import Dict, Optional

import torch

MODES = ("top", "random", "forget_retain", "contrast")
SCORED = ("forget_retain", "contrast")  # the modes ordered by a score per component


def alignments(comps: dict, g: torch.Tensor) -> torch.Tensor:
    """``u_j^T G v_j = <G, E_j> / kappa_j`` per component (float64, on the device of the components)."""
    left, right = comps["left"].double(), comps["right"].double()
    g2 = g.reshape(g.shape[0], -1).to(left.device).double()
    return ((left.transpose(0, 1) @ g2) * right).sum(1)


def forget_retain_score(comps: dict, g_forget: torch.Tensor, g_retain: torch.Tensor) -> torch.Tensor:
    """``-(a_forget,j - a_retain,j) / kappa_j``: removing ``E_j`` changes a loss by ``-kappa_j a_j`` to first order and
    moves ``kappa_j^2`` of the budget, so this is the forget-loss increase minus the retain-loss increase per unit of
    moved energy (0 for a component without energy)."""
    kappa = comps["kappa"].double()
    score = -(alignments(comps, g_forget) - alignments(comps, g_retain)) / kappa.clamp_min(1e-300)
    return torch.where(kappa > 0, score, torch.zeros_like(score))


def _caps(k2: torch.Tensor, strength: float, residual: float) -> torch.Tensor:
    """The largest beta of each component: ``strength``, lowered so that ``(1 - beta) kappa_j >= residual``; 0 for a
    component without energy or with ``kappa_j <= residual``."""
    kappa = k2.sqrt()
    caps = torch.minimum(torch.full_like(kappa, strength), 1.0 - residual / kappa.clamp_min(1e-300))
    return torch.where(k2 > 0, caps.clamp_min(0.0), torch.zeros_like(kappa))


def contraction_beta(kappa: torch.Tensor, mode: str, budget: float, strength: float = 1.0,
                     generator: Optional[torch.Generator] = None, residual: float = 0.0,
                     score: Optional[torch.Tensor] = None) -> torch.Tensor:
    """beta per component (float64, on the device of ``kappa``) with ``sum_j beta_j^2 kappa_j^2 = budget^2 sum_j
    kappa_j^2``: each component in the order of ``mode`` at its cap until the budget is spent, the next one at
    ``sqrt(rest / kappa_j^2)`` below its cap, every other component 0. The cap is ``strength``, lowered so that a
    contracted component keeps at least the singular value ``residual`` (``beta_j <= 1 - residual / kappa_j``). Needs
    ``0 < strength <= 1``, ``0 <= budget <= strength`` and ``residual >= 0``; at ``budget = strength`` without a
    residual every component is contracted by ``strength``. A budget the caps cannot reach (singular values near the
    residual, e.g. a degenerate rank-1 target) contracts every component to its cap and falls short. ``forget_retain``
    and ``contrast`` take the components by ``score`` (``forget_retain_score`` or the input contrast), highest first."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if (mode in SCORED) != (score is not None):
        raise ValueError(f"the modes {SCORED}, and only they, need a score per component")
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must lie in (0, 1], got {strength}")
    if not 0.0 <= budget <= strength:
        raise ValueError(f"budget must lie in [0, strength] = [0, {strength}], got {budget}")
    if residual < 0.0:
        raise ValueError(f"residual must be >= 0, got {residual}")
    if mode == "random" and generator is None:
        raise ValueError("random contraction needs a generator")
    k2 = kappa.detach().double().cpu().square()
    n = k2.numel()
    beta = torch.zeros(n, dtype=torch.float64)
    total = float(k2.sum())
    if n == 0 or total <= 0.0 or budget == 0.0:
        return beta.to(kappa.device)
    if budget == strength and residual == 0.0:  # every component with energy, exactly (no rounding in the rest)
        return torch.where(k2 > 0, torch.full_like(beta, strength), beta).to(kappa.device)
    caps = _caps(k2, strength, residual)
    rest = budget ** 2 * total
    if rest >= float((caps.square() * k2).sum()):  # out of reach (e.g. kappa near the residual): every cap, a shortfall
        return caps.to(kappa.device)
    if mode == "top":
        order = torch.argsort(k2, descending=True, stable=True)
    elif mode == "random":
        order = torch.randperm(n, generator=generator)
    else:
        order = torch.argsort(score.detach().double().cpu(), descending=True, stable=True)
    for j in order.tolist():
        if caps[j] <= 0.0:
            continue
        step = float(caps[j]) ** 2 * float(k2[j])
        if step < rest:
            beta[j] = caps[j]
            rest -= step
        else:
            beta[j] = min(float(caps[j]), math.sqrt(rest / float(k2[j])))
            break
    return beta.to(kappa.device)


def contraction_delta(comps: dict, beta: torch.Tensor) -> torch.Tensor:
    """``W(beta) - W = -sum_j beta_j E_j`` (float64) for the components of ``recast_bases.weight_components``."""
    kappa = comps["kappa"].double()
    return -(comps["left"].double() * (beta.to(kappa.device).double() * kappa)) @ comps["right"].double()


def component_profile(comps: dict, grads: Dict[str, torch.Tensor], contrast: torch.Tensor) -> dict:
    """Per weight-core component j (the order of ``recast_bases.weight_components``, kappa descending): kappa_j, the
    input contrast of v_j and, for each gradient G, the alignment ``u_j^T G v_j = <G, E_j> / kappa_j`` (removing
    ``beta_j E_j`` changes that loss by ``-beta_j kappa_j u_j^T G v_j`` to first order)."""
    out = {"kappa": [float(x) for x in comps["kappa"]], "input_contrast": [float(x) for x in contrast], "alignment": {}}
    for name, g in grads.items():
        out["alignment"][name] = [float(x) for x in alignments(comps, g)]
    return out


def contraction_report(kappa: torch.Tensor, beta: torch.Tensor, strength: float,
                       contrast: Optional[torch.Tensor] = None, residual: float = 0.0) -> dict:
    """What a contraction touches in one target: the number of components, contracted at all and fully (at the cap of
    ``contraction_beta``); the share of
    ``||P(W)||^2`` it moves, ``sum beta_j^2 kappa_j^2 / sum kappa_j^2`` (the budget squared); where the contracted
    components sit in the kappa order (normalised rank, 0 the largest kappa, 1 the smallest), plain and weighted by
    the moved energy; and, given each component's input contrast (``recast_core.input_contrast`` of the rows
    ``v_j^T``), the mean contrast of the moved energy against that of all of ``P(W)``."""
    k2 = kappa.detach().double().cpu().square()
    b = beta.detach().double().cpu()
    n = k2.numel()
    total = float(k2.sum())
    moved = b.square() * k2
    moved_total = float(moved.sum())
    rank = torch.empty(n, dtype=torch.float64)
    rank[torch.argsort(k2, descending=True, stable=True)] = torch.arange(n, dtype=torch.float64) / max(n - 1, 1)
    touched = b > 0
    caps = _caps(k2, strength, residual)
    out = {"components": n, "contracted": int(touched.sum()), "full": int((touched & (b >= caps - 1e-12)).sum()),
           "energy_fraction": moved_total / total if total > 0 else 0.0,
           "kappa_rank_mean": float(rank[touched].mean()) if bool(touched.any()) else None,
           "kappa_rank_weighted": float((rank * moved).sum()) / moved_total if moved_total > 0 else None}
    if contrast is not None:
        c = contrast.detach().double().cpu()
        out["input_contrast"] = {"moved": float((c * moved).sum()) / moved_total if moved_total > 0 else None,
                                 "all": float((c * k2).sum()) / total if total > 0 else None}
    return out
