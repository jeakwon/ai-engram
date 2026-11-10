"""Shrink scores of the ReCAST row factors (Stage 2): the original cosine and projection-consistent alternatives.

The restricted warm start applies ``Delta W(S) = P((S - I) W)`` with ``P(X) = U U^T X Vh^T Vh`` and
``S = diag(s_i)``, ``s_i = max(score_i, floor)``. The original score is ``cos(W_i, -G_r,i)`` in the full space
(``recast_core.row_alignment``). With the cores ``K_X = U^T X Vh^T`` (r x r), ``L_X = U K_X`` (out x r) and ``Vh``
orthonormal, every alternative needs only out x r and r x r matrices, never a projected out x in matrix:

- ``projected_row``: ``cos(P(W)_i, -P(G_r)_i) = -<(L_W)_i, (L_G)_i> / (||(L_W)_i|| ||(L_G)_i||)``;
- ``projected_sensitivity``: shrinking row i moves the weights along its feasible update ``T_i = P(e_i W_i)``, rank one
  ``(U U^T e_i)(W_i Vh^T Vh)``; the first-order retain-loss change per unit of ``s_i`` is
  ``a_i = <G_r, T_i> = <P(G_r), e_i W_i> = <(L_G)_i, (W Vh^T)_i>`` (P is self-adjoint), so the score is
  ``-a_i / (||(L_G)_i|| ||(W Vh^T)_i||) = cos((W Vh^T Vh)_i, -P(G_r)_i)``, in [-1, 1] by Cauchy-Schwarz because
  ``P(G_r)_i`` lies in the row space of ``Vh``. As for the original cosine, a positive score marks a row whose shrink
  raises the retain loss (``s_i - 1 < 0`` times ``a_i < 0``), and the row keeps more of itself;
- ``global_projected``: one cosine per matrix, ``-<K_W, K_G> / (||K_W|| ||K_G||) = cos(P(W), -P(G_r))``; it measures the
  alignment inside the allowed subspace and gives one common factor, not row factors;
- ``core_row`` (diagnostic only): ``cos((K_W)_j, -(K_G)_j)`` per output direction ``U_j`` of the reduced coordinates; it
  scores directions, not neurons, so it is no rule for S.

``projected_row`` and ``projected_sensitivity`` differ row by row because ``U U^T`` is not diagonal (P mixes rows),
while their numerators have the same sum ``<P(W), P(G_r)> = <W, P(G_r)>``; with ``u=None`` (output side free) they
coincide. A zero denominator gives the score 0 and is counted in ``zero_norm_rows``. Everything is float64.
"""

from typing import Dict, Optional, Sequence

import torch

RULES = ("original", "projected_row", "projected_sensitivity", "global_projected")


def _cores(weight: torch.Tensor, grad: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor) -> dict:
    w, g, v = weight.double(), grad.double(), vh.double()
    w_v, g_v = w @ v.transpose(0, 1), g @ v.transpose(0, 1)          # (out, r): rows of W Vh^T, G Vh^T
    if u is None:
        return {"w_v": w_v, "l_w": w_v, "l_g": g_v, "k_w": w_v, "k_g": g_v}
    ud = u.double()
    k_w, k_g = ud.transpose(0, 1) @ w_v, ud.transpose(0, 1) @ g_v      # (r, r)
    return {"w_v": w_v, "l_w": ud @ k_w, "l_g": ud @ k_g, "k_w": k_w, "k_g": k_g}


def _safe_cosine(dot: torch.Tensor, n1: torch.Tensor, n2: torch.Tensor):
    denom = n1 * n2
    ok = denom > 0
    return torch.where(ok, dot / torch.where(ok, denom, torch.ones_like(denom)), torch.zeros_like(dot)), int((~ok).sum())


def shrink_scores(weight: torch.Tensor, grad_retain: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor,
                  eps: float = 1.0e-12) -> dict:
    """Every score of one target (module docstring); ``original`` uses the baseline convention ``dot / (n n + eps)``,
    the others the explicit zero-norm rule. ``sensitivity`` is ``a_i`` itself."""
    w, g = weight.reshape(weight.shape[0], -1).double(), grad_retain.reshape(grad_retain.shape[0], -1).double()
    c = _cores(w, g, u, vh)
    zero = {}
    original = (w * (-g)).sum(1) / (w.norm(dim=1) * g.norm(dim=1) + eps)
    projected_row, zero["projected_row"] = _safe_cosine(-(c["l_w"] * c["l_g"]).sum(1), c["l_w"].norm(dim=1),
                                                        c["l_g"].norm(dim=1))
    a = (c["l_g"] * c["w_v"]).sum(1)
    projected_sensitivity, zero["projected_sensitivity"] = _safe_cosine(-a, c["l_g"].norm(dim=1), c["w_v"].norm(dim=1))
    global_projected, zero["global_projected"] = _safe_cosine(-(c["k_w"] * c["k_g"]).sum().reshape(1),
                                                              c["k_w"].norm().reshape(1), c["k_g"].norm().reshape(1))
    core_row, zero["core_row"] = _safe_cosine(-(c["k_w"] * c["k_g"]).sum(1), c["k_w"].norm(dim=1), c["k_g"].norm(dim=1))
    return {"original": original, "projected_row": projected_row, "projected_sensitivity": projected_sensitivity,
            "sensitivity": a, "global_projected": float(global_projected[0]), "core_row": core_row,
            "zero_norm_rows": zero}


def row_factors(scores: dict, rule: str, floor: float) -> torch.Tensor:
    """``s_i = max(score_i, floor)``; ``global_projected`` gives every row ``max(c_global, floor)``."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if rule == "global_projected":
        return torch.full_like(scores["original"], max(scores["global_projected"], float(floor)))
    return scores[rule].clamp(min=float(floor))


def update_stats(weight: torch.Tensor, factors: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor,
                 grads: Dict[str, torch.Tensor]) -> dict:
    """Of ``Delta W = P((S - I) W)``: its norm ``||U^T X Vh^T||``, the kept energy ``||P(X)||^2 / ||X||^2`` and the
    first-order loss change ``<G, Delta W> = <U^T G Vh^T, U^T X Vh^T>`` for each gradient; cores only."""
    w = weight.reshape(weight.shape[0], -1).double()
    v = vh.double()
    d = (factors.double() - 1.0).unsqueeze(1)
    x_norm = float((d * w).norm())
    x_v = d * (w @ v.transpose(0, 1))                                    # rows of X Vh^T, (out, r)
    k_x = x_v if u is None else u.double().transpose(0, 1) @ x_v
    out = {"update_norm": float(k_x.norm()), "proposal_norm": x_norm,
           "kept_energy_fraction": float(k_x.square().sum()) / x_norm ** 2 if x_norm > 0 else 0.0}
    for name, grad in grads.items():
        g_v = grad.reshape(grad.shape[0], -1).double() @ v.transpose(0, 1)
        k_g = g_v if u is None else u.double().transpose(0, 1) @ g_v
        out[f"predicted_{name}_change"] = float((k_g * k_x).sum())
    return out


def pearson(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    denom = float(a.norm() * b.norm())
    return float((a * b).sum()) / denom if denom > 0 else None


def spearman(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """Pearson of the ranks (ties broken by position)."""
    rank = lambda x: torch.argsort(torch.argsort(x)).double()  # noqa: E731
    return pearson(rank(a), rank(b))


def score_summary(score: torch.Tensor, floors: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
                  reference: Optional[torch.Tensor] = None) -> dict:
    """Quantiles, the share above each floor (the rows the floor does not set) and, against a reference score, the
    Pearson and Spearman correlations."""
    x = score.double()
    qs = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    out = {"mean": float(x.mean()), "std": float(x.std()) if x.numel() > 1 else 0.0,
           "quantiles": {f"q{int(round(100 * q))}": float(v)
                         for q, v in zip(qs, torch.quantile(x, torch.tensor(qs, dtype=x.dtype, device=x.device)))},
           "above_floor": {f"{f:g}": float((x > f).double().mean()) for f in floors}}
    if reference is not None:
        out["pearson_vs_original"] = pearson(x, reference)
        out["spearman_vs_original"] = spearman(x, reference)
    return out
