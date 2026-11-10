"""SVD of the weight core against SVD of the update core (Stage 3): the basis for a future directional contraction.

Notation (kept apart from the update decomposition of ``recast_core.decompose_update``):

- weight core ``K = U^T W Vh^T`` (r x r) of the pretrained weights ``W`` inside the ReCAST subspace, thin SVD
  ``K = P_K diag(kappa) Q_K^T``; its components ``E_j = kappa_j u_j v_j^T`` with ``u_j = U p_j``, ``v_j = Vh^T q_j``
  decompose the projected weights ``P(W) = sum_j E_j``;
- update core ``B = U^T X Vh^T`` of the proposal ``X = (S - I) W``, thin SVD ``B = P_B diag(sigma) Q_B^T``; its
  components ``D_j`` (``decompose_update``) decompose the update ``Delta W = P(X) = sum_j D_j``;
- ``alpha_j`` would scale the update components ``D_j``; ``beta_j`` would contract the weight components ``E_j``:
  ``W(beta) = W - sum_j beta_j E_j``, ``0 <= beta_j <= 1``.

Facts the tests check:

1. ``||W(beta)||^2 = ||W||^2 - sum_j (2 beta_j - beta_j^2) kappa_j^2``: the ``E_j`` are mutually Frobenius-orthogonal
   (orthonormal ``p_j``, ``q_j``), and ``<W, E_j> = <P(W), E_j> = kappa_j^2`` because ``W - P(W)`` is orthogonal to the
   range of P, which holds every ``E_j``; expanding ``||W - sum beta_j E_j||^2`` gives the identity.
2. Uniform ``S = s I``: ``B = (s - 1) K``, so both SVDs share their singular subspaces (up to signs and rotations
   within repeated singular values) and ``sigma = (1 - s) kappa``; ``beta_j = 1 - s`` rebuilds the restricted ReCAST
   update exactly, ``W(beta) = W + P((s - 1) W)``.
3. Non-uniform S: ``B = U^T diag(s - 1) W Vh^T`` and ``K`` need not share singular subspaces. In the weight-core bases
   the update has the full coefficient matrix ``C = P_K^T B Q_K`` (``Delta W = U P_K C Q_K^T Vh``); a diagonal
   ``beta`` represents it only when C is diagonal, with ``beta_j = -C_jj / kappa_j``; the off-diagonal energy
   ``||C - diag(C)||^2 / ||C||^2`` is the relative squared residual of the best diagonal representation (Frobenius).

Subspaces are compared through principal angles, never single singular vectors (repeated or close singular values
make individual vectors arbitrary). Everything runs on the r x r cores in float64.
"""

from typing import Callable, Dict, Optional

import torch


def cores(weight: torch.Tensor, factors: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor) -> dict:
    """``K = U^T W Vh^T`` and ``B = U^T (S - I) W Vh^T`` (``U = I`` when ``u`` is None, then out x r)."""
    w = weight.reshape(weight.shape[0], -1).double()
    w_v = w @ vh.double().transpose(0, 1)
    b_v = (factors.double() - 1.0).unsqueeze(1) * w_v
    if u is None:
        return {"K": w_v, "B": b_v}
    ut = u.double().transpose(0, 1)
    return {"K": ut @ w_v, "B": ut @ b_v}


def principal_angle_cosines(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosines of the principal angles between the column spans of two orthonormal bases (descending)."""
    return torch.linalg.svdvals(a.double().transpose(0, 1) @ b.double()).clamp(max=1.0)


def effective_rank(values: torch.Tensor) -> float:
    """``exp`` of the entropy of the normalised squared singular values (1 to len(values))."""
    e = values.double().square()
    total = float(e.sum())
    if total <= 0:
        return 0.0
    p = e[e > 0] / total
    return float(torch.exp(-(p * p.log()).sum()))


def weight_components(weight: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor) -> dict:
    """The weight-core components: ``kappa`` and the full-space directions ``u_j = U p_j`` (columns of ``left``) and
    ``v_j^T = q_j^T Vh`` (rows of ``right``), so ``E_j = kappa_j left[:, j] right[j]``."""
    k = cores(weight, torch.ones(weight.shape[0], dtype=torch.float64, device=weight.device), u, vh)["K"]
    p, kappa, qh = torch.linalg.svd(k, full_matrices=False)
    left = p if u is None else u.double() @ p
    return {"kappa": kappa, "left": left, "right": qh @ vh.double(), "p": p, "qh": qh}


def contract(weight: torch.Tensor, comps: dict, beta: torch.Tensor) -> torch.Tensor:
    """``W(beta) = W - sum_j beta_j E_j`` (float64)."""
    return weight.reshape(weight.shape[0], -1).double() - (comps["left"] * (beta.double() * comps["kappa"])) @ comps["right"]


def compare_cores(weight: torch.Tensor, factors: torch.Tensor, u: Optional[torch.Tensor], vh: torch.Tensor,
                  ks=(1, 4, 16, 64), grads: Optional[Dict[str, torch.Tensor]] = None,
                  contrast_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, contrast_k: int = 16) -> dict:
    """Weight core against update core of one target for the factors S: spectra (normalised), effective ranks,
    principal-angle cosines of the leading left and right subspaces for each k (mean and minimum), the relative
    rank-k reconstruction error of each core, the representation of B in the weight-core bases (diagonal beta and its
    off-diagonal residual) and, for each gradient, the alignment ``<G, E_j> / kappa_j = p_j^T (U^T G Vh^T) q_j`` of the
    leading weight components and the same for the update components. ``contrast_fn`` (e.g. the forget / retain input
    energy ratio ``recast_core.input_contrast``) scores the input directions ``v_j`` of the leading ``contrast_k``
    components of each core."""
    c = cores(weight, factors, u, vh)
    pk, kappa, qkh = torch.linalg.svd(c["K"], full_matrices=False)
    pb, sigma, qbh = torch.linalg.svd(c["B"], full_matrices=False)
    r = int(kappa.numel())
    out = {"rank": r, "kappa_effective_rank": effective_rank(kappa), "sigma_effective_rank": effective_rank(sigma),
           "spectrum_l2_distance": float((kappa / kappa.norm() - sigma / sigma.norm()).norm()) if sigma.norm() > 0 else None,
           "angles": {}, "reconstruction": {}}
    for k in sorted({min(k, r) for k in ks}):
        left = principal_angle_cosines(pk[:, :k], pb[:, :k])
        right = principal_angle_cosines(qkh[:k].transpose(0, 1), qbh[:k].transpose(0, 1))
        out["angles"][str(k)] = {"left_mean_cos": float(left.mean()), "left_min_cos": float(left.min()),
                                 "right_mean_cos": float(right.mean()), "right_min_cos": float(right.min())}
        out["reconstruction"][str(k)] = {
            "weight_core": float(kappa[k:].square().sum() / kappa.square().sum()) ** 0.5,
            "update_core": float(sigma[k:].square().sum() / sigma.square().sum()) ** 0.5 if sigma.norm() > 0 else None}
    coeff = pk.transpose(0, 1) @ c["B"] @ qkh.transpose(0, 1)          # B in the weight-core bases
    diag = torch.diagonal(coeff)
    total = float(coeff.square().sum())
    beta = torch.where(kappa > 0, -diag / torch.where(kappa > 0, kappa, torch.ones_like(kappa)), torch.zeros_like(kappa))
    out["beta"] = {"offdiag_energy_fraction": (total - float(diag.square().sum())) / total if total > 0 else 0.0,
                   "mean": float(beta.mean()), "min": float(beta.min()), "max": float(beta.max()),
                   "outside_unit_interval": float(((beta < 0) | (beta > 1)).double().mean())}
    if grads:
        out["alignment"] = {}
        ut = None if u is None else u.double().transpose(0, 1)
        for name, g in grads.items():
            g_v = g.reshape(g.shape[0], -1).double() @ vh.double().transpose(0, 1)
            kg = g_v if ut is None else ut @ g_v
            a_k = ((pk.transpose(0, 1) @ kg) * qkh).sum(1)               # p_j^T K_G q_j, weight components
            a_b = ((pb.transpose(0, 1) @ kg) * qbh).sum(1)               # the same for the update components
            out["alignment"][name] = {"weight_leading": [float(x) for x in a_k[:8]],
                                      "update_leading": [float(x) for x in a_b[:8]],
                                      "weight_abs_mean": float(a_k.abs().mean()), "update_abs_mean": float(a_b.abs().mean())}
    if contrast_fn is not None:
        k = min(contrast_k, r)
        v = vh.double()
        out["input_contrast"] = {"k": k, "weight_mean": float(contrast_fn(qkh[:k] @ v).mean()),
                                 "update_mean": float(contrast_fn(qbh[:k] @ v).mean())}
    return out
