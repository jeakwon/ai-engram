"""Output energies of a weight change and the output-contrast directions of a weight matrix (extension, user
2026-10-28).

For a target W (out x in) and the uncentred input covariances C_f, C_r over answer tokens, ``E_x(M) = tr(M C_x M^T)``
is the mean squared output of M on forget (x = f) or retain (x = r) inputs. ``output_energy_report`` gives, for a
change Delta W, the output ratios ``c_x = sqrt(E_x(W + Delta W) / E_x(W))`` (how much a write into the residual stream
shrinks, the factor an h rescaling would have to match) and the removed shares ``E_x(Delta W) / E_x(W)``.

The output-contrast directions solve ``(A_f / tr A_f) u = lambda (A_r / tr A_r + (rho / out) I) u`` with
``A_x = W C_x W^T`` (the output-side analogue of basis D's input side, ridge rho times the mean normalised retain
output energy) and are orthonormalised in the order of lambda (descending), so the leading k span the subspace of the
k leading generalised eigenvectors, the components ``E_j = u_j u_j^T W`` are Frobenius-orthogonal and sum to W, and
``||E_j||^2 = ||u_j^T W||^2``. ``output_contrast_delta`` removes them in that order (beta 1) until an absolute
displacement ``||Delta W||`` is reached, the last one partly.
"""

import math
from typing import Dict, Tuple

import torch


def output_energy(m: torch.Tensor, cov: torch.Tensor) -> float:
    """``tr(M C M^T)`` in float64."""
    m2 = m.reshape(m.shape[0], -1).double()
    return float(((m2 @ cov.to(m2.device).double()) * m2).sum())


def output_energy_report(weight: torch.Tensor, delta: torch.Tensor, cov_forget: torch.Tensor,
                         cov_retain: torch.Tensor) -> Dict[str, float]:
    """For forget and retain inputs: ``c = sqrt(E(W + Delta) / E(W))`` and the removed share ``E(Delta) / E(W)``."""
    w = weight.reshape(weight.shape[0], -1).double()
    d = delta.reshape(w.shape).to(w.device).double()
    out = {}
    for side, cov in (("forget", cov_forget), ("retain", cov_retain)):
        e_w, e_new, e_d = output_energy(w, cov), output_energy(w + d, cov), output_energy(d, cov)
        out[f"{side}_ratio"] = math.sqrt(e_new / e_w) if e_w > 0 else None
        out[f"{side}_removed_share"] = e_d / e_w if e_w > 0 else None
    return out


def output_contrast_directions(weight: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor,
                               rho: float) -> dict:
    """``u`` (out x out, orthonormal columns in the order of lambda), ``lam`` (generalised eigenvalues, descending) and
    ``energy`` (``||u_j^T W||^2``)."""
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    w = weight.reshape(weight.shape[0], -1).double()
    a_f = w @ cov_forget.to(w.device).double() @ w.transpose(0, 1)
    a_r = w @ cov_retain.to(w.device).double() @ w.transpose(0, 1)
    a_f, a_r = a_f / torch.trace(a_f), a_r / torch.trace(a_r)
    n = w.shape[0]
    eye = torch.eye(n, dtype=w.dtype, device=w.device)
    chol = torch.linalg.cholesky(a_r + (rho / n) * eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False)
    m = linv @ a_f @ linv.transpose(0, 1)
    lam, q = torch.linalg.eigh(0.5 * (m + m.transpose(0, 1)))
    order = torch.argsort(lam, descending=True)
    lam, q = lam[order], q[:, order]
    u, r = torch.linalg.qr(torch.linalg.solve_triangular(chol.transpose(0, 1), q, upper=True))
    u = u * torch.sign(torch.diagonal(r)).where(torch.diagonal(r) != 0, torch.ones_like(lam))
    return {"u": u, "lam": lam, "energy": (u.transpose(0, 1) @ w).square().sum(1)}


def output_contrast_beta(energy: torch.Tensor, displacement: float) -> torch.Tensor:
    """beta 1 on the components in their order until ``sum_j beta_j^2 ||E_j||^2 = displacement^2``, the next one
    partly; every component when the displacement exceeds ``||W||``."""
    e = energy.detach().double().cpu()
    beta = torch.zeros_like(e)
    rest = displacement ** 2
    for j in range(e.numel()):
        if e[j] <= 0.0:
            continue
        if float(e[j]) < rest:
            beta[j], rest = 1.0, rest - float(e[j])
        else:
            beta[j] = math.sqrt(rest / float(e[j]))
            break
    return beta.to(energy.device)


def output_contrast_delta(weight: torch.Tensor, dirs: dict, beta: torch.Tensor) -> torch.Tensor:
    """``-sum_j beta_j u_j u_j^T W`` (float64)."""
    w = weight.reshape(weight.shape[0], -1).double()
    u = dirs["u"]
    return -(u * beta.to(u.device).double()) @ (u.transpose(0, 1) @ w)


def output_contrast_contraction(weight: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor, rho: float,
                                displacement: float) -> Tuple[torch.Tensor, dict]:
    """The output-contrast Delta W (float64) at ``rho`` with the given displacement, and its report: directions used,
    the leading lambdas and the output energy report."""
    dirs = output_contrast_directions(weight, cov_forget, cov_retain, rho)
    beta = output_contrast_beta(dirs["energy"], displacement)
    delta = output_contrast_delta(weight, dirs, beta)
    rep = output_energy_report(weight, delta, cov_forget, cov_retain)
    rep.update({"directions": int((beta > 0).sum()), "out": int(beta.numel()),
                "lambda_leading": [float(x) for x in dirs["lam"][:5]], "displacement": float(delta.norm())})
    return delta, rep


def output_contrast_simulation(weight: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor, rho: float,
                               displacement: float) -> dict:
    """The report of ``output_contrast_contraction`` without applying it."""
    return output_contrast_contraction(weight, cov_forget, cov_retain, rho, displacement)[1]
