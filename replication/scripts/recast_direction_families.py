"""Three families of forget-specific directions for one weight matrix, compared at the same removed forget output
(extension, user 2026-10-29).

For a target W (out x in), the uncentred input covariances C_f, C_r over answer tokens and the mean gradients G_f, G_r:

- output contrast: output directions u with ``(W C_f W^T / tr) u = lambda (W C_r W^T / tr + rho / out I) u``;
- gradient contrast: output directions u with ``(G_f G_f^T / tr) u = lambda (G_r G_r^T / tr + rho / out I) u``;
- input contrast: input directions v with ``(C_f / tr) v = lambda (C_r / tr + rho / in I) v`` (the input side of
  basis D, all of them, at this ridge).

Each family is orthonormalised in the order of lambda (descending). An output family removes W's output along its
leading directions, ``Delta W = -sum_j beta_j u_j u_j^T W``; the input family removes W's action on its leading input
directions, ``Delta W = -W sum_j beta_j v_j v_j^T``. Each takes beta 1 in its order until the removed forget output
energy ``E_f(Delta W) = tr(Delta W C_f Delta W^T)`` reaches a target share of ``E_f(W)`` (the share the ReCAST update
removes), the last direction partly, and reports the retain share ``E_r(Delta W) / E_r(W)`` removed on the way: at the
same forget removal, a smaller retain share is more selective. For the output families the energies add over
directions; for the input family they do not (C and W^T W mix the directions), so the leading blocks are summed.
"""

import math
from typing import Dict

import torch

FAMILIES = ("output_contrast", "gradient_contrast", "input_contrast")


def ordered_directions(num: torch.Tensor, den: torch.Tensor, rho: float, return_values: bool = False):
    """Orthonormal columns spanning, in order, the leading generalised eigenvectors of
    ``(num / tr num) x = lambda (den / tr den + rho / n I) x`` (float64); with ``return_values`` also the
    eigenvalues lambda, descending (one per column)."""
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    a, b = num.double(), den.double()
    a, b = a / torch.trace(a), b / torch.trace(b)
    n = a.shape[0]
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    chol = torch.linalg.cholesky(b + (rho / n) * eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False)
    m = linv @ a @ linv.transpose(0, 1)
    lam, q = torch.linalg.eigh(0.5 * (m + m.transpose(0, 1)))
    idx = torch.argsort(lam, descending=True)
    q = q[:, idx]
    u, r = torch.linalg.qr(torch.linalg.solve_triangular(chol.transpose(0, 1), q, upper=True))
    d = torch.diagonal(r)
    u = u * torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    return (u, lam[idx]) if return_values else u


def _additive_at_target(ef: torch.Tensor, er: torch.Tensor, ew: torch.Tensor, target: float) -> dict:
    """Directions in order with additive removed energies ef, er (and displacement energies ew), beta 1 until the
    forget energy reaches ``target``, the last one partly."""
    cf = torch.cumsum(ef, 0)
    k = int(torch.searchsorted(cf, torch.tensor([target], dtype=cf.dtype, device=cf.device)).item())
    if k >= ef.numel():
        return {"directions": int(ef.numel()), "forget": float(cf[-1]), "retain": float(er.sum()),
                "displacement2": float(ew.sum()), "reached": False, "beta_last": 1.0}
    before = float(cf[k - 1]) if k > 0 else 0.0
    b2 = (target - before) / float(ef[k]) if float(ef[k]) > 0 else 0.0
    return {"directions": k + 1, "forget": target, "retain": float(er[:k].sum()) + b2 * float(er[k]),
            "displacement2": float(ew[:k].sum()) + b2 * float(ew[k]), "reached": True, "beta_last": math.sqrt(b2)}


def _input_at_target(hf: torch.Tensor, hr: torch.Tensor, ew: torch.Tensor, target: float) -> dict:
    """The same for the input family, whose removed energy over the leading k directions is the sum of the leading
    k x k block of ``H = (V^T W^T W V) * (V^T C V)``; the last direction partly (the quadratic in its beta)."""
    sf = hf.cumsum(0).cumsum(1).diagonal()
    sr = hr.cumsum(0).cumsum(1).diagonal()
    above = (sf >= target).nonzero()
    if above.numel() == 0:
        return {"directions": int(sf.numel()), "forget": float(sf[-1]), "retain": float(sr[-1]),
                "displacement2": float(ew.sum()), "reached": False, "beta_last": 1.0}
    k = int(above[0])
    if k == 0:
        prev_f, cross_f, prev_r, cross_r = 0.0, 0.0, 0.0, 0.0
    else:
        prev_f, prev_r = float(sf[k - 1]), float(sr[k - 1])
        cross_f, cross_r = float(hf[:k, k].sum()), float(hr[:k, k].sum())
    a, c = float(hf[k, k]), cross_f
    disc = c * c - a * (prev_f - target)
    beta = (-c + math.sqrt(disc)) / a if a > 0 and disc >= 0 else 1.0
    beta = min(max(beta, 0.0), 1.0)
    return {"directions": k + 1, "forget": prev_f + 2 * beta * cross_f + beta * beta * a,
            "retain": prev_r + 2 * beta * cross_r + beta * beta * float(hr[k, k]),
            "displacement2": float(ew[:k].sum()) + beta * beta * float(ew[k]), "reached": True, "beta_last": beta}


def family_report(weight: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor, g_forget: torch.Tensor,
                  g_retain: torch.Tensor, target_share: float, rhos=(0.1,)) -> Dict[str, Dict[str, dict]]:
    """For each family and ridge: the retain output share removed when the forget output share reaches
    ``target_share``, the directions used and the weight displacement."""
    w = weight.reshape(weight.shape[0], -1).double()
    cf, cr = cov_forget.to(w.device).double(), cov_retain.to(w.device).double()
    gf = g_forget.reshape(w.shape).to(w.device).double()
    gr = g_retain.reshape(w.shape).to(w.device).double()
    a_f, a_r = w @ cf @ w.transpose(0, 1), w @ cr @ w.transpose(0, 1)
    e_f, e_r = float(torch.trace(a_f)), float(torch.trace(a_r))
    target = target_share * e_f
    out: Dict[str, Dict[str, dict]] = {f: {} for f in FAMILIES}
    for rho in rhos:
        for family, (num, den) in (("output_contrast", (a_f, a_r)),
                                   ("gradient_contrast", (gf @ gf.transpose(0, 1), gr @ gr.transpose(0, 1)))):
            u = ordered_directions(num, den, rho)
            ef = ((u.transpose(0, 1) @ a_f) * u.transpose(0, 1)).sum(1)
            er = ((u.transpose(0, 1) @ a_r) * u.transpose(0, 1)).sum(1)
            ew = (u.transpose(0, 1) @ w).square().sum(1)
            res = _additive_at_target(ef, er, ew, target)
            out[family][f"{rho:g}"] = res
            del u
        v = ordered_directions(cf, cr, rho)
        wv = w @ v
        n = wv.transpose(0, 1) @ wv
        res = _input_at_target(n * (v.transpose(0, 1) @ cf @ v), n * (v.transpose(0, 1) @ cr @ v), n.diagonal(), target)
        out["input_contrast"][f"{rho:g}"] = res
        del v, wv, n
    for fam in out.values():
        for res in fam.values():
            _to_shares(res, e_f, e_r)
    return out


def _to_shares(res: dict, e_f: float, e_r: float) -> dict:
    res["forget_share"] = res.pop("forget") / e_f
    res["retain_share"] = res.pop("retain") / e_r
    res["displacement"] = math.sqrt(max(res.pop("displacement2"), 0.0))
    return res


def family_contraction_delta(weight: torch.Tensor, cov_forget: torch.Tensor, cov_retain: torch.Tensor, family: str,
                             rho: float, target_share: float):
    """The Delta W (float64) of the output- or input-contrast family at ``rho`` that removes ``target_share`` of the
    forget output energy (beta 1 in the family's order, the last direction partly), and its report: directions,
    the last beta, the forget and retain output shares removed and the displacement."""
    if family not in ("output_contrast", "input_contrast"):
        raise ValueError(f"family must be output_contrast or input_contrast, got {family!r}")
    w = weight.reshape(weight.shape[0], -1).double()
    cf, cr = cov_forget.to(w.device).double(), cov_retain.to(w.device).double()
    a_f, a_r = w @ cf @ w.transpose(0, 1), w @ cr @ w.transpose(0, 1)
    e_f, e_r = float(torch.trace(a_f)), float(torch.trace(a_r))
    target = target_share * e_f
    if family == "output_contrast":
        u = ordered_directions(a_f, a_r, rho)
        ut = u.transpose(0, 1)
        res = _additive_at_target(((ut @ a_f) * ut).sum(1), ((ut @ a_r) * ut).sum(1), (ut @ w).square().sum(1), target)
        k = res["directions"]
        beta = torch.ones(k, dtype=w.dtype, device=w.device)
        beta[-1] = res["beta_last"]
        delta = -(u[:, :k] * beta) @ (ut[:k] @ w)
    else:
        v = ordered_directions(cf, cr, rho)
        wv = w @ v
        n = wv.transpose(0, 1) @ wv
        res = _input_at_target(n * (v.transpose(0, 1) @ cf @ v), n * (v.transpose(0, 1) @ cr @ v), n.diagonal(), target)
        k = res["directions"]
        beta = torch.ones(k, dtype=w.dtype, device=w.device)
        beta[-1] = res["beta_last"]
        delta = -(wv[:, :k] * beta) @ v[:, :k].transpose(0, 1)
    return delta.reshape(weight.shape), _to_shares(res, e_f, e_r)
