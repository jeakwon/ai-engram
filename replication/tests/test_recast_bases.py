"""Numerical tests of Stage 3 (recast_bases.py): the weight core against the update core, on small synthetic cases."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recast_bases as B  # noqa: E402


def _case(seed, out=10, inn=8, r=4, one_sided=False):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    u = None if one_sided else torch.linalg.qr(torch.randn(out, r, generator=gen, dtype=torch.float64))[0]
    vh = torch.linalg.qr(torch.randn(inn, r, generator=gen, dtype=torch.float64))[0].T
    return w, u, vh, gen


def _proj(x, u, vh):
    return (x @ vh.T @ vh) if u is None else u @ u.T @ x @ vh.T @ vh


def test_weight_components_rebuild_the_projected_weights_and_the_norm_identity():
    for one_sided in (False, True):
        w, u, vh, gen = _case(1, one_sided=one_sided)
        comps = B.weight_components(w, u, vh)
        e = [comps["kappa"][j] * torch.outer(comps["left"][:, j], comps["right"][j]) for j in range(comps["kappa"].numel())]
        assert torch.allclose(sum(e), _proj(w, u, vh), atol=1e-10)
        for i in range(len(e)):                                   # E_j orthogonal, <W, E_j> = kappa_j^2
            assert abs(float((w * e[i]).sum()) - float(comps["kappa"][i]) ** 2) < 1e-9
            for j in range(i + 1, len(e)):
                assert abs(float((e[i] * e[j]).sum())) < 1e-10
        for _ in range(3):
            beta = torch.rand(comps["kappa"].numel(), generator=gen, dtype=torch.float64)
            lhs = float(B.contract(w, comps, beta).square().sum())
            rhs = float(w.square().sum()) - float(((2 * beta - beta ** 2) * comps["kappa"] ** 2).sum())
            assert abs(lhs - rhs) < 1e-9 * max(1.0, abs(rhs))


def test_uniform_shrink_shares_the_subspaces_and_beta_rebuilds_recast():
    w, u, vh, _ = _case(2)
    s = 0.3
    factors = torch.full((w.shape[0],), s, dtype=torch.float64)
    c = B.cores(w, factors, u, vh)
    assert torch.allclose(c["B"], (s - 1) * c["K"], atol=1e-12)
    rep = B.compare_cores(w, factors, u, vh, ks=(1, 2, 4))
    for k, a in rep["angles"].items():
        assert a["left_min_cos"] > 1 - 1e-9 and a["right_min_cos"] > 1 - 1e-9
    assert rep["spectrum_l2_distance"] < 1e-12 and rep["beta"]["offdiag_energy_fraction"] < 1e-20
    assert abs(rep["beta"]["mean"] - (1 - s)) < 1e-12 and abs(rep["beta"]["min"] - (1 - s)) < 1e-12
    comps = B.weight_components(w, u, vh)
    recast = w + _proj((s - 1) * w, u, vh)
    assert torch.allclose(B.contract(w, comps, torch.full_like(comps["kappa"], 1 - s)), recast, atol=1e-12)


def test_repeated_singular_values_compare_by_subspace_not_by_vector():
    """K with a repeated singular value: rotating inside the repeated pair changes the vectors, not the span."""
    gen = torch.Generator().manual_seed(3)
    p = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))[0]
    q = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))[0]
    k = p @ torch.diag(torch.tensor([3.0, 2.0, 2.0, 1.0], dtype=torch.float64)) @ q.T
    rot = torch.eye(4, dtype=torch.float64)
    rot[1:3, 1:3] = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
    k2 = (p @ rot) @ torch.diag(torch.tensor([3.0, 2.0, 2.0, 1.0], dtype=torch.float64)) @ (q @ rot).T
    assert torch.allclose(k, k2, atol=1e-12)                      # the same matrix, other singular vectors
    cos = B.principal_angle_cosines(p[:, :3], (p @ rot)[:, :3])
    assert float(cos.min()) > 1 - 1e-12                            # the leading 3-dimensional spans agree
    assert abs(float(p[:, 1] @ (p @ rot)[:, 1])) < 0.7             # while the second vectors do not


def test_non_uniform_shrink_moves_the_subspaces_and_leaves_an_offdiagonal_residual():
    w, u, vh, gen = _case(4, out=12, inn=10, r=5)
    factors = 0.1 + 0.9 * torch.rand(w.shape[0], generator=gen, dtype=torch.float64)
    rep = B.compare_cores(w, factors, u, vh, ks=(1, 2, 5), grads={"forget": torch.randn(w.shape, generator=gen,
                                                                                       dtype=torch.float64)})
    assert rep["angles"]["2"]["left_min_cos"] < 1 - 1e-6
    assert rep["angles"]["5"]["left_min_cos"] > 1 - 1e-9          # at full rank both span the whole core space
    assert rep["beta"]["offdiag_energy_fraction"] > 1e-6
    c = B.cores(w, factors, u, vh)
    pk, kappa, qkh = torch.linalg.svd(c["K"], full_matrices=False)
    coeff = pk.T @ c["B"] @ qkh.T
    assert torch.allclose(u @ pk @ coeff @ qkh @ vh, _proj((factors - 1).unsqueeze(1) * w, u, vh), atol=1e-10)
    diag_only = u @ pk @ torch.diag(torch.diagonal(coeff)) @ qkh @ vh
    resid = float((diag_only - _proj((factors - 1).unsqueeze(1) * w, u, vh)).square().sum() / coeff.square().sum())
    assert abs(resid - rep["beta"]["offdiag_energy_fraction"]) < 1e-10
    assert len(rep["alignment"]["forget"]["weight_leading"]) == 5


def test_effective_rank_limits():
    assert abs(B.effective_rank(torch.tensor([1.0, 1.0, 1.0, 1.0])) - 4.0) < 1e-12
    assert abs(B.effective_rank(torch.tensor([1.0, 0.0, 0.0])) - 1.0) < 1e-12
    assert B.effective_rank(torch.zeros(3)) == 0.0
