"""Numerical tests of the weight-component contraction (recast_contraction.py) on small synthetic cases."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recast_bases as B  # noqa: E402
import recast_contraction as C  # noqa: E402


def _case(seed, out=12, inn=10, r=6):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    u = torch.linalg.qr(torch.randn(out, r, generator=gen, dtype=torch.float64))[0]
    vh = torch.linalg.qr(torch.randn(inn, r, generator=gen, dtype=torch.float64))[0].T
    return w, u, vh


def _proj(x, u, vh):
    return u @ u.T @ x @ vh.T @ vh


def _moved(kappa, beta):
    return float((beta ** 2 * kappa ** 2).sum())


def test_both_modes_move_the_weights_by_the_budget():
    w, u, vh = _case(1)
    comps = B.weight_components(w, u, vh)
    kappa, pw = comps["kappa"], float(_proj(w, u, vh).norm())
    for mode in ("top", "random"):                                      # the modes without a score
        for budget in (0.0, 0.3, 0.8, 1.0):
            gen = torch.Generator().manual_seed(7)
            beta = C.contraction_beta(kappa, mode, budget, 1.0, gen)
            assert float(beta.min()) >= 0.0 and float(beta.max()) <= 1.0
            assert abs(_moved(kappa, beta) - budget ** 2 * float((kappa ** 2).sum())) < 1e-9
            delta = C.contraction_delta(comps, beta)
            assert abs(float(delta.norm()) - budget * pw) < 1e-9
            assert torch.allclose(w + delta, B.contract(w, comps, beta), atol=1e-12)
            assert int(((beta > 0) & (beta < 1.0 - 1e-12)).sum()) <= 1    # at most one partial component


def test_top_takes_the_leading_components_and_the_next_one_partly():
    kappa = torch.tensor([3.0, 5.0, 1.0, 4.0, 2.0], dtype=torch.float64)   # energies 9, 25, 1, 16, 4; total 55
    beta = C.contraction_beta(kappa, "top", 0.8)                            # 0.64 * 55 = 35.2: 25 fully, 10.2 of 16
    assert beta[1] == 1.0 and 0 < beta[3] < 1.0 and beta[0] == beta[2] == beta[4] == 0.0
    assert abs(float(beta[3]) - ((0.64 * 55.0 - 25.0) / 16.0) ** 0.5) < 1e-12
    full = C.contraction_beta(kappa, "top", 1.0)
    assert torch.equal(full, torch.ones_like(kappa))


def test_random_is_reproducible_per_generator_and_differs_between_seeds():
    kappa = torch.linspace(1.0, 2.0, 40, dtype=torch.float64)
    a = C.contraction_beta(kappa, "random", 0.5, 1.0, torch.Generator().manual_seed(0))
    b = C.contraction_beta(kappa, "random", 0.5, 1.0, torch.Generator().manual_seed(0))
    c = C.contraction_beta(kappa, "random", 0.5, 1.0, torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and not torch.equal(a, c)
    top = C.contraction_beta(kappa, "top", 0.5)
    assert int((a > 0).sum()) >= int((top > 0).sum())                       # smaller kappa: more components


def test_budget_equal_to_strength_is_recast_at_floor_one_minus_strength():
    w, u, vh = _case(2)
    comps = B.weight_components(w, u, vh)
    s = 0.2
    for mode in ("top", "random"):                                      # the modes without a score
        beta = C.contraction_beta(comps["kappa"], mode, 1 - s, 1 - s, torch.Generator().manual_seed(3))
        assert torch.allclose(beta, torch.full_like(beta, 1 - s), atol=1e-12)
        assert torch.allclose(w + C.contraction_delta(comps, beta), w + _proj((s - 1) * w, u, vh), atol=1e-10)


def test_components_without_energy_are_never_chosen():
    kappa = torch.tensor([2.0, 0.0, 1.0, 0.0, 3.0], dtype=torch.float64)
    for seed in range(5):
        beta = C.contraction_beta(kappa, "random", 1.0, 1.0, torch.Generator().manual_seed(seed))
        assert beta[1] == 0.0 and beta[3] == 0.0 and torch.equal(beta[[0, 2, 4]], torch.ones(3, dtype=torch.float64))
    assert torch.equal(C.contraction_beta(torch.zeros(3, dtype=torch.float64), "top", 0.5), torch.zeros(3, dtype=torch.float64))


def test_invalid_arguments():
    kappa = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        C.contraction_beta(kappa, "bottom", 0.5)
    with pytest.raises(ValueError):
        C.contraction_beta(kappa, "top", 0.9, 0.8)                          # budget above strength
    with pytest.raises(ValueError):
        C.contraction_beta(kappa, "top", 0.5, 1.5)
    with pytest.raises(ValueError):
        C.contraction_beta(kappa, "random", 0.5, 1.0)                       # no generator


def test_report_counts_energy_rank_and_contrast():
    kappa = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    beta = C.contraction_beta(kappa, "top", 0.9)
    contrast = torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    rep = C.contraction_report(kappa, beta, 1.0, contrast)
    assert rep["components"] == 4 and rep["full"] == int((beta == 1.0).sum()) and rep["contracted"] == int((beta > 0).sum())
    assert abs(rep["energy_fraction"] - 0.81) < 1e-12
    assert rep["kappa_rank_mean"] < 0.5
    moved = beta ** 2 * kappa ** 2
    assert abs(rep["input_contrast"]["moved"] - float((contrast * moved).sum() / moved.sum())) < 1e-12
    assert abs(rep["input_contrast"]["all"] - float((contrast * kappa ** 2).sum() / (kappa ** 2).sum())) < 1e-12
    empty = C.contraction_report(kappa, torch.zeros_like(kappa), 1.0)
    assert empty["contracted"] == 0 and empty["kappa_rank_mean"] is None and empty["energy_fraction"] == 0.0


def test_residual_leaves_every_contracted_component_its_singular_value():
    kappa = torch.tensor([3.0, 5.0, 1.0, 4.0, 2.0], dtype=torch.float64)   # energies 9, 25, 1, 16, 4; total 55
    beta = C.contraction_beta(kappa, "top", 0.8, 1.0, residual=0.1)
    assert abs(_moved(kappa, beta) - 0.64 * 55.0) < 1e-9
    touched = beta > 0
    assert bool(((1 - beta) * kappa)[touched].min() >= 0.1 - 1e-12)
    full = torch.isclose(beta, 1 - 0.1 / kappa)
    assert bool(full[1]) and int((touched & ~full).sum()) <= 1           # 5 keeps 0.1, then 4 partly
    assert C.contraction_report(kappa, beta, 1.0, residual=0.1)["full"] == 1
    assert torch.equal(C.contraction_beta(kappa, "top", 0.8, 1.0, residual=0.0), C.contraction_beta(kappa, "top", 0.8))
    short = C.contraction_beta(kappa, "random", 1.0, 1.0, torch.Generator().manual_seed(0), residual=0.1)
    assert torch.allclose(short, 1 - 0.1 / kappa)                         # out of reach: every component at its cap
    tiny = C.contraction_beta(torch.tensor([1.76e-5], dtype=torch.float64), "top", 0.9, 1.0, residual=1e-5)
    assert abs(float(tiny[0]) - (1 - 1e-5 / 1.76e-5)) < 1e-12             # the degenerate rank-1 down_proj of layer 1
    with pytest.raises(ValueError):
        C.contraction_beta(kappa, "top", 0.5, 1.0, residual=-1.0)


def test_component_profile_matches_the_definitions():
    w, u, vh = _case(3)
    comps = B.weight_components(w, u, vh)
    gen = torch.Generator().manual_seed(5)
    g = torch.randn(w.shape, generator=gen, dtype=torch.float64)
    contrast = torch.rand(comps["kappa"].numel(), generator=gen, dtype=torch.float64)
    prof = C.component_profile(comps, {"forget": g}, contrast)
    assert len(prof["kappa"]) == len(prof["input_contrast"]) == len(prof["alignment"]["forget"]) == comps["kappa"].numel()
    for j in range(comps["kappa"].numel()):
        e = comps["kappa"][j] * torch.outer(comps["left"][:, j], comps["right"][j])
        assert abs(prof["alignment"]["forget"][j] * float(comps["kappa"][j]) - float((g * e).sum())) < 1e-9


def test_forget_retain_takes_the_highest_score_first():
    w, u, vh = _case(4)
    comps = B.weight_components(w, u, vh)
    gen = torch.Generator().manual_seed(6)
    gf = torch.randn(w.shape, generator=gen, dtype=torch.float64)
    gr = torch.randn(w.shape, generator=gen, dtype=torch.float64)
    score = C.forget_retain_score(comps, gf, gr)
    for j in range(comps["kappa"].numel()):                               # -(<G_f, E_j> - <G_r, E_j>) / kappa_j^2
        e = comps["kappa"][j] * torch.outer(comps["left"][:, j], comps["right"][j])
        expect = -(float((gf * e).sum()) - float((gr * e).sum())) / float(comps["kappa"][j]) ** 2
        assert abs(float(score[j]) - expect) < 1e-9
    beta = C.contraction_beta(comps["kappa"], "forget_retain", 0.5, 1.0, score=score)
    touched = (beta > 0).nonzero().flatten().tolist()
    assert set(touched) == set(torch.argsort(score, descending=True)[:len(touched)].tolist())
    assert abs(_moved(comps["kappa"], beta) - 0.25 * float((comps["kappa"] ** 2).sum())) < 1e-9
    with pytest.raises(ValueError):
        C.contraction_beta(comps["kappa"], "forget_retain", 0.5)          # no score
    with pytest.raises(ValueError):
        C.contraction_beta(comps["kappa"], "top", 0.5, score=score)       # a score for another mode


def test_contrast_mode_takes_the_components_by_their_score():
    kappa = torch.tensor([3.0, 1.0, 2.0, 2.5], dtype=torch.float64)
    contrast = torch.tensor([1.1, 1.9, 1.5, 1.2], dtype=torch.float64)     # order: 1, 2, 3, 0
    beta = C.contraction_beta(kappa, "contrast", 0.5, 1.0, score=contrast)
    target = 0.25 * float((kappa ** 2).sum())                               # 5.0625: 1 (1), then 4 (2) and 0.0625 of 6.25
    assert beta[1] == 1.0 and beta[2] == 1.0 and 0 < beta[3] < 1.0 and beta[0] == 0.0
    assert abs(_moved(kappa, beta) - target) < 1e-9
