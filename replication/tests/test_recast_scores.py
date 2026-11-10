"""Numerical tests of the Stage 2 shrink scores (recast_scores.py) against dense definitions on small matrices."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recast_scores as S  # noqa: E402


def _case(seed, out=9, inn=7, r=3, one_sided=False):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    g = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    u = None if one_sided else torch.linalg.qr(torch.randn(out, r, generator=gen, dtype=torch.float64))[0]
    vh = torch.linalg.qr(torch.randn(inn, r, generator=gen, dtype=torch.float64))[0].T
    return w, g, u, vh


def _proj(x, u, vh):
    return (x @ vh.T @ vh) if u is None else u @ u.T @ x @ vh.T @ vh


def _cos(a, b):
    return float(a @ b) / float(a.norm() * b.norm())


def test_scores_match_dense_definitions():
    w, g, u, vh = _case(1)
    sc = S.shrink_scores(w, g, u, vh)
    wp, gp = _proj(w, u, vh), _proj(g, u, vh)
    for i in range(w.shape[0]):
        assert abs(sc["original"][i] - _cos(w[i], -g[i])) < 1e-10
        assert abs(sc["projected_row"][i] - _cos(wp[i], -gp[i])) < 1e-10
        row = torch.zeros_like(w)
        row[i] = w[i]
        t_i = _proj(row, u, vh)                                   # the feasible update of row i
        a_i = float((g * t_i).sum())
        assert abs(a_i - float((gp * row).sum())) < 1e-10           # <G, P(R_i)> = <P(G), R_i>
        assert abs(float(sc["sensitivity"][i]) - a_i) < 1e-10
        assert abs(sc["projected_sensitivity"][i] - _cos((w @ vh.T @ vh)[i], -gp[i])) < 1e-10
    assert abs(sc["global_projected"] - float((wp * -gp).sum() / (wp.norm() * gp.norm()))) < 1e-10
    k_w, k_g = u.T @ w @ vh.T, u.T @ g @ vh.T                     # the reduced cores give the same global cosine
    assert abs(sc["global_projected"] - float((k_w * -k_g).sum() / (k_w.norm() * k_g.norm()))) < 1e-10
    for rule in ("projected_row", "projected_sensitivity"):
        assert float(sc[rule].abs().max()) <= 1 + 1e-12


def test_projection_mixes_rows_but_numerators_share_their_sum():
    w, g, u, vh = _case(2)
    sc = S.shrink_scores(w, g, u, vh)
    wp, gp = _proj(w, u, vh), _proj(g, u, vh)
    num_row = (wp * gp).sum(1)                                     # numerators of projected_row (up to the sign)
    assert not torch.allclose(num_row, sc["sensitivity"], atol=1e-6)
    assert abs(float(num_row.sum()) - float(sc["sensitivity"].sum())) < 1e-10
    assert abs(float(sc["sensitivity"].sum()) - float((w * gp).sum())) < 1e-10


def test_full_bases_reduce_to_the_original_and_one_sided_rules_coincide():
    w, g, _, _ = _case(3, out=6, inn=6)
    eye = torch.eye(6, dtype=torch.float64)
    sc = S.shrink_scores(w, g, eye, eye)
    for rule in ("projected_row", "projected_sensitivity"):
        assert torch.allclose(sc[rule], sc["original"], atol=1e-10)
    w, g, u, vh = _case(4, one_sided=True)
    sc = S.shrink_scores(w, g, u, vh)
    assert torch.allclose(sc["projected_row"], sc["projected_sensitivity"], atol=1e-12)


def test_zero_norm_rows_score_zero_and_are_counted():
    w, g, u, vh = _case(5)
    u = u.clone()
    u[0] = 0.0                                                    # row 0 of U U^T is zero: P(.)_0 = 0
    u = torch.linalg.qr(u)[0]
    u[0] = 0.0
    w = w.clone()
    w[1] = 0.0                                                    # a zero weight row
    sc = S.shrink_scores(w, g, u, vh)
    assert float(sc["projected_row"][0]) == 0.0 and float(sc["projected_sensitivity"][0]) == 0.0
    # P mixes rows: P(W)_1 is not zero although W_1 is, while row 1's own feasible update T_1 = P(e_1 W_1) is
    assert float(sc["projected_row"][1]) != 0.0 and float(sc["projected_sensitivity"][1]) == 0.0
    assert sc["zero_norm_rows"]["projected_row"] == 1 and sc["zero_norm_rows"]["projected_sensitivity"] == 2
    assert torch.isfinite(sc["original"]).all()


def test_row_factors_and_update_stats_match_dense():
    w, g, u, vh = _case(6)
    gf = torch.randn(w.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    sc = S.shrink_scores(w, g, u, vh)
    for rule in S.RULES:
        s = S.row_factors(sc, rule, 0.2)
        assert float(s.min()) >= 0.2 - 1e-15
        x = (s - 1.0).unsqueeze(1) * w
        dw = _proj(x, u, vh)
        st = S.update_stats(w, s, u, vh, {"retain": g, "forget": gf})
        assert abs(st["update_norm"] - float(dw.norm())) < 1e-10
        assert abs(st["kept_energy_fraction"] - float(dw.square().sum() / x.square().sum())) < 1e-10
        assert abs(st["predicted_retain_change"] - float((g * dw).sum())) < 1e-10
        assert abs(st["predicted_forget_change"] - float((gf * dw).sum())) < 1e-10
    uniform = S.update_stats(w, torch.full((w.shape[0],), 0.3, dtype=torch.float64), u, vh, {})
    assert abs(uniform["update_norm"] - 0.7 * float(_proj(w, u, vh).norm())) < 1e-10
    with pytest.raises(ValueError):
        S.row_factors(sc, "cosine", 0.2)


def test_summary_and_correlations():
    x = torch.linspace(-1, 1, 101, dtype=torch.float64)
    out = S.score_summary(x, reference=x)
    assert abs(out["pearson_vs_original"] - 1.0) < 1e-12 and abs(out["spearman_vs_original"] - 1.0) < 1e-12
    assert abs(out["above_floor"]["0.2"] - 40 / 101) < 1e-12
    assert abs(out["quantiles"]["q50"]) < 1e-12 and out["quantiles"]["q0"] == -1.0
    assert S.pearson(x, torch.zeros_like(x)) is None
