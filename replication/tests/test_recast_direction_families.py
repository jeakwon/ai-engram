"""Numerical tests of the three direction families at a common removed forget output (recast_direction_families.py)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recast_direction_families as F  # noqa: E402


def _case(seed, out=5, inn=7, n=300):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    xf = torch.randn(n, inn, generator=gen, dtype=torch.float64) * torch.linspace(0.3, 2.0, inn, dtype=torch.float64)
    xr = torch.randn(n, inn, generator=gen, dtype=torch.float64)
    gf = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    gr = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    return w, xf.T @ xf / n, xr.T @ xr / n, gf, gr


def _energy(m, c):
    return float(torch.trace(m @ c @ m.T))


def test_ordered_directions_are_orthonormal_and_lead_with_the_pencil():
    _, cf, cr, _, _ = _case(1)
    v = F.ordered_directions(cf, cr, 0.1)
    assert torch.allclose(v.T @ v, torch.eye(v.shape[0], dtype=v.dtype), atol=1e-10)
    a, b = cf / torch.trace(cf), cr / torch.trace(cr) + 0.1 / cf.shape[0] * torch.eye(cf.shape[0], dtype=cf.dtype)
    lam = float(v[:, 0] @ a @ v[:, 0]) / float(v[:, 0] @ b @ v[:, 0])
    assert float((a @ v[:, 0] - lam * (b @ v[:, 0])).norm()) < 1e-8


def test_every_family_hits_the_forget_target_and_reports_its_retain_share():
    w, cf, cr, gf, gr = _case(2)
    target = 0.3
    rep = F.family_report(w, cf, cr, gf, gr, target, rhos=(0.1, 1.0))
    for rho in ("0.1", "1"):
        for fam in ("output_contrast", "gradient_contrast"):
            r = rep[fam][rho]
            num, den = (w @ cf @ w.T, w @ cr @ w.T) if fam == "output_contrast" else (gf @ gf.T, gr @ gr.T)
            u = F.ordered_directions(num, den, float(rho))[:, :r["directions"]]
            beta = torch.ones(r["directions"], dtype=torch.float64)
            beta[-1] = r["beta_last"]
            delta = -(u * beta) @ (u.T @ w)
            assert r["reached"] and abs(_energy(delta, cf) / _energy(w, cf) - target) < 1e-9
            assert abs(_energy(delta, cr) / _energy(w, cr) - r["retain_share"]) < 1e-9
            assert abs(float(delta.norm()) - r["displacement"]) < 1e-9
        r = rep["input_contrast"][rho]
        v = F.ordered_directions(cf, cr, float(rho))[:, :r["directions"]]
        beta = torch.ones(r["directions"], dtype=torch.float64)
        beta[-1] = r["beta_last"]
        delta = -(w @ (v * beta)) @ v.T
        assert abs(_energy(delta, cf) / _energy(w, cf) - r["forget_share"]) < 1e-9
        assert abs(_energy(delta, cr) / _energy(w, cr) - r["retain_share"]) < 1e-9
        assert abs(float(delta.norm()) - r["displacement"]) < 1e-9
        assert abs(r["forget_share"] - target) < 1e-9


def test_a_forget_only_output_direction_is_found_first():
    w = torch.eye(4, dtype=torch.float64)
    cf = torch.diag(torch.tensor([10.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    cr = torch.eye(4, dtype=torch.float64)
    rep = F.family_report(w, cf, cr, cf, cr, 10.0 / 13.0 * (1 - 1e-12), rhos=(0.01,))  # just inside e_1's share
    for fam in F.FAMILIES:                                                # every family removes e_1 and nothing else
        r = rep[fam]["0.01"]
        assert r["directions"] == 1 and abs(r["retain_share"] - 0.25) < 1e-9


def test_family_contraction_delta_removes_the_target_forget_share():
    w, cf, cr, gf, gr = _case(3)
    for fam in ("output_contrast", "input_contrast"):
        delta, rep = F.family_contraction_delta(w, cf, cr, fam, 1.0, 0.25)
        assert abs(_energy(delta, cf) / _energy(w, cf) - 0.25) < 1e-9 and abs(rep["forget_share"] - 0.25) < 1e-9
        assert abs(_energy(delta, cr) / _energy(w, cr) - rep["retain_share"]) < 1e-9
        assert abs(float(delta.norm()) - rep["displacement"]) < 1e-9
        other = F.family_report(w, cf, cr, gf, gr, 0.25, rhos=(1.0,))[fam]["1"]
        assert rep["directions"] == other["directions"] and abs(rep["retain_share"] - other["retain_share"]) < 1e-12
