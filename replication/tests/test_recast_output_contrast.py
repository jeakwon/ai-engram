"""Numerical tests of the output energies and the output-contrast contraction (recast_output_contrast.py)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recast_output_contrast as O  # noqa: E402


def _case(seed, out=6, inn=9, n=400):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    xf = torch.randn(n, inn, generator=gen, dtype=torch.float64) * torch.linspace(0.5, 2.0, inn, dtype=torch.float64)
    xr = torch.randn(n, inn, generator=gen, dtype=torch.float64)
    return w, xf, xr, xf.T @ xf / n, xr.T @ xr / n


def test_output_energy_is_the_mean_squared_output_and_the_report_scales():
    w, xf, _, cf, cr = _case(1)
    assert abs(O.output_energy(w, cf) - float((xf @ w.T).square().sum(1).mean())) < 1e-9
    rep = O.output_energy_report(w, (0.3 - 1.0) * w, cf, cr)              # W' = 0.3 W
    for side in ("forget", "retain"):
        assert abs(rep[f"{side}_ratio"] - 0.3) < 1e-12 and abs(rep[f"{side}_removed_share"] - 0.49) < 1e-12
    gone = O.output_energy_report(w, -w, cf, cr)
    assert gone["retain_ratio"] < 1e-12 and abs(gone["forget_removed_share"] - 1.0) < 1e-12


def test_directions_are_orthonormal_complete_and_ordered_by_contrast():
    w, _, _, cf, cr = _case(2)
    d = O.output_contrast_directions(w, cf, cr, rho=0.1)
    u, lam = d["u"], d["lam"]
    assert torch.allclose(u.T @ u, torch.eye(u.shape[0], dtype=u.dtype), atol=1e-10)
    assert torch.allclose(u @ (u.T @ w), w, atol=1e-10)                     # sum_j E_j = W
    assert bool((lam[:-1] >= lam[1:] - 1e-12).all())
    assert abs(float(d["energy"].sum()) - float(w.square().sum())) < 1e-9
    a_f, a_r = w @ cf @ w.T, w @ cr @ w.T
    a_f, a_r = a_f / torch.trace(a_f), a_r / torch.trace(a_r)
    b = a_r + 0.1 / w.shape[0] * torch.eye(w.shape[0], dtype=w.dtype)
    v = u[:, 0]                                                           # the leading direction solves the pencil
    assert float((a_f @ v - lam[0] * (b @ v)).norm()) < 1e-8


def test_contraction_removes_the_forget_specific_output_first():
    w = torch.eye(4, dtype=torch.float64)
    cf, cr = torch.diag(torch.tensor([10.0, 1.0, 1.0, 1.0], dtype=torch.float64)), torch.eye(4, dtype=torch.float64)
    rep = O.output_contrast_simulation(w, cf, cr, rho=0.01, displacement=1.0)
    assert rep["directions"] == 1 and abs(rep["displacement"] - 1.0) < 1e-10
    assert abs(rep["forget_removed_share"] - 10.0 / 13.0) < 1e-9 and abs(rep["retain_removed_share"] - 0.25) < 1e-9


def test_beta_meets_the_displacement_and_saturates():
    w, _, _, cf, cr = _case(3)
    d = O.output_contrast_directions(w, cf, cr, rho=1.0)
    for disp in (0.0, 0.5, 2.0):
        beta = O.output_contrast_beta(d["energy"], disp)
        assert abs(float(O.output_contrast_delta(w, d, beta).norm()) - disp) < 1e-9
        assert int(((beta > 0) & (beta < 1)).sum()) <= 1
    assert torch.equal(O.output_contrast_beta(d["energy"], 1e6), torch.ones_like(d["energy"]))
    with pytest.raises(ValueError):
        O.output_contrast_directions(w, cf, cr, rho=0.0)


def test_contraction_returns_the_delta_of_its_report():
    w, _, _, cf, cr = _case(4)
    delta, rep = O.output_contrast_contraction(w, cf, cr, rho=0.1, displacement=1.5)
    assert abs(float(delta.norm()) - 1.5) < 1e-9 and abs(rep["displacement"] - 1.5) < 1e-9
    assert abs(rep["retain_removed_share"] - O.output_energy(delta, cr) / O.output_energy(w, cr)) < 1e-12
    assert O.output_contrast_simulation(w, cf, cr, 0.1, 1.5) == rep
