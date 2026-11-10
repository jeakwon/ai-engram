"""Numerical tests of the global selection of forget-specific directions over several targets
(recast_global_families.py)."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recast_global_families as G  # noqa: E402

FAMILIES = ("output_contrast", "gradient_contrast", "input_contrast")


def _target(gen, out, inn, n=200):
    w = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    xf = torch.randn(n, inn, generator=gen, dtype=torch.float64) * torch.linspace(0.3, 2.0, inn, dtype=torch.float64)
    xr = torch.randn(n, inn, generator=gen, dtype=torch.float64)
    gf = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    gr = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    return w, xf.T @ xf / n, xr.T @ xr / n, gf, gr


def _case(seed):
    gen = torch.Generator().manual_seed(seed)
    return {"a.o_proj": _target(gen, 5, 5), "a.down_proj": _target(gen, 5, 8), "b.o_proj": _target(gen, 5, 5),
            "b.down_proj": _target(gen, 5, 8)}


def _energy(m, c):
    return float(torch.trace(m @ c @ m.T))


def _parts(case, family, rho=0.5, random_seed=None, key_alpha=None):
    out = {}
    for i, (n, (w, cf, cr, gf, gr)) in enumerate(case.items()):
        gen = None if random_seed is None else torch.Generator().manual_seed(random_seed * 1000 + i)
        out[n] = G.target_directions(w, cf, cr, gf, gr, family, rho, keys_generator=gen, key_alpha=key_alpha)
    return out


@pytest.mark.parametrize("family", FAMILIES)
def test_gains_add_up_to_the_target_energy(family):
    """Over the whole family the gains add up to tr(W C W^T) on both sides, and the directions are orthonormal."""
    case = _case(1)
    for n, p in _parts(case, family).items():
        w, cf, cr = case[n][:3]
        d = p["directions"]
        assert torch.allclose(d.T @ d, torch.eye(d.shape[1], dtype=d.dtype), atol=1e-10)
        assert abs(float(p["gain_f"].sum()) - _energy(w, cf)) < 1e-8 * _energy(w, cf)
        assert abs(float(p["gain_r"].sum()) - _energy(w, cr)) < 1e-8 * _energy(w, cr)
        assert abs(float(p["disp2"].sum()) - float(w.square().sum())) < 1e-8 * float(w.square().sum())


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("floor", (0.04, 0.4))
@pytest.mark.parametrize("keys", ("score", "random", "alpha0", "alpha0.5", "alpha1"))
def test_selection_meets_the_budget_with_the_applied_deltas(family, floor, keys):
    """The deltas of the chosen prefixes remove exactly the budget of forget energy over all targets, and the reported
    per-target energies and displacements are those of the deltas."""
    case = _case(2)
    parts = _parts(case, family, random_seed=3 if keys == "random" else None,
                   key_alpha=float(keys[5:]) if keys.startswith("alpha") else None)
    total = sum(_energy(c[0], c[1]) for c in case.values())
    budget = 0.2 * total
    sel = G.global_selection(parts, floor, budget)
    assert sel["global"]["reached"]
    got = 0.0
    for n, (w, cf, cr, _, _) in case.items():
        r = sel["per_target"][n]
        delta = G.global_delta(parts[n], w, r["directions"], 1.0 - floor, r["beta_last"])
        got += _energy(delta, cf)
        assert abs(_energy(delta, cf) - r["forget_energy_removed"]) < 1e-8 * total
        assert abs(_energy(delta, cr) - r["retain_energy_removed"]) < 1e-8 * total
        assert abs(float(delta.norm()) - r["displacement"]) < 1e-8 * float(w.norm())
        if r["beta_last"] is not None:
            assert 0.0 <= r["beta_last"] <= 1.0 - floor + 1e-12
    assert abs(got - budget) < 1e-8 * total
    assert abs(sel["global"]["forget_energy_removed"] - budget) < 1e-8 * total
    assert sum(r["beta_last"] is not None for r in sel["per_target"].values()) == 1


@pytest.mark.parametrize("family", FAMILIES)
def test_score_order_takes_the_largest_lambda_first(family):
    """With the lambda keys every fully taken direction has a lambda at least that of every direction left out."""
    parts = _parts(_case(4), family)
    total = sum(p["energy_f"] for p in parts.values())
    sel = G.global_selection(parts, 0.04, 0.3 * total)
    taken, left = [], []
    for n, p in parts.items():
        r = sel["per_target"][n]
        full = r["directions"] - (1 if r["beta_last"] is not None else 0)
        taken += p["lam"][:full].tolist()
        left += p["lam"][r["directions"]:].tolist()
    assert taken and left and min(taken) >= max(left) - 1e-12


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("alpha", (0.0, 0.5, 1.0))
def test_energy_ratio_keys_rank_by_the_energies_of_each_direction_alone(family, alpha):
    """energy_ratio keys order every target by e_f / e_r^alpha of each direction alone (the energy of removing only
    that direction), and the global selection takes the largest first."""
    case = _case(8)
    parts = _parts(case, family, key_alpha=alpha)
    for n, p in parts.items():
        w, cf, cr = case[n][:3]
        d = p["directions"]
        if p["side"] == "output":
            alone = [(_energy(d[:, j:j + 1] @ d[:, j:j + 1].T @ w, cf), _energy(d[:, j:j + 1] @ d[:, j:j + 1].T @ w, cr))
                     for j in range(d.shape[1])]
        else:
            alone = [(_energy(w @ d[:, j:j + 1] @ d[:, j:j + 1].T, cf), _energy(w @ d[:, j:j + 1] @ d[:, j:j + 1].T, cr))
                     for j in range(d.shape[1])]
        assert torch.allclose(p["self_f"], torch.tensor([a for a, _ in alone], dtype=torch.float64), rtol=1e-8)
        ratio = torch.tensor([math.log(a) - alpha * math.log(b) for a, b in alone], dtype=torch.float64)
        assert torch.allclose(p["keys"], -ratio, rtol=1e-8, atol=1e-8)
        assert bool((ratio[:-1] >= ratio[1:] - 1e-10).all())
    total = sum(p["energy_f"] for p in parts.values())
    sel = G.global_selection(parts, 0.04, 0.3 * total)
    taken, left = [], []
    for n, p in parts.items():
        r = sel["per_target"][n]
        full = r["directions"] - (1 if r["beta_last"] is not None else 0)
        taken += (-p["keys"][:full]).tolist()
        left += (-p["keys"][r["directions"]:]).tolist()
    assert taken and left and min(taken) >= max(left) - 1e-10


def test_random_order_differs_from_the_score_order():
    case = _case(5)
    total = sum(_energy(c[0], c[1]) for c in case.values())
    a = G.global_selection(_parts(case, "output_contrast"), 0.04, 0.3 * total)
    b = G.global_selection(_parts(case, "output_contrast", random_seed=7), 0.04, 0.3 * total)
    assert ({n: r["directions"] for n, r in a["per_target"].items()}
            != {n: r["directions"] for n, r in b["per_target"].items()}
            or a["global"]["retain_energy_removed"] != b["global"]["retain_energy_removed"])


@pytest.mark.parametrize("family", FAMILIES)
def test_unreachable_budget_takes_every_direction(family):
    """Every direction at beta = 1 - floor removes (1 - floor)^2 of the forget energy: a larger budget is not reached
    and the delta is -(1 - floor) W."""
    case = _case(6)
    parts = _parts(case, family)
    total = sum(p["energy_f"] for p in parts.values())
    sel = G.global_selection(parts, 0.4, 0.5 * total)
    assert not sel["global"]["reached"]
    assert abs(sel["global"]["forget_energy_removed"] - 0.36 * total) < 1e-8 * total
    for n, (w, *_rest) in case.items():
        r = sel["per_target"][n]
        assert r["directions"] == r["of"] and r["beta_last"] is None
        delta = G.global_delta(parts[n], w, r["directions"], 0.6, None)
        assert torch.allclose(delta, -0.6 * w, atol=1e-10)


def test_zero_budget_and_errors():
    parts = _parts(_case(7), "output_contrast")
    sel = G.global_selection(parts, 0.04, 0.0)
    assert sel["global"]["directions"] == 0 and sel["global"]["forget_energy_removed"] == 0.0
    with pytest.raises(ValueError):
        G.global_selection(parts, 1.0, 1.0)
    with pytest.raises(ValueError):
        G.global_selection(parts, 0.04, -1.0)
    w, cf, cr, gf, gr = _case(7)["a.o_proj"]
    with pytest.raises(ValueError):
        G.target_directions(w, cf, cr, gf, gr, "unknown", 0.5)
    with pytest.raises(ValueError):
        G.target_directions(w, cf, cr, gf, gr, "output_contrast", 0.5, keys_generator=torch.Generator(), key_alpha=1.0)
    with pytest.raises(ValueError):
        G.target_directions(w, cf, cr, gf, gr, "output_contrast", 0.5, key_alpha=-1.0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("random_seed", (None, 5))
def test_share_unit_meets_a_budget_of_summed_target_shares(family, random_seed):
    """With the share unit the removed forget energy of every target, divided by its own forget output energy and summed
    over the targets, equals the budget, whatever the scale of each target's energies."""
    case = _case(9)
    case = {n: (w * (10.0 if i % 2 else 0.1), cf, cr, gf, gr) for i, (n, (w, cf, cr, gf, gr)) in enumerate(case.items())}
    parts = _parts(case, family, random_seed=random_seed)
    sel = G.global_selection(parts, 0.04, 0.8, unit="share")
    assert sel["global"]["reached"] and sel["global"]["unit"] == "share"
    got = 0.0
    for n, (w, cf, cr, _, _) in case.items():
        r = sel["per_target"][n]
        delta = G.global_delta(parts[n], w, r["directions"], 0.96, r["beta_last"])
        got += _energy(delta, cf) / _energy(w, cf)
    assert abs(got - 0.8) < 1e-8
    assert abs(sel["global"]["forget_share_sum"] - 0.8) < 1e-8
    with pytest.raises(ValueError):
        G.global_selection(parts, 0.04, 0.8, unit="bits")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("gamma", (0.5, 1.0, 2.0))
@pytest.mark.parametrize("unit", ("energy", "share"))
def test_per_direction_s_meets_the_budget(family, gamma, unit):
    """With s_gamma each direction is contracted by c_j = 1 - clip(lambda_j^-gamma, s_min, 1) (0 for lambda <= 1); the
    deltas of the chosen prefixes remove exactly the budget, in either unit, and match the reported energies."""
    case = _case(10)
    parts = {n: G.target_directions(w, cf, cr, gf, gr, family, 0.5, s_gamma=gamma, s_min=0.04)
             for n, (w, cf, cr, gf, gr) in case.items()}
    for p in parts.values():
        expect = 1.0 - p["lam"].clamp_min(1e-300).pow(-gamma).clamp(min=0.04, max=1.0)
        assert torch.allclose(p["coef"], expect)
        assert bool((p["coef"][p["lam"] <= 1.0] == 0).all())
    full = G.global_selection(parts, 0.0, 1e30, unit=unit)["global"]       # every direction at its own coefficient
    most = full["forget_share_sum"] if unit == "share" else full["forget_energy_removed"]
    assert not G.global_selection(parts, 0.0, 1.5 * most, unit=unit)["global"]["reached"]
    budget = 0.5 * most
    sel = G.global_selection(parts, 0.0, budget, unit=unit)
    assert sel["global"]["reached"]
    got = 0.0
    for n, (w, cf, cr, _, _) in case.items():
        r = sel["per_target"][n]
        delta = G.global_delta(parts[n], w, r["directions"], 1.0, r["beta_last"])
        e = _energy(delta, cf)
        assert abs(e - r["forget_energy_removed"]) < 1e-8 * max(1.0, e)
        assert abs(_energy(delta, cr) - r["retain_energy_removed"]) < 1e-8 * max(1.0, _energy(w, cr))
        got += e / _energy(w, cf) if unit == "share" else e
    assert abs(got - budget) < 1e-8 * max(1.0, budget)
    with pytest.raises(ValueError):
        G.target_directions(*case["a.o_proj"], family, 0.5, s_gamma=0.0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("unit", ("energy", "share"))
def test_retain_share_s_rule(family, unit):
    """s_rule retain_share: s_j = clip(e_r,j / (e_f,j + e_r,j), s_min, 1) from the energies of each direction alone;
    the deltas still remove exactly the budget."""
    case = _case(11)
    parts = {n: G.target_directions(w, cf, cr, gf, gr, family, 0.5, s_min=0.04, s_rule="retain_share")
             for n, (w, cf, cr, gf, gr) in case.items()}
    for n, p in parts.items():
        w, cf, cr = case[n][:3]
        d = p["directions"]
        for j in range(d.shape[1]):
            one = d[:, j:j + 1] @ d[:, j:j + 1].T
            m = one @ w if p["side"] == "output" else w @ one
            ef, er = _energy(m, cf), _energy(m, cr)
            assert abs(float(p["coef"][j]) - (1.0 - min(max(er / (ef + er), 0.04), 1.0))) < 1e-8
    full = G.global_selection(parts, 0.0, 1e30, unit=unit)["global"]
    most = full["forget_share_sum"] if unit == "share" else full["forget_energy_removed"]
    budget = 0.5 * most
    sel = G.global_selection(parts, 0.0, budget, unit=unit)
    got = 0.0
    for n, (w, cf, cr, _, _) in case.items():
        r = sel["per_target"][n]
        e = _energy(G.global_delta(parts[n], w, r["directions"], 1.0, r["beta_last"]), cf)
        got += e / _energy(w, cf) if unit == "share" else e
    assert sel["global"]["reached"] and abs(got - budget) < 1e-8 * max(1.0, budget)
    with pytest.raises(ValueError):
        G.target_directions(*case["a.o_proj"], family, 0.5, s_gamma=1.0, s_rule="retain_share")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("unit", ("energy", "share"))
@pytest.mark.parametrize("gamma", (1.0, 4.0))
def test_calibrated_threshold_meets_the_budget(family, unit, gamma):
    """calibrated_selection: one threshold lambda* for all targets, c_j = 1 - clip((lambda*/lambda_j)^gamma, s_min, 1);
    the deltas remove the budget, c grows with lambda, and directions with lambda <= lambda* stay."""
    case = _case(12)
    parts = {n: G.target_directions(w, cf, cr, gf, gr, family, 0.5, keep_h=True) for n, (w, cf, cr, gf, gr) in case.items()}
    most = G.calibrated_selection(parts, 1e30, gamma, s_min=0.04, unit=unit)["global"]
    assert not most["reached"]
    top = most["forget_share_sum"] if unit == "share" else most["forget_energy_removed"]
    sel = G.calibrated_selection(parts, 0.4 * top, gamma, s_min=0.04, unit=unit)
    assert sel["global"]["reached"]
    got = 0.0
    for n, (w, cf, cr, _, _) in case.items():
        r = sel["per_target"][n]
        c, lam = r["coef"], parts[n]["lam"]
        assert bool((c[lam <= sel["global"]["threshold"]] == 0).all())
        order = torch.argsort(lam)
        assert bool((c[order][1:] >= c[order][:-1] - 1e-12).all())
        delta = G.delta_from_coef(parts[n], w, c)
        e = _energy(delta, cf)
        assert abs(e - r["forget_energy_removed"]) < 1e-5 * max(1.0, e)
        got += e / _energy(w, cf) if unit == "share" else e
    assert abs(got - 0.4 * top) < 1e-5 * max(1.0, top)
