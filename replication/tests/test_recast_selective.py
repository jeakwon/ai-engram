"""Tests of the selective input contrast with retain protection (recast_selective.py)."""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recast_direction_families as F  # noqa: E402
import recast_selective as S  # noqa: E402


def _diag(values):
    return torch.diag(torch.tensor(values, dtype=torch.float64))


def test_common_prefix_length():
    assert S.common_prefix_length([[1, 2, 3, 4], [1, 2, 3, 9], [1, 2, 5]]) == 2
    assert S.common_prefix_length([[7, 8]]) == 2
    assert S.common_prefix_length([[1, 2], [3]]) == 0
    with pytest.raises(ValueError):
        S.common_prefix_length([])


def test_position_masks_split_attended_tokens():
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    labels = torch.tensor([[-100, -100, -100, 5, 6, -100], [-100, -100, -100, -100, 7, 8]])
    m = S.position_masks(attention, labels, 2)
    assert m["prefix"].tolist() == [[True, True, False, False, False, False]] * 2
    assert m["answer"].tolist() == [[False, False, False, True, True, False], [False] * 4 + [True, True]]
    assert m["prompt"].tolist() == [[False, False, True, False, False, False], [False, False, True, True, False, False]]
    assert torch.equal(m["nonprefix"], m["prompt"] | m["answer"])
    assert not (m["prefix"] & m["nonprefix"]).any()
    with pytest.raises(ValueError):  # answer inside the prefix
        S.position_masks(attention, labels, 4)
    with pytest.raises(ValueError):
        S.position_masks(attention, labels, 7)
    # the collator masks in-sequence EOS tokens (here positions 1 and 5): the prefix keeps its positions, a masked
    # answer token stays an answer and a non-prefix token, a masked prompt token is in no class
    m = S.position_masks(torch.tensor([[1, 0, 1, 0, 1, 0]]), torch.tensor([[-100, -100, -100, -100, 5, 6]]), 2)
    assert m["prefix"].tolist() == [[True, True, False, False, False, False]]
    assert m["nonprefix"].tolist() == [[False, False, True, False, True, True]]
    assert m["prompt"].tolist() == [[False, False, True, False, False, False]]


class _Toy(nn.Module):
    """Embedding, then two linear targets; no mixing across positions, so equal prefix tokens give equal inputs."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.emb = nn.Embedding(12, 4)
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(3, 2)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen))

    def forward(self, input_ids, attention_mask):
        return self.b(torch.relu(self.a(self.emb(input_ids))))


def test_collect_position_statistics_matches_manual_covariances():
    model = _Toy()
    batches = [{"input_ids": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
                "labels": torch.tensor([[-100, -100, -100, 4, 5], [-100, -100, -100, 7, -100]])},
               {"input_ids": torch.tensor([[1, 2, 8, 9, 10]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                "labels": torch.tensor([[-100, -100, -100, -100, 10]])}]
    targets = {"a": model.a, "b": model.b}
    model.train()
    covs, counts, prefix, dev = S.collect_position_statistics(model, batches, targets, lambda b: b, 2,
                                                              ("answer", "nonprefix", "prompt"))
    assert model.training
    assert counts == {"answer": 4, "nonprefix": 8, "prompt": 4}
    with torch.no_grad():
        for name, inp in (("a", lambda ids: model.emb(ids)), ("b", lambda ids: torch.relu(model.a(model.emb(ids))))):
            for cls in ("answer", "nonprefix", "prompt"):
                rows = []
                for b in batches:
                    m = S.position_masks(b["attention_mask"], b["labels"], 2)[cls]
                    rows.append(inp(b["input_ids"])[m])
                x = torch.cat(rows).double()
                assert torch.allclose(covs[name][cls].double(), x.T @ x / x.shape[0], atol=1e-5)
            assert torch.allclose(prefix[name], inp(torch.tensor([[1, 2]]))[0], atol=1e-6)
            assert dev[name] < 1e-6


def test_protected_span_caps_the_retain_share_and_holds_the_prefix():
    gen = torch.Generator().manual_seed(1)
    n = 12
    x = torch.randn(400, n, generator=gen, dtype=torch.float64) * torch.linspace(0.2, 3.0, n, dtype=torch.float64)
    cr = x.T @ x / 400
    prefix = torch.randn(3, n, generator=gen, dtype=torch.float64)
    for kappa in (0.5, 1.0, 2.0):
        b, info = S.protected_span(n, cr, kappa, prefix)
        assert torch.allclose(b.T @ b, torch.eye(b.shape[1], dtype=b.dtype), atol=1e-10)
        assert info["prefix_rank"] == 3 and info["protected"] == b.shape[1]
        assert torch.allclose(b @ (b.T @ prefix.T), prefix.T, atol=1e-10)  # the prefix inputs lie in the span
        r = cr / torch.trace(cr)
        lam = torch.linalg.eigvalsh(r)
        assert info["kappa_directions"] == int((lam > kappa / n).sum())
        proj = torch.eye(n, dtype=torch.float64) - b @ b.T
        top = torch.linalg.eigvalsh(proj @ r @ proj).max()
        assert float(top) <= kappa / n + 1e-12  # every unit vector outside the span: v^T R v <= kappa / n
        b2, info2 = S.protected_span(n, None, kappa, prefix, spectrum=S.retain_spectrum(cr))
        assert info2 == info and torch.allclose(b2 @ b2.T, b @ b.T, atol=1e-10)
    b, info = S.protected_span(n)
    assert b.shape == (n, 0) and info["protected"] == 0
    with pytest.raises(ValueError):
        S.protected_span(n, None, 1.0)


def test_selective_directions_without_protection_match_ordered_directions():
    gen = torch.Generator().manual_seed(2)
    n = 9
    xf = torch.randn(300, n, generator=gen, dtype=torch.float64) * torch.linspace(0.3, 2.0, n, dtype=torch.float64)
    xr = torch.randn(300, n, generator=gen, dtype=torch.float64)
    cf, cr = xf.T @ xf / 300, xr.T @ xr / 300
    ref, ref_lam = F.ordered_directions(cf, cr, 1.0, return_values=True)
    v, lam = S.selective_directions(cf, cr, torch.zeros(n, 0, dtype=torch.float64), 1.0)
    assert torch.allclose(lam, ref_lam, atol=1e-10)
    assert torch.allclose(v, ref, atol=1e-8)
    star = float(lam[3])
    v4, _ = S.selective_directions(cf, cr, torch.zeros(n, 0, dtype=torch.float64), 1.0, star)
    assert v4.shape[1] == int((lam >= star).sum()) and torch.allclose(v4, ref[:, :v4.shape[1]], atol=1e-8)
    v0, _ = S.selective_directions(cf, cr, torch.zeros(n, 0, dtype=torch.float64), 1.0, float(lam[0]) * 2)
    assert v0.shape == (n, 0)


def test_protection_drops_a_shared_strong_direction_and_keeps_a_weak_specific_one():
    # axis 0: forget 100, retain 25 (strong in both); axis 1: forget 4, retain 1; the same trace-normalised ratio
    # (lambda 1.16 each), both above lambda* 1, the other axes at 0.29.
    n = 8
    cf = _diag([100.0, 4.0] + [1.0] * (n - 2))
    cr = _diag([25.0, 1.0] + [1.0] * (n - 2))
    none = torch.zeros(n, 0, dtype=torch.float64)
    v, lam = S.selective_directions(cf, cr, none, 1e-3, 1.0)
    assert {int(i) for i in v.abs().argmax(0)} == {0, 1}
    b, info = S.protected_span(n, cr, 4.0)  # retain share of axis 0 is 25 / 32 > 4 / 8, of axis 1 1 / 32
    assert info["kappa_directions"] == 1
    v, lam = S.selective_directions(cf, cr, b, 1e-3, 1.0)
    assert v.shape[1] == 1 and int(v[:, 0].abs().argmax()) == 1
    assert float((b.T @ v).abs().max()) < 1e-12
    assert float(lam[0]) == pytest.approx(float(cf[1, 1] / torch.trace(cf)) / (float(cr[1, 1] / torch.trace(cr)) + 1e-3 / n),
                                          rel=1e-9)


def test_selective_directions_are_orthonormal_and_outside_the_protected_span():
    gen = torch.Generator().manual_seed(3)
    n = 16
    xf = torch.randn(500, n, generator=gen, dtype=torch.float64) * torch.linspace(0.2, 2.5, n, dtype=torch.float64)
    xr = torch.randn(500, n, generator=gen, dtype=torch.float64) * torch.linspace(1.5, 0.5, n, dtype=torch.float64)
    cf, cr = xf.T @ xf / 500, xr.T @ xr / 500
    b, _ = S.protected_span(n, cr, 1.5, torch.randn(2, n, generator=gen, dtype=torch.float64))
    v, lam = S.selective_directions(cf, cr, b, 1.0, 0.5)
    assert v.shape[1] == int((lam >= 0.5).sum()) and v.shape[1] > 0
    assert torch.allclose(v.T @ v, torch.eye(v.shape[1], dtype=v.dtype), atol=1e-10)
    assert float((b.T @ v).abs().max()) < 1e-10
    assert int((lam.abs() < 1e-10).sum()) >= b.shape[1]  # the protected span has lambda 0
    # each orthonormalised column is no eigenvector, but lies in the span of the eigenvectors with lambda >= lambda*,
    # where the pencil's Rayleigh quotient is a weighted mean of their lambdas: every column has a quotient >= lambda*
    proj = torch.eye(n, dtype=torch.float64) - b @ b.T
    a_p = proj @ (cf / torch.trace(cf)) @ proj
    b_p = proj @ (cr / torch.trace(cr)) @ proj + 1.0 / n * torch.eye(n, dtype=torch.float64)
    quotient = (v * (a_p @ v)).sum(0) / (v * (b_p @ v)).sum(0)
    assert float(quotient.min()) >= 0.5 - 1e-10 and float(quotient.max()) <= float(lam[0]) + 1e-10
    with pytest.raises(ValueError):
        S.selective_directions(cf, cr, b, 0.0)


def _plain_case(seed, out=6, inn=9):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=gen)
    gf = torch.randn(out, 3, generator=gen) @ torch.randn(3, inn, generator=gen) + 0.05 * torch.randn(out, inn, generator=gen)
    gr = torch.randn(out, inn, generator=gen)
    return w, gf, gr, gen


def test_plain_restricted_update_without_protection_is_the_contract_warm_start():
    import recast_core as R
    w, gf, gr, _ = _plain_case(5)
    delta, rep = S.plain_restricted_update(w, gf, gr, 0.9, 0.3)
    basis = R.svd_basis(gf, 0.9)
    s = R.shrink_factors(R.row_alignment(w, gr), 0.3)
    ref = R.project((s - 1.0).unsqueeze(1) * w, basis.u, basis.vh)
    assert torch.allclose(delta, ref, atol=1e-6)
    assert rep["rank"] == basis.rank and rep["source_share_after_projection"] == pytest.approx(1.0)
    assert rep["weight_rel_change"] == pytest.approx(float(ref.norm() / w.norm()), rel=1e-5)
    empty = torch.zeros(9, 0, dtype=torch.float64)
    assert torch.allclose(S.plain_restricted_update(w, gf, gr, 0.9, 0.3, empty)[0], ref, atol=1e-6)


def test_plain_restricted_update_leaves_the_protected_inputs_unchanged():
    import recast_core as R
    w, gf, gr, gen = _plain_case(6)
    prefix = torch.randn(2, 9, generator=gen, dtype=torch.float64)
    gf = gf + 3.0 * torch.randn(6, 1, generator=gen) @ prefix[:1].float()  # forget-gradient energy along the prefix
    b, _ = S.protected_span(9, prefix_inputs=prefix)
    delta, rep = S.plain_restricted_update(w, gf, gr, 0.9, 0.3, b)
    assert float((delta.double() @ prefix.T).abs().max()) < 1e-5  # (W + delta) x = W x on the prefix inputs
    assert rep["source_share_after_projection"] < 1.0
    unprotected, _ = S.plain_restricted_update(w, gf, gr, 0.9, 0.3)
    assert float((unprotected.double() @ prefix.T).abs().max()) > 1e-2  # the positive control: it does move them
    proj = (gf.double() - (gf.double() @ b) @ b.T).float()
    assert rep["rank"] == R.svd_basis(proj, 0.9).rank


def _moments(per):
    """(E[g g^T], E[g^T g]) of per-example gradients (examples x out x in)."""
    return torch.einsum("eij,ekj->ik", per, per) / per.shape[0], torch.einsum("eji,ejk->ik", per, per) / per.shape[0]


def _warm_start_targets(seed, out=5, inn=7):
    gen = torch.Generator().manual_seed(seed)
    names = ["model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj"]
    targets = {n: nn.Linear(inn, out, bias=False) for n in names}
    with torch.no_grad():
        for m in targets.values():
            m.weight.copy_(torch.randn(out, inn, generator=gen))
    per = {n: torch.randn(4, out, inn, generator=gen, dtype=torch.float64) for n in names}
    gf = {n: per[n].mean(0).float() for n in names}
    gr = {n: torch.randn(out, inn, generator=gen) for n in names}
    return targets, per, gf, gr


def test_contrast_and_moment_updates_without_protection_are_the_warm_start():
    import recast_core as R
    for family in ("gcontrast", "gmoment"):
        targets, per, gf, gr = _warm_start_targets(11)
        w0 = {n: m.weight.detach().clone() for n, m in targets.items()}
        mom = {n: _moments(p) for n, p in per.items()}
        if family == "gcontrast":
            _, rep = R.warm_start(targets, gr, 0.3, gf, 0.8, whitening_rho=1.0, gradient_contrast="both",
                                  rank_multiplier=1.0)
        else:
            _, rep = R.warm_start(targets, gr, 0.3, gf, 0.8, gradient_moments=mom)
        for n, m in targets.items():
            if family == "gcontrast":
                delta, mine = S.contrast_restricted_update(w0[n], gf[n], gr[n], 0.8, 0.3, 1.0)
                assert mine["reference_rank"] == rep[n]["reference_rank"]
            else:
                delta, mine = S.moment_restricted_update(w0[n], gf[n], gr[n], mom[n], 0.8, 0.3)
            assert mine["rank"] == rep[n]["rank"] and mine["evr_of_source"] == pytest.approx(rep[n]["evr"], abs=1e-6)
            assert mine["source_share_after_projection"] == pytest.approx(1.0)
            assert torch.allclose(w0[n] + delta, m.weight.detach(), atol=1e-5)
            empty = torch.zeros(7, 0, dtype=torch.float64)
            again = (S.contrast_restricted_update(w0[n], gf[n], gr[n], 0.8, 0.3, 1.0, protected=empty)
                     if family == "gcontrast" else S.moment_restricted_update(w0[n], gf[n], gr[n], mom[n], 0.8, 0.3,
                                                                              protected=empty))
            assert torch.allclose(again[0], delta, atol=1e-6)


def test_contrast_and_moment_updates_leave_the_protected_inputs_unchanged():
    import recast_core as R
    targets, per, gf, gr = _warm_start_targets(12)
    gen = torch.Generator().manual_seed(13)
    prefix = torch.randn(2, 7, generator=gen, dtype=torch.float64)
    b, _ = S.protected_span(7, prefix_inputs=prefix)
    proj = torch.eye(7, dtype=torch.float64) - b @ b.T
    for n, m in targets.items():
        w = m.weight.detach()
        # forget-gradient energy along the prefix
        p = per[n] + 3.0 * torch.randn(4, 5, 1, generator=gen, dtype=torch.float64) * prefix[0]
        g = p.mean(0).float()
        r = gr[n] + 2.0 * torch.randn(5, 1, generator=gen) @ prefix[1:].float()
        delta, rep = S.contrast_restricted_update(w, g, r, 0.8, 0.3, 1.0, protected=b)
        assert float((delta.double() @ prefix.T).abs().max()) < 1e-5
        assert rep["rank"] == R.svd_basis(S._off_span(g, b), 0.8).rank and rep["source_share_after_projection"] < 1.0
        unprotected, _ = S.contrast_restricted_update(w, g, r, 0.8, 0.3, 1.0)
        assert float((unprotected.double() @ prefix.T).abs().max()) > 1e-2  # the positive control
        out_p = torch.einsum("eij,ekj->ik", p @ proj, p @ proj) / 4  # E[g P g^T]
        m_out, m_in = _moments(p)
        delta, rep = S.moment_restricted_update(w, g, r, (out_p, m_in), 0.8, 0.3, protected=b)
        assert float((delta.double() @ prefix.T).abs().max()) < 1e-5
        ref = R.gradient_moment_basis(S._off_span(g, b), out_p, proj @ m_in @ proj, energy_threshold=0.8)
        assert rep["rank"] == ref.rank and rep["rank_in"] == ref.rank_in
        unprotected, _ = S.moment_restricted_update(w, g, r, (m_out, m_in), 0.8, 0.3)
        assert float((unprotected.double() @ prefix.T).abs().max()) > 1e-2


class _SeqLoss(nn.Module):
    """Token sequences (batch, time, 4) through two linear targets; squared error per valid token, averaged over the
    batch's valid tokens like a causal LM's mean loss."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(14)
        self.a = nn.Linear(4, 6, bias=False)
        self.b = nn.Linear(6, 3, bias=False)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen))

    def terms(self, input_ids, attention_mask, labels):
        h = self.b(torch.tanh(self.a(input_ids)))
        return (((h - labels) ** 2).sum(-1) * attention_mask).sum(-1)

    def forward(self, input_ids, attention_mask, labels):
        from types import SimpleNamespace
        return SimpleNamespace(loss=self.terms(input_ids, attention_mask, labels).sum() / attention_mask.sum())


def test_collect_example_moments_match_the_core_and_the_projected_gradients():
    import recast_core as R
    gen = torch.Generator().manual_seed(15)
    m = _SeqLoss()
    m.train()
    targets = {"a": m.a, "b": m.b}
    x, y = torch.randn(5, 4, 4, generator=gen), torch.randn(5, 4, 3, generator=gen)
    mask = torch.ones(5, 4)
    mask[1, 3:] = 0
    mask[4, 2:] = 0
    x[mask == 0] = 50.0  # padding
    batches = [{"input_ids": x[i:j], "attention_mask": mask[i:j], "labels": y[i:j]} for i, j in ((0, 3), (3, 5))]
    spans = {"a": torch.linalg.qr(torch.randn(4, 1, generator=gen, dtype=torch.float64))[0],
             "b": torch.linalg.qr(torch.randn(6, 2, generator=gen, dtype=torch.float64))[0]}
    core, n_core = R.collect_example_gradient_moments(m, batches, targets, lambda b: b)
    got, n = S.collect_example_moments(m, batches, targets, lambda b: b, spans)
    assert n == n_core == 5
    weights = [targets[k].weight for k in targets]
    per = {k: [] for k in targets}
    for bt in batches:
        size = bt["input_ids"].shape[0]
        terms = m.terms(**bt) / bt["attention_mask"].sum()
        for e in range(size):
            for k, g in zip(targets, torch.autograd.grad(terms[e], weights, retain_graph=True)):
                per[k].append(size * g.double())
    for k in targets:
        assert torch.allclose(got[k][0], core[k][0], rtol=1e-9, atol=0) and torch.allclose(got[k][1], core[k][1],
                                                                                        rtol=1e-9, atol=0)
        gp = torch.stack(per[k]) @ (torch.eye(spans[k].shape[0], dtype=torch.float64) - spans[k] @ spans[k].T)
        assert torch.allclose(got[k][2], torch.einsum("eij,ekj->ik", gp, gp) / 5, rtol=1e-4, atol=1e-8)
    plain, _ = S.collect_example_moments(m, batches, targets, lambda b: b)
    assert all(v[2] is None for v in plain.values())
    assert m.training and all(p.requires_grad for p in m.parameters()) and all(p.grad is None for p in m.parameters())
    assert all(not mod._forward_hooks for mod in targets.values())


def test_input_shrink_scales_the_chosen_inputs_only():
    gen = torch.Generator().manual_seed(4)
    w = torch.randn(5, 7, generator=gen, dtype=torch.float64)
    v = torch.linalg.qr(torch.randn(7, 2, generator=gen, dtype=torch.float64))[0]
    out = S.input_shrink(w, v, 0.25)
    assert torch.allclose(out @ v, 0.25 * (w @ v), atol=1e-12)
    z = torch.randn(7, generator=gen, dtype=torch.float64)
    z = z - v @ (v.T @ z)
    assert torch.allclose(out @ z, w @ z, atol=1e-12)
    assert torch.allclose(S.input_shrink(w, v, 1.0), w)
    assert torch.allclose(S.input_shrink(w, torch.zeros(7, 0, dtype=torch.float64), 0.0), w)
    with pytest.raises(ValueError):
        S.input_shrink(w, v, 1.5)


def test_direction_report():
    n = 4
    cf, cr = _diag([4.0, 2.0, 1.0, 1.0]), _diag([1.0, 1.0, 1.0, 5.0])
    v = torch.eye(n, dtype=torch.float64)[:, :2]
    prefix = torch.tensor([[3.0, 0.0, 4.0, 0.0]], dtype=torch.float64)
    rep = S.direction_report(v, cf, cr, prefix)
    assert rep["directions"] == 2
    assert rep["forget_share"] == pytest.approx(6 / 8)
    assert rep["retain_share"] == pytest.approx(2 / 8)
    assert rep["max_retain_share_x_n"] == pytest.approx(1 / 8 * 4)
    assert rep["prefix_share"] == pytest.approx(9 / 25)
    empty = S.direction_report(torch.zeros(n, 0, dtype=torch.float64), cf, cr)
    assert empty["directions"] == 0 and empty["prefix_share"] is None


def test_guard_passes():
    assert S.guard_passes({"d_forget": 0.2, "d_retain": 0.1}, 2.0)
    assert not S.guard_passes({"d_forget": 0.19, "d_retain": 0.1}, 2.0)
    assert S.guard_passes({"d_forget": 0.01, "d_retain": -0.1}, 2.0)
    assert not S.guard_passes({"d_forget": 0.0, "d_retain": -0.1}, 2.0)
    assert not S.guard_passes({"d_forget": -0.1, "d_retain": -0.2}, 0.5)


def _toy_measure(effects, interaction=0.0):
    """{name: V} -> the joint change: per target the summed (forget, retain) effects of its first k columns (k from
    V's shape), the forget total scaled by 1 + interaction * (edited targets - 1); and the list of calls."""
    calls = []

    def measure(sel):
        calls.append({n: int(v.shape[1]) for n, v in sel.items()})
        f = sum(sum(e[0] for e in effects[n][:int(v.shape[1])]) for n, v in sel.items())
        r = sum(sum(e[1] for e in effects[n][:int(v.shape[1])]) for n, v in sel.items())
        edited = sum(1 for v in sel.values() if v.shape[1] > 0)
        return {"d_forget": f * (1.0 + interaction * max(edited - 1, 0)), "d_retain": r}
    return measure, calls


def _columns(counts):
    return {n: torch.eye(8, dtype=torch.float64)[:, :k] for n, k in counts.items()}


def test_leave_one_out_contributions_without_the_guard():
    effects = {"a": [(1.0, 0.1)] * 3, "b": [(0.5, 0.5)] * 2, "c": []}
    chosen = _columns({"a": 3, "b": 2, "c": 0})
    measure, calls = _toy_measure(effects)
    kept, rep = S.leave_one_out(chosen, measure)
    assert kept == {"a": 3, "b": 2, "c": 0}
    assert rep["full"]["d_forget"] == pytest.approx(4.0) and rep["full"]["d_retain"] == pytest.approx(1.3)
    # additive measure: the contribution is the target's own effect
    a, b = rep["targets"]["a"]["trials"], rep["targets"]["b"]["trials"]
    assert len(a) == len(b) == 1 and rep["targets"]["c"]["trials"] == []
    assert a[0]["d_forget"] == pytest.approx(3.0) and a[0]["d_retain"] == pytest.approx(0.3)
    assert b[0]["ratio"] == pytest.approx(1.0) and b[0]["directions"] == 2
    assert rep["targets"]["a"]["without"]["d_forget"] == pytest.approx(1.0)
    assert len(calls) == 3  # the full edit, then a and b left out; nothing for the empty target
    assert calls[1] == {"b": 2, "c": 0} and calls[2] == {"a": 3, "c": 0}
    # superadditive forget change: the contribution exceeds the edit alone (3.0)
    measure, _ = _toy_measure(effects, interaction=0.5)
    _, rep = S.leave_one_out(chosen, measure)
    assert rep["targets"]["a"]["trials"][0]["d_forget"] == pytest.approx(4.0 * 1.5 - 1.0)


def test_leave_one_out_guard_halves_or_drops_against_the_same_others():
    effects = {"a": [(1.0, 0.1)] * 4, "b": [(1.0, 0.1)] * 2 + [(0.1, 1.0)] * 2, "c": [(0.1, 1.0)] * 4}
    chosen = _columns({"a": 4, "b": 4, "c": 4})
    measure, calls = _toy_measure(effects)
    kept, rep = S.leave_one_out(chosen, measure, ratio=2.0, steps=4)
    assert kept == {"a": 4, "b": 2, "c": 0}
    b = rep["targets"]["b"]["trials"]
    assert [t["directions"] for t in b] == [4, 2]
    assert b[0]["ratio"] == pytest.approx(1.0) and b[1]["ratio"] == pytest.approx(10.0)
    c = rep["targets"]["c"]["trials"]
    assert [t["directions"] for t in c] == [4, 2, 1]  # 1 // 2 == 0: dropped
    assert {"a": 4, "b": 2, "c": 4} in calls and {"a": 4, "b": 4, "c": 1} in calls  # halved against the others as chosen
    kept, rep = S.leave_one_out(chosen, measure, ratio=2.0, steps=1)
    assert kept["c"] == 0 and [t["directions"] for t in rep["targets"]["c"]["trials"]] == [4, 2]


def test_nonprefix_labels_cover_question_and_answer_tokens_only():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 9], [1, 2, 7, 8, 9, 9, 9]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 6, -100], [-100, -100, -100, 8, -100, -100, -100]])
    out = S.nonprefix_labels(ids, attention, labels, 2)
    assert out.tolist() == [[-100, -100, 3, 4, 5, 6, -100], [-100, -100, 7, 8, -100, -100, -100]]
    assert torch.equal(out[labels != -100], labels[labels != -100])  # the answer labels are kept
