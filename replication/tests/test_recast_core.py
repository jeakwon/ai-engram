"""CPU parity tests for replication/scripts/recast_core.py (RECAST_PORTING_CONTRACT.md, section
"Required parity tests for every port"). Run on a compute node: they import torch."""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import recast_core as R  # noqa: E402


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)


class Toy(nn.Module):
    """Mimics the HF layout: targets under model.layers.*, an excluded lm_head."""

    def __init__(self, d=6, n=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.model.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, d, bias=False)

    def forward(self, input_ids, labels, attention_mask=None):
        h = input_ids
        for blk in self.model.layers:
            h = blk.down_proj(torch.tanh(blk.q_proj(h)))
        out = self.lm_head(self.model.norm(h))
        return SimpleNamespace(loss=((out - labels) ** 2).mean(dim=1).mean())  # mean over examples


def loader(x, y, sizes):
    i = 0
    for s in sizes:
        yield {"input_ids": x[i:i + s], "labels": y[i:i + s]}
        i += s


def test_target_profiles():
    m = Toy()
    blocks = lambda ps: {f"model.layers.{i}.{p}" for i in range(2) for p in ps}  # noqa: E731
    assert set(R.select_targets(m, "decoder_linear")) == blocks(("q_proj", "down_proj"))
    assert set(R.select_targets(m, "mlp")) == blocks(("down_proj",))
    assert set(R.select_targets(m, "mlp_head")) == blocks(("down_proj",)) | {"lm_head"}
    assert set(R.select_targets(m, "mlp_down")) == blocks(("down_proj",))
    assert set(R.select_targets(m, "mlp_gate_down")) == blocks(("down_proj",))  # the toy has no gate_proj
    assert set(R.select_targets(m, "down_up_q")) == blocks(("down_proj", "q_proj"))  # the toy has no up_proj
    assert set(R.select_targets(m, "mlp_o")) == blocks(("down_proj",))  # the toy has no gate/up/o_proj
    with pytest.raises(ValueError):  # the toy has neither up_proj nor gate_proj
        R.select_targets(m, "mlp_up_gate")
    with pytest.raises(ValueError):  # the toy has no up_proj
        R.select_targets(m, "mlp_up")
    assert set(R.select_targets(m, "down_o")) == blocks(("down_proj",))  # the toy has no o_proj
    with pytest.raises(ValueError):  # the toy has neither up_proj nor o_proj
        R.select_targets(m, "up_o")
    assert R.TARGET_PROFILES["down_o"] == (("down_proj", "o_proj"), False)
    assert R.TARGET_PROFILES["up_o"] == (("up_proj", "o_proj"), False)


def test_mean_gradient_is_example_weighted_and_state_restored():
    torch.manual_seed(0)
    m = Toy()
    m.train()
    m.model.norm.weight.requires_grad_(False)
    x, y = torch.randn(7, 6), torch.randn(7, 6)
    targets = R.select_targets(m, "decoder_linear")
    got, n = R.collect_mean_gradients(m, loader(x, y, [3, 3, 1]), targets, lambda b: b)
    assert n == 7
    weights = [targets[k].weight for k in targets]
    full = torch.autograd.grad(m(x, y).loss, weights)  # one batch = exact dataset mean
    for k, g in zip(targets, full):
        assert torch.allclose(got[k], g, atol=1e-6)
    assert m.training and not m.model.norm.weight.requires_grad and m.lm_head.weight.requires_grad
    assert all(p.grad is None for p in m.parameters())


def test_row_alignment_and_fixed_floor():
    w = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    g = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    a = R.row_alignment(w, g)
    assert torch.allclose(a, torch.tensor([1.0, -1.0, 0.0]), atol=1e-6)
    assert torch.allclose(R.shrink_factors(a, 0.1), torch.tensor([1.0, 0.1, 0.1]))
    with pytest.raises(ValueError):
        R.shrink_factors(a, 1.5)


def test_svd_rank_follows_evr_and_zero_energy():
    rank1 = torch.outer(torch.arange(1.0, 5.0), torch.arange(1.0, 4.0))
    assert R.svd_basis(rank1, 0.98).rank == 1
    diag = torch.diag(torch.tensor([3.0, 2.0, 1.0]))  # energies 9, 4, 1 -> EVR .643, .929, 1
    assert R.svd_basis(diag, 0.90).rank == 2
    assert R.svd_basis(diag, 0.93).rank == 3
    assert R.svd_basis(diag, 1.0).rank == 3
    zero = R.svd_basis(torch.zeros(4, 3), 0.98)
    assert not zero.active and zero.reason == "zero_energy"


def test_two_sided_projection_matches_direct_formula():
    torch.manual_seed(1)
    b = R.svd_basis(torch.randn(5, 2) @ torch.randn(2, 4), 0.999)
    d = torch.randn(5, 4)
    direct = b.u @ b.u.T @ d @ b.vh.T @ b.vh
    p = R.project(d, b.u, b.vh)
    assert torch.allclose(p, direct, atol=1e-6)
    assert torch.allclose(R.project(p, b.u, b.vh), p, atol=1e-6)  # idempotent


def test_unrestricted_warm_start_scales_rows_of_targets_only():
    torch.manual_seed(2)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    head = m.lm_head.weight.detach().clone()
    g = {k: torch.randn_like(v.weight) for k, v in t.items()}
    restr, rep = R.warm_start(t, g, 0.05)
    assert restr == {}
    for k, v in t.items():
        s = R.shrink_factors(R.row_alignment(before[k], g[k]), 0.05)
        assert torch.allclose(v.weight, s[:, None] * before[k], atol=1e-6)
        # regression (pilot 24955489): the report compared the float32 weight with itself
        expected = (v.weight.detach() - before[k]).norm() / before[k].norm()
        assert rep[k]["weight_rel_change"] == pytest.approx(float(expected), rel=1e-5) and expected > 0
    assert torch.equal(m.lm_head.weight, head)


def test_floor_one_is_exact_noop():
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    R.warm_start(t, {k: torch.randn_like(v.weight) for k, v in t.items()}, 1.0)
    assert all(torch.equal(t[k].weight, before[k]) for k in t)


def test_restricted_warm_start_delta_lies_in_subspace_and_zero_energy_skips():
    torch.manual_seed(3)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 2) @ torch.randn(2, 6) for k in t}
    zero_key = "model.layers.1.down_proj"
    gf[zero_key] = torch.zeros(6, 6)
    restr, rep = R.warm_start(t, gr, 0.05, gf, 0.98)
    for k, r in restr.items():
        delta = t[k].weight.detach() - before[k]
        if k == zero_key:
            assert not r.basis.active
            s = R.shrink_factors(R.row_alignment(before[k], gr[k]), 0.05)
            assert torch.allclose(t[k].weight, s[:, None] * before[k], atol=1e-6)  # skip_projection
        else:
            assert r.basis.active and r.basis.rank <= 2
            assert torch.allclose(R.project(delta, r.basis.u, r.basis.vh), delta, atol=1e-5)
            assert torch.equal(r.baseline, t[k].weight.detach())


def test_semu_adapter_starts_at_baseline_and_training_stays_in_subspace():
    torch.manual_seed(4)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, _ = R.warm_start(t, gr, 0.05, gf, 0.98)
    rep = R.apply_semu_adapters(m, restr)
    assert rep["adapters"] == 4 and rep["max_rel_error_initial_weight"] < 1e-5
    mods = dict(m.named_modules())
    for p in m.parameters():
        p.requires_grad_(False)
    cores = [mods[k].weight for k in restr]
    for c in cores:
        c.requires_grad_(True)
    checks = R.frozen_checksums(m)
    snap = R.snapshot_trainable(m)
    opt = torch.optim.AdamW(cores, lr=1e-2, weight_decay=0.01)
    x, y = torch.randn(8, 6), torch.randn(8, 6)
    for _ in range(5):
        opt.zero_grad()
        m(x, y).loss.backward()
        opt.step()
    for k, r in restr.items():
        d = mods[k].effective_weight().detach() - r.baseline
        assert d.norm() > 0
        assert torch.allclose(R.project(d, r.basis.u, r.basis.vh), d, atol=1e-5)
    report = R.change_report(m, snap, checks)
    assert report["frozen_tensors_moved_count"] == 0 and report["trainable_entries_changed_fraction"] > 0.9


def test_change_report_flags_a_moved_frozen_tensor():
    m = Toy()
    for p in m.parameters():
        p.requires_grad_(False)
    checks, snap = R.frozen_checksums(m), R.snapshot_trainable(m)
    with torch.no_grad():
        m.lm_head.weight.add_(1.0)
    assert R.change_report(m, snap, checks)["frozen_tensors_moved"] == ["lm_head.weight"]



class TiedToy(Toy):
    """Toy whose head shares the input-embedding matrix, as Llama-3.2-1B."""

    def __init__(self, d=6, vocab=6):
        super().__init__(d)
        self.embed = nn.Embedding(vocab, d)
        self.lm_head.weight = self.embed.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


def test_untie_gives_the_head_an_equal_separate_copy():
    m = TiedToy()
    rep = R.untie_output_embedding(m)
    assert rep == {"was_tied": True, "tied_after": False, "equal_values": True}
    assert m.config.tie_word_embeddings is False
    emb = m.embed.weight.detach().clone()
    with torch.no_grad():
        m.lm_head.weight.mul_(0.5)
    assert torch.equal(m.embed.weight, emb)


def test_restricted_head_outside_restriction_gets_raw_shrink_and_no_adapter():
    torch.manual_seed(5)
    m = Toy()
    t = R.select_targets(m, "mlp_head")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 2) @ torch.randn(2, 6) for k in t if k != "lm_head"}
    restr, rep = R.warm_start(t, gr, 0.1, gf, 0.97, unrestricted={"lm_head"})
    s = R.shrink_factors(R.row_alignment(before["lm_head"], gr["lm_head"]), 0.1)
    assert torch.allclose(m.lm_head.weight, s[:, None] * before["lm_head"], atol=1e-6)
    assert not restr["lm_head"].basis.active and rep["lm_head"]["reason"] == "parameter_unrestricted"
    out = R.apply_semu_adapters(m, restr)
    assert out["adapters"] == 2 and type(m.lm_head) is nn.Linear


def test_piecewise_schedule_points():
    pts = R.parse_points("0:0,1:3e-5,5:1e-5,10:1e-5")
    assert pts == [(0.0, 0.0), (1.0, 3e-5), (5.0, 1e-5), (10.0, 1e-5)]
    f = lambda x: R.piecewise_linear(pts, x)  # noqa: E731
    assert f(0.0) == 0.0 and f(0.5) == pytest.approx(1.5e-5) and f(1.0) == pytest.approx(3e-5)
    assert f(3.0) == pytest.approx(2e-5) and f(5.0) == pytest.approx(1e-5)
    assert f(7.5) == pytest.approx(1e-5) and f(12.0) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        R.parse_points("0:0,0:1")


def test_drift_report_norm_ratio_and_cosine():
    m = Toy()
    t = R.select_targets(m, "mlp")  # down_proj in the toy
    ref = {k: v.weight.detach().clone() for k, v in t.items()}
    with torch.no_grad():
        for v in t.values():
            v.weight.mul_(0.5)
    rep = R.drift_report(m, ref, ("down_proj",))
    assert rep["down_proj"]["norm_ratio"] == pytest.approx(0.5)
    assert rep["down_proj"]["cosine"] == pytest.approx(1.0)


def test_target_blocks_filter():
    m = Toy(n=3)
    assert R.parse_blocks("all") is None and R.parse_blocks("1-2") == {1, 2} and R.parse_blocks("0,2") == {0, 2}
    got = set(R.select_targets(m, "decoder_linear", blocks=R.parse_blocks("1-2")))
    assert got == {f"model.layers.{i}.{p}" for i in (1, 2) for p in ("q_proj", "down_proj")}


def test_full_data_source_is_example_weighted_mean():
    gf, gr = torch.ones(2, 2), torch.zeros(2, 2)
    assert torch.allclose(R.mix_sources(gf, gr, 1, 3), torch.full((2, 2), 0.25))
    with pytest.raises(ValueError):
        R.mix_sources(gf, gr, 0, 3)


def test_subspace_alignment_matches_classification_formula():
    torch.manual_seed(6)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    src = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, _ = R.warm_start(t, gr, 0.2, src, 0.98, alignment_scope="restriction_subspace")
    for k, r in restr.items():
        pw = R.project(before[k], r.basis.u, r.basis.vh)
        s = R.shrink_factors(R.row_alignment(pw, R.project(gr[k], r.basis.u, r.basis.vh)), 0.2)
        expected = before[k] + R.project((s - 1.0)[:, None] * pw, r.basis.u, r.basis.vh)
        assert torch.allclose(t[k].weight, expected, atol=1e-5)


def test_subspace_norm_layer_floors():
    torch.manual_seed(7)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: v.weight.detach().clone() for k, v in t.items()}  # cos(W, -W) = -1: every row at its floor
    src = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, rep = R.warm_start(t, gr, 0.3, src, 0.98, layer_scaling="subspace_norm")
    rel = {k: rep[k]["subspace_rel_norm"] for k in t}
    top = max(rel.values())
    for k, r in restr.items():
        floor = 1.0 - 0.7 * rel[k] / top
        assert rep[k]["floor"] == pytest.approx(floor)
        expected = before[k] + (floor - 1.0) * R.project(before[k], r.basis.u, r.basis.vh)
        assert torch.allclose(t[k].weight, expected, atol=1e-5)
    assert min(rep[k]["floor"] for k in t) == pytest.approx(0.3)
    with pytest.raises(ValueError):  # scaling needs a restriction
        R.warm_start(R.select_targets(Toy(), "mlp"), {k: torch.zeros(6, 6) for k in R.select_targets(Toy(), "mlp")},
                     0.3, layer_scaling="subspace_norm")


def test_subspace_norm_mean_keeps_average_shrink():
    torch.manual_seed(8)
    m = Toy(n=4)
    t = R.select_targets(m, "decoder_linear")
    gr = {k: v.weight.detach().clone() for k, v in t.items()}  # every row at its floor
    src = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.98, layer_scaling="subspace_norm_mean")
    floors = [rep[k]["floor"] for k in t]
    assert all(0.0 <= f <= 1.0 for f in floors)
    if all(0.0 < f < 1.0 for f in floors):  # no clipping: the mean shrink amount is exactly 1 - s_min
        assert sum(1.0 - f for f in floors) / len(floors) == pytest.approx(0.7)
    rel = {k: rep[k]["subspace_rel_norm"] for k in t}
    mean_r = sum(rel.values()) / len(rel)
    for k in t:
        assert rep[k]["floor"] == pytest.approx(min(1.0, max(0.0, 1.0 - 0.7 * rel[k] / mean_r)))


def test_free_parameters_select_groups_and_skip_adapters():
    torch.manual_seed(9)
    m = Toy()
    t = R.select_targets(m, "mlp_down")  # down_proj gets an adapter, q_proj stays nn.Linear
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, _ = R.warm_start(t, gr, 0.3, gf, 0.98)
    R.apply_semu_adapters(m, restr)
    assert R.free_parameters(m, "none") == {}
    norms = R.free_parameters(m, "norms")
    assert set(norms) == {"norms"} and set(norms["norms"]) == {"model.norm.weight", "model.norm.bias"}
    assert norms["norms"]["model.norm.weight"] is m.model.norm.weight
    assert set(R.free_parameters(m, "attention")["attention"]) == {f"model.layers.{i}.q_proj.weight" for i in range(2)}
    every = R.free_parameters(m, "all")
    assert set(every) == {"norms", "attention"}  # the only MLP projection carries an adapter
    assert set(R.free_parameters(m, "norms,attention")) == {"norms", "attention"}
    with pytest.raises(ValueError):  # a named group with nothing to train
        R.free_parameters(m, "mlp")
    with pytest.raises(ValueError):
        R.free_parameters(m, "heads")


def test_piecewise_schedule_step():
    pts = R.parse_points("0:0,1:3e-5,5:3e-5,5:1e-5,10:1e-5")
    f = lambda x: R.piecewise_linear(pts, x)  # noqa: E731
    assert f(0.0) == 0.0 and f(3.0) == pytest.approx(3e-5) and f(4.999) == pytest.approx(3e-5)
    assert f(5.0) == pytest.approx(1e-5) and f(7.0) == pytest.approx(1e-5) and f(10.0) == pytest.approx(1e-5)
    with pytest.raises(ValueError):  # an x three times
        R.parse_points("0:0,5:1,5:2,5:3,10:1")
    with pytest.raises(ValueError):  # decreasing x
        R.parse_points("0:0,5:1,4:1")


class SeqToy(nn.Module):
    """Token sequences (batch, time, d) through one block's down_proj."""

    def __init__(self, d=6):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d)])

    def forward(self, input_ids, attention_mask=None):
        return self.model.layers[0].down_proj(input_ids)


def test_retain_input_covariance_is_answer_token_masked():
    torch.manual_seed(10)
    m = SeqToy()
    t = R.select_targets(m, "mlp_down")
    x = torch.randn(3, 5, 6)
    labels = torch.full((3, 5), -100)
    labels[:2, 2:] = 1  # answer tokens: positions 2-4 of the first two sequences
    batches = [{"input_ids": x[:2], "attention_mask": torch.ones(2, 5), "labels": labels[:2]},
               {"input_ids": x[2:], "attention_mask": torch.ones(1, 5), "labels": labels[2:]}]
    cov, n = R.collect_input_covariances(m, batches, t, lambda b: b)
    sel = x[labels != -100]
    assert n == sel.shape[0] == 6
    assert torch.allclose(cov["model.layers.0.down_proj"], sel.T @ sel / 6, atol=1e-5)
    assert m.training  # mode restored


def test_ridge_input_filter_weights_eigen_directions():
    lam = torch.tensor([4.0, 1.0, 0.0, 0.0])
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(11)))
    cov = q @ torch.diag(lam) @ q.T
    m, mu = R.ridge_input_filter(cov, 1.0)
    assert mu == pytest.approx(5.0 / 4)
    assert torch.allclose(m, q @ torch.diag(mu / (lam + mu)) @ q.T, atol=1e-5)
    big, _ = R.ridge_input_filter(cov, 1e6)  # kappa -> inf: no filter
    assert torch.allclose(big, torch.eye(4), atol=1e-5)
    with pytest.raises(ValueError):
        R.ridge_input_filter(cov, 0.0)


def test_restricted_warm_start_input_filter_right_multiplies_delta():
    torch.manual_seed(12)
    m1, m2 = Toy(), Toy()
    m2.load_state_dict(m1.state_dict())
    t1, t2 = R.select_targets(m1, "decoder_linear"), R.select_targets(m2, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t1.items()}
    gr = {k: torch.randn(6, 6) for k in t1}
    gf = {k: torch.randn(6, 2) @ torch.randn(2, 6) for k in t1}
    filters = {k: R.ridge_input_filter(torch.diag(torch.rand(6) * 3), 1.0)[0] for k in t1}
    R.warm_start(t1, gr, 0.3, gf, 0.98)
    _, rep = R.warm_start(t2, gr, 0.3, gf, 0.98, input_filters=filters)
    for k in t1:
        d1 = t1[k].weight.detach() - before[k]
        d2 = t2[k].weight.detach() - before[k]
        assert torch.allclose(d2, d1 @ filters[k], atol=1e-5)
        assert 0.0 < rep[k]["retain_filter_kept"] <= 1.0 + 1e-6
    with pytest.raises(ValueError):  # a filter needs a restriction
        R.warm_start(R.select_targets(Toy(), "mlp_down"), {k: torch.zeros(6, 6) for k in R.select_targets(Toy(), "mlp_down")},
                     0.3, input_filters={})


def test_whitened_basis_with_isotropic_retain_equals_plain_basis():
    torch.manual_seed(13)
    g = torch.randn(5, 3) @ torch.randn(3, 6)
    plain = R.svd_basis(g, 0.9)
    white = R.whitened_svd_basis(g, torch.eye(6) * 2.0, 0.9, rho=0.5)
    assert white.rank == plain.rank
    assert torch.allclose(white.u @ white.u.T, plain.u @ plain.u.T, atol=1e-5)
    assert torch.allclose(white.vh.T @ white.vh, plain.vh.T @ plain.vh, atol=1e-5)
    assert torch.allclose(white.vh @ white.vh.T, torch.eye(white.rank), atol=1e-5)  # orthonormal covectors


def test_whitened_basis_prefers_directions_retain_rarely_uses():
    # Equal forget-gradient energy on inputs e0 and e1; retain uses e0 a hundred times more.
    u = torch.linalg.qr(torch.randn(3, 2, generator=torch.Generator().manual_seed(14)))[0]
    e = torch.eye(4)
    g = u[:, :1] @ e[:1] + u[:, 1:] @ e[1:2]
    cov = torch.diag(torch.tensor([100.0, 1.0, 1.0, 1.0]))
    assert R.svd_basis(g, 0.9).rank == 2
    white = R.whitened_svd_basis(g, cov, 0.9, rho=0.01)
    assert white.rank == 1
    assert torch.allclose(white.vh.abs(), e[1:2], atol=1e-4)
    t = R.select_targets(Toy(), "mlp_down")
    with pytest.raises(ValueError):  # whitening needs its rho
        R.warm_start(t, {k: torch.zeros(6, 6) for k in t}, 0.3, {k: torch.randn(6, 6) for k in t}, 0.9,
                     whitening={k: torch.eye(6) for k in t})


def test_explicit_rank_budget_matches_plain_and_whitened_bases():
    torch.manual_seed(15)
    g = torch.randn(6, 6)
    plain = R.svd_basis(g, 0.6)
    fixed = R.svd_basis(g, 0.6, rank=plain.rank + 2)
    assert fixed.rank == plain.rank + 2 and fixed.evr > plain.evr
    assert R.svd_basis(g, 0.6, rank=50).rank == 6  # clipped to the total rank
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    gr = {k: torch.randn(6, 6) for k in t}
    src = {k: torch.randn(6, 6) for k in t}
    cov = {k: torch.diag(torch.rand(6) + 0.1) for k in t}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.6, rank_multiplier=1.5)
    m2 = Toy()
    t2 = R.select_targets(m2, "decoder_linear")
    _, rep2 = R.warm_start(t2, gr, 0.3, src, 0.6, whitening=cov, whitening_rho=1.0, rank_multiplier=1.5)
    for k in t:
        r0 = R.svd_basis(src[k], 0.6).rank
        assert rep[k]["reference_rank"] == rep2[k]["reference_rank"] == r0
        assert rep[k]["rank"] == rep2[k]["rank"] == min(6, max(1, round(1.5 * r0)))
    with pytest.raises(ValueError):  # a budget needs a restriction
        R.warm_start(R.select_targets(Toy(), "mlp_down"), {k: torch.zeros(6, 6) for k in R.select_targets(Toy(), "mlp_down")},
                     0.3, rank_multiplier=1.5)


def test_one_sided_projection_and_adapter():
    torch.manual_seed(16)
    vh = torch.linalg.qr(torch.randn(6, 2))[0].T
    d = torch.randn(4, 6)
    assert torch.allclose(R.project(d, None, vh), d @ vh.T @ vh, atol=1e-6)
    basis = R.Basis(True, "", None, vh, 2, 4, 1.0)
    w_star = torch.randn(4, 6)
    core, anchor = R.adapter_core_and_anchor(w_star, basis)
    adapter = R.SemuAdapterLinear(6, 4, core, anchor, None, vh)
    assert torch.allclose(adapter.effective_weight(), w_star, atol=1e-5)
    opt = torch.optim.SGD([adapter.weight], lr=0.1)
    adapter(torch.randn(8, 6)).square().sum().backward()
    opt.step()
    moved = adapter.effective_weight().detach() - w_star
    assert moved.norm() > 0 and torch.allclose(moved @ vh.T @ vh, moved, atol=1e-5)  # rows stay in span(vh)


def test_representation_basis_picks_forget_heavy_inputs():
    e = torch.eye(4)
    cov_forget = torch.diag(torch.tensor([1.0, 5.0, 1.0, 0.1]))
    cov_retain = torch.diag(torch.tensor([5.0, 1.0, 1.0, 0.1]))
    g = torch.randn(3, 4, generator=torch.Generator().manual_seed(17))
    w = torch.randn(3, 4, generator=torch.Generator().manual_seed(18))
    for side in ("identity", "gradient", "representation"):
        b = R.representation_basis(g, cov_forget, cov_retain, w, rank=1, rho=0.01, output_side=side)
        assert b.active and b.rank == 1 and torch.allclose(b.vh.abs(), e[1:2], atol=1e-4)
        assert (b.u is None) == (side == "identity")
        if b.u is not None:
            assert torch.allclose(b.u.T @ b.u, torch.eye(b.u.shape[1]), atol=1e-5)
        assert 0.0 < b.evr <= 1.0 + 1e-6
    g1 = R.representation_basis(g, cov_forget, cov_retain, w, rank=1, rho=0.01, output_side="gradient")
    target = g @ e[1]
    assert torch.allclose((g1.u[:, 0] @ target).abs(), target.norm(), atol=1e-4)  # u along G_f v
    with pytest.raises(ValueError):
        R.representation_basis(g, cov_forget, cov_retain, w, rank=1, rho=0.01, output_side="both")


def test_gradient_contrast_basis_prefers_forget_only_directions():
    # G_f is largest along (e0, e0) but G_r pulls there too; (e1, e2) carries less forget energy and no retain.
    e_out, e_in = torch.eye(4, dtype=torch.float64), torch.eye(5, dtype=torch.float64)
    g_f = 3.0 * torch.outer(e_out[0], e_in[0]) + 1.0 * torch.outer(e_out[1], e_in[2]) + 0.1 * torch.outer(e_out[2], e_in[3])
    g_r = 5.0 * torch.outer(e_out[0], e_in[0]) + 0.1 * torch.outer(e_out[3], e_in[4])
    plain = R.svd_basis(g_f.float(), 0.5)
    assert plain.rank == 1 and torch.allclose(plain.vh.abs().double(), e_in[0:1], atol=1e-5)
    both = R.gradient_contrast_basis(g_f, g_r, rank=1, rho=0.1, sides="both")
    assert both.active and both.rank == 1 and both.total_rank == 4
    assert torch.allclose(both.vh.abs().double(), e_in[2:3], atol=1e-4)
    assert torch.allclose(both.u.abs().double(), e_out[:, 1:2], atol=1e-4)
    assert abs(both.evr - 1.0 / (9.0 + 1.0 + 0.01)) < 1e-4  # ||P(G_f)||^2 / ||G_f||^2
    inp = R.gradient_contrast_basis(g_f, g_r, rank=1, rho=0.1, sides="input")
    assert torch.allclose(inp.vh.abs().double(), e_in[2:3], atol=1e-4)
    assert torch.allclose(inp.u.abs().double(), e_out[:, 0:1], atol=1e-4)  # the plain basis's output side
    assert inp.evr < 1e-6  # u = e0 and v = e2 hold no forget gradient together
    for b in (R.gradient_contrast_basis(g_f, g_r, rank=3, rho=1.0, sides=s) for s in ("both", "input")):
        assert b.rank == 3 and torch.allclose(b.u.T @ b.u, torch.eye(3), atol=1e-5)
        assert torch.allclose(b.vh @ b.vh.T, torch.eye(3), atol=1e-5)
    assert R.gradient_contrast_basis(g_f, g_r, rank=9, rho=1.0, sides="both").rank == 4  # clipped to min(out, in)
    assert not R.gradient_contrast_basis(torch.zeros(4, 5), g_r, rank=1, rho=1.0, sides="both").active
    for bad in (dict(sides="output"), dict(sides="both", rho=0.0), dict(sides="both", rank=0)):
        with pytest.raises(ValueError):
            R.gradient_contrast_basis(g_f, g_r, **{"rank": 1, "rho": 1.0, **bad})


def test_warm_start_with_gradient_contrast_bases():
    torch.manual_seed(31)
    t = R.select_targets(Toy(), "decoder_linear")
    w0 = {k: m.weight.detach().clone() for k, m in t.items()}
    gr = {k: torch.randn(6, 6) for k in t}
    src = {k: torch.randn(6, 6) for k in t}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.6, whitening_rho=1.0, gradient_contrast="both", rank_multiplier=1.0)
    for k, m in t.items():
        r0 = R.svd_basis(src[k], 0.6).rank
        b = R.gradient_contrast_basis(src[k], gr[k], r0, 1.0, "both")
        assert rep[k]["reference_rank"] == r0 and rep[k]["rank"] == r0 and abs(rep[k]["evr"] - b.evr) < 1e-6
        proposal = (R.shrink_factors(R.row_alignment(w0[k], gr[k]), 0.3) - 1.0).unsqueeze(1) * w0[k]
        assert torch.allclose(m.weight.detach(), w0[k] + R.project(proposal, b.u, b.vh), atol=1e-5)
    for bad in (dict(whitening_rho=1.0, gradient_contrast="both", component_selection="full"),  # own-spectrum ranks
                dict(gradient_contrast="both", rank_multiplier=1.0),  # no rho
                dict(whitening_rho=1.0, gradient_contrast="output", rank_multiplier=1.0),
                dict(whitening_rho=1.0, gradient_contrast="both", rank_multiplier=1.0, representation_output="gradient")):
        with pytest.raises(ValueError):
            R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.6, **bad)
    with pytest.raises(ValueError):  # needs a restriction
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, whitening_rho=1.0, gradient_contrast="both",
                     rank_multiplier=1.0)


class SeqLossToy(nn.Module):
    """Token sequences (batch, time, d) through one block; squared error per valid token, averaged over the batch's
    valid tokens like a causal LM's mean loss."""

    def __init__(self, d=6):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d)])

    def terms(self, input_ids, attention_mask, labels):
        blk = self.model.layers[0]
        h = blk.down_proj(torch.tanh(blk.q_proj(input_ids)))
        return (((h - labels) ** 2).sum(-1) * attention_mask).sum(-1)  # one per example

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(loss=self.terms(input_ids, attention_mask, labels).sum() / attention_mask.sum())


def test_example_gradient_moments_match_per_example_gradients():
    torch.manual_seed(41)
    m = SeqLossToy()
    m.train()
    t = R.select_targets(m, "decoder_linear")
    x, y = torch.randn(5, 4, 6), torch.randn(5, 4, 6)
    mask = torch.ones(5, 4)
    mask[1, 3:] = 0
    mask[4, 2:] = 0
    x[mask == 0] = 50.0  # padding
    batches = [{"input_ids": x[i:j], "attention_mask": mask[i:j], "labels": y[i:j]} for i, j in ((0, 3), (3, 5))]
    got, n = R.collect_example_gradient_moments(m, batches, t, lambda b: b)
    mean, _ = R.collect_mean_gradients(m, batches, t, lambda b: b)
    assert n == 5
    weights = [t[k].weight for k in t]
    per = {k: [] for k in t}
    for b in batches:  # g_e = B x the gradient of example e's part of the batch's mean loss
        size = b["input_ids"].shape[0]
        terms = m.terms(**b) / b["attention_mask"].sum()
        for e in range(size):
            for k, g in zip(t, torch.autograd.grad(terms[e], weights, retain_graph=True)):
                per[k].append(size * g.double())
    for k in t:
        g = torch.stack(per[k])
        assert torch.allclose(g.mean(0), mean[k].double(), atol=1e-6)  # the mean of the g_e is the mean gradient
        assert torch.allclose(got[k][0], torch.einsum("eij,ekj->ik", g, g) / 5, rtol=1e-4, atol=1e-8)  # E[g g^T]
        assert torch.allclose(got[k][1], torch.einsum("eji,ejk->ik", g, g) / 5, rtol=1e-4, atol=1e-8)  # E[g^T g]
    assert m.training and all(p.requires_grad for p in m.parameters()) and all(p.grad is None for p in m.parameters())
    assert all(not mod._forward_hooks for mod in t.values())


def test_gradient_moment_basis_keeps_directions_that_cancel_in_the_mean():
    e_out, e_in = torch.eye(4, dtype=torch.float64), torch.eye(5, dtype=torch.float64)
    a, b = 3.0 * torch.outer(e_out[0], e_in[0]), torch.outer(e_out[1], e_in[2])
    per = torch.stack([a + b, -a + b])  # the a parts cancel in the mean
    g_f = per.mean(0)
    m_out, m_in = torch.einsum("eij,ekj->ik", per, per) / 2, torch.einsum("eji,ejk->ik", per, per) / 2
    plain = R.svd_basis(g_f.float(), 0.99)
    assert plain.rank == 1 and torch.allclose(plain.vh.abs().double(), e_in[2:3], atol=1e-5)
    ex = R.gradient_moment_basis(g_f, m_out, m_in, rank=1)
    assert ex.active and ex.rank == 1 and ex.total_rank == 4
    assert torch.allclose(ex.u.abs().double(), e_out[:, 0:1], atol=1e-5)
    assert torch.allclose(ex.vh.abs().double(), e_in[0:1], atol=1e-5)
    assert ex.evr < 1e-6  # (e0, e0) holds none of the mean gradient
    assert abs(R.gradient_moment_basis(g_f, m_out, m_in, rank=2).evr - 1.0) < 1e-6
    torch.manual_seed(43)  # one gradient shared by every example: the plain basis's subspaces
    g = torch.randn(6, 8, dtype=torch.float64)
    same = R.gradient_moment_basis(g, g @ g.T, g.T @ g, rank=3)
    ref = R.svd_basis(g.float(), 0.5, rank=3)
    assert torch.allclose(same.u @ same.u.T, ref.u @ ref.u.T, atol=1e-4)
    assert torch.allclose(same.vh.T @ same.vh, ref.vh.T @ ref.vh, atol=1e-4)
    assert abs(same.evr - ref.evr) < 1e-5
    assert torch.allclose(same.u.T @ same.u, torch.eye(3), atol=1e-5)
    assert torch.allclose(same.vh @ same.vh.T, torch.eye(3), atol=1e-5)
    assert R.gradient_moment_basis(g, g @ g.T, g.T @ g, rank=9).rank == 6  # clipped to min(out, in)
    assert not R.gradient_moment_basis(torch.zeros(4, 5), m_out, m_in, rank=1).active
    with pytest.raises(ValueError):
        R.gradient_moment_basis(g_f, m_out, m_in, rank=0)


def test_warm_start_with_gradient_moment_bases():
    torch.manual_seed(47)
    t = R.select_targets(Toy(), "decoder_linear")
    w0 = {k: m.weight.detach().clone() for k, m in t.items()}
    gr = {k: torch.randn(6, 6) for k in t}
    src = {k: torch.randn(6, 6) for k in t}
    per = {k: torch.randn(3, 6, 6, dtype=torch.float64) for k in t}
    mom = {k: (torch.einsum("eij,ekj->ik", p, p) / 3, torch.einsum("eji,ejk->ik", p, p) / 3) for k, p in per.items()}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.6, gradient_moments=mom, rank_multiplier=1.0)
    for k, m in t.items():
        r0 = R.svd_basis(src[k], 0.6).rank
        b = R.gradient_moment_basis(src[k], *mom[k], r0)
        assert rep[k]["reference_rank"] == r0 and rep[k]["rank"] == r0 and abs(rep[k]["evr"] - b.evr) < 1e-6
        proposal = (R.shrink_factors(R.row_alignment(w0[k], gr[k]), 0.3) - 1.0).unsqueeze(1) * w0[k]
        assert torch.allclose(m.weight.detach(), w0[k] + R.project(proposal, b.u, b.vh), atol=1e-5)
    for bad in (dict(gradient_moments=mom, component_selection="full"),  # own-spectrum ranks
                dict(gradient_moments=mom, rank_multiplier=1.0, whitening_rho=1.0, gradient_contrast="both"),
                dict(gradient_moments=mom, rank_multiplier=1.0, whitening=gr, whitening_rho=1.0)):
        with pytest.raises(ValueError):
            R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.6, **bad)
    with pytest.raises(ValueError):  # needs a restriction
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, gradient_moments=mom, rank_multiplier=1.0)


def test_share_rank():
    e = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert [R.share_rank(e, t) for t in (0.1, 0.4, 0.7, 0.71, 1.0)] == [1, 1, 2, 3, 4]
    assert R.share_rank(torch.tensor([2.0, -1e-12, 0.0]), 1.0) <= 3  # tiny negative eigenvalues count as zero


def test_gradient_contrast_basis_own_spectrum_ranks():
    # The toy of test_gradient_contrast_basis_prefers_forget_only_directions, rho 0.1: lambda orders the input side
    # e2, e0, e3 (forget energies 1, 9, 0.01) and the output side e1, e0, e2 (the same energies).
    e_out, e_in = torch.eye(4, dtype=torch.float64), torch.eye(5, dtype=torch.float64)
    g_f = 3.0 * torch.outer(e_out[0], e_in[0]) + 1.0 * torch.outer(e_out[1], e_in[2]) + 0.1 * torch.outer(e_out[2], e_in[3])
    g_r = 5.0 * torch.outer(e_out[0], e_in[0]) + 0.1 * torch.outer(e_out[3], e_in[4])
    total = 9.0 + 1.0 + 0.01
    one = R.gradient_contrast_basis(g_f, g_r, None, 0.1, "both", energy_threshold=0.05)
    assert one.rank == 1 and one.rank_in == 1 and abs(one.evr - 1.0 / total) < 1e-4
    two = R.gradient_contrast_basis(g_f, g_r, None, 0.1, "both", energy_threshold=0.5)
    assert two.rank == 2 and two.rank_in == 2 and abs(two.evr - 10.0 / total) < 1e-4
    assert torch.allclose((two.vh.T @ two.vh).diagonal().double(), e_in[0] + e_in[2], atol=1e-4)
    inp = R.gradient_contrast_basis(g_f, g_r, None, 0.1, "input", energy_threshold=0.5)  # output side: plain EVR
    assert inp.rank == R.svd_basis(g_f.float(), 0.5).rank == 1 and inp.rank_in == 2 and abs(inp.evr - 9.0 / total) < 1e-4
    assert inp.u.shape == (4, 1) and inp.vh.shape == (2, 5)
    assert R.gradient_contrast_basis(g_f, g_r, 2, 0.1, "both").rank_in is None  # an explicit budget: one rank
    for bad in (dict(rank=None), dict(rank=2, energy_threshold=0.5), dict(rank=None, energy_threshold=0.0)):
        with pytest.raises(ValueError):
            R.gradient_contrast_basis(g_f, g_r, rho=0.1, sides="both", **bad)


def test_gradient_moment_basis_own_spectrum_ranks():
    e_out, e_in = torch.eye(4, dtype=torch.float64), torch.eye(5, dtype=torch.float64)
    a, b = 3.0 * torch.outer(e_out[0], e_in[0]), torch.outer(e_out[1], e_in[2])
    per = torch.stack([a + b, -a + b])
    m_out, m_in = torch.einsum("eij,ekj->ik", per, per) / 2, torch.einsum("eji,ejk->ik", per, per) / 2
    one = R.gradient_moment_basis(per.mean(0), m_out, m_in, energy_threshold=0.5)  # spectra 9, 1 on both sides
    assert one.rank == 1 and one.rank_in == 1 and one.evr < 1e-6
    two = R.gradient_moment_basis(per.mean(0), m_out, m_in, energy_threshold=0.95)
    assert two.rank == 2 and two.rank_in == 2 and abs(two.evr - 1.0) < 1e-6
    torch.manual_seed(53)  # a wide weight: the input side's own rank may exceed min(out, in)
    per = torch.randn(8, 2, 6, dtype=torch.float64)
    m_out, m_in = torch.einsum("eij,ekj->ik", per, per) / 8, torch.einsum("eji,ejk->ik", per, per) / 8
    wide = R.gradient_moment_basis(per.mean(0), m_out, m_in, energy_threshold=0.999)
    want_in = R.share_rank(torch.linalg.eigvalsh(m_in).flip(0), 0.999)
    assert wide.total_rank == 2 and wide.rank <= 2 and wide.rank_in == want_in > 2
    assert wide.u.shape == (2, wide.rank) and wide.vh.shape == (want_in, 6)
    assert torch.allclose(wide.vh @ wide.vh.T, torch.eye(want_in), atol=1e-5)
    with pytest.raises(ValueError):
        R.gradient_moment_basis(per.mean(0), m_out, m_in)  # neither rank nor threshold


def test_warm_start_with_own_spectrum_ranks():
    torch.manual_seed(59)
    gr = {k: torch.randn(6, 6) for k in R.select_targets(Toy(), "decoder_linear")}
    src = {k: torch.randn(6, 6) for k in gr}
    per = {k: torch.randn(3, 6, 6, dtype=torch.float64) for k in gr}
    mom = {k: (torch.einsum("eij,ekj->ik", p, p) / 3, torch.einsum("eji,ejk->ik", p, p) / 3) for k, p in per.items()}
    for kwargs, build in (
            (dict(gradient_moments=mom), lambda k: R.gradient_moment_basis(src[k], *mom[k], energy_threshold=0.6)),
            (dict(gradient_contrast="both", whitening_rho=1.0),
             lambda k: R.gradient_contrast_basis(src[k], gr[k], None, 1.0, "both", energy_threshold=0.6)),
            (dict(gradient_contrast="both", whitening_rho=1.0, contrast_rank_rule="kept_energy"),
             lambda k: R.gradient_contrast_basis(src[k], gr[k], None, 1.0, "both", kept_threshold=0.6))):
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: m.weight.detach().clone() for k, m in t.items()}
        _, rep = R.warm_start(t, gr, 0.3, src, 0.6, **kwargs)
        for k, m in t.items():
            b = build(k)
            assert rep[k]["rank"] == b.rank and rep[k].get("rank_in") == b.rank_in and "reference_rank" not in rep[k]
            proposal = (R.shrink_factors(R.row_alignment(w0[k], gr[k]), 0.3) - 1.0).unsqueeze(1) * w0[k]
            assert torch.allclose(m.weight.detach(), w0[k] + R.project(proposal, b.u, b.vh), atol=1e-5)
    for bad in (dict(gradient_contrast="both", whitening_rho=1.0, contrast_rank_rule="kept_energy", rank_multiplier=1.0),
                dict(gradient_moments=mom, contrast_rank_rule="kept_energy"),
                dict(gradient_contrast="both", whitening_rho=1.0, contrast_rank_rule="lambda")):
        with pytest.raises(ValueError):
            R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.6, **bad)


def test_gradient_contrast_basis_kept_energy_rank():
    # The diagonal toy at rho 0.1 (lambda order: output e1, e0, e2; input e2, e0, e3): kept(r) = 1 / 10.01 at r = 1,
    # (1 + 9) / 10.01 at r = 2 and 1 at r = 3 (G_f[2, 3] = 0.1 joins).
    e_out, e_in = torch.eye(4, dtype=torch.float64), torch.eye(5, dtype=torch.float64)
    g_f = 3.0 * torch.outer(e_out[0], e_in[0]) + 1.0 * torch.outer(e_out[1], e_in[2]) + 0.1 * torch.outer(e_out[2], e_in[3])
    g_r = 5.0 * torch.outer(e_out[0], e_in[0]) + 0.1 * torch.outer(e_out[3], e_in[4])
    total = 9.0 + 1.0 + 0.01
    for threshold, rank, kept in ((0.05, 1, 1.0 / total), (0.5, 2, 10.0 / total), (0.9995, 3, 1.0)):
        b = R.gradient_contrast_basis(g_f, g_r, None, 0.1, "both", kept_threshold=threshold)
        assert b.rank == rank and b.rank_in is None and abs(b.evr - kept) < 1e-4
        assert b.u.shape == (4, rank) and b.vh.shape == (rank, 5)
    plain_like = R.gradient_contrast_basis(g_f, g_r, None, 0.1, "input", kept_threshold=0.5)  # u: e0, then e1
    assert plain_like.rank == 2 and abs(plain_like.evr - 10.0 / total) < 1e-4
    for bad in (dict(rank=2, kept_threshold=0.5), dict(energy_threshold=0.5, kept_threshold=0.5),
                dict(kept_threshold=1.5)):
        with pytest.raises(ValueError):
            R.gradient_contrast_basis(g_f, g_r, **{"rank": None, "rho": 0.1, "sides": "both", **bad})


def test_combined_bases_reduce_to_c_and_d():
    gen = torch.Generator().manual_seed(19)
    g = torch.randn(5, 3, generator=gen) @ torch.randn(3, 6, generator=gen)
    x_r, x_f = torch.randn(40, 6, generator=gen), torch.randn(40, 6, generator=gen) * torch.tensor([1, 3, 1, 1, 2, 1.0])
    cov_r, cov_f = x_r.T @ x_r / 40, x_f.T @ x_f / 40
    w = torch.randn(5, 6, generator=gen)
    proj = lambda m: m @ m.T  # noqa: E731
    c = R.whitened_svd_basis(g, cov_r, 0.9, rho=0.5, rank=2)
    # beta 0 = the gradient contrast: the input side of C; with the whitened-gradient output side it is C
    seq0 = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="whitened_gradient", input_mix=0.0)
    assert torch.allclose(proj(seq0.vh.T), proj(c.vh.T), atol=1e-4)
    assert torch.allclose(proj(seq0.u), proj(c.u), atol=1e-4)
    joint0 = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=0.0)
    assert torch.allclose(proj(joint0.u), proj(c.u), atol=1e-4)  # span(G_f V) = span of C's U
    # beta 1 = D; scaling the forget covariance changes nothing
    d = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient")
    d2 = R.representation_basis(g, 7.0 * cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=1.0)
    assert torch.allclose(proj(d.vh.T), proj(d2.vh.T), atol=1e-5)
    mid = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=0.5)
    assert mid.active and mid.vh.shape == (2, 6) and 0.0 < mid.evr <= 1.0 + 1e-6
    with pytest.raises(ValueError):
        R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=1.5)


def test_v_proj_and_head_profiles():
    m = Toy()
    assert set(R.select_targets(m, "head")) == {"lm_head"}
    assert set(R.select_targets(m, "down_up_v")) == {f"model.layers.{i}.down_proj" for i in range(2)}  # no up/v
    assert set(R.select_targets(m, "down_up_o_v")) == {f"model.layers.{i}.down_proj" for i in range(2)}
    assert set(R.select_targets(m, "attn_q")) == {f"model.layers.{i}.q_proj" for i in range(2)}  # one type only
    with pytest.raises(ValueError):
        R.select_targets(m, "attn_k")


def test_basis_prefix_matches_direct_bases():
    gen = torch.Generator().manual_seed(20)
    g = torch.randn(5, 7, generator=gen)
    x_r, x_f = torch.randn(60, 7, generator=gen), torch.randn(60, 7, generator=gen) * torch.linspace(0.5, 3.0, 7)
    cov_r, cov_f = x_r.T @ x_r / 60, x_f.T @ x_f / 60
    w = torch.randn(5, 7, generator=gen)
    proj = lambda m: m @ m.T  # noqa: E731
    families = {
        "plain": (R.svd_basis(g, 0.9, rank=5), lambda r: R.svd_basis(g, 0.9, rank=r), False),
        "C": (R.whitened_svd_basis(g, cov_r, 0.9, rho=0.5, rank=5),
              lambda r: R.whitened_svd_basis(g, cov_r, 0.9, rho=0.5, rank=r), False),
        "D": (R.representation_basis(g, cov_f, cov_r, w, rank=5, rho=0.5, output_side="gradient"),
              lambda r: R.representation_basis(g, cov_f, cov_r, w, rank=r, rho=0.5, output_side="gradient"), True),
    }
    for name, (full, direct_at, gradient_output) in families.items():
        for r in (1, 2, 3):
            pre, direct = R.basis_prefix(full, r, g, gradient_output), direct_at(r)
            assert pre.rank == r, name
            assert torch.allclose(proj(pre.vh.T), proj(direct.vh.T), atol=1e-4), name
            assert torch.allclose(proj(pre.u), proj(direct.u), atol=1e-4), name
            kept = R.project(g, direct.u, direct.vh).square().sum() / g.square().sum()
            assert abs(pre.evr - float(kept)) < 1e-4, name


def test_forget_output_budget_picks_smallest_rank_reaching_target():
    torch.manual_seed(21)
    gen = torch.Generator().manual_seed(21)
    t = R.select_targets(Toy(), "decoder_linear")
    w0 = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn(6, 6, generator=gen) for k in t}
    src = {k: torch.randn(6, 6, generator=gen) for k in t}
    xs = {k: torch.randn(50, 6, generator=gen) for k in t}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in t}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.9, forget_covariances=cov_f, forget_output_target=0.1)
    for k in t:
        q, r = rep[k]["forget_output_change"], rep[k]["rank"]
        assert abs(R.forget_output_change(t[k].weight.detach() - w0[k], w0[k], cov_f[k]) - q) < 1e-5
        assert q >= 0.1 or r == 6
        assert rep[k]["reference_rank"] == R.svd_basis(src[k], 0.9).rank
        if r > 1:  # one rank less misses the target
            proposal = (R.shrink_factors(R.row_alignment(w0[k], gr[k]), 0.3) - 1.0).unsqueeze(1) * w0[k]
            below = R.svd_basis(src[k], 0.9, rank=r - 1)
            assert R.forget_output_change(R.project(proposal, below.u, below.vh), w0[k], cov_f[k]) < 0.1
    t2 = R.select_targets(Toy(), "decoder_linear")
    _, rep2 = R.warm_start(t2, gr, 0.3, src, 0.9, whitening_rho=0.5, representation_output="gradient",
                           forget_covariances=cov_f, retain_covariances=cov_r, forget_output_target=0.1)
    assert all(rep2[k]["forget_output_change"] >= 0.1 or rep2[k]["rank"] == 6 for k in t2)
    for bad in (dict(rank_multiplier=1.5, forget_covariances=cov_f), dict(), dict(forget_covariances=cov_f,
                                                                              layer_scaling="subspace_norm")):
        with pytest.raises(ValueError):
            R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, forget_output_target=0.1, **bad)


def _component_case(seed, out=7, inn=9, r=4, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    u = torch.linalg.qr(torch.randn(out, r, generator=gen, dtype=torch.float64))[0]
    vh = torch.linalg.qr(torch.randn(inn, r, generator=gen, dtype=torch.float64))[0].T
    x = scale * torch.randn(out, inn, generator=gen, dtype=torch.float64)
    g = torch.randn(out, inn, generator=gen, dtype=torch.float64)
    return u, vh, x, g


def test_component_reconstruction_equals_projected_update():  # 1
    for u_free in (False, True):
        u, vh, x, g = _component_case(30)
        u = None if u_free else u
        parts = R.decompose_update(x, u, vh, g)
        full = R.compose_update(parts, torch.ones_like(parts.sigma))
        direct = R.project(x, u, vh).double()
        assert torch.allclose(full, direct, atol=1e-6), u_free
        assert parts.sigma.numel() == min(vh.shape[0], x.shape[0] if u is None else u.shape[1])


def test_component_contributions_and_their_sum():  # 2, 3
    u, vh, x, g = _component_case(31)
    parts = R.decompose_update(x, u, vh, g)
    for j in range(parts.sigma.numel()):
        d_j = parts.sigma[j] * torch.outer(parts.left[:, j], parts.right[j])
        assert abs(float((g * d_j).sum()) - float(parts.contribution[j])) < 1e-10
    assert abs(float(parts.contribution.sum()) - float((g * R.project(x, u, vh).double()).sum())) < 1e-6


def test_components_are_orthogonal_and_energy_decomposes():  # 4, 5
    u, vh, x, g = _component_case(32)
    parts = R.decompose_update(x, u, vh, g)
    k = parts.sigma.numel()
    comps = [parts.sigma[j] * torch.outer(parts.left[:, j], parts.right[j]) for j in range(k)]
    for i in range(k):
        for j in range(k):
            inner = float((comps[i] * comps[j]).sum())
            assert abs(inner - (float(parts.sigma[i]) ** 2 if i == j else 0.0)) < 1e-9
    alpha = torch.tensor([1.0, 0.5, 0.0, 0.25], dtype=torch.float64)
    upd = R.compose_update(parts, alpha)
    assert abs(float(upd.square().sum()) - float((alpha.square() * parts.sigma.square()).sum())) < 1e-9


def test_signed_mask_keeps_positive_components_only():  # 6
    u, vh, x, g = _component_case(33)
    parts = R.decompose_update(x, u, vh, g)
    alpha = R.component_weights(parts, "signed")
    assert torch.equal(alpha > 0, parts.contribution > 0)
    signed = R.compose_update(parts, alpha)
    expect = sum(parts.sigma[j] * torch.outer(parts.left[:, j], parts.right[j])
                 for j in range(parts.sigma.numel()) if parts.contribution[j] > 0)
    assert torch.allclose(signed, expect, atol=1e-10)
    assert float((g * signed).sum()) >= float((g * R.compose_update(parts, torch.ones_like(alpha))).sum()) - 1e-12
    gamma = R.component_weights(parts, "uniform_matched")
    assert torch.allclose(R.compose_update(parts, gamma).norm(), signed.norm(), atol=1e-10)  # equal displacement


def test_zero_gradient_and_zero_update_cases():  # 7
    u, vh, x, g = _component_case(34)
    parts = R.decompose_update(x, u, vh, torch.zeros_like(g))
    assert torch.all(parts.contribution == 0)
    for sel in ("signed", "random", "uniform_matched"):
        alpha = R.component_weights(parts, sel, torch.Generator().manual_seed(0))
        assert float(alpha.abs().sum()) == 0.0 and torch.all(R.compose_update(parts, alpha) == 0), sel
    zero = R.decompose_update(torch.zeros_like(x), u, vh, g)
    assert torch.all(zero.sigma == 0) and torch.isfinite(zero.contribution).all()
    assert float(R.component_weights(zero, "uniform_matched").abs().sum()) == 0.0
    assert R.effective_rank(zero.sigma) == 0.0
    # warm start: s_min 1 (no shrink) gives a zero proposal; every selection leaves the weights unchanged
    t = R.select_targets(Toy(), "decoder_linear")
    w0 = {k: v.weight.detach().clone() for k, v in t.items()}
    gen = torch.Generator().manual_seed(35)
    gr = {k: torch.randn(6, 6, generator=gen) for k in t}
    src = {k: torch.randn(6, 6, generator=gen) for k in t}
    _, rep = R.warm_start(t, gr, 1.0, src, 0.9, component_selection="signed", component_diagnostics=True)
    assert all(torch.equal(t[k].weight, w0[k]) for k in t) and all(rep[k]["components"]["energy"] == 0 for k in t)


def test_small_singular_values_stay_finite():  # 8
    u, vh, x, g = _component_case(36)
    tiny = R.project(x, u, vh).double()
    p, _, qh = torch.linalg.svd(u.T @ tiny @ vh.T)
    x_small = u @ p @ torch.diag(torch.tensor([1.0, 1e-9, 1e-14, 0.0], dtype=torch.float64)) @ qh @ vh
    parts = R.decompose_update(x_small, u, vh, g)
    assert torch.isfinite(parts.sigma).all() and torch.isfinite(parts.contribution).all()
    assert torch.allclose(R.compose_update(parts, torch.ones_like(parts.sigma)), x_small, atol=1e-10)
    assert abs(float(parts.contribution[-1])) < 1e-12  # a zero component has no first-order effect
    assert 1.0 <= R.effective_rank(parts.sigma) < 1.01


def test_random_control_is_deterministic_and_matches_signed_count():  # 9
    u, vh, x, g = _component_case(37, out=12, inn=15, r=10)
    parts = R.decompose_update(x, u, vh, g)
    n_pos = int((parts.contribution > 0).sum())
    a1 = R.component_weights(parts, "random", torch.Generator().manual_seed(5))
    a2 = R.component_weights(parts, "random", torch.Generator().manual_seed(5))
    a3 = R.component_weights(parts, "random", torch.Generator().manual_seed(6))
    assert torch.equal(a1, a2) and int(a1.sum()) == int(a3.sum()) == n_pos
    assert not torch.equal(a1, a3) or n_pos in (0, parts.sigma.numel())
    t = R.select_targets(Toy(), "decoder_linear")
    gen = torch.Generator().manual_seed(38)
    gr = {k: torch.randn(6, 6, generator=gen) for k in t}
    src = {k: torch.randn(6, 6, generator=gen) for k in t}
    runs = []
    for _ in range(2):  # the same seed twice: the same masks
        torch.manual_seed(39)
        t2 = R.select_targets(Toy(), "decoder_linear")
        R.warm_start(t2, gr, 0.3, src, 0.9, component_selection="random", component_seed=1)
        runs.append({k: v.weight.detach().clone() for k, v in t2.items()})
    assert all(torch.equal(runs[0][k], runs[1][k]) for k in runs[0])


def test_component_modes_default_and_full_reproduce_the_old_update():  # 10
    gen = torch.Generator().manual_seed(40)
    gr = {f"model.layers.{i}.{p}": torch.randn(6, 6, generator=gen) for i in range(2) for p in ("q_proj", "down_proj")}
    src = {k: torch.randn(6, 6, generator=gen) for k in gr}
    xs = {k: torch.randn(50, 6, generator=gen) for k in gr}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in gr}
    extra = dict(whitening_rho=1.0, representation_output="gradient", forget_covariances=cov_f,
                 retain_covariances=cov_r, rank_multiplier=1.25)
    weights = {}
    for label, kw in (("old", {}), ("diag", dict(component_diagnostics=True)), ("full", dict(component_selection="full")),
                      ("signed", dict(component_selection="signed", component_diagnostics=True)),
                      ("matched", dict(component_selection="uniform_matched"))):
        torch.manual_seed(41)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        _, rep = R.warm_start(t, gr, 0.3, src, 0.9, **extra, **kw)
        weights[label] = ({k: v.weight.detach().clone() for k, v in t.items()}, rep, w0)
    old, _, w0 = weights["old"]
    assert all(torch.equal(weights["diag"][0][k], old[k]) for k in old)  # diagnostics alone change nothing
    assert all(torch.allclose(weights["full"][0][k], old[k], atol=1e-6) for k in old)
    for k in old:
        c = weights["signed"][1][k]["components"]
        assert c["kept"] == c["positive"] and c["reconstruction_rel_error"] < 1e-5  # fp32 projection vs fp64 sum
        assert abs(c["contribution_sum"] - c["first_order_forget_direct"]) < 1e-5 * max(1.0, abs(c["contribution_sum"]))
        signed_norm = float((weights["signed"][0][k] - w0[k]).norm())
        matched_norm = float((weights["matched"][0][k] - w0[k]).norm())
        assert abs(signed_norm - matched_norm) < 1e-5 * max(1.0, signed_norm)
        assert len(c["sigma"]) == len(c["contribution"]) == c["components"]
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, component_selection="signed")
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, component_selection="best")


def test_component_control_holds_for_a_near_zero_fp32_update():
    """The fp32 projection of a large proposal onto a direction it barely touches loses relative digits; the fp64
    components still match it to fp32 rounding of the proposal, and sum c_j matches <G, P(X)> within ||G|| times
    the reconstruction gap (the control of recast_tofu.py)."""
    gen = torch.Generator().manual_seed(42)
    u = torch.linalg.qr(torch.randn(64, 1, generator=gen, dtype=torch.float64))[0]
    vh = torch.linalg.qr(torch.randn(256, 1, generator=gen, dtype=torch.float64))[0].T
    y = 10.0 * torch.randn(64, 256, generator=gen, dtype=torch.float64)
    x = (y - u @ (u.T @ y @ vh.T) @ vh + 1e-6 * u @ vh).float()  # core ~1e-6 against entries ~10
    g = torch.randn(64, 256, generator=gen)
    delta = R.project(x, u.float(), vh.float())
    parts = R.decompose_update(x, u.float(), vh.float(), g)
    alpha = torch.ones_like(parts.sigma)
    rep = R.component_report(parts, alpha, delta, R.compose_update(parts, alpha), g, g, keep_arrays=False)
    assert rep["reconstruction_abs_error"] <= 1e-5 * float(x.norm())
    gap = abs(rep["contribution_sum"] - rep["first_order_forget_direct"])
    assert gap <= rep["grad_forget_norm"] * rep["reconstruction_abs_error"] + 1e-6 * abs(rep["first_order_forget_direct"]) + 1e-12


def test_input_contrast_and_direction_weights():
    e = torch.eye(8, dtype=torch.float64)
    cf = torch.diag(torch.tensor([4.0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float64))
    cr = torch.eye(8, dtype=torch.float64)
    k = R.input_contrast(e[:2], cf, cr)  # (4/11)/(1/8), (1/11)/(1/8)
    assert torch.allclose(k, torch.tensor([32 / 11, 8 / 11], dtype=torch.float64))
    w = R.direction_contrast_weights(e[:2], cf, cr, 0.3)  # k / mean(k) = 1.6, 0.4; 1.6 capped at 1 / 0.7
    assert torch.allclose(w, torch.tensor([1 / 0.7, 0.4], dtype=torch.float64))
    gen = torch.Generator().manual_seed(50)
    a = torch.randn(30, 8, generator=gen, dtype=torch.float64)
    c = a.T @ a / 30
    vh = torch.linalg.qr(torch.randn(8, 3, generator=gen, dtype=torch.float64))[0].T
    assert torch.allclose(R.input_contrast(vh, c, 3.0 * c), torch.ones(3, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(R.direction_contrast_weights(vh, c, 2.0 * c, 0.3), torch.ones(3, dtype=torch.float64))


def test_direction_weighting_and_contrast_selection_in_warm_start():
    gen = torch.Generator().manual_seed(51)
    names = [f"model.layers.{i}.{p}" for i in range(2) for p in ("q_proj", "down_proj")]
    gr = {k: torch.randn(6, 6, generator=gen) for k in names}
    src = {k: torch.randn(6, 6, generator=gen) for k in names}
    xs = {k: torch.randn(50, 6, generator=gen) * torch.linspace(0.5, 2.0, 6) for k in names}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in names}
    runs = {}
    for label, kw in (("plain", {}), ("dirw", dict(direction_weighting="contrast")),
                      ("contrast", dict(component_selection="contrast"))):
        torch.manual_seed(52)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        restr, rep = R.warm_start(t, gr, 0.3, src, 0.9, forget_covariances=cov_f, retain_covariances=cov_r, **kw)
        runs[label] = ({k: v.weight.detach() - w0[k] for k, v in t.items()}, restr, rep)
    for k in names:
        vh = runs["plain"][1][k].basis.vh.float()
        w = R.direction_contrast_weights(vh, cov_f[k], cov_r[k], 0.3).float()
        expect = ((runs["plain"][0][k] @ vh.T) * w) @ vh
        assert torch.allclose(runs["dirw"][0][k], expect, atol=1e-5)
        assert abs(runs["dirw"][2][k]["direction_weights"]["mean"] - float(w.mean())) < 1e-6
        # P(X) has the same core U^T X Vh^T as X, so its components are those warm_start decomposed
        parts = R.decompose_update(runs["plain"][0][k], runs["plain"][1][k].basis.u, runs["plain"][1][k].basis.vh, src[k])
        kept = int((R.input_contrast(parts.right, cov_f[k], cov_r[k]) > 1).sum())
        assert runs["contrast"][2][k]["components"]["kept"] == kept
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, direction_weighting="contrast")


def test_scale_update_is_linear_from_init():
    gen = torch.Generator().manual_seed(53)
    w0, ws = torch.randn(5, 7, generator=gen), torch.randn(5, 7, generator=gen)
    assert torch.equal(R.scale_update(w0, ws, 0.0), w0)
    assert torch.allclose(R.scale_update(w0, ws, 1.0), ws, atol=1e-6)
    for t in (0.01, 0.35, 1.4):
        assert abs(float((R.scale_update(w0, ws, t) - w0).norm()) - t * float((ws - w0).norm())) < 1e-5


def test_parse_type_values():
    assert R.parse_type_values("down_proj=0.2,up_proj=0.5, o_proj=0.5") == {
        "down_proj": 0.2, "up_proj": 0.5, "o_proj": 0.5}
    for bad in ("down_proj", "down_proj=1,down_proj=2", "=1", "down_proj=x"):
        with pytest.raises(ValueError):
            R.parse_type_values(bad)


def test_warm_start_per_type_floors():
    """Rows anti-aligned with -G_r (G_r = W) sit at the floor, so each type is scaled by exactly its own floor;
    equal per-type floors reproduce the single floor, also in the restricted (projected) warm start."""
    torch.manual_seed(61)
    floors = {"q_proj": 0.1, "k_proj": 0.2, "v_proj": 0.3, "o_proj": 0.4, "gate_proj": 0.5, "up_proj": 0.6,
              "down_proj": 0.7}
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    R.warm_start(t, {k: w.clone() for k, w in before.items()}, floors)
    for k, v in t.items():
        assert torch.allclose(v.weight, floors[k.rsplit(".", 1)[-1]] * before[k], atol=1e-6)
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), {k: w.clone() for k, w in before.items()},
                     {"q_proj": 0.3})
    results = []
    for s_min in (0.3, {k: 0.3 for k in floors}):
        torch.manual_seed(62)
        m = Toy()
        t = R.select_targets(m, "decoder_linear")
        gr = {k: v.weight.detach().clone() for k, v in t.items()}
        gf = {k: torch.randn_like(v.weight) for k, v in t.items()}
        R.warm_start(t, gr, s_min, gf, 0.97)
        results.append({k: v.weight.detach().clone() for k, v in t.items()})
    assert all(torch.equal(results[0][k], results[1][k]) for k in results[0])


def _restricted_floor_case(floor, seed=71):
    """Restricted warm start of the toy with G_r = W (every row anti-aligned, so at the floor) and a fixed random
    forget gradient: the same bases for every floor. Returns the deltas, the report and the restrictions."""
    torch.manual_seed(seed)
    t = R.select_targets(Toy(), "decoder_linear")
    w0 = {k: v.weight.detach().clone() for k, v in t.items()}
    gf = {k: torch.randn_like(w) for k, w in w0.items()}
    restr, rep = R.warm_start(t, {k: w.clone() for k, w in w0.items()}, floor, gf, 0.9)
    return {k: v.weight.detach() - w0[k] for k, v in t.items()}, rep, restr


def test_floor_equals_scaled_update_for_rows_at_the_floor():
    """Stage 1: with every row at the floor the update at floor 1 - t (1 - s) is t times the update at floor s, so
    W + t Delta W^D is a floor (t = 1 / (1 - s) is floor 0); the report gives applied_norm = (1 - s) ||P(W)||."""
    base, rep, _ = _restricted_floor_case(0.3)
    for t in (0.5, 1.2):
        scaled, _, _ = _restricted_floor_case(1.0 - t * 0.7)
        for k in base:
            assert torch.allclose(scaled[k], t * base[k], atol=1e-5)
    zero, _, _ = _restricted_floor_case(0.0)
    for k in base:
        assert torch.allclose(zero[k], base[k] / 0.7, atol=1e-5)
    for r in rep.values():
        assert r["shrink_at_floor"] == 1.0
        assert abs(r["applied_norm"] - 0.7 * r["projected_weight_norm"]) < 1e-5


def test_update_budget_and_uniform_matched_floor():
    """Stage 1.4: B(s)^2 = sum_k (1 - s_k)^2 E_k is the squared norm of the whole update; the uniform floor of the same
    budget gives an update of that norm; a budget above sqrt(sum E_k) needs s < 0 and is infeasible."""
    floors = {"q_proj": 0.1, "down_proj": 0.6}
    delta, rep, _ = _restricted_floor_case(floors)
    energies = {}
    for name, r in rep.items():
        kind = name.rsplit(".", 1)[-1]
        energies[kind] = energies.get(kind, 0.0) + r["projected_weight_norm"] ** 2
    norm = lambda d: math.sqrt(sum(float(x.square().sum()) for x in d.values()))  # noqa: E731
    b = R.update_budget(floors, energies)
    assert abs(b - norm(delta)) < 1e-5 * max(1.0, b)
    s_u = R.uniform_floor_for_budget(b, energies)
    assert s_u is not None and 0.1 < s_u < 0.6
    matched, _, _ = _restricted_floor_case(s_u)
    assert abs(norm(matched) - b) < 1e-5 * max(1.0, b)
    assert abs(R.uniform_floor_for_budget(math.sqrt(sum(energies.values())), energies)) < 1e-12
    assert R.uniform_floor_for_budget(1.01 * math.sqrt(sum(energies.values())), energies) is None
    with pytest.raises(ValueError):
        R.update_budget({"q_proj": 0.1}, energies)


def test_shrink_score_rules_in_warm_start():
    """Stage 2: the default rule reproduces the baseline bit for bit; every other rule applies P((S - I) W) with S from
    recast_scores at the same floor and basis; the diagnostics report every rule; the rules exclude other changes."""
    import recast_scores as RS
    runs = {}
    for rule, diag in (("baseline", False), ("original", True)) + tuple((r, False) for r in RS.RULES[1:]):
        torch.manual_seed(81)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        gr = {k: torch.randn_like(w) for k, w in w0.items()}
        gf = {k: torch.randn_like(w) for k, w in w0.items()}
        kw = {} if rule == "baseline" else {"shrink_score": rule, "score_diagnostics": diag}
        restr, rep = R.warm_start(t, gr, 0.05, gf, 0.9, **kw)
        runs[rule] = ({k: v.weight.detach() - w0[k] for k, v in t.items()}, restr, rep, w0, gr)
    for k in runs["baseline"][0]:
        assert torch.equal(runs["original"][0][k], runs["baseline"][0][k])
    for rule in RS.RULES[1:]:
        delta, restr, rep, w0, gr = runs[rule]
        for k in delta:
            b = restr[k].basis
            s = RS.row_factors(RS.shrink_scores(w0[k], gr[k], b.u, b.vh), rule, 0.05).float()
            expect = R.project((s - 1.0).unsqueeze(1) * w0[k], b.u, b.vh)
            assert torch.allclose(delta[k], expect, atol=1e-6)
            assert rep[k]["shrink_score"] == rule
    rep = runs["original"][2]
    for k, r in rep.items():
        assert set(r["scores"]) >= {"original", "projected_row", "projected_sensitivity", "global_projected"}
        assert abs(r["applied_first_order"]["update_norm"] - r["applied_norm"]) < 1e-5
        assert set(r["update_norm_by_floor"]) == set(RS.RULES)
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), runs["baseline"][4], 0.05, shrink_score="projected_row")
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), runs["baseline"][4], 0.05, gf, 0.9,
                     shrink_score="cosine")


def test_core_diagnostics_in_warm_start():
    """Stage 3: with every row at the floor the applied S is uniform, so the update core is (s - 1) K (cosines 1, no
    off-diagonal energy, beta = 1 - s); the random control S moves the leading subspaces; no update changes."""
    deltas = {}
    for diag in (False, True):
        torch.manual_seed(91)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        gf = {k: torch.randn_like(w) for k, w in w0.items()}
        _, rep = R.warm_start(t, {k: w.clone() for k, w in w0.items()}, 0.3, gf, 0.9, core_diagnostics=diag)
        deltas[diag] = {k: v.weight.detach() - w0[k] for k, v in t.items()}
    for k in deltas[False]:
        assert torch.equal(deltas[False][k], deltas[True][k])
    for k, r in rep.items():
        applied = r["cores"]["applied"]
        assert applied["beta"]["offdiag_energy_fraction"] < 1e-12
        assert abs(applied["beta"]["mean"] - 0.7) < 1e-6 and abs(applied["beta"]["min"] - 0.7) < 1e-6
        for a in applied["angles"].values():
            assert a["left_min_cos"] > 1 - 1e-6 and a["right_min_cos"] > 1 - 1e-6
        assert r["cores"]["control_uniform_random"]["beta"]["offdiag_energy_fraction"] > 1e-6
        assert set(r["cores"]) == {"applied", "control_uniform_random", "original", "projected_row",
                                   "projected_sensitivity", "global_projected"}


def test_weight_contraction_in_warm_start():
    """Weight-component contraction on basis D: the update is -sum beta_j E_j with beta from recast_contraction at the
    budget 1 - floor (random drawn per seed and target), so it moves each target by (1 - floor) ||P(W)||; the component
    diagnostics change nothing; the contraction needs a representation basis and a reachable budget."""
    import zlib
    import recast_bases as RB
    import recast_contraction as RC
    gen = torch.Generator().manual_seed(40)
    gr = {f"model.layers.{i}.{p}": torch.randn(6, 6, generator=gen) for i in range(2) for p in ("q_proj", "down_proj")}
    src = {k: torch.randn(6, 6, generator=gen) for k in gr}
    xs = {k: torch.randn(50, 6, generator=gen) for k in gr}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in gr}
    extra = dict(whitening_rho=1.0, representation_output="gradient", forget_covariances=cov_f,
                 retain_covariances=cov_r, rank_multiplier=1.25)
    runs = {}
    for label, kw in (("top", dict(weight_contraction="top", contraction_strength=1.0)),
                      ("top_diag", dict(weight_contraction="top", contraction_strength=1.0, contraction_diagnostics=True)),
                      ("random", dict(weight_contraction="random", contraction_strength=1.0, component_seed=3)),
                      ("residual", dict(weight_contraction="top", contraction_strength=1.0, contraction_residual=1e-3))):
        torch.manual_seed(41)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        restr, rep = R.warm_start(t, gr, 0.3, src, 0.9, **extra, **kw)
        runs[label] = ({k: v.weight.detach() - w0[k] for k, v in t.items()}, restr, rep, w0)
    for label, mode, seed, residual in (("top", "top", 0, 0.0), ("random", "random", 3, 0.0), ("residual", "top", 0, 1e-3)):
        delta, restr, rep, w0 = runs[label]
        for k in delta:
            b = restr[k].basis
            if not b.active:
                continue
            comps = RB.weight_components(w0[k].float(), b.u, b.vh)
            g = torch.Generator().manual_seed(seed * 1000003 + zlib.crc32(k.encode()))
            beta = RC.contraction_beta(comps["kappa"], mode, 0.7, 1.0, g, residual=residual)
            assert torch.allclose(delta[k], RC.contraction_delta(comps, beta).float(), atol=1e-5)
            pw = float(R.project(w0[k], b.u, b.vh).norm())
            assert abs(float(delta[k].norm()) - 0.7 * pw) < 1e-4 * max(1.0, pw)
            assert rep[k]["contraction"]["mode"] == mode and abs(rep[k]["contraction"]["energy_fraction"] - 0.49) < 1e-6
    for k in runs["top"][0]:
        assert torch.equal(runs["top"][0][k], runs["top_diag"][0][k])
        prof = runs["top_diag"][2][k].get("contraction_components")
        if prof is not None:
            n = len(prof["kappa"])
            assert len(prof["input_contrast"]) == n == len(prof["alignment"]["forget"]) == len(prof["alignment"]["retain"])
    assert any("contraction_components" in r for r in runs["top_diag"][2].values())
    with pytest.raises(ValueError):                                       # no representation basis
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, weight_contraction="top")
    with pytest.raises(ValueError):                                       # budget 0.7 above the strength 0.5
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, **extra, weight_contraction="top",
                     contraction_strength=0.5)
    with pytest.raises(ValueError):                                       # a residual without a contraction
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, **extra, contraction_residual=1e-3)
    torch.manual_seed(41)                                                 # forget_retain: the score from the same G_f, G_r
    t = R.select_targets(Toy(), "decoder_linear")
    w0 = {k: v.weight.detach().clone() for k, v in t.items()}
    restr, _ = R.warm_start(t, gr, 0.3, src, 0.9, **extra, weight_contraction="forget_retain", contraction_strength=1.0)
    for k, v in t.items():
        b = restr[k].basis
        if not b.active:
            continue
        comps = RB.weight_components(w0[k].float(), b.u, b.vh)
        beta = RC.contraction_beta(comps["kappa"], "forget_retain", 0.7, 1.0,
                                   score=RC.forget_retain_score(comps, src[k], gr[k]))
        assert torch.allclose(v.weight.detach() - w0[k], RC.contraction_delta(comps, beta).float(), atol=1e-5)


def test_output_diagnostics_in_warm_start():
    """The output report of the applied update matches tr(M C M^T) on the input covariances, every SVD ordering is
    simulated at the ReCAST displacement, and the diagnostics change no update."""
    import recast_output_contrast as RO
    gen = torch.Generator().manual_seed(50)
    gr = {f"model.layers.{i}.{p}": torch.randn(6, 6, generator=gen) for i in range(2) for p in ("q_proj", "down_proj")}
    src = {k: torch.randn(6, 6, generator=gen) for k in gr}
    xs = {k: torch.randn(50, 6, generator=gen) for k in gr}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in gr}
    extra = dict(whitening_rho=1.0, representation_output="gradient", forget_covariances=cov_f,
                 retain_covariances=cov_r, rank_multiplier=1.25)
    runs = {}
    for diag in (False, True):
        torch.manual_seed(51)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        _, rep = R.warm_start(t, gr, 0.3, src, 0.9, **extra, output_diagnostics=diag, contraction_diagnostics=diag)
        runs[diag] = ({k: v.weight.detach() - w0[k] for k, v in t.items()}, rep, w0)
    for k in runs[False][0]:
        assert torch.equal(runs[False][0][k], runs[True][0][k])
        r, w = runs[True][1][k], runs[True][2][k]
        if "output_energy" not in r:
            continue
        d = runs[True][0][k]
        expect = RO.output_energy(d, cov_f[k]) / RO.output_energy(w, cov_f[k])
        assert abs(r["output_energy"]["forget_removed_share"] - expect) < 1e-6 * max(1.0, expect)
        assert set(r["output_energy_svd"]) == set(R.CONTRACTION_MODES)
        if k.endswith("down_proj"):
            assert set(r["output_contrast"]) == {f"{rho:g}" for rho in R.OUTPUT_CONTRAST_RHOS}
    assert any("output_energy" in r for r in runs[True][1].values())


def test_write_divisors_match_scaling_the_residual_stream():
    """h <- c h + write_k at write k is the stream before write k times c (the embedding and every earlier write times c)
    or, equivalently, write k and every later write divided by c: every read is an RMSNorm, so the logits agree;
    residual_write_divisors / apply_write_divisors give the second form."""
    import copy
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-12,
                                   tie_word_embeddings=False, max_position_embeddings=32)
    torch.manual_seed(0)
    base = transformers.LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 37, (2, 7))
    names = list(R.residual_write_divisors(3, {}))
    assert names[:2] == ["model.layers.0.self_attn.o_proj", "model.layers.0.mlp.down_proj"] and len(names) == 6
    k_name, c = "model.layers.1.self_attn.o_proj", 0.4
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    with torch.no_grad():
        a.model.embed_tokens.weight.mul_(c)
        mods = dict(a.named_modules())
        for n in names[:names.index(k_name)]:
            mods[n].weight.mul_(c)
    div = R.residual_write_divisors(3, {k_name: c, "model.layers.2.mlp.down_proj": 1.0})
    assert [div[n] for n in names] == [1.0, 1.0, c, c, c, c]
    R.apply_write_divisors(b, div)
    with torch.no_grad():
        la, lb, l0 = a(ids).logits, b(ids).logits, base(ids).logits
    assert torch.allclose(la, lb, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(la, l0, atol=1e-4)                          # it does change the function


def test_output_contrast_in_warm_start():
    """output_contrast_rho replaces each target's update by the output-contrast contraction with the displacement of
    the ReCAST update it replaces; it excludes a weight contraction."""
    import recast_output_contrast as RO
    gen = torch.Generator().manual_seed(52)
    gr = {f"model.layers.{i}.{p}": torch.randn(6, 6, generator=gen) for i in range(2) for p in ("q_proj", "down_proj")}
    src = {k: torch.randn(6, 6, generator=gen) for k in gr}
    xs = {k: torch.randn(50, 6, generator=gen) for k in gr}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in gr}
    extra = dict(whitening_rho=1.0, representation_output="gradient", forget_covariances=cov_f,
                 retain_covariances=cov_r, rank_multiplier=1.25)
    runs = {}
    for rho in (None, 0.1):
        torch.manual_seed(53)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        _, rep = R.warm_start(t, gr, 0.3, src, 0.9, **extra, output_contrast_rho=rho)
        runs[rho] = ({k: v.weight.detach() - w0[k] for k, v in t.items()}, rep, w0)
    seen = 0
    for k, d in runs[0.1][0].items():
        r = runs[0.1][1][k]
        if "output_contrast_applied" not in r:
            continue
        seen += 1
        recast_norm = float(runs[None][0][k].norm())
        assert abs(r["output_contrast_applied"]["recast_update_norm"] - recast_norm) < 1e-4 * max(1.0, recast_norm)
        expect = RO.output_contrast_contraction(runs[0.1][2][k], cov_f[k], cov_r[k], 0.1,
                                                r["output_contrast_applied"]["recast_update_norm"])[0]
        assert torch.allclose(d, expect.float(), atol=1e-5)
        assert abs(float(d.norm()) - recast_norm) < 1e-4 * max(1.0, recast_norm)
    assert seen > 0
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, **extra, output_contrast_rho=0.1,
                     weight_contraction="top")


def test_family_diagnostics_in_warm_start():
    """Every family reaches the applied update's forget output share on down_proj and reports its retain share; the
    diagnostics change no update."""
    gen = torch.Generator().manual_seed(60)
    gr = {f"model.layers.{i}.{p}": torch.randn(6, 6, generator=gen) for i in range(2) for p in ("q_proj", "down_proj")}
    src = {k: torch.randn(6, 6, generator=gen) for k in gr}
    xs = {k: torch.randn(50, 6, generator=gen) for k in gr}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in gr}
    extra = dict(whitening_rho=1.0, representation_output="gradient", forget_covariances=cov_f,
                 retain_covariances=cov_r, rank_multiplier=1.25)
    runs = {}
    for diag in (False, True):
        torch.manual_seed(61)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        _, rep = R.warm_start(t, gr, 0.3, src, 0.9, **extra, family_diagnostics=diag)
        runs[diag] = ({k: v.weight.detach() - w0[k] for k, v in t.items()}, rep)
    seen = 0
    for k in runs[False][0]:
        assert torch.equal(runs[False][0][k], runs[True][0][k])
        fam = runs[True][1][k].get("families")
        if fam is None:
            continue
        seen += 1
        for name in ("output_contrast", "gradient_contrast", "input_contrast"):
            for rho in (f"{x:g}" for x in R.FAMILY_RHOS):
                assert abs(fam[name][rho]["forget_share"] - fam["recast_forget_share"]) < 1e-6
                assert fam[name][rho]["retain_share"] >= 0.0    # the input family can exceed 1 (P C P is not <= C)
    assert seen > 0


def test_family_contraction_in_warm_start():
    """family_contraction replaces each update by the family's removal of the ReCAST update's forget output share."""
    import recast_direction_families as RF
    import recast_output_contrast as RO
    gen = torch.Generator().manual_seed(70)
    gr = {f"model.layers.{i}.{p}": torch.randn(6, 6, generator=gen) for i in range(2) for p in ("q_proj", "down_proj")}
    src = {k: torch.randn(6, 6, generator=gen) for k in gr}
    xs = {k: torch.randn(50, 6, generator=gen) for k in gr}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in gr}
    extra = dict(whitening_rho=1.0, representation_output="gradient", forget_covariances=cov_f,
                 retain_covariances=cov_r, rank_multiplier=1.25)
    runs = {}
    for fam in (None, "input_contrast", "output_contrast"):
        torch.manual_seed(71)
        t = R.select_targets(Toy(), "decoder_linear")
        w0 = {k: v.weight.detach().clone() for k, v in t.items()}
        kw = {} if fam is None else dict(family_contraction=fam, family_rho=1.0)
        _, rep = R.warm_start(t, gr, 0.3, src, 0.9, **extra, **kw)
        runs[fam] = ({k: v.weight.detach() - w0[k] for k, v in t.items()}, rep, w0)
    for fam in ("input_contrast", "output_contrast"):
        seen = 0
        for k, d in runs[fam][0].items():
            r = runs[fam][1][k].get("family_contraction")
            if r is None:
                continue
            seen += 1
            share = RO.output_energy_report(runs[None][2][k], runs[None][0][k], cov_f[k], cov_r[k])["forget_removed_share"]
            assert abs(r["recast_forget_share"] - share) < 1e-5
            expect, _ = RF.family_contraction_delta(runs[fam][2][k], cov_f[k], cov_r[k], fam, 1.0, r["recast_forget_share"])
            assert torch.allclose(d, expect.float(), atol=1e-5)
        assert seen > 0
    with pytest.raises(ValueError):
        R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, **extra,
                     family_contraction="input_contrast")                 # no ridge


def test_output_only_basis_restricts_only_the_output_side():
    """output_only: Vh = I, U from the output-covariance contrast (as the representation side), so P(D) = U U^T D."""
    gen = torch.Generator().manual_seed(80)
    w = torch.randn(5, 7, generator=gen)
    g = torch.randn(5, 7, generator=gen)
    xf, xr = torch.randn(60, 7, generator=gen), torch.randn(60, 7, generator=gen)
    cf, cr = xf.T @ xf / 60, xr.T @ xr / 60
    b = R.representation_basis(g, cf, cr, w, rank=3, rho=1.0, output_side="output_only")
    rep = R.representation_basis(g, cf, cr, w, rank=3, rho=1.0, output_side="representation")
    assert b.active and torch.equal(b.vh, torch.eye(7)) and b.u.shape == (5, 3)
    assert torch.allclose(b.u, rep.u, atol=1e-6)                          # the same output-side contrast
    d = torch.randn(5, 7, generator=gen)
    assert torch.allclose(R.project(d, b.u, b.vh), b.u @ (b.u.T @ d), atol=1e-6)


def _write_targets(seed):
    """down_proj (6 x 9) and o_proj (6 x 6) of two blocks, with gradients and input covariances."""
    torch.manual_seed(seed)
    t = {}
    for i in range(2):
        t[f"model.layers.{i}.self_attn.o_proj"] = nn.Linear(6, 6, bias=False)
        t[f"model.layers.{i}.mlp.down_proj"] = nn.Linear(9, 6, bias=False)
    gen = torch.Generator().manual_seed(seed + 1)
    gr = {k: torch.randn(m.weight.shape, generator=gen) for k, m in t.items()}
    src = {k: torch.randn(m.weight.shape, generator=gen) for k, m in t.items()}
    xs = {k: torch.randn(60, m.weight.shape[1], generator=gen) for k, m in t.items()}
    cov_f = {k: x.T @ x / 60 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(m.weight.shape[1], generator=gen) + 0.1) for k, m in t.items()}
    return t, gr, src, cov_f, cov_r


def test_global_family_in_warm_start():
    """global_family removes, over all targets together, the forget output energy the ReCAST updates remove, with the
    deltas of recast_global_families; score and random orders differ; other targets or a missing ridge are refused."""
    import recast_global_families as RG
    import recast_output_contrast as RO
    extra = dict(whitening_rho=1.0, representation_output="gradient", rank_multiplier=1.0)
    t, gr, src, cov_f, cov_r = _write_targets(90)
    w0 = {k: m.weight.detach().clone() for k, m in t.items()}
    _, ref = R.warm_start(t, gr, 0.1, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r)
    recast = sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) for k in t)
    for family in RG.FAMILIES:
        applied = {}
        for order in ("score", "random"):
            t, gr, src, cov_f, cov_r = _write_targets(90)
            _, rep = R.warm_start(t, gr, 0.1, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                                  global_family=family, global_family_rho=0.5, global_family_floor=0.4,
                                  global_family_order=order, component_seed=3)
            deltas = {k: t[k].weight.detach() - w0[k] for k in t}
            g = rep[next(iter(t))]["global_family"]["global"]
            assert abs(g["budget"] - recast) < 1e-4 * recast and g["reached"]
            assert abs(sum(RO.output_energy(d, cov_f[k]) for k, d in deltas.items()) - recast) < 1e-4 * recast
            for k, d in deltas.items():
                r = rep[k]["global_family"]
                gen = (torch.Generator().manual_seed(3 * 1000003 + R.zlib.crc32(k.encode())) if order == "random"
                       else None)
                part = RG.target_directions(w0[k].float(), cov_f[k], cov_r[k], src[k], gr[k], family, 0.5,
                                            keys_generator=gen)
                expect = RG.global_delta(part, w0[k].float(), r["directions"], 0.6, r["beta_last"])
                assert torch.allclose(d, expect.float(), atol=1e-5)
            applied[order] = deltas
        assert any(not torch.allclose(applied["score"][k], applied["random"][k]) for k in t)
    for family in RG.FAMILIES:                                                                  # energy_ratio order
        t, gr, src, cov_f, cov_r = _write_targets(90)
        _, rep = R.warm_start(t, gr, 0.1, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                              global_family=family, global_family_rho=0.5, global_family_floor=0.04,
                              global_family_order="energy_ratio", global_family_alpha=0.5)
        assert abs(sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) for k in t) - recast) < 1e-4 * recast
        for k in t:
            r = rep[k]["global_family"]
            part = RG.target_directions(w0[k].float(), cov_f[k], cov_r[k], src[k], gr[k], family, 0.5, key_alpha=0.5)
            expect = RG.global_delta(part, w0[k].float(), r["directions"], 0.96, r["beta_last"])
            assert torch.allclose(t[k].weight.detach() - w0[k], expect.float(), atol=1e-5)
        assert rep[k]["global_family"]["global"]["alpha"] == 0.5
    t, gr, src, cov_f, cov_r = _write_targets(90)
    with pytest.raises(ValueError):
        R.warm_start(t, gr, 0.1, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                     global_family="input_contrast", global_family_floor=0.4)                  # no ridge
    with pytest.raises(ValueError):
        R.warm_start(t, gr, 0.1, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                     global_family="output_contrast", global_family_rho=0.5, global_family_floor=0.4,
                     global_family_order="energy_ratio")                                       # no alpha
    q = R.select_targets(Toy(), "decoder_linear")                                              # q_proj targets
    gq = {k: torch.randn(m.weight.shape) for k, m in q.items()}
    cq = {k: torch.eye(m.weight.shape[1]) for k, m in q.items()}
    with pytest.raises(ValueError):
        R.warm_start(q, gq, 0.1, gq, 0.9, **extra, forget_covariances=cq, retain_covariances=cq,
                     global_family="output_contrast", global_family_rho=0.5, global_family_floor=0.4)


def test_global_family_share_unit_admits_other_types():
    """With the share budget unit up_proj and v_proj targets join down_proj and o_proj; the summed shares of their removed
    forget output energy equal the summed shares of the ReCAST updates. The energy unit still refuses them."""
    import recast_output_contrast as RO
    torch.manual_seed(95)
    def make():
        torch.manual_seed(95)
        t = {}
        for i in range(2):
            t[f"model.layers.{i}.self_attn.v_proj"] = nn.Linear(6, 3, bias=False)
            t[f"model.layers.{i}.self_attn.o_proj"] = nn.Linear(6, 6, bias=False)
            t[f"model.layers.{i}.mlp.up_proj"] = nn.Linear(6, 9, bias=False)
            t[f"model.layers.{i}.mlp.down_proj"] = nn.Linear(9, 6, bias=False)
        return t
    t = make()
    gen = torch.Generator().manual_seed(96)
    gr = {k: torch.randn(m.weight.shape, generator=gen) for k, m in t.items()}
    src = {k: torch.randn(m.weight.shape, generator=gen) for k, m in t.items()}
    xs = {k: torch.randn(60, m.weight.shape[1], generator=gen) for k, m in t.items()}
    cov_f = {k: x.T @ x / 60 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(m.weight.shape[1], generator=gen) + 0.1) for k, m in t.items()}
    extra = dict(whitening_rho=1.0, representation_output="gradient", rank_multiplier=1.0, forget_covariances=cov_f,
                 retain_covariances=cov_r)
    w0 = {k: m.weight.detach().clone() for k, m in t.items()}
    R.warm_start(t, gr, 0.2, src, 0.9, **extra)
    recast = sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) / RO.output_energy(w0[k], cov_f[k]) for k in t)
    t = make()
    _, rep = R.warm_start(t, gr, 0.2, src, 0.9, **extra, global_family="input_contrast", global_family_rho=1.0,
                          global_family_floor=0.04, global_family_budget_unit="share")
    got = sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) / RO.output_energy(w0[k], cov_f[k]) for k in t)
    g = rep[next(iter(t))]["global_family"]["global"]
    assert g["unit"] == "share" and g["reached"]
    assert abs(g["budget"] - recast) < 1e-4 * recast and abs(got - recast) < 1e-4 * recast
    with pytest.raises(ValueError):
        R.warm_start(make(), gr, 0.2, src, 0.9, **extra, global_family="input_contrast", global_family_rho=1.0,
                     global_family_floor=0.04)                                         # energy unit, up / v targets


def test_global_family_per_direction_s_in_warm_start():
    """global_family_s_gamma contracts each chosen direction by its own c_j and still removes the ReCAST total."""
    import recast_global_families as RG
    import recast_output_contrast as RO
    extra = dict(whitening_rho=1.0, representation_output="gradient", rank_multiplier=1.0)
    t, gr, src, cov_f, cov_r = _write_targets(97)
    w0 = {k: m.weight.detach().clone() for k, m in t.items()}
    R.warm_start(t, gr, 0.2, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r)
    recast = sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) for k in t)
    t, gr, src, cov_f, cov_r = _write_targets(97)
    _, rep = R.warm_start(t, gr, 0.2, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                          global_family="input_contrast", global_family_rho=0.5, global_family_floor=0.04,
                          global_family_s_gamma=1.0)
    g = rep[next(iter(t))]["global_family"]["global"]
    assert g["s_gamma"] == 1.0
    if g["reached"]:
        assert abs(sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) for k in t) - recast) < 1e-4 * recast
    for k in t:
        r = rep[k]["global_family"]
        part = RG.target_directions(w0[k].float(), cov_f[k], cov_r[k], src[k], gr[k], "input_contrast", 0.5,
                                    s_gamma=1.0, s_min=0.04)
        expect = RG.global_delta(part, w0[k].float(), r["directions"], 1.0, r["beta_last"])
        assert torch.allclose(t[k].weight.detach() - w0[k], expect.float(), atol=1e-5)
    with pytest.raises(ValueError):
        R.warm_start(t, gr, 0.2, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                     global_family="input_contrast", global_family_rho=0.5, global_family_floor=0.04,
                     global_family_s_gamma=-1.0)


def test_global_family_calibrated_threshold_in_warm_start():
    """global_family_s_threshold_gamma removes the ReCAST total with one threshold lambda* over all targets."""
    import recast_global_families as RG
    import recast_output_contrast as RO
    extra = dict(whitening_rho=1.0, representation_output="gradient", rank_multiplier=1.0)
    t, gr, src, cov_f, cov_r = _write_targets(98)
    w0 = {k: m.weight.detach().clone() for k, m in t.items()}
    R.warm_start(t, gr, 0.3, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r)
    recast = sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) for k in t)
    t, gr, src, cov_f, cov_r = _write_targets(98)
    _, rep = R.warm_start(t, gr, 0.3, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                          global_family="input_contrast", global_family_rho=0.5, global_family_floor=0.04,
                          global_family_s_threshold_gamma=2.0)
    g = rep[next(iter(t))]["global_family"]["global"]
    assert g["s_threshold_gamma"] == 2.0 and g["threshold"] > 0
    if g["reached"]:
        assert abs(sum(RO.output_energy(t[k].weight.detach() - w0[k], cov_f[k]) for k in t) - recast) < 1e-3 * recast
    with pytest.raises(ValueError):
        R.warm_start(t, gr, 0.3, src, 0.9, **extra, forget_covariances=cov_f, retain_covariances=cov_r,
                     global_family="input_contrast", global_family_rho=0.5, global_family_floor=0.04,
                     global_family_s_threshold_gamma=2.0, global_family_s_gamma=1.0)
