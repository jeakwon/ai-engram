"""CPU tests of the learned direction gates (recast_direction_gates.py). Run on a compute node: they import torch."""

import copy
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import recast_core as R  # noqa: E402
import recast_direction_gates as G  # noqa: E402
from recast_direction_families import ordered_directions  # noqa: E402
import recast_selective as S  # noqa: E402


def _covs(gen, n, tokens=300):
    xf = torch.randn(tokens, n, generator=gen, dtype=torch.float64) * torch.linspace(0.2, 2.0, n, dtype=torch.float64)
    xr = torch.randn(tokens, n, generator=gen, dtype=torch.float64)
    return xf.T @ xf / tokens, xr.T @ xr / tokens


def _tiny_llama(seed=0, layers=2):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                                   num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False,
                                   max_position_embeddings=32)
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(cfg).eval()


def _items(gen, n, length=8, answer_from=3):
    """QADataset-like items: the first ``answer_from`` positions are the question (label -100)."""
    out = []
    for i in range(n):
        ids = torch.randint(0, 37, (length,), generator=gen)
        labels = ids.clone()
        labels[:answer_from] = G.IGNORE_INDEX
        out.append({"input_ids": ids, "labels": labels, "attention_mask": torch.ones(length, dtype=torch.long),
                    "index": i})
    return out


def _collate(items):
    return {k: torch.stack([x[k] for x in items]) for k in ("input_ids", "labels", "attention_mask")}


def _gates(model, basis="input_contrast", k=6, seed=0):
    targets = R.select_targets(model, "down_o")
    gen = torch.Generator().manual_seed(seed)
    dirs, writes = {}, {}
    for n, m in targets.items():
        cf, cr = _covs(gen, m.in_features)
        v, _, u = G.gate_directions(cf, cr, basis, k, rho=1.0 if basis in G.CONTRAST_BASES else None,
                                    generator=torch.Generator().manual_seed(seed + 1) if basis == "random" else None)
        dirs[n] = v
        if u is not None:
            writes[n] = u
    return targets, G.DirectionGates(targets, dirs, writes)


def _identity(x):
    return x


def test_gate_directions_per_basis():
    """input_contrast: the leading k columns of ordered_directions with their lambdas; forget_input: the leading
    eigenvectors of C_f (V^T C_f V diagonal, descending, scores the shares of tr C_f); random: seeded; all
    orthonormal, k clipped to the width; argument errors."""
    gen = torch.Generator().manual_seed(1)
    cf, cr = _covs(gen, 10)
    d, lam = ordered_directions(cf, cr, 1.0, return_values=True)
    v, s, u = G.gate_directions(cf, cr, "input_contrast", 4, rho=1.0)
    assert torch.allclose(v, d[:, :4]) and torch.allclose(s, lam[:4]) and u is None
    v, s, u = G.gate_directions(cf, None, "forget_input", 4)
    assert u is None
    m = v.T @ cf @ v
    top = torch.linalg.eigvalsh(cf).flip(0)[:4]
    assert torch.allclose(torch.diagonal(m), top, atol=1e-10) and torch.allclose(m, torch.diag(top), atol=1e-10)
    assert torch.allclose(s, top / torch.trace(cf))
    v1, s1, _ = G.gate_directions(cf, None, "random", 4, generator=torch.Generator().manual_seed(3))
    v2, _, _ = G.gate_directions(cf, None, "random", 4, generator=torch.Generator().manual_seed(3))
    assert torch.equal(v1, v2) and torch.isnan(s1).all()
    for basis, kw in (("input_contrast", {"rho": 1.0}), ("forget_input", {}),
                      ("random", {"generator": torch.Generator().manual_seed(0)})):
        v, _, _ = G.gate_directions(cf, cr, basis, 50, **kw)
        assert v.shape == (10, 10)
        assert torch.allclose(v.T @ v, torch.eye(10, dtype=v.dtype), atol=1e-10)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "forget_input", 4, rho=1.0)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "input_contrast", 4)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "input_contrast_dual", 4)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "random", 4)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "forget_input", 0)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "svd", 4)


def test_unlikelihood_is_the_stable_form():
    """-log(1 - p_y) over the shifted answer positions (labels -100 ignored), >= 0, finite for extreme logits, with
    gradient p_y on the true logit; nll_sum is the summed cross-entropy over the same positions."""
    gen = torch.Generator().manual_seed(2)
    logits = torch.randn(2, 5, 7, generator=gen, dtype=torch.float64)
    labels = torch.randint(0, 7, (2, 5), generator=gen)
    labels[0, :2] = G.IGNORE_INDEX
    total, count = G.unlikelihood_sum(logits, labels)
    z, y = logits[:, :-1], labels[:, 1:]
    mask = y != G.IGNORE_INDEX
    p = torch.softmax(z.float(), -1)[mask].gather(1, y[mask].unsqueeze(1)).squeeze(1)
    assert count == int(mask.sum()) == 7
    assert math.isclose(float(total), float(-torch.log1p(-p).sum()), rel_tol=1e-5)
    nll, _ = G.nll_sum(logits, labels)
    assert math.isclose(float(nll), float(-torch.log(p).sum()), rel_tol=1e-5)
    # extreme logits: the memorised token far ahead of every other one
    big = torch.zeros(1, 2, 7)
    big[0, 0, 3] = 1.0e4
    lab = torch.tensor([[0, 3]])
    val, _ = G.unlikelihood_sum(big, lab)
    assert math.isfinite(float(val)) and float(val) > 1.0e3
    zt = torch.randn(1, 2, 7, generator=gen, requires_grad=True)
    lab = torch.tensor([[0, 4]])
    val, _ = G.unlikelihood_sum(zt, lab)
    val.backward()
    p_y = torch.softmax(zt.detach()[0, 0], -1)[4]
    assert torch.isclose(zt.grad[0, 0, 4], p_y, atol=1e-6) and float(val) >= 0.0
    assert torch.all(zt.grad[0, 1] == 0)  # the last position predicts nothing


def test_gates_at_one_are_bitwise_the_model():
    """Every gate at 1: the gated logits equal the plain ones bitwise, with and without bfloat16 autocast; hooks are
    removed after the check, attaching twice raises."""
    model = _tiny_llama()
    _, gates = _gates(model)
    x = _collate(_items(torch.Generator().manual_seed(4), 3))
    for ac in (None, torch.bfloat16):
        G.check_identity(model, gates, x, ac)
    assert not gates._handles
    gates.attach()
    with pytest.raises(RuntimeError):
        gates.attach()
    gates.detach()
    with torch.no_grad():
        gates.gates[0][0] = 0.5
    with pytest.raises(RuntimeError):
        G.check_identity(model, gates, x)


def test_gate_gradient_and_regulariser():
    """On one linear target: dJ/ds_j = a_j^T G v_j with G = dJ/dW_eff (autograd on the materialized weight), equal to a
    central difference; R * sum ||W0||^2 = ||W_eff - W0||_F^2 (orthogonal parts); materialize writes W_eff."""
    gen = torch.Generator().manual_seed(5)
    lin = nn.Linear(6, 4, bias=False).double()
    with torch.no_grad():
        lin.weight.copy_(torch.randn(4, 6, generator=gen, dtype=torch.float64))
    w0 = lin.weight.detach().clone()
    cf, cr = _covs(gen, 6)
    v, _, _ = G.gate_directions(cf, cr, "input_contrast", 4, rho=1.0)
    gates = G.DirectionGates({"t": lin}, {"t": v})
    with torch.no_grad():
        gates.gates[0].copy_(torch.tensor([0.9, 0.3, 1.0, 0.0]))
    x = torch.randn(9, 6, generator=gen, dtype=torch.float64)
    target = torch.randn(9, 4, generator=gen, dtype=torch.float64)
    a, vv = gates.parts(0)[1].double(), gates.parts(0)[0].double()

    def loss(s):
        w_eff = w0 - (a * (1.0 - s)) @ vv.T
        return ((x @ w_eff.T - target) ** 2).sum(), w_eff

    gates.attach()
    out = lin(x)
    gates.detach()
    j = ((out - target) ** 2).sum()
    j.backward()
    s0 = gates.gates[0].detach().double()
    w_eff = (w0 - (a * (1.0 - s0)) @ vv.T).requires_grad_(True)
    ((x @ w_eff.T - target) ** 2).sum().backward()
    analytic = torch.einsum("oj,oi,ij->j", a, w_eff.grad, vv)
    assert torch.allclose(gates.gates[0].grad.double(), analytic, rtol=1e-4, atol=1e-4)
    h = 1e-5
    for jx in range(4):
        e = torch.zeros(4, dtype=torch.float64)
        e[jx] = h
        fd = (loss(s0 + e)[0] - loss(s0 - e)[0]) / (2 * h)
        assert math.isclose(float(analytic[jx]), float(fd), rel_tol=1e-5, abs_tol=1e-6)
    reg = float(gates.regulariser())
    disp = float(((a * (1.0 - s0)) @ vv.T).square().sum())
    assert math.isclose(reg * gates.norm2, disp, rel_tol=1e-5)
    rep = gates.materialize()
    assert torch.allclose(lin.weight.detach(), loss(s0)[1], atol=1e-6)
    assert math.isclose(rep["t"]["displacement"], math.sqrt(disp), rel_tol=1e-5)


def test_removed_energy_matches_the_dense_update():
    """tr(dW C dW^T) from the k x k blocks equals the dense formula; shares against tr(W0 C W0^T)."""
    gen = torch.Generator().manual_seed(6)
    lin = nn.Linear(7, 5, bias=False)
    cf, cr = _covs(gen, 7)
    v, _, _ = G.gate_directions(cf, cr, "input_contrast", 5, rho=1.0)
    gates = G.DirectionGates({"t": lin}, {"t": v})
    with torch.no_grad():
        gates.gates[0].copy_(torch.rand(5, generator=gen))
    rep = G.removed_energy(gates, {"t": cf.float()}, {"t": cr.float()})["t"]
    w = lin.weight.detach().double()
    vv, a = gates.parts(0)[0].double(), gates.parts(0)[1].double()
    delta = -(a * (1.0 - gates.gates[0].detach().double())) @ vv.T
    for side, c in (("forget", cf), ("retain", cr)):
        assert math.isclose(rep[f"{side}_energy_removed"], float(torch.trace(delta @ c @ delta.T)), rel_tol=1e-4)
        assert math.isclose(rep[f"{side}_energy"], float(torch.trace(w @ c @ w.T)), rel_tol=1e-6)
        assert math.isclose(rep[f"{side}_share"], rep[f"{side}_energy_removed"] / rep[f"{side}_energy"])


def test_materialize_reproduces_the_gated_model():
    """Gates in [0, 1]: the materialized model's logits match the gated ones, only the targets change, hooks gone."""
    model = _tiny_llama(seed=1)
    targets, gates = _gates(model, basis="forget_input")
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for g in gates.gates:
            g.copy_(torch.rand(g.shape, generator=gen))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x = _collate(_items(gen, 3))
    out = G.materialize_checked(model, gates, x)
    assert out["check"]["relative"] <= 1e-5 and out["check"]["argmax_agreement"] == 1.0
    assert not gates._handles
    names = {n + ".weight" for n in targets}
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]) != (n in names)
    assert set(out["per_target"]) == set(targets)


def test_forget_anchored_windows():
    """Each epoch covers every forget index once in windows of the given size; partners in range; seeded."""
    w = G.forget_anchored_windows(10, 4, 4, 2, torch.Generator().manual_seed(0))
    assert [len(f) for _, f, _ in w] == [4, 4, 2, 4, 4, 2] and [e for e, _, _ in w] == [0, 0, 0, 1, 1, 1]
    for epoch in (0, 1):
        idx = torch.cat([f for e, f, _ in w if e == epoch])
        assert sorted(idx.tolist()) == list(range(10))
    assert all(int(r.min()) >= 0 and int(r.max()) < 4 for _, _, r in w)
    w2 = G.forget_anchored_windows(10, 4, 4, 2, torch.Generator().manual_seed(0))
    assert all(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) for a, b in zip(w, w2))
    with pytest.raises(ValueError):
        G.forget_anchored_windows(10, 4, 0, 2, torch.Generator())


def _train(model, gates, forget, retain, **kw):
    opts = {"max_epochs": 6, "convergence_epochs": 6, "tol": 0.0, "drift_max": 1.0e-9, "micro_batch": 2,
            "grad_accum": 2, "optim": "adam", "lr": 0.05, "alpha": 1.0, "beta": 0.1, "probe_retain": 5, "seed": 3}
    opts.update(kw)
    return G.train_gates(model, gates, forget, retain, _collate, _identity, **opts)


def test_train_gates_moves_only_the_gates():
    """The gate phase: every model parameter bitwise unchanged and its requires_grad flag restored, gates in [0, 1]
    and moved, one record per window and per epoch (epoch 0 the start), snapshots and per-gate path and reversal
    counts consistent with the records, and the same result for the same seed."""
    model = _tiny_llama(seed=2)
    gen = torch.Generator().manual_seed(8)
    forget, retain = _items(gen, 10), _items(gen, 20)
    runs = []
    for _ in range(2):
        m = copy.deepcopy(model)
        m.model.embed_tokens.weight.requires_grad_(False)  # a flag to restore
        flags = {n: p.requires_grad for n, p in m.named_parameters()}
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        _, gates = _gates(m)
        out = _train(m, gates, forget, retain)
        steps, epochs = out["steps"], out["epochs"]
        assert out["stop"] == "max_epochs" and not out["converged"] and out["epochs_run"] == 6
        assert len(steps) == 6 * 3 and steps[-1]["epoch"] == 6 and [e["epoch"] for e in epochs] == list(range(7))
        assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())
        assert {n: p.requires_grad for n, p in m.named_parameters()} == flags and not m.training
        s = torch.cat([g.detach() for g in gates.gates])
        assert s.dtype == torch.float64 and float(s.min()) >= 0.0 and float(s.max()) <= 1.0 and bool((s < 1).any())
        assert not gates._handles
        snaps = out["contraction_snapshots"]
        assert snaps.shape == (7, s.numel()) and float(snaps[0].abs().max()) == 0.0
        assert torch.allclose(snaps[-1].double(), 1.0 - s, atol=1e-6)
        assert float(out["gate_path"].sum()) > 0.0
        assert float(out["gate_path"].sum()) >= float((1.0 - s).sum()) - 1e-9  # the path is at least the net change
        assert int(out["gate_reversals"].max()) <= len(steps) - 1
        assert all(e["window_converged"] is False for e in epochs[6:]) and "window_drift" not in epochs[5]
        assert len(out["probe_retain_indices"]) == 5 and len(set(out["probe_retain_indices"])) == 5
        runs.append(out)
    a, b = runs
    assert a["steps"] == b["steps"] and a["epochs"] == b["epochs"]
    assert torch.equal(a["contraction_snapshots"], b["contraction_snapshots"])
    for bad in ({"lr": 0.0}, {"optim": "lion"}, {"convergence_epochs": 7}, {"probe_retain": 21}, {"drift_max": 0.0},
                {"tol": -1.0}):
        with pytest.raises(ValueError):
            _train(model, gates, forget, retain, **bad)
        assert all(p.requires_grad for p in model.parameters()) and not gates._handles


def _term_gradients(model, gates, forget, retain, fidx, ridx):
    """At the current gates, for one window (forget items fidx, retain items ridx): the gradients of L_F and of L_R
    in the gates, each alone, and the two mean losses."""
    out = []
    gates.attach()
    try:
        for items, idx, loss_fn in ((forget, fidx, G.unlikelihood_sum), (retain, ridx, G.nll_sum)):
            for p in gates.gates:
                p.grad = None
            x = _collate([items[int(i)] for i in idx])
            total, n = loss_fn(model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits, x["labels"])
            (total / n).backward()
            out.append((torch.cat([p.grad.detach().flatten() for p in gates.gates]), float(total) / n))
    finally:
        gates.detach()
        for p in gates.gates:
            p.grad = None
    (g_f, uf), (g_r, nr) = out
    return g_f, g_r, uf, nr


def _window_gradient(model, gates, forget, retain, fidx, ridx, alpha, gamma=0.0):
    """dJ/ds at the current gates for one window, J = L_R + alpha L_F + gamma O, and the window's mean losses."""
    g_f, g_r, uf, nr = _term_gradients(model, gates, forget, retain, fidx, ridx)
    return g_r + alpha * g_f + gamma, uf, nr


def test_gate_optimizer_steps():
    """One window per epoch: SGD makes exactly s <- clip(1 - lr g); Adam's first step moves every gate pushed down by
    lr (to rounding) and leaves those pushed up at 1. The step record carries the window losses, the gradient of each
    term (the regulariser's 0 at s = 1) and the update; the probe of epoch 0 is the mean NLL of the drawn retain
    items."""
    model = _tiny_llama(seed=4)
    gen = torch.Generator().manual_seed(10)
    forget, retain = _items(gen, 4), _items(gen, 6)
    seed, probe_retain = 5, 3
    g_draw = torch.Generator().manual_seed(seed)
    probe_idx = torch.randperm(len(retain), generator=g_draw)[:probe_retain]
    (_, fidx, ridx), = G.forget_anchored_windows(len(forget), len(retain), 4, 1, g_draw)
    for optim, lr in (("sgd", 0.5), ("adam", 1.0e-3)):
        m = copy.deepcopy(model)
        _, gates = _gates(m)
        grad, uf, nr = _window_gradient(m, gates, forget, retain, fidx, ridx, alpha=1.0)
        out = _train(m, gates, forget, retain, max_epochs=1, convergence_epochs=1, micro_batch=4, grad_accum=1,
                     optim=optim, lr=lr, beta=0.1, probe_retain=probe_retain, seed=seed)
        s = torch.cat([g.detach() for g in gates.gates])
        rec = out["steps"][0]
        assert len(out["steps"]) == 1 and out["probe_retain_indices"] == probe_idx.tolist()
        assert math.isclose(rec["forget_unlikelihood"], uf, rel_tol=1e-6)
        assert math.isclose(rec["retain_nll"], nr, rel_tol=1e-6)
        assert rec["grad_regulariser_l2"] == 0.0 and math.isclose(rec["grad_l2"], float(grad.norm()), rel_tol=1e-5)
        if optim == "sgd":
            assert torch.allclose(s, (1.0 - lr * grad).clamp(0.0, 1.0), atol=1e-6)
        else:  # the first Adam step is lr g / (|g| + eps): no gate moves by more than lr
            assert torch.allclose(s, (1.0 - lr * grad / (grad.abs() + 1e-8)).clamp(0.0, 1.0), atol=1e-9)
            assert float((1.0 - s).max()) <= lr * (1.0 + 1e-9) and bool((grad > 0).any())
            assert bool((s[grad <= 0] == 1.0).all())
        assert math.isclose(rec["update_l2"], float((1.0 - s).norm()), rel_tol=1e-9)
        assert rec["update_cosine"] is None and rec["gates_moving"] == int((s != 1).sum())
        xr = _collate([retain[int(j)] for j in probe_idx])
        nll, n = G.nll_sum(model(input_ids=xr["input_ids"], attention_mask=xr["attention_mask"]).logits, xr["labels"])
        assert math.isclose(out["epochs"][0]["probe_retain_nll"], float(nll) / n, rel_tol=1e-6)
    # the closing term alone (alpha 0): SGD makes s <- clip(1 - lr (g_R + gamma)), the closing gradient is gamma
    m = copy.deepcopy(model)
    _, gates = _gates(m)
    gamma, lr = 0.3, 0.5
    grad, _, _ = _window_gradient(m, gates, forget, retain, fidx, ridx, alpha=0.0, gamma=gamma)
    out = _train(m, gates, forget, retain, max_epochs=1, convergence_epochs=1, micro_batch=4, grad_accum=1,
                 optim="sgd", lr=lr, alpha=0.0, beta=0.0, gamma=gamma, probe_retain=probe_retain, seed=seed)
    rec, s = out["steps"][0], torch.cat([g.detach() for g in gates.gates])
    assert rec["grad_forget_l2"] == 0.0 and rec["grad_regulariser_l2"] == 0.0 and rec["grad_retain_l2"] > 0.0
    assert math.isclose(rec["grad_closing_l2"], gamma * math.sqrt(s.numel()), rel_tol=1e-9)
    assert rec["openness"] == float(s.numel()) and rec["gates_pushed_shut"] == int((grad > 0).sum())
    assert torch.allclose(s, (1.0 - lr * grad).clamp(0.0, 1.0), atol=1e-6)
    e0 = out["epochs"][0]
    assert math.isclose(e0["probe_objective"], e0["probe_retain_nll"] + gamma * s.numel(), rel_tol=1e-9)
    assert math.isclose(e0["probe_openness"], s.numel())
    with pytest.raises(ValueError):  # nothing pushes the gates shut
        _train(m, gates, forget, retain, alpha=0.0, gamma=0.0)


def test_update_stats_and_convergence_check():
    """Update norms, moving gates, the cosine with the previous update and the share of reversals among the gates
    moved by both; the convergence rule on the window's probe objectives and the drift ratio."""
    d = torch.tensor([0.1, -0.2, 0.0, 0.3], dtype=torch.float64)
    p = torch.tensor([0.1, 0.2, 0.5, -0.1], dtype=torch.float64)
    first = G.update_stats(d, None)
    assert first["update_cosine"] is None and first["reversal_share"] is None and first["gates_moving"] == 3
    assert math.isclose(first["update_l2"], math.sqrt(0.14)) and math.isclose(first["update_max"], 0.3)
    rec = G.update_stats(d, p)
    assert math.isclose(rec["update_cosine"], float(d @ p) / float(d.norm() * p.norm()))
    assert math.isclose(rec["reversal_share"], 2.0 / 3.0)  # gates 0, 1, 3 moved twice; 1 and 3 reversed
    assert G.update_stats(torch.zeros(3, dtype=torch.float64), p[:3])["reversal_share"] is None
    c = G.convergence_check([2.0, 1.9, 1.999], net=0.1, path=1.0, tol=1.0e-3, drift_max=0.5)
    assert math.isclose(c["objective_decrease"], 0.0005) and math.isclose(c["drift"], 0.1) and c["converged"]
    assert not G.convergence_check([2.0, 1.0], net=0.1, path=1.0, tol=1.0e-3, drift_max=0.5)["converged"]
    assert not G.convergence_check([2.0, 2.0], net=0.9, path=1.0, tol=1.0e-3, drift_max=0.5)["converged"]
    c = G.convergence_check([2.0, 2.1], net=0.0, path=0.0, tol=0.0, drift_max=0.5)
    assert math.isclose(c["objective_decrease"], -0.05) and c["drift"] == 0.0 and c["converged"]


def test_train_gates_stops_at_convergence():
    """The phase stops at the first epoch whose trailing window meets the rule (here every window does), with the
    window statistics in the epoch record; never earlier than convergence_epochs."""
    model = _tiny_llama(seed=5)
    gen = torch.Generator().manual_seed(11)
    forget, retain = _items(gen, 6), _items(gen, 6)
    _, gates = _gates(model)
    out = _train(model, gates, forget, retain, max_epochs=10, convergence_epochs=3, tol=1.0e9, drift_max=1.0,
                 probe_retain=6)
    assert out["stop"] == "converged" and out["converged"] and out["epochs_run"] == 3
    assert len(out["steps"]) == 3 * 2 and out["contraction_snapshots"].shape[0] == 4
    last = out["epochs"][-1]
    assert last["window_converged"] and 0.0 <= last["window_drift"] <= 1.0 + 1e-12
    assert math.isclose(last["window_objective_decrease"],
                        (out["epochs"][0]["probe_objective"] - last["probe_objective"])
                        / out["epochs"][0]["probe_objective"])


def test_train_gates_descends_on_a_fixed_window():
    """The whole forget split in one window and a single retain item: J is a deterministic function of the gates (the
    probe J equals the window J), and small Adam steps lower it."""
    model = _tiny_llama(seed=3)
    gen = torch.Generator().manual_seed(9)
    forget, retain = _items(gen, 6), _items(gen, 1)
    _, gates = _gates(model)
    out = _train(model, gates, forget, retain, max_epochs=15, convergence_epochs=15, micro_batch=6, grad_accum=1,
                 lr=0.01, probe_retain=1, seed=0)
    assert math.isclose(out["epochs"][0]["probe_objective"], out["steps"][0]["objective"], rel_tol=1e-5)
    assert out["epochs"][-1]["probe_objective"] < out["epochs"][0]["probe_objective"]


def test_dual_basis_removes_one_eigen_coefficient_per_gate():
    """input_contrast_dual: the read directions span the k leading generalised eigenvectors (the span of the QR
    basis), each column's own ratio is lambda_j and the columns' forget parts do not mix; all n gates shut remove W0
    whole; the forget and the ridged-retain energies removed are sums over the gates; R and the summary use the true
    ||dW||^2 of the oblique parts; the gate gradient is a_j^T G v_j and materialize writes W_eff."""
    gen = torch.Generator().manual_seed(12)
    n, k, rho = 7, 4, 1.0
    cf, cr = _covs(gen, n)
    v, lam, u = G.gate_directions(cf, cr, "input_contrast_dual", k, rho=rho)
    q, lam_q, _ = G.gate_directions(cf, cr, "input_contrast", k, rho=rho)
    assert torch.allclose(lam, lam_q) and torch.allclose(v.norm(dim=0), torch.ones(k, dtype=v.dtype))
    p_dual = v @ torch.linalg.pinv(v)
    assert torch.allclose(p_dual, q @ q.T, atol=1e-8)  # the same subspace
    assert torch.allclose(u.T @ v, torch.eye(k, dtype=v.dtype), atol=1e-8)  # u_j^T v_i = delta_ij (dual pairs)
    spec = G.column_specificity(v, cf, cr, rho, lam)
    assert spec["max_rel_gap_to_lambda"] < 1e-8 and spec["offdiag_share"] < 1e-8
    assert math.isclose(spec["spearman_with_lambda"], 1.0)
    spec_q = G.column_specificity(q, cf, cr, rho, lam_q)
    assert math.isclose(float(spec_q["own_ratio"][0]), float(lam_q[0]), rel_tol=1e-8)  # u_1 is x_1
    assert spec_q["offdiag_share"] > 1e-6  # after the QR the columns' forget parts mix
    # every gate of the full dual basis shut: the input is removed whole
    vf, _, uf = G.gate_directions(cf, cr, "input_contrast_dual", n, rho=rho)
    lin = nn.Linear(n, 5, bias=False).double()
    with torch.no_grad():
        lin.weight.copy_(torch.randn(5, n, generator=gen, dtype=torch.float64))
    gates = G.DirectionGates({"t": lin}, {"t": vf}, {"t": uf})
    with torch.no_grad():
        gates.gates[0].zero_()
    reg = float(gates.regulariser())
    assert math.isclose(reg, 1.0, rel_tol=1e-5)  # ||W_eff - W0||^2 = ||W0||^2
    gates.materialize()
    assert float(lin.weight.detach().abs().max()) < 1e-5
    # partial gates: additive energies, true displacement, gradient
    lin = nn.Linear(n, 5, bias=False).double()
    with torch.no_grad():
        lin.weight.copy_(torch.randn(5, n, generator=gen, dtype=torch.float64))
    w0 = lin.weight.detach().clone()
    gates = G.DirectionGates({"t": lin}, {"t": v}, {"t": u})
    with torch.no_grad():
        gates.gates[0].copy_(torch.tensor([0.2, 1.0, 0.0, 0.7], dtype=torch.float64))
    c = 1.0 - gates.gates[0].detach()
    vv, a = gates.parts(0)[0].double(), gates.parts(0)[1].double()
    delta = -(a * c) @ vv.T
    assert torch.allclose(a, w0 @ u, atol=1e-6)
    b_raw = cr + rho * torch.trace(cr) / n * torch.eye(n, dtype=cr.dtype)
    rep = G.removed_energy(gates, {"t": cf}, {"t": b_raw})["t"]
    for side, cov in (("forget", cf), ("retain", b_raw)):
        additive = float((c * c * a.square().sum(0) * ((cov @ vv) * vv).sum(0)).sum())
        assert math.isclose(rep[f"{side}_energy_removed"], additive, rel_tol=1e-5)
        assert math.isclose(rep[f"{side}_energy_removed"], float(torch.trace(delta @ cov @ delta.T)), rel_tol=1e-5)
    assert math.isclose(float(gates.regulariser()) * gates.norm2, float(delta.square().sum()), rel_tol=1e-5)
    assert math.isclose(gates.summary()["t"]["displacement2"], float(delta.square().sum()), rel_tol=1e-5)
    x = torch.randn(6, n, generator=gen, dtype=torch.float64)
    gates.attach()
    out = lin(x)
    gates.detach()
    assert torch.allclose(out, x @ (w0 + delta).T, atol=1e-5)
    out.square().sum().backward()
    w_eff = (w0 + delta).requires_grad_(True)
    (x @ w_eff.T).square().sum().backward()
    assert torch.allclose(gates.gates[0].grad, torch.einsum("oj,oi,ij->j", a, w_eff.grad, vv), rtol=1e-4, atol=1e-4)
    gates.materialize()
    assert torch.allclose(lin.weight.detach(), w0 + delta, atol=1e-5)


def test_dual_gates_on_the_model():
    """The dual basis on the tiny model: identity at s = 1 and materialization within the tolerance."""
    model = _tiny_llama(seed=6)
    _, gates = _gates(model, basis="input_contrast_dual")
    x = _collate(_items(torch.Generator().manual_seed(13), 3))
    G.check_identity(model, gates, x)
    with torch.no_grad():
        for g in gates.gates:
            g.copy_(torch.rand(g.shape, generator=torch.Generator().manual_seed(14), dtype=g.dtype))
    assert G.materialize_checked(model, gates, x)["check"]["relative"] <= 1e-5


def test_gradient_census_and_summary():
    """The census rows are the per-window gradients of L_F and L_R alone at the current gates (no step, model and
    gates unchanged); the summary's guides from fixed rows."""
    model = _tiny_llama(seed=7)
    gen = torch.Generator().manual_seed(15)
    forget, retain = _items(gen, 6), _items(gen, 5)
    _, gates = _gates(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    census = G.gradient_census(model, gates, forget, retain, _collate, _identity, epochs=2, micro_batch=3,
                               grad_accum=1, seed=4)
    windows = G.forget_anchored_windows(6, 5, 3, 2, torch.Generator().manual_seed(4))
    assert census["windows"] == len(windows) == 4 and census["forget"].shape == (4, 2 * 6 + 2 * 6)
    for row, (_, fidx, ridx) in enumerate(windows):
        g_f, g_r, _, _ = _term_gradients(model, gates, forget, retain, fidx, ridx)
        assert torch.allclose(census["forget"][row].double(), g_f, rtol=1e-4, atol=1e-6)
        assert torch.allclose(census["retain"][row].double(), g_r, rtol=1e-4, atol=1e-6)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert all(bool((g == 1).all()) and g.grad is None for g in gates.gates) and not gates._handles
    # fixed rows: 2 windows x 4 gates
    fixed = {"windows": 2, "forget": torch.tensor([[1.0, -1.0, 0.0, 2.0], [1.0, 1.0, 0.0, 2.0]]),
             "retain": torch.tensor([[-0.5, 0.0, 0.2, -3.0], [-0.5, 0.0, -0.2, -3.0]])}
    out = G.census_summary(fixed, alpha=1.0, gamma=0.0)
    g = torch.tensor([[0.5, -1.0, 0.2, -1.0], [0.5, 1.0, -0.2, -1.0]], dtype=torch.float64)
    q99 = float(torch.quantile(g.abs().flatten(), torch.tensor(0.99, dtype=torch.float64)))
    assert math.isclose(out["sgd_lr_for_step"]["0.01"], 0.01 / q99)
    assert math.isclose(out["adam_step_per_lr"]["q50"], 0.5)  # per gate 1, 0, 0, 1
    assert math.isclose(out["pushed_shut_share"], 0.25)  # mean g: 0.5, 0, 0, -1
    assert math.isclose(out["pushed_shut_share_by_alpha"]["10"], 0.5)  # 9.5, 0, 0, 17
    price = torch.tensor([0.5, 0.0, 0.0, 3.0], dtype=torch.float64)
    assert math.isclose(out["closing_price"]["0.5"], float(torch.quantile(price, 0.5)))
    assert math.isclose(out["closing_free_share"], 0.0)
    shut = G.census_summary(fixed, alpha=0.0, gamma=1.0)  # mean g_R + 1: 0.5, 1, 1, -2
    assert math.isclose(shut["pushed_shut_share"], 0.75)


def test_reset_and_openness():
    model = _tiny_llama(seed=8)
    _, gates = _gates(model)
    with torch.no_grad():
        gates.gates[0][:3] = 0.25
    assert math.isclose(float(gates.openness()), sum(g.numel() for g in gates.gates) - 3 * 0.75)
    gates.reset_()
    assert all(bool((g == 1).all()) for g in gates.gates)


def test_binarize_by_share_removes_whole_directions():
    """The chosen gates are the most closed ones (never a gate at 1), the shortest such prefix whose removal at s_hard
    reaches the forget share (one gate fewer does not); every gate ends at s_hard or 1; the reported shares equal
    removed_energy of the binarized gates; an unreachable share raises."""
    model = _tiny_llama(seed=9)
    targets, gates = _gates(model)
    gen = torch.Generator().manual_seed(16)
    with torch.no_grad():
        for g in gates.gates:
            g.copy_(torch.rand(g.shape, generator=gen, dtype=g.dtype))
        gates.gates[0][0] = 1.0
    soft = torch.cat([g.detach().clone() for g in gates.gates])
    covs_f, covs_r = {}, {}
    for n, m in targets.items():
        covs_f[n], covs_r[n] = _covs(gen, m.in_features)
    with pytest.raises(ValueError):
        G.binarize_by_share(gates, covs_f, covs_r, share=0.999, s_hard=0.04)
    assert torch.equal(torch.cat([g.detach() for g in gates.gates]), soft)  # unchanged on the error
    rep = G.binarize_by_share(gates, covs_f, covs_r, share=0.3, s_hard=0.04)
    s = torch.cat([g.detach() for g in gates.gates])
    chosen = s == 0.04
    m = rep["gates_removed"]
    assert bool((chosen | (s == 1.0)).all()) and int(chosen.sum()) == m and not bool(chosen[0])
    order = torch.argsort(soft, stable=True)
    assert torch.equal(chosen, torch.zeros_like(chosen).index_fill_(0, order[:m], True))
    assert rep["threshold"] == float(soft[order[m - 1]]) and sum(rep["per_type"].values()) == m

    def shares():
        e = G.removed_energy(gates, covs_f, covs_r).values()
        return tuple(sum(r[f"{k}_energy_removed"] for r in e) / sum(r[f"{k}_energy"] for r in e)
                     for k in ("forget", "retain"))

    f, r = shares()
    assert f >= 0.3 and math.isclose(f, rep["forget_share"], rel_tol=1e-6)
    assert math.isclose(r, rep["retain_share"], rel_tol=1e-6)
    sizes = [g.numel() for g in gates.gates]
    last = int(order[m - 1])
    i, j = next((t, last - sum(sizes[:t])) for t in range(len(sizes)) if last < sum(sizes[:t + 1]))
    with torch.no_grad():
        gates.gates[i][j] = 1.0
    assert shares()[0] < 0.3


def test_protected_bases_avoid_the_prefix_span():
    """protected_contrast: orthonormal columns orthogonal to the protected span, spanning the leading eigenvectors of
    recast_selective.selective_directions' pencil, with its lambdas; protected_contrast_dual: unit read directions
    orthogonal to the span, dual pairs with the writes, own ratio lambda against the given retain covariance, no
    mixing; a protected span only with these bases."""
    gen = torch.Generator().manual_seed(17)
    n, k, rho = 12, 5, 1.0
    cf, cr = _covs(gen, n)
    prefix = torch.randn(3, n, generator=gen, dtype=torch.float64)
    b, info = S.protected_span(n, prefix_inputs=prefix)
    assert info["prefix_rank"] == 3
    ref, lam_ref = S.selective_directions(cf, cr, b, rho)
    v, lam, u = G.gate_directions(cf, cr, "protected_contrast", k, rho=rho, protected=b)
    assert u is None and torch.allclose(v.T @ v, torch.eye(k, dtype=v.dtype), atol=1e-10)
    assert float((b.T @ v).abs().max()) < 1e-10
    assert torch.allclose(v @ v.T, ref[:, :k] @ ref[:, :k].T, atol=1e-8) and torch.allclose(lam, lam_ref[:k], atol=1e-10)
    vd, lamd, ud = G.gate_directions(cf, cr, "protected_contrast_dual", k, rho=rho, protected=b)
    assert torch.allclose(lamd, lam) and float((b.T @ vd).abs().max()) < 1e-10
    assert torch.allclose(vd.norm(dim=0), torch.ones(k, dtype=vd.dtype)) and torch.allclose(vd @ torch.linalg.pinv(vd), v @ v.T, atol=1e-8)
    assert torch.allclose(ud.T @ vd, torch.eye(k, dtype=vd.dtype), atol=1e-8)
    spec = G.column_specificity(vd, cf, cr, rho, lamd)
    assert spec["max_rel_gap_to_lambda"] < 1e-8 and spec["offdiag_share"] < 1e-8
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "input_contrast", k, rho=rho, protected=b)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "protected_contrast", k, rho=rho)


def test_prefix_change_matches_the_dense_edit():
    """The prefix shares ||dW z||^2 / ||W0 z||^2 and the read leakage equal the dense computation; a protected basis
    leaves the prefix outputs unchanged."""
    gen = torch.Generator().manual_seed(18)
    n, k = 10, 4
    cf, cr = _covs(gen, n)
    z = torch.randn(4, n, generator=gen, dtype=torch.float64)
    lin = nn.Linear(n, 6, bias=False).double()
    with torch.no_grad():
        lin.weight.copy_(torch.randn(6, n, generator=gen, dtype=torch.float64))
    w0 = lin.weight.detach().clone()
    v, _, _ = G.gate_directions(cf, cr, "input_contrast", k, rho=1.0)
    gates = G.DirectionGates({"t": lin}, {"t": v})
    with torch.no_grad():
        gates.gates[0].copy_(torch.tensor([0.1, 0.5, 1.0, 0.0], dtype=torch.float64))
    rep = G.prefix_change(gates, {"t": z})["t"]
    vv, a = gates.parts(0)[0].double(), gates.parts(0)[1].double()
    dw = -(a * (1.0 - gates.gates[0].detach())) @ vv.T
    share = (z @ dw.T).square().sum(1) / (z @ w0.T).square().sum(1)
    assert math.isclose(rep["bos_share"], float(share[0]), rel_tol=1e-5)
    assert math.isclose(rep["prefix_share_mean"], float(share.mean()), rel_tol=1e-5)
    assert math.isclose(rep["prefix_share_max"], float(share.max()), rel_tol=1e-5)
    assert math.isclose(rep["read_leak_max"], float(((z @ vv).norm(dim=1) / z.norm(dim=1)).max()), rel_tol=1e-5)
    b, _ = S.protected_span(n, prefix_inputs=z)
    vp, _, up = G.gate_directions(cf, cr, "protected_contrast_dual", k, rho=1.0, protected=b)
    gp = G.DirectionGates({"t": lin}, {"t": vp}, {"t": up})
    with torch.no_grad():
        gp.gates[0].zero_()
    rp = G.prefix_change(gp, {"t": z})["t"]
    assert rp["prefix_share_max"] < 1e-12 and rp["read_leak_max"] < 1e-6


def test_regrowth_report_groups():
    """After the cut the part of gate j is s_j a_j (orthonormal and dual bases): weights at the cut give back = start
    (= s-weighted), weights back at W0 give regrowth = back = 1, per group cut / partial / open."""
    gen = torch.Generator().manual_seed(19)
    n, k = 9, 5
    cf, cr = _covs(gen, n)
    for basis in ("input_contrast", "input_contrast_dual"):
        lin = nn.Linear(n, 7, bias=False).double()
        with torch.no_grad():
            lin.weight.copy_(torch.randn(7, n, generator=gen, dtype=torch.float64))
        w0 = lin.weight.detach().clone()
        v, _, u = G.gate_directions(cf, cr, basis, k, rho=1.0)
        gates = G.DirectionGates({"t": lin}, {"t": v}, {"t": u} if u is not None else None)
        sv = torch.tensor([0.04, 0.3, 0.7, 1.0, 1.0], dtype=torch.float64)
        with torch.no_grad():
            gates.gates[0].copy_(sv)
        saved = {"t": {"u": (u if u is not None else v).float(), "a": gates.parts(0)[1].float(), "s": sv}}
        gates.materialize()
        cut = G.regrowth_report({"t": lin.weight}, saved)["all"]
        back = G.regrowth_report({"t": w0}, saved)["all"]
        assert [cut[g]["count"] for g in ("cut", "partial", "open")] == [2, 1, 2]
        e = gates.parts(0)[1].double().square().sum(0)
        for g, m in (("cut", sv < 0.5), ("partial", (sv >= 0.5) & (sv < 1)), ("open", sv == 1)):
            assert math.isclose(cut[g]["back"], float((sv[m] * e[m]).sum() / e[m].sum()), rel_tol=1e-4)
            assert math.isclose(cut[g]["start"], float((sv[m] ** 2 * e[m]).sum() / e[m].sum()), rel_tol=1e-6)
            assert math.isclose(cut[g]["regrowth"], cut[g]["start"], rel_tol=1e-4)
            assert math.isclose(back[g]["regrowth"], 1.0, rel_tol=1e-4) and math.isclose(back[g]["back"], 1.0, rel_tol=1e-4)


def test_lambda_star_limits_the_candidates():
    """With lambda_star the basis keeps the directions with lambda >= lambda* (at most k), for every contrast basis;
    a non-contrast basis or a non-positive threshold raises; a threshold above every lambda gives no direction."""
    gen = torch.Generator().manual_seed(20)
    n = 12
    cf, cr = _covs(gen, n)
    b, _ = S.protected_span(n, prefix_inputs=torch.randn(2, n, generator=gen, dtype=torch.float64))
    for basis, kw in (("input_contrast", {}), ("input_contrast_dual", {}), ("protected_contrast", {"protected": b}),
                      ("protected_contrast_dual", {"protected": b})):
        _, lam_all, _ = G.gate_directions(cf, cr, basis, n, rho=1.0, **kw)
        star = float(lam_all[3])  # exactly four directions reach it (lambda distinct)
        v, lam, _ = G.gate_directions(cf, cr, basis, n, rho=1.0, lambda_star=star, **kw)
        assert v.shape[1] == 4 and bool((lam >= star).all())
        v2, _, _ = G.gate_directions(cf, cr, basis, 2, rho=1.0, lambda_star=star, **kw)
        assert v2.shape[1] == 2  # capped at k
        v0, lam0, _ = G.gate_directions(cf, cr, basis, n, rho=1.0, lambda_star=float(lam_all[0]) * 2, **kw)
        assert v0.shape == (n, 0) and lam0.numel() == 0
    with pytest.raises(ValueError):
        G.gate_directions(cf, None, "forget_input", 4, lambda_star=1.0)
    with pytest.raises(ValueError):
        G.gate_directions(cf, cr, "input_contrast", 4, rho=1.0, lambda_star=0.0)


def test_train_gates_without_epochs():
    """max_epochs 0: the probe at the start only, no step, the gates unchanged, the stop reason no_training."""
    model = _tiny_llama(seed=10)
    gen = torch.Generator().manual_seed(21)
    forget, retain = _items(gen, 4), _items(gen, 4)
    _, gates = _gates(model)
    with torch.no_grad():
        for g in gates.gates:
            g.fill_(0.2)
    out = _train(model, gates, forget, retain, max_epochs=0, convergence_epochs=1, micro_batch=4, grad_accum=1,
                 probe_retain=4)
    assert out["steps"] == [] and [e["epoch"] for e in out["epochs"]] == [0] and out["stop"] == "no_training"
    assert not out["converged"] and out["epochs_run"] == 0
    assert all(bool((g == 0.2).all()) for g in gates.gates)
    assert math.isclose(out["epochs"][0]["gate_mean"], 0.2)


def test_train_gates_fixed_schedule():
    """convergence_epochs 0: exactly max_epochs epochs (stop fixed_schedule) without tol and drift_max; with a constant
    lr the gates equal those of a convergence-rule run that never stops; the linear schedule's lr at step t of T is
    lr (T - t + 1) / T (an unclipped SGD step is then -lr_t g); the probe only after every probe_every-th and the last
    epoch; the schedule options outside the fixed schedule are refused."""
    model = _tiny_llama(seed=11)
    gen = torch.Generator().manual_seed(23)
    forget, retain = _items(gen, 6), _items(gen, 8)  # 2 windows of 2 x 2 forget items per epoch
    fixed = {"convergence_epochs": 0, "tol": None, "drift_max": None}
    runs = []
    for kw in ({"convergence_epochs": 4, "tol": 0.0, "drift_max": 1.0e-12}, fixed):
        m = copy.deepcopy(model)
        _, gates = _gates(m)
        out = _train(m, gates, forget, retain, max_epochs=4, probe_retain=4, **kw)
        runs.append((out, torch.cat([g.detach() for g in gates.gates])))
    assert torch.equal(runs[0][1], runs[1][1]) and bool((runs[1][1] != 1.0).any())
    out = runs[1][0]
    assert out["stop"] == "fixed_schedule" and not out["converged"] and out["epochs_run"] == 4
    assert len(out["steps"]) == 8 and all(r["lr"] == 0.05 for r in out["steps"])
    assert all("window_converged" not in e and "probe_objective" in e for e in out["epochs"])
    m = copy.deepcopy(model)
    _, gates = _gates(m)
    lr, epochs = 1.0e-3, 3
    out = _train(m, gates, forget, retain, max_epochs=epochs, optim="sgd", lr=lr, alpha=0.0, beta=0.0, gamma=10.0,
                 probe_retain=4, lr_schedule="linear", probe_every=2, **fixed)
    total = 2 * epochs
    assert out["stop"] == "fixed_schedule" and len(out["steps"]) == total
    for rec in out["steps"]:
        assert math.isclose(rec["lr"], lr * (total - rec["step"] + 1) / total, rel_tol=1e-12)
        assert math.isclose(rec["update_l2"], rec["lr"] * rec["grad_l2"], rel_tol=1e-9)
    s = torch.cat([g.detach() for g in gates.gates])
    assert bool((s < 1.0).all()) and bool((s > 0.0).all())  # every gate pushed shut, none clipped
    assert ["probe_objective" in e for e in out["epochs"]] == [True, False, True, True]
    for bad in ({"lr_schedule": "linear"}, {"probe_every": 2}):  # the convergence rule needs both at their default
        with pytest.raises(ValueError):
            _train(m, gates, forget, retain, max_epochs=2, convergence_epochs=2, **bad)
    with pytest.raises(ValueError):  # the fixed schedule takes no tol and drift_max (the helper passes both)
        _train(m, gates, forget, retain, max_epochs=2, convergence_epochs=0)
    with pytest.raises(ValueError):
        _train(m, gates, forget, retain, max_epochs=2, lr_schedule="cosine", **fixed)


def test_reflection_and_rotation_of_the_cut():
    """s = -1 reflects the part (W0 - 2 a v^T, energy 4 ||a||^2-weighted); rotate_cut keeps every inner product of the
    parts, puts them orthogonal to their own span and to the forget output subspace, and written into the weights
    (W0 + (A' - A) V^T) gives W u_j = a'_j for the rotated parts and a_i for the others (orthonormal and dual bases);
    edit_energy equals the dense trace; the regrowth report puts the rotated parts in their own group."""
    gen = torch.Generator().manual_seed(22)
    n, out, k = 10, 16, 5
    cf, cr = _covs(gen, n)
    for basis in ("input_contrast", "input_contrast_dual"):
        lin = nn.Linear(n, out, bias=False).double()
        with torch.no_grad():
            lin.weight.copy_(torch.randn(out, n, generator=gen, dtype=torch.float64))
        w0 = lin.weight.detach().clone()
        v, _, u = G.gate_directions(cf, cr, basis, k, rho=1.0)
        uu = u if u is not None else v
        gates = G.DirectionGates({"t": lin}, {"t": v}, {"t": u} if u is not None else None)
        a = gates.parts(0)[1].double()
        vv = gates.parts(0)[0].double()
        # reflection
        with torch.no_grad():
            gates.gates[0].copy_(torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        rep = G.removed_energy(gates, {"t": cf}, {"t": cr})["t"]
        dense = -2.0 * torch.outer(a[:, 0], vv[:, 0])
        assert math.isclose(rep["forget_energy_removed"], float(torch.trace(dense @ cf @ dense.T)), rel_tol=1e-5)
        gates.materialize()
        assert torch.allclose(lin.weight.detach(), w0 + dense, atol=1e-5)
        with torch.no_grad():
            lin.weight.copy_(w0)
        # rotation of parts 0 and 2
        idx = torch.tensor([0, 2])
        rot = G.rotate_cut(a[:, idx], w0, cf, 0.5, torch.Generator().manual_seed(1))
        ar = rot["rotated"]
        assert torch.allclose(ar.T @ ar, a[:, idx].T @ a[:, idx], atol=1e-8)
        q, _ = torch.linalg.qr(a[:, idx])
        assert float((q.T @ ar).abs().max()) < 1e-8 and rot["own_overlap"] < 1e-8 and rot["forget_overlap"] < 1e-8
        af = w0 @ cf @ w0.T
        vals, vecs = torch.linalg.eigh(af)
        f = vecs[:, vals.argsort(descending=True)[:rot["forget_dims"]]]
        assert float((f.T @ ar).abs().max()) < 1e-8
        da = ar - a[:, idx]
        e = G.edit_energy(da, vv[:, idx], w0, cf)
        dw = da @ vv[:, idx].T
        assert math.isclose(e["energy"], float(torch.trace(dw @ cf @ dw.T)), rel_tol=1e-6)
        w_new = w0 + dw
        got = w_new @ uu.double()
        assert torch.allclose(got[:, idx], ar, atol=1e-6)
        rest = torch.tensor([1, 3, 4])
        assert torch.allclose(got[:, rest], a[:, rest], atol=1e-6)
        saved = {"t": {"u": uu.float(), "a": a.float(), "s": torch.ones(k, dtype=torch.float64),
                       "rotated": ar.float(), "rotated_index": idx}}
        g = G.regrowth_report({"t": w_new}, saved)["all"]
        assert g["rotated"]["count"] == 2 and g["open"]["count"] == 3
        assert abs(g["rotated"]["start"]) < 1e-5 and math.isclose(g["rotated"]["stay"], 1.0, rel_tol=1e-4)
        assert math.isclose(g["rotated"]["regrowth"], 1.0, rel_tol=1e-4) and abs(g["rotated"]["back"]) < 1e-4
        assert math.isclose(g["open"]["back"], 1.0, rel_tol=1e-4)
    with pytest.raises(ValueError):  # no room: the parts and the forget subspace fill the output space
        G.rotate_cut(torch.randn(4, 3, generator=gen, dtype=torch.float64), torch.randn(4, n, generator=gen,
                                                                                         dtype=torch.float64),
                     cf, 0.99, torch.Generator().manual_seed(2))


def test_partial_rotation_keeps_its_cosine():
    """rotate_cut with keep c: A' = c A + sqrt(1 - c^2) A'_0, A'_0 the quarter turn of the same draw (c 0); the inner
    products kept, sum ||a' - a||^2 = 2 (1 - c) sum ||a||^2, the moved component orthogonal to the parts' span and to the
    forget subspace; in the regrowth report the back at the cut is c and the rotated parts stay; c outside (-1, 1)
    refused."""
    gen = torch.Generator().manual_seed(24)
    n, out = 10, 16
    cf, _ = _covs(gen, n)
    w0 = torch.randn(out, n, generator=gen, dtype=torch.float64)
    v = torch.linalg.qr(torch.randn(n, 3, generator=gen, dtype=torch.float64))[0]
    a = w0 @ v
    quarter = G.rotate_cut(a, w0, cf, 0.5, torch.Generator().manual_seed(3))
    assert quarter["keep"] == 0.0
    for c in (0.2, -0.5):
        rot = G.rotate_cut(a, w0, cf, 0.5, torch.Generator().manual_seed(3), keep=c)
        ar = rot["rotated"]
        assert torch.allclose(ar, c * a + math.sqrt(1.0 - c * c) * quarter["rotated"], atol=1e-12)
        assert torch.allclose(ar.T @ ar, a.T @ a, atol=1e-8)
        assert math.isclose(float((ar - a).square().sum()), 2.0 * (1.0 - c) * float(a.square().sum()), rel_tol=1e-9)
        assert rot["own_overlap"] < 1e-8 and rot["forget_overlap"] < 1e-8
        assert rot["forget_dims"] == quarter["forget_dims"] and rot["keep"] == c
        w_new = w0 + (ar - a) @ v.T
        saved = {"t": {"u": v.float(), "a": a.float(), "s": torch.ones(3, dtype=torch.float64), "rotated": ar.float(),
                       "rotated_index": torch.arange(3)}}
        g = G.regrowth_report({"t": w_new}, saved)["all"]["rotated"]
        assert g["count"] == 3 and math.isclose(g["start"], c, abs_tol=1e-5) and math.isclose(g["back"], c, abs_tol=1e-5)
        assert math.isclose(g["stay"], 1.0, rel_tol=1e-5) and math.isclose(g["regrowth"], 1.0, rel_tol=1e-5)
    for c in (-1.0, 1.0):
        with pytest.raises(ValueError):
            G.rotate_cut(a, w0, cf, 0.5, torch.Generator().manual_seed(3), keep=c)


def test_regrowth_report_counts_reflected_parts():
    """A reflected part (s = -1, the shrink cut to -1) belongs to the cut group: start s^2 = 1 and back -1 right after
    the cut."""
    gen = torch.Generator().manual_seed(25)
    w0 = torch.randn(6, 5, generator=gen, dtype=torch.float64)
    v = torch.linalg.qr(torch.randn(5, 2, generator=gen, dtype=torch.float64))[0]
    a = w0 @ v
    w = w0 - 2.0 * torch.outer(a[:, 0], v[:, 0])
    saved = {"t": {"u": v, "a": a, "s": torch.tensor([-1.0, 1.0], dtype=torch.float64)}}
    g = G.regrowth_report({"t": w}, saved)["all"]
    assert g["cut"]["count"] == 1 and g["open"]["count"] == 1
    assert math.isclose(g["cut"]["start"], 1.0, rel_tol=1e-12) and math.isclose(g["cut"]["back"], -1.0, rel_tol=1e-12)


def test_summarize_gates_totals():
    rows = {"model.layers.0.mlp.down_proj": {"k": 4, "mean": 0.5, "below_0p5": 2, "below_0p1": 1, "at_0": 1, "at_1": 1,
                                              "contraction_sum": 2.0, "displacement2": 4.0, "forget_energy_removed": 1.0,
                                              "forget_energy": 4.0, "retain_energy_removed": 0.5, "retain_energy": 5.0,
                                              "weight_rel_change": 0.2, "displacement": 2.0},
            "model.layers.0.self_attn.o_proj": {"k": 2, "mean": 1.0, "below_0p5": 0, "below_0p1": 0, "at_0": 0,
                                                 "at_1": 2, "contraction_sum": 0.0, "displacement2": 0.0,
                                                 "forget_energy_removed": 0.0, "forget_energy": 4.0,
                                                 "retain_energy_removed": 0.0, "retain_energy": 5.0,
                                                 "weight_rel_change": 0.0, "displacement": 0.0}}
    agg = G.summarize_gates(rows)
    assert set(agg) == {"down_proj", "o_proj", "all"}
    a = agg["all"]
    assert a["k"] == 6 and math.isclose(a["gate_mean"], (0.5 * 4 + 2) / 6) and a["forget_share"] == 1.0 / 8.0
    assert math.isclose(a["retain_share"], 0.05) and math.isclose(a["displacement"], 2.0)
