"""Milestone-1 extraction tests (CPU-only, deterministic, offline).

Run via SLURM (no login-node execution):
    pytest -q tests/
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engram import EditorConfig, EngramEditor, MaskedLinearHandler


def cpu_cfg(**kw) -> EditorConfig:
    base = dict(
        device=torch.device("cpu"),
        storage_device=torch.device("cpu"),
        precision=torch.float64,
        verbose=False,
    )
    base.update(kw)
    return EditorConfig(**base)


# --------------------------------------------------------------------------- #
# T0: package imports and public API surface
# --------------------------------------------------------------------------- #
def test_public_api():
    import engram

    assert isinstance(engram.__version__, str)
    for name in [
        "EditorConfig",
        "EngramEditor",
        "CovarianceCollector",
        "LayerHandler",
        "LinearHandler",
        "Conv1DHandler",
        "MaskedLinearHandler",
    ]:
        assert hasattr(engram, name), name


# --------------------------------------------------------------------------- #
# T1: correctness anchor (no bias) — if Sigma_target == Sigma_total and Sigma is
# full rank, Sigma . pinv(Sigma) == I, so W_engram must equal W exactly.
# --------------------------------------------------------------------------- #
def test_engram_equals_weight_when_target_is_total():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    editor = EngramEditor(model, cpu_cfg())  # absorb_bias auto, but no bias -> off

    X = torch.randn(512, 8)  # >> 8 gaussian samples => full-rank covariance
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    assert cov["0"].shape == (8, 8)  # not augmented (bias-free)

    weights, biases = editor.compute_engram_weights(cov, cov)
    W = model[0].weight.detach().to(torch.float64)
    assert set(weights.keys()) == {"0"}
    assert biases == {}  # nothing to absorb
    assert torch.allclose(weights["0"], W, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- #
# T2: subspace — target inputs spanning a k-dim subspace make Sigma.pinv(Sigma)
# a rank-k projector, so the engram weight has rank <= k (< full weight rank).
# --------------------------------------------------------------------------- #
def test_engram_rank_bounded_by_target_subspace():
    torch.manual_seed(0)
    k = 3
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    editor = EngramEditor(model, cpu_cfg())

    Z = torch.randn(256, k)
    A = torch.randn(k, 8)
    X = Z @ A  # rows live in a k-dimensional subspace of R^8
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    weights, biases = editor.compute_engram_weights(cov, cov)

    W = model[0].weight.detach().to(torch.float64)
    assert weights["0"].shape == (4, 8)
    assert torch.linalg.matrix_rank(W) == 4  # weight itself is full rank
    assert torch.linalg.matrix_rank(weights["0"]) <= k  # projection reduced it


# --------------------------------------------------------------------------- #
# T3: GPT-2 Conv1D path — built from config (no download). Verifies the
# transpose round-trip (engram weight keeps the layer's [in, out] shape) and
# that bias-bearing Conv1D layers produce bias engrams of the right shape.
# --------------------------------------------------------------------------- #
def test_gpt2_conv1d_shapes_and_bias():
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.pytorch_utils import Conv1D

    from engram.handlers import get_conv1d_class

    assert get_conv1d_class() is not None

    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=64, vocab_size=128)
    model = GPT2LMHeadModel(cfg).eval()
    editor = EngramEditor(model, cpu_cfg())  # absorb_bias on by default

    ids = torch.randint(0, 128, (4, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    cov = editor.collect_statistics([batch], batch_fn=lambda b: b)
    weights, biases = editor.compute_engram_weights(cov, cov)

    modules = dict(model.named_modules())
    for name, w in weights.items():
        assert w.shape == modules[name].weight.shape, name
    for name, b in biases.items():
        assert b.shape == modules[name].bias.shape, name

    conv1d_names = [n for n, m in modules.items() if isinstance(m, Conv1D)]
    assert conv1d_names, "no Conv1D modules found in GPT-2"
    # Conv1D always has a bias -> absorbed -> present in both dicts
    assert all(n in weights and n in biases for n in conv1d_names)

    cattn = next(n for n in conv1d_names if n.endswith("attn.c_attn"))
    assert weights[cattn].shape == (32, 96)  # [n_embd, 3*n_embd], orientation kept
    assert biases[cattn].shape == (96,)
    # lm_head is bias-free -> weight only, no bias engram
    assert "lm_head" in weights and "lm_head" not in biases


# --------------------------------------------------------------------------- #
# T4: masking + absorption — MaskedLinearHandler restricts covariance to the
# selected tokens, and the constant-1 column is appended after masking.
# --------------------------------------------------------------------------- #
def test_masked_handler_selects_only_masked_tokens():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3)).eval()  # has bias -> absorbed
    editor = EngramEditor(model, cpu_cfg())
    masked = MaskedLinearHandler()
    editor.registry[nn.Linear] = masked

    X = torch.randn(2, 5, 4)  # [batch, seq, dim]
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask.view(-1)[[0, 3, 7]] = True  # exactly 3 selected tokens

    def batch_fn(batch):
        x, m = batch
        masked.current_mask = m
        return x

    cov = editor.collect_statistics([(X, mask)], batch_fn=batch_fn)

    x_sel = X.reshape(-1, 4)[mask.reshape(-1)].to(torch.float64)  # [3, 4]
    aug = torch.cat([x_sel, torch.ones(x_sel.shape[0], 1, dtype=torch.float64)], dim=1)  # [3, 5]
    expected = aug.mT @ aug
    assert cov["0"].shape == (5, 5)  # augmented
    assert torch.allclose(cov["0"], expected, atol=1e-8)
    assert torch.isclose(cov["0"][4, 4], torch.tensor(3.0, dtype=torch.float64))  # token count


# --------------------------------------------------------------------------- #
# T5: bias absorption — with a bias-bearing layer and Sigma_target == Sigma_total
# (full rank), the engram recovers BOTH W and b exactly.
# --------------------------------------------------------------------------- #
def test_bias_absorption_recovers_weight_and_bias():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4))  # bias=True
    editor = EngramEditor(model, cpu_cfg())  # absorb_bias on by default

    X = torch.randn(512, 8)
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    assert cov["0"].shape == (9, 9)  # augmented [x ; 1]

    weights, biases = editor.compute_engram_weights(cov, cov)
    W = model[0].weight.detach().to(torch.float64)
    b = model[0].bias.detach().to(torch.float64)
    assert weights["0"].shape == W.shape
    assert "0" in biases and biases["0"].shape == b.shape
    assert torch.allclose(weights["0"], W, atol=1e-6, rtol=1e-5)
    assert torch.allclose(biases["0"], b, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- #
# T6: absorb_bias=False reproduces the original W-only behavior (no bias engram).
# --------------------------------------------------------------------------- #
def test_absorb_bias_off_is_weight_only():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4))  # has bias, but absorption disabled
    editor = EngramEditor(model, cpu_cfg(absorb_bias=False))

    X = torch.randn(512, 8)
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    assert cov["0"].shape == (8, 8)  # not augmented

    weights, biases = editor.compute_engram_weights(cov, cov)
    W = model[0].weight.detach().to(torch.float64)
    assert biases == {}
    assert torch.allclose(weights["0"], W, atol=1e-6, rtol=1e-5)
