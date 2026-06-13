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
        storage_device=torch.device("cpu"),
        precision=torch.float64,
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


# --------------------------------------------------------------------------- #
# T7: collector-level mask_fn restricts covariance to selected tokens, for any
# layer type — including GPT-2 Conv1D (which MaskedLinearHandler cannot handle).
# --------------------------------------------------------------------------- #
def test_mask_fn_generic_linear_and_conv1d():
    # (a) nn.Linear — mask_fn keeps exactly the masked token rows
    torch.manual_seed(0)
    lin = nn.Sequential(nn.Linear(4, 3, bias=False)).eval()
    editor = EngramEditor(lin, cpu_cfg())
    X = torch.randn(2, 5, 4)  # [batch, seq, dim]
    labels = torch.full((2, 5), -100)
    labels.view(-1)[[0, 3, 7]] = 1  # 3 answer tokens
    cov = editor.collect_statistics(
        [(X, labels)], batch_fn=lambda b: b[0], mask_fn=lambda b: b[1] != -100
    )
    sel = X.reshape(-1, 4)[labels.reshape(-1) != -100].double()
    assert cov["0"].shape == (4, 4)
    assert torch.allclose(cov["0"], sel.mT @ sel, atol=1e-8)

    # (b) GPT-2 Conv1D — the same mask_fn works (MaskedLinearHandler would crash)
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    gpt = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=16, vocab_size=40)
    ).eval()
    ed = EngramEditor(gpt, cpu_cfg(absorb_bias=False))
    ids = torch.randint(0, 40, (2, 6))
    lab = torch.full((2, 6), -100)
    lab.view(-1)[[1, 4, 9, 10]] = 1  # 4 answer tokens
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}

    # capture the Conv1D layer's input to build the expected masked covariance
    cattn = "transformer.h.0.attn.c_attn"
    cap = {}
    handle = dict(gpt.named_modules())[cattn].register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("x", inp[0].detach().reshape(-1, 16).double())
    )
    with torch.inference_mode():
        gpt(**feats(batch))
    handle.remove()
    sel2 = cap["x"][lab.reshape(-1) != -100]

    cov2 = ed.collect_statistics([batch], batch_fn=feats, mask_fn=lambda b: b["labels"] != -100)
    assert cov2[cattn].shape == (16, 16)
    assert torch.allclose(cov2[cattn], sel2.mT @ sel2, atol=1e-6)


# --------------------------------------------------------------------------- #
# T8: MoE masking — routed expert layers recover their answer-token mask by
# matching rows back to the router input (no routing knowledge needed).
# --------------------------------------------------------------------------- #
def test_mask_fn_moe_mixtral():
    pytest.importorskip("transformers")
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    m = MixtralForCausalLM(
        MixtralConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=64, max_position_embeddings=32,
            num_local_experts=4, num_experts_per_tok=2,
        )
    ).eval()
    # transformers >=5 fuses Mixtral experts into Parameters (no per-expert
    # nn.Linear); this test targets the per-expert .w1 layout — skip otherwise.
    if not any(isinstance(mod, nn.Linear) and n.endswith(".w1") for n, mod in m.named_modules()):
        pytest.skip("no per-expert .w1 nn.Linear (this transformers fuses MoE experts)")
    ed = EngramEditor(m, cpu_cfg(absorb_bias=False))
    ids = torch.randint(0, 64, (2, 8))
    lab = torch.full((2, 8), -100)
    lab.view(-1)[[1, 4, 9, 12]] = 1
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}

    # masking used to crash on the routed expert layers — now it runs
    cov = ed.collect_statistics([batch], batch_fn=feats, mask_fn=lambda b: b["labels"] != -100)
    we, _ = ed.compute_engram_weights(cov, cov)
    expert_w1 = [n for n in cov if n.endswith(".w1")]
    assert expert_w1, "no MoE expert layers were hooked"
    mods = dict(m.named_modules())
    for n in we:
        assert we[n].shape == mods[n].weight.shape, n

    # correctness: brute-force align one expert's w1 against the router input
    w1 = expert_w1[0]
    gate = w1.rsplit(".experts.", 1)[0] + ".gate"
    cap = {}
    hg = mods[gate].register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("g", inp[0].detach().reshape(-1, 32).double())
    )
    he = mods[w1].register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("e", inp[0].detach().reshape(-1, 32).double())
    )
    with torch.inference_mode():
        m(**feats(batch))
    hg.remove()
    he.remove()

    match = (cap["g"][None, :, :] == cap["e"][:, None, :]).all(-1)  # exact row match
    assert (match.sum(1) == 1).all()
    idx = match.float().argmax(1)
    sel = cap["e"][(lab.reshape(-1) != -100)[idx]]
    assert torch.allclose(cov[w1], sel.mT @ sel, atol=1e-5)


# A tiny decoder-style stack: module names are layers.{i}.{down,up}_proj, so the
# index-based selection below has a realistic ".layers.<idx>." path to parse.
class _TinyStack(nn.Module):
    def __init__(self, n: int = 3, d: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {"down_proj": nn.Linear(d, d, bias=False), "up_proj": nn.Linear(d, d, bias=False)}
            )
            for _ in range(n)
        )

    def forward(self, x):
        for blk in self.layers:
            x = blk["up_proj"](blk["down_proj"](x))
        return x


def _stack_loader(d: int = 4):
    return DataLoader(TensorDataset(torch.randn(64, d)), batch_size=16)


# --------------------------------------------------------------------------- #
# T9: target_modules LoRA convention — a list matches by dotted name suffix
# (across every layer); a string is a regex over the full module path.
# --------------------------------------------------------------------------- #
def test_target_modules_suffix_and_regex():
    torch.manual_seed(0)
    model = _TinyStack(n=3).eval()

    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=["down_proj"]
    )
    assert set(cov) == {"layers.0.down_proj", "layers.1.down_proj", "layers.2.down_proj"}

    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=r".*layers\.1\..*"
    )
    assert set(cov) == {"layers.1.down_proj", "layers.1.up_proj"}


# --------------------------------------------------------------------------- #
# T10: layers_to_transform / layers_pattern select by decoder-layer index, and
# combine with target_modules as an AND filter (PEFT convention).
# --------------------------------------------------------------------------- #
def test_layers_to_transform_selects_by_index():
    torch.manual_seed(0)
    model = _TinyStack(n=3).eval()

    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=["down_proj"], layers_to_transform=[0, 2], layers_pattern="layers"
    )
    assert set(cov) == {"layers.0.down_proj", "layers.2.down_proj"}

    # int form + target_modules=None -> every linear in just that one layer
    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), layers_to_transform=1, layers_pattern="layers"
    )
    assert set(cov) == {"layers.1.down_proj", "layers.1.up_proj"}

    # layers_pattern=None auto-detects common containers ("layers", "h", ...)
    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=["down_proj"], layers_to_transform=[0]
    )
    assert set(cov) == {"layers.0.down_proj"}


# --------------------------------------------------------------------------- #
# T11: target_layers still works as a deprecated alias — exact module names match
# exactly as before, and a DeprecationWarning is emitted.
# --------------------------------------------------------------------------- #
def test_target_layers_deprecated_alias():
    import warnings

    torch.manual_seed(0)
    model = _TinyStack(n=2).eval()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cov = EngramEditor(model, cpu_cfg()).collect_statistics(
            _stack_loader(), target_layers=["layers.1.up_proj"]
        )
    assert set(cov) == {"layers.1.up_proj"}
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# --------------------------------------------------------------------------- #
# T12: storage_device defaults to None, which follows the model's device — so on
# a CPU model the covariances are accumulated on CPU (no config needed).
# --------------------------------------------------------------------------- #
def test_default_storage_follows_model_device():
    assert EditorConfig().storage_device is None
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 3, bias=False)).eval()  # lives on CPU
    cov = EngramEditor(model, EditorConfig(precision=torch.float64)).collect_statistics(
        DataLoader(TensorDataset(torch.randn(32, 6)), batch_size=8)
    )
    assert cov["0"].device.type == "cpu"  # followed the (CPU) model, not pinned elsewhere
