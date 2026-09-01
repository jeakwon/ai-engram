"""Sibling layers fed by one tensor share their input covariance.

q/k/v read the same post-attention LayerNorm output, and gate/up the same post-MLP one, so
their input covariances are identical by construction. The collector recognizes the repeat and
skips the redundant ``x^T x`` — the dominant cost of collection — without changing a single
number.

Run offline; CPU-only; deterministic.
"""
import torch
import torch.nn as nn

from engram import EditorConfig, EngramEditor
from engram.collectors import CovarianceCollector


class _Block(nn.Module):
    """One tensor feeding three siblings, then a private input for a fourth."""

    def __init__(self, d=16):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        h = torch.nn.functional.layer_norm(x, x.shape[-1:])
        y = self.q(h) + self.k(h) + self.v(h)      # one tensor -> three modules
        return self.o(y)                            # a different tensor -> one module


def _collect(model, batches, window):
    prev = CovarianceCollector._WINDOW
    CovarianceCollector._WINDOW = window
    try:
        ed = EngramEditor(model, EditorConfig(storage_device="cpu"))
        return ed.collect_statistics(batches, batch_fn=lambda b: b["x"])
    finally:
        CovarianceCollector._WINDOW = prev


def _batches(n=4, rows=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"x": torch.randn(rows, d, generator=g)} for _ in range(n)]


# T1: siblings off one tensor get identical covariances, and a private input does not.
def test_siblings_share_covariance():
    torch.manual_seed(0)
    m = _Block()
    st = _collect(m, _batches(), CovarianceCollector._WINDOW)
    assert torch.equal(st["q"], st["k"]) and torch.equal(st["q"], st["v"])
    assert not torch.equal(st["q"], st["o"])
    assert st.count["q"] == st.count["k"] == st.count["v"] == st.count["o"]


# T2: sharing is a pure optimization — results are bit-identical with it disabled.
def test_sharing_is_bit_identical():
    torch.manual_seed(0)
    m = _Block()
    on = _collect(m, _batches(), CovarianceCollector._WINDOW)
    torch.manual_seed(0)
    m2 = _Block()
    off = _collect(m2, _batches(), 0)
    assert set(on.keys()) == set(off.keys())
    for name in on:
        assert torch.equal(on[name], off[name])
        assert on.count[name] == off.count[name]


# T3: the redundant products are actually skipped (2 of every 4 hooks here).
def test_shared_hits_counted():
    torch.manual_seed(0)
    m = _Block()
    ed = EngramEditor(m, EditorConfig(storage_device="cpu"))
    col = CovarianceCollector(m, EditorConfig(storage_device="cpu"), ed.registry)
    with col:
        for b in _batches():
            m(b["x"])
    assert col.shared_hits == 2 * 4          # k and v reuse q's product, every batch
    assert col.sample_counts["q"] == 4 * 8


# T4: masking still applies per group (the mask is a property of the shared input).
def test_masked_sharing():
    torch.manual_seed(0)
    m = _Block()
    g = torch.Generator().manual_seed(1)
    batches = [{"x": torch.randn(8, 16, generator=g), "m": torch.tensor([True] * 5 + [False] * 3)}
               for _ in range(3)]
    ed = EngramEditor(m, EditorConfig(storage_device="cpu"))
    st = ed.collect_statistics(batches, batch_fn=lambda b: b["x"], mask_fn=lambda b: b["m"])
    assert st.count["q"] == 3 * 5            # masked rows excluded
    assert torch.equal(st["q"], st["v"])     # and the group still shares
