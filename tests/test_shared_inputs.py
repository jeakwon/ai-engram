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


# T5: one accumulator per group — members hold the same object, not a copy.
def test_group_shares_one_buffer():
    torch.manual_seed(0)
    st = _collect(_Block(), _batches(), CovarianceCollector._WINDOW)
    assert st["q"] is st["k"] and st["q"] is st["v"]
    assert st["q"] is not st["o"]
    assert len({id(v) for v in st.cov.values()}) == 2      # {q,k,v} and {o}


# T6: the file stores a shared covariance once, and load restores the sharing.
def test_save_dedupes_and_load_restores_sharing(tmp_path):
    torch.manual_seed(0)
    st = _collect(_Block(), _batches(), CovarianceCollector._WINDOW)
    shared, unshared = tmp_path / "shared.pt", tmp_path / "unshared.pt"
    st.save(shared)
    from engram import Statistics

    Statistics({k: v.clone() for k, v in st.cov.items()}, dict(st.count)).save(unshared)
    assert shared.stat().st_size < unshared.stat().st_size
    back = Statistics.load(shared)
    assert back["q"] is back["k"] and back["q"] is back["v"]
    for name in st:
        assert torch.equal(back[name], st[name])
    assert back.count == st.count


# T7: the engram decomposes each distinct covariance once, and the result is unchanged.
def test_engram_reuses_factors_per_distinct_covariance():
    import engram.editor as editor_mod

    torch.manual_seed(0)
    m = _Block()
    st = _collect(m, _batches(), CovarianceCollector._WINDOW)
    tgt = _collect(m, _batches(seed=1), CovarianceCollector._WINDOW)
    ed = EngramEditor(m, EditorConfig(storage_device="cpu"))
    calls, orig = [0], editor_mod.spectral_factors

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    editor_mod.spectral_factors = counting
    try:
        shared = ed.compute_engram_weights(tgt, st)
    finally:
        editor_mod.spectral_factors = orig
    assert calls[0] == 2                                   # not 4 — one per distinct covariance
    plain = ed.compute_engram_weights(
        tgt, __import__("engram").Statistics({k: v.clone() for k, v in st.cov.items()}, dict(st.count))
    )
    for name in plain.layers:
        assert torch.equal(shared.layers[name].projection, plain.layers[name].projection)


# T8: moving preserves sharing (moving members separately would undo the memory saving).
def test_to_preserves_sharing():
    torch.manual_seed(0)
    st = _collect(_Block(), _batches(), CovarianceCollector._WINDOW)
    moved = st.to("cpu")
    assert moved["q"] is moved["k"] and moved["q"] is moved["v"]
    assert moved["q"] is not moved["o"]
    for name in st:
        assert torch.equal(moved[name], st[name])


# T9: a file carrying an alias map is tagged apart, so a reader that predates aliases fails
# loudly instead of silently dropping the aliased layers.
def test_alias_files_are_tagged_apart(tmp_path):
    from engram import Statistics
    from engram.stats import _FORMAT, _FORMAT_ALIASED

    c = torch.randn(6, 8)
    cov = c.T @ c
    shared = Statistics({"q": cov, "k": cov}, {"q": 6, "k": 6})
    plain = Statistics({"q": cov.clone()}, {"q": 6})
    fs, fp = tmp_path / "s.pt", tmp_path / "p.pt"
    shared.save(fs)
    plain.save(fp)
    assert torch.load(fs, weights_only=True)["format"] == _FORMAT_ALIASED
    assert torch.load(fp, weights_only=True)["format"] == _FORMAT       # 0.9.x can still read it
    back = Statistics.load(fs)
    assert back["q"] is back["k"] and torch.equal(back["q"], cov)


# T10: the sharing window pins only tensor references, never covariance-sized copies.
def test_window_holds_no_covariances():
    torch.manual_seed(0)
    m = _Block()
    ed = EngramEditor(m, EditorConfig(storage_device="cpu"))
    col = CovarianceCollector(m, EditorConfig(storage_device="cpu"), ed.registry)
    with col:
        for b in _batches():
            m(b["x"])
        for entry in col._recent.values():
            assert len(entry) == 2                       # (input tensor, owner name)
            assert isinstance(entry[1], str)
