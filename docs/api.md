# API reference

The public surface of the `engram` package, generated from the source docstrings.

```python
from engram import (
    EditorConfig, EngramEditor,
    CovarianceCollector,
    LayerHandler, LinearHandler, Conv1DHandler, MaskedLinearHandler,
)
```

::: engram.EditorConfig

::: engram.EngramEditor

::: engram.LinearHandler

::: engram.Conv1DHandler

::: engram.MaskedLinearHandler

::: engram.LayerHandler

::: engram.CovarianceCollector

## MoE (optional)

Support for transformers&nbsp;≥5 **fused-expert** MoE layers lives in a separate,
detachable module — import it explicitly; the core never depends on it:

```python
from engram import EngramEditor
from engram.moe import FusedExpertAdapter, apply_engram_weights

editor = EngramEditor(model, adapters=[FusedExpertAdapter()])
```

See [Guide → Mixture-of-experts](guide.md#mixture-of-experts).

::: engram.moe.FusedExpertAdapter

::: engram.moe.apply_engram_weights
