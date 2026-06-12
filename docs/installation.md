# Installation

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- `transformers` (optional) — required for HuggingFace models and for GPT-2
  `Conv1D` support

## From PyPI

```bash
pip install ai-engram          # core: torch, tqdm
pip install "ai-engram[llm]"   # + transformers
```

The distribution is named **`ai-engram`**; the **import package is `engram`**:

```python
import engram
from engram import EngramEditor, EditorConfig
print(engram.__version__)
```

## From source

```bash
git clone https://github.com/jeakwon/ai-engram
cd ai-engram
pip install -e ".[dev]"        # editable install + pytest + transformers
```

The package uses a `src/` layout (`src/engram`) with the
[hatchling](https://hatch.pypa.io) build backend.

## Verify

```bash
python -c "import engram; print(engram.__version__)"
pytest tests/test_extraction.py -q      # fast CPU unit tests
```

The heavy TOFU integration tests are gated behind environment variables and
require a GPU + cached models — see [TOFU validation](tofu.md).
