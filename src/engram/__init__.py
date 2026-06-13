"""engram — minimal, efficient covariance-based engram extraction.

Collect input covariances and extract per-layer *engram weights*
``W_engram = W . Sigma_target . pinv(Sigma_total)`` for PyTorch / HuggingFace
models — forward-only, closed-form, with automatic bias absorption and answer-token
masking. Optional fused-MoE support lives in ``engram.moe``.
See https://jeakwon.github.io/ai-engram/.
"""

from __future__ import annotations

from .collectors import CovarianceCollector
from .config import EditorConfig
from .editor import EngramEditor
from .handlers import (
    Conv1DHandler,
    LayerHandler,
    LinearHandler,
    handler_for,
)

__all__ = [
    "EditorConfig",
    "EngramEditor",
    "CovarianceCollector",
    "LayerHandler",
    "LinearHandler",
    "Conv1DHandler",
    "handler_for",
]

__version__ = "0.5.0"
