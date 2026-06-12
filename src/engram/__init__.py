"""engram — minimal, efficient covariance-based engram extraction.

Milestone 1: collect input covariances and extract per-layer *engram weights*
``W_engram = W . Sigma_target . pinv(Sigma_total)`` for PyTorch / HuggingFace
models. Applying the edit is a later milestone.
"""

from __future__ import annotations

from .collectors import CovarianceCollector
from .config import EditorConfig
from .editor import EngramEditor
from .handlers import (
    Conv1DHandler,
    LayerHandler,
    LinearHandler,
    MaskedLinearHandler,
    handler_for,
)

__all__ = [
    "EditorConfig",
    "EngramEditor",
    "CovarianceCollector",
    "LayerHandler",
    "LinearHandler",
    "Conv1DHandler",
    "MaskedLinearHandler",
    "handler_for",
]

__version__ = "0.4.0"
