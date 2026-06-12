"""Configuration for the engram editor."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EditorConfig:
    """Configuration for engram extraction.

    Attributes:
        device: Device for the closed-form solve (matmul + pseudo-inverse).
        storage_device: Device for accumulating/holding covariance matrices.
            Defaults to CPU so wide layers do not pin large matrices in VRAM.
        precision: Numerical precision for covariance accumulation and the
            closed-form solve. ``float64`` is recommended for stability;
            use ``float32`` for large LLMs to halve covariance memory.
        damping_factor: Tikhonov regularization for the pseudo-inverse
            (``Sigma_total + lambda*I``). ``0.0`` disables it.
        absorb_bias: When ``True`` (default, *automatic*), layers that have a
            bias are treated as the affine map ``y = Wx + b`` via homogeneous
            coordinates: the input is augmented with a constant ``1`` and the
            weight with the bias column, so the engram also corrects ``b``.
            Bias-free layers (e.g. Llama/Mistral projections) are unaffected.
            Set ``False`` to edit ``W`` only (the original behavior).
        verbose: Whether to show progress bars.
    """

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    storage_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    precision: torch.dtype = torch.float64
    damping_factor: float = 0.0
    absorb_bias: bool = True
    verbose: bool = True
