"""Covariance collector.

Registers forward pre-hooks that accumulate the input covariance ``sum(x^T x)``
for each supported layer. Forward-only (no backward pass), in-place
accumulation, with matrices held on ``config.storage_device``.

When ``config.absorb_bias`` is set, layers that have a bias accumulate the
augmented ``(in+1) x (in+1)`` covariance over ``[x ; 1]``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn

from .config import EditorConfig
from .handlers import LayerHandler, handler_for


class CovarianceCollector:
    """Context manager accumulating per-layer input covariance ``sum(x^T x)``.

    On ``__enter__`` it allocates a ``D x D`` matrix per matched layer and
    registers a ``forward_pre_hook`` that flattens the layer input to ``[N, D]``
    and adds ``x^T x`` in place. On ``__exit__`` all hooks are removed.
    """

    def __init__(
        self,
        model: nn.Module,
        config: EditorConfig,
        registry: Dict[Type[nn.Module], LayerHandler],
        target_layers: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.registry = registry
        self.target_layers = target_layers
        self.covariance_matrices: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[Any] = []

    def __enter__(self) -> "CovarianceCollector":
        for name, module in self.model.named_modules():
            if self.target_layers and name not in self.target_layers:
                continue

            handler = handler_for(self.registry, module)
            if handler is None:
                continue

            absorb = self.config.absorb_bias and getattr(module, "bias", None) is not None
            dim = handler.get_input_dim(module, absorb_bias=absorb)
            self.covariance_matrices[name] = torch.zeros(
                (dim, dim),
                device=self.config.storage_device,
                dtype=self.config.precision,
            )

            def make_hook(layer_name: str, layer_handler: LayerHandler, absorb_bias: bool):
                def hook(mod: nn.Module, inputs: Any) -> None:
                    x = layer_handler.reshape_input(mod, inputs, absorb_bias=absorb_bias).to(
                        self.config.precision
                    )
                    cov_chunk = x.mT @ x
                    self.covariance_matrices[layer_name].add_(
                        cov_chunk.to(self.config.storage_device)
                    )

                return hook

            self._hook_handles.append(
                module.register_forward_pre_hook(make_hook(name, handler, absorb))
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
