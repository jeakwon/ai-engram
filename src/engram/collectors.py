"""Covariance collector.

Registers forward pre-hooks that accumulate the input covariance ``sum(x^T x)``
for each supported layer. Forward-only (no backward pass), in-place
accumulation, with matrices held on ``config.storage_device``.

A per-batch token mask (``mask_fn`` in ``EngramEditor.collect_statistics``)
restricts the covariance to selected tokens for **every** layer type. MoE experts
receive only their routed tokens, so the routed rows are matched back to the most
recent full-token reference (the router/gate input) to recover their global mask;
an expert's second projection (whose input has no full-token counterpart) reuses
the routing mask just computed. This is automatic and a no-op for dense models.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .config import EditorConfig
from .handlers import LayerHandler, handler_for

# Container names whose ".<name>.<idx>." segment carries the decoder-layer index,
# scanned when layers_to_transform is set but layers_pattern is not (Llama="layers",
# GPT-2="h", etc.). Mirrors PEFT/LoRA's layer-index selection.
_COMMON_LAYER_PATTERNS = ("layers", "h", "blocks", "block", "layer")


def _module_name_matches(name: str, target_modules: Optional[Union[str, List[str]]]) -> bool:
    """LoRA/PEFT-style module match: None=all, str=regex (fullmatch), list=exact-or-dotted-suffix."""
    if target_modules is None:
        return True
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return name in target_modules or any(name.endswith(f".{t}") for t in target_modules)


def _layer_index_matches(
    name: str,
    layers_to_transform: Optional[Union[int, List[int]]],
    layers_pattern: Optional[Union[str, List[str]]],
) -> bool:
    """PEFT-style index filter: True if the module's decoder-layer index is selected.

    No-op when layers_to_transform is None. A module with no parseable layer index
    (e.g. ``lm_head``) is excluded once an index filter is requested.
    """
    if layers_to_transform is None:
        return True
    want = {layers_to_transform} if isinstance(layers_to_transform, int) else set(layers_to_transform)
    if isinstance(layers_pattern, str):
        patterns = [layers_pattern]
    elif layers_pattern:
        patterns = list(layers_pattern)
    else:
        patterns = list(_COMMON_LAYER_PATTERNS)
    for pat in patterns:
        m = re.search(rf"(?:^|\.){pat}\.(\d+)(?:\.|$)", name)
        if m is not None:
            return int(m.group(1)) in want
    return False


class CovarianceCollector:
    """Context manager accumulating per-layer input covariance ``sum(x^T x)``."""

    def __init__(
        self,
        model: nn.Module,
        config: EditorConfig,
        registry: Dict[Type[nn.Module], LayerHandler],
        target_modules: Optional[Union[str, List[str]]] = None,
        layers_to_transform: Optional[Union[int, List[int]]] = None,
        layers_pattern: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.registry = registry
        self.target_modules = target_modules
        self.layers_to_transform = layers_to_transform
        self.layers_pattern = layers_pattern
        self.covariance_matrices: Dict[str, torch.Tensor] = {}
        self.current_mask: Optional[torch.Tensor] = None
        # per-batch routing-alignment state (used only when a layer sees a subset)
        self._proj: Dict[int, torch.Tensor] = {}                       # H -> random fingerprint vector
        self._ref: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}   # H -> (fingerprint[N], mask[N])
        self._routed_mask: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []

    def set_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Set the per-batch token mask and reset routing-alignment state."""
        self.current_mask = mask
        self._ref.clear()
        self._routed_mask = None

    def _fingerprint(self, x: torch.Tensor) -> torch.Tensor:
        """A per-row scalar (random projection) used to match routed rows back."""
        h = x.shape[1]
        proj = self._proj.get(h)
        if proj is None:
            g = torch.Generator(device=x.device).manual_seed(1234567 + h)  # local RNG, deterministic
            proj = torch.randn(h, generator=g, device=x.device, dtype=x.dtype)
            self._proj[h] = proj
        return x @ proj

    def _select(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the boolean row-mask selecting wanted tokens for this layer input."""
        m = self.current_mask
        if m is None:
            return None
        m = m.reshape(-1).to(x.device)
        n_full, n, h = m.shape[0], x.shape[0], x.shape[1]

        if n == n_full:  # full-token layer: also stash a reference for routed layers
            self._ref[h] = (self._fingerprint(x), m)
            return m

        # n != n_full: a routed subset (e.g. an MoE expert seeing only its tokens)
        ref = self._ref.get(h)
        if ref is not None:  # match rows back to the full-token reference (exact gather)
            fp_ref, mask_full = ref
            d = (self._fingerprint(x)[:, None] - fp_ref[None, :]).abs()
            idx = d.argmin(1)
            if (d.gather(1, idx[:, None]).squeeze(1) > 1e-3 * fp_ref.std().clamp(min=1e-9)).any():
                raise ValueError(
                    f"can't align mask: a layer received {n}/{n_full} rows that don't match the "
                    f"full-token reference (not a token gather)."
                )
            self._routed_mask = mask_full[idx]
            return self._routed_mask
        if self._routed_mask is not None and self._routed_mask.shape[0] == n:
            return self._routed_mask  # 2nd projection of the same expert (no own reference)
        raise ValueError(f"can't align mask: a layer received {n}/{n_full} rows and no reference.")

    def __enter__(self) -> "CovarianceCollector":
        for name, module in self.model.named_modules():
            if not (
                _module_name_matches(name, self.target_modules)
                and _layer_index_matches(name, self.layers_to_transform, self.layers_pattern)
            ):
                continue

            handler = handler_for(self.registry, module)
            if handler is None:
                continue

            absorb = self.config.absorb_bias and getattr(module, "bias", None) is not None
            dim = handler.get_input_dim(module, absorb_bias=absorb)
            self.covariance_matrices[name] = torch.zeros(
                (dim, dim), device=self.config.storage_device, dtype=self.config.precision
            )

            def make_hook(layer_name: str, layer_handler: LayerHandler, absorb_bias: bool):
                def hook(mod: nn.Module, inputs: Any) -> None:
                    x = layer_handler.reshape_input(mod, inputs, absorb_bias=absorb_bias).to(
                        self.config.precision
                    )
                    sel = self._select(x)
                    if sel is not None:
                        x = x[sel]
                    self.covariance_matrices[layer_name].add_(
                        (x.mT @ x).to(self.config.storage_device)
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
