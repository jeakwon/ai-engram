"""EngramEditor: collect input covariances and extract engram weights.

Milestone 1 scope: covariance collection and the closed-form engram weight
    ``W_engram = W . Sigma_target . pinv(Sigma_total)``
per layer (with optional bias absorption). Applying the edit
(``W -= alpha * W_engram``) is intentionally not included yet.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .collectors import CovarianceCollector
from .config import EditorConfig
from .handlers import Conv1DHandler, LinearHandler, get_conv1d_class, handler_for

logger = logging.getLogger("engram")

Stats = Dict[str, torch.Tensor]


class EngramEditor:
    """Covariance-based engram extractor for PyTorch / HuggingFace models.

    The editor hooks every module whose type is in ``self.registry`` (by default
    ``nn.Linear`` and, when ``transformers`` is importable, HF ``Conv1D`` for the
    GPT-2 family). To restrict covariance to answer tokens, swap in a masked
    handler, e.g. ``editor.registry[nn.Linear] = MaskedLinearHandler()``.

    Example::

        editor = EngramEditor(model, EditorConfig(precision=torch.float32))
        target_cov = editor.collect_statistics(forget_loader, batch_fn=bf)
        total_cov = editor.collect_statistics(all_loader, batch_fn=bf)
        weight_engrams, bias_engrams = editor.compute_engram_weights(target_cov, total_cov)
        # weight_engrams[name] has the same shape as the layer's .weight;
        # bias_engrams[name] (if the layer has a bias and absorb_bias is on)
        # has the same shape as the layer's .bias.
    """

    def __init__(self, model: nn.Module, config: Optional[EditorConfig] = None) -> None:
        self.model = model
        self.config = config or EditorConfig()
        self.registry: Dict[type, Any] = {nn.Linear: LinearHandler()}
        conv1d = get_conv1d_class()
        if conv1d is not None:
            self.registry[conv1d] = Conv1DHandler()

    @property
    def _model_device(self) -> torch.device:
        """Device the model lives on — covariances are computed and the solve runs here."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def _storage_device(self) -> Any:
        """Where covariances are held: ``config.storage_device``, or the model device if None."""
        sd = self.config.storage_device
        return sd if sd is not None else self._model_device

    # ---- forward helpers (support tensor / tuple / HF dict batches) ----
    def _move_to_device(self, data: Any) -> Any:
        if torch.is_tensor(data):
            return data.to(self._model_device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        return data

    def _forward(self, inputs: Any) -> None:
        if isinstance(inputs, dict):
            self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) > 0 and isinstance(inputs[0], dict):
                self.model(**inputs[0])
            else:
                self.model(*inputs)
        else:
            self.model(inputs)

    def collect_statistics(
        self,
        dataloader: Iterable[Any],
        target_modules: Optional[Union[str, List[str]]] = None,
        batch_fn: Optional[Callable[[Any], Any]] = None,
        mask_fn: Optional[Callable[[Any], torch.Tensor]] = None,
        layers_to_transform: Optional[Union[int, List[int]]] = None,
        layers_pattern: Optional[Union[str, List[str]]] = None,
        target_layers: Optional[List[str]] = None,
    ) -> Stats:
        """Accumulate input covariance ``sum(x^T x)`` for each supported layer.

        Args:
            dataloader: iterable of batches.
            target_modules: which modules to collect, **LoRA/PEFT convention**. A
                list matches by name suffix (``["down_proj", "q_proj"]`` hits every
                layer's ``down_proj``/``q_proj``); a string is a regex matched
                against the full module path (``r".*layers\\.5\\..*down_proj"``).
                ``None`` (default) collects every supported layer ("all-linear").
            layers_to_transform: restrict to these decoder-layer indices (an int or
                list of ints), like PEFT. Combined with ``target_modules`` as an AND
                filter; ``None`` (default) applies no index restriction.
            layers_pattern: container name(s) holding the layer index
                (``"layers"``, ``"h"``, …). ``None`` scans common patterns.
            target_layers: **deprecated** alias of ``target_modules`` (exact module
                names still match, as they did before).
            batch_fn: maps a batch to model inputs (a tensor, a tuple of
                positional args, or a dict of keyword args). Defaults to
                ``batch[0]`` (vision-style loaders). For HF models pass e.g.
                ``lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}``.
            mask_fn: optional ``batch -> bool tensor`` selecting which tokens enter
                the covariance (one entry per flattened token, e.g.
                ``lambda b: b["labels"] != -100`` for answer-token-only LLM editing).
                Applied to every layer type (``nn.Linear``, ``Conv1D``, …).

        Returns:
            ``{layer_name: covariance[D, D]}`` on ``config.storage_device``.
            ``D`` is the layer's input dim, or ``input dim + 1`` when bias
            absorption applies (``config.absorb_bias`` and the layer has a bias).
        """
        if target_layers is not None:
            if target_modules is None:
                target_modules = target_layers
            warnings.warn(
                "target_layers is deprecated; use target_modules "
                "(LoRA-style: list = name suffix, str = regex).",
                DeprecationWarning,
                stacklevel=2,
            )
        collector = CovarianceCollector(
            self.model,
            self.config,
            self.registry,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
        )
        self.model.eval()

        with collector, torch.inference_mode():
            for batch in tqdm(
                dataloader, disable=None, desc="Collecting covariance"
            ):
                if mask_fn is not None:
                    collector.set_mask(mask_fn(batch))
                raw_inputs = batch_fn(batch) if batch_fn else batch[0]
                self._forward(self._move_to_device(raw_inputs))

        return collector.covariance_matrices

    @staticmethod
    def merge_statistics(*stats_dicts: Stats) -> Stats:
        """Sum multiple covariance dicts layer-wise (used to build totals)."""
        if not stats_dicts:
            return {}
        merged = {k: v.clone() for k, v in stats_dicts[0].items()}
        for d in stats_dicts[1:]:
            for k, v in d.items():
                if k in merged:
                    merged[k].add_(v)
                else:
                    merged[k] = v.clone()
        return merged

    @torch.no_grad()
    def compute_engram_weights(
        self,
        target_covariances: Union[Stats, List[Stats]],
        total_covariance: Stats,
    ) -> Tuple[Stats, Stats]:
        """Compute ``W_engram = W . Sigma_target . pinv(Sigma_total)`` per layer.

        Whether a layer was bias-absorbed is inferred from the covariance size
        (``D == in + 1``), so collection and computation stay consistent without
        re-passing a flag.

        Args:
            target_covariances: covariance of the data to isolate (the
                "forget"/target set). A single dict, or a list of dicts that are
                summed first via :meth:`merge_statistics`.
            total_covariance: covariance of the total/reference set.

        Returns:
            ``(weight_engrams, bias_engrams)``. ``weight_engrams[name]`` matches
            the layer's ``weight`` shape; ``bias_engrams[name]`` matches the
            ``bias`` shape and is present only for bias-absorbed layers
            (``bias_engrams`` is empty for bias-free models).
        """
        if isinstance(target_covariances, list):
            target_covariances = self.merge_statistics(*target_covariances)

        weight_engrams: Stats = {}
        bias_engrams: Stats = {}
        modules = dict(self.model.named_modules())

        for layer_name, pos_cov in tqdm(
            target_covariances.items(),
            disable=None,
            desc="Computing engram",
        ):
            if layer_name not in total_covariance or layer_name not in modules:
                continue

            module = modules[layer_name]
            handler = handler_for(self.registry, module)
            if handler is None:
                continue

            dev, prec = self._model_device, self.config.precision
            sum_cov = total_covariance[layer_name].to(dev, dtype=prec)
            pos_cov = pos_cov.to(dev, dtype=prec)

            # The covariance is self-describing: D == in (+1 if bias was absorbed
            # at collection time). Follow it.
            in_dim = handler.get_input_dim(module, absorb_bias=False)
            absorbed = sum_cov.shape[0] == in_dim + 1 and getattr(module, "bias", None) is not None

            weight = handler.weight_matrix(module, absorb_bias=absorbed).to(dev, dtype=prec)

            engram = weight @ pos_cov @ torch.linalg.pinv(sum_cov)  # [out, D]

            if absorbed:
                weight_engrams[layer_name] = handler.to_weight_shape(engram[:, :in_dim], module)
                bias_engrams[layer_name] = engram[:, in_dim:].reshape(-1)
            else:
                weight_engrams[layer_name] = handler.to_weight_shape(engram, module)

        return weight_engrams, bias_engrams

    def save_statistics(self, stats: Stats, path: Union[str, Path]) -> None:
        """Save a statistics dict with ``torch.save``."""
        torch.save(stats, path)
        logger.info("Saved statistics to %s", path)

    def load_statistics(self, path: Union[str, Path]) -> Stats:
        """Load a statistics dict onto the storage device (the model's device by default)."""
        return torch.load(path, map_location=self._storage_device, weights_only=True)
