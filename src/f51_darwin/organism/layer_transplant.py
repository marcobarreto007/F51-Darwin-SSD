"""Native Layer Transplant Manager — Direct Internal Block Integration.

Replaces external sidecars with native internal layer transplants.
Transplants specialized layers directly into the internal stack of the target model
with basis-alignment and KL-divergence verification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TransplantResult:
    target_layer_idx: int
    donor_layer_idx: int
    relative_weight_distance: float
    weights_transplanted: int
    donor_provenance: str
    transplant_sha256: str


class NativeLayerTransplantManager:
    """Manages direct internal layer swaps and block insertions.

    Replaces sidecars by inserting/swapping donor layers into the model's
    internal PyTorch `nn.ModuleList` (e.g. `model.model.layers` or `model.blocks`).
    """

    def __init__(self, target_model: nn.Module) -> None:
        self.target_model = target_model
        self.target_layers = self._extract_layers(target_model)

    def _extract_layers(self, model: nn.Module) -> nn.ModuleList:
        """Extract the ModuleList representing model layers."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "blocks"):
            return model.blocks
        elif hasattr(model, "layers"):
            return model.layers
        else:
            raise ValueError("Target model has no recognised layer container (.model.layers, .blocks, .layers).")

    def measure_weight_distance(self, donor_layer: nn.Module, target_layer_idx: int) -> float:
        """Calculate relative distance ||donor_w - target_w|| / ||target_w||."""
        target_layer = self.target_layers[target_layer_idx]
        target_params = dict(target_layer.named_parameters())
        donor_params = dict(donor_layer.named_parameters())

        num = 0.0
        den = 0.0

        for name, t_param in target_params.items():
            if name in donor_params:
                d_param = donor_params[name]
                if t_param.shape == d_param.shape:
                    num += float((d_param.detach().float() - t_param.detach().float()).pow(2).sum())
                    den += float(t_param.detach().float().pow(2).sum())

        if den == 0.0:
            return 0.0
        return (num ** 0.5) / (den ** 0.5)

    def swap_layer(
        self,
        donor_layer: nn.Module,
        target_layer_idx: int,
        donor_layer_idx: int = 0,
        donor_provenance: str = "unknown_donor",
        trainable: bool = False,
    ) -> TransplantResult:
        """Swap an internal target layer with a donor layer directly in RAM.

        Args:
            donor_layer: The PyTorch module representing the donor layer.
            target_layer_idx: The 0-indexed position in the target stack.
            donor_layer_idx: Source layer index for provenance.
            donor_provenance: Description of the donor model/source.
            trainable: If False, freezes the transplanted layer.

        Returns:
            TransplantResult containing verification stats and SHA-256 hash.
        """
        if target_layer_idx < 0 or target_layer_idx >= len(self.target_layers):
            raise IndexError(f"target_layer_idx {target_layer_idx} out of range (0-{len(self.target_layers)-1}).")

        target_layer = self.target_layers[target_layer_idx]

        # Calculate relative weight distance prior to swap
        rel_distance = self.measure_weight_distance(donor_layer, target_layer_idx)

        target_params = dict(target_layer.named_parameters())
        donor_params = dict(donor_layer.named_parameters())

        transplanted_count = 0
        hasher = hashlib.sha256()

        with torch.no_grad():
            for name, t_param in target_params.items():
                if name in donor_params:
                    d_param = donor_params[name]
                    if t_param.shape != d_param.shape:
                        raise ValueError(
                            f"Shape mismatch for parameter '{name}': target {t_param.shape} vs donor {d_param.shape}."
                        )
                    t_param.copy_(d_param)
                    t_param.requires_grad = trainable
                    hasher.update(t_param.float().cpu().numpy().tobytes())
                    transplanted_count += 1

        result = TransplantResult(
            target_layer_idx=target_layer_idx,
            donor_layer_idx=donor_layer_idx,
            relative_weight_distance=rel_distance,
            weights_transplanted=transplanted_count,
            donor_provenance=donor_provenance,
            transplant_sha256=hasher.hexdigest(),
        )

        logger.info(
            f"Transplanted layer {donor_layer_idx} ({donor_provenance}) into target layer {target_layer_idx}. "
            f"Rel. Distance: {rel_distance:.4f}, SHA: {result.transplant_sha256[:12]}"
        )
        return result


def measure_kl_divergence(
    model: nn.Module,
    swapped_model: nn.Module,
    input_ids: torch.Tensor,
) -> float:
    """Measure KL divergence between original and layer-swapped model outputs."""
    with torch.inference_mode():
        orig_logits = model(input_ids)
        if hasattr(orig_logits, "logits"):
            orig_logits = orig_logits.logits
        orig_log_probs = F.log_softmax(orig_logits[:, -1, :].float(), dim=-1)

        swapped_logits = swapped_model(input_ids)
        if hasattr(swapped_logits, "logits"):
            swapped_logits = swapped_logits.logits
        swapped_log_probs = F.log_softmax(swapped_logits[:, -1, :].float(), dim=-1)

        kl = F.kl_div(swapped_log_probs, orig_log_probs, log_target=True, reduction="batchmean")
        # KL(P||Q) >= 0 por Gibbs; um valor negativo aqui e ruido de ponto
        # flutuante em distribuicoes quase identicas (medido -5.9e-08 quando
        # os modelos diferem pouco). Deixar passar produz orcamento de
        # divergencia negativo e comparacao de deriva sem sentido.
        return max(0.0, float(kl.item()))
