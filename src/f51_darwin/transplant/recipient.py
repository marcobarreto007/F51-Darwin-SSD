from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .slots import (
    GABAResidualSlot,
    HeartbeatSlot,
    IHSResidualSlot,
    JEPAAuxiliarySlot,
    MTPSlot,
    SpiderSenseSlot,
    TTMResidualSlot,
)


_ORGAN_MODES = {
    "disabled",
    "shadow",
    "adapter_active",
    "organ_unfrozen",
}
_ORGAN_EXECUTION_MODES = {
    "disabled",
    "shadow",
    "active",
    "train_only",
}

GUARDED_TRANSPLANT_POLICY = {
    "gaba.0": "active",
    "jepa": "train_only",
    "ttm": "shadow",
    "spider": "shadow",
    "heartbeat": "shadow",
    "mtp": "disabled",
    "ihs": "disabled",
}


@dataclass
class RecipientOutput:
    logits: torch.Tensor
    jepa_loss: torch.Tensor | None
    spider_loss: torch.Tensor | None
    mtp_loss: torch.Tensor | None
    heartbeat_loss: torch.Tensor | None
    spider_confidence: torch.Tensor | None
    organ_observations: dict[str, Any]
    hidden_states: tuple[torch.Tensor, ...] | None


class TwoDonorRecipient(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        *,
        jepa_slot: JEPAAuxiliarySlot,
        gaba_slots: Mapping[int, tuple[GABAResidualSlot, ...]],
        spider_slot: SpiderSenseSlot | None = None,
        mtp_slot: MTPSlot | None = None,
        ttm_slot: TTMResidualSlot | None = None,
        heartbeat_slot: HeartbeatSlot | None = None,
        ihs_slot: IHSResidualSlot | None = None,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.jepa_slot = jepa_slot
        self.spider_slot = spider_slot
        self.mtp_slot = mtp_slot
        self.ttm_slot = ttm_slot
        self.heartbeat_slot = heartbeat_slot
        self.ihs_slot = ihs_slot
        self.gaba_slots = nn.ModuleDict()
        self._logical_to_safe: dict[str, str] = {}
        self._block_to_logical: dict[int, tuple[str, ...]] = {}

        donor_layer = 0
        block_count = len(self.language_model.transformer.h)
        for block_index in sorted(gaba_slots):
            if block_index < 0 or block_index >= block_count:
                raise ValueError(
                    f"recipient block index out of range: {block_index}"
                )
            logical_names: list[str] = []
            for slot in gaba_slots[block_index]:
                logical = f"gaba.{donor_layer}"
                safe = f"gaba_{donor_layer}"
                self.gaba_slots[safe] = slot
                self._logical_to_safe[logical] = safe
                logical_names.append(logical)
                donor_layer += 1
            self._block_to_logical[int(block_index)] = tuple(logical_names)

        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.eval()

    @classmethod
    def from_language_donor(
        cls,
        language_donor: nn.Module,
        *,
        jepa_slot: JEPAAuxiliarySlot,
        gaba_slots: Mapping[int, tuple[GABAResidualSlot, ...]],
        spider_slot: SpiderSenseSlot | None = None,
        mtp_slot: MTPSlot | None = None,
        ttm_slot: TTMResidualSlot | None = None,
        heartbeat_slot: HeartbeatSlot | None = None,
        ihs_slot: IHSResidualSlot | None = None,
    ) -> "TwoDonorRecipient":
        copied = copy.deepcopy(language_donor)
        return cls(
            copied,
            jepa_slot=jepa_slot,
            gaba_slots=gaba_slots,
            spider_slot=spider_slot,
            mtp_slot=mtp_slot,
            ttm_slot=ttm_slot,
            heartbeat_slot=heartbeat_slot,
            ihs_slot=ihs_slot,
        )

    @property
    def config(self):
        return self.language_model.config

    def organ_slot(
        self,
        logical_name: str,
    ) -> (
        GABAResidualSlot
        | HeartbeatSlot
        | IHSResidualSlot
        | JEPAAuxiliarySlot
        | SpiderSenseSlot
        | MTPSlot
        | TTMResidualSlot
    ):
        if logical_name == "jepa":
            return self.jepa_slot
        if logical_name == "spider" and self.spider_slot is not None:
            return self.spider_slot
        if logical_name == "mtp" and self.mtp_slot is not None:
            return self.mtp_slot
        if logical_name == "ttm" and self.ttm_slot is not None:
            return self.ttm_slot
        if logical_name == "heartbeat" and self.heartbeat_slot is not None:
            return self.heartbeat_slot
        if logical_name == "ihs" and self.ihs_slot is not None:
            return self.ihs_slot
        safe = self._logical_to_safe.get(logical_name)
        if safe is None:
            raise KeyError(f"unknown transplanted organ: {logical_name}")
        return self.gaba_slots[safe]

    def _language_forward(
        self,
        *,
        input_ids: torch.Tensor,
        output_hidden_states: bool,
        **kwargs: Any,
    ):
        options = dict(kwargs)
        options.pop("return_dict", None)
        options.pop("use_cache", None)
        options.pop("output_hidden_states", None)
        return self.language_model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
            **options,
        )

    def _available_organs(self) -> frozenset[str]:
        names = {"jepa", *self._logical_to_safe}
        for name, slot in (
            ("spider", self.spider_slot),
            ("mtp", self.mtp_slot),
            ("ttm", self.ttm_slot),
            ("heartbeat", self.heartbeat_slot),
            ("ihs", self.ihs_slot),
        ):
            if slot is not None:
                names.add(name)
        return frozenset(names)

    def _execution_modes(
        self,
        *,
        organ_mode: str,
        organ_policy: Mapping[str, str] | None,
    ) -> dict[str, str]:
        available = self._available_organs()
        if organ_policy is None:
            if organ_mode not in _ORGAN_MODES:
                raise ValueError(f"unsupported organ mode: {organ_mode}")
            resolved = (
                "active"
                if organ_mode in {"adapter_active", "organ_unfrozen"}
                else organ_mode
            )
            return {name: resolved for name in available}

        unknown = {
            name
            for name, mode in organ_policy.items()
            if name not in available and mode != "disabled"
        }
        if unknown:
            raise ValueError(
                f"unknown organs in execution policy: {sorted(unknown)}"
            )
        invalid = {
            name: mode
            for name, mode in organ_policy.items()
            if mode not in _ORGAN_EXECUTION_MODES
        }
        if invalid:
            raise ValueError(f"unsupported organ execution modes: {invalid}")
        return {
            name: str(organ_policy.get(name, "disabled"))
            for name in available
        }

    def _mode_is_active(self, mode: str) -> bool:
        return mode == "active" or (mode == "train_only" and self.training)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        organ_mode: str = "disabled",
        organ_policy: Mapping[str, str] | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> RecipientOutput:
        execution_modes = self._execution_modes(
            organ_mode=organ_mode,
            organ_policy=organ_policy,
        )
        if all(mode == "disabled" for mode in execution_modes.values()):
            output = self._language_forward(
                input_ids=input_ids,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
            return RecipientOutput(
                logits=output.logits,
                jepa_loss=None,
                spider_loss=None,
                mtp_loss=None,
                heartbeat_loss=None,
                spider_confidence=None,
                organ_observations={},
                hidden_states=(
                    tuple(output.hidden_states)
                    if output_hidden_states and output.hidden_states is not None
                    else None
                ),
            )

        observations: dict[str, Any] = {}
        handles: list[Any] = []

        ihs_mode = execution_modes.get("ihs", "disabled")
        if self.ihs_slot is not None and ihs_mode != "disabled":
            def ihs_hook(_module, _args, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("GPT embedding dropout output contract changed")
                slot_output = self.ihs_slot(
                    output,
                    enabled=self._mode_is_active(ihs_mode),
                )
                observations["ihs"] = slot_output.observation
                return slot_output.hidden

            handles.append(
                self.language_model.transformer.drop.register_forward_hook(
                    ihs_hook
                )
            )

        for block_index, logical_names in self._block_to_logical.items():
            block = self.language_model.transformer.h[block_index]
            enabled_names = tuple(
                name
                for name in logical_names
                if execution_modes.get(name, "disabled") != "disabled"
            )
            if not enabled_names:
                continue

            def hook(
                _module,
                _args,
                output,
                *,
                names=enabled_names,
            ):
                output_is_tensor = isinstance(output, torch.Tensor)
                if output_is_tensor:
                    hidden = output
                elif isinstance(output, tuple) and output:
                    hidden = output[0]
                else:
                    raise TypeError("GPT block output contract changed")
                for logical_name in names:
                    slot = self.organ_slot(logical_name)
                    slot_output = slot(
                        hidden,
                        enabled=self._mode_is_active(
                            execution_modes[logical_name]
                        ),
                        mutate_state=False,
                    )
                    hidden = slot_output.hidden
                    observations[logical_name] = slot_output.observation
                if output_is_tensor:
                    return hidden
                return (hidden, *output[1:])

            handles.append(block.register_forward_hook(hook))

        try:
            output = self._language_forward(
                input_ids=input_ids,
                output_hidden_states=True,
                **kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

        if output.hidden_states is None:
            raise RuntimeError("recipient requires GPT hidden states for JEPA")
        # TTM: memory retrieval + residual injection (modifies hidden_for_logits)
        hidden_for_organs = output.hidden_states[-1]
        final_logits = output.logits  # default: unmodified GPT-2 output
        ttm_mode = execution_modes.get("ttm", "disabled")
        if self.ttm_slot is not None and ttm_mode != "disabled":
            ttm_active = self._mode_is_active(ttm_mode)
            ttm_output = self.ttm_slot(
                hidden_for_organs,
                enabled=ttm_active,
            )
            hidden_for_organs = ttm_output.hidden
            observations["ttm"] = ttm_output.observation
            # Only recompute logits when TTM actually modifies hidden (gate != 0)
            if ttm_active:
                final_logits = self.language_model.lm_head(
                    self.language_model.transformer.ln_f(hidden_for_organs)
                )

        jepa_loss = None
        jepa_error = 0.0
        jepa_mode = execution_modes.get("jepa", "disabled")
        if jepa_mode != "disabled":
            jepa_output = self.jepa_slot(
                hidden_for_organs,
                enabled=self._mode_is_active(jepa_mode),
            )
            jepa_loss = jepa_output.auxiliary_loss
            jepa_error = float(
                jepa_loss.detach() if jepa_loss is not None else 0.0
            )
            observations["jepa"] = jepa_output.observation

        heartbeat_loss = None
        heartbeat_mode = execution_modes.get("heartbeat", "disabled")
        if self.heartbeat_slot is not None and heartbeat_mode != "disabled":
            heartbeat_output = self.heartbeat_slot(
                hidden_for_organs,
                enabled=self._mode_is_active(heartbeat_mode),
                jepa_error=jepa_error,
            )
            heartbeat_loss = heartbeat_output.auxiliary_loss
            observations["heartbeat"] = heartbeat_output.observation

        # Spider-Sense: confidence sidecar (no residual injection)
        spider_loss = None
        spider_conf = None
        spider_mode = execution_modes.get("spider", "disabled")
        if self.spider_slot is not None and spider_mode != "disabled":
            spider_output = self.spider_slot(
                hidden_for_organs,
                enabled=self._mode_is_active(spider_mode),
            )
            spider_loss = spider_output.auxiliary_loss
            spider_conf = spider_output.observation.get("confidence_mean", None)
            observations["spider"] = spider_output.observation

        # MTP: multi-token prediction sidecar (no residual injection)
        mtp_loss = None
        mtp_mode = execution_modes.get("mtp", "disabled")
        if self.mtp_slot is not None and mtp_mode != "disabled":
            mtp_output = self.mtp_slot(
                hidden_for_organs,
                enabled=self._mode_is_active(mtp_mode),
            )
            mtp_loss = mtp_output.auxiliary_loss
            observations["mtp"] = mtp_output.observation

        return RecipientOutput(
            logits=final_logits,
            jepa_loss=jepa_loss,
            spider_loss=spider_loss,
            mtp_loss=mtp_loss,
            heartbeat_loss=heartbeat_loss,
            spider_confidence=spider_conf,
            organ_observations=observations,
            hidden_states=(
                tuple(output.hidden_states) if output_hidden_states else None
            ),
        )
