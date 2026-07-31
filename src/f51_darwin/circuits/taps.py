"""Strict, allowlisted activation taps for portable circuits."""

from __future__ import annotations

import re
from collections.abc import Callable
from types import TracebackType

import torch
from torch import nn

from f51_darwin.circuits.manifest import TapContract


_NAME_SEGMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_INDEX_SEGMENT = re.compile(r"^[0-9]+$")


class CircuitTapError(ValueError):
    """A tap path or captured activation violates its declared contract."""


def resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve only registered submodules using a strict public dotted path."""
    if not isinstance(root, nn.Module):
        raise CircuitTapError("root must be an nn.Module")
    if not isinstance(dotted_path, str) or not dotted_path:
        raise CircuitTapError("module path must be a non-empty dotted path")

    current = root
    for segment in dotted_path.split("."):
        if not segment or segment.startswith("_"):
            raise CircuitTapError(f"private or empty module path segment: {segment!r}")
        if isinstance(current, nn.ModuleList):
            if _INDEX_SEGMENT.fullmatch(segment) is None:
                raise CircuitTapError("ModuleList paths require decimal indices")
            index = int(segment)
            if index >= len(current):
                raise CircuitTapError(f"ModuleList index out of range: {segment}")
            current = current[index]
            continue
        if _NAME_SEGMENT.fullmatch(segment) is None:
            raise CircuitTapError(f"invalid registered-module segment: {segment!r}")
        child = current._modules.get(segment)
        if not isinstance(child, nn.Module):
            raise CircuitTapError(f"registered submodule does not exist: {segment!r}")
        current = child
    return current


class TapCapture:
    """Install one exclusive tap, validate its tensor, and optionally replace it."""

    def __init__(
        self,
        model: nn.Module,
        contract: TapContract,
        *,
        intervention: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if not isinstance(contract, TapContract):
            raise CircuitTapError("contract must be a TapContract")
        if contract.provider != "module_path_v1":
            raise CircuitTapError(f"unsupported tap provider: {contract.provider!r}")
        if contract.output_selector != "tensor":
            raise CircuitTapError(
                f"unsupported output selector: {contract.output_selector!r}"
            )
        self.model = model
        self.contract = contract
        self.module = resolve_module(model, contract.module_path)
        self.intervention = intervention
        self.activation: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def installed(self) -> bool:
        return self._handle is not None

    def _validate(self, activation: object) -> torch.Tensor:
        if not isinstance(activation, torch.Tensor):
            raise CircuitTapError("tap output selector requires a tensor")
        if activation.ndim != self.contract.rank:
            raise CircuitTapError(
                f"tap rank mismatch: expected {self.contract.rank}, got {activation.ndim}"
            )
        if activation.shape[-1] != self.contract.width:
            raise CircuitTapError(
                f"tap width mismatch: expected {self.contract.width}, got {activation.shape[-1]}"
            )
        sequence_length = activation.shape[-2] if activation.ndim >= 2 else 1
        if sequence_length < self.contract.minimum_sequence_length:
            raise CircuitTapError(
                "tap minimum sequence length mismatch: "
                f"expected >= {self.contract.minimum_sequence_length}, got {sequence_length}"
            )
        return activation

    def _post_hook(
        self, _module: nn.Module, _inputs: tuple[object, ...], output: object
    ) -> torch.Tensor:
        activation = self._validate(output)
        self.activation = activation
        if self.intervention is None:
            return activation
        replacement = self.intervention(activation)
        replacement = self._validate(replacement)
        if replacement.shape != activation.shape:
            raise CircuitTapError("intervention changed the activation shape")
        return replacement

    def _pre_hook(
        self, _module: nn.Module, inputs: tuple[object, ...]
    ) -> tuple[object, ...]:
        if not inputs:
            raise CircuitTapError("pre tap received no positional activation")
        activation = self._validate(inputs[0])
        self.activation = activation
        if self.intervention is None:
            return inputs
        replacement = self.intervention(activation)
        replacement = self._validate(replacement)
        if replacement.shape != activation.shape:
            raise CircuitTapError("intervention changed the activation shape")
        return (replacement, *inputs[1:])

    def install(self) -> TapCapture:
        if self.installed:
            raise CircuitTapError("tap is already installed")
        if getattr(self.module, "_f51_circuit_tap_occupied", False):
            raise CircuitTapError("tap is occupied")
        setattr(self.module, "_f51_circuit_tap_occupied", True)
        try:
            if self.contract.position == "post":
                self._handle = self.module.register_forward_hook(self._post_hook)
            else:
                self._handle = self.module.register_forward_pre_hook(self._pre_hook)
        except Exception:
            delattr(self.module, "_f51_circuit_tap_occupied")
            raise
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if getattr(self.module, "_f51_circuit_tap_occupied", False):
            delattr(self.module, "_f51_circuit_tap_occupied")

    def __enter__(self) -> TapCapture:
        return self.install()

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.remove()


__all__ = ["CircuitTapError", "TapCapture", "resolve_module"]
