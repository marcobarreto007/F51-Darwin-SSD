from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from f51_darwin.artifacts import resolve_token_bin
from f51_darwin.darwin_x import DarwinXConfig


@dataclass(frozen=True)
class TokenSourceHealth:
    path: Path
    token_count: int
    sample_min: int
    sample_max: int
    model_vocab_size: int
    tokenizer_vocab_size: int | None

    @property
    def in_vocab(self) -> bool:
        return 0 <= self.sample_min and self.sample_max < self.model_vocab_size

    @property
    def tokenizer_matches_model(self) -> bool:
        return self.tokenizer_vocab_size == self.model_vocab_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "token_count": self.token_count,
            "sample_min": self.sample_min,
            "sample_max": self.sample_max,
            "model_vocab_size": self.model_vocab_size,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "in_vocab": self.in_vocab,
            "tokenizer_matches_model": self.tokenizer_matches_model,
        }


class ConcatenatedTokenBins:
    is_virtual_token_sequence = True

    def __init__(self, arrays: list[np.ndarray], paths: list[Path]) -> None:
        if not arrays:
            raise ValueError("Token index contains no token arrays.")
        self.arrays = arrays
        self.paths = paths
        self.lengths = [int(len(array)) for array in arrays]
        if any(length <= 0 for length in self.lengths):
            raise ValueError("Token index contains an empty token bin.")
        self.cumulative = np.cumsum(self.lengths, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                return np.asarray([self[index] for index in range(start, stop, step)], dtype=np.int32)
            return self.slice(start, stop)
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        array_index = bisect.bisect_right(self.cumulative, item)
        previous = int(self.cumulative[array_index - 1]) if array_index else 0
        return int(self.arrays[array_index][item - previous])

    def slice(self, start: int, stop: int) -> np.ndarray:
        if stop <= start:
            return np.asarray([], dtype=np.int32)
        chunks: list[np.ndarray] = []
        position = start
        while position < stop:
            array_index = bisect.bisect_right(self.cumulative, position)
            previous = int(self.cumulative[array_index - 1]) if array_index else 0
            local_start = position - previous
            local_stop = min(stop - previous, self.lengths[array_index])
            chunks.append(np.asarray(self.arrays[array_index][local_start:local_stop], dtype=np.int32))
            position += local_stop - local_start
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks).astype(np.int32, copy=False)


def _read_int32_token_bin(path: Path, *, ram_limit_bytes: int) -> np.ndarray:
    size = path.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"Token bin is not int32 aligned: {path}")
    if size == 0:
        raise ValueError(f"Token bin is empty: {path}")
    if size < ram_limit_bytes:
        return np.fromfile(path, dtype=np.int32)
    return np.memmap(path, dtype=np.int32, mode="r")


def load_token_ids(
    project_root: Path,
    *,
    token_bin: str | Path | None = None,
    token_index: str | Path | None = None,
    ram_limit_bytes: int = 2_000_000_000,
) -> tuple[np.ndarray, Path]:
    if token_bin is not None and token_index is not None:
        raise ValueError("Use either token_bin or token_index, not both.")
    if token_index is not None:
        return load_token_ids_from_index(project_root, token_index, ram_limit_bytes=ram_limit_bytes)

    selected = Path(token_bin) if token_bin is not None else resolve_token_bin(project_root, min_bytes=4)
    if selected is None:
        raise FileNotFoundError(
            "No usable token bin found. Expected the canonical external "
            "F51-Dataset-Organizado token feast or an explicit --token-bin."
        )
    if not selected.is_absolute():
        selected = project_root / selected
    return _read_int32_token_bin(selected, ram_limit_bytes=ram_limit_bytes), selected


def load_token_ids_from_index(
    project_root: Path,
    token_index: str | Path,
    *,
    ram_limit_bytes: int = 2_000_000_000,
) -> tuple[np.ndarray | ConcatenatedTokenBins, Path]:
    selected = Path(token_index)
    if not selected.is_absolute():
        selected = project_root / selected
    payload = json.loads(selected.read_text(encoding="utf-8"))
    batches = payload.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError(f"Token index has no batches: {selected}")
    arrays: list[np.ndarray] = []
    paths: list[Path] = []
    for record in batches:
        if not isinstance(record, dict):
            raise ValueError("Token index batch record must be a mapping.")
        raw_path = record.get("bin")
        if not raw_path:
            raise ValueError("Token index batch missing bin path.")
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = selected.parent / path
        if not path.exists():
            raise FileNotFoundError(f"Token index bin not found: {path}")
        expected_tokens = int(record.get("tokens", -1))
        if expected_tokens < 2:
            raise ValueError(f"Token index batch has invalid token count: {path}")
        array = _read_int32_token_bin(path, ram_limit_bytes=ram_limit_bytes)
        if len(array) != expected_tokens:
            raise ValueError(
                f"Token index token count mismatch for {path}: "
                f"manifest={expected_tokens}, actual={len(array)}"
            )
        arrays.append(array)
        paths.append(path)
    total_tokens = int(payload.get("total_tokens", -1))
    actual_tokens = sum(len(array) for array in arrays)
    if total_tokens != actual_tokens:
        raise ValueError(
            f"Token index total_tokens mismatch: manifest={total_tokens}, actual={actual_tokens}"
        )
    if len(arrays) == 1:
        return arrays[0], selected
    return ConcatenatedTokenBins(arrays, paths), selected


def _sample_token_ids(token_ids: np.ndarray | ConcatenatedTokenBins, sample_size: int) -> np.ndarray:
    if len(token_ids) == 0:
        raise ValueError("Token bin has zero int32 tokens.")
    take = min(sample_size, len(token_ids))
    indices = np.linspace(0, len(token_ids) - 1, num=take, dtype=np.int64)
    if isinstance(token_ids, np.ndarray):
        return np.asarray(token_ids[indices], dtype=np.int64)
    return np.asarray([token_ids[int(index)] for index in indices], dtype=np.int64)


def validate_token_source(
    token_ids: np.ndarray | ConcatenatedTokenBins,
    path: Path,
    config: DarwinXConfig,
    *,
    tokenizer_vocab_size: int | None = None,
    sample_size: int = 16384,
    strict_tokenizer_vocab: bool = False,
) -> TokenSourceHealth:
    sample = _sample_token_ids(token_ids, sample_size)
    health = TokenSourceHealth(
        path=path,
        token_count=int(len(token_ids)),
        sample_min=int(sample.min()),
        sample_max=int(sample.max()),
        model_vocab_size=int(config.vocab_size),
        tokenizer_vocab_size=tokenizer_vocab_size,
    )
    if not health.in_vocab:
        raise ValueError(
            "Token bin has ids outside model vocab: "
            f"sample_min={health.sample_min}, sample_max={health.sample_max}, "
            f"model_vocab_size={health.model_vocab_size}"
        )
    if strict_tokenizer_vocab and not health.tokenizer_matches_model:
        raise ValueError(
            "Tokenizer vocab does not match model vocab: "
            f"tokenizer={tokenizer_vocab_size}, model={config.vocab_size}"
        )
    return health


def amp_settings(device: torch.device, precision: str) -> tuple[torch.dtype | None, bool]:
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16"
        else:
            precision = "fp32"
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but this CUDA device does not report BF16 support.")
        return torch.bfloat16, True
    if precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError("FP16 AMP is only supported on CUDA for this trainer.")
        return torch.float16, True
    if precision == "fp32":
        return None, False
    raise ValueError(f"Unsupported precision: {precision}")


def require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value.detach()).all():
        raise FloatingPointError(f"Non-finite {name}: {value.detach().float().cpu()}")


def train_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    amp_enabled = amp_dtype is not None and batch.device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=batch.device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        output = model(batch, labels=batch)
        loss = output.loss
    if loss is None:
        raise RuntimeError("Darwin-X model returned no loss for a labeled batch.")
    require_finite("loss", loss)

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        grad_clip,
        error_if_nonfinite=True,
    )
    require_finite("gradient norm", grad_norm)

    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    apply_autonomic = getattr(model, "apply_pending_autonomic_actions", None)
    if callable(apply_autonomic):
        apply_autonomic()

    metrics = {
        "loss": float(loss.detach().float().cpu()),
        "grad_norm": float(grad_norm.detach().float().cpu()),
    }
    for name in ("lm_loss", "mtp_loss", "jepa_loss", "aux_loss"):
        tensor = getattr(output, name, None)
        if tensor is not None:
            require_finite(name, tensor)
            metrics[name] = float(tensor.detach().float().cpu())
    # ── Log MoE routing health ──
    if output.moe_stats:
        last_layer = output.moe_stats[-1]
        lb = last_layer.get("load_balance_loss")
        rz = last_layer.get("router_z_loss")
        dead = last_layer.get("dead_experts", [])
        tokens_per = last_layer.get("tokens_per_expert", [])
        if lb is not None:
            metrics["moelb"] = float(lb.detach().cpu()) if hasattr(lb, 'detach') else float(lb)
        if rz is not None:
            metrics["moerz"] = float(rz.detach().cpu()) if hasattr(rz, 'detach') else float(rz)
        metrics["moe_dead"] = len(dead)
        if tokens_per:
            metrics["moe_max_usage"] = max(tokens_per)
            metrics["moe_min_usage"] = min(tokens_per)
    if getattr(output, "heartbeat_stats", None):
        heartbeat_stats = output.heartbeat_stats or {}
        for source, target in (
            ("beat", "heartbeat_beat"),
            ("dopamine", "heartbeat_dopamine"),
            ("mem_writes", "heartbeat_memory_writes"),
            ("explorations", "heartbeat_explorations"),
            ("thoughts", "heartbeat_thoughts"),
        ):
            value = heartbeat_stats.get(source)
            if isinstance(value, (int, float)):
                metrics[target] = float(value)
    return metrics
