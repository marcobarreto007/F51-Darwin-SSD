from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PLAN_SCHEMA = "darwin-smol-transplant-plan-v1"
IMPLEMENTATION_VERSION = "darwin-smol-surgery-v1"
CALIBRATION_PROMPTS = (
    "O futuro da inteligencia artificial depende de",
    "Explique por que a agua ferve.",
    "If all birds have wings and a robin is a bird, then",
    "Solve 3x + 7 = 22.",
    "Write a Python function that reverses a list.",
    "A causal model differs from correlation because",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("source path must not be empty")
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, field="source sha256"),
        )
        if self.size_bytes < 0:
            raise ValueError("source size_bytes must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SourceFile":
        return cls(
            path=str(raw["path"]),
            sha256=str(raw["sha256"]),
            size_bytes=int(raw["size_bytes"]),
        )


@dataclass(frozen=True)
class SourceIdentity:
    smol_snapshot: str
    smol_config: SourceFile
    smol_weights: SourceFile
    tokenizer_json: SourceFile
    tokenizer_config: SourceFile
    special_tokens: SourceFile
    organ_checkpoint: SourceFile

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SourceIdentity":
        return cls(
            smol_snapshot=str(raw["smol_snapshot"]),
            smol_config=SourceFile.from_mapping(raw["smol_config"]),
            smol_weights=SourceFile.from_mapping(raw["smol_weights"]),
            tokenizer_json=SourceFile.from_mapping(raw["tokenizer_json"]),
            tokenizer_config=SourceFile.from_mapping(raw["tokenizer_config"]),
            special_tokens=SourceFile.from_mapping(raw["special_tokens"]),
            organ_checkpoint=SourceFile.from_mapping(raw["organ_checkpoint"]),
        )

    def files(self) -> tuple[SourceFile, ...]:
        return (
            self.smol_config,
            self.smol_weights,
            self.tokenizer_json,
            self.tokenizer_config,
            self.special_tokens,
            self.organ_checkpoint,
        )


@dataclass(frozen=True)
class TargetAnatomy:
    model_name: str
    config_identity: str
    checkpoint_root: str
    runtime_root: str
    tokenizer_root: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TargetAnatomy":
        return cls(
            model_name=str(raw["model_name"]),
            config_identity=str(raw["config_identity"]),
            checkpoint_root=str(raw["checkpoint_root"]),
            runtime_root=str(raw["runtime_root"]),
            tokenizer_root=str(raw["tokenizer_root"]),
        )


@dataclass(frozen=True)
class TransplantPlan:
    schema: str
    implementation_version: str
    sources: SourceIdentity
    target: TargetAnatomy
    calibration_digest: str
    projection_seed: int
    layer_groups: tuple[tuple[int, ...], ...]
    attention_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema != PLAN_SCHEMA:
            raise ValueError(f"unsupported transplant plan schema: {self.schema}")
        _require_sha256(
            self.calibration_digest,
            field="calibration_digest",
        )
        if self.projection_seed < 0:
            raise ValueError("projection_seed must be non-negative")
        flattened = tuple(
            layer for group in self.layer_groups for layer in group
        )
        if flattened != tuple(range(len(flattened))):
            raise ValueError("layer_groups must cover donor layers monotonically")
        if any(
            block < 0 or block >= len(self.layer_groups)
            for block in self.attention_blocks
        ):
            raise ValueError("attention block index is outside target layer range")

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, *, pretty: bool = False) -> str:
        if pretty:
            return json.dumps(
                self.to_mapping(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        return _canonical_json(self.to_mapping())

    def identity(self) -> str:
        return canonical_sha256(self.to_mapping())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TransplantPlan":
        return cls(
            schema=str(raw["schema"]),
            implementation_version=str(raw["implementation_version"]),
            sources=SourceIdentity.from_mapping(raw["sources"]),
            target=TargetAnatomy.from_mapping(raw["target"]),
            calibration_digest=str(raw["calibration_digest"]),
            projection_seed=int(raw["projection_seed"]),
            layer_groups=tuple(
                tuple(int(layer) for layer in group)
                for group in raw["layer_groups"]
            ),
            attention_blocks=tuple(int(index) for index in raw["attention_blocks"]),
        )

    @classmethod
    def testing(cls, root: Path) -> "TransplantPlan":
        source = SourceFile(
            path=str(root / "source.bin"),
            sha256="0" * 64,
            size_bytes=0,
        )
        return cls(
            schema=PLAN_SCHEMA,
            implementation_version=IMPLEMENTATION_VERSION,
            sources=SourceIdentity(
                smol_snapshot="test-snapshot",
                smol_config=source,
                smol_weights=source,
                tokenizer_json=source,
                tokenizer_config=source,
                special_tokens=source,
                organ_checkpoint=source,
            ),
            target=TargetAnatomy(
                model_name="F51-Darwin-X-Test",
                config_identity="darwin-config-v1:test",
                checkpoint_root=str(root / "checkpoints"),
                runtime_root=str(root / "runtime"),
                tokenizer_root=str(root / "tokenizer"),
            ),
            calibration_digest="1" * 64,
            projection_seed=7,
            layer_groups=((0,), (1,)),
            attention_blocks=(1,),
        )


def write_plan_atomic(plan: TransplantPlan, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"incomplete plan already exists: {temporary}")
    payload = {
        **plan.to_mapping(),
        "plan_id": plan.identity(),
    }
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target
