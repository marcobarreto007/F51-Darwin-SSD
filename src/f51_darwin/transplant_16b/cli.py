from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import yaml

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.organism.checkpoint_root import model_config_identity

from .contracts import (
    CALIBRATION_PROMPTS,
    IMPLEMENTATION_VERSION,
    PLAN_SCHEMA,
    SourceFile,
    SourceIdentity,
    TargetAnatomy,
    TransplantPlan,
    write_plan_atomic,
)
from .layers import monotonic_layer_groups
from .sources import verify_sources


ROOT = Path(__file__).resolve().parents[3]
SMOL_SNAPSHOT = "31b70e2e869a7173562077fd711b654946d38674"
DEFAULT_SOURCE_ROOT = (
    ROOT
    / "workspace"
    / "00_DONORS"
    / "models--HuggingFaceTB--SmolLM2-1.7B-Instruct"
    / "snapshots"
    / SMOL_SNAPSHOT
)
DEFAULT_ORGAN_CHECKPOINT = (
    ROOT
    / "workspace"
    / "03_CHECKPOINTS_100M_FULL_ORGANISM_V3"
    / "organism_cycle_002_step_001750.pt"
)
APPROVED_FILES = {
    "config.json": (
        908,
        "994f50b16abb4ae00880baefe03c10260b5bd608d2bf586f7056ca05a534feea",
    ),
    "model.safetensors": (
        3_422_777_952,
        "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
    ),
    "tokenizer.json": (
        2_104_556,
        "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    ),
    "tokenizer_config.json": (
        3_764,
        "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
    ),
    "special_tokens_map.json": (
        655,
        "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
    ),
}
ORGAN_SIZE = 1_956_993_291
ORGAN_SHA256 = (
    "71c49bc295c0d06d72b3dc640d5b4d846f421ecdcca25820517fffaab6425a30"
)
def _source_file(root: Path, filename: str) -> SourceFile:
    size, digest = APPROVED_FILES[filename]
    return SourceFile(
        path=str((root / filename).resolve()),
        sha256=digest,
        size_bytes=size,
    )


def discover_source_identity(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    organ_checkpoint: str | Path = DEFAULT_ORGAN_CHECKPOINT,
) -> SourceIdentity:
    root = Path(source_root).resolve()
    return SourceIdentity(
        smol_snapshot=SMOL_SNAPSHOT,
        smol_config=_source_file(root, "config.json"),
        smol_weights=_source_file(root, "model.safetensors"),
        tokenizer_json=_source_file(root, "tokenizer.json"),
        tokenizer_config=_source_file(root, "tokenizer_config.json"),
        special_tokens=_source_file(root, "special_tokens_map.json"),
        organ_checkpoint=SourceFile(
            path=str(Path(organ_checkpoint).resolve()),
            sha256=ORGAN_SHA256,
            size_bytes=ORGAN_SIZE,
        ),
    )


def _calibration_digest() -> str:
    payload = json.dumps(
        CALIBRATION_PROMPTS,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_target_config(path: Path) -> tuple[dict, DarwinXConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("target config must contain a YAML mapping")
    config = DarwinXConfig.from_mapping(raw)
    if config.model_name != "F51-Darwin-X-1.6B-Smol-Transplant-V1":
        raise ValueError("target config is not the approved transplant lineage")
    if (
        config.vocab_size,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
    ) != (49_152, 1_920, 16, 30, 30):
        raise ValueError("target config anatomy differs from approved contract")
    return raw, config


def _plan_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    raw, config = _read_target_config(config_path)
    sources = discover_source_identity(
        args.source_root,
        args.organ_checkpoint,
    )
    verify_sources(sources)

    runtime_root = Path(args.runtime_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    tokenizer_root = Path(args.tokenizer_root).resolve()
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
        raise ValueError(
            f"target checkpoint root is not empty: {checkpoint_root}"
        )
    free_bytes = shutil.disk_usage(runtime_root.parent).free
    if free_bytes < 20 * 1024**3:
        raise RuntimeError(
            f"transplant preflight requires 20 GiB free; actual={free_bytes}"
        )
    plan = TransplantPlan(
        schema=PLAN_SCHEMA,
        implementation_version=IMPLEMENTATION_VERSION,
        sources=sources,
        target=TargetAnatomy(
            model_name=config.model_name,
            config_identity=model_config_identity(raw),
            checkpoint_root=str(checkpoint_root),
            runtime_root=str(runtime_root),
            tokenizer_root=str(tokenizer_root),
        ),
        calibration_digest=_calibration_digest(),
        projection_seed=20_260_728,
        layer_groups=monotonic_layer_groups(24, 16),
        attention_blocks=(3, 7, 11, 15),
    )
    plan_path = runtime_root / "plan.json"
    if plan_path.exists():
        existing = _load_plan(plan_path)
        if existing.identity() != plan.identity():
            raise ValueError("existing plan identity differs from requested plan")
        print(f"PLAN_OK plan_id={plan.identity()} reused=true")
        return 0
    write_plan_atomic(plan, plan_path)
    print(
        f"PLAN_OK plan_id={plan.identity()} "
        f"donor_files={len(sources.files())} free_gib={free_bytes / 1024**3:.1f}"
    )
    return 0


def _load_plan(path: str | Path) -> TransplantPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    plan = TransplantPlan.from_mapping(raw)
    if raw.get("plan_id") != plan.identity():
        raise ValueError("transplant plan identity mismatch")
    return plan


def _tiny_build(plan: TransplantPlan) -> int:
    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise ValueError("--tiny-test-mode is allowed only under pytest")
    runtime_root = Path(plan.target.runtime_root)
    marker = runtime_root / "tiny-build-ok.json"
    temporary = marker.with_suffix(".json.tmp")
    payload = {
        "schema": "darwin-smol-tiny-build-v1",
        "plan_id": plan.identity(),
        "calibration_imported": False,
    }
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)
    print(f"BUILD_TINY_OK plan_id={plan.identity()}")
    return 0


def _build_command(args: argparse.Namespace) -> int:
    plan = _load_plan(args.plan)
    verify_sources(plan.sources)
    if args.tiny_test_mode:
        return _tiny_build(plan)
    from .assembly import build_native_transplant

    result = build_native_transplant(plan, resume=args.resume)
    print(
        f"BUILD_OK status={result.status} "
        f"checkpoint={result.checkpoint} plan_id={plan.identity()}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash-bound SmolLM2 to native Darwin 1.6B surgery",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument(
        "--config",
        default=str(ROOT / "src/configs/darwin_x_1.6b_smol_transplant.yaml"),
    )
    plan.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    plan.add_argument(
        "--organ-checkpoint",
        default=str(DEFAULT_ORGAN_CHECKPOINT),
    )
    plan.add_argument(
        "--runtime-root",
        default=str(
            ROOT / "workspace/runtime/darwin_16b_smol_transplant_v1"
        ),
    )
    plan.add_argument(
        "--checkpoint-root",
        default=str(
            ROOT / "workspace/03_CHECKPOINTS_1.6B_SMOL_TRANSPLANT_V1"
        ),
    )
    plan.add_argument(
        "--tokenizer-root",
        default=str(
            ROOT / "workspace/01_TOKENIZER/smol_49152_transplant_v1"
        ),
    )
    build = subparsers.add_parser("build")
    build.add_argument("--plan", required=True)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--tiny-test-mode", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        return _plan_command(args)
    if args.command == "build":
        return _build_command(args)
    raise AssertionError(f"unreachable command: {args.command}")
