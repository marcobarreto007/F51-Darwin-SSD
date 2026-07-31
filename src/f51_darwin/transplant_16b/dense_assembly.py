from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.state_identity import tokenizer_identity

from .assembly import _grow_organs
from .checkpoint import (
    TransplantCheckpointMetadata,
    TransplantCheckpointWriter,
    verify_shard_manifest,
)
from .cli import (
    CALIBRATION_PROMPTS,
    discover_source_identity,
)
from .exact_assembly import (
    _atomic_json,
    _copy,
    _neutralize_first_boot,
    _new_model,
    _parity_gate,
)
from .ledger import CoverageLedger, CoverageRecord
from .organs import derive_organ_projection
from .sources import inventory_safetensor, sha256_file, verify_sources
from .tokenizer import SmolTokenizerAdapter


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "src/configs/darwin_x_1.7b_smol_dense.yaml"
CHECKPOINT_ROOT = (
    ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1"
)
RUNTIME_ROOT = (
    ROOT / "workspace/runtime/darwin_17b_smol_dense_v1"
)
TOKENIZER_ROOT = (
    ROOT / "workspace/01_TOKENIZER/smol_49152_transplant_v1"
)
IMPLEMENTATION = "darwin-smol-native-dense-v1"


def _config() -> tuple[dict[str, Any], DarwinXConfig]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = DarwinXConfig.from_mapping(raw)
    dense = (
        config.feed_forward_kind == "dense_swiglu"
        and config.d_model == 2048
        and config.n_layers == 24
        and config.n_heads == 32
        and config.n_kv_heads == 32
        and config.fine_experts == 1
        and config.shared_experts == 0
        and config.experts_per_token == 1
        and config.fine_expert_hidden_dim == 8192
        and config.attention_layer_indices == tuple(range(24))
        and math.isclose(config.residual_scale, 1.0)
        and config.rms_norm_eps == 1e-5
        and config.rms_norm_fp32
        and not config.dae_enabled
    )
    if not dense:
        raise ValueError("native dense Smol anatomy contract drifted")
    return raw, config


def _plan_payload() -> dict[str, Any]:
    _, config = _config()
    sources = discover_source_identity()
    verify_sources(sources)
    payload = {
        "schema": "darwin-smol-dense-plan-v1",
        "implementation": IMPLEMENTATION,
        "config": dataclasses.asdict(config),
        "config_source": str(CONFIG_PATH.resolve()),
        "sources": dataclasses.asdict(sources),
        "checkpoint_root": str(CHECKPOINT_ROOT.resolve()),
        "runtime_root": str(RUNTIME_ROOT.resolve()),
        "tokenizer_root": str(TOKENIZER_ROOT.resolve()),
        "invariants": {
            "residual_projection": "identity",
            "layer_mapping": "24_to_24",
            "attention_mapping": "32_heads_to_32_heads",
            "feed_forward_mapping": "dense_to_native_dense",
            "moe_modules": 0,
            "dense_ffn_modules": 24,
            "organ_effects_at_first_boot": "neutral",
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = json.loads(canonical)
    payload["plan_id"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def write_dense_plan() -> dict[str, Any]:
    payload = _plan_payload()
    path = RUNTIME_ROOT / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing dense-brain plan identity differs")
        return payload
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("dense-brain checkpoint root is not empty")
    _atomic_json(path, payload)
    return payload


def dense_layer_mapping(layer: int) -> dict[str, tuple[str, ...]]:
    if not 0 <= layer < 24:
        raise ValueError("dense Smol layer must be within [0, 24)")
    source = f"model.layers.{layer}"
    target = f"blocks.{layer}"
    return {
        f"{source}.input_layernorm.weight": (
            f"{target}.norm1.weight",
        ),
        f"{source}.post_attention_layernorm.weight": (
            f"{target}.norm2.weight",
        ),
        f"{source}.self_attn.q_proj.weight": (
            f"{target}.attention.q_proj.weight",
        ),
        f"{source}.self_attn.k_proj.weight": (
            f"{target}.attention.k_proj.weight",
        ),
        f"{source}.self_attn.v_proj.weight": (
            f"{target}.attention.v_proj.weight",
        ),
        f"{source}.self_attn.o_proj.weight": (
            f"{target}.attention.o_proj.weight",
        ),
        f"{source}.mlp.gate_proj.weight": (
            f"{target}.ffn.gate_proj.weight",
        ),
        f"{source}.mlp.up_proj.weight": (
            f"{target}.ffn.up_proj.weight",
        ),
        f"{source}.mlp.down_proj.weight": (
            f"{target}.ffn.down_proj.weight",
        ),
    }


def _language_targets(state: dict[str, torch.Tensor]) -> set[str]:
    targets = {
        "token_embedding.weight",
        "lm_head.weight",
        "norm.weight",
    }
    for layer in range(24):
        for target_keys in dense_layer_mapping(layer).values():
            targets.update(target_keys)
    missing = targets.difference(state)
    if missing:
        raise ValueError(
            f"dense target language anatomy missing: {sorted(missing)}"
        )
    return targets


def _assert_dense_structure(
    model: torch.nn.Module,
    *,
    expected_layers: int,
) -> dict[str, int]:
    blocks = tuple(model.blocks)
    moe_modules = sum(
        int(getattr(block, "moe", None) is not None)
        for block in blocks
    )
    dense_ffn_modules = sum(
        int(getattr(block, "ffn", None) is not None)
        for block in blocks
    )
    if len(blocks) != expected_layers:
        raise ValueError(
            f"dense model has {len(blocks)} blocks, expected "
            f"{expected_layers}"
        )
    if moe_modules:
        raise ValueError(
            f"dense candidate contains {moe_modules} MoE modules"
        )
    if dense_ffn_modules != expected_layers:
        raise ValueError(
            "dense candidate does not have one native FFN per layer"
        )
    if any(".moe." in key for key in model.state_dict()):
        raise ValueError("dense candidate state contains a .moe. key")
    return {
        "moe_modules": moe_modules,
        "dense_ffn_modules": dense_ffn_modules,
    }


def _map_dense_brain(
    model: DarwinXModel,
    donor_path: Path,
    *,
    plan_id: str,
) -> CoverageLedger:
    state = model.state_dict()
    ledger = CoverageLedger.open(
        RUNTIME_ROOT / "coverage.jsonl",
        plan_id=plan_id,
    )
    if ledger.records:
        raise ValueError("dense-brain ledger is not empty")

    def record(source: str, targets: tuple[str, ...], method: str) -> None:
        ledger.append(
            CoverageRecord(
                source_tensor=source,
                target_tensors=targets,
                method=method,
                status="complete",
                metrics={
                    "function_preserving": True,
                    "structurally_dense": True,
                },
            )
        )

    with safe_open(str(donor_path), framework="pt", device="cpu") as donor:
        embedding = donor.get_tensor("model.embed_tokens.weight")
        _copy(state, "token_embedding.weight", embedding)
        _copy(state, "lm_head.weight", embedding)
        record(
            "model.embed_tokens.weight",
            ("token_embedding.weight", "lm_head.weight"),
            "exact_tied_copy",
        )
        _copy(state, "norm.weight", donor.get_tensor("model.norm.weight"))
        record("model.norm.weight", ("norm.weight",), "exact_copy")
        for layer in range(24):
            for source_key, target_keys in dense_layer_mapping(
                layer
            ).items():
                value = donor.get_tensor(source_key)
                for target_key in target_keys:
                    _copy(state, target_key, value)
                record(source_key, target_keys, "exact_native_dense_copy")

    donor_names = {
        tensor.name for tensor in inventory_safetensor(donor_path)
    }
    ledger.assert_complete(donor_names, _language_targets(state))
    model.load_state_dict(state, strict=True)
    return ledger


def build_dense_brain() -> dict[str, Any]:
    plan = write_dense_plan()
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("dense-brain checkpoint root is not empty")
    _, config = _config()
    sources = discover_source_identity()
    verify_sources(sources)
    model = _new_model(config)
    structure = _assert_dense_structure(model, expected_layers=24)
    ledger = _map_dense_brain(
        model,
        Path(sources.smol_weights.path),
        plan_id=plan["plan_id"],
    )
    organ_projection = derive_organ_projection(
        source_dim=512,
        target_dim=2048,
        seed=20_260_729,
    )
    organ_report, organ_cosine = _grow_organs(
        model,
        Path(sources.organ_checkpoint.path),
        organ_projection,
    )
    _neutralize_first_boot(model)
    structure = _assert_dense_structure(model, expected_layers=24)
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(
                f"non-finite tensor after dense surgery: {name}"
            )
    tokenizer = SmolTokenizerAdapter.load(TOKENIZER_ROOT)
    parity = _parity_gate(model, tokenizer)
    metadata = TransplantCheckpointMetadata(
        tokenizer_id=tokenizer_identity(tokenizer),
        plan_id=plan["plan_id"],
        source_identity={
            "smol_snapshot": sources.smol_snapshot,
            "smol_weights": sources.smol_weights.sha256,
            "organ": sources.organ_checkpoint.sha256,
        },
        projection_identity=hashlib.sha256(
            b"native-dense-identity-2048"
        ).hexdigest(),
        organ_projection_identity=organ_projection.identity(),
        coverage_head=ledger.head,
        calibration_digest=hashlib.sha256(
            json.dumps(CALIBRATION_PROMPTS).encode("utf-8")
        ).hexdigest(),
    )
    result = TransplantCheckpointWriter(CHECKPOINT_ROOT).write(
        model,
        config,
        metadata,
    )
    report = {
        "schema": "darwin-smol-dense-build-v1",
        "status": "native_dense_engineering_candidate",
        "feed_forward_kind": config.feed_forward_kind,
        **structure,
        "plan_id": plan["plan_id"],
        "checkpoint": str(result.checkpoint),
        "checkpoint_sha256": result.checkpoint_sha256,
        "base_checkpoint_id": result.base_checkpoint_id,
        "coverage_head": ledger.head,
        "donor_tensor_count": len(ledger.records),
        "organ_report": organ_report,
        "organ_subspace_min_cosine": organ_cosine,
        "parity": parity,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    if report["donor_tensor_count"] != 218:
        raise ValueError(
            "dense donor coverage count is not exactly 218"
        )
    _atomic_json(RUNTIME_ROOT / "build-report.json", report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def publish_dense_candidate() -> dict[str, Any]:
    checkpoint = CHECKPOINT_ROOT / "organism_cycle_000.pt"
    manifest = CHECKPOINT_ROOT / "organism_cycle_000.manifest.json"
    manifest_payload = verify_shard_manifest(checkpoint, manifest)
    build = json.loads(
        (RUNTIME_ROOT / "build-report.json").read_text(encoding="utf-8")
    )
    gates_path = RUNTIME_ROOT / "knowledge-gates.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    optimizer_path = RUNTIME_ROOT / "optimizer-smoke.json"
    optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
    native_log = (RUNTIME_ROOT / "native.stdout.log").read_text(
        encoding="utf-8"
    )
    if build.get("checkpoint_sha256") != manifest_payload.get(
        "checkpoint_sha256"
    ):
        raise ValueError("dense build and checkpoint manifest disagree")
    if build.get("feed_forward_kind") != "dense_swiglu":
        raise ValueError("candidate is not native dense SwiGLU")
    if build.get("moe_modules") != 0:
        raise ValueError("candidate still contains MoE modules")
    if build.get("dense_ffn_modules") != 24:
        raise ValueError("candidate does not contain 24 dense FFNs")
    if build.get("donor_tensor_count") != 218:
        raise ValueError("candidate donor coverage is incomplete")
    if not gates.get("passed") or gates.get("failures"):
        raise ValueError("dense knowledge gates did not pass")
    optimizer_gates = (
        "finite_loss",
        "gate_grad_nonzero",
        "up_grad_nonzero",
        "down_grad_nonzero",
        "weights_changed",
    )
    if not all(optimizer.get(key) is True for key in optimizer_gates):
        raise ValueError("dense optimizer smoke did not pass")
    required_native = (
        "donor_loaded=false",
        '"moe_modules": 0',
        '"dense_ffn_modules": 24',
    )
    if not all(marker in native_log for marker in required_native):
        raise ValueError("dense donor-free native proof is absent")
    candidate = {
        "schema": "darwin-smol-dense-candidate-v1",
        "status": "approved_candidate",
        "feed_forward_kind": "dense_swiglu",
        "moe_modules": 0,
        "dense_ffn_modules": 24,
        "checkpoint": checkpoint.name,
        "checkpoint_manifest": manifest.name,
        "checkpoint_sha256": manifest_payload["checkpoint_sha256"],
        "base_checkpoint_id": manifest_payload["base_checkpoint_id"],
        "plan_id": build["plan_id"],
        "knowledge_gates_sha256": sha256_file(gates_path),
        "optimizer_smoke_sha256": sha256_file(optimizer_path),
        "knowledge_metrics": gates["metrics"],
        "parity": build["parity"],
        "donor_free_native": True,
        "dual_gpu": True,
        "training_steps": gates["training_steps"],
    }
    destination = CHECKPOINT_ROOT / "candidate-manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != candidate:
            raise FileExistsError(
                "refusing to replace approved dense candidate"
            )
        return candidate
    _atomic_json(destination, candidate)
    return candidate
