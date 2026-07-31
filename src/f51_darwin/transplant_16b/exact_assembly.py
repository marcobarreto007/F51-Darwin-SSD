from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from torch.nn import functional as F

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
    DEFAULT_ORGAN_CHECKPOINT,
    DEFAULT_SOURCE_ROOT,
    discover_source_identity,
)
from .ledger import CoverageLedger, CoverageRecord, DriftRecord
from .organs import derive_organ_projection
from .selection import Provenance, tensor_digest
from .sources import inventory_safetensor, sha256_file, verify_sources
from .tokenizer import SmolTokenizerAdapter


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "src/configs/darwin_x_1.7b_smol_exact.yaml"
CHECKPOINT_ROOT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_EXACT_V3"
RUNTIME_ROOT = ROOT / "workspace/runtime/darwin_17b_smol_exact_v3"
TOKENIZER_ROOT = ROOT / "workspace/01_TOKENIZER/smol_49152_transplant_v1"
IMPLEMENTATION = "darwin-smol-exact-brain-v3"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config() -> tuple[dict[str, Any], DarwinXConfig]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = DarwinXConfig.from_mapping(raw)
    exact = (
        config.d_model == 2048
        and config.n_layers == 24
        and config.n_heads == 32
        and config.n_kv_heads == 32
        and config.fine_experts == 1
        and config.shared_experts == 0
        and config.fine_expert_hidden_dim == 8192
        and config.attention_layer_indices == tuple(range(24))
        and math.isclose(config.residual_scale, 1.0)
        and config.rms_norm_eps == 1e-5
        and config.rms_norm_fp32
    )
    if not exact:
        raise ValueError("exact Smol brain anatomy contract drifted")
    return raw, config


def _plan_payload() -> dict[str, Any]:
    raw, config = _config()
    sources = discover_source_identity()
    verify_sources(sources)
    payload = {
        "schema": "darwin-smol-exact-plan-v3",
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
            "mlp_mapping": "dense_to_single_exact_expert",
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
    payload["plan_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_exact_plan() -> dict[str, Any]:
    payload = _plan_payload()
    path = RUNTIME_ROOT / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing exact-brain plan identity differs")
        return payload
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("exact-brain checkpoint root is not empty")
    _atomic_json(path, payload)
    return payload


def _new_model(config: DarwinXConfig) -> DarwinXModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return DarwinXModel(config)
    finally:
        torch.set_default_dtype(previous)


@torch.no_grad()
def _copy(
    state: dict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> None:
    target = state[key]
    if target.shape != value.shape:
        raise ValueError(
            f"exact tensor shape mismatch for {key}: "
            f"{tuple(value.shape)} != {tuple(target.shape)}"
        )
    target.copy_(value.to(dtype=target.dtype))


def _language_targets(state: dict[str, torch.Tensor]) -> set[str]:
    targets = {
        "token_embedding.weight",
        "lm_head.weight",
        "norm.weight",
    }
    for layer in range(24):
        prefix = f"blocks.{layer}"
        targets.update(
            {
                f"{prefix}.norm1.weight",
                f"{prefix}.norm2.weight",
                f"{prefix}.attention.q_proj.weight",
                f"{prefix}.attention.k_proj.weight",
                f"{prefix}.attention.v_proj.weight",
                f"{prefix}.attention.o_proj.weight",
                f"{prefix}.moe.fine_router.router.weight",
                f"{prefix}.moe.fine_experts.0.gate_proj.weight",
                f"{prefix}.moe.fine_experts.0.up_proj.weight",
                f"{prefix}.moe.fine_experts.0.down_proj.weight",
            }
        )
    missing = targets.difference(state)
    if missing:
        raise ValueError(f"exact target language anatomy missing: {missing}")
    return targets


def _map_exact_brain(
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
        raise ValueError("exact-brain ledger is not empty")

    def record(
        source: str,
        targets: tuple[str, ...],
        method: str,
        value: torch.Tensor | None = None,
    ) -> None:
        """Registra cobertura VERIFICANDO a heranca, em vez de afirma-la.

        Quando ``value`` e o tensor do doador, cada alvo e comparado byte a
        byte contra ele. Se todos batem, o registro recebe o digest da
        bijecao identidade -- copia exata e a selecao que mantem todos os
        indices. Se algum diverge, o registro nao pode reivindicar heranca e
        entra como fabricacao, com a distancia medida.
        """
        provenance_sha: str | None = None
        drift: DriftRecord | None = None
        metrics: dict[str, Any] = {}

        if value is not None:
            donor_digest = tensor_digest(value)
            inherited = tuple(
                key for key in targets
                if tensor_digest(state[key]) == donor_digest
            )
            manufactured = tuple(key for key in targets if key not in inherited)

            # Detalhe por alvo: um registro pode cobrir alvos de classes
            # diferentes -- o gate_proj de um expert e copiado do doador
            # enquanto o roteador da mesma origem e zerado. Sem isso o zero
            # some atras de um "function_preserving: True" no agregado.
            metrics["inherited_targets"] = list(inherited)
            metrics["manufactured_targets"] = list(manufactured)

            if not manufactured:
                identity = Provenance(
                    kind="select",
                    axis="full_tensor",
                    source_size=int(value.shape[0]),
                    kept=tuple(range(int(value.shape[0]))),
                )
                provenance_sha = identity.identity()
                drift = DriftRecord("l2", 0.0, measured_on="byte_identity")
            else:
                # Qualquer alvo fabricado rebaixa o registro inteiro: a
                # reivindicacao de heranca so vale se TODOS os bytes batem.
                worst = max(
                    float((state[key].float() - value.float()).norm())
                    for key in manufactured
                )
                drift = DriftRecord("l2", worst, measured_on=",".join(manufactured))

        metrics["function_preserving"] = provenance_sha is not None
        ledger.append(
            CoverageRecord(
                source_tensor=source,
                target_tensors=targets,
                method=method,
                status="complete",
                metrics=metrics,
                provenance_sha256=provenance_sha,
                drift=drift,
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
            embedding,
        )
        norm_weight = donor.get_tensor("model.norm.weight")
        _copy(state, "norm.weight", norm_weight)
        record("model.norm.weight", ("norm.weight",), "exact_copy", norm_weight)
        for layer in range(24):
            source = f"model.layers.{layer}"
            target = f"blocks.{layer}"
            mappings = {
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
                    f"{target}.moe.fine_experts.0.gate_proj.weight",
                    f"{target}.moe.fine_router.router.weight",
                ),
                f"{source}.mlp.up_proj.weight": (
                    f"{target}.moe.fine_experts.0.up_proj.weight",
                ),
                f"{source}.mlp.down_proj.weight": (
                    f"{target}.moe.fine_experts.0.down_proj.weight",
                ),
            }
            for source_key, target_keys in mappings.items():
                value = donor.get_tensor(source_key)
                for target_key in target_keys:
                    if target_key.endswith("fine_router.router.weight"):
                        _copy(state, target_key, torch.zeros_like(state[target_key]))
                    else:
                        _copy(state, target_key, value)
                record(source_key, target_keys, "exact_single_expert_copy", value)
    donor_names = {
        tensor.name for tensor in inventory_safetensor(donor_path)
    }
    ledger.assert_complete(donor_names, _language_targets(state))
    model.load_state_dict(state, strict=True)
    return ledger


@torch.no_grad()
def _neutralize_first_boot(model: DarwinXModel) -> None:
    for name, tensor in model.state_dict().items():
        if name.endswith("gaba.residual_gate"):
            tensor.zero_()
        elif name == "ttm_residual_gate":
            tensor.zero_()
        elif name == "inter_hemispheric_gate":
            tensor.zero_()
        elif ".moe.neuroendocrine.baseline_dopamine" in name:
            tensor.zero_()
        elif ".moe.neuroendocrine.dopamine" in name:
            tensor.zero_()
        elif ".moe.neuroendocrine.cortisol" in name:
            tensor.zero_()


@torch.no_grad()
def _parity_gate(
    model: DarwinXModel,
    tokenizer: SmolTokenizerAdapter,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    donor = AutoModelForCausalLM.from_pretrained(
        DEFAULT_SOURCE_ROOT,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda:1")
    donor.eval()
    model.to(dtype=torch.bfloat16)
    if not model.enable_dual_gpu(gpu0=0, gpu1=1):
        raise RuntimeError("exact-brain parity requires both GPUs")
    model.eval()
    agreement = 0
    count = 0
    kl_sum = 0.0
    max_error = 0.0
    generated: list[int] | None = None
    for prompt in CALIBRATION_PROMPTS:
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input0 = torch.tensor([ids], device="cuda:0")
        input1 = input0.to("cuda:1")
        target = model(input0, heartbeat=False).logits.float()
        teacher = donor(input1).logits.float().to("cuda:0")
        agreement += int(
            target.argmax(dim=-1).eq(teacher.argmax(dim=-1)).sum().item()
        )
        count += target.shape[0] * target.shape[1]
        kl_sum += float(
            F.kl_div(
                F.log_softmax(target, dim=-1),
                F.softmax(teacher, dim=-1),
                reduction="sum",
            ).item()
        )
        max_error = max(
            max_error,
            float((target - teacher).abs().max().item()),
        )
        if generated is None:
            sequence = input0
            for _ in range(8):
                logits = model(sequence, heartbeat=False).logits
                token = logits[:, -1].float().argmax(dim=-1, keepdim=True)
                sequence = torch.cat((sequence, token), dim=1)
            generated = sequence[0].tolist()
    report = {
        "top1_agreement": agreement / count,
        "mean_kl": kl_sum / count,
        "max_logit_error": max_error,
        "generated": tokenizer.decode(generated or []),
        "dual_gpu": True,
        "split_layer": int(model._split_layer),
    }
    if report["top1_agreement"] < 0.98 or report["mean_kl"] > 0.05:
        raise ValueError(f"exact-brain parity gate failed: {report}")
    del donor
    torch.cuda.empty_cache()
    return report


def build_exact_brain() -> dict[str, Any]:
    plan = write_exact_plan()
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("exact-brain checkpoint root is not empty")
    _, config = _config()
    sources = discover_source_identity()
    verify_sources(sources)
    model = _new_model(config)
    ledger = _map_exact_brain(
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
        projection_identity=hashlib.sha256(b"identity-2048").hexdigest(),
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
        "schema": "darwin-smol-exact-build-v3",
        "status": "exact_brain_engineering_candidate",
        "plan_id": plan["plan_id"],
        "checkpoint": str(result.checkpoint),
        "checkpoint_sha256": result.checkpoint_sha256,
        "base_checkpoint_id": result.base_checkpoint_id,
        "coverage_head": ledger.head,
        "donor_tensor_count": 218,
        "organ_report": organ_report,
        "organ_subspace_min_cosine": organ_cosine,
        "parity": parity,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    _atomic_json(RUNTIME_ROOT / "build-report.json", report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def publish_exact_candidate() -> dict[str, Any]:
    checkpoint = CHECKPOINT_ROOT / "organism_cycle_000.pt"
    manifest = CHECKPOINT_ROOT / "organism_cycle_000.manifest.json"
    manifest_payload = verify_shard_manifest(checkpoint, manifest)
    build = json.loads(
        (RUNTIME_ROOT / "build-report.json").read_text(encoding="utf-8")
    )
    gates_path = RUNTIME_ROOT / "knowledge-gates.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    native_log = (RUNTIME_ROOT / "native.stdout.log").read_text(
        encoding="utf-8"
    )
    if build.get("checkpoint_sha256") != manifest_payload.get(
        "checkpoint_sha256"
    ):
        raise ValueError("build report and checkpoint manifest disagree")
    if not gates.get("passed") or gates.get("failures"):
        raise ValueError("knowledge gates did not pass")
    if "donor_loaded=false" not in native_log:
        raise ValueError("donor-free native proof is absent")
    candidate = {
        "schema": "darwin-smol-exact-candidate-v3",
        "status": "approved_candidate",
        "checkpoint": checkpoint.name,
        "checkpoint_manifest": manifest.name,
        "checkpoint_sha256": manifest_payload["checkpoint_sha256"],
        "base_checkpoint_id": manifest_payload["base_checkpoint_id"],
        "plan_id": build["plan_id"],
        "knowledge_gates_sha256": sha256_file(gates_path),
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
            raise FileExistsError("refusing to replace approved candidate")
        return candidate
    _atomic_json(destination, candidate)
    return candidate
