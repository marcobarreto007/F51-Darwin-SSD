from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.state_identity import backbone_identity

from .checkpoint import verify_shard_manifest
from .sources import sha256_file
from .tokenizer import SmolTokenizerAdapter
from .verification import KnowledgeMetrics, evaluate_knowledge_gates


TRAIN_TEXTS = (
    "O futuro da inteligencia artificial depende de sistemas confiaveis.",
    "A agua ferve quando sua pressao de vapor iguala a pressao externa.",
    "If every bird has wings and a robin is a bird, the robin has wings.",
    "Para resolver 3x mais 7 igual a 22, subtraia 7 e divida por 3.",
    "Uma funcao Python pode inverter uma lista usando slicing.",
    "Causalidade exige uma intervencao ou um modelo alem da correlacao.",
    "The capital of France is Paris and the capital of Canada is Ottawa.",
    "Machine learning models estimate patterns from observed data.",
)
HOLDOUT_TEXTS = (
    "Explique em uma frase por que o ceu parece azul.",
    "Solve 5x minus 10 equals 20.",
    "Write a short Python loop that sums integers.",
    "A scientific hypothesis must be testable because",
)
ORGAN_MARKERS = (
    "heartbeat",
    "jepa_predictor",
    "_spider_sense_module",
    "inter_hemispheric",
    "neuroendocrine",
    "ttm_residual_gate",
)

@dataclass(frozen=True)
class CalibrationManifest:
    seed: int
    plan_id: str
    base_checkpoint_id: str
    holdout_sha256: str
    tokenizer_id: str
    trained_steps: int
    status: str = "calibrated_unpublished"


@dataclass(frozen=True)
class CalibrationResult:
    seed: int
    metrics: KnowledgeMetrics
    checkpoint: str
    manifest: str
    mean_train_loss: float
    split_layer: int


@dataclass(frozen=True)
class ExactEvaluationResult:
    metrics: tuple[KnowledgeMetrics, ...]
    candidate_bpb: float
    candidate_kl: float
    candidate_top1: float
    generation_valid: bool


def write_calibration_manifest(
    manifest: CalibrationManifest,
    directory: str | Path,
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"seed-{manifest.seed}.manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != asdict(manifest):
            raise FileExistsError(
                f"refusing to replace calibration lineage: {destination}"
            )
        return destination
    temporary = destination.with_suffix(".json.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(manifest), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def assert_explicit_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if normalized != (17, 29, 43):
        raise ValueError("calibration requires exact isolated seeds 17,29,43")
    return normalized


def _new_model(config: DarwinXConfig) -> DarwinXModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return DarwinXModel(config)
    finally:
        torch.set_default_dtype(previous)


def _place_dual(model: DarwinXModel) -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("calibration requires the two approved local GPUs")
    model.to(dtype=torch.bfloat16)
    if not model.enable_dual_gpu(gpu0=0, gpu1=1):
        raise RuntimeError("Darwin refused dual-GPU placement")
    return int(model._split_layer)


def _tokens(
    tokenizer: SmolTokenizerAdapter,
    text: str,
    *,
    device: torch.device,
    max_length: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)[:max_length]
    if len(ids) < 3:
        raise ValueError("calibration sample is too short")
    tensor = torch.tensor(ids, dtype=torch.long, device=device)
    return tensor[:-1].unsqueeze(0), tensor[1:].unsqueeze(0)


@torch.no_grad()
def _teacher_logits(donor, input_ids: torch.Tensor) -> torch.Tensor:
    return donor(input_ids.to("cuda:1")).logits.float().to("cuda:0")


def _loss(
    target_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    ce = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        labels.reshape(-1),
    )
    kl = F.kl_div(
        F.log_softmax(target_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    ) / target_logits.shape[1]
    return 0.35 * ce + 0.65 * kl


@torch.no_grad()
def _evaluate(
    model: DarwinXModel,
    donor,
    tokenizer: SmolTokenizerAdapter,
) -> tuple[float, float, float, bool]:
    model.eval()
    nll_sum = 0.0
    byte_count = 0
    kl_sum = 0.0
    token_count = 0
    agreement = 0
    for text in HOLDOUT_TEXTS:
        inputs, labels = _tokens(tokenizer, text, device=torch.device("cuda:0"))
        target = model(inputs, heartbeat=False).logits.float()
        teacher = _teacher_logits(donor, inputs)
        nll_sum += float(
            F.cross_entropy(
                target.reshape(-1, target.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            ).item()
        )
        kl_sum += float(
            F.kl_div(
                F.log_softmax(target, dim=-1),
                F.softmax(teacher, dim=-1),
                reduction="sum",
            ).item()
        )
        agreement += int(
            target.argmax(dim=-1).eq(teacher.argmax(dim=-1)).sum().item()
        )
        token_count += labels.numel()
        byte_count += len(text.encode("utf-8"))
    bpb = nll_sum / (math.log(2.0) * byte_count)
    kl = kl_sum / token_count
    top1 = agreement / token_count
    prompt, _ = _tokens(
        tokenizer,
        HOLDOUT_TEXTS[0],
        device=torch.device("cuda:0"),
    )
    generated = prompt
    for _ in range(8):
        logits = model(generated, heartbeat=False).logits
        token = logits[:, -1].float().argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, token), dim=1)
    tail = generated[0, -8:].tolist()
    generation_valid = len(set(tail)) >= 3 and all(
        0 <= token < tokenizer.vocab_size for token in tail
    )
    return bpb, kl, top1, generation_valid


def _freeze_organs(model: DarwinXModel) -> list[torch.nn.Parameter]:
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not any(marker in name for marker in ORGAN_MARKERS))
        if parameter.requires_grad:
            trainable.append(parameter)
    return trainable


def _clip_by_device(parameters: Iterable[torch.nn.Parameter]) -> None:
    groups: dict[torch.device, list[torch.nn.Parameter]] = {}
    for parameter in parameters:
        if parameter.grad is not None:
            groups.setdefault(parameter.device, []).append(parameter)
    for rows in groups.values():
        torch.nn.utils.clip_grad_norm_(rows, 1.0, foreach=False)


def _write_seed_checkpoint(
    *,
    model: DarwinXModel,
    source_payload: dict,
    config: DarwinXConfig,
    seed: int,
    steps: int,
    metrics: KnowledgeMetrics,
    root: Path,
) -> tuple[Path, Path]:
    seed_root = root / f"seed-{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_root / "calibrated.pt"
    manifest = seed_root / "calibrated.manifest.json"
    if checkpoint.exists() or manifest.exists():
        raise FileExistsError(f"calibration seed already exists: {seed_root}")
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    payload = dict(source_payload)
    payload["model_state_dict"] = state
    payload["base_checkpoint_id"] = backbone_identity(state, config)
    payload["transplant"] = {
        **dict(source_payload["transplant"]),
        "status": "calibrated_unpublished",
        "calibration_seed": seed,
        "calibration_steps": steps,
        "knowledge_metrics": asdict(metrics),
    }
    temporary = checkpoint.with_suffix(".pt.incomplete")
    torch.save(payload, temporary)
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    digest = sha256_file(temporary)
    os.replace(temporary, checkpoint)
    manifest_payload = {
        "schema": "darwin-smol-calibrated-checkpoint-v1",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "base_checkpoint_id": payload["base_checkpoint_id"],
        "seed": seed,
        "steps": steps,
        "status": "calibrated_unpublished",
        "metrics": asdict(metrics),
    }
    temporary_manifest = manifest.with_suffix(".json.incomplete")
    temporary_manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest)
    return checkpoint, manifest


def calibrate(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    tokenizer_root: str | Path,
    donor_root: str | Path,
    seed: int,
    steps: int,
    output_root: str | Path,
) -> CalibrationResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    verify_shard_manifest(checkpoint, manifest)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = DarwinXConfig.from_mapping(payload["config"])
    tokenizer = SmolTokenizerAdapter.load(tokenizer_root)
    from transformers import AutoModelForCausalLM

    donor = AutoModelForCausalLM.from_pretrained(
        donor_root,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda:1")
    donor.eval()
    for parameter in donor.parameters():
        parameter.requires_grad_(False)

    random_model = _new_model(config)
    _place_dual(random_model)
    random_bpb, random_kl, random_top1, _ = _evaluate(
        random_model,
        donor,
        tokenizer,
    )
    del random_model
    torch.cuda.empty_cache()

    model = _new_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    split_layer = _place_dual(model)
    trainable = _freeze_organs(model)
    optimizer = torch.optim.SGD(trainable, lr=2e-5)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        text = TRAIN_TEXTS[(step + seed) % len(TRAIN_TEXTS)]
        inputs, labels = _tokens(
            tokenizer,
            text,
            device=torch.device("cuda:0"),
        )
        with torch.no_grad():
            teacher = _teacher_logits(donor, inputs)
        optimizer.zero_grad(set_to_none=True)
        target = model(inputs, heartbeat=False).logits.float()
        loss = _loss(target, teacher, labels)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite calibration loss at step {step}")
        loss.backward()
        _clip_by_device(trainable)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    bpb, kl, top1, generation_valid = _evaluate(model, donor, tokenizer)
    metrics = KnowledgeMetrics(
        seed=seed,
        bpb=bpb,
        random_bpb=random_bpb,
        kl=kl,
        random_kl=random_kl,
        top1=top1,
        random_top1=random_top1,
        generation_valid=generation_valid,
    )
    seed_checkpoint, seed_manifest = _write_seed_checkpoint(
        model=model,
        source_payload=payload,
        config=config,
        seed=seed,
        steps=steps,
        metrics=metrics,
        root=Path(output_root),
    )
    del model, donor
    torch.cuda.empty_cache()
    return CalibrationResult(
        seed=seed,
        metrics=metrics,
        checkpoint=str(seed_checkpoint),
        manifest=str(seed_manifest),
        mean_train_loss=sum(losses) / len(losses),
        split_layer=split_layer,
    )


def evaluate_exact_candidate(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    tokenizer_root: str | Path,
    donor_root: str | Path,
    seeds: Iterable[int] = (17, 29, 43),
) -> ExactEvaluationResult:
    selected_seeds = assert_explicit_seeds(seeds)
    verify_shard_manifest(checkpoint, manifest)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = DarwinXConfig.from_mapping(payload["config"])
    tokenizer = SmolTokenizerAdapter.load(tokenizer_root)
    from transformers import AutoModelForCausalLM

    donor = AutoModelForCausalLM.from_pretrained(
        donor_root,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda:1")
    donor.eval()
    candidate = _new_model(config)
    candidate.load_state_dict(payload["model_state_dict"], strict=True)
    _place_dual(candidate)
    candidate_bpb, candidate_kl, candidate_top1, generation_valid = _evaluate(
        candidate,
        donor,
        tokenizer,
    )
    rows: list[KnowledgeMetrics] = []
    for seed in selected_seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random_model = _new_model(config)
        _place_dual(random_model)
        random_bpb, random_kl, random_top1, _ = _evaluate(
            random_model,
            donor,
            tokenizer,
        )
        rows.append(
            KnowledgeMetrics(
                seed=seed,
                bpb=candidate_bpb,
                random_bpb=random_bpb,
                kl=candidate_kl,
                random_kl=random_kl,
                top1=candidate_top1,
                random_top1=random_top1,
                generation_valid=generation_valid,
            )
        )
        del random_model
        torch.cuda.empty_cache()
    del candidate, donor
    torch.cuda.empty_cache()
    return ExactEvaluationResult(
        metrics=tuple(rows),
        candidate_bpb=candidate_bpb,
        candidate_kl=candidate_kl,
        candidate_top1=candidate_top1,
        generation_valid=generation_valid,
    )
