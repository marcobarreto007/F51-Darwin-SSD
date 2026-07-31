from __future__ import annotations

import copy
import gc
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from safetensors import safe_open

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.organism.checkpoint_root import model_config_identity

from .checkpoint import (
    TransplantCheckpointMetadata,
    TransplantCheckpointWriter,
    verify_shard_manifest,
)
from .contracts import CALIBRATION_PROMPTS, TransplantPlan
from .layers import (
    fold_output_projection,
    select_and_fold_heads,
)
from .ledger import CoverageLedger, CoverageRecord
from .moe import CoupledMLP, cluster_coupled_neurons, fit_router
from .organs import (
    OrganProjection,
    derive_organ_projection,
    expand_square,
    verify_organ_subspace,
)
from .projection import (
    HiddenProjection,
    derive_hidden_projection,
    project_norm,
)
from .sources import inventory_safetensor, sha256_file, verify_sources
from .ssd_fit import SSDFitConfig, TransitionBatch, fit_ssd_block
from .tokenizer import SmolTokenizerAdapter


ROOT = Path(__file__).resolve().parents[3]
TARGET_CONFIG = ROOT / "src/configs/darwin_x_1.6b_smol_transplant.yaml"
LANGUAGE_PREFIXES = (
    "token_embedding.",
    "blocks.",
    "norm.",
    "lm_head.",
)
ORGAN_MARKERS = (
    ".gaba.",
    ".moe.neuroendocrine.",
    "inter_hemispheric.",
    "inter_hemispheric_gate",
    "mtp_heads.",
    "jepa_predictor.",
    "_spider_sense_module.",
    "heartbeat.",
    "ttm_residual_gate",
)


@dataclass(frozen=True)
class DonorCapture:
    hidden_states: tuple[torch.Tensor, ...]
    attention_inputs: tuple[torch.Tensor, ...]
    attention_outputs: tuple[torch.Tensor, ...]
    head_outputs: tuple[torch.Tensor, ...]
    mlp_inputs: tuple[torch.Tensor, ...]
    mlp_activations: tuple[torch.Tensor, ...]
    embedding_samples: torch.Tensor


@dataclass(frozen=True)
class NativeBuildResult:
    checkpoint: str
    manifest: str
    status: str
    coverage_head: str
    projection_identity: str
    organ_projection_identity: str


def _target_config(plan: TransplantPlan) -> DarwinXConfig:
    raw = yaml.safe_load(TARGET_CONFIG.read_text(encoding="utf-8"))
    if model_config_identity(raw) != plan.target.config_identity:
        raise ValueError("target config identity differs from transplant plan")
    config = DarwinXConfig.from_mapping(raw)
    if config.model_name != plan.target.model_name:
        raise ValueError("target model name differs from transplant plan")
    _assert_donor_runtime_contract(config)
    return config


def _assert_donor_runtime_contract(config: DarwinXConfig) -> None:
    """Falha fechado se o runtime alvo divergir do doador Llama-family.

    Espelha o contrato que exact_assembly.py:70-72 ja exigia da linhagem
    densa. Nenhum desses campos aparece em shape de tensor, cobertura do
    ledger ou hash de checkpoint, entao um build inteiro passa por eles em
    silencio -- foi exatamente o que aconteceu em 2026-07-28.
    """
    violations = []
    if config.rope_style != "llama":
        violations.append(
            f"rope_style={config.rope_style!r} (doador exige 'llama': "
            "split-half, nao a convencao GPT-J interleaved)"
        )
    if not math.isclose(config.residual_scale, 1.0):
        violations.append(
            f"residual_scale={config.residual_scale} (transplante zero-shot "
            "exige 1.0; block.py:66,84 escalam cada sub-camada por ele)"
        )
    if config.rms_norm_eps != 1e-5:
        violations.append(
            f"rms_norm_eps={config.rms_norm_eps} (config.json do doador: 1e-5)"
        )
    if not config.rms_norm_fp32:
        violations.append(
            "rms_norm_fp32=False (modelo e construido em bf16; sem upcast a "
            "variancia do RMSNorm e calculada com 7 bits de mantissa)"
        )
    if violations:
        raise ValueError(
            "target config violates the donor runtime contract: "
            + "; ".join(violations)
        )


def _device(index: int) -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > index:
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _donor_snapshot_root(plan: TransplantPlan) -> Path:
    weights = Path(plan.sources.smol_weights.path)
    if weights.name == "model.safetensors":
        return weights.parent
    candidate = (
        weights.parent.parent
        / "snapshots"
        / plan.sources.smol_snapshot
    )
    if not (candidate / "model.safetensors").is_file():
        raise ValueError(
            "cannot recover immutable donor snapshot from resolved blob path: "
            f"{weights}"
        )
    return candidate


def _capture_donor(plan: TransplantPlan) -> DonorCapture:
    from transformers import AutoModelForCausalLM

    source_root = _donor_snapshot_root(plan)
    tokenizer = SmolTokenizerAdapter.load(
        source_root,
        expected_hashes={
            "tokenizer.json": plan.sources.tokenizer_json.sha256,
            "tokenizer_config.json": plan.sources.tokenizer_config.sha256,
            "special_tokens_map.json": plan.sources.special_tokens.sha256,
        },
    )
    encoded = [
        tokenizer.encode(prompt, add_bos=True, add_eos=True)
        for prompt in CALIBRATION_PROMPTS
    ]
    sequence_length = max(len(row) for row in encoded)
    padded = [
        row + [tokenizer.pad_id] * (sequence_length - len(row))
        for row in encoded
    ]
    masks = [
        [1] * len(row) + [0] * (sequence_length - len(row))
        for row in encoded
    ]
    donor_device = _device(1)
    model = AutoModelForCausalLM.from_pretrained(
        source_root,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to(donor_device)
    model.eval()
    layers = model.model.layers
    head_outputs: list[torch.Tensor | None] = [None] * len(layers)
    attention_inputs: list[torch.Tensor | None] = [None] * len(layers)
    attention_outputs: list[torch.Tensor | None] = [None] * len(layers)
    mlp_inputs: list[torch.Tensor | None] = [None] * len(layers)
    mlp_activations: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def pre_store(storage: list, index: int):
        def hook(_module, arguments):
            storage[index] = arguments[0].detach().to("cpu")

        return hook

    def output_store(storage: list, index: int):
        def hook(_module, _arguments, output):
            tensor = output[0] if isinstance(output, tuple) else output
            storage[index] = tensor.detach().to("cpu")

        return hook

    for index, layer in enumerate(layers):
        handles.append(
            layer.self_attn.q_proj.register_forward_pre_hook(
                pre_store(attention_inputs, index)
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                pre_store(head_outputs, index)
            )
        )
        handles.append(
            layer.self_attn.register_forward_hook(
                output_store(attention_outputs, index)
            )
        )
        handles.append(
            layer.mlp.gate_proj.register_forward_pre_hook(
                pre_store(mlp_inputs, index)
            )
        )
        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                pre_store(mlp_activations, index)
            )
        )

    input_ids = torch.tensor(padded, device=donor_device, dtype=torch.long)
    attention_mask = torch.tensor(
        masks,
        device=donor_device,
        dtype=torch.long,
    )
    with torch.inference_mode():
        output = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    for handle in handles:
        handle.remove()
    hidden_states = tuple(
        hidden.detach().to("cpu") for hidden in output.hidden_states
    )
    embedding = model.model.embed_tokens.weight.detach().to("cpu")
    sample_indices = torch.linspace(
        0,
        embedding.shape[0] - 1,
        steps=4096,
    ).round().long()
    embedding_samples = embedding[sample_indices].contiguous()
    del output, model, input_ids, attention_mask
    gc.collect()
    if donor_device.type == "cuda":
        torch.cuda.empty_cache()

    captures = (
        attention_inputs,
        attention_outputs,
        head_outputs,
        mlp_inputs,
        mlp_activations,
    )
    if any(any(item is None for item in storage) for storage in captures):
        raise RuntimeError("donor activation hooks did not cover every layer")
    return DonorCapture(
        hidden_states=hidden_states,
        attention_inputs=tuple(attention_inputs),  # type: ignore[arg-type]
        attention_outputs=tuple(attention_outputs),  # type: ignore[arg-type]
        head_outputs=tuple(head_outputs),  # type: ignore[arg-type]
        mlp_inputs=tuple(mlp_inputs),  # type: ignore[arg-type]
        mlp_activations=tuple(mlp_activations),  # type: ignore[arg-type]
        embedding_samples=embedding_samples,
    )


def _derive_projection(
    capture: DonorCapture,
    *,
    target_dim: int,
    seed: int,
) -> HiddenProjection:
    activation_samples = torch.cat(
        [hidden.reshape(-1, hidden.shape[-1]) for hidden in capture.hidden_states],
        dim=0,
    )
    projection_device = _device(0)
    projection = derive_hidden_projection(
        capture.embedding_samples.to(projection_device),
        activation_samples.to(projection_device),
        target_dim=target_dim,
        seed=seed,
    )
    return HiddenProjection(
        matrix=projection.matrix.cpu(),
        singular_values=projection.singular_values.cpu(),
        explained_energy=projection.explained_energy,
        calibration_digest=projection.calibration_digest,
        seed=projection.seed,
        algorithm=projection.algorithm,
        output_hash=projection.output_hash,
    )


def _new_target_model(config: DarwinXConfig) -> DarwinXModel:
    previous_dtype = torch.get_default_dtype()
    torch.manual_seed(20_260_728)
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = DarwinXModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()
    return model


@torch.no_grad()
def _copy_state(
    state: Mapping[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> None:
    target = state[key]
    if tuple(target.shape) != tuple(value.shape):
        raise ValueError(
            f"target shape mismatch for {key}: "
            f"expected={tuple(target.shape)} actual={tuple(value.shape)}"
        )
    target.copy_(value.to(device=target.device, dtype=target.dtype))


def _project_matrix(
    weight: torch.Tensor,
    projection: HiddenProjection,
) -> torch.Tensor:
    compute_device = _device(0)
    p = projection.matrix.to(compute_device)
    result = p @ weight.float().to(compute_device) @ p.T
    return result.to("cpu")


def _project_rows(
    weight: torch.Tensor,
    projection: HiddenProjection,
) -> torch.Tensor:
    compute_device = _device(0)
    return (
        weight.float().to(compute_device)
        @ projection.matrix.to(compute_device).T
    ).to("cpu")


def _project_columns(
    weight: torch.Tensor,
    projection: HiddenProjection,
) -> torch.Tensor:
    compute_device = _device(0)
    return (
        projection.matrix.to(compute_device)
        @ weight.float().to(compute_device)
    ).to("cpu")


def _average_tensors(
    donor,
    names: Iterable[str],
) -> torch.Tensor:
    tensors = [donor.get_tensor(name).float() for name in names]
    return torch.stack(tensors).mean(dim=0)


def _fast_coupled_expert(
    mlp: CoupledMLP,
    signatures: torch.Tensor,
    source_neurons: tuple[int, ...],
    *,
    target_width: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    indices = torch.tensor(source_neurons, dtype=torch.long)
    gate = mlp.gate[indices].float()
    up = mlp.up[indices].float()
    down = mlp.down[:, indices].float()
    signature = signatures[:, indices].float().T
    source_width = len(source_neurons)
    basis = torch.zeros(target_width, source_width, dtype=torch.float32)
    if source_width <= target_width:
        # OBSERVACAO (2026-07-29): na expansao o basis precisa ser scatter,
        # nao um frame ortonormal denso.
        #
        # A simetria dos neuronios ocultos de um SwiGLU e o grupo de
        # PERMUTACAO, nao o grupo ortogonal: silu e coordenada-a-coordenada e
        # o produto e elementwise, entao
        #     B @ (silu(a) * b) == silu(B @ a) * (B @ b)
        # vale para B permutacao/scatter e FALHA para B ortonormal denso. Todo
        # B de colunas ortonormais zera o erro de reconstrucao do PESO
        # (B.T @ B == I); so o scatter preserva a FUNCAO.
        #
        # A versao anterior replicava cada neuronio-fonte em floor(tw/sw)
        # linhas escaladas por counts.rsqrt(). As partes lineares cancelam,
        # mas sobra silu(s*g) no lugar de silu(g), com s = 1/sqrt(m). Medido
        # no caso de producao (k=512 -> 896): reconstrucao de peso 3.1e-17 e
        # erro FUNCIONAL relativo 0.295 -- ~29% de atenuacao em 75% dos
        # neuronios, invisivel porque reconstruction_error era fixado em 0.0.
        #
        # As linhas excedentes recebem gate/up pequeno e deterministico, mas
        # down exatamente zero. Assim a funcao permanece identica no instante
        # zero, enquanto o primeiro backward gera gradiente em down e permite
        # que a capacidade nova acorde nos passos seguintes.
        rows = torch.arange(source_width)
        basis[rows, rows] = 1.0
        reconstruction_error = 0.0
    else:
        contribution = (
            signature.square().mean(dim=1)
            + gate.square().mean(dim=1)
            + up.square().mean(dim=1)
            + down.T.square().mean(dim=1)
        )
        selected = torch.topk(
            contribution,
            k=target_width,
            largest=True,
            sorted=True,
        ).indices
        selected_set = set(selected.tolist())
        for row, column in enumerate(selected.tolist()):
            basis[row, column] = 1.0
        rejected = [
            column for column in range(source_width)
            if column not in selected_set
        ]
        if rejected:
            selected_signature = torch.nn.functional.normalize(
                signature[selected],
                dim=1,
                eps=1e-12,
            )
            rejected_signature = torch.nn.functional.normalize(
                signature[rejected],
                dim=1,
                eps=1e-12,
            )
            nearest = (
                rejected_signature @ selected_signature.T
            ).abs().argmax(dim=1)
            for rejected_column, selected_row in zip(
                rejected,
                nearest.tolist(),
                strict=True,
            ):
                source = signature[selected[selected_row]]
                residual = signature[rejected_column]
                coefficient = torch.dot(source, residual) / torch.dot(
                    source,
                    source,
                ).clamp_min(1e-12)
                basis[selected_row, rejected_column] = coefficient
        reconstructed = basis.T @ (basis @ signature)
        reconstruction_error = float(
            (reconstructed - signature).square().mean().item()
        )
    compute_device = _device(0)
    basis_device = basis.to(compute_device)
    target_gate = basis_device @ gate.to(compute_device)
    target_up = basis_device @ up.to(compute_device)
    target_down = down.to(compute_device) @ basis_device.T
    if source_width < target_width:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        extra = target_width - source_width
        target_gate[source_width:] = (
            torch.randn(
                extra,
                mlp.d_model,
                generator=generator,
            ).to(compute_device)
            * 1e-4
        )
        target_up[source_width:] = (
            torch.randn(
                extra,
                mlp.d_model,
                generator=generator,
            ).to(compute_device)
            * 1e-4
        )
        target_down[:, source_width:] = 0.0
    return (
        target_gate.cpu(),
        target_up.cpu(),
        target_down.cpu(),
        reconstruction_error,
    )


def _language_target_keys(state: Mapping[str, torch.Tensor]) -> set[str]:
    keys: set[str] = set()
    for key in state:
        if key in {"token_embedding.weight", "lm_head.weight", "norm.weight"}:
            keys.add(key)
            continue
        if not key.startswith("blocks."):
            continue
        if any(marker in key for marker in ORGAN_MARKERS):
            continue
        if ".moe." in key:
            if (
                ".fine_router.router.weight" in key
                or ".fine_experts." in key
                or ".shared_experts." in key
            ):
                keys.add(key)
            continue
        if ".norm" in key or ".attention." in key or ".ssd." in key:
            keys.add(key)
    return keys


def _initialize_ssd(
    block,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
) -> tuple[str, ...]:
    state = block.ssd.state_dict()
    combined = torch.cat((q, k, v, (q + k + v) / 3.0), dim=0)
    _copy_state(state, "ssm.in_proj.weight", combined)
    _copy_state(state, "ssm.out_proj.weight", torch.cat((o, o), dim=1) / 2.0)
    inner = q.shape[0] * 2
    channel_scale = torch.cat(
        (
            q.square().mean(dim=1).sqrt(),
            v.square().mean(dim=1).sqrt(),
        )
    )
    channel_scale = channel_scale / channel_scale.mean().clamp_min(1e-8)
    decay = torch.log(
        torch.arange(1, state["ssm.a_log"].shape[1] + 1).float()
    )
    _copy_state(
        state,
        "ssm.a_log",
        decay[None, :] + channel_scale[:, None].log().clamp(-2, 2) * 0.01,
    )
    _copy_state(state, "ssm.d_skip", channel_scale)
    kernel = torch.tensor([0.4, 0.3, 0.2, 0.1])
    _copy_state(
        state,
        "ssm.conv1d.weight",
        (channel_scale[:, None, None] * kernel[None, None, :]),
    )
    # conv1d.bias recebia channel_scale.log().clamp(-4,4) -- a MESMA formula
    # de dt_proj.bias, apesar dos dois parametros terem papeis matematicos
    # distintos. Confirmado no artefato: os dois tensores eram byte-identicos
    # em 12/12 blocos do checkpoint de 2026-07-28. Bias de conv e zero.
    _copy_state(
        state,
        "ssm.conv1d.bias",
        torch.zeros_like(channel_scale),
    )
    in_proj = combined
    x_proj = (
        in_proj[: state["ssm.x_proj.weight"].shape[0]]
        @ state["ssm.out_proj.weight"].float()
    )
    _copy_state(state, "ssm.x_proj.weight", x_proj)
    dt_rank = state["ssm.dt_proj.weight"].shape[1]
    _copy_state(
        state,
        "ssm.dt_proj.weight",
        in_proj[:inner, :dt_rank],
    )
    # OBSERVACAO (2026-07-29): dt_proj.bias precisa sair do inverse-softplus
    # de dt em [dt_min, dt_max], nao de uma heuristica de escala de canal.
    #
    # dt e a constante de TEMPO da discretizacao do SSM (A_bar = exp(dt*A)),
    # nao um fator de escala -- o RMS por canal das matrizes Q/V do doador
    # nao e uma escala de tempo, e usar log(channel_scale) com channel_scale
    # normalizado para media 1 (linha 513) colocava a maioria dos canais em
    # bias ~0, isto e softplus(0) = 0.6931.
    #
    # Medido no checkpoint de 2026-07-28: 100.0% dos 46.080 canais dos 12
    # blocos SSD com dt acima do teto canonico 0.1, dt mediano global 0.6650
    # -- 6.7x o teto, e os 20 passos de fit nao resgataram (a medicao e do
    # checkpoint final). dt grande faz A_bar decair rapido demais: os blocos
    # esquecem o estado quase por completo a cada passo, e 75% da
    # profundidade do modelo perde a capacidade de integrar contexto.
    #
    # ssm_core.py:289-294 ja define a spec canonica (dt_min=0.001,
    # dt_max=0.1, oficial Mamba) e model.py:613-622 protege dt_proj com
    # _no_reinit justamente por causa do bug V9 de 2026-07-26. A cirurgia
    # escapava dessa guarda porque sobrescreve o state_dict direto, sem
    # passar por _init_weights -- mesma familia de bug, rota diferente.
    #
    # Aqui a varredura e log-uniforme deterministica em vez do sorteio de
    # ssm_core.py: mesma distribuicao marginal, sem RNG, porque o plano de
    # transplante e amarrado por hash.
    dt_min, dt_max = 0.001, 0.1
    spread = torch.linspace(0.0, 1.0, state["ssm.dt_proj.bias"].shape[0])
    dt = torch.exp(
        math.log(dt_min) + spread * (math.log(dt_max) - math.log(dt_min))
    )
    _copy_state(
        state,
        "ssm.dt_proj.bias",
        dt + torch.log(-torch.expm1(-dt)),
    )
    block.ssd.load_state_dict(state, strict=True)
    return tuple(f"blocks.{{block}}.ssd.{key}" for key in state)


def _resize_donor_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    projection: OrganProjection,
    identity_complement: bool = False,
) -> torch.Tensor:
    if source.shape == target.shape:
        return source.detach().clone().to(dtype=target.dtype)
    if source.ndim == 0 and target.ndim == 0:
        return source.detach().clone().to(dtype=target.dtype)
    if (
        source.ndim == 2
        and source.shape
        == (projection.source_dim, projection.source_dim)
        and target.shape
        == (projection.target_dim, projection.target_dim)
    ):
        return expand_square(
            source,
            projection,
            complement="identity" if identity_complement else "zero",
        ).to(dtype=target.dtype)
    if source.ndim == 1 and target.ndim == 1:
        resized = torch.nn.functional.interpolate(
            source.float()[None, None, :],
            size=target.shape[0],
            mode="linear",
            align_corners=True,
        )[0, 0]
        return resized.to(dtype=target.dtype)
    if source.ndim == 2 and target.ndim == 2:
        resized = torch.nn.functional.interpolate(
            source.float()[None, None, :, :],
            size=target.shape,
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        return resized.to(dtype=target.dtype)
    if source.ndim == target.ndim and source.ndim >= 1:
        flat_source = source.float().reshape(1, 1, -1)
        flat_target = torch.nn.functional.interpolate(
            flat_source,
            size=target.numel(),
            mode="linear",
            align_corners=True,
        ).reshape(target.shape)
        return flat_target.to(dtype=target.dtype)
    raise ValueError(
        f"cannot grow organ tensor {tuple(source.shape)} -> {tuple(target.shape)}"
    )


def _source_organ_key(target_key: str, *, target_layers: int = 16) -> str:
    if not target_key.startswith("blocks."):
        return target_key
    parts = target_key.split(".")
    target_layer = int(parts[1])
    source_layer = min(11, target_layer * 12 // target_layers)
    parts[1] = str(source_layer)
    return ".".join(parts)


def _grow_organs(
    model: DarwinXModel,
    organ_checkpoint: Path,
    projection: OrganProjection,
) -> tuple[dict[str, Any], float]:
    payload = torch.load(
        organ_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    source_state = {
        key.replace("_orig_mod.", ""): value
        for key, value in payload["model_state_dict"].items()
    }
    target_state = model.state_dict()
    grown = 0
    subspace_cosines: list[float] = []
    for target_key, target in target_state.items():
        if not any(marker in target_key for marker in ORGAN_MARKERS):
            continue
        source_key = _source_organ_key(
            target_key,
            target_layers=model.config.n_layers,
        )
        source = source_state.get(source_key)
        if source is None:
            continue
        identity_like = any(
            word in target_key.lower()
            for word in ("transition", "recurrent", "memory")
        )
        value = _resize_donor_tensor(
            source,
            target,
            projection=projection,
            identity_complement=identity_like,
        )
        if target_key.endswith(("residual_gate", "external_gate")):
            value = torch.zeros_like(target)
        _copy_state(target_state, target_key, value)
        if (
            source.ndim == 2
            and source.shape
            == (projection.source_dim, projection.source_dim)
            and target.shape
            == (projection.target_dim, projection.target_dim)
        ):
            subspace_cosines.append(
                verify_organ_subspace(source, value.float(), projection)
            )
        grown += 1
    model.load_state_dict(target_state, strict=True)
    del payload, source_state
    gc.collect()
    minimum_cosine = min(subspace_cosines) if subspace_cosines else 1.0
    if minimum_cosine < 0.99:
        raise ValueError(
            f"grown organ subspace cosine below 0.99: {minimum_cosine}"
        )
    return {"grown_tensors": grown}, minimum_cosine


def _copy_tokenizer(plan: TransplantPlan) -> str:
    target = Path(plan.target.tokenizer_root)
    target.mkdir(parents=True, exist_ok=True)
    sources = {
        "tokenizer.json": plan.sources.tokenizer_json,
        "tokenizer_config.json": plan.sources.tokenizer_config,
        "special_tokens_map.json": plan.sources.special_tokens,
    }
    expected = {
        filename: source.sha256 for filename, source in sources.items()
    }
    for filename, source in sources.items():
        digest = source.sha256
        destination = target / filename
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError(
                    f"existing tokenizer target hash mismatch: {destination}"
                )
            continue
        temporary = destination.with_suffix(destination.suffix + ".incomplete")
        shutil.copyfile(source.path, temporary)
        if sha256_file(temporary) != digest:
            raise ValueError(f"copied tokenizer hash mismatch: {filename}")
        os.replace(temporary, destination)
    tokenizer = SmolTokenizerAdapter.load(target, expected_hashes=expected)
    from f51_darwin.state_identity import tokenizer_identity

    return tokenizer_identity(tokenizer)


def build_native_transplant(
    plan: TransplantPlan,
    *,
    resume: bool,
) -> NativeBuildResult:
    verify_sources(plan.sources)
    checkpoint_root = Path(plan.target.checkpoint_root)
    checkpoint = checkpoint_root / "organism_cycle_000.pt"
    manifest = checkpoint_root / "organism_cycle_000.manifest.json"
    if resume and checkpoint.is_file() and manifest.is_file():
        verify_shard_manifest(checkpoint, manifest)
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        transplant = payload["transplant"]
        if transplant["plan_id"] != plan.identity():
            raise ValueError("published checkpoint belongs to another plan")
        return NativeBuildResult(
            checkpoint=str(checkpoint),
            manifest=str(manifest),
            status=str(transplant["status"]),
            coverage_head=str(transplant["coverage_head"]),
            projection_identity=str(transplant["projection_identity"]),
            organ_projection_identity=str(
                transplant["organ_projection_identity"]
            ),
        )
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
        raise ValueError(
            "non-empty transplant root cannot be rebuilt without a valid "
            "published resume artifact"
        )

    config = _target_config(plan)
    capture = _capture_donor(plan)
    projection = _derive_projection(
        capture,
        target_dim=config.d_model,
        seed=plan.projection_seed,
    )
    model = _new_target_model(config)
    state = model.state_dict()
    language_targets = _language_target_keys(state)
    runtime_root = Path(plan.target.runtime_root)
    ledger = CoverageLedger.open(
        runtime_root / "coverage.jsonl",
        plan_id=plan.identity(),
    )
    if ledger.records:
        raise ValueError(
            "partial coverage ledger requires family-shard resume support"
        )
    classified: set[str] = set()

    def record(
        source_name: str,
        target_names: Iterable[str],
        method: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        targets = tuple(sorted(set(target_names)))
        ledger.append(
            CoverageRecord(
                source_tensor=source_name,
                target_tensors=targets,
                method=method,
                status="complete",
                metrics=dict(metrics or {}),
            )
        )
        classified.add(source_name)

    donor_path = Path(plan.sources.smol_weights.path)
    with safe_open(
        str(donor_path),
        framework="pt",
        device="cpu",
    ) as donor:
        embedding = donor.get_tensor("model.embed_tokens.weight")
        compute_device = _device(0)
        p = projection.matrix.to(compute_device)
        projected_parts = []
        for start in range(0, embedding.shape[0], 2048):
            projected_parts.append(
                (
                    embedding[start : start + 2048]
                    .float()
                    .to(compute_device)
                    @ p.T
                ).to("cpu", dtype=torch.bfloat16)
            )
        projected_embedding = torch.cat(projected_parts, dim=0)
        _copy_state(state, "token_embedding.weight", projected_embedding)
        _copy_state(state, "lm_head.weight", projected_embedding)
        record(
            "model.embed_tokens.weight",
            ("token_embedding.weight", "lm_head.weight"),
            "coherent_projection_tied",
            {"projection_identity": projection.identity()},
        )
        final_norm = donor.get_tensor("model.norm.weight")
        _copy_state(state, "norm.weight", project_norm(final_norm, projection))
        record(
            "model.norm.weight",
            ("norm.weight",),
            "coherent_norm_projection",
        )

        for target_layer, group in enumerate(plan.layer_groups):
            prefix = f"blocks.{target_layer}"
            input_norm_names = [
                f"model.layers.{layer}.input_layernorm.weight"
                for layer in group
            ]
            post_norm_names = [
                f"model.layers.{layer}.post_attention_layernorm.weight"
                for layer in group
            ]
            input_norm = _average_tensors(donor, input_norm_names)
            post_norm = _average_tensors(donor, post_norm_names)
            _copy_state(
                state,
                f"{prefix}.norm1.weight",
                project_norm(input_norm, projection),
            )
            _copy_state(
                state,
                f"{prefix}.norm2.weight",
                project_norm(post_norm, projection),
            )
            for name in input_norm_names:
                record(
                    name,
                    (f"{prefix}.norm1.weight",),
                    "group_average_coherent_norm",
                )
            for name in post_norm_names:
                record(
                    name,
                    (f"{prefix}.norm2.weight",),
                    "group_average_coherent_norm",
                )

            attention_names = {
                component: [
                    f"model.layers.{layer}.self_attn.{component}_proj.weight"
                    for layer in group
                ]
                for component in ("q", "k", "v", "o")
            }
            averaged = {
                component: _average_tensors(donor, names)
                for component, names in attention_names.items()
            }
            if target_layer in plan.attention_blocks:
                group_heads = torch.cat(
                    [
                        capture.head_outputs[layer].reshape(
                            -1,
                            32,
                            64,
                        )
                        for layer in group
                    ],
                    dim=0,
                )
                mapping = select_and_fold_heads(
                    group_heads,
                    target_heads=30,
                )
                row_indices = torch.cat(
                    [
                        torch.arange(head * 64, (head + 1) * 64)
                        for head in mapping.selected_heads
                    ]
                )
                target_keys = []
                for component in ("q", "k", "v"):
                    selected = averaged[component][row_indices]
                    value = _project_rows(selected, projection)
                    key = f"{prefix}.attention.{component}_proj.weight"
                    _copy_state(state, key, value)
                    target_keys.append(key)
                    for source_name in attention_names[component]:
                        record(
                            source_name,
                            (key,),
                            "activation_ranked_head_projection",
                            {
                                "selected_heads": list(mapping.selected_heads),
                                "folded_heads": list(mapping.folded_heads),
                            },
                        )
                folded_o = fold_output_projection(
                    averaged["o"],
                    mapping,
                    head_dim=64,
                )
                o_value = _project_columns(folded_o, projection)
                o_key = f"{prefix}.attention.o_proj.weight"
                _copy_state(state, o_key, o_value)
                target_keys.append(o_key)
                for source_name in attention_names["o"]:
                    record(
                        source_name,
                        (o_key,),
                        "least_squares_head_fold_projection",
                    )
            else:
                q = _project_matrix(averaged["q"], projection)
                k = _project_matrix(averaged["k"], projection)
                v = _project_matrix(averaged["v"], projection)
                o = _project_matrix(averaged["o"], projection)
                block = model.blocks[target_layer]
                # Controle pareado real: o bloco recem-construido por
                # DarwinXModel(config) carrega o init aleatorio canonico
                # (dt via _no_reinit, ssm_core.py:289-301). Copiado ANTES de
                # _initialize_ssd, que muta block.ssd in-place.
                random_baseline = copy.deepcopy(block.ssd)
                _initialize_ssd(block, q, k, v, o)
                endpoint = group[-1]
                transition_inputs = _project_rows(
                    capture.attention_inputs[endpoint].reshape(-1, 2048),
                    projection,
                ).reshape(
                    capture.attention_inputs[endpoint].shape[0],
                    capture.attention_inputs[endpoint].shape[1],
                    config.d_model,
                )
                transition_outputs = _project_rows(
                    capture.attention_outputs[endpoint].reshape(-1, 2048),
                    projection,
                ).reshape(
                    capture.attention_outputs[endpoint].shape[0],
                    capture.attention_outputs[endpoint].shape[1],
                    config.d_model,
                )
                fit_device = _device(0)
                block.ssd.to(fit_device)
                random_baseline.to(fit_device)
                report = fit_ssd_block(
                    block.ssd,
                    TransitionBatch(
                        transition_inputs[:4].to(fit_device),
                        transition_outputs[:4].to(fit_device),
                        transition_inputs[4:].to(fit_device),
                        transition_outputs[4:].to(fit_device),
                    ),
                    SSDFitConfig(
                        steps=20,
                        learning_rate=2e-3,
                        seed=plan.projection_seed + target_layer,
                    ),
                    baseline=random_baseline,
                )
                del random_baseline
                block.ssd.to("cpu", dtype=torch.bfloat16)
                ssd_keys = tuple(
                    f"{prefix}.ssd.{key}"
                    for key in block.ssd.state_dict()
                )
                for names in attention_names.values():
                    for source_name in names:
                        record(
                            source_name,
                            ssd_keys,
                            "donor_transition_ssd_identification",
                            {
                                "holdout_mse": report.holdout_mse,
                                "random_holdout_mse": report.random_holdout_mse,
                            },
                        )

            gate_names = [
                f"model.layers.{layer}.mlp.gate_proj.weight"
                for layer in group
            ]
            up_names = [
                f"model.layers.{layer}.mlp.up_proj.weight"
                for layer in group
            ]
            down_names = [
                f"model.layers.{layer}.mlp.down_proj.weight"
                for layer in group
            ]
            gates = [
                _project_rows(donor.get_tensor(name), projection)
                for name in gate_names
            ]
            ups = [
                _project_rows(donor.get_tensor(name), projection)
                for name in up_names
            ]
            downs = [
                _project_columns(donor.get_tensor(name), projection)
                for name in down_names
            ]
            coupled = CoupledMLP(
                gate=torch.cat(gates, dim=0),
                up=torch.cat(ups, dim=0),
                down=torch.cat(downs, dim=1),
            )
            signatures = torch.cat(
                [
                    capture.mlp_activations[layer].reshape(-1, 8192)
                    for layer in group
                ],
                dim=1,
            ).float()
            assignment = cluster_coupled_neurons(
                coupled,
                signatures,
                shared_experts=2,
                fine_experts=14,
                expert_width=896,
                seed=plan.projection_seed + target_layer,
            )
            gate_targets: list[str] = []
            up_targets: list[str] = []
            down_targets: list[str] = []
            reconstruction_errors = []
            ordered_groups = (
                assignment.source_neurons[2:]
                + assignment.source_neurons[:2]
            )
            for expert_index, source_neurons in enumerate(ordered_groups):
                if expert_index < 14:
                    expert_prefix = (
                        f"{prefix}.moe.fine_experts.{expert_index}"
                    )
                else:
                    expert_prefix = (
                        f"{prefix}.moe.shared_experts.{expert_index - 14}"
                    )
                gate, up, down, error = _fast_coupled_expert(
                    coupled,
                    signatures,
                    source_neurons,
                    target_width=896,
                    seed=(
                        plan.projection_seed
                        + target_layer * 100
                        + expert_index
                    ),
                )
                gate_key = f"{expert_prefix}.gate_proj.weight"
                up_key = f"{expert_prefix}.up_proj.weight"
                down_key = f"{expert_prefix}.down_proj.weight"
                _copy_state(state, gate_key, gate)
                _copy_state(state, up_key, up)
                _copy_state(state, down_key, down)
                gate_targets.append(gate_key)
                up_targets.append(up_key)
                down_targets.append(down_key)
                reconstruction_errors.append(error)

            fine_groups = assignment.source_neurons[2:]
            ownership_scores = torch.stack(
                [
                    signatures[:, list(source_neurons)].abs().mean(dim=1)
                    for source_neurons in fine_groups
                ],
                dim=1,
            )
            ownership = ownership_scores.argmax(dim=1)
            router_inputs = torch.stack(
                [
                    _project_rows(
                        capture.mlp_inputs[layer].reshape(-1, 2048),
                        projection,
                    )
                    for layer in group
                ]
            ).mean(dim=0)
            prototype_rows = ownership_scores.argmax(dim=0)
            router_inputs = torch.cat(
                (router_inputs, router_inputs[prototype_rows]),
                dim=0,
            )
            ownership = torch.cat(
                (
                    ownership,
                    torch.arange(
                        len(fine_groups),
                        dtype=ownership.dtype,
                    ),
                )
            )
            router = fit_router(
                router_inputs,
                ownership,
                num_experts=14,
                steps=20,
                learning_rate=5e-3,
            )
            router_key = f"{prefix}.moe.fine_router.router.weight"
            _copy_state(state, router_key, router.weight)
            for source_name in gate_names:
                record(
                    source_name,
                    (*gate_targets, router_key),
                    "coupled_neuron_cluster_gate",
                    {
                        "max_reconstruction_mse": max(
                            reconstruction_errors
                        ),
                        "router_accuracy": router.accuracy,
                    },
                )
            for source_name in up_names:
                record(
                    source_name,
                    up_targets,
                    "coupled_neuron_cluster_up",
                )
            for source_name in down_names:
                record(
                    source_name,
                    down_targets,
                    "coupled_neuron_cluster_down",
                )

    donor_tensors = {
        record.name for record in inventory_safetensor(donor_path)
    }
    ledger.assert_complete(donor_tensors, language_targets)
    if classified != donor_tensors:
        raise AssertionError("source classification drift after ledger gate")
    model.load_state_dict(state, strict=True)
    organ_projection = derive_organ_projection(
        source_dim=512,
        target_dim=config.d_model,
        seed=plan.projection_seed,
    )
    organ_report, organ_cosine = _grow_organs(
        model,
        Path(plan.sources.organ_checkpoint.path),
        organ_projection,
    )
    tokenizer_id = _copy_tokenizer(plan)
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite target tensor after assembly: {name}")
    metadata = TransplantCheckpointMetadata(
        tokenizer_id=tokenizer_id,
        plan_id=plan.identity(),
        source_identity={
            "smol_snapshot": plan.sources.smol_snapshot,
            "smol_weights": plan.sources.smol_weights.sha256,
            "organ": plan.sources.organ_checkpoint.sha256,
        },
        projection_identity=projection.identity(),
        organ_projection_identity=organ_projection.identity(),
        coverage_head=ledger.head,
        calibration_digest=plan.calibration_digest,
    )
    result = TransplantCheckpointWriter(checkpoint_root).write(
        model,
        config,
        metadata,
    )
    build_report = {
        "schema": "darwin-smol-build-report-v1",
        "plan_id": plan.identity(),
        "checkpoint": str(result.checkpoint),
        "checkpoint_sha256": result.checkpoint_sha256,
        "coverage_head": ledger.head,
        "donor_tensor_count": len(donor_tensors),
        "target_language_tensor_count": len(language_targets),
        "projection_identity": projection.identity(),
        "projection_explained_energy": projection.explained_energy,
        "organ_projection_identity": organ_projection.identity(),
        "organ_subspace_min_cosine": organ_cosine,
        "organ_report": organ_report,
        "status": "engineering_transplant_only",
    }
    report_path = Path(plan.target.runtime_root) / "build-report.json"
    temporary = report_path.with_suffix(".json.incomplete")
    temporary.write_text(
        json.dumps(build_report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    return NativeBuildResult(
        checkpoint=str(result.checkpoint),
        manifest=str(result.manifest),
        status="engineering_transplant_only",
        coverage_head=ledger.head,
        projection_identity=projection.identity(),
        organ_projection_identity=organ_projection.identity(),
    )
