from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from f51_darwin.darwin_x_core.losses import (
    CAUSAL_LOSS_SEMANTICS_VERSION,
    LEGACY_LOSS_SEMANTICS_VERSION,
    SUPPORTED_LOSS_SEMANTICS_VERSIONS,
)

from f51_darwin.config import coerce_mapping


@dataclass(frozen=True)
class DarwinXConfig:
    """Contract for the canonical local F51 Darwin-X 1.6B-Nitro architecture.

    Canonical lineages remain random-init. Isolated, hash-bound F51 surgery
    lineages may use ``f51_adapted`` after their source and coverage gates pass.
    """

    model_name: str = "F51-Darwin-X-1.6B-Nitro"
    init: str = "random"
    tokenizer: str = "f51_bpe"
    vocab_size: int = 58_162
    context_length: int = 4096
    inference_context_length: int = 32_768
    d_model: int = 1920
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    ssd_attention_ratio: str = "3:1"
    dropout: float = 0.0
    rope_base_train: float = 10_000.0
    rope_base_infer: float = 1_000_000.0
    rope_style: str = "interleaved"
    residual_scale_multiplier: float = 1.4
    ssm_expand: int = 2
    ssm_state: int = 16
    # P1.1: bias nos projetores QKVO. Default True preserva a linhagem v7
    # (que foi treinada com bias). False economiza ~3% params de atencao e
    # destrava FlashAttention varlen — use apenas em init aleatorio novo,
    # pois muda as chaves do state_dict (resume estrito quebraria).
    qkv_bias: bool = True
    rms_norm_eps: float = 1e-6
    rms_norm_fp32: bool = False
    # Structural feed-forward choice. The default preserves every existing
    # Darwin lineage; dense_swiglu is reserved for isolated fresh roots.
    feed_forward_kind: str = "moe"
    fine_experts: int = 14
    shared_experts: int = 2
    experts_per_token: int = 2
    fine_expert_hidden_dim: int = 896
    shared_expert_hidden_dim: int = 896
    mtp_depth: int = 2
    mtp_weight: float = 0.15
    # Estrutural de propósito (não adicionar a _OPERATIONAL_CONFIG_FIELDS em
    # checkpoint_root.py): muda o TIPO do módulo jepa_predictor (MLP vs.
    # transformer causal), não um hiperparâmetro de treino — precisa que
    # structural_config_identity mude junto, para que preflight_checkpoint_root
    # recuse silenciosamente um resume incompatível em vez de deixar passar.
    jepa_predictor_arch: str = "mlp"  # "mlp" | "transformer"
    jepa_predictor_layers: int = 2  # só usado quando jepa_predictor_arch="transformer"
    jepa_weight: float = 0.05
    jepa_dist_weight: float = 1.0  # multiplier for jepa_dist_loss inside _jepa_loss
    jepa_temporal_mask_ratio: float = 0.0  # 0.0=off, 0.50=mask half: only masked positions count
    ghost_weight: float = 0.07
    ghost_mask_ratio: float = 0.15
    # P2.3: gate do Ghost Token. _ghost_loss faz um SEGUNDO forward completo
    # pelos blocos (~50% de compute extra). Default True preserva a linhagem v7
    # (treinada com Ghost ativo). Setar False para ablation ou para cortar
    # compute enquanto o orgao nao for reescrito para reusar o hidden state do
    # forward principal (single-pass) em vez de recomputar tudo.
    ghost_enabled: bool = True
    curiosity_weight: float = 0.02
    spider_sense_enabled: bool = True
    aux_loss_scale: float = 1.0
    aux_loss_adaptive: bool = True
    loss_semantics_version: int = LEGACY_LOSS_SEMANTICS_VERSION
    heartbeat_enabled: bool = True
    heartbeat_think_interval: int = 1
    heartbeat_explore_interval: int = 10
    heartbeat_memory_capacity: int = 1024
    heartbeat_surprise_threshold: float = 0.3
    heartbeat_ff_layers: int = 3
    # Causal cognitive organs require loss-v2 and therefore a new v8 lineage.
    # Defaults are exact legacy no-ops and add no model parameters.
    ttm_residual_enabled: bool = False
    ttm_residual_max_scale: float = 0.15
    # Weight of the TTM proj_value self-reconstruction auxiliary loss (see
    # DarwinXModel._jepa_loss neighbourhood in model.py for the TTM term).
    # 0.0 fully disables it without touching architecture.
    ttm_associative_weight: float = 0.02
    # Endereçamento por entidade (2026-07-27): em vez de escrever/ler UM
    # vetor pooled por sequencia (media borrada de todos os tokens -- ver
    # achado de que isso reforca anisotropia e "esquece" identidade), marca
    # as posicoes mais salientes (via Spider-Sense) e faz uma escrita/leitura
    # de memoria POR POSICAO marcada, injetando o resultado so nela (esparso,
    # nao broadcast). Default False preserva o comportamento pooled exato
    # (legacy no-op, zero parametro novo -- reaproveita proj_key/proj_value).
    ttm_entity_addressing: bool = False
    ttm_entity_top_k: int = 4
    spider_calibration_enabled: bool = False
    spider_calibration_weight: float = 0.0
    legacy_tiers: int = 7
    nitro_enabled: bool = True
    nitro_gpu_expert_capacity: int = 5
    # Darwin Active Gradient Engine.  Defaults preserve the trained v7
    # lineage: evidence is collected, but gradients are not changed.
    dae_enabled: bool = False
    dae_shadow_mode: bool = True
    dae_update_gain_min: float = 0.75
    dae_update_gain_max: float = 1.25
    vertical_routing_scale: float = 0.1
    # ── Three-organ cognition (disabled preserves every legacy lineage) ──
    cognitive_architecture_version: str = "disabled"
    cognitive_shadow_enabled: bool = False
    cognitive_pulse_enabled: bool = False
    cognitive_organ_width: int = 512
    cognitive_residual_max_scale: float = 0.15
    # ── Órgãos integrados (toggles) ──
    # Default False preserva a linhagem v7 (checkpoints treinados sem estes órgãos).
    # True ativa o órgão no forward pass. Requer init aleatório novo se o órgão
    # adicionar parâmetros treináveis (GABA, Spider-Sense MLP, Inter-Hemispheric).
    gaba_enabled: bool = False
    inter_hemispheric_enabled: bool = False
    inter_hemispheric_residual_enabled: bool = False
    sleep_enabled: bool = False
    decision_engine_enabled: bool = False
    unified_mesh_enabled: bool = False
    # ── P0.1/P0.3 (experimentais, default preserva linhagem treinada) ──
    # Tamanho do bloco (tokens) do selective scan chunked. 512 oferece
    # melhor granularidade de chunking que 1024, reduzindo pico de memoria
    # de O(B*D*S*seq) para O(B*D*S*512) sem perda de qualidade (testado em
    # _test_chunking_equivalence com max_err < 1e-4).
    scan_chunk_size: int = 512
    gradient_checkpointing: bool = False
    # Cadencia explicita de camadas de atencao. Vazio = auto (3,7,11,15 para
    # n_layers=16). Use para experimentos (ex.: [0,7,11,15] porem requer init
    # aleatorio — NAO resumir de checkpoint com cadencia 3,7,11,15).
    attention_indices_override: tuple[int, ...] = field(default_factory=tuple)
    module_states: tuple[str, ...] = field(
        default_factory=lambda: ("candidate", "active", "frozen", "quarantine", "merged", "dead")
    )

    def __post_init__(self) -> None:
        if self.init not in {"random", "f51_adapted"}:
            raise ValueError("Darwin-X init must be random or f51_adapted.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.rope_style not in {"interleaved", "llama"}:
            raise ValueError("rope_style must be interleaved or llama.")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be finite and positive.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA.")
        if self.ssd_attention_ratio != "3:1":
            raise ValueError("Darwin-X 4B contract requires ssd_attention_ratio='3:1'.")
        if self.feed_forward_kind not in {"moe", "dense_swiglu"}:
            raise ValueError(
                "feed_forward_kind must be 'moe' or 'dense_swiglu'."
            )
        if self.feed_forward_kind == "dense_swiglu":
            if self.fine_experts != 1:
                raise ValueError(
                    "dense_swiglu requires fine_experts=1."
                )
            if self.shared_experts != 0:
                raise ValueError(
                    "dense_swiglu requires shared_experts=0."
                )
            if self.experts_per_token != 1:
                raise ValueError(
                    "dense_swiglu requires experts_per_token=1."
                )
        if self.experts_per_token > self.fine_experts:
            raise ValueError("experts_per_token cannot exceed fine_experts.")
        if self.context_length <= 1 or self.inference_context_length < self.context_length:
            raise ValueError("invalid context length contract.")
        if self.heartbeat_think_interval < 1:
            raise ValueError("heartbeat_think_interval must be >= 1.")
        if self.heartbeat_explore_interval < 1:
            raise ValueError("heartbeat_explore_interval must be >= 1.")
        if self.heartbeat_memory_capacity < 1:
            raise ValueError("heartbeat_memory_capacity must be >= 1.")
        if self.heartbeat_ff_layers < 1:
            raise ValueError("heartbeat_ff_layers must be >= 1.")
        for name in (
            "mtp_weight",
            "jepa_weight",
            "ghost_weight",
            "aux_loss_scale",
            "spider_calibration_weight",
            "ttm_residual_max_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if self.loss_semantics_version not in SUPPORTED_LOSS_SEMANTICS_VERSIONS:
            raise ValueError(
                "loss_semantics_version must be one of "
                f"{sorted(SUPPORTED_LOSS_SEMANTICS_VERSIONS)}."
            )
        if (
            self.ttm_residual_enabled or self.spider_calibration_enabled
        ) and self.loss_semantics_version != CAUSAL_LOSS_SEMANTICS_VERSION:
            raise ValueError(
                "causal cognitive toggles require loss_semantics_version=2."
            )
        if self.ttm_residual_enabled and not self.heartbeat_enabled:
            raise ValueError(
                "ttm_residual_enabled requires heartbeat_enabled=True."
            )
        if self.spider_calibration_enabled and not self.spider_sense_enabled:
            raise ValueError(
                "spider_calibration_enabled requires spider_sense_enabled=True."
            )
        if not 0.0 <= self.vertical_routing_scale <= 1.0:
            raise ValueError("vertical_routing_scale must be in [0, 1].")
        if self.scan_chunk_size < 1:
            raise ValueError("scan_chunk_size must be >= 1.")
        if self.dae_update_gain_min <= 0:
            raise ValueError("dae_update_gain_min must be positive.")
        if self.dae_update_gain_max < self.dae_update_gain_min:
            raise ValueError(
                "dae_update_gain_max must be >= dae_update_gain_min."
            )
        for idx in self.attention_indices_override:
            if not 0 <= idx < self.n_layers:
                raise ValueError(
                    f"attention_indices_override tem indice {idx} fora do intervalo [0,{self.n_layers})."
                )
        if self.cognitive_architecture_version not in {
            "disabled",
            "three_organs_v1",
        }:
            raise ValueError("unsupported cognitive_architecture_version")
        if (
            self.cognitive_architecture_version == "three_organs_v1"
            and self.cognitive_organ_width != 512
        ):
            raise ValueError("three_organs_v1 requires organ_width=512")
        if (
            self.cognitive_shadow_enabled or self.cognitive_pulse_enabled
        ) and self.cognitive_architecture_version != "three_organs_v1":
            raise ValueError(
                "cognitive shadow/pulse require three_organs_v1"
            )
        if (
            not math.isfinite(self.cognitive_residual_max_scale)
            or not 0.0 <= self.cognitive_residual_max_scale <= 1.0
        ):
            raise ValueError(
                "cognitive_residual_max_scale must be finite and in [0, 1]"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def attention_layer_indices(self) -> tuple[int, ...]:
        # P0.3: override explicito tem precedencia (para experimentos); vazio
        # preserva a cadencia canonica 3:1 (indices 3,7,11,15 p/ 16 camadas),
        # compativel com os checkpoints treinados v7.
        if self.attention_indices_override:
            return tuple(sorted(set(self.attention_indices_override)))
        return tuple(index for index in range(self.n_layers) if (index + 1) % 4 == 0)

    @property
    def ssd_layer_indices(self) -> tuple[int, ...]:
        attention = set(self.attention_layer_indices)
        return tuple(index for index in range(self.n_layers) if index not in attention)

    @property
    def residual_scale(self) -> float:
        return self.residual_scale_multiplier / math.sqrt(self.n_layers)

    @property
    def causal_cognitive_enabled(self) -> bool:
        return bool(
            self.loss_semantics_version == CAUSAL_LOSS_SEMANTICS_VERSION
            and (self.ttm_residual_enabled or self.spider_calibration_enabled)
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DarwinXConfig":
        kwargs = coerce_mapping(cls, raw)
        for tuple_field in ("module_states", "attention_indices_override"):
            if tuple_field in kwargs:
                kwargs[tuple_field] = tuple(kwargs[tuple_field])
        return cls(**kwargs)


@dataclass(frozen=True)
class DarwinXOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    jepa_loss: torch.Tensor | None = None
    jepa_dist_loss: torch.Tensor | None = None
    spider_confidence: torch.Tensor | None = None
    spider_loss: torch.Tensor | None = None
    effective_spider_loss: torch.Tensor | None = None
    ttm_memory_retrieved: bool | None = None
    aux_loss: torch.Tensor | None = None
    effective_aux_loss: torch.Tensor | None = None
    raw_ghost_loss: torch.Tensor | None = None
    ghost_loss: torch.Tensor | None = None
    ghost_mask_digest: str | None = None
    effective_jepa_loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    moe_stats: list[dict[str, Any]] = field(default_factory=list)
    heartbeat_stats: dict[str, Any] | None = None
    gaba_observations: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )
    cognitive_pulse_events: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )
