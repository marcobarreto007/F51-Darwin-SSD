from __future__ import annotations

import hashlib
import math
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from f51_darwin.cognition import CognitiveForwardMetadata

from f51_darwin.darwin_x_core.block import DarwinXBlock
from f51_darwin.darwin_x_core.config import DarwinXConfig, DarwinXOutput
from f51_darwin.darwin_x_core.losses import LossPolicy, LossTerms, compose_loss
from f51_darwin.darwin_x_core.state import _stable_cross_entropy
from f51_darwin.heartbeat import bound_memory_residual
from f51_darwin.ssd_block import RMSNorm


_GHOST_MASK_ALGORITHM_VERSION = "ghost-next-hidden-v2"


def _deterministic_ghost_mask(
    input_ids: torch.Tensor,
    step_digest: str,
    ratio: float,
    *,
    eligible: torch.Tensor,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    if not isinstance(step_digest, str) or not step_digest:
        raise ValueError("step_digest must be a non-empty string")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or not 0.0 <= float(ratio) <= 1.0
    ):
        raise ValueError("ghost ratio must be finite and within [0, 1]")
    if eligible.shape != input_ids.shape or eligible.dtype is not torch.bool:
        raise ValueError("eligible must be a boolean input-shaped tensor")

    seed = (
        int.from_bytes(
            hashlib.sha256(step_digest.encode("utf-8")).digest()[:8],
            "little",
        )
        & 0x7FFFFFFF
    )
    columns = torch.arange(
        input_ids.size(1),
        device=input_ids.device,
        dtype=torch.int64,
    )[None, :]
    mixed = (
        input_ids.to(torch.int64) * 0x45D9F3B
        + columns * 0x165667B1
        + seed
    ) & 0x7FFFFFFF
    threshold = int(float(ratio) * 0x80000000)
    mask = (mixed < threshold) & eligible
    if input_ids.size(1) == 0:
        return mask
    needs_fallback = (
        eligible.any(dim=1)
        & ~mask.any(dim=1)
        & (float(ratio) > 0.0)
    )
    if float(ratio) > 0.0:
        sentinel = torch.full_like(mixed, 0x80000000)
        fallback_columns = torch.where(
            eligible,
            mixed,
            sentinel,
        ).argmin(dim=1, keepdim=True)
        fallback = torch.zeros_like(mask)
        fallback.scatter_(
            1,
            fallback_columns,
            needs_fallback[:, None],
        )
        mask = mask | fallback
    return mask


def deterministic_ghost_mask(
    input_ids: torch.Tensor,
    step_digest: str,
    ratio: float,
) -> torch.Tensor:
    """Build an RNG-free, sample-local Ghost mask for causal probes."""

    columns = torch.arange(
        input_ids.size(1),
        device=input_ids.device,
    )[None, :]
    return _deterministic_ghost_mask(
        input_ids,
        step_digest,
        ratio,
        eligible=input_ids.ge(4) & columns.ge(1),
    )


def _ghost_mask_digest(
    step_digest: str,
    ratio: float,
    shape: torch.Size,
) -> str:
    """Hash trusted mask provenance without reading device mask contents."""

    dimensions = ",".join(str(size) for size in shape)
    provenance = (
        f"{_GHOST_MASK_ALGORITHM_VERSION}\0{step_digest}\0"
        f"{float(ratio).hex()}\0{dimensions}"
    )
    return hashlib.sha256(provenance.encode("utf-8")).hexdigest()


class _ModelPlacementMixin:
    """Device placement and organism safe-boundary control."""

    def enable_dual_gpu(self, gpu0: int = 0, gpu1: int = 1, split_layer: int | None = None):
        """Pipeline parallelism: split blocks across two GPUs.

        Blocks 0..split_layer-1 → GPU0 (primary: embedding, norm, lm_head, heartbeat)
        Blocks split_layer..N-1 → GPU1 (secondary)
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return False
        self._gpu0 = torch.device(f"cuda:{gpu0}")
        self._gpu1 = torch.device(f"cuda:{gpu1}")
        self._split_layer = (
            self.recommended_dual_gpu_split(gpu0=gpu0, gpu1=gpu1)
            if split_layer is None
            else int(split_layer)
        )
        if not 1 <= self._split_layer < self.config.n_layers:
            raise ValueError(
                f"split_layer must be between 1 and {self.config.n_layers - 1}; "
                f"got {self._split_layer}."
            )

        # Move blocks to their GPUs
        for i, block in enumerate(self.blocks):
            target = self._gpu0 if i < self._split_layer else self._gpu1
            block.to(target)

        # Keep core modules on GPU0
        self.token_embedding.to(self._gpu0)
        self.norm.to(self._gpu0)
        self.lm_head.to(self._gpu0)
        self.mtp_heads.to(self._gpu0)
        self.jepa_predictor.to(self._gpu0)
        if self.inter_hemispheric is not None:
            self.inter_hemispheric.to(self._gpu0)
        if self.inter_hemispheric_gate is not None:
            self.inter_hemispheric_gate.data = (
                self.inter_hemispheric_gate.data.to(self._gpu0)
            )
        if self._spider_sense_module is not None:
            self._spider_sense_module.to(self._gpu0)
        if self.ttm_residual_gate is not None:
            self.ttm_residual_gate.data = self.ttm_residual_gate.data.to(
                self._gpu0
            )
        # Moving tied modules separately can replace one Parameter object.
        # Re-establish the weight-sharing contract after placement.
        self.lm_head.weight = self.token_embedding.weight
        if self.heartbeat is not None:
            self.heartbeat.to(device=self._gpu0)
        if self.cognitive_runtime is not None:
            self.cognitive_runtime.to(self._gpu0)

        self._dual_gpu = True
        return True

    @staticmethod
    def _unique_parameter_bytes(modules: list[nn.Module]) -> int:
        """Count parameter storage once, including tied weights only once."""
        seen: set[int] = set()
        total = 0
        for module in modules:
            for parameter in module.parameters():
                identity = id(parameter)
                if identity in seen:
                    continue
                seen.add(identity)
                total += parameter.numel() * parameter.element_size()
        return total

    def recommended_dual_gpu_split(
        self,
        *,
        gpu0: int = 0,
        gpu1: int = 1,
        gpu0_total_bytes: int | None = None,
        gpu1_total_bytes: int | None = None,
    ) -> int:
        """Balance parameter bytes against the available VRAM on both GPUs."""
        if self.config.n_layers < 2:
            raise ValueError("Dual-GPU placement requires at least two layers.")

        if gpu0_total_bytes is None or gpu1_total_bytes is None:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError(
                    "GPU capacities are required when two CUDA devices are unavailable."
                )
            gpu0_total_bytes = torch.cuda.get_device_properties(gpu0).total_memory
            gpu1_total_bytes = torch.cuda.get_device_properties(gpu1).total_memory

        if gpu0_total_bytes <= 0 or gpu1_total_bytes <= 0:
            raise ValueError("GPU capacities must be positive.")

        gpu0_modules: list[nn.Module] = [
            self.token_embedding,
            self.norm,
            self.lm_head,
            self.mtp_heads,
            self.jepa_predictor,
        ]
        if self.heartbeat is not None:
            gpu0_modules.extend(
                [self.heartbeat.ff_stack, self.heartbeat.tt_memory, self.heartbeat.thinker]
            )
        gpu0_fixed = self._unique_parameter_bytes(gpu0_modules)
        block_bytes = [self._unique_parameter_bytes([block]) for block in self.blocks]

        best_split = 1
        best_delta = float("inf")
        for split in range(1, self.config.n_layers):
            pressure0 = (gpu0_fixed + sum(block_bytes[:split])) / gpu0_total_bytes
            pressure1 = sum(block_bytes[split:]) / gpu1_total_bytes
            delta = abs(pressure0 - pressure1)
            if delta < best_delta:
                best_delta = delta
                best_split = split
        return best_split

    def activate_organism(self, *, curiosity=None, ghost_brain=None, jepa=None, organism_ref=None):
        """Ligar o organismo completo — coração, curiosidade, ghost brain."""
        if not self.config.heartbeat_enabled:
            return
        if self.heartbeat is not None:
            self.curiosity = curiosity
            self.ghost_brain_ref = ghost_brain
            # object.__setattr__ (nao self.jepa_ref = jepa): jepa e o MESMO
            # nn.Module que self.jepa_predictor -- via nn.Module.__setattr__
            # normal, atribuir um nn.Module a um atributo o registra de novo
            # como submodulo, duplicando jepa_predictor.* no state_dict
            # salvo (mesmos tensores, dois caminhos -- achado 2026-07-26).
            object.__setattr__(self, "jepa_ref", jepa)
            self.heartbeat.curiosity = curiosity
            self.heartbeat.ghost_brain = ghost_brain
            self.heartbeat.jepa = jepa
            self.heartbeat.organism = organism_ref
            self.heartbeat.explorer.curiosity = curiosity
            self.heartbeat.explorer.ghost_brain = ghost_brain
            return
        from f51_darwin.heartbeat import Heartbeat, HeartbeatConfig
        self.curiosity = curiosity
        self.ghost_brain_ref = ghost_brain
        object.__setattr__(self, "jepa_ref", jepa)  # see comment above
        self.heartbeat = Heartbeat(
            self.config.d_model,
            HeartbeatConfig(
                think_interval=self.config.heartbeat_think_interval,
                explore_interval=self.config.heartbeat_explore_interval,
                self_reward_interval=self.config.heartbeat_explore_interval,
                ff_layers=self.config.heartbeat_ff_layers,
                memory_capacity=self.config.heartbeat_memory_capacity,
                surprise_threshold=self.config.heartbeat_surprise_threshold,
            ),
            ghost_brain=ghost_brain,
            curiosity=curiosity,
            jepa=jepa,
            organism=organism_ref,
        )
        # Mover pra GPU
        device = next(self.parameters()).device
        self.heartbeat.to(device=device, dtype=next(self.parameters()).dtype)

    @torch.no_grad()
    def apply_pending_autonomic_actions(self) -> list[dict[str, Any]]:
        """Commit shape-safe layer decisions after the optimizer boundary."""
        return [
            block.moe.apply_pending_autonomic_actions()
            for block in self.blocks
            if block.moe is not None
        ]

    @torch.no_grad()
    def apply_active_gradient_actions(self) -> list[dict[str, Any]]:
        """Apply bounded DAE actions once at the optimizer boundary."""
        return [
            block.moe.apply_active_gradient_actions()
            for block in self.blocks
            if block.moe is not None
        ]

    @torch.no_grad()
    def execute_structural_actions(self) -> list[dict[str, Any]]:
        """Execute topology changes at SAFE boundary (between cycles).

        This is the ONLY place where expert birth/death/expansion is safe.
        Called after optimizer.step() and before the next forward().
        """
        return [
            block.moe.execute_structural_actions()
            for block in self.blocks
            if block.moe is not None
        ]


class _ModelTopologyMixin:
    """Topology manifest serialization and strict restoration."""

    def topology_manifest(self) -> dict:
        """Serialize the anatomy required before tensor/optimizer restoration.

        The manifest is deliberately lightweight: tensors remain in the regular
        model state dict, while this DNA reconstructs their shapes and identities.
        """
        causal_cognitive = self.config.causal_cognitive_enabled
        manifest = {
            "version": 8 if causal_cognitive else 7,
            "base_config": {
                "model_name": self.config.model_name,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "fine_experts": self.config.fine_experts,
                "experts_per_token": self.config.experts_per_token,
                "fine_expert_hidden_dim": self.config.fine_expert_hidden_dim,
            },
            "topology": [],
            "neuroendocrine_state": [],
            "organism_memory": {
                "replay_buffer": [],  # filled by organism
                "signature_inputs": [],  # filled below
                "ghost_predator_buffer": [],
                "structural_event_log": [],
            },
        }
        if self.config.feed_forward_kind != "moe":
            manifest["base_config"]["feed_forward_kind"] = (
                self.config.feed_forward_kind
            )
        if causal_cognitive:
            manifest["causal_cognitive_contract"] = {
                "loss_semantics_version": self.config.loss_semantics_version,
                "ttm_residual_enabled": self.config.ttm_residual_enabled,
                "ttm_residual_max_scale": self.config.ttm_residual_max_scale,
                "spider_calibration_enabled": (
                    self.config.spider_calibration_enabled
                ),
                "spider_calibration_weight": (
                    self.config.spider_calibration_weight
                ),
                "causal_state_identity_schema": (
                    "darwin-causal-cognitive-state-v1"
                ),
                "required_model_state": (
                    ["ttm_residual_gate"]
                    if self.config.ttm_residual_enabled
                    else []
                ),
                "heartbeat_state_required": (
                    self.config.ttm_residual_enabled
                ),
            }
        for li, block in enumerate(self.blocks):
            moe = block.moe
            if moe is None:
                if block.ffn is None:
                    raise ValueError(
                        f"layer {li} has no feed-forward implementation"
                    )
                manifest["topology"].append({
                    "layer": li,
                    "feed_forward_kind": "dense_swiglu",
                    "hidden_dim": block.ffn.gate_proj.out_features,
                })
                manifest["organism_memory"]["signature_inputs"].append(0)
                manifest["organism_memory"][
                    "ghost_predator_buffer"
                ].append(0)
                manifest["organism_memory"][
                    "structural_event_log"
                ].append([])
                continue
            ne = moe.neuroendocrine
            # Expert manifest with stable identity
            experts = []
            for ei in range(len(moe.fine_experts)):
                expert = moe.fine_experts[ei]
                experts.append({
                    "uuid": moe._expert_ids[ei],
                    "parent_uuid": moe._expert_parent_ids[ei],
                    "birth_step": moe._expert_birth_steps[ei],
                    "layer": li,
                    "index": ei,
                    "hidden_dim": expert.gate_proj.out_features,
                    "device": str(moe._expert_gpu.get(ei, "cpu")),
                    "requires_grad": any(p.requires_grad for p in expert.parameters()),
                })
            manifest["topology"].append({
                "layer": li,
                "num_experts": len(moe.fine_experts),
                "gpu_capacity": moe.gpu_capacity,
                "router_order": list(moe._expert_ids),
                "disabled_expert_ids": [
                    moe._expert_ids[index]
                    for index in sorted(moe._disabled_experts)
                    if index < len(moe._expert_ids)
                ],
                "experts": experts,
            })
            # Hormonal state
            manifest["neuroendocrine_state"].append({
                "layer": li,
                "dopamine": ne.dopamine.tolist(),
                "baseline_dopamine": ne.baseline_dopamine.tolist(),
                "cortisol": ne.cortisol.item(),
                "bdnf": ne.bdnf.tolist(),
                "norepinephrine": ne.norepinephrine.item(),
                "ach_pressure": ne.ach_pressure.item(),
                "expert_grad_norm_ema": ne.expert_grad_norm_ema.tolist(),
                "expert_usage_ema": ne.expert_usage_ema.tolist(),
                "ghost_loss_ema": ne.ghost_loss_ema.item(),
                "ghost_memory_size": ne.ghost_memory_size.item(),
                "router_entropy_ema": ne.router_entropy_ema.item(),
                "router_entropy_initialized": bool(ne.router_entropy_initialized.item()),
                "step_count": int(ne.step_count.item()),
            })
            # Organism memory
            manifest["organism_memory"]["signature_inputs"].append(
                len(moe._signature_inputs)
            )
            manifest["organism_memory"]["ghost_predator_buffer"].append(
                len(moe._ghost_predator_buffer)
            )
            manifest["organism_memory"]["structural_event_log"].append(
                [dict(event) for event in moe._structural_event_log]
            )
        manifest["cognition"] = (
            self.cognitive_runtime.manifest()
            if self.cognitive_runtime is not None
            else None
        )
        return manifest

    @classmethod
    def restore_topology(cls, manifest: dict, model: "DarwinXModel") -> int:
        """Restore organism anatomy from a topology manifest.

        Returns the number of structural mismatches repaired.
        This does NOT load weights — only reconstructs anatomy.
        """
        manifest_version = int(manifest.get("version", 0))
        causal_cognitive = model.config.causal_cognitive_enabled
        if causal_cognitive and manifest_version != 8:
            raise ValueError(
                "causal cognitive topology requires explicit v8 manifest"
            )
        if not causal_cognitive and manifest_version != 7:
            raise ValueError("legacy topology manifest version 7 required")
        if causal_cognitive:
            expected_contract = {
                "loss_semantics_version": model.config.loss_semantics_version,
                "ttm_residual_enabled": model.config.ttm_residual_enabled,
                "ttm_residual_max_scale": model.config.ttm_residual_max_scale,
                "spider_calibration_enabled": (
                    model.config.spider_calibration_enabled
                ),
                "spider_calibration_weight": (
                    model.config.spider_calibration_weight
                ),
                "causal_state_identity_schema": (
                    "darwin-causal-cognitive-state-v1"
                ),
                "required_model_state": (
                    ["ttm_residual_gate"]
                    if model.config.ttm_residual_enabled
                    else []
                ),
                "heartbeat_state_required": (
                    model.config.ttm_residual_enabled
                ),
            }
            if manifest.get("causal_cognitive_contract") != expected_contract:
                raise ValueError(
                    "causal cognitive topology contract does not match model"
                )
        base = manifest.get("base_config", {})
        if int(base.get("n_layers", model.config.n_layers)) != model.config.n_layers:
            raise ValueError("topology manifest layer count does not match model config")
        if int(base.get("d_model", model.config.d_model)) != model.config.d_model:
            raise ValueError("topology manifest d_model does not match model config")
        manifest_feed_forward = str(
            base.get("feed_forward_kind", "moe")
        )
        if manifest_feed_forward != model.config.feed_forward_kind:
            raise ValueError(
                "topology feed-forward kind does not match model config"
            )
        if model.config.feed_forward_kind == "dense_swiglu":
            topology = manifest.get("topology", [])
            if len(topology) != model.config.n_layers:
                raise ValueError(
                    "dense topology must describe every model layer"
                )
            if manifest.get("neuroendocrine_state"):
                raise ValueError(
                    "dense topology cannot contain neuroendocrine expert state"
                )
            for layer, row in enumerate(topology):
                block = model.blocks[layer]
                if block.ffn is None or block.moe is not None:
                    raise ValueError(
                        f"dense topology layer {layer} has invalid anatomy"
                    )
                if (
                    int(row.get("layer", -1)) != layer
                    or row.get("feed_forward_kind") != "dense_swiglu"
                    or int(row.get("hidden_dim", -1))
                    != block.ffn.gate_proj.out_features
                ):
                    raise ValueError(
                        f"dense topology layer {layer} does not match model"
                    )
            return 0

        repaired = 0
        hormone_by_layer = {
            int(state["layer"]): state
            for state in manifest.get("neuroendocrine_state", [])
        }
        event_logs = manifest.get("organism_memory", {}).get("structural_event_log", [])
        for topo in manifest.get("topology", []):
            li = topo["layer"]
            if li >= len(model.blocks):
                continue
            moe = model.blocks[li].moe
            target_count = topo["num_experts"]
            current_count = len(moe.fine_experts)

            # Grow if needed (neurogenesis)
            while len(moe.fine_experts) < target_count:
                moe._create_expert()
                repaired += 1

            # Shrink if needed (apoptosis)
            while len(moe.fine_experts) > target_count:
                if not moe._apoptosis(len(moe.fine_experts) - 1):
                    raise ValueError("manifest requests topology below apoptosis safety floor")
                repaired += 1

            # Restore expert hidden dims
            for ei_info in topo.get("experts", []):
                ei = ei_info["index"]
                if ei < len(moe.fine_experts):
                    expert = moe.fine_experts[ei]
                    target_hidden = ei_info.get("hidden_dim")
                    while target_hidden and expert.gate_proj.out_features < target_hidden:
                        moe._expand_expert(ei)
                        repaired += 1
                    if target_hidden and expert.gate_proj.out_features != target_hidden:
                        raise ValueError(
                            f"cannot reconstruct expert hidden size {target_hidden}"
                        )

            expert_infos = topo.get("experts", [])
            expert_ids = [str(info["uuid"]) for info in expert_infos]
            if len(expert_ids) != target_count or len(set(expert_ids)) != target_count:
                raise ValueError("manifest expert UUIDs must be complete and unique per layer")
            router_order = topo.get("router_order", expert_ids)
            if list(router_order) != expert_ids:
                raise ValueError("manifest router order does not match expert order")
            moe._expert_ids = expert_ids
            moe._expert_parent_ids = [info.get("parent_uuid") for info in expert_infos]
            moe._expert_birth_steps = [int(info.get("birth_step", 0)) for info in expert_infos]
            disabled_ids = set(topo.get("disabled_expert_ids", []))
            moe._disabled_experts = {
                index for index, expert_id in enumerate(expert_ids) if expert_id in disabled_ids
            }
            moe.gpu_capacity = int(topo.get("gpu_capacity", moe.gpu_capacity))

            hormonal = hormone_by_layer.get(li)
            if hormonal:
                vector_names = (
                    "dopamine", "baseline_dopamine", "bdnf",
                    "expert_grad_norm_ema", "expert_usage_ema",
                )
                scalar_names = (
                    "cortisol", "norepinephrine", "ach_pressure",
                    "ghost_loss_ema", "ghost_memory_size", "router_entropy_ema",
                )
                with torch.no_grad():
                    for name in vector_names:
                        if name in hormonal:
                            target = getattr(moe.neuroendocrine, name)
                            target.copy_(torch.as_tensor(
                                hormonal[name], device=target.device, dtype=target.dtype
                            ))
                    for name in scalar_names:
                        if name in hormonal:
                            getattr(moe.neuroendocrine, name).fill_(hormonal[name])
                    moe.neuroendocrine.router_entropy_initialized.fill_(
                        bool(hormonal.get("router_entropy_initialized", False))
                    )
                    moe.neuroendocrine.step_count.fill_(int(hormonal.get("step_count", 0)))
            if li < len(event_logs) and isinstance(event_logs[li], list):
                moe._structural_event_log = [dict(event) for event in event_logs[li]][-128:]

        return repaired


class _ModelForwardMixin:
    """Initialization, training forward pass and heartbeat state."""

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # dt_proj tem init proprio (canonical Mamba-1, ver ssm_core.py) e
            # marca _no_reinit=True. Antes so o bias era protegido -- o weight
            # era pisado incondicionalmente pelo init generico, reduzindo a
            # input-dependence do delta (bug irmao do V9, achado 2026-07-26).
            if not getattr(module, '_no_reinit', False):
                nn.init.normal_(module.weight, mean=0.0, std=min(0.02, 0.30 / math.sqrt(self.config.d_model)))
            if module.bias is not None and not getattr(module, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=min(0.02, 0.30 / math.sqrt(self.config.d_model)))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        domain: str = "train",
        heartbeat: bool | None = None,
        step_digest: str | None = None,
        cognitive_metadata: CognitiveForwardMetadata | None = None,
        cognitive_mode: str = "shadow",
    ) -> DarwinXOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("sequence length exceeds training context_length.")

        x = self.token_embedding(input_ids)
        # Inter-Hemispheric lateralization (entre embedding e blocos)
        if self.inter_hemispheric is not None:
            lateralized = self.inter_hemispheric(x)
            if self.inter_hemispheric_gate is None:
                x = lateralized
            else:
                gate = torch.tanh(self.inter_hemispheric_gate)
                x = x + gate * (lateralized - x)
        total_aux = torch.tensor(0.0, device=x.device)
        moe_stats: list[dict[str, Any]] = []
        gaba_observations: list[dict[str, Any]] = []

        if self._dual_gpu:
            # ── Pipeline: GPU0 blocks → GPU1 blocks → back to GPU0 ──
            for i in range(self._split_layer):
                x, aux = self.blocks[i](x)
                if "aux_loss" in aux:
                    total_aux = (
                        total_aux
                        + aux["aux_loss"].to(total_aux.device)
                    )
                if self.blocks[i].moe is not None:
                    moe_stats.append(aux)
                if "gaba_observation" in aux:
                    gaba_observations.append(aux["gaba_observation"])
                # ── Vertical: propagate routing bias to next layer ──
                if i + 1 < len(self.blocks):
                    current_moe = self.blocks[i].moe
                    next_moe = self.blocks[i + 1].moe
                    if current_moe is not None and next_moe is not None:
                        next_moe._vertical_bias = (
                            current_moe._emit_vertical_bias()
                        )
            x = x.to(self._gpu1)
            for i in range(self._split_layer, len(self.blocks)):
                x, aux = self.blocks[i](x)
                if "aux_loss" in aux:
                    total_aux = (
                        total_aux
                        + aux["aux_loss"].to(total_aux.device)
                    )
                if self.blocks[i].moe is not None:
                    moe_stats.append(aux)
                if "gaba_observation" in aux:
                    gaba_observations.append(aux["gaba_observation"])
                if i + 1 < len(self.blocks):
                    current_moe = self.blocks[i].moe
                    next_moe = self.blocks[i + 1].moe
                    if current_moe is not None and next_moe is not None:
                        next_moe._vertical_bias = (
                            current_moe._emit_vertical_bias()
                        )
            x = x.to(self._gpu0)
        else:
            for i, block in enumerate(self.blocks):
                x, aux = block(x)
                if "aux_loss" in aux:
                    total_aux = total_aux + aux["aux_loss"]
                if block.moe is not None:
                    moe_stats.append(aux)
                if "gaba_observation" in aux:
                    gaba_observations.append(aux["gaba_observation"])
                # ── Vertical: propagate routing bias to next layer ──
                if i + 1 < len(self.blocks):
                    current_moe = block.moe
                    next_moe = self.blocks[i + 1].moe
                    if current_moe is not None and next_moe is not None:
                        next_moe._vertical_bias = (
                            current_moe._emit_vertical_bias()
                        )

        # Media entre camadas, nao soma cru -- os coeficientes internos de
        # aux_loss (0.01*load_balance + 1e-3*router_z, moe.py) sao pensados
        # por camada (Switch Transformer faz media entre as camadas MoE);
        # somar por 12 camadas inflava o peso efetivo do balanceamento
        # proporcionalmente a profundidade (achado 2026-07-26, parcialmente
        # mascarado pelo aux_loss_adaptive que so recorta o excesso em vez
        # de corrigir na fonte).
        if len(self.blocks) > 0:
            total_aux = total_aux / len(self.blocks)

        hidden = self.norm(x)
        cognitive_pulse_events: tuple[dict[str, Any], ...] = ()
        if self.cognitive_runtime is not None and cognitive_metadata is not None:
            if cognitive_mode == "active":
                self.cognitive_runtime.set_active()
                hidden, pulse_record = self.cognitive_runtime.observe_active(
                    hidden,
                    cognitive_metadata,
                )
                if self.config.cognitive_pulse_enabled:
                    cognitive_pulse_events = (pulse_record.to_payload(),)
            elif self.config.cognitive_shadow_enabled:
                hidden, pulse_record = self.cognitive_runtime.observe_shadow(
                    hidden,
                    cognitive_metadata,
                )
                if self.config.cognitive_pulse_enabled:
                    cognitive_pulse_events = (pulse_record.to_payload(),)
        elif (
            self.cognitive_runtime is not None
            and self.config.cognitive_shadow_enabled
        ):
            if cognitive_metadata is None:
                raise ValueError(
                    "cognitive_metadata is required in cognitive shadow mode"
                )
            hidden, pulse_record = self.cognitive_runtime.observe_shadow(
                hidden,
                cognitive_metadata,
            )
            if self.config.cognitive_pulse_enabled:
                cognitive_pulse_events = (pulse_record.to_payload(),)
        hidden_for_logits = hidden
        ttm_memory_retrieved = False
        ttm_residual_applied = False
        entity_positions = None
        if (
            self.config.ttm_entity_addressing
            and self.config.spider_sense_enabled
            and self._spider_sense_module is not None
        ):
            # Saliencia por token, calculada sobre hidden PURO (antes do TTM,
            # nao hidden_for_logits) -- decide QUAIS posicoes valem virar
            # chave/valor de memoria. Reusa o Spider-Sense ja existente como
            # detector de saliencia (baixa confianca = token novo/dificil de
            # prever = candidato a entidade/evento), sem parametro novo.
            # Calculo separado do bloco "SPIDER SENSE" mais abaixo (que usa
            # hidden_for_logits pra calibracao) -- duplica um MLP de 2
            # camadas barato em vez de arriscar mudar o comportamento
            # existente de spider_calibration (2026-07-27).
            with torch.no_grad():
                entity_confidence = self._spider_sense_module(hidden.detach())  # [B, T]
                saliencia = 1.0 - entity_confidence
                k = min(self.config.ttm_entity_top_k, saliencia.size(1))
                entity_positions = saliencia.topk(k, dim=1).indices  # [B, k]

        if self.ttm_residual_gate is not None and self.heartbeat is not None:
            if self.config.ttm_entity_addressing:
                # O readout da linguagem e decidido em T-1. A escrita
                # associativa usa o mesmo regime [B,D], evitando comparar uma
                # media de sequencia com tokens Spider arbitrarios.
                query = hidden[:, -1, :].detach()
                value = self.heartbeat.tt_memory.retrieve(
                    query,
                    top_k=1,
                    already_pooled=True,
                )
                if value is not None:
                    ttm_memory_retrieved = True
                    residual_scale = self.ttm_residual_scale().to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    bounded = bound_memory_residual(
                        value.to(device=hidden.device, dtype=hidden.dtype),
                        query,
                    )
                    hidden_for_logits = hidden.clone()
                    hidden_for_logits[:, -1, :] = (
                        hidden[:, -1, :] + residual_scale * bounded
                    )
                    ttm_residual_applied = bool(
                        residual_scale.detach().ne(0).item()
                    )
            else:
                memory = self.heartbeat.tt_memory.retrieve(hidden.detach())
                if memory is not None:
                    ttm_memory_retrieved = True
                    residual_scale = self.ttm_residual_scale().to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    hidden_for_logits = hidden + residual_scale * memory.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                    ttm_residual_applied = bool(
                        residual_scale.detach().ne(0).item()
                    )
            if self.training and self.config.ttm_associative_weight > 0.0:
                # proj_value only ever sees gradient through this term: the
                # values it writes to memory come from a non-differentiable
                # Python-list cache write (write_if_surprised), so unlike
                # proj_key (trained via retrieve(), see heartbeat.py) it has
                # no other path to a real training signal. A same-step
                # self-reconstruction loss keeps it from staying at its
                # random init forever; it is not full Titans (no inner-loop
                # associative K->V objective), just the minimal fix that is
                # safe to add without changing any parameter shape.
                pooled = hidden.detach().mean(dim=1)
                ttm_value = self.heartbeat.tt_memory.proj_value(pooled)
                ttm_recon_loss = (
                    1.0 - F.cosine_similarity(ttm_value, pooled, dim=-1)
                ).mean()
                total_aux = (
                    total_aux
                    + self.config.ttm_associative_weight * ttm_recon_loss
                )
        logits = self.lm_head(hidden_for_logits)
        lm_loss = mtp_loss = jepa_loss_value = loss = effective_aux_loss = None
        effective_jepa_loss = jepa_dist_loss_value = None
        jepa_loss_value = None
        spider_loss = effective_spider_loss = None
        if labels is not None:
            lm_loss = _stable_cross_entropy(logits[:, :-1], labels[:, 1:])
            # hidden_for_logits (nao hidden) -- mesma representacao que
            # lm_head/JEPA consomem. Antes MTP via o hidden PRE-TTM enquanto
            # lm_head/JEPA viam POS-TTM: o mesmo lm_head tied recebia
            # gradiente de dois regimes de representacao diferentes quando
            # ttm_residual_enabled (achado 2026-07-26).
            mtp_loss = self._mtp_loss(hidden_for_logits, labels)
            if self.config.jepa_weight > 0:
                # JEPA agora recebe hidden_for_logits (com residual TTM quando
                # houver memoria recuperada). Spider detecta "isso e novo?",
                # TTM responde "ja vi algo parecido", e JEPA usa esse contexto
                # pra prever melhor a representacao futura.
                jepa_loss_value, jepa_dist_loss_value = self._jepa_loss(
                    hidden_for_logits,
                    detach_target_for_anticollapse=self.config.ttm_residual_enabled,
                    temporal_mask_ratio=self.config.jepa_temporal_mask_ratio,
                )
            else:
                jepa_loss_value = torch.tensor(0.0, device=hidden.device)
                jepa_dist_loss_value = torch.tensor(0.0, device=hidden.device)

        # ── SPIDER SENSE: danger score from hidden states ──
        spider_danger = torch.zeros(
            (hidden.size(0), 1, 1),
            device=x.device,
            dtype=hidden.dtype,
        )
        spider_confidence = None
        internal_spider_confidence = None
        if self.config.spider_sense_enabled and hidden is not None:
            # Spider-Sense: MLP real ou fallback heurístico
            if self._spider_sense_module is not None:
                internal_spider_confidence = self._spider_sense_module(
                    hidden_for_logits
                )
                spider_danger = (
                    1.0
                    - internal_spider_confidence.mean(
                        dim=1, keepdim=True
                    ).unsqueeze(-1)
                ).detach().to(dtype=hidden.dtype)
            else:
                spider_danger = torch.sigmoid(
                    hidden.float().var(dim=-1).mean(dim=1, keepdim=True)
                ).unsqueeze(-1).detach().to(dtype=hidden.dtype)
        if (
            self.config.spider_calibration_enabled
            and internal_spider_confidence is not None
        ):
            spider_confidence = internal_spider_confidence
            if labels is not None:
                from f51_darwin.spider_sense import SpiderSense

                correctness = SpiderSense.token_correctness(logits, labels)
                spider_loss = SpiderSense.brier_loss(
                    spider_confidence[:, :-1],
                    correctness,
                    valid_mask=labels[:, 1:].detach().ne(-100),
                )

        # ── GHOST TOKEN: auxiliary survival prediction (optional, heavy) ──
        # P2.3: _ghost_loss recomputa um forward COMPLETO pelos blocos
        # (~50% de compute extra). Gate ghost_enabled permite desligar para
        # ablation/corte de compute. REWRITE futuro: injetar a mascara no
        # forward principal e reusar o hidden state, evitando a 2a passagem.
        raw_ghost_loss = torch.tensor(0.0, device=x.device)
        ghost_mask_digest = None
        if (
            labels is not None
            and self.config.ghost_enabled
            and self.config.ghost_mask_ratio > 0
        ):
            if step_digest is None:
                raw_ghost_loss = self._ghost_loss(input_ids, labels, hidden)
            else:
                raw_ghost_loss, ghost_mask_digest = (
                    self._causal_ghost_loss(
                        input_ids,
                        labels,
                        hidden,
                        step_digest=step_digest,
                    )
                )

        ghost_loss = raw_ghost_loss
        if labels is not None:
            assert lm_loss is not None
            assert mtp_loss is not None
            assert jepa_loss_value is not None
            composed = compose_loss(
                LossTerms(
                    lm=lm_loss,
                    mtp=mtp_loss,
                    jepa=jepa_loss_value,
                    aux=total_aux,
                    ghost=raw_ghost_loss,
                    spider=spider_loss,
                ),
                LossPolicy(
                    mtp_scale=self.config.mtp_weight,
                    jepa_scale=self.config.jepa_weight,
                    aux_scale=self.config.aux_loss_scale,
                    ghost_scale=self.config.ghost_weight,
                    spider_scale=self.config.spider_calibration_weight,
                    aux_adaptive=self.config.aux_loss_adaptive,
                    loss_semantics_version=self.config.loss_semantics_version,
                ),
            )
            loss = composed.total
            effective_aux_loss = composed.effective.aux
            effective_jepa_loss = composed.effective.jepa
            if spider_loss is not None:
                effective_spider_loss = composed.effective.spider
            # Preserve the existing DarwinXOutput contract: ghost_loss has
            # always exposed the weighted term used by the objective.
            ghost_loss = composed.effective.ghost

        if step_digest is None:
            ghost_signal = float(ghost_loss.detach().float().cpu())
            for block in self.blocks:
                if block.moe is not None:
                    block.moe._ghost_loss = ghost_signal

        heartbeat_stats = None
        heartbeat_enabled = self.config.heartbeat_enabled if heartbeat is None else heartbeat
        if heartbeat_enabled and self.heartbeat is not None:
            with torch.no_grad():
                # jepa_dist_loss (previsao real), nao jepa_loss_value
                # (dist+VICReg combinado) -- os termos VICReg dominam a
                # magnitude (~0.05 cada) sobre jepa_dist_loss (~0.0008 apos
                # rebalanceamento), entao usar o total como "surpresa" reage
                # a regularizacao anti-colapso, nao a erro de predicao real
                # (achado 2026-07-26, ver hipotese de interferencia TTM+JEPA).
                jepa_error = (
                    float(jepa_dist_loss_value.detach().float().clamp(min=0.0, max=10.0).cpu())
                    if jepa_dist_loss_value is not None
                    else 0.0
                )
                loss_value = (
                    float(loss.detach().float().clamp(min=0.0, max=100.0).cpu())
                    if loss is not None
                    else 0.0
                )
                self.heartbeat.to(device=hidden.device, dtype=hidden.dtype)
                if (
                    self.config.ttm_entity_addressing
                    and entity_positions is not None
                ):
                    # Escrita endereçada por entidade: um slot por posicao
                    # marcada (nao um pooled da sequencia inteira) -- so
                    # assim a memoria acumula identidade de entidade/evento
                    # em vez de media borrada (ver comentario no bloco de
                    # retrieve acima, 2026-07-27).
                    for b in range(hidden.size(0)):
                        for pos in entity_positions[b].tolist():
                            pos_hidden = hidden[b : b + 1, pos, :].detach()
                            self.heartbeat.tt_memory.write_if_surprised(
                                pos_hidden,
                                jepa_error=jepa_error,
                                domain=domain,
                                already_pooled=True,
                            )
                heartbeat_stats = self.heartbeat.beat(
                    hidden.detach(),
                    domain=domain,
                    jepa_error=jepa_error,
                    loss=loss_value,
                    expose_jepa_signal=self.config.causal_cognitive_enabled,
                    memory_used=(
                        ttm_memory_retrieved
                        if self.config.ttm_residual_enabled
                        else None
                    ),
                )
                if self.config.ttm_residual_enabled:
                    heartbeat_stats["ttm_memory_retrieved"] = (
                        ttm_memory_retrieved
                    )
                    heartbeat_stats["ttm_residual_applied"] = (
                        ttm_residual_applied
                    )
                    heartbeat_stats["ttm_residual_scale"] = float(
                        self.ttm_residual_scale().detach().float().cpu()
                    )

        return DarwinXOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            mtp_loss=mtp_loss,
            jepa_loss=jepa_loss_value,
            jepa_dist_loss=jepa_dist_loss_value,
            spider_confidence=spider_confidence,
            spider_loss=spider_loss,
            effective_spider_loss=effective_spider_loss,
            ttm_memory_retrieved=(
                ttm_memory_retrieved
                if self.config.ttm_residual_enabled
                else None
            ),
            aux_loss=total_aux,
            effective_aux_loss=effective_aux_loss,
            raw_ghost_loss=raw_ghost_loss,
            ghost_loss=ghost_loss,
            ghost_mask_digest=ghost_mask_digest,
            effective_jepa_loss=effective_jepa_loss,
            hidden_states=(
                hidden_for_logits
                if not self.config.spider_sense_enabled
                else hidden_for_logits * (1.0 - spider_danger)
            ),
            moe_stats=moe_stats,
            heartbeat_stats=heartbeat_stats,
            gaba_observations=tuple(gaba_observations),
            cognitive_pulse_events=cognitive_pulse_events,
        )

    def commit_gaba_observations(self, output: DarwinXOutput) -> int:
        """Commit ordered GABA evidence after a successful attempt outcome."""
        gaba_layers = tuple(
            block.gaba
            for block in self.blocks
            if block.gaba is not None
        )
        observations = output.gaba_observations
        if len(observations) != len(gaba_layers):
            raise ValueError(
                "GABA observation count does not match enabled blocks"
            )
        layer_observations = tuple(zip(
            gaba_layers,
            observations,
            strict=True,
        ))
        validated = (
            gaba_layers[0].validate_observations_batch(layer_observations)
            if layer_observations
            else ()
        )
        snapshots = tuple(
            layer._snapshot_commit_state()
            for layer, _ in layer_observations
        )
        try:
            for (layer, _), observation in zip(
                layer_observations,
                validated,
                strict=True,
            ):
                layer._commit_validated_observation(observation)
        except Exception:
            for (layer, _), snapshot in zip(
                layer_observations,
                snapshots,
                strict=True,
            ):
                layer._restore_commit_state(snapshot)
            raise
        return len(observations)

    def heartbeat_state_dict(self) -> dict[str, Any] | None:
        if self.heartbeat is None:
            return None
        return self.heartbeat.state_dict()

    def load_heartbeat_state_dict(self, state: dict[str, Any] | None) -> None:
        if self.heartbeat is not None and state is not None:
            self.heartbeat.load_state_dict(state)

    def ttm_residual_scale(self) -> torch.Tensor:
        """Bound the learned TTM residual gate symmetrically around zero."""
        if self.ttm_residual_gate is None:
            return self.token_embedding.weight.new_zeros(())
        return (
            self.config.ttm_residual_max_scale
            * torch.tanh(self.ttm_residual_gate)
        )


class _ModelGenerationMixin:
    """Autoregressive local inference."""

    @torch.no_grad()
    def live_generate(
        self,
        prompt: str,
        tokenizer,
        *,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        eos_id: int | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Geração VIVA — o coração bate e influencia cada token.

        Diferente de generate(), aqui o heartbeat:
        - Recupera memórias relevantes (Test-Time Memory)
        - Injeta pensamentos internos (Quiet-STaR)
        - Pondera confiança (Spider Sense)
        - Aprende com Forward-Forward (dopamina)
        - Explora domínios (Curiosity)

        Returns dict com: text, tokens, heartbeat_stats, memories_used, thoughts
        """
        device = next(self.parameters()).device
        config = self.config
        stop_ids = {eos_id} if eos_id else {tokenizer.eos_id}

        # Encode prompt
        ids = tokenizer.encode(prompt)
        context = torch.tensor([ids[-config.context_length:]], dtype=torch.long, device=device)

        generated: list[int] = []
        finish_reason = "max_length"
        heartbeat_log: list[dict] = []
        memories_used = 0
        thoughts_used = 0

        for step in range(max_tokens):
            # ── Forward pass ──
            # live_generate owns the inference heartbeat below. Disabling the
            # forward hook here prevents two beats (and two memory writes) for
            # every generated token when heartbeat_enabled=True.
            output = self.forward(context, heartbeat=False)
            logits = output.logits[:, -1, :] / max(temperature, 1e-8)

            # ── HEARTBEAT: aprender + lembrar + pensar ──
            if self.heartbeat is not None and output.hidden_states is not None:
                self.heartbeat.to(device=device, dtype=output.hidden_states.dtype)
                hb = self.heartbeat.beat(
                    output.hidden_states.detach(),
                    domain="inference",
                    jepa_error=0.0,
                    loss=0.0,
                    memory_used=(
                        output.ttm_memory_retrieved
                        if config.ttm_residual_enabled
                        else None
                    ),
                )
                heartbeat_log.append(hb)

                # 💾 Recuperar memórias relevantes
                if config.ttm_residual_enabled:
                    # The bounded residual was already applied before lm_head
                    # by forward(). Never add the historical 0.15 logit bias a
                    # second time under the causal TTM contract.
                    if output.ttm_memory_retrieved:
                        memories_used += 1
                    memory = None
                else:
                    memory = self.heartbeat.tt_memory.retrieve(
                        output.hidden_states
                    )
                if memory is not None:
                    # Injetar memória como viés nos logits
                    # Projetar memória [1,1,d_model] → [vocab_size]
                    memory_bias = self.lm_head(memory.to(device=device, dtype=logits.dtype)).squeeze()
                    logits = logits + 0.15 * memory_bias  # peso leve
                    memories_used += 1
                    if verbose:
                        print(f"  💾 Memória recuperada (beat {hb.get('beat', '?')})")

                # 🤔 Quiet-STaR: pensamento interno
                thought = self.heartbeat.thinker.think(output.hidden_states)
                if thought is not None:
                    thought_logits = self.lm_head(thought.to(device=device, dtype=logits.dtype)).mean(dim=1)
                    logits = logits + 0.05 * thought_logits.squeeze(0)  # peso leve
                    thoughts_used += 1

            # ── Amostrar próximo token ──
            if temperature < 1e-8:
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[1:] = cutoff[:-1].clone()
                    cutoff[0] = False
                    logits_flat = logits.clone()
                    logits_flat.scatter_(-1, sorted_indices, sorted_logits.masked_fill(cutoff, float("-inf")))
                else:
                    logits_flat = logits
                probs = F.softmax(logits_flat, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token in stop_ids:
                finish_reason = "eos"
                break

            generated.append(next_token)
            context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)
            # Sem isso, gerar mais tokens do que context_length - len(prompt)
            # estoura o ValueError de forward() (bug achado 2026-07-26).
            if context.size(1) > config.context_length:
                context = context[:, -config.context_length:]

        all_ids = ids + generated
        text = tokenizer.decode(all_ids, skip_special=True) if tokenizer else " ".join(map(str, all_ids))

        return {
            "text": text,
            "token_ids": generated,
            "finish_reason": finish_reason,
            "heartbeat_beats": len(heartbeat_log),
            "memories_used": memories_used,
            "thoughts_used": thoughts_used,
            "dopamine": heartbeat_log[-1].get("dopamine", 0.5) if heartbeat_log else 0.5,
        }


class _ModelAuxiliaryLossMixin:
    """Auxiliary MTP, Ghost and JEPA objectives."""

    def _mtp_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = []
        for offset, head in enumerate(self.mtp_heads, start=2):
            if hidden.size(1) <= offset:
                continue
            pred_hidden = head(hidden[:, :-offset])
            logits = self.lm_head(pred_hidden)
            losses.append(_stable_cross_entropy(logits, labels[:, offset:]))
        if not losses:
            return torch.tensor(0.0, device=hidden.device)
        return torch.stack(losses).mean()

    def _ghost_loss(
        self, input_ids: torch.Tensor, labels: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Jurassic Park Ghost Token: strategic exploration of 15% dark tokens."""
        batch, seq = input_ids.shape
        prob = torch.rand(batch, seq, device=input_ids.device)
        special_mask = input_ids < 4
        mask = (prob < self.config.ghost_mask_ratio) & (~special_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input_ids.device)
        masked_ids = input_ids.clone()
        masked_ids[mask] = 2  # <mask> token
        x = self.token_embedding(masked_ids)
        total_aux = torch.tensor(0.0, device=x.device)
        if self._dual_gpu:
            for i in range(self._split_layer):
                x, aux = self.blocks[i](x)
                if "aux_loss" in aux:
                    total_aux = (
                        total_aux
                        + aux["aux_loss"].to(total_aux.device)
                    )
            x = x.to(self._gpu1)
            for i in range(self._split_layer, len(self.blocks)):
                x, aux = self.blocks[i](x)
                if "aux_loss" in aux:
                    total_aux = (
                        total_aux
                        + aux["aux_loss"].to(total_aux.device)
                    )
            x = x.to(self._gpu0)
        else:
            for block in self.blocks:
                x, aux = block(x)
                if "aux_loss" in aux:
                    total_aux = total_aux + aux["aux_loss"]
        hidden_ghost = self.norm(x)
        logits_ghost = self.lm_head(hidden_ghost)
        return _stable_cross_entropy(logits_ghost[mask], labels[mask])

    def _causal_ghost_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        hidden: torch.Tensor,
        *,
        step_digest: str,
    ) -> tuple[torch.Tensor, str]:
        """Compute one deterministic Ghost probe without a second organ pass."""

        columns = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        )[None, :]
        eligible = (
            input_ids.ge(4)
            & labels.ne(-100)
            & columns.ge(1)
        )
        mask = _deterministic_ghost_mask(
            input_ids,
            step_digest,
            self.config.ghost_mask_ratio,
            eligible=eligible,
        )
        digest = _ghost_mask_digest(
            step_digest,
            self.config.ghost_mask_ratio,
            input_ids.shape,
        )
        current = hidden.detach()[:, :-1]  # backbone isolated — Ghost must not
        # compete with LM gradients through the 12 transformer blocks.
        predictor = getattr(
            self.jepa_predictor,
            "predictor",
            self.jepa_predictor,
        )
        # Ghost trains the JEPA predictor (no .detach()).  The predictor's
        # job is to map current hidden → next hidden; Ghost uses it to map
        # current hidden → masked-token hidden.  Both are representation-
        # prediction tasks with aligned gradients.  The earlier double-detach
        # (commit a0de9a0) made Ghost a constant-term with zero training
        # signal — the largest auxiliary loss (20% of total) was dead weight.
        predicted_hidden = predictor(current)
        logits_ghost = self.lm_head(predicted_hidden)
        target_labels = labels[:, 1:]
        target_mask = mask[:, 1:] & target_labels.ne(-100)
        selected_logits = logits_ghost[target_mask]
        selected_labels = target_labels[target_mask]
        if selected_labels.numel() == 0:
            return logits_ghost.sum() * 0.0, digest
        return (
            _stable_cross_entropy(selected_logits, selected_labels),
            digest,
        )

    # TODO(fresh-start apenas — adiciona um submodulo novo, quebra resume
    # estrito): nao ha EMA target encoder real aqui (diferente do I-JEPA/
    # V-JEPA canonico) -- o "alvo" e a mesma saida do backbone ao vivo, so
    # com stop-gradient simples (.detach() abaixo), nao uma rede separada
    # com pesos atualizados por momentum (theta_ema <- m*theta_ema +
    # (1-m)*theta). E o organ com melhor resultado empirico no ablation
    # (2026-07-26: heldout_lm 6.97 vs 7.34 do baseline puro) -- um EMA
    # target encoder de verdade e a maior alavanca de melhoria identificada,
    # mas exige nova linhagem fresh-start pra validar (custo de memoria de
    # ~dobrar os params copiados, m~=0.996-0.999 como no BYOL/I-JEPA).
    def _jepa_loss(
        self, hidden: torch.Tensor, *, detach_target_for_anticollapse: bool = False,
        temporal_mask_ratio: float = 0.50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.size(1) <= 1:
            zero = torch.tensor(0.0, device=hidden.device)
            return zero, zero
        if not isinstance(self.jepa_predictor, nn.Sequential):
            predicted, target, _ = self.jepa_predictor(hidden)
        else:
            predicted = self.jepa_predictor(hidden[:, :-1])
            target = hidden[:, 1:]

        # ── Temporal masking (2026-07-29): JEPA canonico usa mascaramento
        # espacial.  Aqui mascara posicoes TEMPORAIS para romper a correlacao
        # trivial hidden[t] ~= hidden[t+1] (cosine 0.9975, model.py:1381-1383).
        # Sem mascara, a tarefa "prever o vizinho temporal identico" e
        # degenerada e o predictor colapsa para identidade.
        # Com mascara, o predictor so avalia posicoes mascaradas (default 50%),
        # forcando o modelo a aprender representacoes nao-locais.
        B, Tp, D = predicted.shape  # Tp = T-1 (MLP) or T (Transformer)
        if temporal_mask_ratio > 0.0 and Tp > 4:
            mask = torch.rand(B, Tp, 1, device=hidden.device, dtype=predicted.dtype)
            mask = (mask < temporal_mask_ratio)
            # Apply mask: only masked positions contribute to prediction loss
            # Unmasked positions receive zero weight (no gradient from pred loss)
            # VICReg terms still operate on full tensor to prevent collapse
            pred_flat = predicted[mask.expand(-1, -1, D)].reshape(-1, D)
            target_flat_for_pred = target[mask.expand(-1, -1, D)].reshape(-1, D)
            # For anti-collapse: still use full tensor
            target_for_anticollapse = target.detach() if detach_target_for_anticollapse else target
            target_flat_live = target_for_anticollapse.flatten(0, 1)
        else:
            pred_flat = predicted.flatten(0, 1)
            target_flat_for_pred = None  # signal to use full pred_flat
            target_for_anticollapse = target.detach() if detach_target_for_anticollapse else target
            target_flat_live = target_for_anticollapse.flatten(0, 1)

        # ── Backbone anisotropy fix (measured, not assumed — see Exp7 in
        # DIARIO_DE_BORDO.md: shuffled-target cosine == real-target cosine,
        # 0.9975 vs 0.9975, meaning `hidden` vectors were nearly parallel
        # regardless of content). Center + decorrelate the RAW backbone
        # representation (target_flat_live = hidden[:, 1:], pre-normalization)
        # so the anti-collapse gradient reaching `hidden` does not pass
        # through F.normalize's Jacobian on its way back to the backbone.
        # This is additive to (not a replacement for) the normalized-space
        # terms below, which regularize the predictor's own output.
        # backbone_cov_loss: RMS normalize BEFORE covariance to prevent
        # O(||h||^3) gradient explosion on the backbone.
        target_rms = target_flat_live * torch.rsqrt(target_flat_live.pow(2).mean(-1, keepdim=True) + 1e-6)
        target_centered = target_rms - target_rms.mean(dim=0, keepdim=True)
        backbone_cov = (target_centered.T @ target_centered) / (target_centered.size(0) - 1)
        backbone_cov_loss = (
            backbone_cov.pow(2).sum() - backbone_cov.diag().pow(2).sum()
        ) / target_centered.size(-1)

        # ── Scale-invariant distance WITHOUT reintroducing 1/||z|| decay ──
        # F.normalize(x)'s backward pass projects the incoming gradient onto
        # the tangent plane of the unit sphere at x/||x|| and divides by
        # ||x||: exactly the sin(phi)/||z|| suppression this whole fix exists
        # to remove (measured: with this suppression live for the distance
        # AND the VICReg terms below, jepa_predictor's own gradient dropped
        # to ~1.3e-5 for this checkpoint's ||predicted||~=508 — five orders
        # of magnitude smaller than the ~0.9 pre-fix baseline, i.e. worse,
        # not better; scratchpad/jepa_backbone_fix_smoketest.py reproduces
        # this). Dividing by a DETACHED norm keeps the loss scale-invariant
        # (same value as true cosine distance) but its gradient is a plain
        # per-row rescale, not a radial projection — no suppression.
        eps = 1e-8
        pred_scale = pred_flat.norm(dim=-1, keepdim=True).detach() + eps
        target_scale = target_flat_live.norm(dim=-1, keepdim=True).detach() + eps
        pred_norm = pred_flat / pred_scale
        target_norm_live = target_flat_live / target_scale

        # Bilateral Covariance Decorrelation on unit hypersphere
        target_ctr = target_norm_live - target_norm_live.mean(dim=0)
        target_cov = (target_ctr.T @ target_ctr) / (target_norm_live.size(0) - 1)
        target_cov_loss = (
            target_cov.pow(2).sum() - target_cov.diag().pow(2).sum()
        ) / target_norm_live.size(-1)

        target_norm = target_norm_live.detach()
        target_flat = target_flat_live.detach()

        # Canonical L2 MSE distance on (detached-norm) normalized embeddings.
        # When temporal masking is active, only masked positions count.
        if target_flat_for_pred is not None:
            target_for_dist = target_flat_for_pred / (
                target_flat_for_pred.norm(dim=-1, keepdim=True).detach() + eps)
            jepa_dist_loss = F.mse_loss(pred_norm, target_for_dist.detach())
        else:
            jepa_dist_loss = F.mse_loss(pred_norm, target_norm)

        # Variance & Covariance anti-collapse on the RAW predictor output
        # (not the normalized one): these terms exist to keep the predictor
        # from collapsing, and need their own gradient to reach
        # jepa_predictor's weights at real magnitude — routing them through
        # the same detached-norm division would still scale them down by
        # ~1/||predicted|| per row, just without the extra radial-projection
        # penalty of a live F.normalize. Keeping them on pred_flat directly
        # (as before 6c18b08) sidesteps that scaling entirely.
        pred_ctr = pred_flat - pred_flat.mean(dim=0)
        pred_cov = (pred_ctr.T @ pred_ctr) / (pred_flat.size(0) - 1)
        pred_cov_loss = (pred_cov.pow(2).sum() - pred_cov.diag().pow(2).sum()) / pred_flat.size(-1)

        # Normalize by mean per-dim std before hinge so gamma=1 is reachable.
        pred_scale = pred_flat.std(dim=0).mean().detach() + eps
        pred_scaled = pred_flat / pred_scale
        std = torch.sqrt(pred_scaled.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()

        # Weight rebalance (2026-07-25): measured in production that
        # jepa_dist_loss (~0.0008) was ~60x smaller than the summed
        # VICReg/backbone anti-collapse terms (~0.05 each) — the gradient
        # reaching the predictor/backbone was dominated almost entirely by
        # "don't collapse", with almost no signal from "predict well".
        # jepa_dist_loss itself (returned unweighted below, for logging) is
        # the actual prediction-quality metric; only its contribution to
        # the combined loss is scaled up here.
        jepa_dist_weight = self.config.jepa_dist_weight
        total_jepa_loss = (
            jepa_dist_weight * jepa_dist_loss
            + 0.05 * var_loss + 0.025 * pred_cov_loss + 0.05 * target_cov_loss
            + 0.05 * backbone_cov_loss
        )

        with torch.no_grad():
            if target_flat_for_pred is not None:
                cos_sim = F.cosine_similarity(
                    pred_flat.detach(), target_flat_for_pred.detach())
            else:
                cos_sim = F.cosine_similarity(pred_flat.detach(), target_flat.detach())

        if hasattr(self.jepa_predictor, "update_metrics"):
            self.jepa_predictor.update_metrics(
                total_jepa_loss.detach(), cos_sim.mean().detach(),
            )
        # Exposed separately (not just folded into total_jepa_loss) so the
        # actual prediction-quality signal can be told apart from the
        # VICReg/backbone regularization terms, which can dominate the
        # combined number and mask whether the distance term is moving at
        # all — this was an observability gap discovered while validating
        # the backbone anisotropy fix in production.
        return total_jepa_loss, jepa_dist_loss.detach()


class DarwinXModel(
    _ModelPlacementMixin,
    _ModelTopologyMixin,
    _ModelForwardMixin,
    _ModelGenerationMixin,
    _ModelAuxiliaryLossMixin,
    nn.Module,
):
    """F51 Darwin-X model with stable module registration and checkpoint topology."""

    def __init__(self, config: DarwinXConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                DarwinXBlock(config, is_attention_layer=index in config.attention_layer_indices)
                for index in range(config.n_layers)
            ]
        )
        # Inter-Hemispheric lateralization (opcional)
        self.inter_hemispheric = None
        self.inter_hemispheric_gate = None
        if getattr(config, 'inter_hemispheric_enabled', False):
            try:
                from f51_darwin.inter_hemispheric import InterHemisphericSystem, HemisphereConfig
                hemi_cfg = HemisphereConfig(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    dropout=config.dropout,
                )
                self.inter_hemispheric = InterHemisphericSystem(hemi_cfg)
                if config.inter_hemispheric_residual_enabled:
                    self.inter_hemispheric_gate = nn.Parameter(torch.zeros(()))
            except ImportError:
                pass
        self.norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            fp32=config.rms_norm_fp32,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        # TODO(fresh-start apenas — adiciona parametros, quebra resume
        # estrito): cabecas MTP sao nn.Linear cru, sem RMSNorm/residuo antes
        # de alimentar o lm_head tied -- diferente do caminho principal, que
        # sempre normaliza antes de lm_head. DeepSeek-V3/Meta (Gloeckle 2024)
        # normalizam a saida de cada trunk head. Nao aplicado em 2026-07-26
        # porque adicionar RMSNorm por cabeca muda o state_dict e quebraria
        # o resume estrito de David 1/David 2 em andamento.
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(config.mtp_depth)]
        )
        # JEPA predictor — V2 com surprise momentum se disponível.
        # jepa_predictor_arch (config, estrutural — ver config.py) escolhe
        # entre o MLP posição-a-posição (JEPAHeadV2, default, compatível com
        # todo checkpoint v9 existente) e o transformer causal
        # (JEPAHeadTransformer) — este último SÓ deve rodar contra um
        # checkpoint_root isolado até validação empírica (ver
        # DIARIO_DE_BORDO.md), nunca resumindo direto o checkpoint de
        # produção atual (foi treinado com JEPAHeadV2; strict=True no resume
        # abortaria de qualquer forma por chaves incompatíveis).
        predictor_arch = getattr(config, "jepa_predictor_arch", "mlp")
        try:
            if predictor_arch == "transformer":
                from f51_darwin.jepa_v2 import JEPAHeadTransformer
                self.jepa_predictor = JEPAHeadTransformer(
                    config,
                    n_layers=getattr(config, "jepa_predictor_layers", 2),
                )
            else:
                from f51_darwin.jepa_v2 import JEPAHeadV2
                self.jepa_predictor = JEPAHeadV2(
                    d_model=config.d_model,
                    bottleneck_dim=config.d_model // 2,
                )
        except ImportError:
            self.jepa_predictor = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                RMSNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
            )
        self.heartbeat = None
        self.ttm_residual_gate = None
        if config.ttm_residual_enabled:
            # A newly enabled causal residual must be an exact no-op.  Using
            # randn here both altered logits before evidence existed and
            # consumed RNG state, changing otherwise identical backbone
            # initialization.
            self.ttm_residual_gate = nn.Parameter(torch.zeros(()))
        # Spider-Sense MLP (não heurística de variância)
        self._spider_sense_module = None
        if getattr(config, 'spider_sense_enabled', False):
            try:
                from f51_darwin.spider_sense import SpiderSense
                self._spider_sense_module = SpiderSense(d_model=config.d_model)
            except ImportError:
                pass
        self.curiosity = None
        self.ghost_brain_ref = None
        self.jepa_ref = None
        # Dual-GPU pipeline state
        self._dual_gpu = False
        self._gpu0 = None
        self._gpu1 = None
        self._split_layer = 0
        self.apply(self._init_weights)
        self.cognitive_runtime = None
        if config.cognitive_architecture_version == "three_organs_v1":
            from f51_darwin.cognition import CognitiveRuntime

            self.cognitive_runtime = CognitiveRuntime(
                d_model=config.d_model,
                max_scale=config.cognitive_residual_max_scale,
            )
        if self.config.heartbeat_enabled:
            self.activate_organism(jepa=self.jepa_predictor)
