from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.attention_block import SparseCausalAttentionBlock
from f51_darwin.config import DarwinConfig
from f51_darwin.expert_pool import ExpertPool
from f51_darwin.router import DynamicDepthRouter
from f51_darwin.ssd_block import RMSNorm, SSDBlock


@dataclass(frozen=True)
class DarwinOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    router_scores: torch.Tensor | None = None
    expert_used: list[str] | None = None
    aux_loss: torch.Tensor | None = None
    moe_stats: list[dict] = field(default_factory=list)
    hidden_states: torch.Tensor | None = None  # for JEPA/Ghost Token losses


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    token_ids: list[int]
    finish_reason: str  # "max_length" | "eos" | "stop_token"


class F51DarwinModel(nn.Module):
    """F51 Darwin-SSD — Hybrid SSM-Transformer with optional MoE routing.

    When experts_enabled=True, regular FFN blocks are replaced by MoE layers
    with domain-specialized experts and Nitro-tiered placement (GPU/RAM).

    Args:
        config: DarwinConfig with model hyperparameters.
        expert_pool: Optional ExpertPool for organism lifecycle management.
        router_enabled: Enable DynamicDepthRouter.
        experts_enabled: Enable MoE — replaces FFN with expert routing.
        moe_config: Optional MoE config. If None, uses sensible defaults.
    """

    def __init__(
        self,
        config: DarwinConfig,
        *,
        expert_pool: ExpertPool | None = None,
        router_enabled: bool = False,
        experts_enabled: bool = False,
        moe_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.experts_enabled = experts_enabled
        self.router_enabled = router_enabled

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Build blocks — with or without MoE
        if experts_enabled:
            from f51_darwin.moe_layer import MoEConfig, MoETransformerLayer

            moe_cfg = moe_config or {}
            moe = MoEConfig(
                d_model=config.d_model,
                num_experts=moe_cfg.get('num_experts', 8),
                experts_per_token=moe_cfg.get('experts_per_token', 2),
                expert_hidden_mult=moe_cfg.get('expert_hidden_mult', 4),
                dropout=config.dropout,
                use_nitro_tiering=moe_cfg.get('use_nitro_tiering', True),
                nitro_gpu_capacity=moe_cfg.get('nitro_gpu_capacity', 4),
            )

            self.blocks = nn.ModuleList()
            for index in range(config.n_layers):
                is_attn = index in config.attention_layer_indices
                layer = MoETransformerLayer(
                    d_model=config.d_model,
                    moe_config=moe,
                    is_attention_layer=is_attn,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                self.blocks.append(layer)

            # Label experts by domain
            domains = moe_cfg.get('expert_domains', {})
            for block in self.blocks:
                if hasattr(block, 'moe'):
                    for idx, domain in domains.items():
                        if int(idx) < moe.num_experts:
                            block.moe.label_expert(int(idx), domain)

            self._moe_config = moe
        else:
            self.blocks = nn.ModuleList()
            for index in range(config.n_layers):
                if index in config.attention_layer_indices:
                    self.blocks.append(
                        SparseCausalAttentionBlock(
                            config.d_model, config.n_heads,
                            config.mlp_ratio, config.dropout,
                        )
                    )
                else:
                    self.blocks.append(
                        SSDBlock(config.d_model, config.mlp_ratio, config.dropout)
                    )

        self.router = DynamicDepthRouter(config.d_model)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight
        self.expert_pool = expert_pool
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module is self.lm_head:
                if self.config.weight_tying:
                    return
                nn.init.normal_(module.weight, mean=0.0, std=self._lm_head_init_std())
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = self._lm_head_init_std() if self.config.weight_tying else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _lm_head_init_std(self) -> float:
        # RMSNorm emits roughly unit-variance hidden states; keep random logits
        # near entropy as d_model scales into the 1B+ MoE range.
        return min(0.02, 0.30 / math.sqrt(self.config.d_model))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> DarwinOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("sequence length exceeds config.context_length.")
        x = self.token_embedding(input_ids)

        router_scores = None
        expert_used: list[str] = []
        total_aux_loss = torch.tensor(0.0, device=x.device)
        moe_stats: list[dict] = []

        for i, block in enumerate(self.blocks):
            if self.experts_enabled:
                x, aux_info = block(x)
                if aux_info.get('aux_loss') is not None:
                    total_aux_loss = total_aux_loss + aux_info['aux_loss']
                load_balance_loss = aux_info.get('load_balance_loss')
                router_z_loss = aux_info.get('router_z_loss')
                if isinstance(load_balance_loss, torch.Tensor):
                    load_balance_loss = float(load_balance_loss.detach().cpu())
                if isinstance(router_z_loss, torch.Tensor):
                    router_z_loss = float(router_z_loss.detach().cpu())
                moe_stats.append({
                    'layer': i,
                    'expert_usage': aux_info.get('expert_usage', []),
                    'tokens_per_expert': aux_info.get('tokens_per_expert', []),
                    'expert_usage_fraction': aux_info.get('expert_usage_fraction', []),
                    'dead_experts': aux_info.get('dead_experts', []),
                    'router_entropy': aux_info.get('router_entropy'),
                    'router_entropy_norm': aux_info.get('router_entropy_norm'),
                    'load_balance_loss': load_balance_loss,
                    'router_z_loss': router_z_loss,
                    'gpu_experts': aux_info.get('gpu_experts', []),
                })
            else:
                x = block(x)

            if self.router_enabled and i == len(self.blocks) - 1:
                routing = self.router(x)
                router_scores = routing["scores"]

            if not self.experts_enabled and self.expert_pool is not None:
                active = self.expert_pool.active_records()
                if active:
                    x = self.expert_pool.route_active(x)
                    expert_used = [r.id for r in active]

        x = self.norm(x)
        hidden_states = x  # preserve for JEPA/Ghost Token losses
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape.")
            ce_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + 0.01 * total_aux_loss
        return DarwinOutput(
            logits=logits,
            loss=loss,
            router_scores=router_scores,
            expert_used=expert_used if expert_used else None,
            aux_loss=total_aux_loss if self.experts_enabled else None,
            moe_stats=moe_stats,
            hidden_states=hidden_states,  # for JEPA + Ghost Token
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[int],
        tokenizer=None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token_ids: list[int] | None = None,
        eos_id: int | None = None,
        use_cache: bool = True,
    ) -> GenerationOutput:
        if use_cache:
            return self._generate_cached(
                prompt, tokenizer, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, stop_token_ids=stop_token_ids, eos_id=eos_id,
            )
        return self._generate_legacy(
            prompt, tokenizer, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, stop_token_ids=stop_token_ids, eos_id=eos_id,
        )

    def _generate_cached(self, prompt, tokenizer, **kw):
        """Geração com KV cache — O(1) por token."""
        from f51_darwin.kv_cache import generate_with_cache
        if isinstance(prompt, str):
            if tokenizer is None:
                raise ValueError("tokenizer required when prompt is a string")
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = list(prompt)
        stop_ids = set(kw.get('stop_token_ids') or [])
        eos = kw.get('eos_id') or (tokenizer.eos_id if tokenizer else None)
        text, token_ids, reason = generate_with_cache(
            self, input_ids, tokenizer,
            max_tokens=kw.get('max_tokens', 256),
            temperature=kw.get('temperature', 0.8),
            top_p=kw.get('top_p', 0.95),
            eos_id=eos, stop_ids=stop_ids,
        )
        return GenerationOutput(text=text, token_ids=token_ids, finish_reason=reason)

    def _generate_legacy(self, prompt, tokenizer, **kw):
        """Geração legada — sem KV cache."""
        max_tokens = kw.get('max_tokens', 256)
        temperature = kw.get('temperature', 0.8)
        top_p = kw.get('top_p', 0.95)
        stop_token_ids = kw.get('stop_token_ids')
        eos_id = kw.get('eos_id')

        self.eval()
        device = next(self.parameters()).device

        if isinstance(prompt, str):
            if tokenizer is None:
                raise ValueError("tokenizer required when prompt is a string")
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = list(prompt)

        if eos_id is None and tokenizer is not None:
            eos_id = tokenizer.eos_id

        stop_ids = set(stop_token_ids or [])
        if eos_id is not None:
            stop_ids.add(eos_id)

        generated: list[int] = []
        context = torch.tensor([input_ids], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            if context.size(1) > self.config.context_length:
                context = context[:, -self.config.context_length :]

            output = self.forward(context)
            logits = output.logits[:, -1, :] / max(temperature, 1e-8)

            if temperature < 1e-8:
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                logits_flat = logits.squeeze(0)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_flat, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[1:] = cutoff[:-1].clone()
                    cutoff[0] = False
                    logits_flat[sorted_indices[cutoff]] = -float("inf")
                probs = F.softmax(logits_flat, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token in stop_ids:
                break

            generated.append(next_token)
            context = torch.cat(
                [context, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1
            )

        finish_reason = "max_length" if len(generated) == max_tokens else "eos"

        if tokenizer is not None:
            text = tokenizer.decode(input_ids + generated, skip_special=True)
        else:
            text = " ".join(str(t) for t in input_ids + generated)

        return GenerationOutput(
            text=text,
            token_ids=generated,
            finish_reason=finish_reason,
        )

