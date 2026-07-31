from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.rope import RoPECache, apply_rope, apply_rope_llama
from f51_darwin.ssd_block import SelectiveSSM

if TYPE_CHECKING:
    from f51_darwin.darwin_x_core.neuroendocrine import NeuroendocrineSystem


class GQACausalAttention(nn.Module):
    """Grouped-query causal attention with RoPE.

    Q has n_heads. K/V have n_kv_heads and are repeated across query groups,
    reducing KV-cache size by n_heads / n_kv_heads.
    """

    def __init__(self, config: DarwinXConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeat = config.n_heads // config.n_kv_heads
        kv_dim = config.n_kv_heads * self.head_dim
        # P1.1: bias configuravel. Default True (compat v7); False economiza
        # params e destrava FlashAttention varlen (init novo apenas).
        qkv_bias = config.qkv_bias
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=qkv_bias)
        self.dropout = config.dropout
        self.rope_base_train = config.rope_base_train
        self.rope_base_infer = config.rope_base_infer
        self.rope_style = config.rope_style
        self.context_length = config.context_length
        # P1.2: cache das tabelas cos/sin do RoPE (nao reconstruir a cada forward).
        self._rope_cache = RoPECache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Base RoPE depende do COMPRIMENTO real da sequencia, nao de
        # self.training -- antes, model.eval() (usado pelo holdout/canary)
        # trocava a base pra rope_base_infer mesmo dentro do context_length
        # de treino, medindo o holdout com uma matematica de posicao
        # diferente da do lm_loss de treino (achado 2026-07-26).
        rope_base = self.rope_base_train if seq_len <= self.context_length else self.rope_base_infer
        cos, sin = self._rope_cache.get(
            seq_len, self.head_dim, base=rope_base, device=x.device
        )
        rope = apply_rope_llama if self.rope_style == "llama" else apply_rope
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        # enable_gqa=True (SDPA nativo, torch>=2.5) evita materializar K/V
        # repetidos via repeat_interleave -- mesmo resultado numerico, menos
        # banda de memoria.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                enable_gqa=True,
            )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.o_proj(attn)


class SSDMixerOnly(nn.Module):
    """Selective SSM mixer without a dense FFN.

    Darwin-X places the expensive feed-forward capacity in MoE experts, so the
    SSD branch remains a sequence mixer rather than duplicating FFN cost.
    """

    def __init__(self, config: DarwinXConfig) -> None:
        super().__init__()
        # P0.1: repassa scan_chunk_size do config para o selective scan chunked.
        self.ssm = SelectiveSSM(
            config.d_model,
            expand=config.ssm_expand,
            d_state=config.ssm_state,
            scan_chunk_size=config.scan_chunk_size,
            gradient_checkpointing=config.gradient_checkpointing,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        conv_state: torch.Tensor | None = None,
        ssm_state: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        return self.ssm(
            x, conv_state=conv_state, ssm_state=ssm_state, return_state=return_state
        )


class DenseSwiGLU(nn.Module):
    """Native dense SwiGLU without routing or expert-side control state."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, neuroendocrine: NeuroendocrineSystem | None = None, expert_idx: int = 0) -> torch.Tensor:
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        if neuroendocrine is not None:
            gate = neuroendocrine.expert_gate(expert_idx)
        else:
            gate = 1.0

        return self.dropout(out) * gate


class FineRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # expert_bias: buffer (nao nn.Parameter) atualizado por regra
        # deterministica de balanceamento, nao por gradiente -- estilo
        # DeepSeek-V3 "aux-loss-free load balancing". So afeta a SELECAO
        # top-k (ver forward); o peso de combine (softmax) usa os logits
        # SEM este bias, entao ele nunca compete com o gradiente que
        # orienta a especializacao dos experts. Antes era nn.Parameter
        # treinado por backprop E mutado manualmente (.item(), fora do
        # grafo) pelo jitter anti-colapso -- dois mecanismos de atualizacao
        # no mesmo tensor, com o momentum do Adam ficando desatualizado
        # apos cada salto manual (achado 2026-07-26).
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.expert_bias_update_rate = 0.05
        self.register_buffer("expert_usage_count", torch.zeros(num_experts, dtype=torch.float32))
        # expert_usage_count e zerado a cada 100 steps (janela do
        # anti-colapso, precisa disso pra detectar experts MORTOS agora, nao
        # historicamente). get_hot_experts (usado pelo Nitro pra decidir
        # quem fica na GPU) lia esse MESMO buffer -- logo apos cada reset,
        # sum()==0 e a "lista quente" virava arbitraria (primeiros indices),
        # so recuperando sentido conforme o uso reacumulava, ate ser zerada
        # de novo 100 steps depois. Bug achado 2026-07-26: causaria
        # thrashing (evict/restore CPU<->GPU) se o Nitro fosse ligado.
        # expert_usage_ema NUNCA e resetado -- media movel exponencial de
        # longo prazo, e a fonte de verdade pra "quente" agora.
        self.register_buffer("expert_usage_ema", torch.zeros(num_experts, dtype=torch.float32))
        self.usage_ema_decay = 0.99
        self.register_buffer("expert_miss_count", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("total_steps", torch.zeros(1, dtype=torch.long))
        # GATE 0.2: External bias from ExpertPool (Legacy tiering)
        self.register_buffer("external_bias", torch.zeros(num_experts, dtype=torch.float32))
        # Anti-collapse: track per-expert stats for jitter
        self._init_noise()

    def _init_noise(self) -> None:
        """One-time noise injection to break symmetry."""
        with torch.no_grad():
            noise = torch.randn_like(self.expert_bias) * 0.01
            self.expert_bias.copy_(noise)

    def get_hot_experts(self, top_n: int) -> list[int]:
        """Return indices of most-used experts for Nitro GPU tiering.

        Le expert_usage_ema (nunca resetado), nao expert_usage_count (zerado
        a cada 100 steps pelo anti-colapso) -- ver comentario no __init__.
        """
        if top_n <= 0 or self.expert_usage_ema.sum() == 0:
            return list(range(min(top_n, self.num_experts)))
        _, indices = torch.topk(self.expert_usage_ema, min(top_n, self.num_experts))
        return sorted(indices.tolist())

    def _anti_collapse_jitter(self) -> None:
        """Deterministic load-balance push (DeepSeek-V3 aux-loss-free style).

        Experts below the window's mean usage get expert_bias pushed up by
        a fixed step; experts above the mean get pushed down. No randomness
        (jitter was noise before 2026-07-26) and no gradient (expert_bias is
        a buffer) -- a simple, bounded, deterministic rule that only ever
        affects top-k SELECTION, never the softmax combine weight.
        """
        total_usage = self.expert_usage_count.sum()
        if total_usage == 0:
            return
        mean_usage = total_usage / self.num_experts
        step = self.expert_bias_update_rate
        with torch.no_grad():
            below = self.expert_usage_count < mean_usage
            self.expert_bias += torch.where(
                below,
                torch.full_like(self.expert_bias, step),
                torch.full_like(self.expert_bias, -step),
            )

    def forward(
        self,
        x: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # raw_logits: sinal de roteamento "de verdade" (router aprendido +
        # vies estrutural legitimo de external_bias/routing_bias) -- e o que
        # forma o peso de combine E o que as aux losses (load_balance_loss/
        # router_z_loss em moe.py) medem, entao o gradiente delas continua
        # chegando em self.router.weight normalmente.
        raw_logits = F.linear(x.float(), self.router.weight.float(), None) + self.external_bias.float()
        if routing_bias is not None:
            if routing_bias.numel() != self.num_experts:
                raise ValueError("routing_bias must have one value per expert")
            raw_logits = raw_logits + routing_bias.to(device=raw_logits.device, dtype=raw_logits.dtype)
        # expert_bias entra SO na selecao (nao-diferenciavel, buffer) --
        # ver comentario no __init__.
        selection_logits = raw_logits + self.expert_bias.float()
        top_indices = torch.topk(selection_logits, self.top_k, dim=-1).indices
        top_logits = torch.gather(raw_logits, -1, top_indices)
        weights = F.softmax(top_logits, dim=-1).to(dtype=x.dtype)
        logits = raw_logits
        if self.training:
            with torch.no_grad():
                flat = top_indices.reshape(-1)
                step_usage = torch.zeros_like(self.expert_usage_count)
                step_usage.scatter_add_(
                    0, flat, torch.ones_like(flat, dtype=self.expert_usage_count.dtype)
                )
                self.expert_usage_count += step_usage
                # EMA de longo prazo -- atualizado TODO step, nunca zerado,
                # sobrevive ao reset de 100 steps abaixo. Fonte de verdade
                # pro Nitro (get_hot_experts).
                self.expert_usage_ema.mul_(self.usage_ema_decay).add_(
                    step_usage, alpha=1.0 - self.usage_ema_decay
                )
                self.total_steps += 1
                # Run anti-collapse every 100 steps
                if self.total_steps.cpu().item() % 100 == 0:
                    self._anti_collapse_jitter()
                    self.expert_usage_count.zero_()
        return weights, top_indices, logits
