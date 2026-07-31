from __future__ import annotations

import math

import torch
from torch import nn


class DiscreteSSDMixer(nn.Module):
    """Reference discrete SSD mixer with scalar decay per head.

    The recurrence is:

        H_t = alpha_t H_{t-1} + B_t outer X_t
        Y_t = C_t H_t

    It intentionally exposes both recurrent and materialized matrix forms so
    architecture-transfer losses can compare it directly with causal attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.input_norm = nn.Identity()
        self.c_proj = nn.Linear(self.d_model, self.d_model)
        self.b_proj = nn.Linear(self.d_model, self.d_model)
        self.x_proj = nn.Linear(self.d_model, self.d_model)
        self.decay_proj = nn.Linear(self.d_model, self.n_heads)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            self.decay_proj.weight.zero_()
            self.decay_proj.bias.fill_(4.0)

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape

        def heads(projection: nn.Linear) -> torch.Tensor:
            return (
                projection(hidden_states)
                .view(batch, sequence, self.n_heads, self.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        c = heads(self.c_proj)
        b = heads(self.b_proj)
        x = heads(self.x_proj)
        alpha = torch.sigmoid(self.decay_proj(hidden_states)).permute(0, 2, 1)
        return c, b, x, alpha

    def project_values(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._project(hidden_states)[2]

    def mix_recurrent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        c, b, x, alpha = self._project(hidden_states)
        batch, heads, sequence, head_dim = b.shape
        state = torch.zeros(
            batch,
            heads,
            head_dim,
            head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        outputs: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(head_dim)
        for position in range(sequence):
            state = (
                alpha[:, :, position, None, None] * state
                + torch.einsum(
                    "bhn,bhp->bhnp",
                    b[:, :, position],
                    x[:, :, position],
                )
            )
            output = torch.einsum(
                "bhn,bhnp->bhp",
                c[:, :, position],
                state,
            )
            outputs.append(output * scale)
        return torch.stack(outputs, dim=2)

    def materialize_mixer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        c, b, _, alpha = self._project(hidden_states)
        _, _, sequence, head_dim = b.shape
        scores = torch.einsum("bhid,bhjd->bhij", c, b)
        log_prefix = torch.cumsum(
            torch.log(alpha.clamp_min(torch.finfo(alpha.dtype).tiny)),
            dim=-1,
        )
        log_decay = log_prefix.unsqueeze(-1) - log_prefix.unsqueeze(-2)
        causal = torch.ones(
            sequence,
            sequence,
            device=hidden_states.device,
            dtype=torch.bool,
        ).tril()
        decay = torch.exp(
            log_decay.masked_fill(~causal, -torch.inf)
        )
        return scores * decay / math.sqrt(head_dim)

    def mix_parallel(self, hidden_states: torch.Tensor) -> torch.Tensor:
        matrix = self.materialize_mixer(hidden_states)
        values = self.project_values(hidden_states)
        return torch.einsum("bhij,bhjd->bhid", matrix, values)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(hidden_states)
        mixed = (
            self.mix_recurrent(normalized)
            if normalized.shape[1] == 1
            else self.mix_parallel(normalized)
        )
        batch, heads, sequence, head_dim = mixed.shape
        merged = (
            mixed.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, sequence, heads * head_dim)
        )
        return self.dropout(self.out_proj(merged))

    def load_gpt2_attention_projections(
        self,
        *,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        out_weight: torch.Tensor,
        out_bias: torch.Tensor,
    ) -> None:
        expected_qkv = (self.d_model, self.d_model * 3)
        if tuple(qkv_weight.shape) != expected_qkv:
            raise ValueError(
                f"qkv weight shape mismatch: expected={expected_qkv} actual={tuple(qkv_weight.shape)}"
            )
        if tuple(qkv_bias.shape) != (self.d_model * 3,):
            raise ValueError("qkv bias shape mismatch")
        if tuple(out_weight.shape) != (self.d_model, self.d_model):
            raise ValueError("output weight shape mismatch")
        if tuple(out_bias.shape) != (self.d_model,):
            raise ValueError("output bias shape mismatch")

        q_weight, k_weight, v_weight = qkv_weight.split(self.d_model, dim=1)
        q_bias, k_bias, v_bias = qkv_bias.split(self.d_model, dim=0)
        with torch.no_grad():
            self.c_proj.weight.copy_(q_weight.T)
            self.c_proj.bias.copy_(q_bias)
            self.b_proj.weight.copy_(k_weight.T)
            self.b_proj.bias.copy_(k_bias)
            self.x_proj.weight.copy_(v_weight.T)
            self.x_proj.bias.copy_(v_bias)
            self.out_proj.weight.copy_(out_weight.T)
            self.out_proj.bias.copy_(out_bias)
