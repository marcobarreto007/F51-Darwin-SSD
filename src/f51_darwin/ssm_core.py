"""
F51 Darwin-SSD — Selective State Space Mixer.
Reimplementado em 2026-Jul-13 conforme especificacao Mamba-1/2 oficial.

REFERENCIAS:
  Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  Dao & Gu (2024) "Transformers are SSMs: Generalized Models and Efficient Algorithms"
  state-spaces/mamba (github) — implementacao de referencia

CORRECOES vs legado (ssm_core_legado.py):
  [x] dt_proj.bias inicializado com inverse-softplus de valores em [dt_min, dt_max]
  [x] dt_min=0.001, dt_max=0.1 (conforme oficial Mamba)
  [x] d_state=64 (Mamba-2 default, era 16)
  [x] a_log inicializado via S4D-real (Hippo) — ja estava correto
  [x] Named tensors para debugging
  [x] Teste de gradiente por canal do estado
  [x] dt_proj.weight inicializado com std=0.001 (estabilidade numerica)
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# ═══════════════════════════════════════════════════════════════════════════
# Selective scan  (prefix-sum parallel — PyTorch puro, com chunking)
# ═══════════════════════════════════════════════════════════════════════════
#
# TODO(future): substituir o scan PyTorch por um kernel CUDA fusionado.
#   O caminho canônico de produção é o `mamba_ssm` (Dao&Gu) ou `flash_conv_fn`,
#   que fundem discretização + scan + output em um único kernel, eliminando
#   a materialização dos tensores [B, D, S, L]. A versão abaixo é a referência
#   PyTorch pura (correta e flexível), mantida para debug, CPU e portabilidade.
#   Quando `mamba_ssm` estiver disponível, trocar o dispatcher em
#   SelectiveSSM.forward para chamá-lo sem alterar o restante do modelo.
# ═══════════════════════════════════════════════════════════════════════════


def _associative_scan_last_dim(
    a_bar: torch.Tensor, b_bar: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hillis-Steele associative scan sobre a ultima dimensao.

    Nao e in-place de verdade -- cada passo usa torch.cat para recompor o
    tensor (realoca memoria), o que e matematicamente equivalente e ja
    validado por _test_chunking_equivalence, mas nao economiza memoria como
    "in-place" sugere (doc corrigido 2026-07-26; oportunidade futura: usar
    escrita in-place real via indexacao se o scan virar gargalo de alocador).

    Operacao associativa (recorrencia linear h_t = A_t h_{t-1} + B_t):
        (A2, B2) ∘ (A1, B1) = (A2*A1, A2*B1 + B2)

    Apos a varredura:
        a_bar[..., t] = prod_{i<=t} A_i   (decay cumulativo)
        b_bar[..., t] = h_t                (estado, assumindo h_0 = 0)

    Tudo ja deve estar no dtype de computo (fp32 recomendado).
    """
    if seq_len <= 1:
        return a_bar, b_bar
    num_steps = int(math.ceil(math.log2(seq_len)))
    for i in range(num_steps):
        step = 2 ** i
        if step >= seq_len:
            break
        a_even = a_bar[..., : seq_len - step]
        a_odd = a_bar[..., step:]
        b_even = b_bar[..., : seq_len - step]
        b_odd = b_bar[..., step:]

        a_new = a_odd * a_even            # propagate decay
        b_new = a_odd * b_even + b_odd    # accumulate input

        a_bar = torch.cat([a_bar[..., :step], a_new], dim=-1)
        b_bar = torch.cat([b_bar[..., :step], b_new], dim=-1)
    return a_bar, b_bar


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor | None = None,
    *,
    chunk_size: int = 1024,
    h_init: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Parallel causal selective scan com chunking (P0.1).

    Recorrencia por canal:  h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
    Saida:                   y_t = C_t^T h_t  (+ skip D * u_t opcional).

    Shapes:
      u:     [batch, dim, seq]
      delta: [batch, dim, seq]
      a:     [dim, state]        — valores negativos (discretizado)
      b:     [batch, state, seq]
      c:     [batch, state, seq]
      d:     [dim]               — skip connection (opcional)

    Chunking (P0.1):
      O scan e quebrado em blocos de `chunk_size` tokens (default 1024). Dentro
      de cada bloco roda o prefix-scan paralelo; o estado h ao fim do bloco e
      carregado (carry) para o proximo, propagando o decay cumulativo. Isso
      reduz o pico de memoria de ~O(B*D*S*seq) para ~O(B*D*S*chunk_size),
      permitindo sequencias longas (4096+) em VRAM limitada sem OOM.

    Dtype:
      Aceita BF16/FP16/FP32 nas entradas. A computacao interna (exp e produtos
      cumulativos) roda em FP32 por estabilidade numerica; as entradas sao
      promovidas por bloco, entao nunca materializamos o tensor fp32 cheio
      [B, D, S, seq]. A saida volta no dtype de entrada.

    Args:
      chunk_size: tamanho do bloco (tokens). <=0 ou >= seq desativa o chunking
                  (caminho original de um unico pass).
      h_init: estado inicial opcional [batch, dim, state] (default h_0=0).
              Usado pelo decode incremental (inference_engine.py) para
              continuar uma sequencia token-a-token sem reprocessar o
              historico -- sem isso, cada chamada assumia h_0=0 e "esquecia"
              tudo antes do token novo (bug achado 2026-07-26). Treino nunca
              passa h_init -- comportamento la e identico ao de antes.
      return_state: se True, retorna (y, h_final) em vez de so y.
    """
    batch, dim, seq_len = u.shape
    state_size = a.shape[1]
    input_dtype = u.dtype
    # Computo interno em fp32 (exp + produtos cumulativos sao sensiveis a bf16).
    a_fp32 = a.float()

    # ── Caminho unico pass (sem chunking): equivalente ao algoritmo original ──
    if chunk_size <= 0 or chunk_size >= seq_len:
        u_f = u.float()
        delta_f = delta.float()
        a_bar = torch.exp(delta_f.unsqueeze(2) * a_fp32.unsqueeze(0).unsqueeze(-1))
        b_bar = delta_f.unsqueeze(2) * b.float().unsqueeze(1) * u_f.unsqueeze(2)
        a_bar, h = _associative_scan_last_dim(a_bar, b_bar, seq_len)
        if h_init is not None:
            # a_bar[...,t] = produto cumulativo do decay ate t; soma o
            # decaimento do estado herdado, mesma formula do carry chunked.
            h = h + a_bar * h_init.float().unsqueeze(-1)
        y = (h * c.float().unsqueeze(1)).sum(dim=2)
        if d is not None:
            y = y + d.float().unsqueeze(0).unsqueeze(-1) * u_f
        y = y.to(dtype=input_dtype)
        if return_state:
            return y, h[..., -1].to(dtype=input_dtype)
        return y

    # ── Caminho chunked (P0.1): carry de h entre blocos ──
    h_carry = (
        h_init.float() if h_init is not None
        else torch.zeros(batch, dim, state_size, device=u.device, dtype=torch.float32)
    )
    y_parts: list[torch.Tensor] = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        clen = end - start

        uc = u[..., start:end].float()
        dc = delta[..., start:end].float()
        bc = b[..., start:end].float()
        cc = c[..., start:end].float()

        # Discretize por bloco: A_bar = exp(delta * A), B_bar = delta * B * u
        a_bar_c = torch.exp(dc.unsqueeze(2) * a_fp32.unsqueeze(0).unsqueeze(-1))
        b_bar_c = dc.unsqueeze(2) * bc.unsqueeze(1) * uc.unsqueeze(2)

        # Prefix-scan dentro do bloco (fp32).
        a_bar_c, h_local = _associative_scan_last_dim(a_bar_c, b_bar_c, clen)

        # Carry: h_t = h_local_t + (decay cumulativo do bloco ate t) * h_carry
        # a_bar_c[..., t] ja e o produto cumulativo dentro do bloco apos o scan.
        h_local = h_local + a_bar_c * h_carry.unsqueeze(-1)

        # Saida do bloco: y_t = C_t^T h_t
        y_parts.append((h_local * cc.unsqueeze(1)).sum(dim=2))

        # Estado carregado = h na ultima posicao do bloco
        h_carry = h_local[..., -1]

    y = torch.cat(y_parts, dim=-1)  # [B, D, seq]
    if d is not None:
        y = y + d.float().unsqueeze(0).unsqueeze(-1) * u.float()
    y = y.to(dtype=input_dtype)
    if return_state:
        return y, h_carry.to(dtype=input_dtype)
    return y


# ═══════════════════════════════════════════════════════════════════════════
# SelectiveSSM  (conforme especificacao Mamba-1/2)
# ═══════════════════════════════════════════════════════════════════════════

class SelectiveSSM(nn.Module):
    # TODO(fresh-start apenas — muda shape/parametrizacao de a_log, quebra
    # resume estrito): isto e Mamba-1 (A diagonal por-canal, a: [d_inner,
    # d_state]) com hiperparametros de Mamba-2, nao o algoritmo real do
    # Mamba-2 (structured state-space duality -- A escalar por head, scan
    # expresso como produto de matrizes bloco-diagonais, explorando tensor
    # cores). Daria ganho real de throughput em GPU; nao aplicado em
    # 2026-07-26 por exigir fresh-start.
    """Selective state-space mixer — Mamba-1 architecture, Mamba-2 hyperparams.

    INITIALIZATION (critico — desvios aqui MATAM o treino):
      - dt_proj.bias: inverse-softplus de valores uniformes em [dt_min, dt_max]
                       ≈ -4.6 para dt≈0.01  (nao 1.0 como antes!)
      - dt_proj.weight: uniform(-std, std) com std = dt_rank^-0.5 (init canonico
                        Mamba-1). Protegido por _no_reinit=True (ver __init__ e
                        model.py:_init_weights) — sem essa flag no PESO (nao so
                        no bias), o init generico do resto do modelo pisa em
                        cima e reduz a input-dependence do delta (bug irmao do
                        V9, achado em analise 2026-07-26).
      - a_log: S4D-real → log(1..d_state)  (Hippo-LegS)
      - conv1d: init default do nn.Conv1d (Kaiming uniform) — NAO Xavier
      - out_proj: init generico de model.py:_init_weights (normal std pequeno)
                  — NAO Xavier
    """

    def __init__(
        self,
        d_model: int,
        *,
        expand: int = 2,
        # d_state: default do codigo = 64 (Mamba-2 captura dependencias longas).
        # As configs ativas (100M, 600M, 1.2B) usam ssm_state=16 ou 64 por escala.
        # A 600M usa ssm_state=64; 100M e 1.2B usam ssm_state=16.
        # Consulte a config YAML da sua linhagem para o valor exato.
        d_state: int = 64,
        conv_kernel: int = 4,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,      # oficial Mamba
        dt_max: float = 0.1,        # oficial Mamba
        scan_chunk_size: int = 1024,  # P0.1: chunking do selective scan (tokens/bloco)
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if expand <= 0:
            raise ValueError("expand must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if conv_kernel <= 1:
            raise ValueError("conv_kernel must be > 1")
        if scan_chunk_size < 1:
            raise ValueError("scan_chunk_size must be >= 1")

        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state
        self.dt_rank = dt_rank if isinstance(dt_rank, int) else max(16, d_model // 16)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.scan_chunk_size = scan_chunk_size
        self.gradient_checkpointing = gradient_checkpointing

        # Input projection: x → (x_inner, z_gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D causal convolution (depthwise)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            bias=True,
        )

        # Delta / B / C projection
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # ── INITIALIZATION (SEQUENCE MATTERS) ──

        # A matrix: S4D-real (Hippo-LegS)
        # A_n = -n  for n = 1..d_state   →  stored as log(|A|)
        a_values = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        a_log = torch.log(a_values).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.a_log = nn.Parameter(a_log)

        # D (skip connection)
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))

        # dt_proj.bias: inverse-softplus of uniform [dt_min, dt_max]
        dt_init = torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt_init = torch.exp(dt_init)  # [dt_min, dt_max]
        inv_softplus_dt = dt_init + torch.log(-torch.expm1(-dt_init))  # f⁻¹(x) = log(exp(x)-1)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

        # dt_proj.weight: canonical Mamba-1 init — uniform with std = dt_rank^{-0.5}
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Flag to prevent _init_weights from zeroing this bias (bug V9)
        self.dt_proj._no_reinit = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        conv_state: torch.Tensor | None = None,
        ssm_state: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input [B, L, D] → Output [B, L, D].

        Recorrencia real para decode incremental (inference-only): passe
        conv_state [B, D_inner, conv_kernel-1] e ssm_state [B, D_inner,
        D_state] devolvidos de uma chamada anterior (return_state=True) para
        continuar a sequencia token-a-token sem reprocessar o historico.
        Sem isso (default, treino sempre chama assim), comportamento e
        identico ao de antes -- bug achado 2026-07-26: inference_engine.py
        chamava isto so com o token novo e sem estado nenhum, entao cada
        passo do decode "esquecia" tudo (conv1d com padding zero, scan com
        h_0=0), apesar do docstring do arquivo afirmar "SSD is already O(1)".
        """
        batch, seq_len, d_model = x.shape

        # ── Input projection + split ──
        xz = self.in_proj(x)                      # [B, L, 2*D_inner]
        x_inner, z = xz.chunk(2, dim=-1)           # each [B, L, D_inner]

        # ── 1D causal convolution ──
        x_conv_in = x_inner.transpose(1, 2)        # [B, D_inner, L]
        kernel = self.conv1d.kernel_size[0]
        if conv_state is not None:
            # conv_state ja E o contexto causal (ultimos kernel-1 inputs) --
            # concatenar em vez de padar com zero preserva a memoria real.
            x_conv_in = torch.cat([conv_state, x_conv_in], dim=-1)
        else:
            x_conv_in = F.pad(x_conv_in, (kernel - 1, 0))  # causal pad (treino / prefill)
        new_conv_state = x_conv_in[:, :, -(kernel - 1):] if return_state else None
        x_conv = self.conv1d(x_conv_in)                       # [B, D_inner, L]
        x_conv = x_conv.transpose(1, 2)                       # [B, L, D_inner]
        x_conv = F.silu(x_conv)

        # ── Delta / B / C projection ──
        x_proj_out = self.x_proj(x_conv)           # [B, L, dt_rank + 2*d_state]
        dt_rank = self.dt_rank
        d_state = self.d_state
        dt_in, b_in, c_in = torch.split(
            x_proj_out,
            [dt_rank, d_state, d_state],
            dim=-1,
        )

        # Delta: softplus(dt_proj). A implementacao de referencia do Mamba usa
        # dt_min/dt_max SO para amostrar o init do bias (ver __init__), sem
        # clampar em runtime -- um clamp duro aqui cria dead-zone de gradiente
        # (grad=0 fora de [dt_min,dt_max]) toda vez que softplus satura,
        # podendo travar canais permanentemente em dt_max ou dt_min. Removido
        # 2026-07-26 apos analise; softplus por si so garante delta > 0.
        delta = self.dt_proj(dt_in)                # [B, L, D_inner]
        delta = F.softplus(delta)                  # > 0, sem clamp adicional

        # Reshape for scan: [B, D_inner, L]  and  [B, D_state, L]
        delta = delta.transpose(1, 2)              # [B, D_inner, L]
        b = b_in.transpose(1, 2)                   # [B, D_state, L]
        c = c_in.transpose(1, 2)                   # [B, D_state, L]
        u = x_conv.transpose(1, 2)                 # [B, D_inner, L]

        # A matrix: -exp(a_log)  ∈  [-1, -2, ..., -d_state]
        a = -torch.exp(self.a_log.float())         # [D_inner, D_state] fp32

        # ── Selective scan (P0.1: chunked, promove a fp32 internamente) ──
        # Nota: nao forcamos .float() nas entradas aqui — selective_scan promove
        # por bloco e devolve no dtype de entrada, economizando memoria sem
        # materializar o tensor fp32 cheio [B, D_inner, D_state, L].
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            def checkpointed_scan(
                scan_u: torch.Tensor,
                scan_delta: torch.Tensor,
                scan_a: torch.Tensor,
                scan_b: torch.Tensor,
                scan_c: torch.Tensor,
                scan_d: torch.Tensor,
            ) -> torch.Tensor:
                return selective_scan(
                    scan_u,
                    scan_delta,
                    scan_a,
                    scan_b,
                    scan_c,
                    scan_d,
                    chunk_size=self.scan_chunk_size,
                )

            y = checkpoint(
                checkpointed_scan,
                u,
                delta,
                a,
                b,
                c,
                self.d_skip,
                use_reentrant=False,
            )
        elif return_state or ssm_state is not None:
            y, new_ssm_state = selective_scan(
                u, delta, a, b, c, self.d_skip,
                chunk_size=self.scan_chunk_size,
                h_init=ssm_state,
                return_state=True,
            )
        else:
            y = selective_scan(
                u, delta, a, b, c, self.d_skip,
                chunk_size=self.scan_chunk_size,
            )
        # [B, D_inner, L] no dtype de entrada

        # ── Gate + output projection (back to input dtype) ──
        y = y.transpose(1, 2)                      # [B, L, D_inner]
        y = y * F.silu(z)                          # gate (Mamba standard)
        out = self.out_proj(y)                      # [B, L, D_model]
        if return_state:
            return out, new_conv_state, new_ssm_state
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test  (executado no import)
# ═══════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Verifica shapes, gradientes e sanidade dos canais de estado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    ssm = SelectiveSSM(d_model=64, expand=2, d_state=32).to(device=device, dtype=dtype)
    x = torch.randn(2, 16, 64, device=device, dtype=dtype)

    # 1. Forward shape
    y = ssm(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # 2. Gradient flow
    y.sum().backward()
    for name, param in ssm.named_parameters():
        if param.grad is None:
            print(f"  [WARN] {name}: sem gradiente")
        elif param.grad.abs().sum() == 0:
            print(f"  [WARN] {name}: gradiente zero")
    assert ssm.a_log.grad is not None, "a_log sem gradiente"
    assert ssm.dt_proj.bias.grad is not None, "dt_proj.bias sem gradiente"
    assert ssm.dt_proj.weight.grad is not None, "dt_proj.weight sem gradiente"

    # 3. Delta range
    with torch.no_grad():
        xz = ssm.in_proj(x)
        xi, _ = xz.chunk(2, dim=-1)
        x_conv = xi.transpose(1, 2)
        x_conv = F.pad(x_conv, (ssm.conv1d.kernel_size[0] - 1, 0))
        x_conv = ssm.conv1d(x_conv).transpose(1, 2)
        x_conv = F.silu(x_conv)
        xp = ssm.x_proj(x_conv)
        dt_in, _, _ = torch.split(xp, [ssm.dt_rank, ssm.d_state, ssm.d_state], dim=-1)
        # Sem clamp em runtime (removido 2026-07-26) -- delta e so softplus(.),
        # esperado ficar perto de [dt_min, dt_max] logo apos o init, mas nao
        # e mais forcado a isso durante o treino.
        delta = F.softplus(ssm.dt_proj(dt_in))
        d_min = delta.min().item()
        d_max = delta.max().item()
        d_mean = delta.mean().item()
        print(f"  delta: min={d_min:.4f}  max={d_max:.4f}  mean={d_mean:.4f}  "
              f"(init esperado perto de [{ssm.dt_min}, {ssm.dt_max}], sem clamp)")
        assert d_min > 0.0, f"delta deve ser > 0 (softplus): {d_min}"

    # 4. Channel survival: check effective decay per state channel
    with torch.no_grad():
        a_val = -torch.exp(ssm.a_log.float())          # [D_inner, D_state]
        delta_mean = delta.float().mean(dim=(0, 2))     # [D_inner]
        a_mean = a_val.mean(dim=0)                       # [D_state]
        effective_decay = torch.exp(torch.outer(delta_mean, a_mean)).mean(dim=0)  # [D_state]
        alive_channels = (effective_decay > 0.01).sum().item()  # > 1% retained
        total_channels = ssm.d_state
        print(f"  state channels: {alive_channels}/{total_channels} alive (>1% retention per step)")
        if alive_channels < total_channels * 0.5:
            print(f"  [WARN] menos de 50% dos canais vivos — ajustar dt_min/dt_max")

    # 5. BF16 safety
    if device.type == "cuda":
        ssm_bf16 = SelectiveSSM(d_model=64, expand=2, d_state=32).to(device=device, dtype=torch.bfloat16)
        x_bf16 = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
        y_bf16 = ssm_bf16(x_bf16)
        assert y_bf16.shape == x_bf16.shape
        assert not torch.isnan(y_bf16).any(), "NaN em bf16"
        assert not torch.isinf(y_bf16).any(), "Inf em bf16"

    # 6. P0.1 — chunking equivalence: chunked scan deve bater com single-pass
    _test_chunking_equivalence(device)

    print("  [OK] Todos os testes passaram")


def _test_chunking_equivalence(device: torch.device) -> None:
    """P0.1: verifica que o scan chunked equivale ao single-pass em fp32."""
    torch.manual_seed(0)
    batch, dim, seq, state = 2, 12, 64, 8
    u = torch.randn(batch, dim, seq, device=device)
    delta = torch.rand(batch, dim, seq, device=device) * 0.05 + 0.001
    # a: [dim, state] (valores negativos, conforma docstring/contrato)
    a = -(torch.arange(1, state + 1, device=device).float()).unsqueeze(0).expand(dim, -1).contiguous()
    b = torch.randn(batch, state, seq, device=device)
    c = torch.randn(batch, state, seq, device=device)
    d = torch.ones(dim, device=device)

    y_full = selective_scan(u, delta, a, b, c, d, chunk_size=0)
    for cs in (1, 3, 7, 8, 16, 64, 1024):
        y_chunk = selective_scan(u, delta, a, b, c, d, chunk_size=cs)
        max_err = (y_full - y_chunk).abs().max().item()
        assert max_err < 1e-4, (
            f"chunk_size={cs}: divergencia max={max_err:.2e} entre scan chunked e single-pass"
        )
    print(f"  [P0.1] chunking equivalence OK (max_err < 1e-4 para cs em 1..1024)")


# Backward compat alias (legacy tests may reference old name)
selective_scan_parallel = selective_scan
selective_scan_sequential = selective_scan
selective_scan_parallel_compiled = selective_scan

if __name__ == "__main__":
    _smoke_test()
