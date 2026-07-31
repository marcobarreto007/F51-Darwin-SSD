from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


@torch.no_grad()
def zeropower_via_newton_schulz(
    matrix: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the polar factor of a 2-D update without an SVD.

    The quintic iteration is the Muon-style orthogonalization primitive.  CPU
    tests stay fp32; CUDA BF16 inputs remain BF16 to avoid a second full-size
    fp32 work buffer.
    """
    if matrix.ndim != 2:
        raise ValueError("zeropower_via_newton_schulz requires a 2-D tensor")
    if steps < 1:
        raise ValueError("steps must be positive")
    original_dtype = matrix.dtype
    work_dtype = (
        torch.float32 if matrix.device.type == "cpu" else matrix.dtype
    )
    x = matrix.to(dtype=work_dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / x.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    if transposed:
        x = x.mT
    return x.to(dtype=original_dtype)


class DAEHybridOptimizer(torch.optim.Optimizer):
    """Darwin spectral-matrix + tensor-scalar adaptive optimizer.

    Matrix parameters in spectral groups use an orthogonalized momentum
    direction.  Embeddings, heads, vectors and scalar parameters use one RMS
    accumulator per tensor rather than AdamW's parameter-sized second moment.
    """

    identity = "dae_hybrid_v1"

    def __init__(
        self,
        params,
        *,
        lr: float = 1.5e-4,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        beta2: float = 0.99,
        eps: float = 1e-8,
        ns_steps: int = 5,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
            eps=eps,
            ns_steps=ns_steps,
            spectral=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Validate the complete update before mutating any parameter.
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is not None and not bool(
                    torch.isfinite(gradient).all().item()
                ):
                    raise FloatingPointError("DAEHybridOptimizer: non-finite gradient")

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum_beta = float(group["momentum"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            spectral_group = bool(group.get("spectral", True))
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state: dict[str, Any] = self.state[parameter]
                momentum = state.get("momentum")
                if momentum is None:
                    momentum = torch.zeros_like(parameter)
                    state["momentum"] = momentum
                    state["step"] = torch.zeros(
                        (), device=parameter.device, dtype=torch.long
                    )
                state["step"].add_(1)
                grad = gradient.to(dtype=parameter.dtype)
                momentum.mul_(momentum_beta).add_(grad, alpha=1.0 - momentum_beta)

                if spectral_group and parameter.ndim >= 2 and min(parameter.shape) >= 2:
                    rows = parameter.shape[0]
                    columns = parameter.numel() // rows
                    update = zeropower_via_newton_schulz(
                        momentum.reshape(rows, columns),
                        steps=int(group["ns_steps"]),
                        eps=eps,
                    ).reshape_as(parameter)
                    update = update * math.sqrt(max(1.0, rows / columns))
                else:
                    rms = state.get("rms")
                    if rms is None:
                        rms = torch.zeros(
                            (), device=parameter.device, dtype=torch.float32
                        )
                        state["rms"] = rms
                    mean_square = gradient.float().square().mean()
                    rms.mul_(beta2).add_(mean_square, alpha=1.0 - beta2)
                    update = momentum / rms.sqrt().clamp_min(eps).to(
                        dtype=momentum.dtype
                    )

                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)
        return loss


def build_optimizer(
    name: str,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build an optimizer with explicit, checkpointable identity."""
    items = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    normalized = name.strip().lower()
    if normalized == "adamw":
        parameters = [parameter for _, parameter in items]
        fused = bool(parameters) and all(
            parameter.device.type == "cuda" for parameter in parameters
        )
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            fused=fused,
        )
    if normalized != "dae_hybrid":
        raise ValueError(f"unsupported optimizer: {name}")

    adaptive_markers = (
        "embedding", "token_embed", "lm_head", "output_head", "norm", "bias"
    )
    spectral: list[torch.nn.Parameter] = []
    adaptive: list[torch.nn.Parameter] = []
    for parameter_name, parameter in items:
        is_matrix = parameter.ndim >= 2 and min(parameter.shape) >= 2
        use_adaptive = any(marker in parameter_name.lower() for marker in adaptive_markers)
        (spectral if is_matrix and not use_adaptive else adaptive).append(parameter)
    groups = []
    if spectral:
        groups.append({"params": spectral, "spectral": True})
    if adaptive:
        groups.append({"params": adaptive, "spectral": False})
    return DAEHybridOptimizer(groups, lr=lr, weight_decay=weight_decay)
