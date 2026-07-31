from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransitionBatch:
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    holdout_inputs: torch.Tensor
    holdout_targets: torch.Tensor

    def __post_init__(self) -> None:
        if self.train_inputs.shape != self.train_targets.shape:
            raise ValueError("SSD train transition shapes differ")
        if self.holdout_inputs.shape != self.holdout_targets.shape:
            raise ValueError("SSD holdout transition shapes differ")
        if self.train_inputs.ndim != 3 or self.holdout_inputs.ndim != 3:
            raise ValueError("SSD transitions must be [batch, sequence, hidden]")
        if self.train_inputs.shape[-1] != self.holdout_inputs.shape[-1]:
            raise ValueError("SSD train/holdout hidden dimensions differ")


@dataclass(frozen=True)
class SSDFitConfig:
    steps: int
    learning_rate: float
    seed: int
    weight_decay: float = 0.0
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("SSD fit steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("SSD fit learning_rate must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("SSD fit gradient_clip must be positive")


@dataclass(frozen=True)
class SSDFitReport:
    seed: int
    steps: int
    initial_train_mse: float
    train_mse: float
    random_holdout_mse: float
    holdout_mse: float
    finite: bool
    parameter_hash: str


def _mse(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (model(inputs) - targets).square().mean()


def _parameter_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().to(device="cpu").contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(contiguous.dtype).encode("ascii"))
        hasher.update(str(tuple(contiguous.shape)).encode("ascii"))
        hasher.update(
            memoryview(
                contiguous.reshape(-1).view(torch.uint8).numpy()
            ).cast("B")
        )
    return hasher.hexdigest()


def fit_ssd_block(
    candidate: nn.Module,
    transitions: TransitionBatch,
    config: SSDFitConfig,
    baseline: nn.Module | None = None,
) -> SSDFitReport:
    """Ajusta o bloco SSD contra transicoes do doador.

    ``baseline`` e o modulo de init aleatorio pareado contra o qual o gate
    de holdout compara. Sem ele o gate compara o candidato contra ele
    mesmo pre-treino, o que nao e um controle -- ver observacao abaixo.
    """
    torch.manual_seed(config.seed)
    train_inputs = transitions.train_inputs.detach().float()
    train_targets = transitions.train_targets.detach().float()
    holdout_inputs = transitions.holdout_inputs.detach().float()
    holdout_targets = transitions.holdout_targets.detach().float()
    candidate.float()
    candidate.train()

    with torch.no_grad():
        initial_train_mse = float(
            _mse(candidate, train_inputs, train_targets).item()
        )
        # OBSERVACAO (2026-07-29): sem `baseline`, isto mede o proprio
        # candidato antes do treino -- nao um modulo aleatorio. No call site
        # de producao o candidato ja chega mutado in-place por
        # _initialize_ssd, entao "random_holdout_mse" era o erro da
        # inicializacao heuristica, e o gate da linha ~155 so verificava se
        # 20 passos de Adam reduziram alguma coisa. Isso explica o paradoxo
        # do build de 2026-07-28: os 12 blocos batiam o "random" interno por
        # 3 a 6 ordens de grandeza (ate 179000x) enquanto o modelo inteiro
        # ficava PIOR que o random real do harness de calibracao -- eram
        # dois baselines estruturalmente diferentes, e o interno estava
        # contaminado pela propria inicializacao ruim que deveria julgar.
        reference = candidate if baseline is None else baseline.float().eval()
        random_holdout_mse = float(
            _mse(reference, holdout_inputs, holdout_targets).item()
        )
    optimizer = torch.optim.AdamW(
        candidate.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    finite = True
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _mse(candidate, train_inputs, train_targets)
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in candidate.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(
            not torch.isfinite(gradient).all() for gradient in gradients
        ):
            finite = False
            break
        torch.nn.utils.clip_grad_norm_(
            candidate.parameters(),
            config.gradient_clip,
        )
        optimizer.step()

    candidate.eval()
    with torch.no_grad():
        train_mse = float(
            _mse(candidate, train_inputs, train_targets).item()
        )
        holdout_mse = float(
            _mse(candidate, holdout_inputs, holdout_targets).item()
        )
    finite = bool(
        finite
        and torch.isfinite(torch.tensor(train_mse))
        and torch.isfinite(torch.tensor(holdout_mse))
        and all(
            torch.isfinite(tensor).all()
            for tensor in candidate.state_dict().values()
        )
    )
    if not finite:
        raise ValueError("SSD system identification produced non-finite state")
    if holdout_mse >= random_holdout_mse:
        raise ValueError(
            "SSD system identification did not beat paired random baseline: "
            f"candidate={holdout_mse}; random={random_holdout_mse}"
        )
    return SSDFitReport(
        seed=config.seed,
        steps=config.steps,
        initial_train_mse=initial_train_mse,
        train_mse=train_mse,
        random_holdout_mse=random_holdout_mse,
        holdout_mse=holdout_mse,
        finite=finite,
        parameter_hash=_parameter_hash(candidate),
    )
