from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CoupledMLP:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def __post_init__(self) -> None:
        if self.gate.ndim != 2 or self.up.ndim != 2 or self.down.ndim != 2:
            raise ValueError("MLP gate/up/down must be rank-2")
        if self.gate.shape != self.up.shape:
            raise ValueError("MLP gate and up shapes differ")
        if self.down.shape != (self.gate.shape[1], self.gate.shape[0]):
            raise ValueError(
                "MLP down shape must couple every gate/up neuron"
            )

    @property
    def neurons(self) -> int:
        return int(self.gate.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.gate.shape[1])


@dataclass(frozen=True)
class ExpertAssignment:
    source_neurons: tuple[tuple[int, ...], ...]
    target_widths: tuple[int, ...]
    shared_experts: int
    fine_experts: int
    activation_centroids: torch.Tensor


@dataclass(frozen=True)
class ExpertState:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    basis: torch.Tensor
    reconstruction_mse: torch.Tensor
    source_neurons: tuple[int, ...]


@dataclass(frozen=True)
class RouterFit:
    weight: torch.Tensor
    initial_loss: float
    final_loss: float
    accuracy: float


def _cosine_kmeans(
    features: torch.Tensor,
    clusters: int,
    *,
    seed: int,
    iterations: int = 30,
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] < clusters:
        raise ValueError("not enough neuron signatures for requested experts")
    normalized = F.normalize(features.float(), dim=1, eps=1e-12)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(
        normalized.shape[0],
        generator=generator,
    )[:clusters]
    centroids = normalized[initial].clone()
    assignments = torch.zeros(
        normalized.shape[0],
        dtype=torch.long,
    )
    for _ in range(iterations):
        similarities = normalized @ centroids.T
        assignments = similarities.argmax(dim=1)
        counts = torch.bincount(assignments, minlength=clusters)
        for empty in torch.where(counts == 0)[0].tolist():
            assigned_similarity = similarities[
                torch.arange(assignments.numel()),
                assignments,
            ]
            movable = counts[assignments] > 1
            if not movable.any():
                raise ValueError("cannot repair empty k-means cluster")
            candidates = torch.where(
                movable,
                assigned_similarity,
                torch.full_like(assigned_similarity, float("inf")),
            )
            replacement = int(candidates.argmin().item())
            previous = int(assignments[replacement].item())
            assignments[replacement] = int(empty)
            counts[previous] -= 1
            counts[empty] += 1
        updated = []
        for cluster in range(clusters):
            members = normalized[assignments == cluster]
            updated.append(
                F.normalize(members.mean(dim=0), dim=0, eps=1e-12)
            )
        next_centroids = torch.stack(updated)
        if torch.equal(
            assignments,
            (normalized @ next_centroids.T).argmax(dim=1),
        ):
            centroids = next_centroids
            break
        centroids = next_centroids
    return assignments


def cluster_coupled_neurons(
    mlp: CoupledMLP,
    activation_signatures: torch.Tensor,
    *,
    shared_experts: int,
    fine_experts: int,
    expert_width: int,
    seed: int,
) -> ExpertAssignment:
    if activation_signatures.ndim != 2:
        raise ValueError("activation_signatures must be [samples, neurons]")
    if activation_signatures.shape[1] != mlp.neurons:
        raise ValueError("activation signatures do not cover every neuron")
    total_experts = shared_experts + fine_experts
    if min(shared_experts, fine_experts, expert_width) < 1:
        raise ValueError("expert counts and width must be positive")
    if mlp.neurons < total_experts:
        raise ValueError("every expert requires at least one source neuron")

    signatures = activation_signatures.detach().float().cpu()
    magnitude = signatures.abs().mean(dim=0)
    variation = signatures.abs().std(dim=0).clamp_min(1e-6)
    ubiquity = magnitude / variation
    shared_count = max(
        shared_experts,
        round(mlp.neurons * shared_experts / total_experts),
    )
    shared_count = min(
        shared_count,
        mlp.neurons - fine_experts,
    )
    ranked = sorted(
        range(mlp.neurons),
        key=lambda index: (-float(ubiquity[index]), index),
    )
    shared_indices = ranked[:shared_count]
    fine_indices = ranked[shared_count:]

    groups: list[tuple[int, ...]] = []
    for expert in range(shared_experts):
        group = tuple(sorted(shared_indices[expert::shared_experts]))
        if not group:
            raise ValueError("shared expert received no donor neurons")
        groups.append(group)

    fine_features = signatures[:, fine_indices].T.contiguous()
    fine_assignment = _cosine_kmeans(
        fine_features,
        fine_experts,
        seed=seed,
    )
    for expert in range(fine_experts):
        group = tuple(
            sorted(
                fine_indices[position]
                for position in torch.where(
                    fine_assignment == expert
                )[0].tolist()
            )
        )
        if not group:
            raise ValueError("fine expert received no donor neurons")
        groups.append(group)

    flattened = sorted(index for group in groups for index in group)
    if flattened != list(range(mlp.neurons)):
        raise AssertionError("coupled-neuron coverage is incomplete")
    centroids = torch.stack(
        [signatures[:, list(group)].mean(dim=1) for group in groups]
    )
    return ExpertAssignment(
        source_neurons=tuple(groups),
        target_widths=(expert_width,) * total_experts,
        shared_experts=shared_experts,
        fine_experts=fine_experts,
        activation_centroids=centroids,
    )


def _coupled_basis(features: torch.Tensor, target_width: int) -> torch.Tensor:
    source_width = int(features.shape[0])
    if target_width <= source_width:
        left, _, _ = torch.linalg.svd(
            features.float(),
            full_matrices=False,
        )
        return left[:, :target_width].T.contiguous()

    # Ver OBSERVACAO em assembly.py:_fast_coupled_expert. Na expansao o unico
    # basis que preserva a funcao SwiGLU e o scatter: a simetria dos neuronios
    # ocultos e o grupo de permutacao, nao o ortogonal. O frame DCT+QR anterior
    # era ortonormal -- logo reconstruia o PESO exatamente -- e ainda assim
    # misturava os neuronios antes da silu, alterando a FUNCAO.
    return torch.eye(target_width, source_width, dtype=torch.float32)


def build_expert_state(
    mlp: CoupledMLP,
    source_neurons: tuple[int, ...],
    *,
    target_width: int,
) -> ExpertState:
    if not source_neurons:
        raise ValueError("expert source_neurons must not be empty")
    indices = torch.tensor(source_neurons, dtype=torch.long)
    gate = mlp.gate.float()[indices]
    up = mlp.up.float()[indices]
    down = mlp.down.float()[:, indices]
    features = torch.cat((gate, up, down.T), dim=1)
    basis = _coupled_basis(features, target_width)
    target_gate = basis @ gate
    target_up = basis @ up
    target_down = down @ basis.T
    reconstructed = basis.T @ (basis @ features)
    mse = (reconstructed - features).square().mean()
    return ExpertState(
        gate=target_gate,
        up=target_up,
        down=target_down,
        basis=basis,
        reconstruction_mse=mse,
        source_neurons=source_neurons,
    )


def fit_router(
    activation_inputs: torch.Tensor,
    ownership: torch.Tensor,
    *,
    num_experts: int,
    steps: int = 100,
    learning_rate: float = 1e-2,
) -> RouterFit:
    if activation_inputs.ndim != 2 or ownership.ndim != 1:
        raise ValueError("router inputs/ownership have invalid ranks")
    if activation_inputs.shape[0] != ownership.shape[0]:
        raise ValueError("router inputs and ownership lengths differ")
    if ownership.min().item() < 0 or ownership.max().item() >= num_experts:
        raise ValueError("router ownership is outside expert range")

    inputs = activation_inputs.detach().float()
    labels = ownership.detach().long()
    centroids = []
    for expert in range(num_experts):
        members = inputs[labels == expert]
        if members.numel() == 0:
            raise ValueError(f"router expert {expert} has no ownership samples")
        centroids.append(F.normalize(members.mean(dim=0), dim=0, eps=1e-12))
    weight = torch.nn.Parameter(torch.stack(centroids))
    optimizer = torch.optim.AdamW([weight], lr=learning_rate)
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(inputs @ weight.T, labels).item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(inputs @ weight.T, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = inputs @ weight.T
        final_loss = float(F.cross_entropy(logits, labels).item())
        accuracy = float(logits.argmax(dim=1).eq(labels).float().mean().item())
    return RouterFit(
        weight=weight.detach().clone(),
        initial_loss=initial_loss,
        final_loss=final_loss,
        accuracy=accuracy,
    )
