from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch


PROJECTION_ALGORITHM = "coherent-residual-svd-v1"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(
        memoryview(
            contiguous.reshape(-1).view(torch.uint8).numpy()
        ).cast("B")
    ).hexdigest()


def _calibration_sha256(*tensors: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            [str(contiguous.dtype), *contiguous.shape],
            separators=(",", ":"),
        ).encode("ascii")
        hasher.update(len(metadata).to_bytes(8, "big"))
        hasher.update(metadata)
        view = memoryview(
            contiguous.reshape(-1).view(torch.uint8).numpy()
        ).cast("B")
        hasher.update(len(view).to_bytes(8, "big"))
        hasher.update(view)
    return hasher.hexdigest()


@dataclass(frozen=True)
class HiddenProjection:
    matrix: torch.Tensor
    singular_values: torch.Tensor
    explained_energy: float
    calibration_digest: str
    seed: int
    algorithm: str
    output_hash: str

    @property
    def source_dim(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.matrix.shape[0])

    def identity(self) -> str:
        payload = {
            "algorithm": self.algorithm,
            "calibration_digest": self.calibration_digest,
            "explained_energy": float(self.explained_energy).hex(),
            "output_hash": self.output_hash,
            "seed": self.seed,
            "shape": list(self.matrix.shape),
            "singular_values_hash": _tensor_sha256(self.singular_values),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def derive_hidden_projection(
    embedding_samples: torch.Tensor,
    activation_samples: torch.Tensor,
    *,
    target_dim: int,
    seed: int,
) -> HiddenProjection:
    if embedding_samples.ndim != 2 or activation_samples.ndim != 2:
        raise ValueError("projection samples must be rank-2 matrices")
    if embedding_samples.shape[1] != activation_samples.shape[1]:
        raise ValueError("embedding and activation dimensions differ")
    source_dim = int(embedding_samples.shape[1])
    if not 0 < target_dim <= source_dim:
        raise ValueError("target_dim must be in (0, source_dim]")

    calibration_digest = _calibration_sha256(
        embedding_samples,
        activation_samples,
    )
    samples = torch.cat(
        (embedding_samples.float(), activation_samples.float()),
        dim=0,
    )
    samples = samples - samples.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(
        samples,
        full_matrices=False,
    )
    matrix = right_vectors[:target_dim].contiguous()
    pivot_indices = matrix.abs().argmax(dim=1)
    pivot_values = matrix[
        torch.arange(target_dim, device=matrix.device),
        pivot_indices,
    ]
    signs = torch.where(
        pivot_values < 0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    matrix = (matrix * signs[:, None]).contiguous()
    energy = singular_values.square()
    explained_energy = float(
        energy[:target_dim].sum().div(energy.sum().clamp_min(1e-30)).item()
    )
    return HiddenProjection(
        matrix=matrix,
        singular_values=singular_values[:target_dim].contiguous(),
        explained_energy=explained_energy,
        calibration_digest=calibration_digest,
        seed=int(seed),
        algorithm=PROJECTION_ALGORITHM,
        output_hash=_tensor_sha256(matrix),
    )


def project_linear(
    weight: torch.Tensor,
    output_space: HiddenProjection,
    input_space: HiddenProjection,
) -> torch.Tensor:
    expected = (output_space.source_dim, input_space.source_dim)
    if tuple(weight.shape) != expected:
        raise ValueError(
            f"linear weight shape mismatch: expected={expected}; "
            f"actual={tuple(weight.shape)}"
        )
    return (
        output_space.matrix
        @ weight.float()
        @ input_space.matrix.T
    )


def project_norm(
    weight: torch.Tensor,
    projection: HiddenProjection,
) -> torch.Tensor:
    if tuple(weight.shape) != (projection.source_dim,):
        raise ValueError("norm weight does not match projection source")
    transformed = (
        projection.matrix
        @ torch.diag(weight.float())
        @ projection.matrix.T
    )
    return transformed.diagonal().contiguous()
