from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class OrganProjection:
    matrix: torch.Tensor
    source_dim: int
    target_dim: int
    seed: int
    algorithm: str = "organ-semi-orthogonal-qr-v1"

    def identity(self) -> str:
        contiguous = self.matrix.detach().cpu().contiguous()
        matrix_hash = hashlib.sha256(
            memoryview(
                contiguous.reshape(-1).view(torch.uint8).numpy()
            ).cast("B")
        ).hexdigest()
        payload = {
            "algorithm": self.algorithm,
            "matrix_hash": matrix_hash,
            "seed": self.seed,
            "shape": list(self.matrix.shape),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class OrganTensorGrowth:
    key: str
    method: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    finite: bool


def derive_organ_projection(
    *,
    source_dim: int,
    target_dim: int,
    seed: int,
) -> OrganProjection:
    if source_dim < 1 or target_dim < source_dim:
        raise ValueError("organ projection requires target_dim >= source_dim")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    seed_matrix = torch.randn(
        target_dim,
        source_dim,
        generator=generator,
        dtype=torch.float32,
    )
    matrix, _ = torch.linalg.qr(seed_matrix, mode="reduced")
    pivots = matrix.abs().argmax(dim=0)
    values = matrix[pivots, torch.arange(source_dim)]
    signs = torch.where(
        values < 0,
        -torch.ones_like(values),
        torch.ones_like(values),
    )
    matrix = (matrix * signs[None, :]).contiguous()
    return OrganProjection(
        matrix=matrix,
        source_dim=source_dim,
        target_dim=target_dim,
        seed=seed,
    )


def expand_square(
    source: torch.Tensor,
    projection: OrganProjection,
    *,
    complement: str,
) -> torch.Tensor:
    if tuple(source.shape) != (
        projection.source_dim,
        projection.source_dim,
    ):
        raise ValueError("square organ tensor does not match source dimension")
    if complement not in {"zero", "identity"}:
        raise ValueError("organ complement must be zero or identity")
    q = projection.matrix.to(device=source.device)
    grown = q @ source.float() @ q.T
    if complement == "identity":
        projector = q @ q.T
        grown = grown + torch.eye(
            projection.target_dim,
            device=source.device,
        ) - projector
    return grown


def expand_linear(
    source: torch.Tensor,
    *,
    output_projection: OrganProjection | None,
    input_projection: OrganProjection | None,
) -> torch.Tensor:
    if source.ndim != 2:
        raise ValueError("linear organ tensor must be rank-2")
    result = source.float()
    if output_projection is not None:
        if result.shape[0] != output_projection.source_dim:
            raise ValueError("linear output dimension does not match projection")
        result = output_projection.matrix.to(result.device) @ result
    if input_projection is not None:
        if result.shape[1] != input_projection.source_dim:
            raise ValueError("linear input dimension does not match projection")
        result = result @ input_projection.matrix.to(result.device).T
    return result


def verify_organ_subspace(
    source: torch.Tensor,
    grown: torch.Tensor,
    projection: OrganProjection,
) -> float:
    source_inputs = torch.eye(
        projection.source_dim,
        dtype=torch.float32,
        device=source.device,
    )
    embedded_inputs = source_inputs @ projection.matrix.to(source.device).T
    expected = (source_inputs @ source.float().T) @ projection.matrix.to(
        source.device
    ).T
    actual = embedded_inputs @ grown.float().T
    cosine = torch.nn.functional.cosine_similarity(
        expected.reshape(1, -1),
        actual.reshape(1, -1),
        dim=1,
        eps=1e-12,
    )
    return float(cosine.item())


def grow_organ_state(
    source_state: Mapping[str, torch.Tensor],
    target_template: Mapping[str, torch.Tensor],
    projection: OrganProjection,
    *,
    identity_transition_keys: frozenset[str] = frozenset(),
    zero_gate_suffixes: tuple[str, ...] = (
        "residual_gate",
        "external_gate",
    ),
) -> tuple[dict[str, torch.Tensor], tuple[OrganTensorGrowth, ...]]:
    result = {
        key: value.detach().clone()
        for key, value in target_template.items()
    }
    reports: list[OrganTensorGrowth] = []
    for key, source in source_state.items():
        target = target_template.get(key)
        if target is None:
            continue
        source_shape = tuple(source.shape)
        target_shape = tuple(target.shape)
        method: str
        if source_shape == target_shape:
            grown = source.detach().clone()
            method = "copy"
        elif (
            source.ndim == 2
            and source_shape
            == (projection.source_dim, projection.source_dim)
            and target_shape
            == (projection.target_dim, projection.target_dim)
        ):
            complement = (
                "identity" if key in identity_transition_keys else "zero"
            )
            grown = expand_square(
                source,
                projection,
                complement=complement,
            )
            method = f"square_{complement}"
        elif (
            source.ndim == 2
            and source_shape[0] == projection.source_dim
            and target_shape
            == (projection.target_dim, source_shape[1])
        ):
            grown = expand_linear(
                source,
                output_projection=projection,
                input_projection=None,
            )
            method = "linear_output"
        elif (
            source.ndim == 2
            and source_shape[1] == projection.source_dim
            and target_shape
            == (source_shape[0], projection.target_dim)
        ):
            grown = expand_linear(
                source,
                output_projection=None,
                input_projection=projection,
            )
            method = "linear_input"
        elif (
            source_shape
            and source_shape[-1] == projection.source_dim
            and target_shape
            == (*source_shape[:-1], projection.target_dim)
        ):
            grown = source.float() @ projection.matrix.to(source.device).T
            method = "vector_last_dim"
        else:
            raise ValueError(
                f"unsupported organ tensor growth for {key}: "
                f"source={source_shape}; target={target_shape}"
            )
        grown = grown.to(dtype=target.dtype, device=target.device)
        if key.endswith(zero_gate_suffixes):
            grown = torch.zeros_like(target)
            method = "zero_authority_gate"
        finite = bool(torch.isfinite(grown).all())
        if not finite:
            raise ValueError(f"grown organ tensor is non-finite: {key}")
        result[key] = grown
        reports.append(
            OrganTensorGrowth(
                key=key,
                method=method,
                source_shape=source_shape,
                target_shape=target_shape,
                finite=finite,
            )
        )
    return result, tuple(reports)
