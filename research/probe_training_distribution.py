#!/usr/bin/env python
"""Probe checkpoint loss on the exact token stream used by organism training.

This diagnostic exists to prevent two invalid comparisons:

* evaluating a sequentially trained checkpoint on a uniformly random corpus
  position and comparing that loss with the live training panel;
* feeding ``DarwinXOutput.hidden_states`` back through ``lm_head``.  That
  public tensor is Spider-Sense gated, while the real logits are produced
  from the ungated ``hidden_for_logits`` tensor.

The probe reconstructs batches with :class:`CausalLMDataLoader`, validates
their SHA-256 digest against the causal ledger when available, and captures
the exact input to ``lm_head`` with a forward pre-hook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.data import CausalLMDataLoader
from f51_darwin.darwin_x import (
    DarwinXConfig,
    DarwinXModel,
    migrate_mutational_state_for_load,
)
from f51_darwin.darwin_x_core.state import _cross_entropy_sum, _stable_cross_entropy


@dataclass(frozen=True)
class ProbeRequest:
    step: int
    cycle: int
    expected_digest: str | None = None


def parse_probe_request(raw: str) -> ProbeRequest:
    """Parse ``STEP:CYCLE[:EXPECTED_SHA256]``."""

    fields = raw.split(":")
    if len(fields) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "probe must use STEP:CYCLE[:EXPECTED_SHA256]"
        )
    try:
        step = int(fields[0])
        cycle = int(fields[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step and cycle must be integers") from exc
    if step < 0 or cycle < 0:
        raise argparse.ArgumentTypeError("step and cycle must be non-negative")
    expected = fields[2].lower() if len(fields) == 3 else None
    if expected is not None and (
        len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise argparse.ArgumentTypeError("expected digest must be a SHA-256 hex digest")
    return ProbeRequest(step=step, cycle=cycle, expected_digest=expected)


def parse_non_negative_float(raw: str) -> float:
    """Parse an explicit, finite, non-negative CLI threshold."""

    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return value


def contextual_signal_metrics(rank_results: dict[str, float]) -> dict[str, float]:
    """Derive auditable CE penalties for position-specific context.

    Positive penalties mean the aligned full hidden state beats the indicated
    control.  ``contextual_gain`` is conservative: both the rank-0 and reversed
    alignment controls must be worse than the aligned representation.
    """

    required = ("ce_full_sample", "ce_rank0", "ce_reversed")
    missing = [key for key in required if key not in rank_results]
    if missing:
        raise ValueError(
            "contextual signal metrics require " + ", ".join(missing)
        )
    values = {key: float(rank_results[key]) for key in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("contextual signal CE values must be finite")
    rank0_penalty = values["ce_rank0"] - values["ce_full_sample"]
    alignment_break_penalty = (
        values["ce_reversed"] - values["ce_full_sample"]
    )
    return {
        "rank0_penalty": rank0_penalty,
        "alignment_break_penalty": alignment_break_penalty,
        "contextual_gain": min(rank0_penalty, alignment_break_penalty),
    }


def contextual_signal_passes(
    rank_results: dict[str, float],
    minimum_gain: float,
) -> bool:
    """Return whether both contextual controls clear the explicit CE delta."""

    if not math.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("minimum contextual gain must be finite and non-negative")
    metrics = contextual_signal_metrics(rank_results)
    return metrics["contextual_gain"] >= minimum_gain


def contextual_gate_exit_code(failed_steps: Iterable[int]) -> int:
    """Map any rejected rank probe to a process-level acceptance failure."""

    return 1 if any(True for _ in failed_steps) else 0


def infer_tokens_per_step(training_state: dict[str, object]) -> int:
    """Infer ``batch_size * block_size`` from a real checkpoint."""

    step = int(training_state.get("step", 0))
    seen = int(training_state.get("train_tokens_seen", 0))
    if step <= 0 or seen <= 0 or seen % step:
        raise ValueError(
            "checkpoint cannot infer tokens/step exactly from "
            f"step={step}, train_tokens_seen={seen}"
        )
    return seen // step


def loader_seed(base_seed: int, cycle: int) -> int:
    """Mirror ``training.py``: ``seed=cfg.seed + self.cycle``."""

    return int(base_seed) + int(cycle)


def tensor_digest(batch: torch.Tensor) -> str:
    return hashlib.sha256(batch.detach().cpu().numpy().tobytes()).hexdigest()


def load_token_stream(
    checkpoint: dict[str, object],
    explicit_path: Path | None,
) -> tuple[np.memmap, Path]:
    token_source = checkpoint.get("token_source")
    if not isinstance(token_source, dict):
        raise ValueError("checkpoint is missing token_source metadata")
    path = explicit_path or Path(str(token_source.get("path", "")))
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"token stream not found: {path}")
    stat = path.stat()
    expected_size = int(token_source.get("size_bytes", -1))
    expected_count = int(token_source.get("token_count", -1))
    if stat.st_size != expected_size:
        raise ValueError(
            f"token stream size mismatch: checkpoint={expected_size}, disk={stat.st_size}"
        )
    if stat.st_size % np.dtype(np.int32).itemsize:
        raise ValueError("token stream is not aligned int32")
    token_ids = np.memmap(path, dtype=np.int32, mode="r")
    if len(token_ids) != expected_count:
        raise ValueError(
            f"token count mismatch: checkpoint={expected_count}, disk={len(token_ids)}"
        )
    return token_ids, path


def make_batch(
    token_ids: np.memmap,
    *,
    step: int,
    cycle: int,
    block_size: int,
    batch_size: int,
    base_seed: int,
    sampler_mode: str = "sequential",
    sampler_version: int = 1,
) -> tuple[torch.Tensor, int]:
    effective_seed = (
        int(base_seed)
        if sampler_mode == "permuted_blocks"
        else loader_seed(base_seed, cycle)
    )
    loader = CausalLMDataLoader(
        token_ids,
        block_size=block_size,
        batch_size=batch_size,
        seed=effective_seed,
        sampler_mode=sampler_mode,
        sampler_version=sampler_version,
        device="cpu",
    )
    loader.set_step(step)
    return loader.next_batch(), effective_seed


def ledger_digest_for_step(
    ledger_paths: Iterable[Path],
    *,
    step: int,
    cycle: int,
) -> str | None:
    matches: set[str] = set()
    for path in ledger_paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_step = event.get("step") or {}
                if (
                    event.get("event_type") == "STEP_OUTCOME"
                    and int(event_step.get("optimizer_step", -1)) == step
                    and int(event_step.get("cycle", -1)) == cycle
                ):
                    digest = str(event_step.get("batch_digest", "")).lower()
                    if digest:
                        matches.add(digest)
    if len(matches) > 1:
        raise ValueError(
            f"ledger has conflicting batch digests for step={step}, cycle={cycle}: "
            f"{sorted(matches)}"
        )
    return next(iter(matches), None)


def load_model(
    checkpoint: dict[str, object],
    *,
    dual_gpu: bool,
) -> tuple[DarwinXModel, torch.device]:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing embedded model config")
    config = DarwinXConfig.from_mapping(raw_config)
    model = DarwinXModel(config)
    state = {
        key.replace("_orig_mod.", ""): value
        for key, value in dict(checkpoint["model_state_dict"]).items()
    }
    migrate_mutational_state_for_load(state, model)
    model.load_state_dict(state, strict=True)
    model.load_heartbeat_state_dict(checkpoint.get("heartbeat_state"))

    if torch.cuda.is_available():
        if dual_gpu and torch.cuda.device_count() >= 2:
            model.enable_dual_gpu(gpu0=0, gpu1=1)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:0")
            model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)
    model.train()
    return model, device


def lm_head_ce(
    model: DarwinXModel,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute CE from hidden vectors without materializing all logits."""

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    valid = 0
    device_type = hidden.device.type
    amp_enabled = device_type == "cuda"
    for offset in range(0, flat_hidden.shape[0], chunk_size):
        stop = min(offset + chunk_size, flat_hidden.shape[0])
        with torch.amp.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model.lm_head(flat_hidden[offset:stop])
        target = flat_labels[offset:stop]
        total = total + _cross_entropy_sum(logits, target)
        valid += int(target.ne(-100).sum().item())
    if valid == 0:
        raise ValueError("probe labels contain no valid tokens")
    return total / valid


@torch.no_grad()
def functional_rank_probe(
    model: DarwinXModel,
    hidden_for_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ranks: tuple[int, ...],
    sample_tokens: int,
) -> dict[str, float]:
    """Measure how much next-token signal survives rank-k reconstruction."""

    flat = hidden_for_logits[:, :-1].reshape(-1, hidden_for_logits.shape[-1]).float()
    flat_labels = labels[:, 1:].reshape(-1)
    count = min(sample_tokens, flat.shape[0])
    positions = torch.linspace(
        0,
        flat.shape[0] - 1,
        steps=count,
        device=flat.device,
    ).round().long()
    sampled = flat.index_select(0, positions)
    sampled_labels = flat_labels.index_select(0, positions)

    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(explained * explained.clamp_min(1e-12).log()).sum()
    )

    results = {
        "effective_rank": float(effective_rank.cpu()),
        "variance_top1": float(explained[0].cpu()),
        "variance_top10": float(explained[:10].sum().cpu()),
        "ce_full_sample": float(
            lm_head_ce(model, sampled, sampled_labels).detach().cpu()
        ),
    }
    # Alignment controls.  A rank-0 vector computed from the same sequence can
    # carry that sequence's aggregate token distribution, so rank-0 alone
    # cannot prove whether position-specific information is being used.
    # Reversal and circular shifts preserve the exact hidden-vector multiset
    # while breaking hidden<->next-token alignment.
    results["ce_reversed"] = float(
        lm_head_ce(model, sampled.flip(0), sampled_labels).detach().cpu()
    )
    for shift in (1, 17, max(1, count // 2)):
        results[f"ce_roll{shift}"] = float(
            lm_head_ce(
                model,
                sampled.roll(shifts=shift, dims=0),
                sampled_labels,
            )
            .detach()
            .cpu()
        )
    results["ce_zero"] = float(
        lm_head_ce(model, torch.zeros_like(sampled), sampled_labels)
        .detach()
        .cpu()
    )
    centered_sample = sampled - mean
    results["ce_rank0"] = float(
        lm_head_ce(
            model,
            mean.expand_as(sampled),
            sampled_labels,
        )
        .detach()
        .cpu()
    )
    for rank in ranks:
        if rank == 0:
            continue
        basis = vh[: min(rank, vh.shape[0])]
        reconstructed = mean + (centered_sample @ basis.T) @ basis
        results[f"ce_rank{rank}"] = float(
            lm_head_ce(model, reconstructed, sampled_labels).detach().cpu()
        )
    results.update(contextual_signal_metrics(results))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-bin", type=Path)
    parser.add_argument(
        "--probe",
        type=parse_probe_request,
        action="append",
        required=True,
        help="STEP:CYCLE[:EXPECTED_SHA256]; may be repeated",
    )
    parser.add_argument("--base-seed", type=int, default=51)
    parser.add_argument(
        "--sampler-mode",
        choices=["sequential", "permuted_blocks"],
        help="defaults to the checkpoint training_data_contract, then sequential",
    )
    parser.add_argument(
        "--sampler-version",
        type=int,
        choices=[1],
        help="defaults to the checkpoint training_data_contract, then 1",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--block-size",
        type=int,
        help="defaults to exact train_tokens_seen / checkpoint step / batch_size",
    )
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument("--ranks", default="0,1,8,32,128")
    parser.add_argument(
        "--rank-step",
        type=int,
        action="append",
        help="run the functional rank probe only for this step",
    )
    parser.add_argument(
        "--min-contextual-gain",
        type=parse_non_negative_float,
        metavar="CE_DELTA",
        help=(
            "enable the contextual-signal gate with this explicit minimum CE "
            "penalty for both rank-0 and reversed-alignment controls"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        metavar="PATH",
        help="optionally write a machine-readable probe report to this path",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="do not mirror the production dual-GPU placement",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rank_steps = set(args.rank_step or ())
    if args.min_contextual_gain is not None and not rank_steps:
        parser.error(
            "--min-contextual-gain requires at least one --rank-step"
        )
    missing_rank_steps = rank_steps.difference(
        request.step for request in args.probe
    )
    if args.min_contextual_gain is not None and missing_rank_steps:
        parser.error(
            "every gated --rank-step must also appear in --probe; missing "
            + ",".join(str(step) for step in sorted(missing_rank_steps))
        )
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("checkpoint is missing training_state")
    tokens_per_step = infer_tokens_per_step(training_state)
    saved_data_contract = checkpoint.get("training_data_contract")
    saved_sampler = (
        dict(saved_data_contract.get("sampler") or {})
        if isinstance(saved_data_contract, dict)
        else {}
    )
    sampler_mode = args.sampler_mode or str(
        saved_sampler.get("mode") or "sequential"
    )
    sampler_version = args.sampler_version or int(
        saved_sampler.get("version") or 1
    )
    if (
        args.sampler_mode is not None
        and saved_sampler
        and args.sampler_mode != saved_sampler.get("mode")
    ):
        raise ValueError("requested sampler mode disagrees with checkpoint")
    if (
        args.sampler_version is not None
        and saved_sampler
        and args.sampler_version != saved_sampler.get("version")
    ):
        raise ValueError("requested sampler version disagrees with checkpoint")
    if args.block_size is None:
        if tokens_per_step % args.batch_size:
            raise ValueError(
                f"tokens/step={tokens_per_step} is not divisible by "
                f"batch_size={args.batch_size}"
            )
        block_size = tokens_per_step // args.batch_size
    else:
        block_size = args.block_size
        if block_size * args.batch_size != tokens_per_step:
            raise ValueError(
                "requested batch geometry disagrees with checkpoint: "
                f"{block_size}*{args.batch_size} != {tokens_per_step}"
            )

    token_ids, token_path = load_token_stream(checkpoint, args.token_bin)
    train_token_count = (
        int(saved_data_contract.get("train_token_count", len(token_ids)))
        if isinstance(saved_data_contract, dict)
        else len(token_ids)
    )
    if not block_size < train_token_count <= len(token_ids):
        raise ValueError(
            "checkpoint training_data_contract has invalid train_token_count"
        )
    training_token_ids = token_ids[:train_token_count]
    ledger_paths = tuple(checkpoint_path.parent.glob("causal_events*.jsonl"))
    prepared: list[
        tuple[ProbeRequest, torch.Tensor, str, int, dict[str, object]]
    ] = []
    report: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "token_bin": str(token_path),
        "checkpoint_step": int(training_state.get("step", 0)),
        "checkpoint_cycle": int(training_state.get("cycle", 0)),
        "tokens_per_step": tokens_per_step,
        "batch_size": args.batch_size,
        "block_size": block_size,
        "sampler_mode": sampler_mode,
        "sampler_version": sampler_version,
        "train_token_count": train_token_count,
        "probes": [],
    }
    report_probes = report["probes"]
    assert isinstance(report_probes, list)

    print(f"checkpoint={checkpoint_path}")
    print(f"token_bin={token_path}")
    print(
        f"checkpoint_step={training_state.get('step')} "
        f"checkpoint_cycle={training_state.get('cycle')} "
        f"tokens_per_step={tokens_per_step} "
        f"batch_size={args.batch_size} block_size={block_size} "
        f"sampler={sampler_mode}:v{sampler_version} "
        f"train_token_count={train_token_count}"
    )
    for request in args.probe:
        batch, effective_seed = make_batch(
            training_token_ids,
            step=request.step,
            cycle=request.cycle,
            block_size=block_size,
            batch_size=args.batch_size,
            base_seed=args.base_seed,
            sampler_mode=sampler_mode,
            sampler_version=sampler_version,
        )
        digest = tensor_digest(batch)
        ledger_digest = ledger_digest_for_step(
            ledger_paths,
            step=request.step,
            cycle=request.cycle,
        )
        expected = request.expected_digest or ledger_digest
        status = "UNRECORDED"
        if expected is not None:
            if digest != expected:
                raise RuntimeError(
                    f"batch digest mismatch at step={request.step}, "
                    f"cycle={request.cycle}: computed={digest}, expected={expected}"
                )
            status = "MATCH"
        print(
            f"batch step={request.step} cycle={request.cycle} "
            f"seed={effective_seed} digest={digest} digest_status={status}"
        )
        probe_report: dict[str, object] = {
            "step": request.step,
            "cycle": request.cycle,
            "effective_seed": effective_seed,
            "digest": digest,
            "digest_status": status,
        }
        report_probes.append(probe_report)
        prepared.append(
            (request, batch, digest, effective_seed, probe_report)
        )

    model, device = load_model(checkpoint, dual_gpu=not args.single_gpu)
    print(
        f"device={device} dual_gpu={model._dual_gpu} "
        f"split_layer={model._split_layer if model._dual_gpu else 'n/a'}"
    )
    ranks = tuple(int(value) for value in args.ranks.split(",") if value.strip())
    if any(rank < 0 for rank in ranks):
        raise ValueError("ranks must be non-negative")
    failed_context_steps: list[int] = []

    for request, cpu_batch, digest, _, probe_report in prepared:
        batch = cpu_batch.to(device)
        captured: dict[str, torch.Tensor] = {}

        def capture_lm_head_input(_module, inputs):
            captured["hidden_for_logits"] = inputs[0].detach()

        hook = model.lm_head.register_forward_pre_hook(capture_lm_head_input)
        try:
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(batch, labels=None, heartbeat=False)
        finally:
            hook.remove()
        hidden = captured.get("hidden_for_logits")
        if hidden is None:
            raise RuntimeError("lm_head pre-hook did not capture its input")

        ce_logits = _stable_cross_entropy(output.logits[:, :-1], batch[:, 1:])
        ce_hook = lm_head_ce(model, hidden[:, :-1], batch[:, 1:])
        delta = abs(float(ce_logits.detach().cpu()) - float(ce_hook.detach().cpu()))
        if delta > 1e-5:
            raise RuntimeError(
                f"lm_head control mismatch at step={request.step}: "
                f"ce_logits={float(ce_logits):.8f}, ce_hook={float(ce_hook):.8f}"
            )
        ce_value = float(ce_logits.detach().cpu())
        print(
            f"loss step={request.step} cycle={request.cycle} "
            f"digest={digest} ce_logits={ce_value:.6f} "
            f"ce_hook={float(ce_hook.detach().cpu()):.6f} "
            f"ppl={math.exp(min(ce_value, 20.0)):.3f} "
            f"input_t={batch.shape[1]} logits_t={output.logits.shape[1]} "
            f"hidden_t={hidden.shape[1]}"
        )
        probe_report["loss"] = {
            "ce_logits": ce_value,
            "ce_hook": float(ce_hook.detach().cpu()),
            "ppl": math.exp(min(ce_value, 20.0)),
            "input_t": batch.shape[1],
            "logits_t": output.logits.shape[1],
            "hidden_t": hidden.shape[1],
        }

        if request.step in rank_steps:
            rank_results = functional_rank_probe(
                model,
                hidden,
                batch,
                ranks=ranks,
                sample_tokens=args.sample_tokens,
            )
            gate_status = "NOT_REQUESTED"
            if args.min_contextual_gain is not None:
                if contextual_signal_passes(
                    rank_results,
                    args.min_contextual_gain,
                ):
                    gate_status = "PASS"
                else:
                    gate_status = "FAIL"
                    failed_context_steps.append(request.step)
            threshold_text = (
                "disabled"
                if args.min_contextual_gain is None
                else f"{args.min_contextual_gain:.9g}"
            )
            print(
                "functional_rank "
                + " ".join(
                    f"{key}={value:.9g}"
                    for key, value in rank_results.items()
                )
                + f" context_gate={gate_status}"
                + f" min_contextual_gain={threshold_text}"
            )
            probe_report["functional_rank"] = {
                **rank_results,
                "context_gate": gate_status,
                "min_contextual_gain": args.min_contextual_gain,
            }

        del output, hidden, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gate_enabled = args.min_contextual_gain is not None
    report["context_gate"] = {
        "enabled": gate_enabled,
        "minimum_gain": args.min_contextual_gain,
        "passed": None if not gate_enabled else not failed_context_steps,
        "failed_steps": failed_context_steps,
    }
    if args.json_output is not None:
        json_path = args.json_output.resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json_output={json_path}")
    gate_exit_code = contextual_gate_exit_code(failed_context_steps)
    if gate_exit_code:
        print(
            "context_gate=FAIL "
            f"failed_steps={','.join(str(step) for step in failed_context_steps)}"
        )
        return gate_exit_code
    if gate_enabled:
        print("context_gate=PASS")
    return gate_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
