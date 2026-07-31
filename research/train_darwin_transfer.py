from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from f51_darwin.transfer.checkpoint import (
    inherit_matching_weights,
    load_transfer_checkpoint,
    save_transfer_checkpoint,
)
from f51_darwin.transfer.data import (
    evaluation_partitions,
    load_or_build_token_cache,
    split_transfer_holdouts,
)
from f51_darwin.transfer.donor import load_frozen_teacher, sha256_file
from f51_darwin.transfer.ngram import fit_ngram_baseline
from f51_darwin.transfer.schedule import (
    default_progressive_schedule,
    parse_schedule,
    publication_gates,
    validate_schedule,
)
from f51_darwin.transfer.student import DarwinTransferStudent
from f51_darwin.transfer.trainer import TransferTrainer


STAGES = ("orientation", "alignment", "distillation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--corpus-max-bytes", type=int, default=96_000_000)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--layers", help="single-generation compatibility mode")
    parser.add_argument("--schedule", help="semicolon-separated progressive layers")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--monitor-holdout-chunks", type=int, default=256)
    parser.add_argument("--promotion-holdout-tokens", type=int, default=5_000_000)
    parser.add_argument("--promotion-holdout-batches", type=int, default=256)
    parser.add_argument("--ngram-train-chunks", type=int, default=200_000)
    parser.add_argument("--ngram-alpha", type=float, default=0.5)
    parser.add_argument("--ngram-margin", type=float, default=0.0)
    parser.add_argument("--steps-per-stage", type=int)
    parser.add_argument("--orientation-steps", type=int, default=250)
    parser.add_argument("--alignment-steps", type=int, default=250)
    parser.add_argument("--distillation-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--defer-promotion",
        action="store_true",
        help="finish the requested training horizon without running publication holdout",
    )
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint(
    trainer: TransferTrainer,
    path: Path,
    *,
    donor_hash: str,
    generation: int,
    stage: str,
    stage_step: int,
    include_optimizer: bool = True,
) -> None:
    save_transfer_checkpoint(
        trainer.student,
        path,
        donor_manifest_sha256=donor_hash,
        stage=stage,
        step=trainer.step,
        generation=generation,
        stage_step=stage_step,
        optimizer_state=(
            trainer.optimizer_state_dict() if include_optimizer else None
        ),
    )


def main() -> None:
    args = parse_args()
    if args.steps_per_stage is not None:
        args.orientation_steps = args.steps_per_stage
        args.alignment_steps = args.steps_per_stage
        args.distillation_steps = args.steps_per_stage
    stage_steps = {
        "orientation": args.orientation_steps,
        "alignment": args.alignment_steps,
        "distillation": args.distillation_steps,
    }
    if (
        any(value <= 0 for value in stage_steps.values())
        or args.checkpoint_every <= 0
        or args.batch_size <= 0
    ):
        raise ValueError("steps and checkpoint interval must be positive")
    device = torch.device(args.device)
    teacher, tokenizer, _ = load_frozen_teacher(args.donor, device=str(device))
    layer_count = len(teacher.transformer.h)
    if args.layers:
        only = tuple(sorted({int(value) for value in args.layers.split(",")}))
        schedule = (only,)
    elif args.schedule:
        schedule = validate_schedule(
            parse_schedule(args.schedule), layer_count=layer_count
        )
    else:
        schedule = default_progressive_schedule(layer_count)

    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    donor_hash = sha256_file(args.donor / "donor_manifest.json")
    tokenizer.model_max_length = int(1e30)
    corpus_root = (
        args.corpus_root
        if args.corpus_root is not None
        else args.repo_root / "workspace" / "02_CORPUS" / "corpus"
    ).resolve()
    chunks, cache_hit = load_or_build_token_cache(
        tokenizer,
        corpus_root,
        max_bytes=args.corpus_max_bytes,
        sequence_length=args.sequence_length,
        cache_path=args.runtime_root / "corpus_chunks.pt",
        tokenizer_identity=donor_hash,
    )
    train, monitor_holdout, promotion_holdout = split_transfer_holdouts(
        chunks,
        sequence_length=args.sequence_length,
        monitor_chunks=args.monitor_holdout_chunks,
        promotion_tokens=args.promotion_holdout_tokens,
    )
    promotion_partitions = evaluation_partitions(
        promotion_holdout,
        batches=args.promotion_holdout_batches,
    )
    metrics = args.runtime_root / "metrics.jsonl"
    stop_path = args.runtime_root / "STOP_REQUESTED"
    latest = args.checkpoint_root / "latest.pt"

    resume_metadata = None
    resume_generation = 1
    previous_checkpoint: Path | None = None
    global_step = 0
    teacher_monitor_nll: float | None = None
    student_promotion_nll: float | None = None
    if args.resume and latest.is_file():
        raw = torch.load(latest, map_location="cpu", weights_only=True)["metadata"]
        if raw["donor_manifest_sha256"] != donor_hash:
            raise ValueError("latest checkpoint belongs to another donor")
        resume_generation = int(raw.get("generation", 1))
        resume_metadata = raw
        previous_checkpoint = latest
        global_step = int(raw["step"])

    for generation, layers in enumerate(schedule, start=1):
        if generation < resume_generation:
            continue
        student = DarwinTransferStudent.from_teacher(
            teacher, replaced_layers=layers
        ).to(device)
        if resume_metadata is not None and generation == resume_generation:
            if tuple(resume_metadata["replaced_layers"]) != layers:
                raise ValueError("resume schedule does not match checkpoint lineage")
        elif previous_checkpoint is not None:
            inherited = inherit_matching_weights(
                student,
                previous_checkpoint,
                expected_donor_manifest_sha256=donor_hash,
            )
            if inherited <= 0:
                raise ValueError("progressive generation inherited no weights")

        trainer = TransferTrainer(
            teacher,
            student,
            learning_rate=args.learning_rate,
            metrics_path=metrics,
            metric_context={
                "run_id": args.run_id,
                "generation": generation,
                "layers": layers,
                "batch_size": args.batch_size,
            },
        )
        trainer.step = global_step
        if teacher_monitor_nll is None:
            teacher_monitor_nll = trainer.evaluate_teacher_nll(
                monitor_holdout, device
            )
            trainer.log_metric(
                {
                    "event": "teacher_monitor_baseline",
                    "run_id": args.run_id,
                    "holdout_nll": teacher_monitor_nll,
                    "monitor_tokens": (
                        len(monitor_holdout) * args.sequence_length
                    ),
                    "promotion_tokens": (
                        len(promotion_holdout) * args.sequence_length
                    ),
                    "promotion_batches": len(promotion_partitions),
                    "token_cache_hit": cache_hit,
                    "corpus_chunks": len(chunks),
                }
            )
        start_stage = 0
        start_stage_step = 0
        if resume_metadata is not None and generation == resume_generation:
            start_stage = STAGES.index(str(resume_metadata["stage"]))
            start_stage_step = int(resume_metadata.get("stage_step", 0))
            if start_stage_step >= stage_steps[STAGES[start_stage]]:
                start_stage += 1
                start_stage_step = 0
            if start_stage < len(STAGES):
                trainer.configure_stage(STAGES[start_stage])
                load_transfer_checkpoint(
                    student,
                    latest,
                    expected_donor_manifest_sha256=donor_hash,
                    optimizer=trainer.optimizer,
                )
            else:
                load_transfer_checkpoint(
                    student,
                    latest,
                    expected_donor_manifest_sha256=donor_hash,
                )

        for stage_index in range(start_stage, len(STAGES)):
            stage = STAGES[stage_index]
            target_stage_steps = stage_steps[stage]
            stage_start = start_stage_step if stage_index == start_stage else 0
            batch_indices = [
                (trainer.step * args.batch_size + offset) % len(train)
                for offset in range(args.batch_size)
            ]
            baseline_batch = torch.stack(
                [train[index] for index in batch_indices]
            ).to(device=device, dtype=torch.long)
            baseline_loss = trainer.measure_loss(stage, baseline_batch)
            trainer.log_metric(
                {
                    "event": "stage_start",
                    "run_id": args.run_id,
                    "generation": generation,
                    "layers": layers,
                    "attention_layers_remaining": student.attention_layer_count,
                    "stage": stage,
                    "stage_step": stage_start,
                    "step": trainer.step,
                    "baseline_loss": baseline_loss,
                }
            )
            for stage_step in range(stage_start + 1, target_stage_steps + 1):
                batch_indices = [
                    (trainer.step * args.batch_size + offset) % len(train)
                    for offset in range(args.batch_size)
                ]
                batch = torch.stack(
                    [train[index] for index in batch_indices]
                ).to(device=device, dtype=torch.long)
                trainer.train_step(stage, batch)
                if stage_step % args.checkpoint_every == 0 or stop_path.is_file():
                    checkpoint(
                        trainer,
                        latest,
                        donor_hash=donor_hash,
                        generation=generation,
                        stage=stage,
                        stage_step=stage_step,
                    )
                if stop_path.is_file():
                    stop_path.unlink(missing_ok=True)
                    trainer.log_metric(
                        {
                            "event": "graceful_stop",
                            "run_id": args.run_id,
                            "generation": generation,
                            "stage": stage,
                            "stage_step": stage_step,
                            "step": trainer.step,
                        }
                    )
                    return
            holdout_nll = trainer.evaluate_nll(monitor_holdout, device)
            end_loss = trainer.measure_loss(stage, baseline_batch)
            trainer.log_metric(
                {
                    "event": "stage_end",
                    "run_id": args.run_id,
                    "generation": generation,
                    "stage": stage,
                    "step": trainer.step,
                    "baseline_loss": baseline_loss,
                    "end_loss": end_loss,
                    "relative_loss": end_loss / max(baseline_loss, 1e-12),
                    "holdout_nll": holdout_nll,
                }
            )
            checkpoint(
                trainer,
                latest,
                donor_hash=donor_hash,
                generation=generation,
                stage=stage,
                stage_step=target_stage_steps,
            )

        final_path = args.checkpoint_root / f"generation-{generation:02d}-final.pt"
        checkpoint(
            trainer,
            final_path,
            donor_hash=donor_hash,
            generation=generation,
            stage="distillation",
            stage_step=stage_steps["distillation"],
            include_optimizer=False,
        )
        previous_checkpoint = final_path
        global_step = trainer.step
        resume_metadata = None

    assert teacher_monitor_nll is not None
    assert previous_checkpoint is not None
    if args.defer_promotion:
        trainer.log_metric(
            {
                "event": "promotion_deferred",
                "run_id": args.run_id,
                "generation": len(schedule),
                "step": trainer.step,
                "checkpoint": str(previous_checkpoint.resolve()),
                "attention_layers_remaining": student.attention_layer_count,
                "reason": "explicit_training_continuation",
            }
        )
        return

    teacher_promotion_nll = trainer.evaluate_teacher_nll(
        promotion_holdout, device
    )
    student_promotion_nll = trainer.evaluate_nll(promotion_holdout, device)
    baseline = fit_ngram_baseline(
        train,
        vocab_size=int(teacher.config.vocab_size),
        device=device,
        alpha=args.ngram_alpha,
        max_chunks=args.ngram_train_chunks,
    )
    ngram_promotion_nll = baseline.nll(promotion_holdout, device=device)
    trainer.log_metric(
        {
            "event": "ngram_baseline",
            "run_id": args.run_id,
            "holdout_nll": ngram_promotion_nll,
            "train_tokens": baseline.train_tokens,
            "distinct_pairs": baseline.distinct_pairs,
            "alpha": args.ngram_alpha,
        }
    )
    gates = publication_gates(
        attention_layers_remaining=student.attention_layer_count,
        teacher_holdout_nll=teacher_promotion_nll,
        student_holdout_nll=student_promotion_nll,
        ngram_holdout_nll=ngram_promotion_nll,
        ngram_margin=args.ngram_margin,
    )
    candidate = {
        "schema_version": 2,
        "status": "accepted" if all(gates.values()) else "rejected",
        "run_id": args.run_id,
        "checkpoint": str(previous_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(previous_checkpoint),
        "donor_manifest_sha256": donor_hash,
        "replaced_layers": list(schedule[-1]),
        "attention_layers_remaining": student.attention_layer_count,
        "teacher_holdout_nll": teacher_promotion_nll,
        "student_holdout_nll": student_promotion_nll,
        "ngram_holdout_nll": ngram_promotion_nll,
        "ngram_train_tokens": baseline.train_tokens,
        "ngram_distinct_pairs": baseline.distinct_pairs,
        "promotion_holdout_tokens": len(promotion_holdout) * args.sequence_length,
        "promotion_holdout_batches": len(promotion_partitions),
        "gates": gates,
    }
    atomic_json(candidate, args.checkpoint_root / "candidate.json")
    if all(gates.values()):
        atomic_json(candidate, args.checkpoint_root / "published.json")


if __name__ == "__main__":
    main()
