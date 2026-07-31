from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from f51_darwin.transfer.donor import load_frozen_teacher
from f51_darwin.transplant.bundle import (
    extract_first_slice_bundle,
    load_organ_bundle,
    sha256_file,
)
from f51_darwin.transplant.checkpoint import save_recipient_checkpoint
from f51_darwin.transplant.experiment import (
    ArmBudget,
    ExperimentalArm,
    build_first_slice_arms,
)
from f51_darwin.transplant.ledger import (
    OrganEvidence,
    OrganLedger,
    organ_utility,
)
from f51_darwin.transplant.lifecycle import (
    OrganState,
    audit_gradients,
    configure_trainable_state,
    configure_trainable_states,
)
from f51_darwin.transplant.organs import (
    build_original_gaba,
    build_original_heartbeat,
    build_original_ihs,
    build_original_jepa,
    build_original_mtp,
    build_original_spider,
    load_ttm_memory,
)
from f51_darwin.transplant.recipient import (
    GUARDED_TRANSPLANT_POLICY,
    TwoDonorRecipient,
)
from f51_darwin.transplant.slots import (
    GABAResidualSlot,
    HeartbeatSlot,
    IHSResidualSlot,
    JEPAAuxiliarySlot,
    MTPSlot,
    SpiderSenseSlot,
    TTMResidualSlot,
)


EXPECTED_ORGAN_DONOR_SHA256 = (
    "71c49bc295c0d06d72b3dc640d5b4d846f421ecdcca25820517fffaab6425a30"
)
TRANSPLANTED_ORGANS = (
    "jepa",
    "gaba.0",
    "spider",
    "mtp",
    "ttm",
    "heartbeat",
    "ihs",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded two-donor Darwin transplant canary"
    )
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--organ-checkpoint", type=Path, required=True)
    parser.add_argument("--organ-donor-sha256", default=EXPECTED_ORGAN_DONOR_SHA256)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps-per-arm", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--guarded-stack-canary", action="store_true")
    return parser.parse_args()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _language_donor_identity(manifest_path: Path) -> str:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"donor manifest missing: {manifest_path}")
    return sha256_file(manifest_path)


def _load_verified_bundle(args: argparse.Namespace):
    actual_checkpoint_hash = sha256_file(args.organ_checkpoint)
    if actual_checkpoint_hash != args.organ_donor_sha256:
        raise ValueError(
            "organ donor checkpoint hash mismatch: "
            f"expected={args.organ_donor_sha256} actual={actual_checkpoint_hash}"
        )
    extract_bundle = not args.bundle_path.is_file()
    if not extract_bundle:
        bundle = load_organ_bundle(args.bundle_path)
        if bundle.manifest.source_checkpoint_sha256 != actual_checkpoint_hash:
            raise ValueError("organ bundle donor identity mismatch")
        missing_organs = sorted(
            set(TRANSPLANTED_ORGANS) - set(bundle.manifest.organs)
        )
        if missing_organs:
            extract_bundle = True
    if extract_bundle:
        extract_first_slice_bundle(
            args.organ_checkpoint,
            args.bundle_path,
            source_checkpoint_sha256=actual_checkpoint_hash,
            gaba_layers=(0,),
            extra_organs=(
                "_spider_sense_module.",
                "mtp_heads.",
                "inter_hemispheric.",
            ),
            include_heartbeat=True,
        )
        bundle = load_organ_bundle(args.bundle_path)
        missing_organs = sorted(
            set(TRANSPLANTED_ORGANS) - set(bundle.manifest.organs)
        )
        if missing_organs:
            raise ValueError(
                "fresh organ bundle is incomplete: "
                f"missing={missing_organs}"
            )
    if bundle.verify_tensor_hashes():
        raise ValueError("organ bundle tensor verification failed")
    return bundle, actual_checkpoint_hash


def _recipient_factory(language_donor, bundle, organ_checkpoint=None) -> TwoDonorRecipient:
    jepa = build_original_jepa(bundle)
    gaba = build_original_gaba(bundle, donor_layer=0)
    spider = build_original_spider(bundle)
    mtp = build_original_mtp(bundle)
    heartbeat = build_original_heartbeat(bundle)
    ihs = build_original_ihs(bundle)
    kwargs = {}
    if organ_checkpoint is not None:
        proj_key, proj_value = load_ttm_memory(organ_checkpoint)
        kwargs["ttm_slot"] = TTMResidualSlot(proj_key, proj_value)
    return TwoDonorRecipient.from_language_donor(
        language_donor,
        jepa_slot=JEPAAuxiliarySlot(jepa.module),
        gaba_slots={0: (GABAResidualSlot(gaba.module),)},
        spider_slot=SpiderSenseSlot(spider.module),
        mtp_slot=MTPSlot(mtp.module, depth=2),
        heartbeat_slot=HeartbeatSlot(heartbeat.module),
        ihs_slot=IHSResidualSlot(ihs.module),
        **kwargs,
    )


def _nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        input_ids[:, 1:].contiguous().view(-1),
    )


def _finite_recipient_output(output) -> None:
    if not torch.isfinite(output.logits).all():
        raise FloatingPointError("non-finite recipient logits")
    if output.jepa_loss is not None and not torch.isfinite(output.jepa_loss):
        raise FloatingPointError("non-finite recipient JEPA loss")


def _load_corpus_tokens(tokenizer, corpus_root: Path, minimum: int) -> list[int]:
    tokens: list[int] = []
    files = sorted(corpus_root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no local corpus text found under {corpus_root}")
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")[:20000]
        if text:
            remaining = max(minimum - len(tokens), 1)
            tokens.extend(
                tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=min(remaining, 1024),
                )
            )
        if len(tokens) >= minimum:
            break
    if len(tokens) < minimum:
        raise ValueError(
            f"local corpus too small: required={minimum} actual={len(tokens)}"
        )
    return tokens


def _batch(
    token_ids: list[int],
    *,
    start: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    maximum_start = len(token_ids) - sequence_length
    if maximum_start < 0:
        raise ValueError("token stream shorter than sequence length")
    resolved = start % (maximum_start + 1)
    return torch.tensor(
        [token_ids[resolved : resolved + sequence_length]],
        dtype=torch.long,
        device=device,
    )


def _zero_and_backward(
    arm: ExperimentalArm,
    input_ids: torch.Tensor,
    *,
    organ_name: str,
) -> tuple[float, float]:
    parameters = [
        parameter
        for name, parameter in arm.recipient.named_parameters()
        if name in arm.declared_parameters
    ]
    if not parameters:
        with torch.no_grad():
            output = arm.recipient(
                input_ids=input_ids,
                organ_mode=arm.organ_mode,
            )
            _finite_recipient_output(output)
            value = float(_nll(output.logits, input_ids))
        return value, value

    optimizer = torch.optim.AdamW(parameters, lr=3e-5)
    with torch.no_grad():
        before_output = arm.recipient(
            input_ids=input_ids,
            organ_mode=arm.organ_mode,
        )
        before = float(_nll(before_output.logits, input_ids))
    started = time.perf_counter()
    for _ in range(arm.budget.steps):
        optimizer.zero_grad(set_to_none=True)
        output = arm.recipient(
            input_ids=input_ids,
            organ_mode=arm.organ_mode,
        )
        _finite_recipient_output(output)
        loss = _nll(output.logits, input_ids)
        if organ_name == "jepa":
            if output.jepa_loss is None:
                raise RuntimeError("JEPA experiment produced no auxiliary loss")
            loss = loss + 0.1 * output.jepa_loss
        if organ_name == "spider":
            if output.spider_loss is not None:
                loss = loss + output.spider_loss
        if organ_name == "mtp":
            if output.mtp_loss is not None:
                loss = loss + 0.05 * output.mtp_loss
        if organ_name == "heartbeat":
            if output.heartbeat_loss is None:
                raise RuntimeError(
                    "Heartbeat experiment produced no state anchor"
                )
            loss = loss + output.heartbeat_loss
        loss.backward()
        audit_gradients(arm.recipient, arm.declared_parameters)
        optimizer.step()
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        after_output = arm.recipient(
            input_ids=input_ids,
            organ_mode=arm.organ_mode,
        )
        _finite_recipient_output(after_output)
        after = float(_nll(after_output.logits, input_ids))
    if elapsed < 0.0:
        raise RuntimeError("monotonic clock moved backwards")
    return before, after


def _evaluate(
    arm: ExperimentalArm,
    input_ids: torch.Tensor,
) -> float:
    arm.recipient.eval()
    with torch.no_grad():
        output = arm.recipient(
            input_ids=input_ids,
            organ_mode=arm.organ_mode,
        )
        _finite_recipient_output(output)
        return float(_nll(output.logits, input_ids))


def _arm_measurement(
    arm: ExperimentalArm,
    *,
    train_ids: torch.Tensor,
    old_ids: torch.Tensor,
    future_ids: torch.Tensor,
    organ_name: str,
    seed: int,
) -> dict[str, Any]:
    arm.recipient.to(train_ids.device)
    torch.manual_seed(seed)
    if train_ids.is_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(train_ids.device)
    started = time.perf_counter()
    before, after = _zero_and_backward(
        arm,
        train_ids,
        organ_name=organ_name,
    )
    old_nll = _evaluate(arm, old_ids)
    future_nll = _evaluate(arm, future_ids)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(train_ids.device))
        if train_ids.is_cuda
        else 0
    )
    arm.recipient.to("cpu")
    if train_ids.is_cuda:
        torch.cuda.empty_cache()
    return {
        "train_nll_before": before,
        "train_nll_after": after,
        "old_holdout_nll": old_nll,
        "future_holdout_nll": future_nll,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_memory,
        "declared_parameter_count": len(arm.declared_parameters),
    }


def _evidence(
    control: dict[str, Any],
    living: dict[str, Any],
    immutable: dict[str, Any],
) -> OrganEvidence:
    control_seconds = max(float(control["elapsed_seconds"]), 1e-9)
    living_seconds = float(living["elapsed_seconds"])
    immutable_old = float(immutable["old_holdout_nll"])
    living_old = float(living["old_holdout_nll"])
    return OrganEvidence(
        control_nll=float(control["old_holdout_nll"]),
        living_nll=living_old,
        future_gain=(
            float(control["future_holdout_nll"])
            - float(living["future_holdout_nll"])
        ),
        forgetting=max(0.0, living_old - immutable_old),
        compute_cost=max(0.0, (living_seconds - control_seconds) / control_seconds),
        instability=0.0,
        compute_saving=max(
            0.0,
            (control_seconds - living_seconds) / control_seconds,
        ),
    )


def _engineering_smoke(
    recipient: TwoDonorRecipient,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for state in (OrganState.ADAPTER_ACTIVE, OrganState.ORGAN_UNFROZEN):
        declared = configure_trainable_state(
            recipient,
            organ_name="gaba.0",
            state=state,
        )
        recipient.zero_grad(set_to_none=True)
        parameters = [
            parameter
            for name, parameter in recipient.named_parameters()
            if name in declared
        ]
        optimizer = torch.optim.AdamW(parameters, lr=3e-5)
        output = recipient(
            input_ids=input_ids,
            organ_mode=state.value,
        )
        loss = _nll(output.logits, input_ids)
        loss.backward()
        audit_gradients(recipient, declared)
        optimizer.step()
        results[state.value] = {
            "loss": float(loss.detach()),
            "declared_parameter_count": len(declared),
            "gradient_audit": "passed",
        }
        with torch.no_grad():
            recipient.organ_slot("gaba.0").external_gate.fill_(0.05)
    recipient.zero_grad(set_to_none=True)
    return results


def _heartbeat_summary(recipient: TwoDonorRecipient) -> dict[str, Any]:
    heartbeat = recipient.organ_slot("heartbeat").organ
    state = heartbeat.export_state()
    return {
        "beat": int(state.get("beat", 0)),
        "dopamine": float(state.get("dopamine", 0.5)),
        "memory_slots": len(state.get("tt_memory_slots", ())),
    }


def _policy_evaluate(
    recipient: TwoDonorRecipient,
    input_ids: torch.Tensor,
    *,
    policy: dict[str, str] | None,
) -> tuple[float, Any]:
    recipient.eval()
    with torch.no_grad():
        output = recipient(
            input_ids=input_ids,
            organ_mode="disabled" if policy is None else "shadow",
            organ_policy=policy,
        )
        _finite_recipient_output(output)
        return float(_nll(output.logits, input_ids)), output


def _run_guarded_stack_canary(
    *,
    args: argparse.Namespace,
    language_donor,
    bundle,
    corpus_tokens: list[int],
    language_identity: str,
    organ_identity: str,
    device: torch.device,
) -> dict[str, Any]:
    policy = dict(GUARDED_TRANSPLANT_POLICY)
    isolation_policy = {
        name: ("active" if name == "gaba.0" else "disabled")
        for name in TRANSPLANTED_ORGANS
    }
    ledger_path = args.runtime_root / f"{args.run_id}.guarded.ledger.jsonl"
    ledger = OrganLedger(ledger_path)
    checkpoint_run_root = args.checkpoint_root / args.run_id
    results: list[dict[str, Any]] = []
    final_checkpoint = None

    for seed_index in range(args.seeds):
        seed = 2003 + seed_index
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        starts = tuple(
            (seed_index * args.steps_per_arm + step) * args.sequence_length
            for step in range(args.steps_per_arm)
        )
        budget = ArmBudget(
            tokens=args.sequence_length * args.steps_per_arm,
            steps=args.steps_per_arm,
            seed=seed,
            starts=starts,
        )
        recipient = _recipient_factory(
            language_donor,
            bundle,
            args.organ_checkpoint,
        ).to(device)
        declared = configure_trainable_states(
            recipient,
            organ_states={
                "gaba.0": OrganState.ADAPTER_ACTIVE,
                "jepa": OrganState.ADAPTER_ACTIVE,
            },
        )
        parameters = [
            parameter
            for name, parameter in recipient.named_parameters()
            if name in declared
        ]
        optimizer = torch.optim.AdamW(parameters, lr=3e-5)
        old_ids = _batch(
            corpus_tokens,
            start=args.sequence_length * 5,
            sequence_length=args.sequence_length,
            device=device,
        )
        future_ids = _batch(
            corpus_tokens,
            start=args.sequence_length * 7,
            sequence_length=args.sequence_length,
            device=device,
        )
        old_before, disabled_old = _policy_evaluate(
            recipient,
            old_ids,
            policy=None,
        )
        future_before, _ = _policy_evaluate(
            recipient,
            future_ids,
            policy=None,
        )
        heartbeat_before = _heartbeat_summary(recipient)
        ttm_before = len(recipient.organ_slot("ttm")._memory)
        gate_before = float(
            recipient.organ_slot("gaba.0").external_gate.detach()
        )
        gradient_paths = {"gaba.0": False, "jepa": False}
        train_loss_before = None
        train_loss_after = None
        started = time.perf_counter()

        recipient.train()
        recipient.language_model.eval()
        for step, start in enumerate(starts):
            train_ids = _batch(
                corpus_tokens,
                start=start,
                sequence_length=args.sequence_length,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = recipient(
                input_ids=train_ids,
                organ_policy=policy,
            )
            _finite_recipient_output(output)
            if output.jepa_loss is None or output.jepa_loss.item() <= 0.0:
                raise RuntimeError("guarded stack produced no active JEPA loss")
            loss = _nll(output.logits, train_ids) + 0.1 * output.jepa_loss
            if step == 0:
                train_loss_before = float(loss.detach())
            loss.backward()
            audit_gradients(recipient, declared)
            for name, parameter in recipient.named_parameters():
                if name not in declared or parameter.grad is None:
                    continue
                nonzero = bool(parameter.grad.detach().abs().max().item() > 0.0)
                if name.startswith("gaba_slots.gaba_0."):
                    gradient_paths["gaba.0"] |= nonzero
                if name.startswith("jepa_slot."):
                    gradient_paths["jepa"] |= nonzero
            optimizer.step()
            train_loss_after = float(loss.detach())

        elapsed = time.perf_counter() - started
        old_after, active_old = _policy_evaluate(
            recipient,
            old_ids,
            policy=policy,
        )
        future_after, active_future = _policy_evaluate(
            recipient,
            future_ids,
            policy=policy,
        )
        _, isolated = _policy_evaluate(
            recipient,
            old_ids,
            policy=isolation_policy,
        )
        _, disabled_after = _policy_evaluate(
            recipient,
            old_ids,
            policy=None,
        )
        heartbeat_after = _heartbeat_summary(recipient)
        ttm_after = len(recipient.organ_slot("ttm")._memory)
        gate_after = float(
            recipient.organ_slot("gaba.0").external_gate.detach()
        )
        shadow_logit_error = float(
            (active_old.logits - isolated.logits).abs().max()
        )
        disabled_drift = float(
            (disabled_old.logits - disabled_after.logits).abs().max()
        )
        active_logit_delta = float(
            (active_old.logits - disabled_after.logits).abs().max()
        )
        if not all(gradient_paths.values()):
            raise RuntimeError(
                f"guarded stack missing gradient path: {gradient_paths}"
            )
        if heartbeat_after != heartbeat_before:
            raise RuntimeError("observer Heartbeat mutated during guarded stack")
        if ttm_after != ttm_before:
            raise RuntimeError("shadow TTM wrote memory during guarded stack")
        if shadow_logit_error > 1e-5:
            raise RuntimeError(
                "shadow organs changed logits: "
                f"max_abs_error={shadow_logit_error}"
            )
        if disabled_drift > 1e-5:
            raise RuntimeError(
                "frozen language donor drifted: "
                f"max_abs_error={disabled_drift}"
            )
        if active_logit_delta <= 0.0 or gate_after == gate_before:
            raise RuntimeError("active GABA did not affect guarded stack logits")

        evidence = OrganEvidence(
            control_nll=old_before,
            living_nll=old_after,
            future_gain=future_before - future_after,
            forgetting=max(0.0, old_after - old_before),
            compute_cost=0.0,
            instability=0.0,
        )
        record = ledger.append(
            organ_name="gaba.0",
            language_donor_sha256=language_identity,
            organ_donor_sha256=organ_identity,
            recipient_parent_sha256=None,
            arm="D",
            budget=budget,
            evidence=evidence,
            lifecycle_decision="inconclusive_guarded_stack_canary",
        )
        result = {
            "seed": seed,
            "budget": asdict(budget),
            "policy": policy,
            "declared_optimizer_parameters": sorted(declared),
            "gradient_paths": gradient_paths,
            "train_loss_before": train_loss_before,
            "train_loss_after": train_loss_after,
            "old_holdout_nll_before": old_before,
            "old_holdout_nll_after": old_after,
            "future_holdout_nll_before": future_before,
            "future_holdout_nll_after": future_after,
            "gaba_gate_before": gate_before,
            "gaba_gate_after": gate_after,
            "active_max_abs_logit_delta": active_logit_delta,
            "shadow_max_abs_logit_error": shadow_logit_error,
            "disabled_max_abs_logit_drift": disabled_drift,
            "heartbeat_before": heartbeat_before,
            "heartbeat_after": heartbeat_after,
            "ttm_memory_before": ttm_before,
            "ttm_memory_after": ttm_after,
            "spider_observed": "spider" in active_future.organ_observations,
            "elapsed_seconds": elapsed,
            "evidence_id": record.evidence_id,
        }
        results.append(result)

        if seed_index == args.seeds - 1:
            checkpoint_path = checkpoint_run_root / (
                f"guarded-stack-seed-{seed}.pt"
            )
            final_checkpoint = save_recipient_checkpoint(
                recipient,
                checkpoint_path,
                language_donor_sha256=language_identity,
                organ_donor_sha256=organ_identity,
                parent_checkpoint_sha256=None,
                ledger_sha256=ledger.sha256,
                organ_states=policy,
                declared_optimizer_parameters=declared,
            )
        recipient.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_checkpoint is None:
        raise RuntimeError("guarded stack produced no checkpoint")
    return {
        "policy": policy,
        "scientific_status": "inconclusive_engineering_canary",
        "runs": results,
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger.sha256,
        "checkpoint_path": str(final_checkpoint.path),
        "checkpoint_sha256": final_checkpoint.sha256,
    }


def main() -> int:
    args = _parse_args()
    if args.sequence_length < 3:
        raise ValueError("sequence length must be at least 3")
    if args.seeds < 0 or args.steps_per_arm <= 0:
        raise ValueError("invalid engineering budget")
    if not args.verify_only and args.seeds == 0:
        raise ValueError("a launched engineering run requires at least one seed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.runtime_root.mkdir(parents=True, exist_ok=True)
    language_identity = _language_donor_identity(args.donor_manifest)
    language_donor, tokenizer, manifest = load_frozen_teacher(
        args.donor_root,
        allow_download=False,
        device="cpu",
    )
    bundle, organ_identity = _load_verified_bundle(args)
    recipient = _recipient_factory(language_donor, bundle, args.organ_checkpoint).to(device)

    corpus_tokens = _load_corpus_tokens(
        tokenizer,
        args.corpus_root,
        max(args.sequence_length * 8, 128),
    )
    identity_ids = _batch(
        corpus_tokens,
        start=0,
        sequence_length=args.sequence_length,
        device=device,
    )
    language_donor.to(device)
    with torch.no_grad():
        teacher_logits = language_donor(input_ids=identity_ids).logits
        disabled = recipient(
            input_ids=identity_ids,
            organ_mode="disabled",
        )
        shadow = recipient(
            input_ids=identity_ids,
            organ_mode="shadow",
        )
    _finite_recipient_output(disabled)
    _finite_recipient_output(shadow)
    disabled_error = float((teacher_logits - disabled.logits).abs().max())
    shadow_error = float((teacher_logits - shadow.logits).abs().max())
    if disabled_error > 1e-5 or shadow_error > 1e-5:
        raise ValueError(
            "zero-impact GPT equivalence failed: "
            f"disabled={disabled_error} shadow={shadow_error}"
        )
    language_donor.to("cpu")
    smoke = _engineering_smoke(recipient, identity_ids)
    recipient.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_metrics: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "verify_only": bool(args.verify_only),
        "device": str(device),
        "language_donor_manifest_sha256": language_identity,
        "language_donor_model_id": manifest.model_id,
        "organ_donor_checkpoint_sha256": organ_identity,
        "bundle_tensor_count": len(bundle.manifest.tensors),
        "bundle_parameter_count": sum(
            tensor.numel() for tensor in bundle.tensors.values()
        ),
        "transplanted_organs": list(TRANSPLANTED_ORGANS),
        "disabled_max_abs_logit_error": disabled_error,
        "shadow_max_abs_logit_error": shadow_error,
        "gradient_smoke": smoke,
        "experiments": [],
    }
    if args.verify_only:
        output = args.runtime_root / f"{args.run_id}.verify.json"
        _atomic_json(base_metrics, output)
        print(
            "TWO_DONOR_VERIFY_OK "
            f"organs={len(TRANSPLANTED_ORGANS)} "
            f"disabled_error={disabled_error:.3e} "
            f"shadow_error={shadow_error:.3e} metrics={output}",
            flush=True,
        )
        return 0

    if args.guarded_stack_canary:
        guarded = _run_guarded_stack_canary(
            args=args,
            language_donor=language_donor,
            bundle=bundle,
            corpus_tokens=corpus_tokens,
            language_identity=language_identity,
            organ_identity=organ_identity,
            device=device,
        )
        base_metrics["guarded_stack"] = guarded
        final_organ_hash = sha256_file(args.organ_checkpoint)
        if final_organ_hash != organ_identity:
            raise RuntimeError("organ donor changed during guarded canary")
        output = args.runtime_root / f"{args.run_id}.guarded.metrics.json"
        _atomic_json(base_metrics, output)
        print(
            "GUARDED_STACK_CANARY_OK "
            f"runs={len(guarded['runs'])} "
            f"checkpoint={guarded['checkpoint_path']} "
            f"metrics={output}",
            flush=True,
        )
        return 0

    ledger_path = args.runtime_root / f"{args.run_id}.ledger.jsonl"
    ledger = OrganLedger(ledger_path)
    checkpoint_run_root = args.checkpoint_root / args.run_id
    for organ_name in TRANSPLANTED_ORGANS:
        for seed_index in range(args.seeds):
            seed = 1009 + seed_index
            starts = tuple(
                seed_index * args.sequence_length + step * args.sequence_length
                for step in range(args.steps_per_arm)
            )
            budget = ArmBudget(
                tokens=args.sequence_length * args.steps_per_arm,
                steps=args.steps_per_arm,
                seed=seed,
                starts=starts,
            )
            budgets = {name: budget for name in "ABCD"}
            arms = build_first_slice_arms(
                lambda: _recipient_factory(language_donor, bundle, args.organ_checkpoint),
                organ_name=organ_name,
                budgets=budgets,
            )
            train_ids = _batch(
                corpus_tokens,
                start=starts[0],
                sequence_length=args.sequence_length,
                device=device,
            )
            old_ids = _batch(
                corpus_tokens,
                start=args.sequence_length * 4,
                sequence_length=args.sequence_length,
                device=device,
            )
            future_ids = _batch(
                corpus_tokens,
                start=args.sequence_length * 6,
                sequence_length=args.sequence_length,
                device=device,
            )
            arm_metrics = {
                name: _arm_measurement(
                    arm,
                    train_ids=train_ids,
                    old_ids=old_ids,
                    future_ids=future_ids,
                    organ_name=organ_name,
                    seed=seed,
                )
                for name, arm in arms.items()
            }
            records: dict[str, Any] = {}
            for arm_name in ("C", "D"):
                evidence = _evidence(
                    arm_metrics["B"],
                    arm_metrics[arm_name],
                    arm_metrics["A"],
                )
                record = ledger.append(
                    organ_name=organ_name,
                    language_donor_sha256=language_identity,
                    organ_donor_sha256=organ_identity,
                    recipient_parent_sha256=None,
                    arm=arm_name,
                    budget=budget,
                    evidence=evidence,
                    lifecycle_decision="inconclusive_engineering_canary",
                )
                records[arm_name] = {
                    "evidence_id": record.evidence_id,
                    "utility": organ_utility(evidence),
                    "evidence": asdict(evidence),
                }
            base_metrics["experiments"].append(
                {
                    "organ_name": organ_name,
                    "seed": seed,
                    "budget": asdict(budget),
                    "budget_identity": budget.identity,
                    "arms": arm_metrics,
                    "ledger": records,
                }
            )
            if seed_index == args.seeds - 1:
                d_arm = arms["D"]
                checkpoint_path = (
                    checkpoint_run_root
                    / f"{organ_name.replace('.', '_')}-seed-{seed}.pt"
                )
                save_recipient_checkpoint(
                    d_arm.recipient,
                    checkpoint_path,
                    language_donor_sha256=language_identity,
                    organ_donor_sha256=organ_identity,
                    parent_checkpoint_sha256=None,
                    ledger_sha256=ledger.sha256,
                    organ_states={
                        candidate: (
                            "organ_unfrozen"
                            if organ_name == candidate
                            else "shadow"
                        )
                        for candidate in TRANSPLANTED_ORGANS
                    },
                    declared_optimizer_parameters=d_arm.declared_parameters,
                )

    final_organ_hash = sha256_file(args.organ_checkpoint)
    if final_organ_hash != organ_identity:
        raise RuntimeError("organ donor changed during canary")
    base_metrics["ledger_path"] = str(ledger_path)
    base_metrics["ledger_sha256"] = ledger.sha256
    base_metrics["checkpoint_root"] = str(checkpoint_run_root)
    output = args.runtime_root / f"{args.run_id}.metrics.json"
    _atomic_json(base_metrics, output)
    print(
        "TWO_DONOR_RUN_OK "
        f"experiments={len(base_metrics['experiments'])} "
        f"ledger={ledger_path} metrics={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
