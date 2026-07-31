#!/usr/bin/env python3
"""Darwin Transfer V2 — TTM + JEPA experience test.

Compara TTM-only (V1) vs TTM+JEPA (V2) no mesmo teste de aprendizado pelo uso.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from f51_darwin.transfer.checkpoint import load_transfer_checkpoint
from f51_darwin.transfer.donor import load_frozen_teacher, sha256_file
from f51_darwin.transfer.experience import ExperienceMemory
from f51_darwin.transfer.jepa_predictor import JEPAPredictor
from f51_darwin.transfer.runtime import DarwinTransferRuntime
from f51_darwin.transfer.runtime_v2 import DarwinTransferRuntimeV2
from f51_darwin.transfer.student import DarwinTransferStudent


def checkpoint_from_pointer(root: Path) -> Path:
    pointer = root / "published.json"
    if pointer.is_file():
        return Path(json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"])
    latest = root / "latest.pt"
    if latest.is_file():
        return latest
    raise FileNotFoundError(f"no published or latest checkpoint below {root}")


def run_test(
    runtime,
    tokenizer,
    observe_text: str,
    prompt: str,
    max_new_tokens: int,
    control: str,
    device: str,
) -> dict:
    """Run one experience test and return metrics."""
    # Phase 1: Observe
    if observe_text:
        observed = tokenizer(
            observe_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        for position in range(1, observed.shape[1]):
            runtime.observe(
                observed[:, :position],
                target_id=int(observed[0, position]),
                vocab_size=runtime.backbone.config.vocab_size,
            )

    # Phase 2: Generate
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated, paths = runtime.generate(
        input_ids, max_new_tokens=max_new_tokens, learn=(control == "correct")
    )

    result = {
        "text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "paths": paths,
        "backbone_calls": runtime.backbone_calls,
        "experience_hits": runtime.experience_hits,
    }

    if hasattr(runtime, "jepa_hits"):
        result["jepa_hits"] = runtime.jepa_hits

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--memory", type=Path, default=Path("workspace/runtime/transfer/experience_memory_v2.pt"))
    parser.add_argument("--prompt", default="Darwin learns through")
    parser.add_argument("--observe-text", default="Darwin learns from experience and adapts.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jepa-threshold", type=float, default=0.8)
    parser.add_argument("--compare", action="store_true", help="Compare V1 (TTM-only) vs V2 (TTM+JEPA)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint = checkpoint_from_pointer(args.checkpoint_root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    layers = tuple(payload["metadata"]["replaced_layers"])
    teacher, tokenizer, _ = load_frozen_teacher(args.donor, device=device)
    student = DarwinTransferStudent.from_teacher(teacher, replaced_layers=layers).to(device)
    load_transfer_checkpoint(
        student, checkpoint,
        expected_donor_manifest_sha256=sha256_file(args.donor / "donor_manifest.json"),
    )
    student.eval()
    d_model = student.config.n_embd
    print(f"Model loaded: {d_model}d, layers={layers}, attention_remaining=0")

    if args.compare:
        # ── JEPA warmup: run a few tokens through backbone to train predictor ──
        print("Warming up JEPA...")
        warmup_text = "Darwin learns from experience and adapts to new challenges every day"
        warmup_ids = tokenizer(warmup_text, return_tensors="pt").input_ids.to(device)
        jepa_warmup = JEPAPredictor(d_model=d_model, bottleneck=32).to(device)
        opt = torch.optim.Adam(jepa_warmup.parameters(), lr=1e-2)
        # Run backbone (WITH gradients so JEPA can learn)
        out = student(input_ids=warmup_ids, output_hidden_states=True)
        if out.hidden_states:
            hiddens = out.hidden_states[-1][0]  # [seq, d_model]
            for i in range(len(hiddens) - 1):
                loss = jepa_warmup.train_step(
                    hiddens[i].detach().unsqueeze(0),
                    hiddens[i+1].detach().unsqueeze(0),
                    opt)
        print(f"  JEPA warmup done ({len(hiddens)-1} steps, final_loss={loss:.4f})")

        # ── V1: TTM only ──
        print("\n" + "=" * 60)
        print("  V1 — TTM Only")
        print("=" * 60)
        mem_v1 = ExperienceMemory(capacity=4096, top_k=64, threshold=0.995, min_observations=1)
        rt_v1 = DarwinTransferRuntime(student, student.model.transformer.wte, mem_v1)
        r1 = run_test(rt_v1, tokenizer, args.observe_text, args.prompt,
                      args.max_new_tokens, "correct", device)

        # ── V2: TTM + JEPA ──
        print("\n" + "=" * 60)
        print("  V2 — TTM + JEPA")
        print("=" * 60)
        mem_v2 = ExperienceMemory(capacity=4096, top_k=64, threshold=0.995, min_observations=1)
        jepa = JEPAPredictor(d_model=d_model, bottleneck=32).to(device)
        rt_v2 = DarwinTransferRuntimeV2(
            student, student.model.transformer.wte, mem_v2, jepa,
            jepa_confidence_threshold=args.jepa_threshold, device=str(device),
        )
        r2 = run_test(rt_v2, tokenizer, args.observe_text, args.prompt,
                      args.max_new_tokens, "correct", device)

        # ── Controls (V2 only) ──
        print("\n" + "=" * 60)
        print("  Controls (V2)")
        print("=" * 60)
        controls = {}
        for control in ["frozen", "shuffled", "reset"]:
            mem_c = ExperienceMemory(capacity=4096, top_k=64, threshold=0.995, min_observations=1)
            if control == "frozen":
                mem_c.freeze()
            elif control == "shuffled":
                mem_c.shuffle(seed=51)
            elif control == "reset":
                mem_c.reset()
            jepa_c = JEPAPredictor(d_model=d_model, bottleneck=32).to(device)
            rt_c = DarwinTransferRuntimeV2(
                student, student.model.transformer.wte, mem_c, jepa_c,
                jepa_confidence_threshold=args.jepa_threshold, device=str(device),
            )
            rc = run_test(rt_c, tokenizer, args.observe_text, args.prompt,
                          args.max_new_tokens, control, device)
            controls[control] = rc

        # ── Report ──
        print("\n" + "=" * 60)
        print("  COMPARISON")
        print("=" * 60)
        print(f"  {'Metric':<25s} {'V1 (TTM)':>12s} {'V2 (TTM+JEPA)':>15s}")
        print(f"  {'─'*52}")
        print(f"  {'Backbone calls':<25s} {r1['backbone_calls']:>12d} {r2['backbone_calls']:>15d}")
        print(f"  {'Experience hits':<25s} {r1['experience_hits']:>12d} {r2['experience_hits']:>15d}")
        print(f"  {'JEPA hits':<25s} {'—':>12s} {r2.get('jepa_hits', 0):>15d}")
        total_v1 = r1['backbone_calls'] + r1['experience_hits']
        total_v2 = r2['backbone_calls'] + r2['experience_hits'] + r2.get('jepa_hits', 0)
        backbone_pct_v1 = r1['backbone_calls'] / max(1, total_v1) * 100
        backbone_pct_v2 = r2['backbone_calls'] / max(1, total_v2) * 100
        print(f"  {'Backbone %':<25s} {backbone_pct_v1:>11.1f}% {backbone_pct_v2:>14.1f}%")
        print(f"  {'Text V1':<25s} {r1['text'][:80]}")
        print(f"  {'Text V2':<25s} {r2['text'][:80]}")

        print(f"\n  Controls (V2):")
        print(f"  {'Control':<15s} {'Backbone':>10s} {'Exp Hits':>10s} {'JEPA Hits':>10s} {'Text'}")
        for cname, cr in [("correct", r2)] + list(controls.items()):
            print(f"  {cname:<15s} {cr['backbone_calls']:>10d} {cr['experience_hits']:>10d} "
                  f"{cr.get('jepa_hits', 0):>10d} {cr['text'][:50]}")

        # Save results
        report = {
            "v1_ttm_only": r1,
            "v2_ttm_jepa": r2,
            "controls": controls,
        }
        out = Path("workspace/runtime/transfer/experience_test_v2.json")
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(f"\nReport: {out}")

    else:
        # Single V2 run
        memory = ExperienceMemory(capacity=4096, top_k=64, threshold=0.995, min_observations=1)
        jepa = JEPAPredictor(d_model=d_model, bottleneck=32).to(device)
        runtime = DarwinTransferRuntimeV2(
            student, student.model.transformer.wte, memory, jepa,
            jepa_confidence_threshold=args.jepa_threshold, device=str(device),
        )
        result = run_test(runtime, tokenizer, args.observe_text, args.prompt,
                          args.max_new_tokens, "correct", device)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
