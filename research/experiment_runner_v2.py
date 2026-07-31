#!/usr/bin/env python3
"""F51 Overnight Benchmark v2 — causal continual-learning experiment.

Phase A shared → Phase B per condition. 5 conditions, 5 seeds.
Replay replaces fresh tokens (same budget). Shadow mode only.
"""
from __future__ import annotations

import argparse, json, math, os, shutil, sys, time, traceback, uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import msvcrt, numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.dataset_layout import TOKENIZED_RELATIVE, resolve_dataset_path

# ═══════════════════════════════════════════════════════════════
# TIERS
# ═══════════════════════════════════════════════════════════════

TIERS = {
    "smoke": {"phase_a_steps": 20, "phase_b_steps": 20, "eval_samples": 16},
    "main": {"phase_a_steps": 500, "phase_b_steps": 500, "eval_samples": 256},
    "long": {"phase_a_steps": 500, "phase_b_steps": 2000, "eval_samples": 512},
    "scale-smoke": {"phase_a_steps": 20, "phase_b_steps": 20, "eval_samples": 16},
}

MAIN_CONDITIONS = ["C0_STATIC", "C1_REPLAY_ONLY", "C2_GHOST_ONLY", "C3_ORGANS_NO_REPLAY", "F51_FULL_SHADOW"]
LONG_CONDITIONS = ["C0_STATIC", "C1_REPLAY_ONLY", "F51_FULL_SHADOW"]
SCALE_CONDITIONS = ["C0_STATIC", "C1_REPLAY_ONLY", "F51_FULL_SHADOW"]

# ═══════════════════════════════════════════════════════════════
# LOCK
# ═══════════════════════════════════════════════════════════════

class ExperimentAlreadyRunning(RuntimeError):
    pass

class SingleRunLock(AbstractContextManager):
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.handle.close()
            raise ExperimentAlreadyRunning("another runner is active")
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}).encode())
        self.handle.flush()
        return self
    def __exit__(self, *args):
        if self.handle:
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, PermissionError):
                pass
            self.handle.close()

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def save_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

@torch.no_grad()
def perplexity(model, tokens, block, samples, dev) -> float:
    model.eval()
    tl, tt = 0.0, 0
    for _ in range(samples):
        s = torch.randint(0, max(1, len(tokens) - block - 1), (1,)).item()
        b = torch.as_tensor(np.array(tokens[s:s+block], copy=True), dtype=torch.long, device=dev).unsqueeze(0)
        o = model(b, labels=b)
        if o.lm_loss is not None and math.isfinite(o.lm_loss.item()):
            tl += o.lm_loss.item() * block
            tt += block
    model.train()
    return math.exp(tl / tt) if tt > 0 else float("inf")

def build_model(config_path: Path, condition: str) -> tuple[DarwinXModel, dict]:
    raw = yaml.safe_load(open(config_path, encoding="utf-8"))
    # C0, C1: disable all auxiliary losses
    if condition in ("C0_STATIC", "C1_REPLAY_ONLY"):
        for k in ["heartbeat_enabled", "ghost_mask_ratio", "mtp_weight", "jepa_weight",
                   "ghost_weight", "curiosity_weight"]:
            raw[k] = False if k == "heartbeat_enabled" else 0.0

    cfg = DarwinXConfig.from_mapping(raw)

    # C2: Ghost only (modify raw dict before frozen config)
    if condition == "C2_GHOST_ONLY":
        raw["mtp_weight"] = 0.0
        raw["jepa_weight"] = 0.0
        raw["curiosity_weight"] = 0.0
        raw["heartbeat_enabled"] = False
        cfg = DarwinXConfig.from_mapping(raw)

    model = DarwinXModel(cfg)
    return model, cfg.__dict__ if hasattr(cfg, '__dict__') else {}

def condition_has_replay(condition: str) -> bool:
    return condition in ("C1_REPLAY_ONLY", "F51_FULL_SHADOW")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run(args):
    run_dir = ROOT / "runs" / "experiments_v2" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lock = SingleRunLock(run_dir / "lock")
    with lock:
        # Tier config
        tier = TIERS[args.tier]
        phase_a_steps = args.phase_a_steps or tier["phase_a_steps"]
        phase_b_steps = args.phase_b_steps or tier["phase_b_steps"]
        eval_samples = args.eval_samples or tier["eval_samples"]
        block = args.block
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

        conditions = args.conditions.split(",") if args.conditions else MAIN_CONDITIONS

        # Load data
        token_root = resolve_dataset_path(ROOT, TOKENIZED_RELATIVE, require=True)
        math_tokens = np.memmap(token_root / "MATEMATICA/tokens_openwebmath2.bin", dtype=np.int32, mode="r")
        lit_tokens = np.memmap(token_root / "CLASSICOS/tokens_classical.bin", dtype=np.int32, mode="r")

        # Split: last N tokens for validation, rest for training
        val_size = eval_samples * block * 4  # generous margin
        val_math = math_tokens[-val_size:].copy()
        val_lit = lit_tokens[-val_size:].copy()
        train_math = math_tokens[:-val_size]
        train_lit = lit_tokens[:-val_size]

        config_path = ROOT / "src" / "configs" / "darwin_x_600m_ckpt239.yaml"

        if args.plan:
            print(f"PLAN: tier={args.tier} conditions={conditions} seeds={seeds}")
            print(f"  phase_a_steps={phase_a_steps} phase_b_steps={phase_b_steps}")
            print(f"  eval_samples={eval_samples} block={block}")
            print(f"  math_tokens={len(train_math)/1e6:.1f}M lit_tokens={len(train_lit)/1e6:.1f}M")
            print(f"  val_math={len(val_math)} val_lit={len(val_lit)}")
            print(f"  estimated_time_per_seed: {phase_a_steps * 0.6 + phase_b_steps * len(conditions) * 0.6:.0f}s")
            save_json_atomic(run_dir / "readiness.json", {
                "tier": args.tier, "conditions": conditions, "seeds": seeds,
                "phase_a_steps": phase_a_steps, "phase_b_steps": phase_b_steps,
                "block": block, "eval_samples": eval_samples,
                "math_tokens": int(len(train_math)), "lit_tokens": int(len(train_lit)),
                "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1),
                "plan_only": True,
            })
            return

        dev = torch.device("cuda:0")
        all_results = {}
        all_results["meta"] = {
            "run_id": args.run_id, "tier": args.tier,
            "phase_a_steps": phase_a_steps, "phase_b_steps": phase_b_steps,
            "block": block, "eval_samples": eval_samples, "seeds": seeds,
            "conditions": conditions, "started_at": datetime.now(timezone.utc).isoformat(),
        }

        for seed in seeds:
            print(f"\n=== SEED {seed} ===")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # ═══════════════ PHASE A: SHARED ═══════════════════
            print(f"  Phase A: Math ({phase_a_steps} steps)...")
            m_shared, _ = build_model(config_path, "C0_STATIC")
            m_shared = m_shared.to(dtype=torch.bfloat16, device=dev)
            if torch.cuda.device_count() >= 2:
                split = m_shared.recommended_dual_gpu_split(gpu0=0, gpu1=1)
                m_shared.enable_dual_gpu(gpu0=0, gpu1=1, split_layer=split)
            opt_shared = torch.optim.AdamW(m_shared.parameters(), lr=3e-4, weight_decay=0.01, fused=True)

            for step in range(phase_a_steps):
                s = torch.randint(0, max(1, len(train_math) - block - 1), (1,)).item()
                b = torch.as_tensor(np.array(train_math[s:s+block], copy=True), dtype=torch.long, device=dev).unsqueeze(0)
                o = m_shared(b, labels=b)
                opt_shared.zero_grad()
                o.loss.backward()
                torch.nn.utils.clip_grad_norm_(m_shared.parameters(), 1.0)
                opt_shared.step()

            ppl_math_before = perplexity(m_shared, val_math, block, eval_samples, dev)
            ppl_lit_before = perplexity(m_shared, val_lit, block, eval_samples, dev)
            print(f"    PPL math: {ppl_math_before:.1f}  PPL lit: {ppl_lit_before:.1f}")

            # Save shared state
            shared_state = {k: v.cpu().clone() for k, v in m_shared.state_dict().items()}
            shared_manifest = m_shared.topology_manifest()
            del m_shared, opt_shared
            torch.cuda.empty_cache()

            # ═══════════════ PHASE B: PER CONDITION ═════════════
            for cond in conditions:
                print(f"    {cond}...", end=" ", flush=True)
                try:
                    m, _ = build_model(config_path, cond)
                    m = m.to(dtype=torch.bfloat16, device=dev)
                    if torch.cuda.device_count() >= 2:
                        split = m.recommended_dual_gpu_split(gpu0=0, gpu1=1)
                        m.enable_dual_gpu(gpu0=0, gpu1=1, split_layer=split)
                    m.load_state_dict(shared_state, strict=False)
                    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01, fused=True)

                    use_replay = condition_has_replay(cond)
                    # Build replay buffer from math (from shared phase A)
                    replay_buf = []
                    if use_replay:
                        for _ in range(128):
                            s = torch.randint(0, max(1, len(train_math) - block - 1), (1,)).item()
                            rb = torch.as_tensor(np.array(train_math[s:s+block], copy=True), dtype=torch.long, device=dev).unsqueeze(0)
                            replay_buf.append(rb.cpu())

                    fresh_count = 0
                    replay_count = 0
                    for step in range(phase_b_steps):
                        s = torch.randint(0, max(1, len(train_lit) - block - 1), (1,)).item()
                        b = torch.as_tensor(np.array(train_lit[s:s+block], copy=True), dtype=torch.long, device=dev).unsqueeze(0)
                        fresh_count += 1

                        # Replay: every 5th step, use math instead of lit (same budget)
                        if use_replay and step % 5 == 0 and len(replay_buf) > 0:
                            ri = torch.randint(0, len(replay_buf), (1,)).item()
                            b = replay_buf[ri].to(dev)
                            replay_count += 1

                        o = m(b, labels=b)
                        opt.zero_grad()
                        o.loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()

                    ppl_math_after = perplexity(m, val_math, block, eval_samples, dev)
                    ppl_lit_after = perplexity(m, val_lit, block, eval_samples, dev)
                    fr = ppl_math_after / max(ppl_math_before, 1.0)

                    result = {
                        "seed": seed, "condition": cond,
                        "ppl_math_before": ppl_math_before, "ppl_math_after": ppl_math_after,
                        "ppl_lit_before": ppl_lit_before, "ppl_lit_after": ppl_lit_after,
                        "forgetting_ratio": fr, "fresh_steps": fresh_count, "replay_steps": replay_count,
                    }
                    print(f"FR={fr:.3f}")
                    all_results[f"{cond}_seed{seed}"] = result
                    save_json_atomic(run_dir / "partial_results.json", all_results)

                    del m, opt
                    torch.cuda.empty_cache()

                except Exception as e:
                    traceback.print_exc()
                    all_results[f"{cond}_seed{seed}"] = {"seed": seed, "condition": cond, "error": str(e)}
                    save_json_atomic(run_dir / "partial_results.json", all_results)

        # ═══════════════ SAVE FINAL ═══════════════════
        all_results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json_atomic(run_dir / "results.json", all_results)

        # Summary
        print("\n=== SUMMARY ===")
        for cond in conditions:
            frs = [all_results[f"{cond}_seed{s}"].get("forgetting_ratio", float("nan"))
                   for s in seeds if f"{cond}_seed{s}" in all_results]
            valid = [f for f in frs if math.isfinite(f)]
            if valid:
                mu = np.mean(valid)
                sd = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
                print(f"  {cond}: FR = {mu:.3f} +/- {sd:.3f}  (n={len(valid)})")
            else:
                print(f"  {cond}: no valid results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F51 Overnight Benchmark v2")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--tier", default="smoke", choices=list(TIERS.keys()))
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--seeds", default="51")
    parser.add_argument("--phase-a-steps", type=int, default=None)
    parser.add_argument("--phase-b-steps", type=int, default=None)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=9.0)
    parser.add_argument("--plan", action="store_true", default=True)
    parser.add_argument("--launch", dest="plan", action="store_false")
    args = parser.parse_args()
    run(args)
