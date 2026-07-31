#!/usr/bin/env python
"""F51 Neural Organism — The heartbeat CLI.

Runs the full organism lifecycle:
    bootstrap  → initialize all organs
    cycle      → run one evolution cycle
    status     → organism health report
    generate   → generate text from the cortex
    extract    → extract project knowledge (grounded)

Usage:
    python scripts/run_neural_organism.py bootstrap
    python scripts/run_neural_organism.py cycle --extract --max-steps 500
    python scripts/run_neural_organism.py status
    python scripts/run_neural_organism.py generate --prompt "F51 Darwin-SSD is"
    python scripts/run_neural_organism.py extract --source f51_darwin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from f51_darwin.config import DarwinConfig
from f51_darwin.grounded_extractor import GroundedExtractor
from f51_darwin.organism import F51NeuralOrganism, OrganismConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="F51 Neural Organism — heartbeat CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_neural_organism.py bootstrap
  python scripts/run_neural_organism.py cycle --extract
  python scripts/run_neural_organism.py cycle --generate --prompt "The F51 architecture"
  python scripts/run_neural_organism.py status
  python scripts/run_neural_organism.py generate --prompt "def hello():"
  python scripts/run_neural_organism.py extract --source f51_darwin
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    bootstrap_parser = sub.add_parser("bootstrap", help="Initialize all organs")
    bootstrap_parser.add_argument(
        "--organism-config", default=str(ROOT / "configs" / "organism.yaml"),
        help="Organism config YAML"
    )
    bootstrap_parser.add_argument(
        "--model-config", default=None,
        help="Optional legacy Darwin-SSD model config. Inline defaults are used when omitted."
    )

    # cycle
    cycle_parser = sub.add_parser("cycle", help="Run one organism cycle")
    cycle_parser.add_argument("--extract", action="store_true",
                              help="Extract project-grounded candidates")
    cycle_parser.add_argument("--generate", action="store_true",
                              help="Generate synthetic candidates from Darwin-SSD")
    cycle_parser.add_argument("--prompt", action="append", default=[],
                              help="Prompts for generation (repeatable)")
    cycle_parser.add_argument("--max-generations", type=int, default=50)
    cycle_parser.add_argument("--max-steps", type=int, default=None,
                              help="Override max steps per cycle")
    cycle_parser.add_argument("--organism-config",
                              default=str(ROOT / "configs" / "organism.yaml"))

    # status
    sub.add_parser("status", help="Organism health report")

    # generate
    gen_parser = sub.add_parser("generate", help="Generate text from the cortex")
    gen_parser.add_argument("--prompt", required=True, help="Input prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=256)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--tokenizer", default="tokenizer/f51_bpe",
                            help="Tokenizer directory")

    # extract
    ext_parser = sub.add_parser("extract", help="Extract grounded project knowledge")
    ext_parser.add_argument("--source", default="f51_darwin",
                            help="Source directory to extract from")
    ext_parser.add_argument("--type", dest="extract_type", default="module_contract",
                            choices=["module_contract", "architecture_note",
                                     "test_behavior", "decision_record"])

    return parser


def load_organism_config(path: str) -> OrganismConfig:
    cfg_path = Path(path)
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        raw["project_root"] = str(ROOT)
        return OrganismConfig(**{k: v for k, v in raw.items()
                                if k in OrganismConfig.__dataclass_fields__})
    return OrganismConfig(project_root=str(ROOT))


def load_model_config(path: str | None) -> DarwinConfig:
    if path is None:
        return DarwinConfig()
    cfg_path = Path(path)
    if cfg_path.exists():
        return DarwinConfig.from_yaml(cfg_path)
    raise FileNotFoundError(f"Model config not found: {cfg_path}")


def cmd_bootstrap(args) -> int:
    org_cfg = load_organism_config(args.organism_config)
    model_cfg = load_model_config(args.model_config)

    organism = F51NeuralOrganism(org_cfg, model_config=model_cfg)
    status = organism.bootstrap()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def cmd_cycle(args) -> int:
    org_cfg = load_organism_config(args.organism_config)

    organism = F51NeuralOrganism(org_cfg)
    status = organism.bootstrap()
    print("=== BOOTSTRAP ===")
    print(json.dumps(status, indent=2))

    if args.max_steps:
        organism.config.max_steps_per_cycle = args.max_steps

    print(f"\n=== CYCLE {organism.cycle + 1} ===")
    report = organism.run_cycle(
        extract_from_project=args.extract,
        generate_synthetic=args.generate,
        synthetic_prompts=args.prompt if args.prompt else None,
        max_generations=args.max_generations,
    )

    print(json.dumps({
        "cycle": report.cycle,
        "status": report.status,
        "steps_trained": report.steps_trained,
        "train_loss_start": report.train_loss_start,
        "train_loss_end": report.train_loss_end,
        "eval_loss": report.eval_loss,
        "replay_loss": report.replay_loss,
        "forgetting_proxy": report.forgetting_proxy,
        "candidates_generated": report.candidates_generated,
        "candidates_approved": report.candidates_approved,
        "candidates_rejected": report.candidates_rejected,
        "modules_active": report.modules_active,
        "modules_died": report.modules_died,
        "brainstem_alive": report.brainstem_alive,
        "warnings": report.warnings,
        "violations": report.violations,
        "checkpoint": report.checkpoint_path,
    }, indent=2, sort_keys=True))

    organism.save_state()
    return 0 if report.status == "completed" else 1


def cmd_status(args) -> int:
    org_cfg = OrganismConfig(project_root=str(ROOT))
    organism = F51NeuralOrganism(org_cfg)
    status = organism.bootstrap()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def cmd_generate(args) -> int:
    from f51_darwin.config import DarwinConfig
    from f51_darwin.model import F51DarwinModel

    model_cfg = load_model_config(None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load trained model from checkpoint
    checkpoint_path = ROOT / "checkpoints" / "base" / "latest.json"
    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text())
        ckpt = ROOT / payload["path"]
        if ckpt.exists():
            from f51_darwin.checkpointing import load_model_from_checkpoint
            model, model_cfg, _ = load_model_from_checkpoint(ckpt, map_location=device)
        else:
            model = F51DarwinModel(model_cfg).to(device)
    else:
        model = F51DarwinModel(model_cfg).to(device)

    # Load tokenizer
    tokenizer = None
    tokenizer_dir = ROOT / args.tokenizer
    if tokenizer_dir.exists() and (tokenizer_dir / "vocab.json").exists():
        from f51_darwin.tokenizer import F51BPETokenizer
        tokenizer = F51BPETokenizer.load(tokenizer_dir)

    output = model.generate(
        args.prompt,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(json.dumps({
        "prompt": args.prompt,
        "generated": output.text,
        "tokens": len(output.token_ids),
        "finish_reason": output.finish_reason,
    }, indent=2, sort_keys=True))

    return 0


def cmd_extract(args) -> int:
    extractor = GroundedExtractor(ROOT)
    extractions = extractor.extract_project_knowledge(
        args.source,
        extraction_types=[args.extract_type],
    )

    results = []
    for ext in extractions:
        results.append({
            "id": ext.id,
            "topic": ext.topic,
            "epistemic_level": ext.epistemic_level.value,
            "source_file": ext.evidence.source_file,
            "content_preview": ext.content[:200] + "..." if len(ext.content) > 200 else ext.content,
        })

    print(json.dumps({
        "source": args.source,
        "extraction_type": args.extract_type,
        "count": len(results),
        "results": results,
    }, indent=2, sort_keys=True))

    return 0


def main() -> int:
    args = build_parser().parse_args()

    commands = {
        "bootstrap": cmd_bootstrap,
        "cycle": cmd_cycle,
        "status": cmd_status,
        "generate": cmd_generate,
        "extract": cmd_extract,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
