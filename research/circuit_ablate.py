#!/usr/bin/env python3
"""Descoberta e prova causal de circuitos — CLI fina sobre `f51_darwin.circuits`.

Comandos:
  discover  ranqueia canais do residual stream no split de DESCOBERTA
  run       roda os 5 arms pareados no split de CONFIRMAÇÃO e emite o veredito
  verify    reverifica um relatório em disco, sem tocar em modelo

A lógica vive na biblioteca. Este script só resolve caminhos, materializa
contratos e serializa evidência.

Códigos de saída: 0 sucesso verificado, 2 incompatibilidade de contrato ou
evidência, 4 adulteração ou identidade divergente.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from f51_darwin.circuits.ablation import (  # noqa: E402
    CausalGateConfig,
    CircuitSelector,
    build_causal_verdict,
    discover_residual_channels,
    run_paired_ablation,
)
from f51_darwin.circuits.identity import canonical_sha256  # noqa: E402
from f51_darwin.circuits.manifest import TapContract  # noqa: E402
from f51_darwin.circuits.skills import (  # noqa: E402
    SkillContract,
    build_skill_batches,
    token_pool_identity,
)
from f51_darwin.circuits.taps import CircuitTapError  # noqa: E402
from f51_darwin.transfer.donor import load_frozen_teacher, sha256_file  # noqa: E402


class ContractError(ValueError):
    """Contrato, identidade ou evidência não satisfaz o exigido."""


class TamperError(ValueError):
    """Um artefato em disco não bate com a identidade declarada."""


# ───────────────────────────── carregamento ─────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"arquivo ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_donor(donor: Path, device: str, *, trainable: bool):
    """Carrega o doador verificado.

    `trainable` religa `requires_grad` porque a descoberta pontua canais por
    gradiente × ativação, e `load_frozen_teacher` congela tudo. Nenhum passo
    de otimização é executado: o gradiente é usado só como atribuição.
    """
    model, tokenizer, _ = load_frozen_teacher(donor, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    model.eval()
    return model, tokenizer


def _batches(contract: SkillContract, tokenizer, device: str) -> list[dict[str, Any]]:
    built = build_skill_batches(contract, tokenizer.get_vocab())
    return [
        {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        for batch in built
    ]


def _tap(path: Path) -> TapContract:
    return TapContract.from_dict(_read_json(path))


def _selector_to_dict(selector: CircuitSelector) -> dict[str, Any]:
    return {
        "layer_path": selector.layer_path,
        "channels": list(selector.channels),
        "baseline": list(selector.baseline),
        "discovery_input_sha256": selector.discovery_input_sha256,
        "discovery_item_sha256": list(selector.discovery_item_sha256),
    }


def _selector_from_dict(payload: dict[str, Any]) -> CircuitSelector:
    return CircuitSelector(
        payload["layer_path"],
        tuple(int(value) for value in payload["channels"]),
        tuple(float(value) for value in payload["baseline"]),
        discovery_input_sha256=payload["discovery_input_sha256"],
        discovery_item_sha256=tuple(payload["discovery_item_sha256"]),
    )


def _write_report(path: Path, payload: dict[str, Any]) -> str:
    """Grava relatório com auto-hash, para `verify` detectar adulteração."""
    body = dict(payload)
    body.pop("report_sha256", None)
    digest = canonical_sha256(body)
    body["report_sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return digest


# ─────────────────────────────── comandos ───────────────────────────────


def cmd_discover(args: argparse.Namespace) -> dict[str, Any]:
    donor_sha = sha256_file(args.donor / "donor_manifest.json")
    if args.expected_donor_sha256 and args.expected_donor_sha256 != donor_sha:
        raise TamperError(
            f"donor manifest sha mismatch: esperado {args.expected_donor_sha256}, "
            f"medido {donor_sha}"
        )
    tap = _tap(args.tap_contract)
    contract = SkillContract.from_dict(_read_json(args.discovery_contract))
    if contract.split != "discovery":
        raise ContractError(f"contrato de descoberta tem split={contract.split!r}")

    model, tokenizer = _load_donor(args.donor, args.device, trainable=True)
    batches = _batches(contract, tokenizer, args.device)
    selector = discover_residual_channels(
        model, batches, tap, args.channels, center=args.center
    )

    payload = {
        "schema_version": 1,
        "command": "discover",
        "donor_manifest_sha256": donor_sha,
        "device": args.device,
        "tap": tap.to_dict(),
        "tap_sha256": canonical_sha256(tap.to_dict()),
        "skill_contract": contract.to_dict(),
        "skill_identity": contract.identity,
        "token_pool_sha256": token_pool_identity(
            contract.token_pool, tokenizer.get_vocab()
        ),
        "channels_requested": args.channels,
        "score": "centered_attribution" if args.center else "attribution",
        "selector": _selector_to_dict(selector),
    }
    digest = _write_report(args.output, payload)
    return {
        "status": "DISCOVER_OK",
        "report": str(args.output.resolve()),
        "report_sha256": digest,
        "layer_path": selector.layer_path,
        "channels": len(selector.channels),
    }


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    donor_sha = sha256_file(args.donor / "donor_manifest.json")
    if args.expected_donor_sha256 and args.expected_donor_sha256 != donor_sha:
        raise TamperError("donor manifest sha mismatch")

    discovery_report = _read_json(args.selector)
    if discovery_report.get("donor_manifest_sha256") != donor_sha:
        raise ContractError("o seletor foi descoberto com outro doador")
    selector = _selector_from_dict(discovery_report["selector"])
    tap = TapContract.from_dict(discovery_report["tap"])

    base_contract = SkillContract.from_dict(_read_json(args.confirmation_contract))
    if base_contract.split != "confirmation":
        raise ContractError(
            f"contrato de confirmação tem split={base_contract.split!r}"
        )

    config = CausalGateConfig(minimum_seeds=args.seeds)
    _model, tokenizer = _load_donor(args.donor, args.device, trainable=False)

    def factory():
        model, _tokenizer = _load_donor(args.donor, args.device, trainable=False)
        return model

    runs = []
    per_seed = []
    for offset in range(args.seeds):
        seed = base_contract.seed + offset
        # Trocar a semente muda a identidade do contrato, logo o fluxo de
        # aleatoriedade: cada seed recebe itens de confirmação próprios.
        contract = SkillContract.from_dict({**base_contract.to_dict(), "seed": seed})
        batches = _batches(contract, tokenizer, args.device)
        # O mesmo tap da descoberta: intervir noutro ponto mediria outro circuito.
        result = run_paired_ablation(factory, batches, selector, seed, tap=tap)
        runs.append(result)
        per_seed.append(
            {
                "seed": seed,
                "skill_identity": contract.identity,
                "clean_nll": result.clean.mean_nll,
                "ablate_nll": result.ablate.mean_nll,
                "restore_nll": result.restore.mean_nll,
                "shuffle_nll": result.shuffle.mean_nll,
                "random_matched_nll": result.random_matched.mean_nll,
                "evidence_sha256": result.evidence_sha256,
            }
        )

    verdict = build_causal_verdict(runs, config)
    payload = {
        "schema_version": 1,
        "command": "run",
        "donor_manifest_sha256": donor_sha,
        "device": args.device,
        "tap": tap.to_dict(),
        "selector": _selector_to_dict(selector),
        "discovery_report_sha256": discovery_report["report_sha256"],
        "confirmation_contract": base_contract.to_dict(),
        "seeds": [entry["seed"] for entry in per_seed],
        "per_seed": per_seed,
        "gate_config": {
            "minimum_seeds": config.minimum_seeds,
            "minimum_items_per_domain": config.minimum_items_per_domain,
            "minimum_effect_nats": config.minimum_effect_nats,
            "minimum_control_ratio": config.minimum_control_ratio,
            "minimum_restore_fraction": config.minimum_restore_fraction,
            "minimum_shuffle_fraction": config.minimum_shuffle_fraction,
            "alpha": config.alpha,
        },
        "verdict": {
            "causal": verdict.causal,
            "reason": verdict.reason,
            "target_effect_mean": verdict.target_effect_mean,
            "target_effect_ci_low": verdict.target_effect_ci_low,
            "target_effect_ci_high": verdict.target_effect_ci_high,
            "target_to_control_ratio": verdict.target_to_control_ratio,
            "restore_fraction": verdict.restore_fraction,
            "shuffle_fraction": verdict.shuffle_fraction,
            "holm_rejected": verdict.holm_rejected,
        },
    }
    digest = _write_report(args.output, payload)
    return {
        "status": "ABLATION_CAUSAL" if verdict.causal else "ABLATION_NOT_CAUSAL",
        "causal": verdict.causal,
        "reason": verdict.reason,
        "report": str(args.output.resolve()),
        "report_sha256": digest,
    }


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(args.report)
    declared = payload.get("report_sha256")
    body = {key: value for key, value in payload.items() if key != "report_sha256"}
    recomputed = canonical_sha256(body)
    if declared != recomputed:
        raise TamperError(
            f"report sha mismatch: declarado {declared}, recomputado {recomputed}"
        )
    if args.expected_report_sha256 and args.expected_report_sha256 != declared:
        raise TamperError("report sha differs from the expected identity")
    return {
        "status": "VERIFY_OK",
        "report": str(args.report.resolve()),
        "report_sha256": declared,
        "command": payload.get("command"),
        "causal": payload.get("verdict", {}).get("causal"),
    }


# ──────────────────────────────── parser ────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover")
    discover.add_argument("--donor", type=Path, required=True)
    discover.add_argument("--expected-donor-sha256")
    discover.add_argument("--tap-contract", type=Path, required=True)
    discover.add_argument("--discovery-contract", type=Path, required=True)
    discover.add_argument("--channels", type=int, required=True)
    discover.add_argument(
        "--center",
        action="store_true",
        help="pontua |(ativação − média) × gradiente|, isolando desvio "
        "específico do item em vez de magnitude bruta",
    )
    discover.add_argument("--output", type=Path, required=True)
    discover.add_argument("--device", default="cpu")
    discover.set_defaults(handler=cmd_discover)

    run = sub.add_parser("run")
    run.add_argument("--donor", type=Path, required=True)
    run.add_argument("--expected-donor-sha256")
    run.add_argument("--selector", type=Path, required=True)
    run.add_argument("--confirmation-contract", type=Path, required=True)
    run.add_argument("--seeds", type=int, default=5)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    run.set_defaults(handler=cmd_run)

    verify = sub.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--expected-report-sha256")
    verify.set_defaults(handler=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except TamperError as exc:
        print(json.dumps({"status": "tampered", "error": str(exc)}, sort_keys=True))
        return 4
    except (ContractError, CircuitTapError, ValueError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
