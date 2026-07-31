#!/usr/bin/env python3
"""Remontagem certificada do SmolLM2-1.7B: doador -> doador, tensor por tensor.

Fase 1.1 do roadmap (governance/docs/ROADMAP_RECEPTOR_UNIVERSAL.md, secao
"1.1 Remontagem certificada (baseline de confianca)"). O aparato de
transplante (`f51_darwin.transplant_16b.selection` e `.ledger`) nunca foi
calibrado contra uma resposta conhecida. Este script cria um alvo vazio da
MESMA arquitetura do doador (`LlamaForCausalLM`, d_model 2048, 24 camadas,
vocab 49152), copia cada tensor do doador usando a bijecao identidade
(`Provenance(kind="select", kept=range(n))`), certifica cada copia com
`verify_inheritance`, grava tudo no `CoverageLedger` com `provenance_sha256`
e deriva 0.0, e mede KL(alvo || doador) em >= 8 prompts variados.

Criterio de sucesso declarado ANTES de rodar: KL < 1e-6 em todos os prompts
E 100% dos tensores com certificado de heranca valido.

Criterio de fracasso, declarado antes: qualquer tensor sem certificado, ou
KL acima do limiar. Se isso acontecer o script REPORTA o numero medido; ele
nunca afrouxa o limiar nem a asserção para "passar".

Uso:
    python research/reassemble_certified.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Regra 8 do CLAUDE.md: nao usar cloud como autoridade e nao contatar
# servicos externos. O doador ja esta no disco local; forcar modo offline
# torna isso um invariante do script, nao uma esperanca.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

# Sem mutacao de sys.path: o pacote e instalado em modo editavel
# (pyproject: where = ["src"]) e o pytest recebe pythonpath = ["src"], entao
# f51_darwin importa direto. Inserir src/ no path manualmente e o que
# check_architecture_boundaries sinaliza como sys_path_mutation, e com razao:
# mascara um ambiente mal instalado em vez de falhar nele.
from f51_darwin.transplant_16b.ledger import (
    CoverageLedger,
    CoverageRecord,
    DriftRecord,
)
from f51_darwin.transplant_16b.selection import (
    Provenance,
    tensor_digest,
    verify_inheritance,
)

DEVICE = "cuda:0"  # cuda:1 esta em uso por outro agente; regra explicita da tarefa.

DONOR_ROOT = (
    ROOT
    / "workspace"
    / "00_DONORS"
    / "models--HuggingFaceTB--SmolLM2-1.7B"
    / "snapshots"
    / "effd688a12921b4cc83e3312b6feb579f70f9c71"
)

RUNTIME_ROOT = ROOT / "workspace" / "runtime" / "reassembly"
LEDGER_PATH = RUNTIME_ROOT / "coverage.jsonl"
REPORT_PATH = RUNTIME_ROOT / "report.json"
PLAN_ID = "reassemble-certified-smollm2-1.7b-identity-v1"

# >= 8 prompts variados: raciocinio, causalidade, matematica, codigo, fato,
# ciencia, narrativa, continuacao de codigo. Os 6 primeiros sao os prompts
# de calibracao ja usados em f51_darwin.transplant_16b.contracts; os demais
# sao novos, para nao herdar cegamente a cobertura de outro experimento.
PROMPTS: tuple[str, ...] = (
    "O futuro da inteligencia artificial depende de",
    "Explique por que a agua ferve.",
    "If all birds have wings and a robin is a bird, then",
    "Solve 3x + 7 = 22.",
    "Write a Python function that reverses a list.",
    "A causal model differs from correlation because",
    "The capital of France is",
    "In quantum mechanics, superposition means",
    "def fibonacci(n):",
    "Yesterday I went to the market and bought",
)

KL_THRESHOLD = 1e-6
CERTIFIED_FRACTION_THRESHOLD = 1.0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not DONOR_ROOT.exists():
        raise FileNotFoundError(f"donor snapshot not found: {DONOR_ROOT}")

    donor = AutoModelForCausalLM.from_pretrained(
        str(DONOR_ROOT), local_files_only=True, dtype=torch.bfloat16
    )
    donor.to(DEVICE)
    donor.eval()

    # Alvo vazio da MESMA arquitetura: from_config faz init aleatorio, entao
    # antes da copia o alvo NAO tem relacao com o doador. A copia certificada
    # e a unica coisa que pode fazer os pesos coincidirem depois.
    target = AutoModelForCausalLM.from_config(donor.config, dtype=torch.bfloat16)
    target.to(DEVICE)
    target.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(DONOR_ROOT), local_files_only=True)
    return donor, target, tokenizer


def _reassemble(donor, target, ledger: CoverageLedger) -> list[dict[str, Any]]:
    donor_state = donor.state_dict()
    target_state = target.state_dict()

    donor_names = set(donor_state.keys())
    target_names = set(target_state.keys())
    if donor_names != target_names:
        raise ValueError(
            "donor and target state_dict keys differ: "
            f"only_donor={sorted(donor_names - target_names)} "
            f"only_target={sorted(target_names - donor_names)}"
        )

    records: list[dict[str, Any]] = []
    for name in sorted(donor_names):
        donor_tensor = donor_state[name]
        target_tensor = target_state[name]

        if donor_tensor.ndim < 1:
            raise ValueError(f"tensor {name!r} is 0-dim; identity axis undefined")
        if donor_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"shape mismatch for {name!r}: donor={tuple(donor_tensor.shape)} "
                f"target={tuple(target_tensor.shape)}"
            )

        pre_copy_matched = tensor_digest(target_tensor) == tensor_digest(donor_tensor)

        with torch.no_grad():
            target_tensor.data.copy_(donor_tensor.data)

        source_size = int(donor_tensor.shape[0])
        provenance = Provenance(
            kind="select",
            axis="full_tensor",
            source_size=source_size,
            kept=tuple(range(source_size)),
        )
        certificate = verify_inheritance(donor_tensor, target_tensor, provenance, dim=0)

        metrics: dict[str, Any] = {
            "shape": list(donor_tensor.shape),
            "dtype": str(donor_tensor.dtype),
            "pre_copy_matched_donor": bool(pre_copy_matched),
            "certificate_valid": bool(certificate.valid),
            "donor_sha256": certificate.donor_sha256,
            "expected_sha256": certificate.expected_sha256,
            "actual_sha256": certificate.actual_sha256,
        }

        if certificate.valid:
            provenance_sha = certificate.provenance_sha256
            drift = DriftRecord("l2", 0.0, measured_on="byte_identity")
        else:
            l2 = float((target_tensor.float() - donor_tensor.float()).norm().item())
            provenance_sha = None
            drift = DriftRecord("l2", l2, measured_on=name)

        ledger.append(
            CoverageRecord(
                source_tensor=name,
                target_tensors=(name,),
                method="identity_reassembly",
                status="complete",
                metrics=metrics,
                provenance_sha256=provenance_sha,
                drift=drift,
            )
        )
        records.append({"tensor": name, **metrics})

    return records


@torch.no_grad()
def _measure_kl(donor, target, tokenizer) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        donor_logits = donor(ids).logits.float()
        target_logits = target(ids).logits.float()

        token_count = int(ids.shape[0] * ids.shape[1])
        kl_sum = float(
            F.kl_div(
                F.log_softmax(target_logits, dim=-1),
                F.softmax(donor_logits, dim=-1),
                reduction="sum",
            ).item()
        )
        mean_kl = kl_sum / token_count
        max_abs_logit_error = float((target_logits - donor_logits).abs().max().item())
        top1_agreement = float(
            (target_logits.argmax(dim=-1) == donor_logits.argmax(dim=-1))
            .float()
            .mean()
            .item()
        )
        results.append(
            {
                "prompt": prompt,
                "token_count": token_count,
                "kl_sum": kl_sum,
                "kl_mean_per_token": mean_kl,
                "max_abs_logit_error": max_abs_logit_error,
                "top1_agreement": top1_agreement,
            }
        )
    return results


def main() -> dict[str, Any]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    # Este ledger e propriedade exclusiva deste script (workspace/runtime e
    # estado descartavel de laboratorio, regra 3 do CLAUDE.md); recomeco
    # limpo a cada execucao para que reruns nao colidam com "source tensor
    # already classified".
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()

    print(f"[reassemble] donor root: {DONOR_ROOT}", flush=True)
    donor_weights_path = DONOR_ROOT / "model.safetensors"
    donor_weights_sha256 = _sha256_file(donor_weights_path)
    print(f"[reassemble] donor model.safetensors sha256: {donor_weights_sha256}", flush=True)

    donor, target, tokenizer = _load_models()
    print(
        f"[reassemble] donor params: {sum(p.numel() for p in donor.parameters())} "
        f"target params: {sum(p.numel() for p in target.parameters())}",
        flush=True,
    )

    ledger = CoverageLedger.open(LEDGER_PATH, plan_id=PLAN_ID)

    start = time.time()
    tensor_records = _reassemble(donor, target, ledger)
    reassembly_seconds = time.time() - start

    total_tensors = len(tensor_records)
    certified_count = sum(1 for r in tensor_records if r["certificate_valid"])
    certified_fraction = certified_count / total_tensors if total_tensors else 0.0
    failed_tensors = [r["tensor"] for r in tensor_records if not r["certificate_valid"]]
    pre_copy_already_matched = [
        r["tensor"] for r in tensor_records if r["pre_copy_matched_donor"]
    ]

    print(
        f"[reassemble] tensors={total_tensors} certified={certified_count} "
        f"fraction={certified_fraction:.6f} time={reassembly_seconds:.1f}s",
        flush=True,
    )

    kl_results = _measure_kl(donor, target, tokenizer)
    kl_means = [r["kl_mean_per_token"] for r in kl_results]
    kl_mean = sum(kl_means) / len(kl_means)
    kl_max = max(kl_means)
    max_logit_error = max(r["max_abs_logit_error"] for r in kl_results)
    min_top1_agreement = min(r["top1_agreement"] for r in kl_results)

    for r in kl_results:
        print(
            f"[reassemble] KL prompt={r['prompt']!r} "
            f"kl_mean_per_token={r['kl_mean_per_token']:.10e} "
            f"max_abs_logit_error={r['max_abs_logit_error']:.6e} "
            f"top1_agreement={r['top1_agreement']:.6f}",
            flush=True,
        )

    passed = (
        certified_fraction >= CERTIFIED_FRACTION_THRESHOLD
        and not failed_tensors
        and kl_max < KL_THRESHOLD
    )

    notes: list[str] = []
    if pre_copy_already_matched:
        notes.append(
            "tensores cujo init aleatorio ja coincidia com o doador antes da "
            f"copia (nao deveria acontecer para pesos treinados): {pre_copy_already_matched}"
        )
    if failed_tensors:
        notes.append(f"tensores sem certificado valido: {failed_tensors}")
    if kl_max >= KL_THRESHOLD:
        notes.append(
            f"KL maximo medido ({kl_max:.10e}) nao ficou abaixo do limiar "
            f"declarado ({KL_THRESHOLD:.1e})"
        )

    report = {
        "schema": "reassemble-certified-smollm2-1.7b-v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": DEVICE,
        "plan_id": PLAN_ID,
        "donor_root": str(DONOR_ROOT),
        "donor_weights_sha256": donor_weights_sha256,
        "reassembly_seconds": reassembly_seconds,
        "tensor_count": total_tensors,
        "certified_count": certified_count,
        "certified_fraction": certified_fraction,
        "failed_tensors": failed_tensors,
        "ledger_path": str(LEDGER_PATH),
        "ledger_head": ledger.head,
        "ledger_record_count": len(ledger.records),
        "prompts": list(PROMPTS),
        "kl_by_prompt": kl_results,
        "kl_mean": kl_mean,
        "kl_max": kl_max,
        "max_abs_logit_error": max_logit_error,
        "min_top1_agreement": min_top1_agreement,
        "success_criteria": {
            "kl_threshold": KL_THRESHOLD,
            "certified_fraction_threshold": CERTIFIED_FRACTION_THRESHOLD,
        },
        "passed": passed,
        "notes": notes,
    }
    _atomic_json(REPORT_PATH, report)
    print(f"[reassemble] report written to {REPORT_PATH}", flush=True)
    print(
        f"[reassemble] PASSED={passed} kl_mean={kl_mean:.10e} kl_max={kl_max:.10e} "
        f"certified_fraction={certified_fraction:.6f}",
        flush=True,
    )

    del donor, target
    gc.collect()
    torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["passed"] else 1)
