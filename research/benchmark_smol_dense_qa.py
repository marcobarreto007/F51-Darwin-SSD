#!/usr/bin/env python3
"""Paired, deterministic QA benchmark for SmolLM2 and Darwin Dense V1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.transplant_16b.checkpoint import verify_shard_manifest
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
from f51_darwin.transplant_16b.dense_assembly import _assert_dense_structure
from f51_darwin.transplant_16b.exact_assembly import _new_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
)
MANIFEST = CHECKPOINT.with_suffix(".manifest.json")
RUNTIME_ROOT = ROOT / "workspace/runtime/darwin_17b_smol_dense_v1"
JSON_REPORT = RUNTIME_ROOT / "qa-benchmark.json"
MARKDOWN_REPORT = RUNTIME_ROOT / "qa-benchmark.md"
SYSTEM_PROMPT = (
    "Você é um assistente preciso. Siga exatamente a instrução do usuário "
    "e não invente informações."
)


MCQ: tuple[dict[str, Any], ...] = (
    {
        "id": "knowledge-01",
        "category": "conhecimento",
        "question": "Qual é a capital do Brasil?",
        "options": ("Rio de Janeiro", "São Paulo", "Salvador", "Brasília"),
        "answer": "D",
    },
    {
        "id": "knowledge-02",
        "category": "conhecimento",
        "question": "Qual é a fórmula química da água?",
        "options": ("H2O", "O2", "CO2", "NaCl"),
        "answer": "A",
    },
    {
        "id": "knowledge-03",
        "category": "conhecimento",
        "question": "Qual é o maior planeta do Sistema Solar?",
        "options": ("Marte", "Terra", "Saturno", "Júpiter"),
        "answer": "D",
    },
    {
        "id": "knowledge-04",
        "category": "conhecimento",
        "question": "Quem escreveu o romance Dom Casmurro?",
        "options": (
            "Machado de Assis",
            "José de Alencar",
            "Clarice Lispector",
            "Carlos Drummond de Andrade",
        ),
        "answer": "A",
    },
    {
        "id": "portuguese-01",
        "category": "português",
        "question": "Qual é o plural correto de cidadão?",
        "options": ("cidadãos", "cidadães", "cidadões", "cidadans"),
        "answer": "A",
    },
    {
        "id": "portuguese-02",
        "category": "português",
        "question": "Qual palavra é antônimo de escasso?",
        "options": ("raro", "pequeno", "frágil", "abundante"),
        "answer": "D",
    },
    {
        "id": "portuguese-03",
        "category": "português",
        "question": "Qual palavra está grafada corretamente?",
        "options": ("excessão", "exceção", "esceção", "excesssão"),
        "answer": "B",
    },
    {
        "id": "portuguese-04",
        "category": "português",
        "question": (
            'Na frase "Apesar da chuva, ela saiu", a expressão inicial '
            "indica qual relação?"
        ),
        "options": ("causa", "finalidade", "concessão", "conclusão"),
        "answer": "C",
    },
    {
        "id": "math-01",
        "category": "matemática",
        "question": "Quanto é 17 multiplicado por 8?",
        "options": ("126", "136", "146", "156"),
        "answer": "B",
    },
    {
        "id": "math-02",
        "category": "matemática",
        "question": "Qual é o resultado de (45 dividido por 5) mais 7?",
        "options": ("16", "15", "14", "17"),
        "answer": "A",
    },
    {
        "id": "math-03",
        "category": "matemática",
        "question": "Se 3x + 5 = 20, qual é o valor de x?",
        "options": ("3", "4", "5", "6"),
        "answer": "C",
    },
    {
        "id": "math-04",
        "category": "matemática",
        "question": "Quanto é 2 elevado à quinta potência?",
        "options": ("10", "16", "25", "32"),
        "answer": "D",
    },
    {
        "id": "reasoning-01",
        "category": "raciocínio",
        "question": (
            "Todos os mamíferos têm sangue quente. Uma baleia é mamífero. "
            "Logo, o que é necessariamente verdadeiro?"
        ),
        "options": (
            "A baleia é um peixe",
            "A baleia tem sangue quente",
            "Todo animal marinho é mamífero",
            "A baleia vive em terra",
        ),
        "answer": "B",
    },
    {
        "id": "reasoning-02",
        "category": "raciocínio",
        "question": "Qual é o próximo número: 2, 6, 12, 20, ...?",
        "options": ("26", "28", "30", "32"),
        "answer": "C",
    },
    {
        "id": "reasoning-03",
        "category": "raciocínio",
        "question": (
            "Ana é mais velha que Bia, e Bia é mais velha que Carla. "
            "Quem é a mais velha?"
        ),
        "options": ("Ana", "Bia", "Carla", "Não é possível saber"),
        "answer": "A",
    },
    {
        "id": "reasoning-04",
        "category": "raciocínio",
        "question": "Qual item não pertence ao mesmo grupo dos demais?",
        "options": ("triângulo", "quadrado", "círculo", "cubo"),
        "answer": "D",
    },
    {
        "id": "code-01",
        "category": "programação",
        "question": "Em Python, qual é o resultado de len([1, 2, 3])?",
        "options": ("2", "3", "4", "Erro"),
        "answer": "B",
    },
    {
        "id": "code-02",
        "category": "programação",
        "question": "Em Python, qual é o valor de bool(0)?",
        "options": ("True", "False", "0.0", "None"),
        "answer": "B",
    },
    {
        "id": "code-03",
        "category": "programação",
        "question": "Em Python, o que list(range(2, 5)) produz?",
        "options": ("[2, 3, 4]", "[2, 3, 4, 5]", "[1, 2, 3, 4]", "[2, 5]"),
        "answer": "A",
    },
    {
        "id": "code-04",
        "category": "programação",
        "question": "Qual comando SQL seleciona todas as colunas da tabela clientes?",
        "options": (
            "GET ALL FROM clientes",
            "SELECT ALL clientes",
            "READ clientes",
            "SELECT * FROM clientes",
        ),
        "answer": "D",
    },
    {
        "id": "english-01",
        "category": "inglês",
        "question": 'Qual é o passado do verbo inglês "go"?',
        "options": ("goed", "gone", "went", "goes"),
        "answer": "C",
    },
    {
        "id": "english-02",
        "category": "inglês",
        "question": 'Qual palavra é sinônimo de "rapid" em inglês?',
        "options": ("slow", "fast", "late", "weak"),
        "answer": "B",
    },
    {
        "id": "english-03",
        "category": "inglês",
        "question": 'Qual é a tradução de "book" para o português?',
        "options": ("mesa", "caneta", "livro", "porta"),
        "answer": "C",
    },
    {
        "id": "english-04",
        "category": "inglês",
        "question": "Qual frase em inglês está gramaticalmente correta?",
        "options": (
            "She don't like coffee.",
            "She doesn't likes coffee.",
            "She doesn't like coffee.",
            "She not like coffee.",
        ),
        "answer": "C",
    },
)

OPEN_QA: tuple[dict[str, Any], ...] = (
    {
        "id": "open-math",
        "category": "matemática",
        "question": "Quanto é 23 + 19?",
        "accepted": ("42",),
    },
    {
        "id": "open-knowledge-01",
        "category": "conhecimento",
        "question": "Qual é a capital do Canadá?",
        "accepted": ("ottawa",),
    },
    {
        "id": "open-knowledge-02",
        "category": "conhecimento",
        "question": "Quem escreveu o romance 1984?",
        "accepted": ("george orwell", "orwell"),
    },
    {
        "id": "open-code",
        "category": "programação",
        "question": "Em Python, qual é o resultado de sum([2, 3, 5])?",
        "accepted": ("10",),
    },
    {
        "id": "open-reasoning",
        "category": "raciocínio",
        "question": "Complete a sequência: 1, 1, 2, 3, 5, ...",
        "accepted": ("8",),
    },
    {
        "id": "open-translation",
        "category": "inglês",
        "question": 'Traduza "good morning" para o português.',
        "accepted": ("bom dia",),
    },
)


def normalize_answer(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def extract_choice(value: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", value.upper())
    return match.group(1) if match else None


def accepted_answer(value: str, accepted: Sequence[str]) -> bool:
    normalized = normalize_answer(value)
    for expected in accepted:
        token = normalize_answer(expected)
        if re.search(rf"(?:^|\s){re.escape(token)}(?:\s|$)", normalized):
            return True
    return False


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prompt(tokenizer: Any, item: dict[str, Any], kind: str) -> torch.Tensor:
    if kind == "mcq":
        rendered_options = "\n".join(
            f"{letter}) {option}"
            for letter, option in zip("ABCD", item["options"], strict=True)
        )
        user = (
            f"{item['question']}\n{rendered_options}\n"
            "Responda somente com uma letra: A, B, C ou D."
        )
    else:
        user = (
            f"{item['question']}\n"
            "Responda apenas com a resposta curta, sem explicação."
        )
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids


@torch.inference_mode()
def _greedy(
    forward: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    eos_token_id: int | None,
    max_new_tokens: int,
) -> tuple[list[int], float]:
    generated = input_ids.to(device)
    new_tokens: list[int] = []
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = forward(generated)
        next_token = int(logits[0, -1].float().argmax().item())
        new_tokens.append(next_token)
        generated = torch.cat(
            (
                generated,
                torch.tensor([[next_token]], dtype=torch.long, device=device),
            ),
            dim=1,
        )
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return new_tokens, time.perf_counter() - started


def _run_items(
    *,
    candidate_name: str,
    tokenizer: Any,
    forward: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    open_tokens: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in MCQ:
        input_ids = _prompt(tokenizer, item, "mcq")
        tokens, elapsed = _greedy(
            forward,
            input_ids,
            device=device,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1,
        )
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        parsed = extract_choice(raw)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "kind": "mcq",
                "response": raw,
                "parsed": parsed,
                "reference": item["answer"],
                "correct": parsed == item["answer"],
                "valid": parsed is not None,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[{candidate_name}] {item['id']} "
            f"answer={parsed or 'INVALID'} correct={parsed == item['answer']}",
            flush=True,
        )
    for item in OPEN_QA:
        input_ids = _prompt(tokenizer, item, "open")
        tokens, elapsed = _greedy(
            forward,
            input_ids,
            device=device,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=open_tokens,
        )
        raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        correct = accepted_answer(raw, item["accepted"])
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "kind": "open",
                "response": raw,
                "parsed": normalize_answer(raw),
                "reference": list(item["accepted"]),
                "correct": correct,
                "valid": bool(raw),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[{candidate_name}] {item['id']} "
            f"answer={raw!r} correct={correct}",
            flush=True,
        )
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    return {
        "name": candidate_name,
        "rows": rows,
        "mcq_correct": sum(row["correct"] for row in rows if row["kind"] == "mcq"),
        "mcq_total": len(MCQ),
        "open_correct": sum(row["correct"] for row in rows if row["kind"] == "open"),
        "open_total": len(OPEN_QA),
        "overall_correct": sum(row["correct"] for row in rows),
        "overall_total": len(rows),
        "valid_responses": sum(row["valid"] for row in rows),
        "latency_seconds": {
            "total": sum(latencies),
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
        },
    }


def _candidate(tokenizer: Any, open_tokens: int) -> dict[str, Any]:
    verify_shard_manifest(CHECKPOINT, MANIFEST)
    payload = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = DarwinXConfig.from_mapping(payload["config"])
    model = _new_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    _assert_dense_structure(model, expected_layers=config.n_layers)
    model.to(dtype=torch.bfloat16)
    if not model.enable_dual_gpu(gpu0=0, gpu1=1):
        raise RuntimeError("Darwin QA benchmark requires both GPUs")
    model.eval()
    result = _run_items(
        candidate_name="darwin-smol-native-dense-v1",
        tokenizer=tokenizer,
        forward=lambda ids: model(ids, heartbeat=False).logits,
        device=torch.device("cuda:0"),
        open_tokens=open_tokens,
    )
    result["split_layer"] = int(model._split_layer)
    result["moe_modules"] = sum(block.moe is not None for block in model.blocks)
    result["dense_ffn_modules"] = sum(block.ffn is not None for block in model.blocks)
    del model, payload
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _donor(tokenizer: Any, open_tokens: int) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_SOURCE_ROOT,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()
    result = _run_items(
        candidate_name="smollm2-1.7b-instruct",
        tokenizer=tokenizer,
        forward=lambda ids: model(ids).logits,
        device=torch.device("cuda:0"),
        open_tokens=open_tokens,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _environment() -> dict[str, Any]:
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
    except Exception:
        ram_bytes = 0
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def _comparison(
    darwin: dict[str, Any],
    donor: dict[str, Any],
) -> dict[str, Any]:
    donor_by_id = {row["id"]: row for row in donor["rows"]}
    comparisons = []
    for row in darwin["rows"]:
        baseline = donor_by_id[row["id"]]
        comparisons.append(
            {
                "id": row["id"],
                "category": row["category"],
                "kind": row["kind"],
                "darwin_response": row["response"],
                "donor_response": baseline["response"],
                "darwin_correct": row["correct"],
                "donor_correct": baseline["correct"],
                "answer_equal": row["parsed"] == baseline["parsed"],
            }
        )
    return {
        "answer_equal": sum(row["answer_equal"] for row in comparisons),
        "correctness_equal": sum(
            row["darwin_correct"] == row["donor_correct"]
            for row in comparisons
        ),
        "total": len(comparisons),
        "darwin_wins": sum(
            row["darwin_correct"] and not row["donor_correct"]
            for row in comparisons
        ),
        "donor_wins": sum(
            row["donor_correct"] and not row["darwin_correct"]
            for row in comparisons
        ),
        "rows": comparisons,
    }


def _markdown(payload: dict[str, Any]) -> str:
    darwin = payload["candidates"]["darwin"]
    donor = payload["candidates"]["donor"]
    comparison = payload["comparison"]
    lines = [
        "# Benchmark pareado de perguntas e respostas",
        "",
        f"Data: {payload['created_at']}",
        "",
        "## Resultado",
        "",
        "| Modelo | Objetivas | Abertas | Total | Respostas válidas |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Darwin Dense V1 | {darwin['mcq_correct']}/{darwin['mcq_total']} "
            f"| {darwin['open_correct']}/{darwin['open_total']} "
            f"| {darwin['overall_correct']}/{darwin['overall_total']} "
            f"| {darwin['valid_responses']}/{darwin['overall_total']} |"
        ),
        (
            f"| SmolLM2 doador | {donor['mcq_correct']}/{donor['mcq_total']} "
            f"| {donor['open_correct']}/{donor['open_total']} "
            f"| {donor['overall_correct']}/{donor['overall_total']} "
            f"| {donor['valid_responses']}/{donor['overall_total']} |"
        ),
        "",
        (
            f"Respostas iguais: {comparison['answer_equal']}/"
            f"{comparison['total']}. Decisões de acerto iguais: "
            f"{comparison['correctness_equal']}/{comparison['total']}. "
            f"Darwin venceu {comparison['darwin_wins']}; "
            f"Smol venceu {comparison['donor_wins']}."
        ),
        "",
        "## Respostas lado a lado",
        "",
        "| ID | Darwin | Smol | D correto | S correto | Igual |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        darwin_text = str(row["darwin_response"]).replace("|", "\\|")
        donor_text = str(row["donor_response"]).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {darwin_text} | {donor_text} "
            f"| {'sim' if row['darwin_correct'] else 'não'} "
            f"| {'sim' if row['donor_correct'] else 'não'} "
            f"| {'sim' if row['answer_equal'] else 'não'} |"
        )
    lines.extend(
        [
            "",
            "## Protocolo",
            "",
            "- 24 questões objetivas e 6 respostas curtas.",
            "- Seis categorias: conhecimento, português, matemática, raciocínio, programação e inglês.",
            "- Mesmo tokenizer, template de chat e decodificação greedy.",
            "- Temperatura zero; nenhuma amostragem; avaliação determinística.",
            "- Darwin carregado donor-free; Smol carregado separadamente depois de liberar o Darwin.",
            "- Latência registrada apenas como diagnóstico: Darwin usa duas GPUs e Smol uma, portanto não é comparação de velocidade justa.",
            "",
            "Reprodução:",
            "",
            "```powershell",
            "python research/benchmark_smol_dense_qa.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real QA answers from SmolLM2 and Darwin Dense V1."
    )
    parser.add_argument("--open-tokens", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.open_tokens < 1 or args.open_tokens > 32:
        raise ValueError("--open-tokens must be between 1 and 32")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("paired QA benchmark requires both local GPUs")
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_SOURCE_ROOT,
        local_files_only=True,
    )
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    darwin = _candidate(tokenizer, args.open_tokens)
    donor = _donor(tokenizer, args.open_tokens)
    checkpoint_sha = json.loads(MANIFEST.read_text(encoding="utf-8"))[
        "checkpoint_sha256"
    ]
    payload = {
        "schema": "darwin-smol-paired-qa-benchmark-v1",
        "created_at": created_at,
        "protocol_sha256": hashlib.sha256(
            json.dumps(
                {"mcq": MCQ, "open": OPEN_QA, "system": SYSTEM_PROMPT},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha,
        "donor": str(DEFAULT_SOURCE_ROOT),
        "decoding": {
            "strategy": "greedy",
            "temperature": 0.0,
            "mcq_tokens": 1,
            "open_tokens": args.open_tokens,
        },
        "environment": _environment(),
        "candidates": {"darwin": darwin, "donor": donor},
        "comparison": _comparison(darwin, donor),
    }
    _atomic_json(JSON_REPORT, payload)
    MARKDOWN_REPORT.write_text(_markdown(payload), encoding="utf-8")
    comparison = payload["comparison"]
    print(
        "DENSE_QA_BENCHMARK_OK "
        f"darwin={darwin['overall_correct']}/{darwin['overall_total']} "
        f"donor={donor['overall_correct']}/{donor['overall_total']} "
        f"equal={comparison['answer_equal']}/{comparison['total']} "
        f"darwin_wins={comparison['darwin_wins']} "
        f"donor_wins={comparison['donor_wins']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
