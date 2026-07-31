#!/usr/bin/env python3
"""F51 Darwin-SSD - Diagnostico de checkpoint (FASE 0).

Este script nao assume qual linhagem e "a correta". Ele carrega o checkpoint
mais recente (ou o apontado por organism_latest.json), conta parametros reais,
detecta init aleatorio vs treinado, compara contra a config canonica e contra as
afirmacoes de docs/operacao/STATUS_ATUAL.md, e grava achados em
.f51/baseline/checkpoint_info.json.

Uso:
    python scripts/verify_checkpoint_25b.py
    python scripts/verify_checkpoint_25b.py --checkpoint <path.pt>
    python scripts/verify_checkpoint_25b.py --no-load   # so metadados, sem torch.load

Background:
    O CLAUDE.md original tratava o 2.5B como alvo. O STATUS_ATUAL.md
    (2026-07-15) e o AGENTS.md atualizados definem o alvo local como
    F51-Darwin-X-1.6B-Nitro (1.764B parametros), reservando o 2.5B para a
    nuvem. Este diagnostico reporta FATOS e deixa a interpretacao humana
    decidir o que e discrepancia.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "darwin_x_1.6b_nitro.yaml"
STATUS_PATH = REPO_ROOT / "docs" / "operacao" / "STATUS_ATUAL.md"
OUT_DIR = REPO_ROOT / ".f51" / "baseline"
OUT_PATH = OUT_DIR / "checkpoint_info.json"


# ---------------------------------------------------------------------------
# Workspace resolution (corpo fora do repo, conforme contrato AGENTS.md)
# ---------------------------------------------------------------------------

def resolve_workspace() -> Path:
    """Resolve o workspace externo de checkpoints."""
    import os

    root = os.environ.get("F51_DATASET_ROOT")
    if root:
        p = Path(root)
        if p.exists():
            return p
    sibling = REPO_ROOT.parent / "F51-Dataset-Organizado"
    if sibling.exists():
        return sibling
    # fallback: junction legado dentro do repo
    legacy = REPO_ROOT / "data"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        "Workspace externo nao encontrado. Defina F51_DATASET_ROOT ou garanta que "
        "../F51-Dataset-Organizado exista."
    )


def find_checkpoints(workspace: Path) -> list[Path]:
    ckpt_dir = workspace / "03_CHECKPOINTS"
    if not ckpt_dir.exists():
        return []
    return sorted(
        ckpt_dir.glob("organism_cycle_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )


def read_latest_pointer(workspace: Path) -> dict[str, Any] | None:
    latest = workspace / "03_CHECKPOINTS" / "organism_latest.json"
    if not latest.exists():
        return None
    try:
        # utf-8-sig tolera BOM (PowerShell Export-Clixml/Out-File costuma gravar com BOM)
        return json.loads(latest.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # pragma: no cover - diagnostico
        return {"_error": f"falha lendo organism_latest.json: {exc}"}


# ---------------------------------------------------------------------------
# Config / STATUS parsing
# ---------------------------------------------------------------------------

def load_canonical_config() -> dict[str, Any] | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        import yaml  # type: ignore

        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return {"_error": f"falha lendo config yaml: {exc}"}


def parse_status_claims() -> dict[str, Any]:
    """Extrai afirmacoes numericas do STATUS_ATUAL.md para comparacao."""
    claims: dict[str, Any] = {"_path": str(STATUS_PATH), "parsed": False}
    if not STATUS_PATH.exists():
        claims["_error"] = "STATUS_ATUAL.md ausente"
        return claims
    text = STATUS_PATH.read_text(encoding="utf-8")
    claims["parsed"] = True
    # captura o snapshot date
    m = re.search(r"Snapshot local verificado em \*\*([0-9A-Za-z\-: ]+)\*\*", text)
    if m:
        claims["snapshot_date"] = m.group(1).strip()
    # "1.764.019.648 parametros"  ou  "1,764,019,648"
    m = re.search(r"([\d\.,]+)\s+parametros", text, re.IGNORECASE)
    if m:
        num = m.group(1).replace(".", "").replace(",", "")
        try:
            claims["status_params_claim"] = int(num)
        except ValueError:
            claims["status_params_claim_raw"] = m.group(1)
    # modelo real
    m = re.search(r"Modelo real\s*\|([^\n|]+)\|", text)
    if m:
        claims["status_model_real"] = m.group(1).strip()
    # ultimo checkpoint
    m = re.search(r"organism_cycle_(\d+)\.pt[^0-9]*v(\d+)[^0-9]*cycle\s*(\d+)[^0-9]*step\s*(\d+)",
                  text, re.IGNORECASE)
    if m:
        claims["status_last_checkpoint"] = {
            "cycle": int(m.group(1)),
            "version": int(m.group(2)),
            "cycle_field": int(m.group(3)),
            "step": int(m.group(4)),
            "path": f"organism_cycle_{m.group(1)}.pt",
        }
    return claims


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def human_params(n: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.3f}{unit}"
    return str(n)


def summarize_model_state(model_state: dict[str, Any]) -> dict[str, Any]:
    total = 0
    by_prefix: dict[str, int] = {}
    sample_keys: list[str] = []
    for i, (key, tensor) in enumerate(model_state.items()):
        n = getattr(tensor, "numel", lambda: 0)()
        total += n
        prefix = key.split(".")[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + n
        if i < 12:
            sample_keys.append(f"{key}: {tuple(tensor.shape)}")
    return {
        "total_params": total,
        "total_params_human": human_params(total),
        "n_tensors": len(model_state),
        "by_prefix_top": dict(sorted(by_prefix.items(), key=lambda kv: -kv[1])[:10]),
        "sample_keys": sample_keys,
    }


def _resolve_model_state(state: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("model_state_dict", "model", "state_dict", "model_state"):
        v = state.get(key)
        if isinstance(v, dict):
            return v
    return None


def _resolve_optimizer_state(state: dict[str, Any]) -> Any:
    for key in ("optimizer_state_dict", "optimizer", "optim"):
        if key in state:
            return state[key]
    return None


def _resolve_training_state(state: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("training_state", "train_state"):
        v = state.get(key)
        if isinstance(v, dict):
            return v
    return None


def detect_init_kind(state: dict[str, Any], training_state: dict[str, Any] | None) -> dict[str, Any]:
    """Heuristica: init aleatorio puro vs treinado.

    Sinais de treinado: presenca de optimizer_state_dict, step > 0, cycle > 0,
    train_loss/eval_loss presentes, e identificadores de checkpoint.
    """
    opt = _resolve_optimizer_state(state)
    has_optimizer = bool(opt) and (
        bool(opt.get("state", {})) if isinstance(opt, dict) else bool(opt)
    )
    ts = training_state or {}
    # step pode estar em training_state ou no topo
    step_keys = ("step", "iterations", "updates", "global_step", "total_steps")
    step_val = None
    for src in (ts, state):
        for k in step_keys:
            if k in src:
                try:
                    step_val = int(src[k])
                except Exception:
                    step_val = src[k]
                break
        if step_val is not None:
            break
    cycle_val = None
    for src in (ts, state):
        for k in ("cycle", "current_cycle"):
            if k in src:
                try:
                    cycle_val = int(src[k])
                except Exception:
                    cycle_val = src[k]
                break
        if cycle_val is not None:
            break
    loss_keys = ("train_loss", "eval_loss", "replay_loss", "loss", "last_loss")
    has_loss = any(k in ts for k in loss_keys) or any(k in state for k in loss_keys)
    ckpt_version = state.get("checkpoint_version") or state.get("version")
    base_id = state.get("base_checkpoint_id")

    signals = {
        "has_optimizer": has_optimizer,
        "step": step_val,
        "cycle": cycle_val,
        "has_loss_metrics": has_loss,
        "checkpoint_version": ckpt_version,
        "base_checkpoint_id": base_id,
    }
    trained_score = 0
    if has_optimizer:
        trained_score += 1
    if isinstance(step_val, int) and step_val > 0:
        trained_score += 1
    if isinstance(cycle_val, int) and cycle_val > 0:
        trained_score += 1
    if has_loss:
        trained_score += 1
    if base_id:
        trained_score += 1
    init_kind = "trained" if trained_score >= 3 else ("random_init" if trained_score == 0 else "ambiguous")
    signals["trained_score"] = trained_score
    signals["init_kind"] = init_kind
    return signals


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    t0 = time.time()
    size_gb = path.stat().st_size / 1e9
    state = torch.load(path, map_location="cpu", weights_only=False)
    load_secs = time.time() - t0
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_gb": round(size_gb, 3),
        "load_secs": round(load_secs, 2),
        "top_keys": sorted(state.keys()),
    }
    model_state = _resolve_model_state(state) if isinstance(state, dict) else None
    training_state = _resolve_training_state(state) if isinstance(state, dict) else None
    if training_state is not None:
        info["training_state_keys"] = sorted(training_state.keys())
    if isinstance(model_state, dict):
        info["model"] = summarize_model_state(model_state)
        info["init"] = detect_init_kind(state, training_state)
    else:
        info["model"] = None
        info["init"] = detect_init_kind(state, training_state)
    # metricas uteis (topo + training_state)
    metric_sources: list[tuple[dict[str, Any], str]] = [(state, "top")]
    if training_state is not None:
        metric_sources.append((training_state, "training_state"))
    for k in ("step", "cycle", "epoch", "train_loss", "eval_loss", "learning_rate",
              "gradient_norm", "checkpoint_version", "version", "base_checkpoint_id",
              "timestamp", "created_at", "saved_at", "total_steps"):
        for src, origin in metric_sources:
            if k in src:
                v = src[k]
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        v = str(v)
                info.setdefault("metrics", {})[k] = v
                break
    return info


# ---------------------------------------------------------------------------
# Expected params from config (estimativa barata sem instanciar o modelo)
# ---------------------------------------------------------------------------

def estimate_params_from_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Tenta usar estimate_darwin_x_parameters do proprio modelo."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from f51_darwin.darwin_x import DarwinXConfig, estimate_darwin_x_parameters  # type: ignore

        dc = DarwinXConfig.from_mapping(dict(cfg))
        est = estimate_darwin_x_parameters(dc)
        # ATENCAO: o dict retornado ja contem a chave "total" (soma dos componentes).
        # Somar todos os valores causaria double-count. Use "total" se presente.
        total = est.get("total") if "total" in est else sum(
            v for k, v in est.items() if isinstance(v, int)
        )
        return {"total_params_estimate": total,
                "total_params_estimate_human": human_params(total),
                "breakdown": est,
                "config_model_name": cfg.get("model_name"),
                "config_ssm_state": cfg.get("ssm_state"),
                "config_n_layers": cfg.get("n_layers"),
                "config_d_model": cfg.get("d_model")}
    except Exception as exc:
        return {"_error": f"estimativa indisponivel: {exc}",
                "config_model_name": cfg.get("model_name") if cfg else None,
                "config_ssm_state": cfg.get("ssm_state") if cfg else None}


# ---------------------------------------------------------------------------
# Cross-check / discrepancies
# ---------------------------------------------------------------------------

def cross_check(ckpt_info: dict[str, Any], cfg_est: dict[str, Any] | None,
                latest_ptr: dict[str, Any] | None,
                all_ckpts: list[Path]) -> list[str]:
    disc: list[str] = []
    model = ckpt_info.get("model") or {}
    actual = model.get("total_params")

    # 1. Param count vs STATUS_ATUAL claim
    status_params = None
    if STATUS_PATH.exists():
        m = re.search(r"([\d\.,]+)\s+parametros", STATUS_PATH.read_text(encoding="utf-8"), re.IGNORECASE)
        if m:
            try:
                status_params = int(m.group(1).replace(".", "").replace(",", ""))
            except ValueError:
                pass
    if actual and status_params and abs(actual - status_params) > max(1000, status_params * 0.001):
        disc.append(
            f"params reais ({actual:,}) != STATUS_ATUAL.md ({status_params:,}). "
            f"Delta = {actual - status_params:+,}"
        )

    # 2. Param count vs config estimate
    if actual and cfg_est and (est := cfg_est.get("total_params_estimate")):
        if abs(actual - est) > max(1000, est * 0.02):
            disc.append(
                f"params reais ({actual:,}) divergem da estimativa da config ({est:,}). "
                f"Delta = {actual - est:+,}"
            )

    # 3. init aleatorio vs treinado
    init_kind = (ckpt_info.get("init") or {}).get("init_kind")
    if init_kind in ("random_init", "ambiguous"):
        disc.append(f"checkpoint parece nao-treinado ou ambiguo (init_kind={init_kind}). "
                    f"Verifique step/cycle/optimizer.")

    # 4. latest pointer vs newest on disk
    if latest_ptr and all_ckpts:
        ptr_path = latest_ptr.get("path")
        newest = all_ckpts[-1].name
        if ptr_path and ptr_path != newest:
            disc.append(
                f"organism_latest.json aponta para '{ptr_path}' mas o .pt mais recente "
                f"no disco e '{newest}'. Resume pode pegar linha errada."
            )
        ptr_cycle = latest_ptr.get("cycle")
        ptr_step = latest_ptr.get("step")
        ckpt_cycle = (ckpt_info.get("metrics") or {}).get("cycle")
        ckpt_step = (ckpt_info.get("metrics") or {}).get("step")
        if ckpt_cycle is not None and ptr_cycle is not None and ckpt_cycle != ptr_cycle:
            disc.append(f"cycle do checkpoint carregado ({ckpt_cycle}) != cycle do latest ({ptr_cycle}).")
        if ckpt_step is not None and ptr_step is not None and ckpt_step != ptr_step:
            disc.append(f"step do checkpoint carregado ({ckpt_step}) != step do latest ({ptr_step}).")

    # 5. classe 2.5B vs 1.6B (contexto)
    if actual:
        if 2_400_000_000 <= actual <= 2_600_000_000:
            disc.append("Checkpoint na classe ~2.5B. AGENTS.md regra 6 reserva 2.5B para nuvem "
                        "- nao e alvo local sem decisao nova.")
        elif actual < 1_500_000_000 or (1_900_000_000 < actual < 2_400_000_000):
            disc.append(f"Checkpoint fora das classes oficiais (600M / 1.6B-Nitro): {actual:,} params.")
    return disc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnostico de checkpoint F51 Darwin-X")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Caminho explicito para .pt (default: apontado por organism_latest.json)")
    ap.add_argument("--no-load", action="store_true",
                    help="Nao carregar o .pt (apenas metadados do disco + latest.json)")
    ap.add_argument("--json-only", action="store_true",
                    help="Suprime output humano, so escreve JSON")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    findings: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(REPO_ROOT),
    }

    try:
        workspace = resolve_workspace()
    except FileNotFoundError as exc:
        findings["error"] = str(exc)
        OUT_PATH.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    findings["workspace"] = str(workspace)
    all_ckpts = find_checkpoints(workspace)
    findings["checkpoints_on_disk"] = [
        {"name": p.name, "size_gb": round(p.stat().st_size / 1e9, 3),
         "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))}
        for p in all_ckpts
    ]
    latest_ptr = read_latest_pointer(workspace)
    findings["organism_latest_pointer"] = latest_ptr

    # Escolhe checkpoint a inspecionar
    target: Path | None = args.checkpoint
    if target is None:
        ptr_name = (latest_ptr or {}).get("path")
        if ptr_name:
            cand = workspace / "03_CHECKPOINTS" / ptr_name
            if cand.exists():
                target = cand
        if target is None and all_ckpts:
            target = all_ckpts[-1]  # mais recente no disco

    cfg = load_canonical_config()
    cfg_est = estimate_params_from_config(cfg) if cfg else None
    findings["canonical_config"] = {
        "path": str(CONFIG_PATH),
        "model_name": (cfg or {}).get("model_name"),
        "ssm_state": (cfg or {}).get("ssm_state"),
        "n_layers": (cfg or {}).get("n_layers"),
        "d_model": (cfg or {}).get("d_model"),
    } if cfg else {"_error": "config ausente"}
    findings["config_param_estimate"] = cfg_est
    findings["status_claims"] = parse_status_claims()

    if target is None:
        findings["error"] = "Nenhum checkpoint encontrado para inspecionar."
        OUT_PATH.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[FAIL] Nenhum checkpoint encontrado.", file=sys.stderr)
        return 3

    if args.no_load:
        findings["checkpoint_inspected"] = {
            "path": str(target),
            "size_gb": round(target.stat().st_size / 1e9, 3),
            "loaded": False,
        }
    else:
        try:
            ckpt_info = load_checkpoint(target)
            findings["checkpoint_inspected"] = ckpt_info
            findings["discrepancies"] = cross_check(ckpt_info, cfg_est, latest_ptr, all_ckpts)
        except Exception as exc:
            findings["checkpoint_inspected"] = {"path": str(target), "load_error": repr(exc)}
            findings["discrepancies"] = [f"falha carregando checkpoint: {exc!r}"]

    OUT_PATH.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.json_only:
        print("=" * 70)
        print("F51 DARWIN-X - DIAGNOSTICO DE CHECKPOINT")
        print("=" * 70)
        if cfg:
            print(f"Config canonica......: {cfg.get('model_name')} "
                  f"(d_model={cfg.get('d_model')}, n_layers={cfg.get('n_layers')}, "
                  f"ssm_state={cfg.get('ssm_state')})")
        if target:
            print(f"Checkpoint inspecionado: {target.name}")
        model = (findings.get("checkpoint_inspected") or {}).get("model") or {}
        if model:
            print(f"Parametros reais.....: {model.get('total_params'):,} "
                  f"({model.get('total_params_human')})")
        init = (findings.get("checkpoint_inspected") or {}).get("init") or {}
        print(f"init_kind............: {init.get('init_kind')}  "
              f"(score={init.get('trained_score')})")
        metrics = (findings.get("checkpoint_inspected") or {}).get("metrics") or {}
        if metrics:
            print(f"step/cycle/version...: step={metrics.get('step')} "
                  f"cycle={metrics.get('cycle')} v={metrics.get('checkpoint_version') or metrics.get('version')}")
        if latest_ptr:
            print(f"latest.json aponta...: cycle {latest_ptr.get('cycle')} step {latest_ptr.get('step')} "
                  f"({latest_ptr.get('path')})")
        print(f"Status params claim..: {(findings.get('status_claims') or {}).get('status_params_claim')}")
        disc = findings.get("discrepancies") or []
        print("-" * 70)
        if disc:
            print(f"DISCREPANCIAS ({len(disc)}):")
            for d in disc:
                print(f"  [!] {d}")
        else:
            print("DISCREPANCIAS: nenhuma detectada.")
        print("-" * 70)
        print(f"Findings salvos em: {OUT_PATH}")

    return 0 if not findings.get("discrepancies") else 1


if __name__ == "__main__":
    raise SystemExit(main())
