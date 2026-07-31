#!/usr/bin/env python3
"""
Gradient Attribution Protocol — F51 Darwin-X
=============================================
Atribui norma L2 de gradiente por termo de loss no backbone (parametros com
'blocks' no nome) e implementa formula de calibracao de pesos.

Checkpoint: workspace/03_CHECKPOINTS_100M_FULL_V9/organism_cycle_071_step_023000.pt
Config:     src/configs/darwin_x_100m.yaml

Tarefas:
  T1 — Script de atribuicao de gradiente
  T2 — Validacao contra checkpoint V9
  T3 — Formula de calibracao
  T4 — Piso de ruido (noise floor)

Autoridade:
  src/f51_darwin/darwin_x_core/model.py:631-733  (forward pass e loss terms)
  src/f51_darwin/darwin_x_core/losses.py:105-157 (compose_loss / LossPolicy)
  src/f51_darwin/darwin_x_core/config.py:233-256 (DarwinXOutput)
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.darwin_x import DarwinXConfig, DarwinXModel
from f51_darwin.organism.checkpoint import load_model_from_checkpoint


# ── Constantes ─────────────────────────────────────────────────────
CKPT_PATH = (
    ROOT / "workspace" / "03_CHECKPOINTS_100M_FULL_V9"
    / "organism_cycle_071_step_023000.pt"
)
BATCH_SIZE = 1
SEQ_LEN = 256
SEED = 42
STEP_DIGEST = "gradient-attribution-v1-c7a3f"
EPSILON = 1e-8
N_NOISE_RUNS = 5
PERTURBATION_SCALE = 1e-5


# ── Helpers ────────────────────────────────────────────────────────

def _backbone_param_names(model: DarwinXModel) -> list[str]:
    """src/f51_darwin/darwin_x_core/model.py: percorre self.blocks (DarwinXBlock)."""
    return [name for name, _ in model.named_parameters() if "blocks" in name]


def _l2_norm_of_grads(model: DarwinXModel, param_names: set[str]) -> float:
    total_sq = 0.0
    for name, param in model.named_parameters():
        if name in param_names and param.grad is not None:
            total_sq += param.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total_sq)


def _zero_all_grads(model: DarwinXModel) -> None:
    for param in model.parameters():
        param.grad = None


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _sep(title: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")


# ── Carregamento ───────────────────────────────────────────────────

def load_model_and_validate() -> tuple[DarwinXModel, DarwinXConfig, dict]:
    """Carrega modelo do checkpoint V9, congela heartbeat, valida identidade."""
    _sep("CARREGAMENTO DO MODELO")

    print(f"Checkpoint : {CKPT_PATH}")
    print(f"Existe     : {CKPT_PATH.exists()}")

    # CPU porque o codigo em src/f51_darwin/darwin_x_core/layers.py:62-66
    # força FlashAttention via torch.backends.cuda.sdp_kernel(enable_math=False)
    # e float32 nao tem kernel FlashAttention disponivel neste ambiente.
    device = "cpu"
    print(f"Device     : {device}")

    model, config, metrics = load_model_from_checkpoint(str(CKPT_PATH), map_location=device)
    # Usar train() para que o modelo se comporte como durante treino
    # (dropout=0.0 na config, entao nao ha ruido de dropout)
    model.train()
    # Congela heartbeat: src/f51_darwin/darwin_x_core/model.py:741-766
    model.heartbeat = None

    total_p = sum(p.numel() for p in model.parameters())
    bb_p = sum(p.numel() for n, p in model.named_parameters() if "blocks" in n)
    print(f"Total params        : {total_p:>15,}")
    print(f"Backbone ('blocks') : {bb_p:>15,}  ({bb_p / total_p * 100:.1f}%)")
    print(f"d_model={config.d_model}  n_layers={config.n_layers}  "
          f"context_length={config.context_length}")
    print(f"loss_semantics_version  : {config.loss_semantics_version}")
    print(f"gradient_checkpointing  : {config.gradient_checkpointing}")
    print(f"heartbeat_enabled       : {config.heartbeat_enabled}  (congelado: model.heartbeat=None)")
    print(f"ghost_enabled           : {config.ghost_enabled}")
    print(f"spider_calibration_enabled: {config.spider_calibration_enabled}")
    print(f"gaba_enabled            : {getattr(config, 'gaba_enabled', '?')}")
    print(f"mtp_depth               : {config.mtp_depth}")
    print(f"fine_experts            : {config.fine_experts}")
    print(f"experts_per_token       : {config.experts_per_token}")

    return model, config, metrics


# ── TAREFA 1 + 2: Atribuicao de Gradiente ─────────────────────────

def run_gradient_attribution(
    model: DarwinXModel,
    config: DarwinXConfig,
) -> dict:
    """
    T1/T2: Forward com batch fixo, backward por termo de loss RAW,
    norma L2 concatenada nos parametros 'blocks'.
    """
    _sep("TAREFA 1+2 — ATRIBUICAO DE GRADIENTE POR TERMO DE LOSS")

    device = next(model.parameters()).device
    _set_seeds(SEED)
    batch = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
    print(f"Batch: [{BATCH_SIZE}, {SEQ_LEN}] tokens, seed={SEED}, "
          f"vocab=[{batch.min().item()}, {batch.max().item()}]")

    # ── Forward ──
    output = model(batch, labels=batch, domain="test", step_digest=STEP_DIGEST)

    # Coletar tensores RAW disponiveis no DarwinXOutput
    raw_terms: dict[str, torch.Tensor] = {}
    for attr in ("lm_loss", "mtp_loss", "jepa_loss", "aux_loss",
                 "raw_ghost_loss", "spider_loss"):
        t = getattr(output, attr, None)
        if isinstance(t, torch.Tensor):
            raw_terms[attr] = t
    for attr in ("effective_jepa_loss", "effective_spider_loss"):
        t = getattr(output, attr, None)
        if isinstance(t, torch.Tensor):
            raw_terms[attr] = t

    print(f"\nTermos RAW disponiveis: {list(raw_terms.keys())}")
    print(f"\n{'─' * 70}")
    print(f"  {'Termo':<28s} {'Valor RAW':>15s}  {'Valor EFFECTIVE':>15s}")
    print(f"{'─' * 70}")
    for name, tensor in raw_terms.items():
        marker = " <-- weighted" if name.startswith("effective_") else ""
        print(f"  {name:<28s} {tensor.item():>15.6f}{marker}")
    print(f"{'─' * 70}")

    # Pesos atuais da config
    weights = {
        "lm": 1.0,
        "mtp": config.mtp_weight,
        "jepa": config.jepa_weight,
        "aux": float(getattr(config, "aux_loss_scale", 1.0)),
        "ghost": config.ghost_weight,
        "spider": float(getattr(config, "spider_calibration_weight", 0.0)),
    }
    print(f"\nPesos atuais (w_atual) da config:")
    for k, v in weights.items():
        print(f"  w_{k:<10s} = {v:.4f}")

    # Mapeamento: nome do termo raw -> (tensor, chave do peso)
    attribution_map = [
        ("lm_loss",        output.lm_loss,        weights["lm"]),
        ("mtp_loss",       output.mtp_loss,       weights["mtp"]),
        ("jepa_loss",      output.jepa_loss,      weights["jepa"]),
        ("aux_loss",       output.aux_loss,       weights["aux"]),
        ("raw_ghost_loss", output.raw_ghost_loss, weights["ghost"]),
        ("spider_loss",    output.spider_loss,    weights["spider"]),
    ]
    attribution_map = [(n, t, w) for n, t, w in attribution_map if t is not None]

    bb_names = set(_backbone_param_names(model))
    print(f"\nBackbone params: {len(bb_names)} keys (com 'blocks' no nome)")

    # ── Computar gradientes ──
    results = {}
    grad_norms = {}

    for term_name, loss_tensor, weight in attribution_map:
        _zero_all_grads(model)
        try:
            loss_tensor.backward(retain_graph=True)
        except Exception as exc:
            print(f"  ERRO backward({term_name}): {exc}")
            grad_norms[term_name] = 0.0
            continue

        norm = _l2_norm_of_grads(model, bb_names)
        grad_norms[term_name] = norm

        value = loss_tensor.item()
        eff_norm = norm * weight
        lm_norm = grad_norms.get("lm_loss", 1.0)
        frac_vs_lm = norm / lm_norm if lm_norm > 1e-16 else float("nan")

        # Contar quantos parametros backbone receberam gradiente nao-zero
        nonzero_grad_params = sum(
            1 for n, p in model.named_parameters()
            if n in bb_names and p.grad is not None and p.grad.abs().max().item() > 0
        )

        results[term_name] = {
            "loss_value": value,
            "grad_norm_raw": norm,
            "weight": weight,
            "grad_norm_effective": eff_norm,
            "fraction_vs_lm": frac_vs_lm,
            "nonzero_params": nonzero_grad_params,
        }

    # ── TABELA ──
    lm_norm_raw = grad_norms.get("lm_loss", 1.0)

    def _fmt_grad(v: float) -> str:
        """Formata norma de gradiente com notacao cientifica se < 0.01."""
        if abs(v) < 1e-12:
            return "      0.0"
        if abs(v) < 0.01:
            return f"{v:>10.4e}"
        return f"{v:>10.4f}"

    print(f"\n{'=' * 130}")
    print(f"  TABELA DE ATRIBUICAO DE GRADIENTE — Checkpoint V9 "
          f"(cycle 71, step 23000, {len(bb_names)} backbone params)")
    print(f"  ||g_lm|| (ref) = {_fmt_grad(lm_norm_raw)}")
    print(f"{'=' * 130}")
    header = (
        f"  {'Termo':<22s} {'Loss RAW':>10s} {'Peso w':>8s} "
        f"{'||g_raw||':>12s} {'||g_eff||':>12s} {'Fracao/LM':>12s} "
        f"{'Params c/ g':>12s} {'Dominancia':>12s}"
    )
    print(header)
    print(f"  {'─' * 128}")

    display_order = ["lm_loss", "mtp_loss", "jepa_loss", "aux_loss",
                     "raw_ghost_loss", "spider_loss"]
    for term_name in display_order:
        if term_name not in results:
            continue
        r = results[term_name]
        domination = "REFERENCIA" if term_name == "lm_loss" else ""
        if r["fraction_vs_lm"] > 100:
            domination = f"DOMINA {r['fraction_vs_lm']:.0f}x"
        elif r["fraction_vs_lm"] > 10:
            domination = f"FORTE {r['fraction_vs_lm']:.0f}x"
        elif r["fraction_vs_lm"] > 1:
            domination = f"+{r['fraction_vs_lm']:.1f}x"
        elif r["fraction_vs_lm"] > 0.01:
            domination = f"fraco {r['fraction_vs_lm']:.3f}x"
        else:
            domination = f"minimo {r['fraction_vs_lm']:.1e}x"

        print(
            f"  {term_name:<22s} {r['loss_value']:>10.4f} {r['weight']:>8.4f} "
            f"{_fmt_grad(r['grad_norm_raw']):>12s} {_fmt_grad(r['grad_norm_effective']):>12s} "
            f"{r['fraction_vs_lm']:>12.2e} {r['nonzero_params']:>10d}  {domination:>14s}"
        )

    print(f"  {'─' * 128}")
    total_eff = sum(r["grad_norm_effective"] for r in results.values())
    print(f"  {'SOMA ||g_eff||':>56s} {_fmt_grad(total_eff):>12s}")
    print(f"{'=' * 130}")

    return {
        "results": results,
        "grad_norms": grad_norms,
        "lm_norm_raw": lm_norm_raw,
        "backbone_names": bb_names,
        "weights": weights,
    }


# ── TAREFA 3: Formula de Calibracao ───────────────────────────────

def calibrate_weights(attribution_data: dict) -> dict:
    """
    T3: Formula de calibracao de pesos por termo de loss.

        w_novo = w_atual * (alvo * ||g_lm||) / (w_atual * ||g_termo|| + epsilon)

    Regra de seguranca: clamp(w_novo / w_atual, 1/3, 3).
    """
    _sep("TAREFA 3 — FORMULA DE CALIBRACAO DE PESOS")

    grad_norms = attribution_data["grad_norms"]
    weights = attribution_data["weights"]
    lm_norm_raw = grad_norms.get("lm_loss", 1.0)

    # Alvos: fracao desejada de contribuicao de gradiente relativa a ||g_lm||
    targets = {
        "mtp": 0.10,    # 10% da forca do LM
        "jepa": 0.20,   # 20% — sinal de representacao latente
        "aux": 0.05,    #  5% — load balancing nao deve dominar
        "ghost": 0.05,  #  5% — auxiliar
        "spider": 0.05, #  5% — calibracao de confianca
    }

    term_to_key = {
        "mtp_loss": "mtp",
        "jepa_loss": "jepa",
        "aux_loss": "aux",
        "raw_ghost_loss": "ghost",
        "spider_loss": "spider",
    }

    def clip_ratio(ratio: float) -> float:
        return max(1.0 / 3.0, min(3.0, ratio))

    print(f"\n||g_lm|| (referencia) = {lm_norm_raw:.6e}")
    print(f"epsilon = {EPSILON}")
    print()
    print(f"  {'Termo':<22s} {'w_atual':>8s} {'||g_raw||':>14s} "
          f"{'Alvo':>8s} {'w_novo_ideal':>14s} {'Ratio':>10s} "
          f"{'Clamp':>10s} {'w_final':>10s}")
    print(f"  {'─' * 106}")

    calibrated = {}
    clamped_any = False
    for term_name, key in term_to_key.items():
        w_atual = weights.get(key, 1.0)
        g_term_norm = grad_norms.get(term_name, 1e-8)
        alvo = targets.get(key, 0.05)

        if w_atual <= 0.0:
            print(f"  {term_name:<22s} {'---':>8s} {'---':>14s} "
                  f"{'---':>8s} {'---':>14s} {'---':>10s} {'---':>10s} "
                  f"{'---':>10s}  (w=0, pulando)")
            calibrated[key] = 0.0
            continue

        denom = w_atual * g_term_norm + EPSILON
        w_novo_ideal = w_atual * (alvo * lm_norm_raw) / denom

        ratio = w_novo_ideal / w_atual
        ratio_clamped = clip_ratio(ratio)
        w_final = w_atual * ratio_clamped

        calibrated[key] = w_final

        clamped = "SIM" if abs(ratio - ratio_clamped) > 1e-6 else ""
        if clamped:
            clamped_any = True

        print(
            f"  {term_name:<22s} {w_atual:>8.4f} {g_term_norm:>14.6e} "
            f"{alvo:>8.2f} {w_novo_ideal:>14.6e} {ratio:>10.2f} "
            f"{clamped:>10s} {w_final:>10.4f}"
        )

    print(f"  {'─' * 106}")

    if clamped_any:
        print(f"\n  [AVISO] Regra de seguranca clamp(ratio, 1/3, 3) ativada.")
        print(f"  Mudancas > 3x foram limitadas. Apos 1 ciclo, re-medir e re-aplicar.")

    print(f"\n  RESUMO DE PESOS CALIBRADOS:")
    print(f"  {'Termo':<15s} {'Antes':>10s} {'Depois':>10s} {'Mudanca':>10s} {'Notas':>30s}")
    print(f"  {'─' * 78}")
    for key in ["mtp", "jepa", "aux", "ghost", "spider"]:
        antes = weights.get(key, 1.0)
        depois = calibrated.get(key, antes)
        delta_pct = (depois / antes - 1.0) * 100 if antes > 0 else 0

        g_term_norm_key = [k for k, v in term_to_key.items() if v == key][0]
        g_term = grad_norms.get(g_term_norm_key, 0)
        eff_before = antes * g_term
        eff_after = depois * g_term
        target_eff = targets.get(key, 0.05) * lm_norm_raw
        notes = f"eff: {eff_before:.2e} -> {eff_after:.2e} (target {target_eff:.2e})"

        print(f"  {key:<15s} {antes:>10.4f} {depois:>10.4f} {delta_pct:>+9.1f}%  {notes}")

    return calibrated


# ── TAREFA 4: Piso de Ruido ───────────────────────────────────────

def measure_noise_floor(
    model: DarwinXModel,
    config: DarwinXConfig,
    n_runs: int = N_NOISE_RUNS,
) -> dict:
    """
    T4: Mede o piso de ruido do forward com mesmo input/modelo/seed.

    Executa N forwards consecutivos, reseta seed entre cada um, e mede
    max_delta na loss total e lm_loss.

    Estima o piso equivalente em norma de gradiente via:
      1. Medicao direta de variabilidade no forward
      2. Teste de perturbacao parametrica (Delta_w -> Delta_L)
      3. Inferencia: ||g_min|| = epsilon_L / delta_w_characteristic
    """
    _sep("TAREFA 4 — PISO DE RUIDO (NOISE FLOOR)")

    device = next(model.parameters()).device

    # ── Fase 1: Ruido forward-a-forward ──
    print(f"\n  --- Fase 1: Variabilidade forward-a-forward ({n_runs} runs) ---")
    loss_vals = []
    lm_vals = []
    all_loss_terms = []

    for run_idx in range(n_runs):
        _set_seeds(SEED)
        batch = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)

        with torch.no_grad():
            output = model(batch, labels=batch, domain="test", step_digest=STEP_DIGEST)

        lv = output.loss.item() if output.loss is not None else float("nan")
        lm = output.lm_loss.item() if output.lm_loss is not None else float("nan")
        loss_vals.append(lv)
        lm_vals.append(lm)

        terms = {
            "lm": lm,
            "mtp": output.mtp_loss.item() if output.mtp_loss is not None else 0.0,
            "jepa": output.jepa_loss.item() if output.jepa_loss is not None else 0.0,
            "aux": output.aux_loss.item() if output.aux_loss is not None else 0.0,
            "ghost": output.raw_ghost_loss.item() if output.raw_ghost_loss is not None else 0.0,
            "spider": output.spider_loss.item() if output.spider_loss is not None else 0.0,
        }
        all_loss_terms.append(terms)
        print(f"  Run {run_idx + 1}/{n_runs}: loss={lv:.6f}, lm={lm:.6f}")

    max_delta_loss = max(loss_vals) - min(loss_vals) if loss_vals else 0.0
    max_delta_lm = max(lm_vals) - min(lm_vals) if lm_vals else 0.0

    # Max delta por termo entre runs
    print(f"\n  --- Variabilidade entre runs ---")
    print(f"  Loss values : {[f'{v:.6f}' for v in loss_vals]}")
    print(f"  LM values   : {[f'{v:.6f}' for v in lm_vals]}")
    for key in ["mtp", "jepa", "aux", "ghost", "spider"]:
        vals = [t[key] for t in all_loss_terms]
        maxd = max(vals) - min(vals) if vals else 0.0
        print(f"  {key:>8s}     : {[f'{v:.6f}' for v in vals]}  max_delta={maxd:.6f}")

    mean_loss = sum(loss_vals) / len(loss_vals) if loss_vals else 0.0
    std_loss = float(np.std(loss_vals)) if loss_vals else 0.0
    print(f"\n  max_delta(loss)  = {max_delta_loss:.6e}")
    print(f"  max_delta(lm)    = {max_delta_lm:.6e}")
    print(f"  mean(loss)       = {mean_loss:.6f}")
    print(f"  std(loss)        = {std_loss:.6e}")

    epsilon_loss = max(max_delta_loss, max_delta_lm)

    # ── Fase 2: Determinismo com seed fixo ──
    print(f"\n  --- Fase 2: Mesmo batch, seed fixo (determinismo) ---")
    _set_seeds(SEED)
    batch_fixed = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)

    fixed_vals = []
    for run_idx in range(n_runs):
        _set_seeds(SEED)
        with torch.no_grad():
            output = model(batch_fixed, labels=batch_fixed, domain="test", step_digest=STEP_DIGEST)
        fixed_vals.append(output.loss.item() if output.loss is not None else float("nan"))
        print(f"  Run {run_idx + 1}/{n_runs}: loss={fixed_vals[-1]:.10f}")

    max_delta_fixed = max(fixed_vals) - min(fixed_vals) if fixed_vals else 0.0
    print(f"  max_delta (batch fixo, seed fixo): {max_delta_fixed:.6e}")

    # ── Fase 3: Perturbacao parametrica ──
    print(f"\n  --- Fase 3: Teste de perturbacao parametrica (Delta_w = {PERTURBATION_SCALE}) ---")
    _set_seeds(SEED)
    batch = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)

    # Baseline
    _set_seeds(SEED)
    with torch.no_grad():
        base_out = model(batch, labels=batch, domain="test", step_digest=STEP_DIGEST)
    loss_base = base_out.loss.item()

    # Perturbar: adicionar ruido gaussiano * perturbation_scale a todos os params backbone
    bb_names_set = set(_backbone_param_names(model))
    saved = {}
    for name, param in model.named_parameters():
        if name in bb_names_set:
            saved[name] = param.data.clone()
            perturb = torch.randn_like(param.data) * PERTURBATION_SCALE
            param.data.add_(perturb)

    # Forward perturbado
    _set_seeds(SEED)
    with torch.no_grad():
        pert_out = model(batch, labels=batch, domain="test", step_digest=STEP_DIGEST)
    loss_pert = pert_out.loss.item()

    # Restaurar
    for name, param in model.named_parameters():
        if name in saved:
            param.data.copy_(saved[name])

    delta_l_emp = abs(loss_pert - loss_base)
    sensitivity = delta_l_emp / PERTURBATION_SCALE  # |Delta L| / |Delta w|

    print(f"  Loss baseline      : {loss_base:.12f}")
    print(f"  Loss perturbado    : {loss_pert:.12f}")
    print(f"  |Delta L| emp      : {delta_l_emp:.6e}")
    print(f"  Sensibilidade (~||g_eff||): {sensitivity:.4f}")

    # ── Fase 4: Gradiente real para comparacao ──
    print(f"\n  --- Fase 4: ||g_lm|| real para SNR ---")
    _set_seeds(SEED)
    output = model(batch, labels=batch, domain="test", step_digest=STEP_DIGEST)
    _zero_all_grads(model)
    output.lm_loss.backward()
    g_lm_measured = _l2_norm_of_grads(model, bb_names_set)
    print(f"  ||g_lm|| medido (backbone): {g_lm_measured:.6e}")

    # ── Piso de gradiente ──
    # Se o ruido forward e epsilon_L e a perturbacao parametrica
    # Delta_w produz Delta_L, entao o gradiente minimo detectavel e:
    #   ||g_min|| = epsilon_L / (Delta_w_que_produz_epsilon_L)
    #            = epsilon_L / (epsilon_L / sensitivity)
    #            = sensitivity * (epsilon_L / delta_l_emp)
    #            ≈ sensitivity           (se epsilon_L ≈ delta_l_emp)
    #
    # Abordagem mais direta: o ruido em L se propaga como ruido em g
    # via a sensibilidade S = delta_L / delta_w.
    # grad_noise = epsilon_L / (delta_w_caracteristico)
    # Usamos perturbation_scale como delta_w_caracteristico:
    grad_noise_floor = epsilon_loss / PERTURBATION_SCALE if PERTURBATION_SCALE > 0 else 0.0

    # SNR: quantas vezes o gradiente de LM excede o piso
    snr = g_lm_measured / max(grad_noise_floor, 1e-16)

    print(f"\n  --- Piso de Gradiente ---")
    print(f"  epsilon_L (max delta loss entre runs) : {epsilon_loss:.6e}")
    print(f"  Perturbation scale (Delta_w)           : {PERTURBATION_SCALE:.1e}")
    print(f"  Sensibilidade (|Delta L| / |Delta w|)   : {sensitivity:.4f}")
    print(f"  Piso ||g|| (epsilon_L / Delta_w_scale) : {grad_noise_floor:.6e}")
    print(f"  ||g_lm|| medido                        : {g_lm_measured:.6e}")
    print(f"  SNR_g = ||g_lm|| / piso                : {snr:.2f}")

    # ── Conclusao ──
    print(f"\n  --- Conclusao ---")
    if epsilon_loss < 1e-10:
        print(f"  Modelo DETERMINISTICO com heartbeat=None + step_digest fixo.")
        print(f"  max_delta = {max_delta_fixed:.2e} (ruido de ponto flutuante).")
        print(f"  Nao ha piso de ruido — qualquer gradiente > 0 e detectavel.")
        print(f"  O ruido de 0.226 reportado e provavelmente do heartbeat ATIVO.")
    elif snr > 100:
        print(f"  [OK] SNR_g = {snr:.1f}x. Gradientes sao claramente detectaveis.")
    elif snr > 10:
        print(f"  [WARN] SNR_g = {snr:.1f}x. Margem moderada.")
    else:
        print(f"  [CRIT] SNR_g = {snr:.1f}x. Gradientes submersos em ruido!")

    return {
        "loss_values": loss_vals,
        "lm_values": lm_vals,
        "max_delta_loss": max_delta_loss,
        "max_delta_lm": max_delta_lm,
        "max_delta_fixed": max_delta_fixed,
        "epsilon_loss": epsilon_loss,
        "perturbation_scale": PERTURBATION_SCALE,
        "delta_loss_empirical": delta_l_emp,
        "sensitivity": sensitivity,
        "grad_noise_floor": grad_noise_floor,
        "g_lm_measured": g_lm_measured,
        "snr": snr,
    }


# ── MAIN ──────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 100)
    print("  F51 Darwin-X — GRADIENT ATTRIBUTION PROTOCOL")
    print(f"  Checkpoint : {CKPT_PATH}")
    print(f"  Batch      : {BATCH_SIZE}x{SEQ_LEN}, Seed={SEED}")
    print(f"  Timestamp  : {__import__('datetime').datetime.now().isoformat()}")
    print("=" * 100)

    model, config, metrics = load_model_and_validate()

    # T1 + T2
    attr_data = run_gradient_attribution(model, config)

    # T3
    calibrated = calibrate_weights(attr_data)

    # T4
    noise_data = measure_noise_floor(model, config, n_runs=N_NOISE_RUNS)

    # ── Resumo Final ──
    _sep("RESUMO FINAL")
    print(f"  Checkpoint      : organism_cycle_071_step_023000.pt (V9)")
    print(f"  Backbone params : {len(attr_data['backbone_names'])}")
    print(f"  ||g_lm|| (raw)  : {attr_data['lm_norm_raw']:.6e}")
    print(f"  Ruido max (L)   : {noise_data['max_delta_loss']:.6e}")
    print(f"  Piso ||g||      : {noise_data['grad_noise_floor']:.6e}")
    print(f"  SNR_g           : {noise_data['snr']:.2f}x")
    print()
    print(f"  Pesos calibrados (clamp 1/3x..3x ativado):")
    for key in ["mtp", "jepa", "aux", "ghost", "spider"]:
        antes = attr_data["weights"].get(key, 1.0)
        depois = calibrated.get(key, antes)
        delta = (depois / antes - 1.0) * 100 if antes > 0 else 0
        print(f"    w_{key:<10s} : {antes:.4f} -> {depois:.4f}  ({delta:+.1f}%)")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
