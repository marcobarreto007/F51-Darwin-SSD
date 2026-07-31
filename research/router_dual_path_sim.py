#!/usr/bin/env python3
"""
Router Dual-Path com Confianca Calibrada e Compute Floor
=========================================================
Simulacao executavel do mecanismo de roteamento dual-path inspirado nas
fontes primarias de 1940-1951. Implementa um router que escolhe entre um
caminho barato (memoria declarativa) e um caminho caro (JEPA / residual)
usando confianca calibrada e um piso minimo de compute.

ANCORAGEM TEORICA (governance/docs/pesquisa/FONTES_PRIMARIAS_1940_1951.md)

1. DOIS CAMINHOS COM CAPACIDADE DIFERENCIAL
   - Shannon (1948): BSTJ 27(3-4). Canais com taxas diferentes exigem
     codificacao adaptada.
   - von Neumann (1952): Automata Studies. Componentes nao-confiaveis
     podem ser organizados em sistema confiavel com redundancia.

2. ROUTER COM CONFIANCA (ORGAO RESTAURADOR)
   - von Neumann (1952): orgao restaurador (majority gate).
   - Wiener (1948): Cybernetics caps. III-IV. Feedback como decisao.

3. CALIBRACAO DE ESCALA
   - Shannon (1948): Secao 21. Entropia diferencial requer referencia de escala.
   - von Neumann (1952): threshold adaptativo proporcional a amplitude.

4. COMPUTE FLOOR
   - Ashby (1940, 1947, 1949): Lei da Variedade Requerida V(R) >= V(D)/V(S).
   - Wiener (1948): feedback negativo periodico necessario.

5. ALVO MOVEL
   - Wiener (1942/1949): predicao sob nao-estacionariedade.
   - Ashby (1947): adaptacao como processo continuo.

6. BIAS SUAVE NO MoE
   - Hebb (1949): reforco GRADUAL, nao binario.
   - Ashby (1949): mudanca de organizacao com RESTRICAO.

Uso:
    python research/router_dual_path_sim.py \
        --num-examples 2000 --compute-floor 0.05 \
        --memory-update-rate 100 --seed 42 \
        --output-dir workspace/runtime/router_sim \
        --router-mode correct

Modos: correct | random | all-expensive
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

EPSILON = 1e-8
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Router Dual-Path Simulation")
    p.add_argument("--num-examples", type=int, default=2000)
    p.add_argument("--compute-floor", type=float, default=0.05)
    p.add_argument("--memory-update-rate", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "workspace" / "runtime" / "router_sim"))
    p.add_argument("--router-mode", choices=["correct", "random", "all-expensive"], default="correct")
    p.add_argument("--cost-ratio", type=float, default=10.0)
    p.add_argument("--confidence-threshold", type=float, default=0.80)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--n-categories", type=int, default=8)
    p.add_argument("--input-dim", type=int, default=32)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def generate_example_stream(num_examples, n_classes, n_categories, input_dim, rng,
                            belief_freeze_category=None, freeze_start=0):
    examples = []
    true_prototypes = {}
    for cat in range(n_categories):
        proto = rng.randn(input_dim).astype(np.float32)
        proto /= np.linalg.norm(proto) + EPSILON
        true_prototypes[cat] = proto
    category_label_weights = {}
    for cat in range(n_categories):
        W = rng.randn(n_classes, input_dim).astype(np.float32) * 0.3
        b = rng.randn(n_classes).astype(np.float32) * 0.1
        category_label_weights[cat] = (W, b)
    for idx in range(num_examples):
        if rng.random() < 0.6 or n_categories <= 4:
            cat = rng.randint(0, max(1, n_categories // 2))
        else:
            cat = rng.randint(max(1, n_categories // 2), n_categories)
        noise_level = 0.15 + 0.35 * (cat / max(1, n_categories - 1))
        input_vec = true_prototypes[cat] + rng.randn(input_dim).astype(np.float32) * noise_level
        input_vec /= np.linalg.norm(input_vec) + EPSILON
        W, b = category_label_weights[cat]
        logits = input_vec @ W.T + b
        true_label = int(np.argmax(logits))
        difficulty = 0.1 + 0.2 * rng.random() if cat < n_categories // 2 else 0.5 + 0.5 * rng.random()
        is_freeze_trap = (belief_freeze_category is not None and cat == belief_freeze_category and idx >= freeze_start)
        examples.append({"index": idx, "category": cat, "input_vec": input_vec,
                         "true_label": true_label, "difficulty": difficulty,
                         "is_freeze_trap": is_freeze_trap})
    return examples, category_label_weights


class MemoryPath:
    """Memoria associativa baseada em prototipos. Custo: 1 FLOP."""
    def __init__(self, n_classes, n_categories, input_dim, rng, learning_rate=0.1,
                 temperature=1.0, freeze_category=None):
        self.n_classes, self.n_categories, self.input_dim = n_classes, n_categories, input_dim
        self.lr, self.temperature, self.freeze_category = learning_rate, temperature, freeze_category
        self.prototypes = rng.randn(n_categories, input_dim).astype(np.float32) * 0.1
        for c in range(n_categories):
            self.prototypes[c] /= np.linalg.norm(self.prototypes[c]) + EPSILON
        self.output_biases = np.zeros((n_categories, n_classes), dtype=np.float32)
        if freeze_category is not None and freeze_category < n_categories:
            wrong_dir = rng.randn(input_dim).astype(np.float32)
            wrong_dir /= np.linalg.norm(wrong_dir) + EPSILON
            self.prototypes[freeze_category] = wrong_dir
            self.output_biases[freeze_category] = rng.randn(n_classes).astype(np.float32) * 0.5

    def predict(self, input_vec, category):
        sim = float(np.dot(input_vec, self.prototypes[category]))
        logits = np.full(self.n_classes, sim / self.temperature, dtype=np.float32)
        logits += self.output_biases[category]
        logits_max = logits.max()
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum()
        return logits.astype(np.float32), float(probs.max())

    def update(self, category, input_vec, true_label, confidence, was_used):
        alpha = self.lr * (0.5 if was_used else 1.5)
        old_proto = self.prototypes[category].copy()
        new_proto = (1.0 - alpha) * old_proto + alpha * input_vec
        new_proto /= np.linalg.norm(new_proto) + EPSILON
        self.prototypes[category] = new_proto
        bias_update = np.zeros(self.n_classes, dtype=np.float32)
        bias_update[true_label] = alpha * 0.1
        self.output_biases[category] += bias_update
        self.output_biases[category] = np.clip(self.output_biases[category], -1.0, 1.0)

    def get_flop_cost(self): return 1.0


class JEPAPath:
    """Caminho caro (oraculo com ruido). Custo: cost_ratio FLOPs."""
    def __init__(self, n_classes, input_dim, cost_ratio=10.0, oracle_noise=0.02):
        self.n_classes, self.input_dim = n_classes, input_dim
        self.cost_ratio, self.oracle_noise = cost_ratio, oracle_noise

    def predict(self, input_vec, category_label_weights, category):
        W, b = category_label_weights[category]
        true_logits = input_vec @ W.T + b
        noise = np.random.randn(self.n_classes).astype(np.float32) * self.oracle_noise
        return (true_logits + noise).astype(np.float32)

    def get_flop_cost(self): return self.cost_ratio


class ScaleCalibrator:
    """Normaliza escalas dos logits com running mean/MAD (momentum 0.99)."""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.mem_mean = self.mem_mad = self.jepa_mean = self.jepa_mad = None

    def update(self, mem_logits, jepa_logits):
        mem_arr, jepa_arr = mem_logits.astype(np.float64), jepa_logits.astype(np.float64)
        if self.mem_mean is None:
            self.mem_mean = mem_arr.copy(); self.mem_mad = np.ones_like(mem_arr)
            self.jepa_mean = jepa_arr.copy(); self.jepa_mad = np.ones_like(jepa_arr)
        else:
            m = self.momentum
            n_mem, n_jepa = min(len(self.mem_mean), len(mem_arr)), min(len(self.jepa_mean), len(jepa_arr))
            self.mem_mean[:n_mem] = m * self.mem_mean[:n_mem] + (1.0 - m) * mem_arr[:n_mem]
            self.mem_mad[:n_mem] = m * self.mem_mad[:n_mem] + (1.0 - m) * np.abs(mem_arr[:n_mem] - self.mem_mean[:n_mem])
            self.jepa_mean[:n_jepa] = m * self.jepa_mean[:n_jepa] + (1.0 - m) * jepa_arr[:n_jepa]
            self.jepa_mad[:n_jepa] = m * self.jepa_mad[:n_jepa] + (1.0 - m) * np.abs(jepa_arr[:n_jepa] - self.jepa_mean[:n_jepa])

    def calibrate(self, logits, source):
        mean, mad = (self.mem_mean, self.mem_mad) if source == "memory" else (self.jepa_mean, self.jepa_mad)
        if mean is None or mad is None: return logits
        n = min(len(mean), len(logits))
        calibrated = logits.copy().astype(np.float64)
        calibrated[:n] = (logits[:n] - mean[:n]) / (mad[:n] + EPSILON)
        return calibrated.astype(np.float32)


class DualPathRouter:
    """Decide entre caminho barato (memoria) e caro (JEPA)."""
    def __init__(self, memory, jepa, calibrator, compute_floor=0.05,
                 confidence_threshold=0.80, mode="correct", rng=None):
        self.memory, self.jepa, self.calibrator = memory, jepa, calibrator
        self.compute_floor = max(0.0, min(1.0, compute_floor))
        self.confidence_threshold, self.mode = confidence_threshold, mode
        self.rng = rng or np.random.RandomState(42)
        self.n_cheap_routed = self.n_expensive_routed = self.n_floor_forced = 0

    def decide(self, input_vec, category, category_label_weights):
        mem_logits, _ = self.memory.predict(input_vec, category)
        jepa_logits = self.jepa.predict(input_vec, category_label_weights, category)
        self.calibrator.update(mem_logits, jepa_logits)
        cal_mem_logits = self.calibrator.calibrate(mem_logits, "memory")
        cal_mem_max = cal_mem_logits.max()
        cal_mem_probs = np.exp(cal_mem_logits - cal_mem_max) / np.exp(cal_mem_logits - cal_mem_max).sum()
        calibrated_confidence = float(cal_mem_probs.max())
        was_floor_forced = False
        if self.mode == "correct":
            if self.rng.random() < self.compute_floor:
                use_cheap, was_floor_forced = False, True
            else:
                use_cheap = calibrated_confidence >= self.confidence_threshold
        elif self.mode == "random":
            use_cheap = self.rng.random() >= self.compute_floor
        else:
            use_cheap = False
        chosen_logits = mem_logits if use_cheap else jepa_logits
        if use_cheap: self.n_cheap_routed += 1
        else: self.n_expensive_routed += 1
        if was_floor_forced: self.n_floor_forced += 1
        return use_cheap, calibrated_confidence, chosen_logits, was_floor_forced


# ====================================================================
# SIMULACAO + VISUALIZACAO + MAIN
# ====================================================================

def run_simulation(args):
    rng = np.random.RandomState(args.seed)
    freeze_cat = min(5, args.n_categories - 1) if args.n_categories > 2 else None
    freeze_start = args.num_examples // 3
    examples, category_label_weights = generate_example_stream(
        args.num_examples, args.n_classes, args.n_categories, args.input_dim, rng,
        belief_freeze_category=freeze_cat, freeze_start=freeze_start)
    rng_mem = np.random.RandomState(args.seed)
    memory = MemoryPath(args.n_classes, args.n_categories, args.input_dim, rng_mem,
                        learning_rate=0.15, temperature=1.0, freeze_category=freeze_cat)
    jepa = JEPAPath(args.n_classes, args.input_dim, cost_ratio=args.cost_ratio, oracle_noise=0.02)
    calibrator = ScaleCalibrator(momentum=0.99)
    router = DualPathRouter(memory=memory, jepa=jepa, calibrator=calibrator,
                            compute_floor=args.compute_floor,
                            confidence_threshold=args.confidence_threshold,
                            mode=args.router_mode, rng=rng)
    total_correct_from_cheap = total_correct_from_expensive = 0
    total_cheap_routed = total_expensive_routed = 0
    total_flops = total_correct = total_examples = 0.0
    freeze_correct_with_floor = freeze_total_with_floor = 0
    freeze_correct_without_floor = freeze_total_without_floor = 0
    window_size = max(50, args.num_examples // 40)
    recent_routed, recent_correct, recent_flops = [], [], []
    per_example_log, metric_history, timesteps = [], [], []
    log_interval = max(1, args.num_examples // 200)

    for t, ex in enumerate(examples):
        use_cheap, conf, logits, was_floor_forced = router.decide(ex["input_vec"], ex["category"], category_label_weights)
        pred_label = int(np.argmax(logits))
        correct = pred_label == ex["true_label"]
        if ex.get("is_freeze_trap"):
            if was_floor_forced:
                freeze_total_with_floor += 1
                if correct: freeze_correct_with_floor += 1
            else:
                freeze_total_without_floor += 1
                if correct: freeze_correct_without_floor += 1
        total_examples += 1
        if correct: total_correct += 1
        if use_cheap:
            total_cheap_routed += 1; total_flops += memory.get_flop_cost()
            if correct: total_correct_from_cheap += 1
        else:
            total_expensive_routed += 1; total_flops += jepa.get_flop_cost()
            if correct: total_correct_from_expensive += 1
        recent_routed.append(use_cheap); recent_correct.append(correct)
        recent_flops.append(memory.get_flop_cost() if use_cheap else jepa.get_flop_cost())
        if len(recent_routed) > window_size:
            recent_routed.pop(0); recent_correct.pop(0); recent_flops.pop(0)
        per_example_log.append({"index": t, "category": ex["category"],
                                "difficulty": ex["difficulty"], "use_cheap": use_cheap,
                                "confidence": conf, "correct": correct,
                                "is_freeze_trap": ex.get("is_freeze_trap", False),
                                "was_floor_forced": was_floor_forced})
        if t % log_interval == 0 or t == len(examples) - 1:
            w = max(1, len(recent_routed))
            correct_cheap_w = sum(recent_routed[i] and recent_correct[i] for i in range(w))
            total_correct_w = correct_cheap_w + sum((not recent_routed[i]) and recent_correct[i] for i in range(w))
            actual_flops_w = sum(recent_flops)
            all_exp_flops = w * jepa.get_flop_cost()
            timesteps.append(t)
            metric_history.append({
                "step": t,
                "frac_gain_memory": correct_cheap_w / max(1, total_correct_w),
                "frac_routed_cheap": sum(recent_routed) / w,
                "frac_flops_saved": 1.0 - actual_flops_w / max(EPSILON, all_exp_flops),
                "window_accuracy": sum(recent_correct) / w,
                "mean_confidence": float(np.mean([p["confidence"] for p in per_example_log[-w:]]))})
        if (t + 1) % args.memory_update_rate == 0 and t > 0:
            for rw in per_example_log[-args.memory_update_rate:]:
                if rw["category"] < args.n_categories:
                    memory.update(category=rw["category"], input_vec=examples[rw["index"]]["input_vec"],
                                  true_label=examples[rw["index"]]["true_label"],
                                  confidence=rw["confidence"], was_used=rw["use_cheap"])

    final_frac_gain_memory = total_correct_from_cheap / max(1, total_correct)
    final_frac_routed_cheap = total_cheap_routed / max(1, total_examples)
    final_frac_flops_saved = 1.0 - total_flops / (total_examples * jepa.get_flop_cost())
    return {
        "config": vars(args),
        "final_metrics": {
            "frac_gain_memory": round(final_frac_gain_memory, 4),
            "frac_routed_cheap": round(final_frac_routed_cheap, 4),
            "frac_flops_saved": round(final_frac_flops_saved, 4),
            "overall_accuracy": round(total_correct / max(1, total_examples), 4),
            "total_cheap_routed": total_cheap_routed,
            "total_expensive_routed": total_expensive_routed,
            "total_floor_forced": router.n_floor_forced,
            "total_flops": round(total_flops, 1)},
        "freeze_trap": {
            "accuracy_with_floor": round(freeze_correct_with_floor / max(1, freeze_total_with_floor), 4),
            "accuracy_without_floor": round(freeze_correct_without_floor / max(1, freeze_total_without_floor), 4),
            "total_with_floor": freeze_total_with_floor,
            "total_without_floor": freeze_total_without_floor},
        "metric_history": metric_history, "timesteps": timesteps,
        "per_example_log": per_example_log[-500:]}


def visualize(results, output_dir, args):
    if not HAS_MPL: return
    output_dir.mkdir(parents=True, exist_ok=True)
    history = results["metric_history"]
    timesteps = results["timesteps"]
    per_example = results["per_example_log"]
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(f"Router Dual-Path | modo={args.router_mode}  floor={args.compute_floor}  thresh={args.confidence_threshold}  N={args.num_examples}  seed={args.seed}", fontsize=14, fontweight="bold")
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(timesteps, [h["frac_gain_memory"] for h in history], "b-", linewidth=1.5, alpha=0.85)
    ax1.axhline(y=results["final_metrics"]["frac_gain_memory"], color="b", linestyle="--", alpha=0.4)
    ax1.set_ylabel("Frac. Ganho Memoria"); ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)
    ax1.set_title("1) Fracao do Ganho Preditivo explicada pela Memoria")
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(timesteps, [h["frac_routed_cheap"] for h in history], "g-", linewidth=1.5, alpha=0.85)
    ax2.axhline(y=results["final_metrics"]["frac_routed_cheap"], color="g", linestyle="--", alpha=0.4)
    ax2.set_ylabel("Frac. Roteado Barato"); ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3)
    ax2.set_title("2) Fracao de Exemplos Roteados ao Caminho Barato")
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(timesteps, [h["frac_flops_saved"] for h in history], "orange", linewidth=1.5, alpha=0.85)
    ax3.axhline(y=results["final_metrics"]["frac_flops_saved"], color="orange", linestyle="--", alpha=0.4)
    ax3.set_ylabel("Frac. FLOPs Economizada"); ax3.set_ylim(0, 1.05); ax3.grid(True, alpha=0.3)
    ax3.set_title("3) Fracao de FLOPs Economizada vs. All-Expensive")
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(timesteps, [h["window_accuracy"] for h in history], "purple", linewidth=1.5, alpha=0.85)
    ax4.set_ylabel("Acuracia"); ax4.set_ylim(0, 1.05); ax4.grid(True, alpha=0.3)
    ax4.set_title("Acuracia em Janela Movel")
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(timesteps, [h["mean_confidence"] for h in history], "brown", linewidth=1.5, alpha=0.85)
    ax5.axhline(y=args.confidence_threshold, color="red", linestyle="--", alpha=0.7)
    ax5.set_ylabel("Confianca Media"); ax5.set_ylim(0, 1.05); ax5.grid(True, alpha=0.3)
    ax5.set_title("Confianca Media da Memoria")
    ax6 = fig.add_subplot(3, 3, 6)
    confidences = [p["confidence"] for p in per_example]
    correct_mask = [p["correct"] for p in per_example]
    conf_correct = [c for c, cor in zip(confidences, correct_mask) if cor]
    conf_wrong = [c for c, cor in zip(confidences, correct_mask) if not cor]
    bins = np.linspace(0, 1, 31)
    ax6.hist(conf_correct, bins=bins, alpha=0.6, color="green", label=f"Corretos (n={len(conf_correct)})")
    ax6.hist(conf_wrong, bins=bins, alpha=0.6, color="red", label=f"Errados (n={len(conf_wrong)})")
    ax6.axvline(x=args.confidence_threshold, color="black", linestyle="--", linewidth=1.5)
    ax6.set_xlabel("Confianca Calibrada"); ax6.set_title("Histograma de Confianca c/ Threshold"); ax6.legend(fontsize=7)
    ax7 = fig.add_subplot(3, 3, 7)
    difficulties = np.array([p["difficulty"] for p in per_example])
    use_cheap_arr = np.array([p["use_cheap"] for p in per_example])
    diff_bins = np.linspace(0, 1, 11)
    cheap_by_diff = []
    for i in range(len(diff_bins) - 1):
        mask = (difficulties >= diff_bins[i]) & (difficulties < diff_bins[i + 1])
        cheap_by_diff.append(use_cheap_arr[mask].mean() if mask.sum() > 0 else np.nan)
    ax7.bar((diff_bins[:-1] + diff_bins[1:]) / 2, cheap_by_diff, width=0.08, color="steelblue", edgecolor="navy", alpha=0.8)
    ax7.axhline(y=1.0 - args.compute_floor, color="red", linestyle="--", alpha=0.5)
    ax7.set_xlabel("Dificuldade"); ax7.set_ylim(0, 1.05); ax7.grid(True, alpha=0.3, axis="y")
    ax7.set_title("Roteamento por Faixa de Dificuldade")
    ax8 = fig.add_subplot(3, 3, 8)
    categories = np.array([p["category"] for p in per_example])
    correct_arr = np.array([p["correct"] for p in per_example])
    cat_ids = sorted(set(categories))
    cheap_by_cat = [use_cheap_arr[categories == cat].mean() for cat in cat_ids]
    acc_by_cat = [correct_arr[categories == cat].mean() for cat in cat_ids]
    x_cat, w_bar = np.arange(len(cat_ids)), 0.35
    ax8.bar(x_cat - w_bar / 2, cheap_by_cat, w_bar, color="steelblue", alpha=0.8, label="Frac. Barato")
    ax8.bar(x_cat + w_bar / 2, acc_by_cat, w_bar, color="darkgreen", alpha=0.8, label="Acuracia")
    ax8.set_xticks(x_cat); ax8.set_xticklabels([str(c) for c in cat_ids])
    ax8.set_ylim(0, 1.05); ax8.legend(fontsize=7); ax8.grid(True, alpha=0.3, axis="y")
    ax8.set_title("Roteamento e Acuracia por Categoria")
    ax9 = fig.add_subplot(3, 3, 9); ax9.axis("off")
    m, ft = results["final_metrics"], results["freeze_trap"]
    delta_freeze = ft["accuracy_with_floor"] - ft["accuracy_without_floor"]
    for i, line in enumerate([
        f"MODO: {args.router_mode.upper()}", "",
        "Metricas Finais:",
        f"  Ganho Memoria:       {m['frac_gain_memory']:.1%}",
        f"  Roteado Barato:       {m['frac_routed_cheap']:.1%}",
        f"  FLOPs Economizados:   {m['frac_flops_saved']:.1%}",
        f"  Acuracia Global:      {m['overall_accuracy']:.1%}",
        f"  Total Barato / Caro:  {m['total_cheap_routed']} / {m['total_expensive_routed']}",
        f"  Forcados p/ Floor:    {m['total_floor_forced']}", "",
        "Freeze Trap (crenca congelada):",
        f"  Acc s/ floor: {ft['accuracy_without_floor']:.1%}  (n={ft['total_without_floor']})",
        f"  Acc c/ floor: {ft['accuracy_with_floor']:.1%}  (n={ft['total_with_floor']})",
        f"  Delta (floor): {delta_freeze:+.1%}"]):
        ax9.text(0.05, 0.96 - i * 0.063, line, transform=ax9.transAxes, fontfamily="monospace", fontsize=8.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / f"router_sim_{args.router_mode}_n{args.num_examples}_s{args.seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grafico salvo: {output_dir}")


def print_report(results, args):
    m, ft = results["final_metrics"], results["freeze_trap"]
    delta = ft["accuracy_with_floor"] - ft["accuracy_without_floor"]
    print("\n" + "=" * 72)
    print("  ROUTER DUAL-PATH SIMULATION REPORT")
    print("=" * 72)
    print(f"  Mode: {args.router_mode} | N: {args.num_examples} | Floor: {args.compute_floor} | Thresh: {args.confidence_threshold} | Seed: {args.seed}")
    print("-" * 72)
    print(f"  Fracao Ganho Memoria:     {m['frac_gain_memory']:.2%}")
    print(f"  Fracao Roteado Barato:    {m['frac_routed_cheap']:.2%}")
    print(f"  Fracao FLOPs Economizada: {m['frac_flops_saved']:.2%}")
    print(f"  Acuracia Global:          {m['overall_accuracy']:.2%}")
    print(f"  Freeze Delta (floor):     {delta:+.2%}")
    print("=" * 72 + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_simulation(args)
    json_path = output_dir / f"results_{args.router_mode}_n{args.num_examples}_s{args.seed}.json"
    json_payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "args": vars(args), **results}
    for entry in json_payload.get("per_example_log", []):
        for k, v in list(entry.items()):
            if isinstance(v, (np.integer,)): entry[k] = int(v)
            elif isinstance(v, (np.floating,)): entry[k] = float(v)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] JSON: {json_path}")
    print_report(results, args)
    if not args.no_plot: visualize(results, output_dir, args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
