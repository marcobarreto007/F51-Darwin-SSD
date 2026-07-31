#!/usr/bin/env python3
"""
Simulacao de Formacao de Memoria Episodica Entidade--Evento
============================================================
Implementa uma simulacao executavel da formacao de memoria episodica baseada em
entidades e eventos, ancorada nas fontes primarias de 1940-1951.

Uso:
    python research/memory_entity_sim.py \
        --num-tokens 5000 \
        --num-entities 8 \
        --seed 42 \
        --output-dir workspace/runtime/memory_sim/

    # Com controle embaralhado:
    python research/memory_entity_sim.py \
        --num-tokens 5000 \
        --num-entities 8 \
        --seed 42 \
        --shuffle-memory

ANCORAGEM TEORICA (Parte B do requisito)
------------------------------------------
Cada conceito implementado e sua fonte primaria exata:

1. ENTIDADE — Padrao distribuido de features que co-ocorrem estavelmente.
   Fonte: Hebb, D.O. (1949). The Organization of Behavior. Wiley.
   Conceito: "cell assembly" — co-ativacao repetida de features.
   Fonte complementar: Ashby, W.R. (1940). "Adaptiveness and Equilibrium."
   J. Mental Science, 86:478-483. Conceito: restricao reduz variedade.

2. EVENTO — Transicao de estado com delta temporal e assimetria causal.
   Fonte: Reichenbach, H. (1956). The Direction of Time. UC Press.
   (Palestras originais na UCLA, 1947.)
   Conceito: "Mark Method" — assimetria temporal distingue causa de efeito.

3. RELACAO — Conexao entre entidades computada de forma BARATA.
   Fonte: Hebb, D.O. (1949). regra de Hebb — co-ativacao repetida.
   Fonte complementar: Ashby, W.R. (1947). restricao como essencia da organizacao.

4. RELATION_SCORE — Jaccard e PMI incrementais. O(1) por par.
   Fonte: Shannon, C.E. (1948). informacao mutua.

5. STATE_CHANGE — Running surprisal + JS divergence. O(1) por token.
   Fonte: Shannon, C.E. (1948). equivocation. Wiener, N. (1948). innovation.

O QUE E "BARATO" E POR QUE IMPORTA
------------------------------------
Ashby, Lei da Variedade Requerida: V(R) >= V(D)/V(S).
"Barato" = O(1) ou O(log n) por token. NUNCA O(n^2).
Se o mecanismo de memoria custasse O(n^2), a variedade do regulador
seria insuficiente para acompanhar a variedade do ambiente.

VIESES DO GROUND TRUTH SINTETICO
----------------------------------
1. Entidades tem conjuntos fixos de features — cell assemblies reais sao dinamicas
2. Eventos sao discretos — na realidade se sobrepoem
3. Features sao condicionalmente independentes
4. Nao ha esquecimento — Lashley (1950) mostra que memoria se degrada
5. A sequencia e estacionaria — sem concept drift

Autoridade: governance/docs/pesquisa/FONTES_PRIMARIAS_1940_1951.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EPSILON = 1e-10
ROOT = Path(__file__).resolve().parents[1]


# ===================================================================
# GROUND TRUTH: Entidades e Eventos Sinteticos
# ===================================================================

@dataclass
class GroundTruthEntity:
    entity_id: int
    core_features: set[int]
    variable_features: set[int]
    label: str = ""


@dataclass
class GroundTruthEvent:
    time: int
    subject_entity: int
    object_entity: int
    relation_type: int
    state_change_magnitude: float


class GroundTruthGenerator:
    """Gera sequencia sintetica com estrutura entidade-evento conhecida."""

    def __init__(
        self, num_entities: int = 8, num_features: int = 200,
        features_per_entity: int = 6, variable_features_per_entity: int = 2,
        feature_overlap: int = 1, seed: int = 42,
    ):
        self.num_entities = num_entities
        self.num_features = num_features
        self.features_per_entity = features_per_entity
        self.variable_features_per_entity = variable_features_per_entity
        self.feature_overlap = feature_overlap
        self.rng = np.random.RandomState(seed)
        self.entities: list[GroundTruthEntity] = []
        self.events: list[GroundTruthEvent] = []
        self._relation_matrix: np.ndarray | None = None
        self._generate_entities()

    def _generate_entities(self) -> None:
        all_features = list(range(self.num_features))
        used_core: list[set[int]] = []
        for eid in range(self.num_entities):
            core_pool: set[int] = set()
            if used_core and self.feature_overlap > 0:
                prev_core = used_core[-1]
                overlap_take = min(self.feature_overlap, len(prev_core))
                core_pool.update(sorted(prev_core)[:overlap_take])
            remaining = [f for f in all_features if f not in core_pool]
            self.rng.shuffle(remaining)
            needed = self.features_per_entity - len(core_pool)
            core_pool.update(remaining[:needed])
            remaining2 = [f for f in all_features if f not in core_pool]
            self.rng.shuffle(remaining2)
            var_pool = set(remaining2[:self.variable_features_per_entity])
            self.entities.append(GroundTruthEntity(
                entity_id=eid, core_features=core_pool,
                variable_features=var_pool, label=f"E{eid}",
            ))
            used_core.append(core_pool)
        self._relation_matrix = np.zeros((self.num_entities, self.num_entities))
        for i in range(self.num_entities):
            for j in range(i + 1, self.num_entities):
                overlap = len(self.entities[i].core_features & self.entities[j].core_features)
                base = 0.1 + 0.15 * overlap + self.rng.uniform(-0.05, 0.15)
                self._relation_matrix[i, j] = max(0.0, min(1.0, base))
                self._relation_matrix[j, i] = self._relation_matrix[i, j]

    def generate_sequence(self, num_tokens: int) -> list[set[int]]:
        sequence: list[set[int]] = []
        self.events = []
        t = 0
        event_duration_mean = 12
        gap_duration_mean = 20
        event_prob = 0.25
        current_event: GroundTruthEvent | None = None
        event_remaining = 0
        gap_remaining = 0
        while t < num_tokens:
            if event_remaining > 0:
                features: set[int] = set()
                assert current_event is not None
                if current_event.subject_entity >= 0:
                    ent_s = self.entities[current_event.subject_entity]
                    features.update(ent_s.core_features)
                    for vf in ent_s.variable_features:
                        if self.rng.random() < 0.6:
                            features.add(vf)
                if current_event.object_entity >= 0:
                    ent_o = self.entities[current_event.object_entity]
                    features.update(ent_o.core_features)
                    for vf in ent_o.variable_features:
                        if self.rng.random() < 0.6:
                            features.add(vf)
                if self.rng.random() < 0.05:
                    features.add(self.rng.randint(0, self.num_features))
                sequence.append(features)
                event_remaining -= 1
                t += 1
                if event_remaining == 0:
                    current_event = None
                    gap_remaining = self.rng.poisson(gap_duration_mean)
            else:
                features_bg: set[int] = set()
                for _ in range(self.rng.poisson(1.5)):
                    features_bg.add(self.rng.randint(0, self.num_features))
                sequence.append(features_bg)
                t += 1
                gap_remaining -= 1
                if gap_remaining <= 0 and self.rng.random() < event_prob:
                    s = self.rng.randint(0, self.num_entities)
                    o = self.rng.randint(0, self.num_entities)
                    while o == s and self.rng.random() < 0.7:
                        o = self.rng.randint(0, self.num_entities)
                    rel_str = self._relation_matrix[s, o] if s != o else 0.5
                    magnitude = float(0.3 + 0.7 * rel_str + 0.1 * self.rng.randn())
                    magnitude = max(0.0, min(1.0, magnitude))
                    ev = GroundTruthEvent(
                        time=t, subject_entity=s, object_entity=o,
                        relation_type=self.rng.randint(0, 5),
                        state_change_magnitude=magnitude,
                    )
                    self.events.append(ev)
                    current_event = ev
                    event_remaining = max(3, self.rng.poisson(event_duration_mean))
        return sequence

    def get_ground_truth_entity_features(self) -> dict[int, set[int]]:
        return {e.entity_id: e.core_features | e.variable_features for e in self.entities}

    def get_ground_truth_relation_strengths(self) -> np.ndarray:
        assert self._relation_matrix is not None
        return self._relation_matrix.copy()

    def get_event_boundaries(self) -> set[int]:
        return {ev.time for ev in self.events}


# ===================================================================
# BANCO DE MEMORIA EPISODICA (Incremental)
# ===================================================================

@dataclass
class DiscoveredEntity:
    entity_id: int
    features: set[int]
    created_at: int
    last_seen: int


@dataclass
class RelationRecord:
    entity_a: int
    entity_b: int
    co_occurrence_count: int = 0
    a_alone_count: int = 0
    b_alone_count: int = 0
    jaccard_score: float = 0.0
    pmi_score: float = 0.0


class EntityBank:
    """Banco incremental de entidades. O(1) amortizado por token.
    Hebb (1949): cell assembly via co-ativacao."""

    def __init__(self, window_size: int = 64, cooccur_threshold: float = 0.35):
        self.window_size = window_size
        self.cooccur_threshold = cooccur_threshold
        self.entities: dict[int, DiscoveredEntity] = {}
        self._next_entity_id = 0
        self._cooccur: dict[tuple[int, int], int] = {}
        self._window_features: list[set[int]] = []
        self._total_windows = 0
        self._feature_to_entity: dict[int, int] = {}

    def process_token(self, features: set[int]) -> set[int]:
        if not features:
            return set()
        self._window_features.append(features)
        if len(self._window_features) > self.window_size:
            self._window_features.pop(0)
        self._total_windows += 1
        features_list = sorted(features)
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                key = (features_list[i], features_list[j])
                self._cooccur[key] = self._cooccur.get(key, 0) + 1
        if self._total_windows % self.window_size == 0:
            self._discover_entities()
        active_entities: set[int] = set()
        for f in features:
            if f in self._feature_to_entity:
                active_entities.add(self._feature_to_entity[f])
        for eid in active_entities:
            if eid in self.entities:
                self.entities[eid].last_seen = self._total_windows
        return active_entities

    def _discover_entities(self) -> None:
        if len(self._window_features) < 4:
            return
        total_possible = len(self._window_features)
        edges: dict[int, set[int]] = defaultdict(set)
        all_nodes: set[int] = set()
        for (f1, f2), count in self._cooccur.items():
            rate = count / max(1, total_possible)
            if rate >= self.cooccur_threshold:
                edges[f1].add(f2)
                edges[f2].add(f1)
                all_nodes.add(f1)
                all_nodes.add(f2)
        visited: set[int] = set()
        components: list[set[int]] = []
        for node in all_nodes:
            if node in visited:
                continue
            comp: set[int] = set()
            stack = [node]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                for neighbor in edges.get(n, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)
            if len(comp) >= 2:
                components.append(comp)
        known_entity_features = {eid: ent.features for eid, ent in self.entities.items()}
        matched_entities: set[int] = set()
        for comp in components:
            best_jaccard = 0.0
            best_eid = -1
            for eid, efeatures in known_entity_features.items():
                if eid in matched_entities:
                    continue
                inter = len(comp & efeatures)
                union = len(comp | efeatures)
                jaccard = inter / max(1, union)
                if jaccard > best_jaccard and jaccard > 0.2:
                    best_jaccard = jaccard
                    best_eid = eid
            if best_eid >= 0:
                self.entities[best_eid].features |= comp
                self.entities[best_eid].last_seen = self._total_windows
                matched_entities.add(best_eid)
                for f in comp:
                    self._feature_to_entity[f] = best_eid
            else:
                eid = self._next_entity_id
                self._next_entity_id += 1
                self.entities[eid] = DiscoveredEntity(
                    entity_id=eid, features=comp,
                    created_at=self._total_windows, last_seen=self._total_windows,
                )
                for f in comp:
                    self._feature_to_entity[f] = eid


class RelationBank:
    """Banco incremental de relacoes. Hebb (1949) + Shannon (1948)."""

    def __init__(self):
        self.relations: dict[tuple[int, int], RelationRecord] = {}
        self._window_active_sets: list[set[int]] = []
        self._max_window_history = 128

    def process_window(self, active_entities: set[int]) -> None:
        self._window_active_sets.append(active_entities)
        if len(self._window_active_sets) > self._max_window_history:
            self._window_active_sets.pop(0)
        active_list = sorted(active_entities)
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                a, b = active_list[i], active_list[j]
                key = (a, b) if a < b else (b, a)
                if key not in self.relations:
                    self.relations[key] = RelationRecord(entity_a=key[0], entity_b=key[1])
                self.relations[key].co_occurrence_count += 1
        self._recompute_scores()

    def _recompute_scores(self) -> None:
        total = max(1, len(self._window_active_sets))
        for rec in self.relations.values():
            co = rec.co_occurrence_count
            a_total = co + rec.a_alone_count
            b_total = co + rec.b_alone_count
            union_approx = a_total + b_total - co
            rec.jaccard_score = co / max(1, union_approx) if union_approx > 0 else 0.0
            p_ab = co / total
            p_a = a_total / total
            p_b = b_total / total
            if p_ab > 0 and p_a > 0 and p_b > 0:
                pmi = math.log(p_ab / (p_a * p_b))
                rec.pmi_score = max(0.0, pmi)
            else:
                rec.pmi_score = 0.0

    def get_relation_scores(self, score_type: str = "jaccard") -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for key, rec in self.relations.items():
            result[key] = rec.jaccard_score if score_type == "jaccard" else rec.pmi_score
        return result

    def shuffle(self, seed: int = 999) -> "RelationBank":
        """Controle embaralhado (Reichenbach 1947)."""
        shuffled = RelationBank()
        shuffled._window_active_sets = self._window_active_sets.copy()
        shuffled._max_window_history = self._max_window_history
        rng = np.random.RandomState(seed)
        keys = list(self.relations.keys())
        values = list(self.relations.values())
        indices = list(range(len(values)))
        rng.shuffle(indices)
        for orig_key, shuffled_idx in zip(keys, indices):
            orig_rec = values[shuffled_idx]
            shuffled.relations[orig_key] = RelationRecord(
                entity_a=orig_key[0], entity_b=orig_key[1],
                co_occurrence_count=orig_rec.co_occurrence_count,
                a_alone_count=orig_rec.a_alone_count,
                b_alone_count=orig_rec.b_alone_count,
                jaccard_score=orig_rec.jaccard_score,
                pmi_score=orig_rec.pmi_score,
            )
        return shuffled


class StateTracker:
    """Rastreador de state_change barato. Shannon (1948) + Wiener (1948)."""

    def __init__(self, window_recent: int = 32, window_background: int = 128, num_features: int = 200):
        self.window_recent = window_recent
        self.window_background = window_background
        self.num_features = num_features
        self._recent_counts: np.ndarray = np.zeros(num_features, dtype=np.float64)
        self._recent_total = 0
        self._recent_history: list[set[int]] = []
        self._bg_counts: np.ndarray = np.zeros(num_features, dtype=np.float64)
        self._bg_total = 0
        self._bg_history: list[set[int]] = []
        self.state_change_history: list[float] = []

    def process_token(self, features: set[int]) -> float:
        if not features:
            self.state_change_history.append(0.0)
            return 0.0
        self._recent_history.append(features)
        for f in features:
            if 0 <= f < self.num_features:
                self._recent_counts[f] += 1
                self._recent_total += 1
        if len(self._recent_history) > self.window_recent:
            old = self._recent_history.pop(0)
            for f in old:
                if 0 <= f < self.num_features:
                    self._recent_counts[f] -= 1
                    self._recent_total -= 1
        self._bg_history.append(features)
        for f in features:
            if 0 <= f < self.num_features:
                self._bg_counts[f] += 1
                self._bg_total += 1
        if len(self._bg_history) > self.window_background:
            old = self._bg_history.pop(0)
            for f in old:
                if 0 <= f < self.num_features:
                    self._bg_counts[f] -= 1
                    self._bg_total -= 1
        surprisals = []
        for f in features:
            if 0 <= f < self.num_features:
                p_bg = (self._bg_counts[f] + 1.0) / max(1, self._bg_total + self.num_features)
                surprisals.append(-math.log(max(p_bg, EPSILON)))
        avg_surprisal = float(np.mean(surprisals)) if surprisals else 0.0
        js_div = self._compute_js_divergence()
        surprisal_norm = min(1.0, avg_surprisal / 5.0)
        js_norm = min(1.0, js_div)
        state_change = 0.7 * surprisal_norm + 0.3 * js_norm
        self.state_change_history.append(float(state_change))
        return float(state_change)

    def _compute_js_divergence(self) -> float:
        if self._recent_total < 4 or self._bg_total < 4:
            return 0.0
        r_prob = (self._recent_counts + EPSILON) / (self._recent_total + EPSILON * self.num_features)
        b_prob = (self._bg_counts + EPSILON) / (self._bg_total + EPSILON * self.num_features)
        m_prob = 0.5 * (r_prob + b_prob)
        kl_rm = np.sum(r_prob * np.log(r_prob / m_prob))
        kl_bm = np.sum(b_prob * np.log(b_prob / m_prob))
        return float(max(0.0, 0.5 * kl_rm + 0.5 * kl_bm))


class EpisodicMemory:
    """Integra EntityBank, RelationBank e StateTracker."""

    def __init__(self, window_size: int = 64, cooccur_threshold: float = 0.35,
                 num_features: int = 200, window_recent: int = 32, window_background: int = 128):
        self.entity_bank = EntityBank(window_size=window_size, cooccur_threshold=cooccur_threshold)
        self.relation_bank = RelationBank()
        self.state_tracker = StateTracker(
            window_recent=window_recent, window_background=window_background, num_features=num_features)
        self._step_counter = 0

    def process_token(self, features: set[int]) -> dict[str, Any]:
        active = self.entity_bank.process_token(features)
        self.relation_bank.process_window(active)
        sc = self.state_tracker.process_token(features)
        self._step_counter += 1
        return {"active_entities": active, "state_change": sc,
                "is_event_boundary": sc > 0.45, "step": self._step_counter}

    def get_discovered_entities(self) -> dict[int, DiscoveredEntity]:
        return self.entity_bank.entities

    def get_relation_scores(self, score_type: str = "jaccard") -> dict[tuple[int, int], float]:
        return self.relation_bank.get_relation_scores(score_type)

    def get_state_change_history(self) -> list[float]:
        return self.state_tracker.state_change_history


# ===================================================================
# AVALIACAO
# ===================================================================

def match_entities(discovered: dict[int, DiscoveredEntity],
                   ground_truth: dict[int, set[int]]) -> tuple[dict[int, int], float]:
    """Greedy Hungarian approximation para emparelhar entidades."""
    if not discovered or not ground_truth:
        return {}, 0.0
    disc_ids = list(discovered.keys())
    gt_ids = list(ground_truth.keys())
    scores = np.zeros((len(disc_ids), len(gt_ids)))
    for di, did in enumerate(disc_ids):
        for gj, gid in enumerate(gt_ids):
            inter = len(discovered[did].features & ground_truth[gid])
            union = len(discovered[did].features | ground_truth[gid])
            scores[di, gj] = inter / max(1, union)
    matching: dict[int, int] = {}
    used_gt: set[int] = set()
    pairs = [(scores[di, gj], di, gj) for di in range(len(disc_ids)) for gj in range(len(gt_ids))]
    pairs.sort(reverse=True)
    for score, di, gj in pairs:
        if score < 0.1:
            break
        if gj not in used_gt:
            matching[disc_ids[di]] = gt_ids[gj]
            used_gt.add(gj)
    matched_scores = [scores[disc_ids.index(did), gt_ids.index(gid)] for did, gid in matching.items()]
    mean_jaccard = float(np.mean(matched_scores)) if matched_scores else 0.0
    return matching, mean_jaccard


def evaluate_relation_quality(predicted_scores: dict[tuple[int, int], float],
                              ground_truth_matrix: np.ndarray,
                              entity_matching: dict[int, int]) -> dict[str, float]:
    if not entity_matching:
        return {"relation_mae": 1.0, "relation_corr": 0.0, "n_pairs": 0}
    gt_to_disc = {v: k for k, v in entity_matching.items()}
    n_gt = ground_truth_matrix.shape[0]
    gt_vals, pred_vals = [], []
    for i in range(n_gt):
        for j in range(i + 1, n_gt):
            if i in gt_to_disc and j in gt_to_disc:
                di, dj = gt_to_disc[i], gt_to_disc[j]
                key = (di, dj) if di < dj else (dj, di)
                gt_vals.append(ground_truth_matrix[i, j])
                pred_vals.append(predicted_scores.get(key, 0.0))
    if not gt_vals:
        return {"relation_mae": 1.0, "relation_corr": 0.0, "n_pairs": 0}
    gt_arr, pred_arr = np.array(gt_vals), np.array(pred_vals)
    mae = float(np.mean(np.abs(gt_arr - pred_arr)))
    gt_std, pred_std = np.std(gt_arr), np.std(pred_arr)
    corr = float(np.corrcoef(gt_arr, pred_arr)[0, 1]) if gt_std > EPSILON and pred_std > EPSILON else 0.0
    return {"relation_mae": mae, "relation_corr": max(-1.0, min(1.0, corr)), "n_pairs": len(gt_vals)}


def evaluate_event_detection(state_change_history: list[float],
                             event_boundaries: set[int],
                             threshold: float = 0.45) -> dict[str, float]:
    if not state_change_history or not event_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    predicted = {i for i, sc in enumerate(state_change_history) if sc > threshold}
    tp = len(predicted & event_boundaries)
    fp = len(predicted - event_boundaries)
    fn = len(event_boundaries - predicted)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(EPSILON, p + r)
    return {"precision": p, "recall": r, "f1": f1}


# ===================================================================
# VISUALIZACAO
# ===================================================================

def setup_matplotlib() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass


def plot_results(sequence, ground_truth, memory, metrics, output_dir, prefix="memory_sim") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib nao disponivel — pulando visualizacao.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        plt.style.use("default")
    _plot_entity_comparison(memory, ground_truth, output_dir, prefix, plt)
    _plot_relation_matrix(memory, ground_truth, metrics, output_dir, prefix, plt)
    _plot_formation_curve(memory, ground_truth, output_dir, prefix, plt)
    _plot_state_change(memory, ground_truth, output_dir, prefix, plt)
    print(f"[viz] Graficos salvos em: {output_dir}")


def _plot_entity_comparison(memory, ground_truth, output_dir, prefix, plt) -> None:
    discovered = memory.get_discovered_entities()
    gt_features = ground_truth.get_ground_truth_entity_features()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    gt_map = np.zeros((len(gt_features), ground_truth.num_features))
    for eid, feats in sorted(gt_features.items()):
        for f in feats:
            if f < ground_truth.num_features:
                gt_map[eid, f] = 1
    ax1.imshow(gt_map, aspect="auto", cmap="Blues", interpolation="none")
    ax1.set_title("Ground Truth: Entity Features")
    ax1.set_xlabel("Feature ID"); ax1.set_ylabel("Entity ID")
    disc_ids = sorted(discovered.keys())
    if disc_ids:
        disc_map = np.zeros((len(disc_ids), ground_truth.num_features))
        for di, did in enumerate(disc_ids):
            for f in discovered[did].features:
                if f < ground_truth.num_features:
                    disc_map[di, f] = 1
        ax2.imshow(disc_map, aspect="auto", cmap="Oranges", interpolation="none")
        ax2.set_title("Discovered: Entity Features")
        ax2.set_xlabel("Feature ID"); ax2.set_ylabel("Discovered Entity Index")
    else:
        ax2.text(0.5, 0.5, "No entities discovered", ha="center", va="center", transform=ax2.transAxes)
    fig.suptitle("Entity Discovery: Ground Truth vs Incremental Memory", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_entity_comparison.png", dpi=150)
    plt.close(fig)


def _plot_relation_matrix(memory, ground_truth, metrics, output_dir, prefix, plt) -> None:
    gt_matrix = ground_truth.get_ground_truth_relation_strengths()
    n_gt = gt_matrix.shape[0]
    entity_matching = metrics.get("entity_matching", {})
    gt_to_disc = {v: k for k, v in entity_matching.items()} if entity_matching else {}
    pred_matrix = np.zeros((n_gt, n_gt))
    if gt_to_disc:
        scores = memory.get_relation_scores("jaccard")
        for i in range(n_gt):
            for j in range(i + 1, n_gt):
                if i in gt_to_disc and j in gt_to_disc:
                    di, dj = gt_to_disc[i], gt_to_disc[j]
                    key = (di, dj) if di < dj else (dj, di)
                    pred_matrix[i, j] = scores.get(key, 0.0)
                    pred_matrix[j, i] = pred_matrix[i, j]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axes[0].imshow(gt_matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    axes[0].set_title("Ground Truth Relations"); plt.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(pred_matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    axes[1].set_title("Predicted Relations (Jaccard)"); plt.colorbar(im1, ax=axes[1], shrink=0.8)
    diff = np.abs(gt_matrix - pred_matrix)
    im2 = axes[2].imshow(diff, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    axes[2].set_title("|GT - Pred| Difference"); plt.colorbar(im2, ax=axes[2], shrink=0.8)
    fig.suptitle("Relation Matrix Comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_relation_matrix.png", dpi=150)
    plt.close(fig)


def _plot_formation_curve(memory, ground_truth, output_dir, prefix, plt) -> None:
    gt_features = ground_truth.get_ground_truth_entity_features()
    num_gt = len(gt_features)
    discovered = memory.get_discovered_entities()
    history_len = len(memory.get_state_change_history())
    steps = np.linspace(0, history_len - 1, min(50, history_len), dtype=int)
    n_disc, match_jac = [], []
    for step in steps:
        n_d = len([e for e in discovered.values() if e.created_at <= step])
        n_disc.append(n_d)
        disc_at_step = {eid: ent for eid, ent in discovered.items() if ent.created_at <= step}
        _, mj = match_entities(disc_at_step, gt_features)
        match_jac.append(mj)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, n_disc, "b-", linewidth=2, label="Discovered entities")
    ax1.axhline(y=num_gt, color="gray", linestyle="--", label=f"GT entities ({num_gt})")
    ax1.set_xlabel("Token"); ax1.set_ylabel("Cumulative Entities")
    ax1.set_title("Entity Discovery Over Time"); ax1.legend(); ax1.set_ylim(bottom=0)
    ax2.plot(steps, match_jac, "g-", linewidth=2, label="Mean Jaccard (disc. vs GT)")
    ax2.set_xlabel("Token"); ax2.set_ylabel("Jaccard Score")
    ax2.set_title("Entity Matching Quality Over Time"); ax2.legend(); ax2.set_ylim(0, 1.05)
    fig.suptitle("Memory Formation Curves", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_formation_curve.png", dpi=150)
    plt.close(fig)


def _plot_state_change(memory, ground_truth, output_dir, prefix, plt) -> None:
    sc_history = memory.get_state_change_history()
    event_boundaries = ground_truth.get_event_boundaries()
    if not sc_history:
        return
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(np.arange(len(sc_history)), sc_history, "b-", linewidth=0.8, alpha=0.7)
    for t in sorted(event_boundaries):
        if t < len(sc_history):
            ax.axvline(x=t, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=0.45, color="orange", linestyle=":", linewidth=1, label="Threshold (0.45)")
    ax.set_xlabel("Token"); ax.set_ylabel("State Change Score")
    ax.set_title("State Change Detection vs Ground Truth Event Boundaries")
    ax.set_ylim(0, 1.05)
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="red", linestyle="--", alpha=0.5, label="GT Event Boundary"),
        Line2D([0], [0], color="blue", linewidth=1, alpha=0.7, label="State Change"),
        Line2D([0], [0], color="orange", linestyle=":", label="Threshold"),
    ]
    ax.legend(handles=custom_lines, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_state_change.png", dpi=150)
    plt.close(fig)


def _plot_control_comparison(metrics_correct, metrics_shuffled, metrics_empty, output_dir) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    conditions = ["Correct", "Shuffled", "Empty"]
    mae_vals = [metrics_correct.get("relation_mae", 0),
                metrics_shuffled.get("relation_mae", 0),
                metrics_empty.get("relation_mae", 0)]
    corr_vals = [metrics_correct.get("relation_corr", 0),
                 metrics_shuffled.get("relation_corr", 0),
                 metrics_empty.get("relation_corr", 0)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    bars1 = ax1.bar(conditions, mae_vals, color=colors, edgecolor="black")
    ax1.set_title("Relation MAE (lower is better)"); ax1.set_ylabel("Mean Absolute Error")
    for bar, val in zip(bars1, mae_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", fontsize=10)
    bars2 = ax2.bar(conditions, corr_vals, color=colors, edgecolor="black")
    ax2.set_title("Relation Correlation (higher is better)"); ax2.set_ylabel("Pearson r")
    ax2.set_ylim(-0.1, 1.1)
    for bar, val in zip(bars2, corr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", fontsize=10)
    fig.suptitle("Control Comparison: Correct vs Shuffled vs Empty Memory", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "memory_sim_control_comparison.png", dpi=150)
    plt.close(fig)


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory Entity Simulation — F51 Darwin-X")
    p.add_argument("--num-tokens", type=int, default=5000)
    p.add_argument("--num-entities", type=int, default=8)
    p.add_argument("--num-features", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="workspace/runtime/memory_sim")
    p.add_argument("--shuffle-memory", action="store_true")
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--cooccur-threshold", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    print("=" * 70)
    print("  F51 Darwin-X — Memory Entity Simulation")
    print("  Formacao de Memoria Episodica Entidade--Evento")
    print("=" * 70)
    print(f"  Tokens: {args.num_tokens} | Entidades: {args.num_entities} | Seed: {args.seed}")
    print(f"  Shuffle: {args.shuffle_memory}")
    print("=" * 70)

    print("\n[1/5] Gerando ground truth sintetico...")
    gt = GroundTruthGenerator(
        num_entities=args.num_entities, num_features=args.num_features, seed=args.seed)
    sequence = gt.generate_sequence(args.num_tokens)
    print(f"  Sequencia: {len(sequence)} tokens | Eventos: {len(gt.events)} | Fronteiras: {len(gt.get_event_boundaries())}")

    print("\n[2/5] Construindo memoria episodica...")
    memory = EpisodicMemory(
        window_size=args.window_size, cooccur_threshold=args.cooccur_threshold, num_features=args.num_features)
    for idx, features in enumerate(sequence):
        memory.process_token(features)
        if idx % 1000 == 0 and idx > 0:
            n_ent = len(memory.get_discovered_entities())
            print(f"  Token {idx}/{args.num_tokens} | Entidades: {n_ent}")

    discovered = memory.get_discovered_entities()
    relation_scores = memory.get_relation_scores("jaccard")
    print(f"  Final: {len(discovered)} entidades, {len(relation_scores)} relacoes")

    print("\n[3/5] Avaliando contra ground truth...")
    gt_features = gt.get_ground_truth_entity_features()
    gt_relations = gt.get_ground_truth_relation_strengths()
    event_boundaries = gt.get_event_boundaries()

    entity_matching, mean_jaccard = match_entities(discovered, gt_features)
    print(f"  Entity matching: {len(entity_matching)}/{len(gt_features)} matched | Mean Jaccard: {mean_jaccard:.4f}")

    rel_metrics = evaluate_relation_quality(relation_scores, gt_relations, entity_matching)
    print(f"  Relation MAE: {rel_metrics['relation_mae']:.4f} | Corr: {rel_metrics['relation_corr']:.4f}")

    event_metrics = evaluate_event_detection(memory.get_state_change_history(), event_boundaries)
    print(f"  Event Detection — P:{event_metrics['precision']:.3f} R:{event_metrics['recall']:.3f} F1:{event_metrics['f1']:.3f}")

    metrics_correct = {
        "entity_matching": {str(k): v for k, v in entity_matching.items()},
        "mean_jaccard": mean_jaccard, **rel_metrics, **event_metrics,
        "num_discovered_entities": len(discovered),
        "num_gt_entities": len(gt_features),
        "num_relations": len(relation_scores),
        "condition": "correct_memory",
    }

    metrics_shuffled = {}
    metrics_empty = {}
    if args.shuffle_memory:
        print("\n[3b/5] Controle embaralhado...")
        shuffled = memory.relation_bank.shuffle(seed=999)
        rel_shuf = evaluate_relation_quality(shuffled.get_relation_scores("jaccard"), gt_relations, entity_matching)
        metrics_shuffled = {"condition": "shuffled_memory", **rel_shuf, "num_relations": len(shuffled.relations)}
        print(f"  Shuffled MAE: {rel_shuf['relation_mae']:.4f} | Corr: {rel_shuf['relation_corr']:.4f}")
        print("\n[3c/5] Controle vazio...")
        rel_emp = evaluate_relation_quality({}, gt_relations, entity_matching)
        metrics_empty = {"condition": "empty_memory", **rel_emp, "num_relations": 0}
        print(f"  Empty MAE: {rel_emp['relation_mae']:.4f}")

    print("\n[4/5] Gerando visualizacoes...")
    plot_results(sequence, gt, memory, metrics_correct, output_dir)
    if args.shuffle_memory:
        _plot_control_comparison(metrics_correct, metrics_shuffled, metrics_empty, output_dir)

    print("\n[5/5] Salvando metricas...")
    report = {
        "config": {k: v for k, v in vars(args).items()},
        "correct_memory": {k: v for k, v in metrics_correct.items() if k != "entity_matching"},
        "entity_matching": metrics_correct.get("entity_matching", {}),
    }
    if args.shuffle_memory:
        report["shuffled_memory"] = metrics_shuffled
        report["empty_memory"] = metrics_empty
    report_path = output_dir / "memory_sim_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  SUMARIO")
    print("=" * 70)
    print(f"  Entidades GT: {len(gt_features)} | Descobertas: {len(discovered)} | Matched: {len(entity_matching)}")
    print(f"  Mean Jaccard:  {mean_jaccard:.4f}")
    print(f"  Relation MAE:  {rel_metrics['relation_mae']:.4f} | Corr: {rel_metrics['relation_corr']:.4f}")
    print(f"  Event F1:      {event_metrics['f1']:.3f}")
    if args.shuffle_memory:
        print(f"  Shuffled MAE:  {metrics_shuffled.get('relation_mae', float('nan')):.4f}")
        print(f"  Empty MAE:     {metrics_empty.get('relation_mae', float('nan')):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
