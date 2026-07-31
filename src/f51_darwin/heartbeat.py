"""
F51 HEARTBEAT — O Coracao Autonomo do Organismo.
5 inovacoes ligadas: Forward-Forward + Test-Time Memory +
Quiet-STaR + Self-Rewarding + Voyager.
TUDO LIGADO. DOPAMINA COMO COMBUSTIVEL. NUNCA PARA.
"""

from __future__ import annotations
import json, math, random, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import torch, torch.nn.functional as F
from torch import nn

# ═══════════════════════ 1. FORWARD-FORWARD ═══════════════════════

class ForwardForwardLayer(nn.Module):
    """Camada que aprende LOCALMENTE com sinal de dopamina."""
    def __init__(self, dim, threshold=2.0, lr=0.001):
        super().__init__()
        self.dim, self.threshold, self.lr = dim, threshold, lr
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.ones(dim) * 0.1)

    def goodness(self, x):
        return (x.pow(2).mean(dim=-1) - self.threshold).mean()

    def forward(self, x, dopamine=0.0):
        h = self.norm(x)
        if abs(dopamine) > 0.1:
            with torch.no_grad():
                g = self.goodness(h).item()
                delta = (2.0 - g) * dopamine * 0.1
                self.weight.data += self.lr * delta * torch.sign(self.weight.data)
                self.weight.data.clamp_(-1.0, 1.0)
        return h * self.weight


class ForwardForwardStack(nn.Module):
    """Stack de camadas Forward-Forward."""
    def __init__(self, dim, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([ForwardForwardLayer(dim) for _ in range(n_layers)])

    def forward(self, x, dopamine=0.0):
        for layer in self.layers:
            x = layer(x, dopamine)
        return x

    def total_goodness(self):
        layer = self.layers[0]
        with torch.no_grad():
            z = torch.zeros(1, 1, layer.dim, device=layer.weight.device, dtype=layer.weight.dtype)
            goodness = sum(l.goodness(l.norm(z)).detach() for l in self.layers) / len(self.layers)
        return float(goodness.cpu())


# ═══════════════════ 2. TEST-TIME MEMORY (Titans 2025) ═══════════════════════

@dataclass
class SurpriseMemorySlot:
    key: torch.Tensor; value: torch.Tensor; timestamp: str
    domain: str = ""; surprise_score: float = 0.0; access_count: int = 0


def bound_memory_residual(
    value: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Limit each memory vector to the norm of its hidden-state reference."""
    if value.shape != reference.shape:
        raise ValueError("memory value and reference must share shape")
    if not bool(torch.isfinite(value).all()) or not bool(
        torch.isfinite(reference).all()
    ):
        raise ValueError("memory value and reference must be finite")
    value_norm = value.float().norm(dim=-1, keepdim=True)
    reference_norm = reference.float().norm(dim=-1, keepdim=True)
    factor = (reference_norm / value_norm.clamp_min(eps)).clamp(max=1.0)
    return value * factor.to(device=value.device, dtype=value.dtype)


class TestTimeMemory(nn.Module):
    """Memoria que aprende DURANTE inferencia. Surpresa -> armazena."""
    def __init__(self, d_model, capacity=1024):
        super().__init__()
        self.d_model, self.capacity = d_model, capacity
        self.slots: list[SurpriseMemorySlot] = []
        self.proj_key = nn.Linear(d_model, d_model // 4)
        self.proj_value = nn.Linear(d_model, d_model)
        self.surprise_threshold = 0.3

    def _append_slot(self, key, value, *, jepa_error=0.0, domain=""):
        if len(self.slots) >= self.capacity:
            self.slots.sort(key=lambda s: s.access_count)
            self.slots.pop(0)
        self.slots.append(SurpriseMemorySlot(
            key=key.detach(), value=value.detach(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            domain=domain, surprise_score=jepa_error))

    def write_if_surprised(self, x, jepa_error=0.0, domain="", already_pooled=False):
        """already_pooled=True: x e ja [batch, d_model] (ex: hidden numa UNICA
        posicao marcada por saliencia), pula o mean(dim=1) sobre a sequencia
        inteira. Endereçamento por entidade (2026-07-27) -- ver
        ttm_entity_addressing em config.py. Default False preserva o
        comportamento pooled exato de antes."""
        if jepa_error < self.surprise_threshold:
            return False
        with torch.no_grad():
            src = x if already_pooled else x.mean(dim=1)
            k = self.proj_key(src); k = k.mean(dim=0) if k.ndim > 1 else k
            v = self.proj_value(src); v = v.mean(dim=0) if v.ndim > 1 else v
        self._append_slot(k, v, jepa_error=jepa_error, domain=domain)
        return True

    def write_association(
        self,
        key_source,
        value_source,
        *,
        jepa_error=0.0,
        domain="",
    ):
        """Store an aligned [1,D] key and an answer-bearing residual value."""
        if jepa_error < self.surprise_threshold:
            return False
        if key_source.ndim != 2 or value_source.ndim != 2:
            raise ValueError(
                "association sources must have shape [batch, d_model]"
            )
        if (
            tuple(key_source.shape) != (1, self.d_model)
            or tuple(value_source.shape) != (1, self.d_model)
        ):
            raise ValueError(
                "association sources must both have shape [1, d_model]"
            )
        if not bool(torch.isfinite(key_source).all()) or not bool(
            torch.isfinite(value_source).all()
        ):
            raise ValueError("association sources must be finite")
        with torch.no_grad():
            key = self.proj_key(key_source)[0]
            value = value_source[0]
        self._append_slot(
            key,
            value,
            jepa_error=jepa_error,
            domain=domain,
        )
        return True

    def retrieve(self, query, top_k=4, min_similarity=0.5, already_pooled=False):
        """Retrieve one independent memory residual per batch item.

        No batch reduction is permitted here: pooling queries across samples
        leaks one sample's retrieval decision into every other sample.

        Unlike ``write_if_surprised`` (an inherently non-differentiable
        Python-list cache write), this method is NOT wrapped in
        ``torch.no_grad()``. Stored ``slot.key``/``slot.value`` tensors are
        already ``.detach()``'d at write time, so they act as frozen
        constants here — but ``self.proj_key`` is applied to the *live*
        query, so its weights receive a real gradient from whatever
        downstream loss consumes the retrieved memory (e.g. the LM loss via
        the TTM residual). This is what lets ``proj_key`` actually learn a
        useful query space instead of staying at its random init forever.
        The caller already detaches ``query`` from the backbone
        (``model.py``'s ``hidden.detach()``), so no gradient leaks upstream.

        already_pooled=True: query is already [batch, d_model] (e.g. hidden
        at ONE saliency-marked position), skips the mean(dim=1) over the
        whole sequence -- entity addressing (2026-07-27, see
        ttm_entity_addressing in config.py). Result is then also returned
        as [batch, d_model] (no seq dim) so the caller can scatter it into
        the specific marked position instead of broadcasting it everywhere.
        """
        if not self.slots:
            return None
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if already_pooled:
            if query.ndim != 2:
                raise ValueError("query must have shape [batch, d_model] when already_pooled=True")
            q = self.proj_key(query)  # [batch, d_model//4]
        else:
            if query.ndim != 3:
                raise ValueError("query must have shape [batch, sequence, d_model]")
            q = self.proj_key(query.mean(dim=1))  # [batch, d_model//4]
        with torch.no_grad():
            compatible = [
                (index, slot)
                for index, slot in enumerate(self.slots)
                if tuple(slot.key.shape) == (q.shape[-1],)
                and tuple(slot.value.shape) == (self.d_model,)
            ]
        if not compatible:
            return None
        keys = torch.stack(
            [
                slot.key.to(device=q.device, dtype=q.dtype)
                for _, slot in compatible
            ]
        )
        values = torch.stack(
            [
                slot.value.to(device=query.device, dtype=query.dtype)
                for _, slot in compatible
            ]
        )
        similarities = F.cosine_similarity(
            q.unsqueeze(1),
            keys.unsqueeze(0),
            dim=-1,
        )
        count = min(int(top_k), len(compatible))
        top_similarities, top_indices = similarities.topk(count, dim=1)
        matched = top_similarities[:, 0] >= float(min_similarity)
        weights = top_similarities.clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        selected = values[top_indices]
        result = (selected * weights.unsqueeze(-1)).sum(dim=1)
        result = torch.where(
            matched.unsqueeze(-1),
            result,
            torch.zeros_like(result),
        )
        if not bool(matched.any().item()):
            return None
        with torch.no_grad():
            for batch_index in matched.nonzero(as_tuple=False).flatten().tolist():
                for local_index, weight in zip(
                    top_indices[batch_index].tolist(),
                    weights[batch_index].tolist(),
                ):
                    if weight > 0.0:
                        slot_index = compatible[local_index][0]
                        self.slots[slot_index].access_count += 1
        return result if already_pooled else result.unsqueeze(1)

    def stats(self):
        return {"slots_used": len(self.slots), "capacity": self.capacity,
                "avg_surprise": sum(s.surprise_score for s in self.slots) / max(1, len(self.slots))}


# ═══════════════════ 3. QUIET-STAR (Stanford 2024) ═══════════════════════

class QuietStarThinker(nn.Module):
    """Gera pensamentos internos antes de responder."""
    def __init__(self, d_model, num_thoughts=4, thought_len=8):
        super().__init__()
        self.d_model, self.num_thoughts, self.thought_len = d_model, num_thoughts, thought_len
        self.thought_start = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.thought_end = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.evaluator = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(),
            nn.Linear(d_model, 1), nn.Sigmoid())
        self.scores: list[float] = []

    def think(self, context):
        B, T, D = context.shape
        if B != 1:  # batch > 1: skip thoughts, just return dummy
            return torch.zeros(
                1,
                self.thought_len,
                D,
                device=context.device,
                dtype=context.dtype,
            )
        thoughts, scores = [], []
        ctx_pool = context.mean(dim=1)
        for _ in range(self.num_thoughts):
            noise = torch.randn(
                1,
                self.thought_len,
                D,
                device=context.device,
                dtype=context.dtype,
            ) * 0.01
            thought = self.thought_start.expand(1, self.thought_len, -1) + noise
            thoughts.append(thought)
            combined = torch.cat([ctx_pool.expand(1, -1), thought.mean(dim=1)], dim=-1)
            scores.append(self.evaluator(combined).item())
        self.scores.extend(scores)
        if len(self.scores) > 1000:
            self.scores = self.scores[-500:]
        return thoughts[scores.index(max(scores))]

    def stats(self):
        return {"avg_score": sum(self.scores) / max(1, len(self.scores)),
                "best_score": max(self.scores) if self.scores else 0,
                "total": len(self.scores)}


# ═══════════════════ 4. SELF-REWARDING EXPLORER (Voyager-style) ═══════════════════════

@dataclass
class ExplorationTarget:
    domain: str; difficulty: float = 0.5
    questions: list[str] = field(default_factory=list)
    explored: bool = False; discoveries: int = 0; errors: int = 0


class SelfRewardingExplorer(nn.Module):
    """Exploracao autonoma: escolhe, explora, verifica, recompensa."""
    def __init__(self, curiosity_drive=None, ghost_brain=None):
        super().__init__()
        self.curiosity, self.ghost_brain = curiosity_drive, ghost_brain
        self.explored: dict[str, ExplorationTarget] = {}
        self._history: list[dict] = []

    def register_domain(self, domain, difficulty=0.5):
        if domain not in self.explored:
            self.explored[domain] = ExplorationTarget(domain=domain, difficulty=difficulty)

    def choose_next(self):
        candidates = [d for d in self.explored.values() if not d.explored]
        if not candidates:
            return None
        if self.curiosity:
            for c in candidates:
                n = self.curiosity.state.novelty_scores.get(c.domain, 1.0)
                c.difficulty *= n
        candidates.sort(key=lambda c: -abs(c.difficulty - 0.5) + c.difficulty * 0.3)
        return candidates[-1].domain

    def explore(self, domain):
        if domain not in self.explored:
            self.register_domain(domain)
        t = self.explored[domain]
        questions = []
        if self.ghost_brain and self.ghost_brain.ghost:
            docs = self.ghost_brain.ghost.curiosity.select_next_batch(
                list(self.ghost_brain.ghost.documents.values()), 3)
            for doc in docs:
                qs = self.ghost_brain.ghost.questioner.generate_questions(doc, domain)
                questions.extend(qs)
        t.questions = questions[:5]
        t.explored = True
        self._history.append({"domain": domain, "questions": len(questions),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
        return t.questions

    def stats(self):
        return {"domains": len(self.explored),
                "explored": sum(1 for d in self.explored.values() if d.explored),
                "discoveries": sum(d.discoveries for d in self.explored.values())}


# ═══════════════════ 5. HEARTBEAT — O CORACAO ═══════════════════════

@dataclass
class HeartbeatConfig:
    think_interval: int = 1; explore_interval: int = 10
    memory_consolidate: int = 50; ff_learn_interval: int = 5
    self_reward_interval: int = 20; ff_layers: int = 3
    ff_threshold: float = 2.0; ff_lr: float = 0.001
    memory_capacity: int = 1024; surprise_threshold: float = 0.3
    num_thoughts: int = 4; thought_len: int = 8
    auto_register_domains: bool = True


class Heartbeat:
    """O coracao autonomo do organismo F51.

    EXPLORA (curiosity) -> PENSA (Quiet-STaR) -> LEMBRA (Titans) ->
    SENTE (dopamine) -> APRENDE (Forward-Forward) -> EVOLUI (lessons)
    """

    def __init__(self, d_model, config=None, *,
                 ghost_brain=None, curiosity=None, jepa=None, organism=None):
        self.d_model = d_model
        self.cfg = config or HeartbeatConfig()
        self.ff_stack = ForwardForwardStack(d_model, self.cfg.ff_layers)
        self.tt_memory = TestTimeMemory(d_model, self.cfg.memory_capacity)
        self.tt_memory.surprise_threshold = self.cfg.surprise_threshold
        self.thinker = QuietStarThinker(d_model, self.cfg.num_thoughts, self.cfg.thought_len)
        self.explorer = SelfRewardingExplorer(curiosity_drive=curiosity, ghost_brain=ghost_brain)
        self.ghost_brain = ghost_brain
        self.curiosity = curiosity
        self.jepa = jepa
        self.organism = organism
        self.total_beats = 0
        self.dopamine = 0.5
        self.memory_writes = 0
        self.explorations = 0
        self.thoughts_generated = 0
        self.beat_history: list[dict] = []
        # Dopamina tripla — estado persistente
        self._ema_loss: float | None = None
        self._ema_jepa: float | None = None
        self._ema_jepa_sq: float | None = None
        self._gmc_signal: float = 0.0  # injetado pelo training loop

    def to(self, device):
        """Move all PyTorch submodules to target device."""
        if hasattr(self, "ff_stack") and hasattr(self.ff_stack, "to"):
            self.ff_stack = self.ff_stack.to(device)
        if hasattr(self, "tt_memory") and hasattr(self.tt_memory, "to"):
            self.tt_memory = self.tt_memory.to(device)
        if hasattr(self, "thinker") and hasattr(self.thinker, "to"):
            self.thinker = self.thinker.to(device)
        return self

    def beat(
        self,
        x,
        *,
        domain="default",
        jepa_error=0.0,
        loss=0.0,
        expose_jepa_signal=False,
        memory_used=None,
    ):
        """Uma batida. Chamado a cada forward pass."""
        with torch.no_grad():
            x = x.detach()
            self.total_beats += 1
            stats = {}
            if expose_jepa_signal:
                stats["jepa_surprise"] = float(jepa_error)

            # Guard: heartbeat organs assume batch=1
            if x.size(0) != 1:
                return {
                    "beat": self.total_beats,
                    "batch_skipped": x.size(0),
                    **stats,
                }

            if self.total_beats % self.cfg.think_interval == 0:
                self.thinker.think(x); self.thoughts_generated += 1
                stats["thought"] = self.thinker.stats()

            # Retrieval evidence belongs to the memory state that existed at
            # the start of this beat. A surprise written below is available
            # only to future beats and must not count as same-step memory use.
            if memory_used is None:
                memory_used = self.tt_memory.retrieve(x) is not None
            else:
                memory_used = bool(memory_used)
            wrote = self.tt_memory.write_if_surprised(x, jepa_error, domain)
            if wrote: self.memory_writes += 1
            stats["memory"] = self.tt_memory.stats()
            stats["memory_used"] = memory_used

            # ── Dopamina tripla: delta relativo + GMC + surpresa JEPA ──
            # Sinal 1: Delta relativo (SOAR / GLOW)
            # Recompensa quando a loss MELHORA em relacao ao proprio passado,
            # nao quando atinge um threshold absoluto improvavel.
            loss_f = float(loss)
            if self._ema_loss is None:
                self._ema_loss = loss_f
            else:
                self._ema_loss = 0.995 * self._ema_loss + 0.005 * loss_f  # horizonte ~200 passos
            frac = (self._ema_loss - loss_f) / (self._ema_loss + 1e-8)
            delta_reward = 0.1 * math.tanh(frac * 5.0)  # continuo, ±0.1

            # Sinal 2: GMC (Gradient-Momentum Coupling)
            # Mede o alinhamento entre o gradiente atual e o momentum do
            # AdamW. Alto = progresso real. Baixo/negativo = ruido ou conflito.
            gmc = self._gmc_signal
            gmc_reward = 0.05 * gmc  # gmc in [-1, 1] → [-0.05, 0.05]

            # Sinal 3: Surpresa JEPA (Z-score dinâmico)
            # Concede bônus exploratório quando o erro do JEPA dá um pico estatístico
            # acima do desvio padrão recente (evita bônus travado em 0.0 na descida).
            jepa_f = float(jepa_error)
            if self._ema_jepa is None:
                self._ema_jepa = jepa_f
                self._ema_jepa_sq = jepa_f ** 2
            else:
                self._ema_jepa = 0.995 * self._ema_jepa + 0.005 * jepa_f
                self._ema_jepa_sq = 0.995 * self._ema_jepa_sq + 0.005 * (jepa_f ** 2)
            var_jepa = max(1e-6, self._ema_jepa_sq - (self._ema_jepa ** 2))
            std_jepa = math.sqrt(var_jepa)
            z_score = (jepa_f - self._ema_jepa) / (std_jepa + 1e-5)
            jepa_bonus = 0.03 * max(0.0, min(z_score / 2.0, 1.0))

            reward = delta_reward + gmc_reward + jepa_bonus
            self.dopamine = 0.9 * self.dopamine + reward
            self.dopamine = max(-1.0, min(1.0, self.dopamine))

            stats["dopamine_raw"] = round(self.dopamine, 4)
            stats["dopamine_delta"] = round(delta_reward, 4)
            stats["dopamine_gmc"] = round(gmc_reward, 4)
            stats["dopamine_jepa"] = round(jepa_bonus, 4)

            if self.total_beats % self.cfg.ff_learn_interval == 0 and abs(self.dopamine) > 0.1:
                self.ff_stack.forward(x, self.dopamine)
                stats["ff"] = {"goodness": self.ff_stack.total_goodness(), "dopamine": self.dopamine}

            if self.cfg.auto_register_domains and domain not in self.explorer.explored:
                self.explorer.register_domain(domain)

            if self.total_beats % self.cfg.self_reward_interval == 0:
                nd = self.explorer.choose_next()
                if nd:
                    qs = self.explorer.explore(nd); self.explorations += 1
                    stats["explore"] = {"domain": nd, "questions": len(qs)}

            record = {"beat": self.total_beats, "dopamine": round(self.dopamine, 3),
                      "mem_writes": self.memory_writes, "explorations": self.explorations,
                      "thoughts": self.thoughts_generated, **stats}
            self.beat_history.append(record)
            if len(self.beat_history) > 500: self.beat_history = self.beat_history[-200:]
            return record

    def inject_gmc(self, value: float) -> None:
        """Recebe sinal GMC do training loop apos backward.

        O GMC (Gradient-Momentum Coupling) mede o alinhamento entre o gradiente
        atual e o momentum do AdamW. Alto = progresso real, baixo/negativo =
        ruido ou conflito. Injete APOS backward e ANTES do optimizer.step().
        O valor sera consumido no PROXIMO beat(), evitando circularidade.
        """
        self._gmc_signal = float(value)

    def consolidate(self):
        recent = self.beat_history[-100:] if self.beat_history else [{"dopamine": 0.5}]
        result = {"total_beats": self.total_beats,
                  "dopamine_avg": sum(b.get("dopamine", 0) for b in recent) / max(1, len(recent)),
                  "memory": self.tt_memory.stats(), "thinker": self.thinker.stats(),
                  "explorer": self.explorer.stats(), "ff_goodness": self.ff_stack.total_goodness()}
        if self.ghost_brain:
            lessons = self.ghost_brain.consolidate_lessons()
            result["lessons_consolidated"] = len(lessons)
            result["ghost"] = self.ghost_brain.stats()
        return result

    @staticmethod
    def _module_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        """Return a checkpoint-safe CPU snapshot, detached from live tensors."""
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    def state_dict(self):
        return {
            "version": 3,
            "beat": self.total_beats,
            "dopamine": self.dopamine,
            "ff_goodness": self.ff_stack.total_goodness(),
            "memory_slots": len(self.tt_memory.slots),
            "domains": list(self.explorer.explored.keys()),
            "explorer_targets": {
                domain: {
                    "difficulty": target.difficulty,
                    "questions": list(target.questions),
                    "explored": target.explored,
                    "discoveries": target.discoveries,
                    "errors": target.errors,
                }
                for domain, target in self.explorer.explored.items()
            },
            "explorer_history": [
                dict(item) for item in self.explorer._history
            ],
            "memory_writes": self.memory_writes,
            "explorations": self.explorations,
            "thoughts_generated": self.thoughts_generated,
            "ff_stack_state_dict": self._module_state_dict(self.ff_stack),
            "tt_memory_state_dict": self._module_state_dict(self.tt_memory),
            "thinker_state_dict": self._module_state_dict(self.thinker),
            "tt_memory_surprise_threshold": self.tt_memory.surprise_threshold,
            "tt_memory_slots": [
                {
                    "key": slot.key.detach().cpu().clone(),
                    "value": slot.value.detach().cpu().clone(),
                    "timestamp": slot.timestamp,
                    "domain": slot.domain,
                    "surprise_score": slot.surprise_score,
                    "access_count": slot.access_count,
                }
                for slot in self.tt_memory.slots
            ],
            "thinker_scores": list(self.thinker.scores),
            "beat_history": list(self.beat_history),
        }

    def load_state_dict(self, state):
        ff_state = state.get("ff_stack_state_dict")
        if ff_state is not None:
            self.ff_stack.load_state_dict(ff_state)
        memory_state = state.get("tt_memory_state_dict")
        if memory_state is not None:
            self.tt_memory.load_state_dict(memory_state)
        thinker_state = state.get("thinker_state_dict")
        if thinker_state is not None:
            self.thinker.load_state_dict(thinker_state)

        self.total_beats = state.get("beat", 0)
        self.dopamine = state.get("dopamine", 0.5)
        self.memory_writes = state.get("memory_writes", 0)
        self.explorations = state.get("explorations", 0)
        self.thoughts_generated = state.get("thoughts_generated", 0)
        self.tt_memory.surprise_threshold = state.get(
            "tt_memory_surprise_threshold",
            self.tt_memory.surprise_threshold,
        )
        self.thinker.scores = list(state.get("thinker_scores", []))
        self.beat_history = list(state.get("beat_history", []))

        self.tt_memory.slots.clear()
        memory_parameter = next(self.tt_memory.parameters())
        expected_key_shape = (self.d_model // 4,)
        expected_value_shape = (self.d_model,)
        for raw_slot in state.get("tt_memory_slots", [])[-self.tt_memory.capacity:]:
            key = raw_slot["key"].detach().to(
                device=memory_parameter.device,
                dtype=memory_parameter.dtype,
            ).clone()
            value = raw_slot["value"].detach().to(
                device=memory_parameter.device,
                dtype=memory_parameter.dtype,
            ).clone()
            if tuple(key.shape) != expected_key_shape:
                raise ValueError(
                    f"invalid heartbeat memory key shape: {tuple(key.shape)}; "
                    f"expected {expected_key_shape}"
                )
            if tuple(value.shape) != expected_value_shape:
                raise ValueError(
                    f"invalid heartbeat memory value shape: {tuple(value.shape)}; "
                    f"expected {expected_value_shape}"
                )
            self.tt_memory.slots.append(
                SurpriseMemorySlot(
                    key=key,
                    value=value,
                    timestamp=str(raw_slot.get("timestamp", "")),
                    domain=str(raw_slot.get("domain", "")),
                    surprise_score=float(raw_slot.get("surprise_score", 0.0)),
                    access_count=int(raw_slot.get("access_count", 0)),
                )
            )

        self.explorer.explored.clear()
        raw_targets = state.get("explorer_targets")
        if isinstance(raw_targets, dict):
            for domain, raw_target in raw_targets.items():
                if not isinstance(raw_target, dict):
                    raise TypeError(
                        "heartbeat explorer target state must be a mapping"
                    )
                self.explorer.explored[str(domain)] = ExplorationTarget(
                    domain=str(domain),
                    difficulty=float(raw_target.get("difficulty", 0.5)),
                    questions=[
                        str(item)
                        for item in raw_target.get("questions", [])
                    ],
                    explored=bool(raw_target.get("explored", False)),
                    discoveries=int(raw_target.get("discoveries", 0)),
                    errors=int(raw_target.get("errors", 0)),
                )
        else:
            for domain in state.get("domains", []):
                self.explorer.register_domain(domain)
        self.explorer._history = [
            dict(item) for item in state.get("explorer_history", [])
        ]

    def to(self, *args, **kwargs):
        self.ff_stack.to(*args, **kwargs)
        self.tt_memory.to(*args, **kwargs)
        self.thinker.to(*args, **kwargs)
        memory_parameter = next(self.tt_memory.parameters())
        for slot in self.tt_memory.slots:
            slot.key = slot.key.to(
                device=memory_parameter.device,
                dtype=memory_parameter.dtype,
            )
            slot.value = slot.value.to(
                device=memory_parameter.device,
                dtype=memory_parameter.dtype,
            )
        return self
