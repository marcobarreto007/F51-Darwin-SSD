"""
F51 Legacy Layers — 7 Camadas de Memória Infinita

O modelo nunca esquece. Só arquiva o que não usa e transforma
erros em inibidores (cicatrizes).

Arquitetura:
    1. GPU (Quente)    — 70% ativos, auto-treino contínuo
    2. RAM (Morno)     — 20% standby, prontos para subir
    3. SSD (Frio)      — 7% arquivado, domínios raros
    4. Deep (Gelo)     — 2% emergência, sobrevivência only
    5. Ashes (Cinzas) — 0.9% mortos, mas mantidos
    6. Scar (Cicatriz) — 0.09% inibidores, "NUNCA mais"
    7. DNA (Núcleo)    — 0.01% imutável, semente original

Doutrina:
    "O que funciona fica ativo."
    "O que falha vira inibidor, mas nunca é deletado."
    "Memória infinita = camadas profundas nunca morrem."
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import torch


class LayerTier(Enum):
    """As 7 camadas da memória neural."""
    GPU = 1      # 🔥 Quente — 70% ativos
    RAM = 2      # 🌤️ Morno — 20% standby
    SSD = 3      # ❄️ Frio — 7% arquivado
    DEEP = 4     # 🧊 Gelo — 2% emergência
    ASHES = 5    # 🪦 Cinzas — 0.9% mortos
    SCAR = 6     # 🩹 Cicatriz — 0.09% inibidores
    DNA = 7      # 🧬 Núcleo — 0.01% imutável


@dataclass
class NeuralGrave:
    """Um neurônio/módulo que morreu mas virou memória."""
    expert_id: str
    death_cycle: int
    death_cause: str  # "ablation", "never_fired", "always_wrong"
    original_layer: LayerTier
    score_at_death: float
    error_patterns: list[str] = field(default_factory=list)
    danger_detections: int = 0

    def to_inhibitor(self) -> Inhibitor:
        """Transforma um morto em inibidor (cicatriz)."""
        return Inhibitor(
            expert_id=self.expert_id,
            birth_cycle=self.death_cycle,
            blocked_patterns=self.error_patterns,
            danger_score=self.danger_detections / 100.0,
            reason=f"Died by {self.death_cause} — never fire this path again",
        )


@dataclass
class Inhibitor:
    """Um inibidor permanente — marca caminhos que NUNCA devem ser tomados."""
    expert_id: str
    birth_cycle: int
    blocked_patterns: list[str]
    danger_score: float  # 0.0 a 1.0
    reason: str
    activation_count: int = 0  # quantas vezes bloqueou um disparo

    def should_block(self, context: str) -> bool:
        """Verifica se deve bloquear este contexto."""
        for pattern in self.blocked_patterns:
            if pattern.lower() in context.lower():
                self.activation_count += 1
                return True
        return False

    def to_dict(self) -> dict:
        return {
            'expert_id': self.expert_id,
            'birth_cycle': self.birth_cycle,
            'blocked_patterns': self.blocked_patterns,
            'danger_score': self.danger_score,
            'reason': self.reason,
            'activation_count': self.activation_count,
        }


@dataclass
class DNALayer:
    """O núcleo imutável — semente que nunca muda."""
    seed_architecture: dict = field(default_factory=dict)
    immutable_params: dict[str, float] = field(default_factory=dict)
    tokenizer_config: dict = field(default_factory=dict)
    birth_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            'seed_architecture': self.seed_architecture,
            'immutable_params': self.immutable_params,
            'tokenizer_config': self.tokenizer_config,
            'birth': self.birth_timestamp,
        }


@dataclass
class LayerState:
    """Estado de uma camada de memória."""
    tier: LayerTier
    count: int = 0
    total_params: int = 0
    experts: set[str] = field(default_factory=set)
    last_access: str = ""
    activation_rate: float = 0.0  # proporção de tempo ativo


class LegacyLayers:
    """Gerenciador das 7 camadas de memória infinita.

    O modelo decide SOZINHO qual camada cada expert pertence.
    Baseado em temperatura (Ghost/JEPA/Spider/Consensus/Curiosity).
    """

    # Distribuição alvo de parâmetros
    TARGET_RATIOS = {
        LayerTier.GPU:   0.70,  # 70% ativos
        LayerTier.RAM:   0.20,  # 20% standby
        LayerTier.SSD:   0.07,  # 7% arquivado
        LayerTier.DEEP:  0.02,  # 2% emergência
        LayerTier.ASHES: 0.009, # 0.9% mortos
        LayerTier.SCAR:  0.0009, # 0.09% inibidores
        LayerTier.DNA:   0.0001, # 0.01% núcleo
    }

    def __init__(self, save_path: str | Path = "workspace/runtime/organism/legacy"):
        self.save_dir = Path(save_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / "legacy_state.json"  # Arquivo, não diretório

        # Camadas
        self.layers: dict[LayerTier, LayerState] = {
            tier: LayerState(tier=tier)
            for tier in LayerTier
        }

        # Cemitério (Ashes) + Cicatrizes (Scar)
        self.graves: dict[str, NeuralGrave] = {}
        self.inhibitors: dict[str, Inhibitor] = {}

        # DNA (núcleo imutável)
        self.dna: DNALayer | None = None

        # Histórico
        self.cycle: int = 0
        self.temperature_log: dict[str, list[float]] = {}

        self._load()

    def _load(self):
        """Carrega estado do disco."""
        if self.save_path.exists():
            try:
                data = json.loads(self.save_path.read_text())
                self.cycle = data.get('cycle', 0)

                # Recarrega camadas
                for tier_name, tier_data in data.get('layers', {}).items():
                    tier = LayerTier[tier_name]
                    self.layers[tier] = LayerState(
                        tier=tier,
                        count=tier_data.get('count', 0),
                        total_params=tier_data.get('total_params', 0),
                        experts=set(tier_data.get('experts', [])),
                    )

                # Recarrega cemitério
                for exp_id, grave_data in data.get('graves', {}).items():
                    grave_payload = dict(grave_data)
                    original_layer = grave_payload.get('original_layer', LayerTier.GPU)
                    if isinstance(original_layer, str):
                        original_layer = LayerTier[original_layer]
                    grave_payload['original_layer'] = original_layer
                    self.graves[exp_id] = NeuralGrave(**grave_payload)

                # Recarrega inibidores
                for exp_id, inh_data in data.get('inhibitors', {}).items():
                    self.inhibitors[exp_id] = Inhibitor(**inh_data)

                # Recarrega DNA
                dna_data = data.get('dna')
                if dna_data:
                    self.dna = DNALayer(
                        seed_architecture=dna_data.get('seed_architecture', {}),
                        immutable_params=dna_data.get('immutable_params', {}),
                        tokenizer_config=dna_data.get('tokenizer_config', {}),
                        birth_timestamp=dna_data.get('birth_timestamp', dna_data.get('birth', '')),
                    )

                self._normalize_layer_membership()

            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid LegacyLayers state: {self.save_path}") from exc

    def _normalize_layer_membership(self):
        """Repair stale counts and keep each expert in exactly one tier."""
        seen: set[str] = set()
        for tier in LayerTier:
            state = self.layers[tier]
            if tier == LayerTier.DNA:
                state.experts.clear()
                state.count = 1 if self.dna is not None else 0
                continue
            state.experts.difference_update(seen)
            state.count = len(state.experts)
            seen.update(state.experts)

    def save(self):
        """Salva estado no disco."""
        self._normalize_layer_membership()
        state = {
            'cycle': self.cycle,
            'layers': {
                tier.name: {
                    'count': state.count,
                    'total_params': state.total_params,
                    'experts': list(state.experts),
                }
                for tier, state in self.layers.items()
            },
            'graves': {
                exp_id: {
                    'expert_id': g.expert_id,
                    'death_cycle': g.death_cycle,
                    'death_cause': g.death_cause,
                    'original_layer': g.original_layer.name,
                    'score_at_death': g.score_at_death,
                    'error_patterns': g.error_patterns,
                    'danger_detections': g.danger_detections,
                }
                for exp_id, g in self.graves.items()
            },
            'inhibitors': {
                exp_id: inh.to_dict()
                for exp_id, inh in self.inhibitors.items()
            },
            'dna': self.dna.to_dict() if self.dna else None,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        temporary_path = self.save_path.with_name(
            f".{self.save_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            for attempt in range(6):
                try:
                    os.replace(temporary_path, self.save_path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            temporary_path.unlink(missing_ok=True)

    def set_dna(self, architecture: dict, tokenizer_config: dict):
        """Define o núcleo imutável (DNA). Só pode ser definido UMA vez."""
        if self.dna is not None:
            return  # DNA já foi definido, não muda

        self.dna = DNALayer(
            seed_architecture=architecture,
            tokenizer_config=tokenizer_config,
            birth_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.layers[LayerTier.DNA].count = 1
        self.save()

    def compute_tier(self, temperature: float, expert_id: str) -> LayerTier:
        """Decide qual camada um expert pertence baseado em temperatura.

        Temperatura vem dos 5 preditores:
            ghost × 3 + jepa × 2 + spider × 4 + consensus × 2 + curiosity × 1

        Escala: 0 a 12
        """
        # Log de temperatura para análise
        if expert_id not in self.temperature_log:
            self.temperature_log[expert_id] = []
        self.temperature_log[expert_id].append(temperature)

        # Verifica inibidores PRIMEIRO
        if expert_id in self.inhibitors:
            # Já é cicatriz — mantém no SCAR
            return LayerTier.SCAR

        # Verifica se já morreu
        if expert_id in self.graves:
            # Está no cemitério — pode promover a SCAR
            grave = self.graves[expert_id]
            if self._should_promote_to_scar(grave):
                inhibitor = grave.to_inhibitor()
                self.inhibitors[expert_id] = inhibitor
                del self.graves[expert_id]  # Remove de Ashes
                self.layers[LayerTier.ASHES].experts.discard(expert_id)
                self.layers[LayerTier.SCAR].experts.add(expert_id)
                self.layers[LayerTier.SCAR].count += 1
                self.save()
            return LayerTier.ASHES

        # Classifica por temperatura
        if temperature > 5.0:
            return LayerTier.GPU   # 🔥 Quente
        elif temperature > 2.0:
            return LayerTier.RAM   # 🌤️ Morno
        elif temperature > 0.5:
            return LayerTier.SSD   # ❄️ Frio
        elif temperature > 0.0:
            return LayerTier.DEEP  # 🧊 Gelo
        else:
            return LayerTier.ASHES # 🪦 Morto

    def _should_promote_to_scar(self, grave: NeuralGrave) -> bool:
        """Decide se um morto vira cicatriz (inibidor)."""
        # Só vira inibidor se:
        # 1. Morreu por erro (não por desuso)
        # 2. Tem padrões de erro claros
        # 3. Detectou perigo antes
        return (
            grave.death_cause in ("always_wrong", "danger_detected")
            and len(grave.error_patterns) > 0
            and grave.danger_detections > 0
        )

    # GATE 0.3: Referencia ao ExpertPool e modelo para acoes causais
    _expert_pool = None
    _model = None

    def set_causal_refs(self, expert_pool, model) -> None:
        """Conecta Legacy Layers ao ExpertPool e DarwinXModel para acoes reais."""
        self._expert_pool = expert_pool
        self._model = model

    # GATE 0.3: Mapeamento Tier → Acoes Reais
    TIER_ACTIONS = {
        LayerTier.GPU:   {"grad": True,  "bias": 0.0},
        LayerTier.RAM:   {"grad": True,  "bias": 0.0},
        LayerTier.SSD:   {"grad": False, "bias": 0.0},
        LayerTier.DEEP:  {"grad": False, "bias": -0.5},
        LayerTier.ASHES: {"grad": False, "bias": -5.0},
        LayerTier.SCAR:  {"grad": False, "bias": float("-inf")},
        LayerTier.DNA:   {"grad": False, "bias": 0.0},
    }

    def _apply_tier_actions(self, expert_id: str, tier: LayerTier) -> None:
        """GATE 0.3: Aplica acoes reais no modelo baseado no tier."""
        if self._expert_pool is None:
            return
        actions = self.TIER_ACTIONS.get(tier, {"grad": True, "bias": 0.0})
        record = self._expert_pool.records.get(expert_id)
        if record is None:
            return
        # Freeze ou activate
        if actions["grad"]:
            self._expert_pool.activate_expert(expert_id)
        else:
            self._expert_pool.freeze_expert(expert_id)
        # Router bias
        if actions["bias"] == float("-inf"):
            self._expert_pool.mask_expert(expert_id, -10.0)
        elif actions["bias"] != 0.0:
            self._expert_pool.mask_expert(expert_id, actions["bias"])
        else:
            self._expert_pool.unmask_expert(expert_id)
        # Propaga bias pro FineRouter do bloco correspondente
        if self._model is not None and record.block_index >= 0:
            block = self._model.blocks[record.block_index]
            if hasattr(block, 'moe') and hasattr(block.moe, 'fine_router'):
                bias_val = -10.0 if actions["bias"] == float("-inf") else actions["bias"]
                with torch.no_grad():
                    block.moe.fine_router.external_bias[record.expert_index] = bias_val

    def update_expert_tier(self, expert_id: str, old_tier: LayerTier | None, new_tier: LayerTier):
        """Move um expert entre camadas. GATE 0.3: com acoes causais reais."""
        # The persisted old_tier can be stale. Remove the id from every
        # mutable tier first so repeated bootstrap calls remain idempotent.
        for tier, state in self.layers.items():
            if tier != LayerTier.DNA and tier != new_tier:
                state.experts.discard(expert_id)

        if new_tier != LayerTier.DNA:
            self.layers[new_tier].experts.add(expert_id)
            self.layers[new_tier].last_access = datetime.now(timezone.utc).isoformat()

        self._normalize_layer_membership()
        # GATE 0.3: Aplica acoes reais no modelo
        self._apply_tier_actions(expert_id, new_tier)
        self.save()

    def expert_tier(self, expert_id: str) -> LayerTier | None:
        for tier, state in self.layers.items():
            if expert_id in state.experts:
                return tier
        return None

    def resurrect_expert(self, expert_id: str) -> None:
        """Return an archived real expert to the trainable GPU tier."""
        old_tier = self.expert_tier(expert_id)
        self.graves.pop(expert_id, None)
        self.update_expert_tier(expert_id, old_tier, LayerTier.GPU)

    def bury_expert(self, expert_id: str, death_cause: str, score: float, error_patterns: list[str]):
        """Mata um expert mas mantém no cemitério (Ashes)."""
        # Encontra a camada atual
        current_tier = None
        for tier, state in self.layers.items():
            if expert_id in state.experts:
                current_tier = tier
                break

        # Cria a sepultura
        grave = NeuralGrave(
            expert_id=expert_id,
            death_cycle=self.cycle,
            death_cause=death_cause,
            original_layer=current_tier or LayerTier.GPU,
            score_at_death=score,
            error_patterns=error_patterns,
        )
        self.graves[expert_id] = grave

        # Move para Ashes
        self.update_expert_tier(expert_id, current_tier, LayerTier.ASHES)
        self.save()

    def check_inhibitors(self, context: str) -> list[str]:
        """Verifica se algum inibidor deve bloquear este contexto."""
        blocked = []
        for expert_id, inhibitor in self.inhibitors.items():
            if inhibitor.should_block(context):
                blocked.append(expert_id)
                blocked.append(inhibitor.reason)
        return blocked

    def status(self) -> dict:
        """Relatório completo das 7 camadas."""
        total = sum(s.count for s in self.layers.values())
        return {
            'cycle': self.cycle,
            'total_experts': total,
            'layers': {
                tier.name: {
                    'count': state.count,
                    'percentage': round(state.count / total * 100, 2) if total > 0 else 0,
                    'target': round(self.TARGET_RATIOS[tier] * 100, 2),
                    'experts': list(state.experts)[:5],  # Primeiros 5
                }
                for tier, state in self.layers.items()
            },
            'graves': len(self.graves),
            'inhibitors': len(self.inhibitors),
            'dna_defined': self.dna is not None,
        }

    def visualize(self) -> str:
        """Visualização ASCII das 7 camadas."""
        status = self.status()
        lines = [
            "\n" + "=" * 60,
            "  🧬 F51 LEGACY LAYERS — Memória Infinita",
            "=" * 60,
        ]

        for tier in LayerTier:
            tier_name = tier.name
            tier_status = status['layers'][tier_name]
            count = tier_status['count']
            percentage = tier_status['percentage']
            target = tier_status['target']

            # Emoji baseado na camada
            emojis = {
                'GPU': '🔥',
                'RAM': '🌤️',
                'SSD': '❄️',
                'DEEP': '🧊',
                'ASHES': '🪦',
                'SCAR': '🩹',
                'DNA': '🧬',
            }
            emoji = emojis.get(tier_name, '•')

            # Barra de progresso
            bar_len = int(percentage / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)

            lines.append(
                f"  {emoji} {tier_name:<6s} │ [{bar}] {percentage:>5.1f}% "
                f"(target: {target:>5.1f}%) │ {count:>6} experts"
            )

        lines.extend([
            f"  🪦 Graves (cemitério): {status['graves']}",
            f"  🩹 Inhibitors (cicatrizes): {status['inhibitors']}",
            f"  🧬 DNA definido: {'SIM' if status['dna_defined'] else 'NÃO'}",
            "=" * 60,
        ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# INTEGRAÇÃO COM ORGANISMO
# ═══════════════════════════════════════════════════════════

@dataclass
class LegacyConfig:
    """Configuração do sistema de legado."""
    save_path: str = "workspace/runtime/organism/legacy"
    enable_inhibitors: bool = True
    auto_bury_threshold: float = 0.0  # temperatura < 0 = morre
    scar_promotion_cycles: int = 3  # ciclos antes de promover a cicatriz


class LegacyManager:
    """Manager que conecta LegacyLayers com o resto do organismo."""

    def __init__(self, config: LegacyConfig | None = None):
        self.config = config or LegacyConfig()
        self.legacy = LegacyLayers(self.config.save_path)

    def on_expert_scored(self, expert_id: str, temperature: float, score: float):
        """Chamado quando um expert recebe sua temperatura/score."""
        # Encontra camada atual
        current_tier = None
        for tier, state in self.legacy.layers.items():
            if expert_id in state.experts:
                current_tier = tier
                break

        # Calcula nova camada
        new_tier = self.legacy.compute_tier(temperature, expert_id)

        # Se mudou, atualiza
        if new_tier != current_tier:
            self.legacy.update_expert_tier(expert_id, current_tier, new_tier)

        # Se morreu, enterra
        if new_tier == LayerTier.ASHES and current_tier != LayerTier.ASHES:
            self.legacy.bury_expert(
                expert_id,
                death_cause="low_temperature",
                score=score,
                error_patterns=[],
            )

    def on_cycle_complete(self):
        """Chamado ao fim de cada ciclo."""
        self.legacy.cycle += 1
        self.legacy.save()

    def status_dashboard(self) -> str:
        """Painel de status do legado."""
        return self.legacy.visualize()
