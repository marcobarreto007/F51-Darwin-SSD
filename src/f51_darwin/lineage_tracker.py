"""F51 Lineage Tracker — Árvore genealógica dos módulos do organismo.

Registra nascimento, morte, parentesco e gerações de cada módulo.
Isso não é log. Isso é o registro fóssil do organismo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LineageEvent:
    timestamp: str
    event: str  # "born", "promoted", "quarantined", "ablated", "died", "merged", "frozen"
    module_id: str
    generation: int
    cycle: int
    score: float = 0.0
    parent_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "module_id": self.module_id,
            "generation": self.generation,
            "cycle": self.cycle,
            "score": self.score,
            "parent_ids": self.parent_ids,
            "details": self.details,
        }


class LineageTracker:
    """Rastreia a árvore genealógica completa do organismo."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[LineageEvent] = []
        self._module_generations: dict[str, int] = {}
        self._module_parents: dict[str, list[str]] = {}
        self._current_generation: int = 0
        self._total_births: int = 0
        self._total_deaths: int = 0
        self._survivors: set[str] = set()
        self._load()

    def _load(self) -> None:
        path = self.output_dir / "lineage.jsonl"
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(raw_line)
                event = LineageEvent(
                    timestamp=str(payload["timestamp"]),
                    event=str(payload["event"]),
                    module_id=str(payload["module_id"]),
                    generation=int(payload.get("generation", 0)),
                    cycle=int(payload.get("cycle", 0)),
                    score=float(payload.get("score", 0.0)),
                    parent_ids=list(payload.get("parent_ids", [])),
                    details=dict(payload.get("details", {})),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self.events.append(event)
            self._current_generation = max(self._current_generation, event.generation)
            if event.event == "born":
                is_new_module = event.module_id not in self._module_generations
                self._module_generations.setdefault(event.module_id, event.generation)
                self._module_parents.setdefault(event.module_id, list(event.parent_ids))
                if is_new_module:
                    self._total_births += 1
                self._survivors.add(event.module_id)
            elif event.event == "died":
                self._total_deaths += 1
                self._survivors.discard(event.module_id)
            elif event.event == "promoted" and event.details.get("to") in {"active", "GPU"}:
                self._survivors.add(event.module_id)

    def has_module(self, module_id: str) -> bool:
        return module_id in self._module_generations

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def new_generation(self) -> int:
        """Avança para a próxima geração. Retorna o número da geração."""
        self._current_generation += 1
        return self._current_generation

    # ═══════════════════════════════════════════════════════════
    # EVENTOS DE VIDA
    # ═══════════════════════════════════════════════════════════

    def record_birth(
        self,
        module_id: str,
        cycle: int,
        *,
        parent_ids: list[str] | None = None,
        score: float = 0.0,
        d_model: int = 0,
        mlp_ratio: int = 2,
    ) -> LineageEvent:
        """Registra o nascimento de um módulo."""
        parents = parent_ids or []
        # Herda a geração dos pais + 1, ou começa na 0
        if parents and parents[0] in self._module_generations:
            gen = self._module_generations[parents[0]] + 1
        else:
            gen = self._current_generation

        self._module_generations[module_id] = gen
        self._module_parents[module_id] = list(parents)
        self._total_births += 1
        self._survivors.add(module_id)

        event = LineageEvent(
            timestamp=self._now(),
            event="born",
            module_id=module_id,
            generation=gen,
            cycle=cycle,
            score=score,
            parent_ids=list(parents),
            details={"d_model": d_model, "mlp_ratio": mlp_ratio},
        )
        self.events.append(event)
        self._save()
        return event

    def record_promotion(
        self, module_id: str, cycle: int, *, from_state: str, to_state: str, score: float = 0.0
    ) -> LineageEvent:
        if to_state in {"active", "GPU"}:
            self._survivors.add(module_id)
        event = LineageEvent(
            timestamp=self._now(),
            event="promoted",
            module_id=module_id,
            generation=self._module_generations.get(module_id, -1),
            cycle=cycle,
            score=score,
            details={"from": from_state, "to": to_state},
        )
        self.events.append(event)
        self._save()
        return event

    def record_death(
        self,
        module_id: str,
        cycle: int,
        *,
        cause: str = "ablation",
        score: float = 0.0,
        lifespan_cycles: int = 0,
    ) -> LineageEvent:
        """Registra a morte de um módulo."""
        self._total_deaths += 1
        self._survivors.discard(module_id)

        event = LineageEvent(
            timestamp=self._now(),
            event="died",
            module_id=module_id,
            generation=self._module_generations.get(module_id, -1),
            cycle=cycle,
            score=score,
            details={"cause": cause, "lifespan_cycles": lifespan_cycles},
        )
        self.events.append(event)
        self._save()
        return event

    def record_quarantine(self, module_id: str, cycle: int, *, score: float = 0.0) -> LineageEvent:
        event = LineageEvent(
            timestamp=self._now(),
            event="quarantined",
            module_id=module_id,
            generation=self._module_generations.get(module_id, -1),
            cycle=cycle,
            score=score,
        )
        self.events.append(event)
        self._save()
        return event

    def record_ablation(
        self, module_id: str, cycle: int, *, baseline: float, ablated: float, decision: str
    ) -> LineageEvent:
        event = LineageEvent(
            timestamp=self._now(),
            event="ablated",
            module_id=module_id,
            generation=self._module_generations.get(module_id, -1),
            cycle=cycle,
            details={"baseline_score": baseline, "ablated_score": ablated, "decision": decision},
        )
        self.events.append(event)
        self._save()
        return event

    # ═══════════════════════════════════════════════════════════
    # CONSULTAS
    # ═══════════════════════════════════════════════════════════

    def family_tree(self, module_id: str, depth: int = 0) -> dict[str, Any]:
        """Retorna a árvore genealógica de um módulo."""
        parents = self._module_parents.get(module_id, [])
        events_of_module = [e for e in self.events if e.module_id == module_id]

        return {
            "module_id": module_id,
            "generation": self._module_generations.get(module_id, -1),
            "parents": parents,
            "children": [
                mid for mid, pids in self._module_parents.items()
                if module_id in pids
            ],
            "events": [e.to_dict() for e in events_of_module[-10:]],
            "alive": module_id in self._survivors,
        }

    def population_stats(self) -> dict[str, Any]:
        """Estatísticas da população atual."""
        alive = len(self._survivors)
        dead = self._total_deaths
        total = self._total_births

        generations: dict[int, int] = {}
        for mid, gen in self._module_generations.items():
            if mid in self._survivors:
                generations[gen] = generations.get(gen, 0) + 1

        return {
            "total_born": total,
            "total_died": dead,
            "alive": alive,
            "extinction_rate": round(dead / max(total, 1), 3),
            "survival_rate": round(alive / max(total, 1), 3),
            "current_generation": self._current_generation,
            "generations": dict(sorted(generations.items())),
            "oldest_survivor_gen": min(generations.keys()) if generations else 0,
            "newest_survivor_gen": max(generations.keys()) if generations else 0,
        }

    def timeline(self, last_n: int = 20) -> list[dict[str, Any]]:
        """Últimos N eventos da linha do tempo."""
        return [e.to_dict() for e in self.events[-last_n:]]

    def full_lineage_report(self) -> dict[str, Any]:
        """Relatório completo da linhagem."""
        survivors = []
        for mid in sorted(self._survivors):
            survivors.append(self.family_tree(mid))

        dead_modules = []
        for mid in sorted(set(self._module_generations.keys()) - self._survivors):
            dead_modules.append({
                "module_id": mid,
                "generation": self._module_generations.get(mid, -1),
                "death_event": next(
                    (e.to_dict() for e in reversed(self.events)
                     if e.module_id == mid and e.event == "died"),
                    None,
                ),
            })

        return {
            "organism": "F51-Darwin-SSD",
            "total_events": len(self.events),
            "population": self.population_stats(),
            "survivors": survivors,
            "dead": dead_modules[-20:],  # last 20 dead
            "recent_timeline": self.timeline(30),
        }

    # ═══════════════════════════════════════════════════════════
    # PERSISTÊNCIA
    # ═══════════════════════════════════════════════════════════

    def _save(self) -> Path:
        path = self.output_dir / "lineage.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.events[-1].to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def save_report(self) -> Path:
        path = self.output_dir / "lineage_report.json"
        path.write_text(
            json.dumps(self.full_lineage_report(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def print_tree(self, module_id: str | None = None, indent: int = 0) -> str:
        """Imprime a árvore genealógica como texto."""
        lines: list[str] = []

        if module_id is None:
            # Começa pelos módulos sem pais (geração 0 — Adão e Eva)
            roots = [mid for mid, pids in self._module_parents.items() if not pids]
            for root in sorted(roots):
                lines.append(self._print_subtree(root, indent=0))
        else:
            lines.append(self._print_subtree(module_id, indent=indent))

        return "\n".join(lines)

    def _print_subtree(self, module_id: str, indent: int) -> str:
        gen = self._module_generations.get(module_id, "?")
        alive = "🟢" if module_id in self._survivors else "☠️"
        prefix = "  " * indent

        # Find the birth and last event
        module_events = [e for e in self.events if e.module_id == module_id]
        birth = module_events[0] if module_events else None
        last = module_events[-1] if module_events else None

        line = f"{prefix}{alive} {module_id} (gen {gen})"
        if birth:
            line += f" — born cycle {birth.cycle}"
        if last and last.event == "died":
            line += f", died cycle {last.cycle} [{last.details.get('cause', '?')}]"

        lines = [line]

        children = [mid for mid, pids in self._module_parents.items() if module_id in pids]
        for child in sorted(children):
            lines.append(self._print_subtree(child, indent + 1))

        return "\n".join(lines)
