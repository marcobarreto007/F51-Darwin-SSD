"""
Organ Senate — dynamic loss weight redistribution based on on-chain reputation.

The Senate convenes at cycle boundaries, reads organ reputations from the
blockchain, and redistributes loss weights. Decisions are recorded as
special blocks (Proof-of-Learning) in the blockchain.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SenateDecision:
    """One senate decision recorded as a Proof-of-Learning block."""
    cycle: int
    timestamp_unix_ms: int
    previous_weights: dict[str, float]
    new_weights: dict[str, float]
    reputations: dict[str, float]          # organ_name -> reputation_score
    promotions: list[str] = field(default_factory=list)
    demotions: list[str] = field(default_factory=list)
    deaths: list[str] = field(default_factory=list)
    newborns: list[str] = field(default_factory=list)
    reasoning: str = ""
    senate_block_hash: str = ""


class OrganSenate:
    """
    Reads blockchain reputations and redistributes loss weights.

    Policies:
    - Every active organ gets a minimum weight (senate_min_weight, default 0.02)
    - Reputation below death_threshold (0.05): weight = 0.01 (near death)
    - Reputation 0.05-0.3: weight gradually decreases
    - Reputation 0.3-0.7: weight stable
    - Reputation > 0.7: weight increases with "rising star" bonus
    - Newborn: initial weight 0.05, 3-cycle trial period
    - Dead: weight = 0, gate closes, proposed for pruning
    - Total auxiliary loss weight is normalized to not exceed 1.5
    """

    # Organ -> loss weight key mapping
    ORGAN_WEIGHT_KEYS = {
        "jepa": "jepa_weight",
        "mtp": "mtp_weight",
        "ghost": "ghost_weight",
        "spider_sense": "spider_calibration_weight",
        "spider_calibration": "spider_calibration_weight",
        "ttm_residual": "ttm_associative_weight",
        "gaba": "gaba_weight",
        "inter_hemispheric": "inter_hemispheric_weight",
        "curiosity": "curiosity_weight",
    }

    # Default weights from 100M config
    DEFAULT_WEIGHTS = {
        "jepa_weight": 0.4,
        "mtp_weight": 0.15,
        "ghost_weight": 0.12,
        "spider_calibration_weight": 0.10,
        "ttm_associative_weight": 0.02,
        "gaba_weight": 0.05,
        "inter_hemispheric_weight": 0.05,
        "curiosity_weight": 0.06,
    }

    def __init__(
        self,
        min_weight: float = 0.02,
        death_threshold: float = 0.05,
        rising_bonus: float = 0.10,
        newborn_period: int = 3,
        newborn_weight: float = 0.05,
        max_total_aux: float = 1.5,
    ):
        self.min_weight = min_weight
        self.death_threshold = death_threshold
        self.rising_bonus = rising_bonus
        self.newborn_period = newborn_period
        self.newborn_weight = newborn_weight
        self.max_total_aux = max_total_aux

        self._newborn_cycles: dict[str, int] = {}  # organ -> cycles since birth
        self._previous_decision: Optional[SenateDecision] = None

    def convene(
        self,
        reputations: dict,
        current_weights: dict[str, float],
        cycle: int,
        ledger_head_hash: str,
    ) -> SenateDecision:
        """
        Convene the senate: read reputations, decide new weights, produce decision.

        Args:
            reputations: dict[organ_name, OrganReputation]
            current_weights: current loss weights
            cycle: current training cycle
            ledger_head_hash: hash of the current blockchain head

        Returns:
            SenateDecision with the new weight distribution
        """
        # Track newborns
        for name, rep in reputations.items():
            if hasattr(rep, 'trend') and rep.trend == "newborn":
                if name not in self._newborn_cycles:
                    self._newborn_cycles[name] = 0
                else:
                    self._newborn_cycles[name] += 1

        # Compute new weights
        new_weights = self._redistribute_weights(reputations, current_weights, cycle)

        # Detect changes
        promotions, demotions, deaths, newborns = [], [], [], []

        for name, rep in reputations.items():
            if not hasattr(rep, 'trend'):
                continue
            if rep.trend == "rising":
                promotions.append(name)
            elif rep.trend == "declining":
                demotions.append(name)
            elif rep.trend == "dead" and name not in deaths:
                deaths.append(name)
            elif rep.trend == "newborn":
                newborns.append(name)

        # Generate reasoning
        reasoning_parts = []
        if promotions:
            reasoning_parts.append(f"Promoted: {', '.join(promotions)}")
        if demotions:
            reasoning_parts.append(f"Demoted: {', '.join(demotions)}")
        if deaths:
            reasoning_parts.append(f"Marked for death: {', '.join(deaths)}")
        if newborns:
            reasoning_parts.append(f"Newborns in trial: {', '.join(newborns)}")

        decision = SenateDecision(
            cycle=cycle,
            timestamp_unix_ms=int(time.time() * 1000),
            previous_weights=dict(current_weights),
            new_weights=new_weights,
            reputations={
                name: getattr(rep, 'reputation_score', 0.5)
                for name, rep in reputations.items()
            },
            promotions=promotions,
            demotions=demotions,
            deaths=deaths,
            newborns=newborns,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No changes",
            senate_block_hash="",
        )

        self._previous_decision = decision
        return decision

    def _redistribute_weights(
        self,
        reputations: dict,
        current_weights: dict[str, float],
        cycle: int,
    ) -> dict[str, float]:
        """Core redistribution algorithm.

        ORGAN_WEIGHT_KEYS maps more than one organ name to the same
        weight_key (spider_sense and spider_calibration both resolve to
        spider_calibration_weight). Computing per-organ and writing
        `new_weights[weight_key] = weight` directly used to let whichever
        organ iterated last silently overwrite the other's contribution
        (dict iteration order, not policy), while still adding BOTH organs'
        weights into `total` — inflating the aux-loss cap check and
        scaling down every other organ's weight for a phantom amount that
        never actually existed in new_weights. Aggregating by weight_key
        first (averaging shared contributions) fixes both: no silent
        overwrite, and `total` reflects only what is actually distributed.
        """
        weights_by_key: dict[str, list[float]] = {}

        for organ_name, rep in reputations.items():
            weight_key = self.ORGAN_WEIGHT_KEYS.get(organ_name)
            if weight_key is None:
                continue

            score = getattr(rep, 'reputation_score', 0.5)
            trend = getattr(rep, 'trend', 'stable')

            # Check newborn status
            newborn_cycles = self._newborn_cycles.get(organ_name, 999)
            is_newborn = newborn_cycles < self.newborn_period

            if is_newborn:
                # Newborn: fixed trial weight
                weight = self.newborn_weight
            elif score < self.death_threshold and not is_newborn:
                # Near death
                weight = self.min_weight * 0.5
            elif score < 0.3:
                # Declining zone
                weight = self.min_weight + (score - self.death_threshold) * 0.3
            elif score <= 0.7:
                # Stable zone
                base = current_weights.get(weight_key, self.DEFAULT_WEIGHTS.get(weight_key, 0.05))
                weight = base  # keep current
            else:
                # Rising zone
                base = current_weights.get(weight_key, self.DEFAULT_WEIGHTS.get(weight_key, 0.05))
                bonus = self.rising_bonus * (score - 0.7) / 0.3
                weight = base + bonus

            weights_by_key.setdefault(weight_key, []).append(max(0.0, weight))

        new_weights = {
            key: sum(values) / len(values) for key, values in weights_by_key.items()
        }
        total = sum(new_weights.values())

        # Normalize: cap total auxiliary at max_total_aux
        if total > self.max_total_aux:
            scale = self.max_total_aux / total
            new_weights = {k: v * scale for k, v in new_weights.items()}

        return new_weights

    def validate_decision(self, decision: SenateDecision) -> bool:
        """Validate that a senate decision is internally consistent."""
        if not decision.new_weights:
            return False

        for weight in decision.new_weights.values():
            if weight < 0:
                return False
            if weight > 1.0:
                return False

        return True


class SenateLedger:
    """Records senate decisions as Proof-of-Learning blocks in the blockchain."""

    @staticmethod
    def create_senate_block(
        decision: SenateDecision,
        prev_block_hash: str,
        block_number: int,
        step_key: str,
    ) -> dict:
        """
        Create a senate decision block (checkpoint_flag=False,
        transaction type="SENATE_DECISION").

        Returns a dict that can be passed to OrganLedger.append_block().
        """
        import json

        tx = {
            "tx_id": f"senate-decision-v1:{decision.cycle}:{decision.timestamp_unix_ms}",
            "phase": "CYCLE_BOUNDARY",
            "tx_type": "SENATE_DECISION",
            "cycle": decision.cycle,
            "previous_weights": decision.previous_weights,
            "new_weights": decision.new_weights,
            "reputations": decision.reputations,
            "promotions": decision.promotions,
            "demotions": decision.demotions,
            "deaths": decision.deaths,
            "newborns": decision.newborns,
            "reasoning": decision.reasoning,
        }

        # Compute merkle root using standard MerkleTree.build (canonical raw SHA-256 hex)
        from f51_darwin.organism.blockchain import MerkleTree
        merkle_root = MerkleTree.build([tx]).root

        block = {
            "header": {
                "schema": "darwin-organ-block-v1",
                "block_number": block_number,
                "prev_block_hash": prev_block_hash,
                "merkle_root": merkle_root,
                "timestamp_unix_ms": decision.timestamp_unix_ms,
                "step_key": step_key,
                "cycle": decision.cycle,
                "optimizer_step": 0,
                "checkpoint_flag": False,
                "checkpoint_sha256": None,
                "organ_identity_merkle_root": "0" * 64,
            },
            "transactions": [tx],
            "block_hash": "",  # filled by ledger on append
        }

        return block


def _sha256_hex(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()
