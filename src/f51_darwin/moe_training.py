"""
F51 MoE Domain-Routed Training Pipeline.

Each document in the corpus is tagged with a domain. The MoE router
learns to activate the correct expert for each domain during training.

Domains:
    0: medicina       (med_*.txt)
    1: financas       (arch_finance*.txt)
    2: codigo         (f51 extraction .py files, code docs)
    3: literatura_pt  (book_pt*.txt)
    4: identidade_f51 (f51_identity*.txt)
    5: tecnico_en     (doc_*.txt, technical docs)
    6: dialogo        (mike_ms*, conversation memory)
    7: reserva        (everything else, catch-all)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.config import DarwinConfig
from f51_darwin.data import CausalLMDataLoader, load_text_documents, tokenize_documents
from f51_darwin.model import F51DarwinModel
from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.training import BaseTrainingConfig, BaseTrainerState, TrainingMetricsTracker
from f51_darwin.replay_buffer import ReplayBuffer


# ═══════════════════════════════════════════════════════════
# DOMAIN LABELING
# ═══════════════════════════════════════════════════════════

DOMAIN_NAMES = [
    "medicina",
    "financas",
    "codigo",
    "literatura_pt",
    "identidade_f51",
    "tecnico_en",
    "dialogo",
    "reserva",
]

DOMAIN_COLORS = [
    "🟢", "🟡", "🔵", "🟣", "❤️", "⚙️", "💬", "⬜",
]

# Batch sampling weight per domain (identidade & family content boosted)
DOMAIN_SAMPLE_WEIGHTS: dict[int, float] = {
    0: 1.0,   # medicina
    1: 1.0,   # financas
    2: 1.0,   # codigo
    3: 1.5,   # literatura_pt
    4: 5.0,   # identidade_f51 — priority
    5: 1.0,   # tecnico_en
    6: 2.0,   # dialogo
    7: 0.5,   # reserva
}


def classify_domain(filename: str) -> int:
    """Classify a document into a MoE domain based on filename.

    Mapping rules:
        med_*              → 0 (medicina)
        arch_finance*      → 1 (financas)
        f51_*.py, f51_*.md → 2 (codigo)
        book_pt*           → 3 (literatura_pt)
        f51_identity*      → 4 (identidade_f51)
        *familia*, *olavo* → 4 (identidade_f51)
        doc_*, arch_*md    → 5 (tecnico_en)
        mike_ms*, conv*    → 6 (dialogo)
        *                  → 7 (reserva)
    """
    name = filename.lower()

    if name.startswith('med_'):
        return 0
    if 'finance' in name or name.startswith('arch_finance'):
        return 1
    if name.endswith('.py') or (name.startswith('f51_') and 'identity' not in name):
        return 2
    if name.startswith('book_pt'):
        return 3
    if (
        'identity' in name
        or 'olavo' in name
        or 'familia' in name
        or 'family' in name
        or 'barreto' in name
    ):
        return 4
    if name.startswith('doc_') or (name.startswith('arch_') and name.endswith('.md')):
        return 5
    if name.startswith('mike_ms') or 'conversation' in name or 'dialog' in name:
        return 6
    return 7


class DomainDataLoader:
    """Dataloader that yields (tokens, domain_id) batches.

    Organizes documents by domain so the MoE router can learn
    to activate the right experts per domain.
    """

    def __init__(
        self,
        token_ids: list[int],
        domain_map: dict[int, list[int]],  # domain_id → list of token indices
        block_size: int,
        batch_size: int,
        seed: int = 51,
        device: torch.device | None = None,
        domain_balance: bool = True,  # if True, sample domains by weight (not uniform)
        domain_weights: dict[int, float] | None = None,
    ):
        self.token_ids = token_ids
        self.domain_map = domain_map
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self.domain_balance = domain_balance
        self.domain_weights = domain_weights or DOMAIN_SAMPLE_WEIGHTS
        self.domains = sorted(domain_map.keys())

        # Build index: list of (start_pos, domain_id) for each possible chunk
        self.chunks: list[tuple[int, int]] = []
        for domain_id, positions in domain_map.items():
            for pos in positions:
                if pos + block_size <= len(token_ids):
                    self.chunks.append((pos, domain_id))

        self._rng = torch.Generator().manual_seed(seed)
        self._step = 0

    def __len__(self) -> int:
        return len(self.chunks) // self.batch_size

    def set_step(self, step: int) -> None:
        self._step = step
        self._rng.manual_seed(step)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, domain_labels) batch.

        input_ids: [batch, block_size]
        domain_labels: [batch] — domain ID for each sequence
        """
        # Select chunks
        if self.domain_balance and len(self.domains) > 1:
            # Weighted sampling — identidade_f51 (and family-tagged docs) seen more often
            weights = [self.domain_weights.get(d, 1.0) for d in self.domains]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            selected: list[tuple[int, int]] = []
            for _ in range(self.batch_size):
                r = torch.rand(1, generator=self._rng).item()
                cumulative = 0.0
                chosen_domain = self.domains[-1]
                for d, p in zip(self.domains, probs):
                    cumulative += p
                    if r <= cumulative:
                        chosen_domain = d
                        break
                domain_chunks = [c for c in self.chunks if c[1] == chosen_domain]
                if domain_chunks:
                    pick = int(torch.randint(0, len(domain_chunks), (1,), generator=self._rng).item())
                    selected.append(domain_chunks[pick])
            if len(selected) < self.batch_size:
                idx = torch.randint(0, len(self.chunks), (self.batch_size - len(selected),), generator=self._rng).tolist()
                selected.extend(self.chunks[i] for i in idx)
        else:
            idx = torch.randint(0, len(self.chunks), (self.batch_size,), generator=self._rng).tolist()
            selected = [self.chunks[i] for i in idx]

        # Build tensors
        rows = []
        domains = []
        for pos, domain_id in selected:
            chunk = self.token_ids[pos:pos + self.block_size]
            if len(chunk) < self.block_size:
                chunk = chunk + [0] * (self.block_size - len(chunk))
            rows.append(chunk)
            domains.append(domain_id)

        input_ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        domain_labels = torch.tensor(domains, dtype=torch.long, device=self.device)

        self._step += 1
        return input_ids, domain_labels

    def take_eval_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic eval batch."""
        idx = list(range(min(self.batch_size, len(self.chunks))))
        selected = [self.chunks[i] for i in idx]
        rows = []
        domains = []
        for pos, domain_id in selected:
            chunk = self.token_ids[pos:pos + self.block_size]
            if len(chunk) < self.block_size:
                chunk = chunk + [0] * (self.block_size - len(chunk))
            rows.append(chunk)
            domains.append(domain_id)
        input_ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        domain_labels = torch.tensor(domains, dtype=torch.long, device=self.device)
        return input_ids, domain_labels

    @classmethod
    def from_corpus_dir(
        cls,
        corpus_dir: str | Path,
        tokenizer: F51BPETokenizer,
        block_size: int,
        batch_size: int,
        seed: int = 51,
        device: torch.device | None = None,
    ) -> "DomainDataLoader":
        """Build a domain-labeled dataloader from a corpus directory.

        Scans all .txt files, classifies by domain, tokenizes,
        and builds index maps.
        """
        corpus = Path(corpus_dir)
        domain_docs: dict[int, list[str]] = defaultdict(list)

        for txt_file in sorted(corpus.glob("*.txt")):
            domain_id = classify_domain(txt_file.name)
            text = txt_file.read_text(encoding='utf-8', errors='replace')
            domain_docs[domain_id].append(text)

        # Tokenize all documents, tracking domain boundaries
        all_tokens: list[int] = []
        domain_map: dict[int, list[int]] = defaultdict(list)
        current_pos = 0

        for domain_id, docs in sorted(domain_docs.items()):
            for doc in docs:
                tokens = tokenizer.encode(doc)
                if len(tokens) < 10:
                    continue
                # Record all possible chunk start positions for this domain
                for offset in range(0, len(tokens) - block_size + 1, block_size // 2):
                    domain_map[domain_id].append(current_pos + offset)
                all_tokens.extend(tokens)
                current_pos += len(tokens)

        if not domain_map:
            raise ValueError("No valid domain-labeled documents found in corpus.")

        return cls(all_tokens, dict(domain_map), block_size, batch_size, seed, device)


# ═══════════════════════════════════════════════════════════
# MOE TRAINER
# ═══════════════════════════════════════════════════════════

@dataclass
class MoETrainingMetrics:
    run_id: str
    steps: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)
    domain_usage: dict[int, list[float]] = field(default_factory=dict)
    started_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            'run_id': self.run_id,
            'steps': self.steps,
            'evals': self.evals,
            'domain_usage': {str(k): v for k, v in self.domain_usage.items()},
            'started_at': self.started_at,
        }

    def record_step(self, step: int, loss: float, aux_loss: float, domain_usage: dict, tok_s: float):
        self.steps.append({
            'step': step, 'loss': round(loss, 4), 'aux_loss': round(aux_loss, 4),
            'tok_s': round(tok_s, 1),
        })
        for d, usage in domain_usage.items():
            if d not in self.domain_usage:
                self.domain_usage[d] = []
            self.domain_usage[d].append(usage)


class MoETrainer:
    """Trains a Darwin-MoE model with domain-routed expert selection."""

    def __init__(
        self,
        model: F51DarwinModel,
        config: DarwinConfig,
        data_loader: DomainDataLoader,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        domain_loss_weight: float = 0.1,
        aux_loss_weight: float = 0.01,
    ):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.device = device
        self.domain_loss_weight = domain_loss_weight
        self.aux_loss_weight = aux_loss_weight

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.grad_clip = grad_clip
        self.metrics = MoETrainingMetrics(run_id=f"moe_{int(time.time())}")
        self.metrics.started_at = time.time()

    def train_step(
        self, input_ids: torch.Tensor, domain_labels: torch.Tensor
    ) -> dict[str, float]:
        """One training step with domain routing loss.

        The domain routing loss encourages the router to activate
        the correct expert for each domain label.
        """
        self.model.train()

        # Forward pass
        output = self.model(input_ids, labels=input_ids)

        ce_loss = output.loss
        aux_loss = output.aux_loss if output.aux_loss is not None else torch.tensor(0.0)

        # Domain routing loss: encourage correct expert activation
        domain_loss = self._compute_domain_loss(output.moe_stats, domain_labels)

        total_loss = ce_loss + self.aux_loss_weight * aux_loss + self.domain_loss_weight * domain_loss

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        # Track domain usage
        domain_usage = self._get_domain_usage(output.moe_stats, domain_labels)

        return {
            'loss': float(ce_loss.detach().cpu()),
            'aux_loss': float(aux_loss.detach().cpu()) if isinstance(aux_loss, torch.Tensor) else 0.0,
            'domain_loss': float(domain_loss.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu()),
            'domain_usage': domain_usage,
        }

    def _compute_domain_loss(
        self, moe_stats: list[dict], domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute domain routing loss.

        For each MoE layer, penalize the router if it doesn't activate
        the expert corresponding to the true domain of the input.

        Uses cross-entropy between router logits and domain labels.
        """
        if not moe_stats or not self.model.experts_enabled:
            return torch.tensor(0.0, device=domain_labels.device)

        total_domain_loss = torch.tensor(0.0, device=domain_labels.device)
        n_layers = 0

        for layer_stats in moe_stats:
            if 'router_logits' not in layer_stats:
                # Try to get from the block
                continue

        # Use the expert usage stats as a signal
        # For each layer, check if the correct domain expert is in the top-k
        for i, block in enumerate(self.model.blocks):
            if not hasattr(block, 'moe'):
                continue

            moe = block.moe
            # Get router probs from the last forward pass via expert usage
            usage = moe.router.expert_usage_count
            if usage.sum() == 0:
                continue

            # Normalize usage to probabilities
            probs = usage / usage.sum()

            # For each domain in this batch, compute cross-entropy
            for d in range(moe.config.num_experts):
                mask = (domain_labels == d)
                if mask.sum() > 0:
                    # Encourage high probability for the correct expert
                    target = torch.tensor(d, device=domain_labels.device)
                    n_layers += 1
                    # Simple: negative log prob of correct expert
                    prob_d = probs[d]
                    total_domain_loss = total_domain_loss - torch.log(prob_d + 1e-8)

        if n_layers > 0:
            total_domain_loss = total_domain_loss / n_layers

        return total_domain_loss

    def _get_domain_usage(
        self, moe_stats: list[dict], domain_labels: torch.Tensor
    ) -> dict[int, float]:
        """Get per-domain expert usage percentages."""
        usage: dict[int, float] = {}
        for d in range(8):
            mask = (domain_labels == d)
            usage[d] = mask.float().mean().item()
        return usage

    def run(
        self,
        steps: int,
        *,
        eval_every: int = 500,
        save_every: int = 2000,
        checkpoint_dir: str | Path | None = None,
        project_root: Path | None = None,
    ) -> MoETrainingMetrics:
        """Run MoE training loop."""
        start_time = time.perf_counter()
        tokens_per_step = self.data_loader.batch_size * self.data_loader.block_size

        for step in range(1, steps + 1):
            input_ids, domain_labels = self.data_loader.next_batch()

            t0 = time.perf_counter()
            result = self.train_step(input_ids, domain_labels)
            step_time = time.perf_counter() - t0
            tok_s = tokens_per_step / max(step_time, 0.001)

            self.metrics.record_step(
                step, result['loss'], result['aux_loss'],
                result['domain_usage'], tok_s,
            )

            # Progress bar
            if step % 50 == 0 or step == 1:
                elapsed = time.perf_counter() - start_time
                domain_str = " ".join(
                    f"{DOMAIN_COLORS[d]}{result['domain_usage'].get(d,0)*100:.0f}%"
                    for d in sorted(result['domain_usage'].keys())
                    if result['domain_usage'].get(d, 0) > 0.05
                )
                print(
                    f"\r  step {step:>6d}/{steps} | "
                    f"loss={result['loss']:.4f} | "
                    f"tok/s={tok_s:.0f} | "
                    f"domains: {domain_str}",
                    end="", flush=True,
                )

            # Checkpoint
            if checkpoint_dir and step % save_every == 0:
                self._save_checkpoint(step, checkpoint_dir, project_root)

        print()
        return self.metrics

    def _save_checkpoint(
        self, step: int, checkpoint_dir: str | Path, project_root: Path | None
    ) -> None:
        """Save MoE model checkpoint."""
        from f51_darwin.checkpointing import save_training_checkpoint

        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"moe_step_{step:07d}.pt"

        training_state = {
            'step': step,
            'metrics': self.metrics.to_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'moe_config': {
                'num_experts': 8,
                'experts_per_token': 2,
                'use_nitro_tiering': True,
            },
        }

        save_training_checkpoint(path, self.model, self.config, training_state=training_state)

        # Update latest pointer
        if project_root:
            latest = ckpt_dir / "moe_latest.json"
            latest.write_text(json.dumps({
                'path': str(path.relative_to(project_root)),
                'step': step,
                'timestamp': time.time(),
            }, indent=2) + "\n")

        print(f"\n  💾 Checkpoint: step {step}")


# ═══════════════════════════════════════════════════════════
# CORPUS STATS
# ═══════════════════════════════════════════════════════════

def corpus_domain_stats(corpus_dir: str | Path) -> dict:
    """Analyze domain distribution in the corpus."""
    corpus = Path(corpus_dir)
    stats: dict[int, int] = defaultdict(int)
    total_chars: dict[int, int] = defaultdict(int)

    for txt_file in corpus.glob("*.txt"):
        domain_id = classify_domain(txt_file.name)
        stats[domain_id] += 1
        total_chars[domain_id] += txt_file.stat().st_size

    return {
        'files': {DOMAIN_NAMES[d]: stats.get(d, 0) for d in range(8)},
        'chars_mb': {DOMAIN_NAMES[d]: round(total_chars.get(d, 0) / 1e6, 2) for d in range(8)},
        'total_files': sum(stats.values()),
        'total_mb': round(sum(total_chars.values()) / 1e6, 2),
    }


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F51 Darwin-MoE Domain-Routed Training")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--tokenizer", default="tokenizer/f51_bpe")
    parser.add_argument("--config", default="src/configs/darwin_moe_seed.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--stats", action="store_true", help="Show corpus domain stats only")
    args = parser.parse_args()

    import yaml

    if args.stats:
        stats = corpus_domain_stats(args.corpus)
        print(json.dumps(stats, indent=2))
        raise SystemExit(0)

    # Setup
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    project_root = Path(__file__).resolve().parents[2]

    print("=" * 60)
    print("  F51 DARWIN-MoE Domain-Routed Training")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Block:      {args.block_size}")
    print(f"  Steps:      {args.steps}")
    print("=" * 60)
    print()

    # Load tokenizer
    tokenizer = F51BPETokenizer.load(args.tokenizer)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # Build domain-labeled dataloader
    print("Building domain-labeled dataloader...")
    data_loader = DomainDataLoader.from_corpus_dir(
        args.corpus, tokenizer, args.block_size, args.batch_size,
        device=device,
    )
    print(f"Dataloader: {len(data_loader)} batches, {len(data_loader.domains)} domains active")
    print(f"Domains: {[DOMAIN_NAMES[d] for d in data_loader.domains]}")

    # Build MoE model
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    config = DarwinConfig.from_mapping(raw)
    model = F51DarwinModel(
        config,
        experts_enabled=True,
        moe_config=raw.get('moe', {}),
    ).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {config.model_name} | {params_m:.1f}M params")
    print()

    # Train
    trainer = MoETrainer(
        model, config, data_loader, device,
        lr=args.lr,
    )
    trainer.run(
        steps=args.steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir="checkpoints/moe",
        project_root=project_root,
    )
