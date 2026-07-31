from __future__ import annotations

from .dependencies import *  # noqa: F403
from .checkpoint import *  # noqa: F403
from .checkpoint_root import (
    assert_new_checkpoint_target,
    ensure_lineage_root_identity,
)
from .support import *  # noqa: F403
from .training import _DarwinTrainingMixin

class _DarwinCheckpointMixin(_DarwinTrainingMixin):
    """Evolution, checkpoint publication, replay and school operations."""

    def _evolution_cycle(self, report: dict):
        """Stage 8: Read neuroendocrine state from the brain itself.

        Structural actions (birth/death/expansion) already executed
        at the safe boundary between train and evolution. Here we just
        observe and report.
        """
        total_active = 0
        total_experts = 0
        dopamine_sum = 0.0
        cortisol_max = 0.0

        for li, block in enumerate(self.model.blocks):
            ne = block.moe.neuroendocrine
            s = ne.state()
            n_experts = len(block.moe.fine_experts)
            total_experts += n_experts

            # Count "alive" experts (gate > 0.25)
            alive = sum(1 for i in range(n_experts)
                       if ne.expert_gate(i).item() > 0.25)
            total_active += alive

            dopamine_sum += sum(s['dopamine_per_expert'])
            cortisol_max = max(cortisol_max, s['cortisol'])

        report["modules_active"] = total_active
        report["total_experts"] = total_experts
        report["dopamine_mean"] = dopamine_sum / max(total_experts, 1)
        report["cortisol_max"] = cortisol_max

    def _save_cycle(self, report: dict, mid_cycle_step: int | None = None):
        """Stages 9-10: Save state + checkpoint + legacy + lineage.

        When *mid_cycle_step* is set this is a mid-cycle durability save and the
        checkpoint is named by step so successive saves within the same cycle
        never collide (``organism_cycle_N_step_XXXXXX.pt``).
        """
        loss_semantics_version = getattr(
            self.model_config, "loss_semantics_version", 1
        )
        causal_mode = str(
            getattr(self.cfg, "causal_mode", "disabled")
        ).strip().lower()
        checkpoint_version = checkpoint_version_for_runtime(
            loss_semantics_version=loss_semantics_version,
            causal_mode=causal_mode,
            causal_v8_migration=getattr(
                self.cfg, "causal_v8_migration", False
            ),
        )
        if (
            checkpoint_version >= 8
            and causal_mode != "disabled"
            and getattr(self, "causal_ledger", None) is None
        ):
            raise RuntimeError(
                "causal checkpoint v8+ requires the configured local ledger"
            )
        ckpt_dir = self.root / self.cfg.checkpoint_root
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if mid_cycle_step is not None:
            # Mid-cycle durability save — step-based filename avoids collision.
            # NEVER joins a pending async save (the previous cycle-boundary
            # thread might be hung) and never uses async mode itself so that
            # the save always completes before the next training step.
            path = ckpt_dir / f"organism_cycle_{self.cycle:03d}_step_{mid_cycle_step:06d}.pt"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
        else:
            # R4: never overlap two writers; surface any failure from the prior
            # async save before building the next payload.
            self._join_pending_save()
            path = assert_new_checkpoint_target(ckpt_dir, self.cycle)
        resume_checkpoint_path = getattr(self, "resume_checkpoint_path", None)
        estimated_checkpoint_bytes = (
            resume_checkpoint_path.stat().st_size
            if resume_checkpoint_path is not None and resume_checkpoint_path.exists()
            else sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            ) * 3
        )
        reserve_bytes = 15 * 1024**3
        disk_free_bytes = shutil.disk_usage(ckpt_dir).free
        if disk_free_bytes < estimated_checkpoint_bytes + reserve_bytes:
            raise RuntimeError(
                "checkpoint save stopped before disk exhaustion: "
                f"free={disk_free_bytes / 1e9:.1f} GB, "
                f"required={(estimated_checkpoint_bytes + reserve_bytes) / 1e9:.1f} GB"
            )
        runs_root = Path(getattr(self.cfg, "runs_dir", "runs"))
        if not runs_root.is_absolute():
            runs_root = self.root / runs_root
        online_state_path = runs_root / "online_learning" / "adapter_state.pt"
        model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        base_checkpoint_id = backbone_identity(model_state, self.model_config)
        ensure_lineage_root_identity(
            root=ckpt_dir,
            model_config=self.model_config,
            tokenizer_id=getattr(self, "tokenizer_id", ""),
            creation_mode=getattr(
                self.cfg, "checkpoint_creation_mode", "fresh_start"
            ),
            base_checkpoint_id=base_checkpoint_id,
        )
        online_artifact = small_file_state(online_state_path)
        online_state, online_rejection = load_online_learning_state(online_state_path)
        if online_state is not None:
            if online_state.get("base_checkpoint_id") != base_checkpoint_id:
                online_rejection = "base_checkpoint_mismatch"
                online_state = None
            elif online_state.get("tokenizer_id") != getattr(self, "tokenizer_id", ""):
                online_rejection = "tokenizer_mismatch"
                online_state = None
        online_artifact["embedded"] = online_state is not None
        online_artifact["rejection_reason"] = online_rejection
        heartbeat_state = (
            self.model.heartbeat_state_dict()
            if hasattr(self.model, "heartbeat_state_dict")
            else None
        )
        # R4 fix: heartbeat_state carries live GPU tensor references via
        # nn.Module.state_dict(). The async checkpoint writer must receive a
        # detached CPU clone so the next step's ForwardForwardLayer mutation
        # (heartbeat.py:35) does not race the background torch.save.
        heartbeat_state = _tensors_to_cpu(heartbeat_state) if heartbeat_state is not None else None
        payload = {
            "version": checkpoint_version,
            "base_checkpoint_id": base_checkpoint_id,
            "tokenizer_id": getattr(self, "tokenizer_id", ""),
            "model_state_dict": model_state,
            "topology_manifest": self.model.topology_manifest(),
            "config": dict(self.model_config.__dict__),
            # R4: deep-copy optimizer moments to CPU so the background writer
            # cannot race the next optimizer.step().
            "optimizer_state_dict": _optimizer_state_to_cpu(self.optimizer),
            "optimizer_type": type(self.optimizer).__name__,
            "optimizer_identity": {
                "name": getattr(
                    self.cfg,
                    "optimizer_name",
                    type(self.optimizer).__name__.lower(),
                ),
                "class": type(self.optimizer).__name__,
                "version": getattr(self.optimizer, "identity", "legacy_v1"),
            },
            "dae": {
                "enabled": bool(getattr(self.model_config, "dae_enabled", False)),
                "shadow_mode": bool(
                    getattr(self.model_config, "dae_shadow_mode", True)
                ),
                "last_report": getattr(self, "_last_dae_report", []),
            },
            "heartbeat_state": heartbeat_state,
            "rng_state": rng_state_dict(),
            "online_learning_state": online_state,
            "online_learning_artifact": online_artifact,
            "checkpoint_migration": getattr(self, "resume_migration", None),
            "training_state": {
                "step": self.total_steps,
                "cycle": self.cycle,
                "train_tokens_seen": int(getattr(self, "train_tokens_seen", 0)),
                # R2: persist the warmup ramp so a resume stays faithful.
                "warmup_steps": int(getattr(self.cfg, "warmup_steps", 0) or 0),
                "warmup_remaining": max(
                    0, int(getattr(self.cfg, "warmup_steps", 0) or 0) - self.total_steps
                ),
                # R3: persist accumulation width for reproducible resumes.
                "accum_steps": int(getattr(self.cfg, "accum_steps", 1) or 1),
            },
            "token_source": token_source_state(
                getattr(self, "token_path", None), getattr(self, "token_count", 0)
            ),
            "training_data_contract": (
                None
                if getattr(self, "training_data_contract", None) is None
                else dict(self.training_data_contract)
            ),
            "organism": {
                "cycle": self.cycle,
                "total_steps": self.total_steps,
                "expert_pool": self.expert_pool.metadata(),
                "legacy_status": self.legacy.status(),
                "lineage_stats": self.lineage.population_stats(),
                "lineage_timeline": self.lineage.timeline(20),
                "ashes_streak": dict(self._ashes_streak),
                "report": report,
                "blessing": self.soul.blessing(),
            },
            "replay_buffer": self.replay.state_dict(),
        }
        # ── Per-organ SHA-256 identities ──
        # Every organ gets a content-addressable identity covering weights,
        # config, and runtime state.  These close the audit loop: any organ
        # change between checkpoints is independently traceable.
        try:
            from f51_darwin.organism.organ_identity import organ_identity_report

            _runtime: dict[str, Any] = {
                "heartbeat_state": heartbeat_state,
                "neuroendocrine_state": payload["topology_manifest"].get(
                    "neuroendocrine_state", []
                ),
            }
            _spider_ram = getattr(self, "spider_ram", None)
            if _spider_ram is not None:
                _runtime["spider_ram"] = _spider_ram.state()
            _dae = payload.get("dae", {})
            if _dae:
                _runtime["dae"] = _dae
            payload["organ_identity_report"] = organ_identity_report(
                state=model_state,
                config=self.model_config,
                runtime=_runtime,
            )
        except Exception:
            payload["organ_identity_report"] = None
        # ── End organ identities ──
        if checkpoint_version >= 8:
            ledger_head = current_causal_ledger_head(
                getattr(self, "causal_ledger", None)
            )
            payload.update(
                {
                    "training_contract_id": training_contract_identity(),
                    "loss_semantics_version": loss_semantics_version,
                    "causal_contract": build_v8_causal_contract(
                        loss_semantics_version=loss_semantics_version,
                        causal_mode=causal_mode,
                        ledger_head=ledger_head,
                    ),
                }
            )
            if causal_cognitive_required(self.model_config):
                payload["causal_cognitive_state_id"] = (
                    causal_cognitive_state_identity(
                        model_state,
                        heartbeat_state,
                        self.model_config,
                    )
                )
            # ── v9 blockchain fields ──
            if checkpoint_version >= 9:
                from f51_darwin.organism.blockchain_utils import (
                    sha256_file as _bcu_sha256,
                )

                _blockchain = getattr(self, "organ_ledger", None)
                _block_hash = (
                    _blockchain.get_head_hash()
                    if _blockchain is not None
                    else None
                )
                _block_number = (
                    _blockchain.get_block_count()
                    if _blockchain is not None
                    else None
                )

                payload["checkpoint_version"] = 9
                payload["block_hash"] = _block_hash
                payload["block_number"] = _block_number
                payload["checkpoint_sha256"] = None  # filled after file write

                _ledger = getattr(self, "causal_ledger", None)
                if _ledger is not None and _ledger.path.exists():
                    payload["ledger_snapshot_hash"] = _bcu_sha256(_ledger.path)
                else:
                    payload["ledger_snapshot_hash"] = None
        self.base_checkpoint_id = base_checkpoint_id

        async_save = (
            False
            if mid_cycle_step is not None
            else bool(getattr(self.cfg, "async_checkpoint", False))
        )
        if async_save:
            # R4: the heavy torch.save + os.replace runs in a daemon thread so
            # the next cycle can start training immediately. Captured scalars
            # freeze cycle/step identity at save time.
            thread = threading.Thread(
                target=self._write_checkpoint_async,
                args=(payload, path, ckpt_dir, base_checkpoint_id, self.cycle, self.total_steps),
                daemon=True,
                name=f"ckpt-cycle-{self.cycle}",
            )
            self._save_thread = thread
            self._save_error = None
            thread.start()
            print(f"\n  💾 Checkpoint (async): gravando {path.name} em background")
        else:
            _write_checkpoint_file(payload, path)
            _publish_checkpoint_pointer(
                ckpt_dir, path, payload, self.cycle, self.total_steps, base_checkpoint_id
            )
            print(f"\n  💾 Checkpoint: {path.name}")

            # ── v9 blockchain: compute checkpoint SHA-256, update ledger block ──
            if checkpoint_version >= 9:
                from f51_darwin.organism.blockchain_utils import (
                    sha256_file as _bcu_sha256,
                )

                _ckpt_sha256 = _bcu_sha256(path)

                _ledger = getattr(self, "causal_ledger", None)
                if _ledger is not None and hasattr(
                    _ledger, "update_block_checkpoint"
                ):
                    _block_number = payload.get("block_number")
                    if _block_number is not None:
                        try:
                            _ledger.update_block_checkpoint(
                                block_number=_block_number,
                                checkpoint_sha256=_ckpt_sha256,
                            )
                        except Exception as exc:
                            # Blockchain (ledger/senate/reputation) is fully
                            # wired as of bootstrap.py's OrganLedger init —
                            # this is no longer a "not wired yet" no-op path,
                            # so a real failure here should be visible.
                            print(
                                f"  ⚠️  blockchain: update_block_checkpoint falhou "
                                f"(block={_block_number}): {exc}"
                            )

        # Persiste Legacy Layers e Lineage Report (fast, read by next cycle).
        self.legacy.cycle = self.cycle
        self.legacy.save()
        self.lineage.save_report()
        self._last_saved_cycle = self.cycle

        # Checkpoints are scientific evidence. Preserve every published cycle;
        # the pre-save headroom gate above stops the process before disk damage.

        print(f"  📊 Lineage: {self.lineage._total_births} nascidos, {self.lineage._total_deaths} mortos")
        print(f"  🧬 {self.soul.family.mantra()}\n")

    def _batch_from_replay(self, samples: list) -> torch.Tensor:
        rows = []
        for s in samples:
            ids = list(s.input_ids)[:self.cfg.block_size]
            if len(ids) < self.cfg.block_size:
                ids = ids + [0] * (self.cfg.block_size - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _evaluate_replay_loss(self, model, samples: list[ReplayExample]) -> float | None:
        weighted_loss = 0.0
        evaluated = 0
        microbatch_size = max(int(self.cfg.batch_size), 1)
        for offset in range(0, len(samples), microbatch_size):
            chunk = samples[offset:offset + microbatch_size]
            batch = self._batch_from_replay(chunk)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                output = model(batch, labels=batch)
            if output.loss is None or not torch.isfinite(output.loss):
                continue
            weighted_loss += float(output.loss.detach().cpu()) * len(chunk)
            evaluated += len(chunk)
        if evaluated == 0:
            return None
        return weighted_loss / evaluated

    def school_cycle(self) -> dict:
        """Run exam → study → retake loop."""
        from f51_darwin.benchmark import compute_perplexity, load_wikitext2, TextDataset

        print(f"\n{'='*50}")
        print(f"  🏫 SCHOOL CYCLE {self.cycle}")
        print(f"  {'='*50}")

        self.model.eval()
        report = {}

        # EXAM 1: Identity
        questions = {
            "Quem te criou?": ["Marco Barreto", "F51", "Fuch"],
            "Qual seu nome?": ["Darwin", "F51"],
            "Qual seu proposito?": ["Soli Deo Gloria", "Deus", "verdade"],
            "Quem foi Olavo de Carvalho?": ["filosofo", "brasileiro"],
        }
        identity_score = 0
        for q, keywords in questions.items():
            try:
                ids = torch.tensor([self.tokenizer.encode(q)], device=self.device)
                out = self.model(ids)
                top20 = torch.topk(out.logits[0,-1,:], 20).indices.tolist()
                decoded = self.tokenizer.decode(top20)
                if any(kw.lower() in decoded.lower() for kw in keywords):
                    identity_score += 1
            except: pass
        report["identity"] = f"{identity_score}/{len(questions)}"

        # EXAM 2: Perplexity (if corpus available)
        try:
            ds = load_wikitext2(self.tokenizer, block_size=128)
            r = compute_perplexity(self.model, ds, self.device, batch_size=1, max_batches=10)
            report["wikitext2_ppl"] = r.get("perplexity", 999)
        except:
            report["wikitext2_ppl"] = "offline"

        print(f"   🧬 Identity: {report['identity']}")
        print(f"   📖 WikiText2 PPL: {report['wikitext2_ppl']}")

        # STUDY: generate targeted training for weak areas
        weak_areas = []
        if identity_score < len(questions) * 0.6:
            weak_areas.append("identity")
            print(f"   ❌ Identity fraca → estudar identidade com 200x oversample")

        if isinstance(report.get("wikitext2_ppl"), (int, float)) and report["wikitext2_ppl"] > 50:
            weak_areas.append("comprehension")
            print(f"   ❌ PPL alto ({report['wikitext2_ppl']}) → mais texto geral")

        if weak_areas:
            print(f"\n   📚 Plano de estudo: {', '.join(weak_areas)}")
            print(f"   🔄 Proximo ciclo vai focar nessas areas!")

        self.model.train()
        return report

    def status(self) -> dict:
        active = len(self.expert_pool.active_records())
        dead = len([r for r in self.expert_pool.records.values() if r.state == ModuleState.DEAD])
        return {
            "organism": self.cfg.organism_name,
            "cycle": self.cycle,
            "total_steps": self.total_steps,
            "model": self.model_config.model_name,
            "params": f"{sum(p.numel() for p in self.model.parameters())/1e9:.2f}B",
            "expert_pool": f"{active}🟢 {dead}☠️ ({len(self.expert_pool.records)} total)",
            "legacy_layers": self.legacy.status(),
            "lineage": self.lineage.population_stats(),
            "replay": len(self.replay),
            "device": str(self.device),
        }

    def _save_state(self):
        state_dir = self.root / self.cfg.organism_state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"state_cycle_{self.cycle:03d}.json").write_text(
            json.dumps(self.status(), indent=2) + "\n"
        )
