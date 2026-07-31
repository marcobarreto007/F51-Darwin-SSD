#!/usr/bin/env python3
"""
F51 UNIFIED TRAIN — Todos os órgãos integrados.

Ghost Token + JEPA + Curiosity + ExpertWeightRouter + NaN Rollback + LR Warmup.
Salva checkpoints a cada 250 steps com rollback automático se NaN detectado.

Uso: python3 unified_train.py [--resume CHECKPOINT]
"""

import sys, time, struct, torch, math, os, psutil
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from f51_darwin.config import DarwinConfig
from f51_darwin.model import F51DarwinModel
from f51_darwin.data import CausalLMDataLoader, tokenize_corpus_dir, discover_corpus_files
from f51_darwin.training import BaseTrainer, BaseTrainingConfig, BaseTrainerState
from f51_darwin.tokenizer import F51BPETokenizer
from f51_darwin.ghost_token import GhostTokenTrainer
from f51_darwin.jepa import JEPAHead, jepa_loss
from f51_darwin.curiosity import CuriosityDrive
from f51_darwin.expert_weight_router import ExpertWeightRouter
from f51_darwin.quality_gate import validate_output
from f51_darwin.decision_engine import decide, DecisionFactors

class CPUOffloadedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.gpu_params = [p for p in params if p.requires_grad]
        self.cpu_params = []
        for p in self.gpu_params:
            p_cpu = p.detach().cpu().float().clone().requires_grad_(True)
            self.cpu_params.append(p_cpu)
            
        self.opt = torch.optim.AdamW(self.cpu_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(self.cpu_params, self.opt.defaults)

    @property
    def param_groups(self):
        return self.opt.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.opt.param_groups = value

    @property
    def defaults(self):
        return self.opt.defaults

    @defaults.setter
    def defaults(self, value):
        self.opt.defaults = value

    @property
    def state(self):
        return self.opt.state

    @state.setter
    def state(self, value):
        self.opt.state = value

    def step(self, closure=None):
        for p_gpu, p_cpu in zip(self.gpu_params, self.cpu_params):
            if p_gpu.grad is not None:
                if p_cpu.grad is None:
                    p_cpu.grad = torch.zeros_like(p_cpu)
                p_cpu.grad.copy_(p_gpu.grad)
                
        self.opt.step(closure)
        
        with torch.no_grad():
            for p_gpu, p_cpu in zip(self.gpu_params, self.cpu_params):
                p_gpu.copy_(p_cpu)

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)
        for p_gpu in self.gpu_params:
            if p_gpu.grad is not None:
                if set_to_none:
                    p_gpu.grad = None
                else:
                    p_gpu.grad.zero_()

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
MODEL_CONFIG = {
    "vocab_size": 58162,
    "d_model": 1536,
    "n_layers": 12,
    "n_heads": 12,
}

MOE_CONFIG = {
    "num_experts": 16,          # mantido compatível com checkpoint ghost_1.9b
    "experts_per_token": 2,
    "expert_hidden_mult": 1,
    "use_nitro_tiering": True,
    "nitro_gpu_capacity": 4,    # reduzido: 4 experts GPU, 12 em CPU RAM
}

TRAINING_CONFIG = {
    "batch_size": 1,
    "block_size": 256,           # reduzido de 512 → menos ativações
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "max_steps": 999_999,
    "eval_every": 500,
    "save_every": 200,           # salva mais frequente pra não perder progresso
    "grad_clip": 1.0,
    "seed": 51,
    "warmup_steps": 500,
    "checkpoint_dir": "checkpoints/unified",
}

LOSS_WEIGHTS = {
    "ghost": 0.10,
    "jepa": 0.05,
    "curiosity": 0.02,
}


def summarize_moe_stats(moe_stats: list[dict]) -> str:
    if not moe_stats:
        return "moe=off"

    entropies = [
        float(stats["router_entropy_norm"])
        for stats in moe_stats
        if stats.get("router_entropy_norm") is not None
    ]
    z_losses = [
        float(stats["router_z_loss"])
        for stats in moe_stats
        if stats.get("router_z_loss") is not None
    ]
    dead_total = sum(len(stats.get("dead_experts", [])) for stats in moe_stats)
    last_stats = moe_stats[-1]
    usage = last_stats.get("tokens_per_expert") or []
    top = sorted(enumerate(usage), key=lambda item: item[1], reverse=True)[:3] if usage else []
    top_text = ",".join(f"{idx}:{count:.0f}" for idx, count in top) if top else "-"
    entropy_min = min(entropies) if entropies else 0.0
    z_max = max(z_losses) if z_losses else 0.0
    return f"moe_ent_min={entropy_min:.3f} moe_dead={dead_total} moe_z_max={z_max:.3f} moe_top={top_text}"

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Caminho do arquivo YAML de config do modelo")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--corpus", default="data/unified_corpus")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"], help="Otimizador a usar para o treino (adamw ou adafactor)")
    parser.add_argument("--offload-optimizer", action="store_true", help="Desloca os estados do AdamW para a RAM da CPU para economizar VRAM")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Precisao do treino: auto, bf16, fp16 ou fp32")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 solicitado, mas esta GPU/PyTorch nao reporta suporte a BF16.")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    amp_enabled = amp_dtype is not None and device.type == "cuda"

    project_root = ROOT
    checkpoint_dir = project_root / TRAINING_CONFIG["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ═══ TOKENIZER ═══
    tok_path = project_root / "tokenizer" / "f51_bpe_80k"
    tokenizer = F51BPETokenizer.load(tok_path)
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # ═══ CARREGAR TOKENS ═══
    # Tenta tokens_full.bin ou tokens.bin, senão tokeniza
    tokens_bin = project_root / "data" / "tokens_full.bin"
    if not tokens_bin.exists():
        tokens_bin = project_root / "data" / "tokens.bin"
    token_ids = None

    if tokens_bin.exists() and tokens_bin.stat().st_size > 1_000_000:
        print(f"Carregando {tokens_bin.name} ({tokens_bin.stat().st_size/1e9:.1f}GB) via memmap...")
        import numpy as np
        token_ids = np.memmap(tokens_bin, dtype=np.int32, mode='r')
        print(f"OK: {len(token_ids)/1e6:.1f}M tokens (memmap)")

    if token_ids is None:
        corpus_dir = project_root / args.corpus
        if not corpus_dir.exists():
            corpus_dir = project_root / "data" / "corpus"
        print(f"Tokenizando {corpus_dir}...")
        token_ids = tokenize_corpus_dir(corpus_dir, tokenizer)
        print(f"OK: {len(token_ids)/1e6:.1f}M tokens")

    if len(token_ids) < TRAINING_CONFIG["block_size"] * 2:
        print(f"ERRO: Corpus muito pequeno ({len(token_ids)} tokens)")
        return 1

    # ═══ MODELO ═══
    if args.config:
        print(f"Carregando config de {args.config}...")
        config = DarwinConfig.from_yaml(args.config)
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        moe_config = raw_cfg.get("moe", MOE_CONFIG)
        experts_enabled = moe_config.get("enabled", True)
    else:
        config = DarwinConfig(**MODEL_CONFIG)
        moe_config = MOE_CONFIG
        experts_enabled = True

    model = F51DarwinModel(config, experts_enabled=experts_enabled, moe_config=moe_config)
    if precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif precision == "fp16":
        model = model.to(dtype=torch.float16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {total_params/1e9:.2f}B params ({precision})")

    # Carregar checkpoint se fornecido (na CPU para evitar OOM na GPU)
    start_step = 0
    ckpt = None
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Resumindo de {ckpt_path.name}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)  # raw state_dict
            start_step = ckpt.get("training_state", {}).get("step", 0)
            print(f"Step inicial: {start_step}")
    model = model.to(device)  # move pra GPU só depois de carregar checkpoint

    # ═══ ÓRGÃOS ═══
    active_num_experts = int(moe_config.get("num_experts", MOE_CONFIG["num_experts"]))
    ghost = GhostTokenTrainer(mask_ratio=0.15, mask_token_id=2, num_experts=active_num_experts)
    jepa = JEPAHead(d_model=config.d_model).to(device)
    curiosity = CuriosityDrive(exploration_budget=0.30)
    expert_router = ExpertWeightRouter(num_experts=active_num_experts, recency_window=3600)

    print(f"Órgãos: Ghost+JEPA+Curiosity+ExpertRouter — TODOS ATIVOS")

    # ═══ DATA LOADER + TRAINER ═══
    loader = CausalLMDataLoader(token_ids, block_size=TRAINING_CONFIG["block_size"],
                                 batch_size=TRAINING_CONFIG["batch_size"],
                                 seed=TRAINING_CONFIG["seed"], device=device)
    loader.set_step(start_step)

    cfg = BaseTrainingConfig(
        batch_size=TRAINING_CONFIG["batch_size"],
        block_size=TRAINING_CONFIG["block_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        max_steps=TRAINING_CONFIG["max_steps"],
        eval_every=TRAINING_CONFIG["eval_every"],
        save_every=TRAINING_CONFIG["save_every"],
        grad_clip=TRAINING_CONFIG["grad_clip"],
        seed=TRAINING_CONFIG["seed"],
        replay_capacity=512,
        replay_sample_size=16,
        replay_seed_every=50,
        checkpoint_dir=TRAINING_CONFIG["checkpoint_dir"],
        run_id=f"unified_{int(time.time())}",
    )

    # ═══ OTIMIZADOR ═══
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 0
    force_offload = (vram_gb > 0 and vram_gb < 24) and not args.offload_optimizer and args.optimizer != "adafactor"
    if force_offload:
        print(f"⚠️  VRAM={vram_gb:.1f}GB < 24GB → ativando CPU offload automaticamente!")
        print(f"    (use --optimizer adafactor para alternativa sem offload)")
    if args.optimizer == "adafactor":
        print("Ativando otimizador GPU Adafactor (Ultra-Low Memory, Sem Offload necessário)!")
        optimizer = torch.optim.Adafactor(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"]
        )
    elif args.offload_optimizer or force_offload:
        print("Ativando blindagem de VRAM: Offloading do otimizador para CPU RAM ativo!")
        optimizer = CPUOffloadedAdamW(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"]
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"]
        )

    # Carrega estado do otimizador se resumindo e disponivel
    if ckpt is not None and "optimizer_state_dict" in ckpt:
        print("Carregando estados do otimizador do checkpoint...")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16" and device.type == "cuda"))

    trainer = BaseTrainer(
        model=model, model_config=config, training_config=cfg,
        data_loader=loader, device=device,
        tokenizer_path=str(tok_path), project_root=project_root,
        optimizer=optimizer,
        state=BaseTrainerState(step=start_step, run_id=cfg.run_id),
    )

    # ═══ TREINO ═══
    print(f"\n{'='*60}")
    print(f"  F51 UNIFIED TRAIN — Todos os órgãos")
    print(f"  Ghost Token + JEPA + Curiosity + ExpertRouter")
    print(f"  NaN Rollback + LR Warmup + Quality Gate")
    print(f"  Save every {TRAINING_CONFIG['save_every']} steps")
    print(f"{'='*60}\n")

    total_tokens = start_step * cfg.batch_size * cfg.block_size
    started_at = time.time()
    loss_history = []
    best_loss = float("inf")
    last_good_checkpoint = None
    steps_since_save = 0
    nan_detected = False

    try:
        for step in range(TRAINING_CONFIG["max_steps"]):
            # ═══ LR WARMUP ═══
            current_step = trainer.state.step
            if current_step < TRAINING_CONFIG["warmup_steps"]:
                lr_scale = (current_step + 1) / TRAINING_CONFIG["warmup_steps"]
                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = TRAINING_CONFIG["learning_rate"] * lr_scale

            # ═══ FORWARD ═══
            batch = loader.next_batch()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(batch, labels=batch)

            # ═══ NaN DETECTION + ROLLBACK ═══
            if output.loss is not None and (torch.isnan(output.loss) or torch.isinf(output.loss)):
                nan_detected = True
                print(f"\n  🚨 NaN/Inf detectado no step {current_step}!")
                if last_good_checkpoint and last_good_checkpoint.exists():
                    print(f"  ⏪ Rollback para {last_good_checkpoint.name}")
                    ckpt = torch.load(last_good_checkpoint, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                    model = model.to(device)
                print(f"  ⏭️  Pulando batch com NaN, continuando...")
                continue

            lm_loss = output.loss if output.loss is not None else torch.tensor(0.0, device=device)
            total_loss = lm_loss
            loss_dict = {"lm": lm_loss.item()}

            # ═══ GHOST TOKEN LOSS ═══
            if output.hidden_states is not None and current_step > 10:
                try:
                    # Seleciona tokens aleatórios para mascarar (evita padding/speciais)
                    prob = torch.rand(batch.shape, device=device)
                    special_mask = batch < 4  # <pad>, <unk>, <mask>, <bos>
                    mask = (prob < ghost.mask_ratio) & (~special_mask)

                    # Cria input mascarado
                    masked_ids = batch.clone()
                    masked_ids[mask] = ghost.mask_token_id

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        masked_out = model(masked_ids)
                    g_logits = masked_out.logits.clamp(-15, 15)
                    if mask.sum() > 0:
                        g_loss = torch.nn.functional.cross_entropy(
                            g_logits[mask].view(-1, config.vocab_size),
                            batch[mask].view(-1), reduction="mean",
                        )
                        total_loss = total_loss + LOSS_WEIGHTS["ghost"] * g_loss
                        loss_dict["ghost"] = g_loss.item()
                except Exception as exc:
                    raise RuntimeError("Ghost Token loss failed") from exc

            # ═══ JEPA LOSS ═══
            if output.hidden_states is not None and current_step > 10:
                try:
                    j_loss = jepa_loss(output.hidden_states.float(), jepa)  # fp32 pra JEPA
                    total_loss = total_loss + LOSS_WEIGHTS["jepa"] * j_loss
                    loss_dict["jepa"] = j_loss.item()
                except Exception as exc:
                    raise RuntimeError("JEPA loss failed") from exc

            # ═══ BACKWARD ═══
            trainer.optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()

            # ═══ GRADIENT CLEANING & SHIELDING (nan_to_num) ═══
            grad_nan_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grad_nan_count += 1
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=65000.0, neginf=-65000.0)
            if grad_nan_count > 0:
                print(f"  🛡️  NaN/Inf detectado em {grad_nan_count} tensores de gradiente! Blindado com sucesso.")

            scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["grad_clip"])
            scaler.step(trainer.optimizer)
            scaler.update()
            trainer.state.step += 1
            total_tokens += cfg.batch_size * cfg.block_size
            steps_since_save += 1

            # ═══ EXPERT WEIGHT UPDATE ═══
            if current_step % 100 == 0 and output.moe_stats:
                for layer_stats in output.moe_stats:
                    usage = layer_stats.get("expert_usage", [])
                    for eid, count in enumerate(usage):
                        if count > 0:
                            expert_router.record_call(eid, correct=(lm_loss.item() < best_loss),
                                                     ghost_accuracy=loss_dict.get("ghost", 0.5))

            # ═══ CURIOSITY REWARD ═══
            if current_step % 100 == 0:
                jepa_error = loss_dict.get("jepa", 0.5)
                curiosity.evaluate_curiosity_reward(
                    output.hidden_states.float() if output.hidden_states is not None else torch.zeros(1,1,384),
                    jepa_prediction_error=jepa_error,
                    domain="training",
                    loss_improved=(lm_loss.item() < best_loss),
                )

            # ═══ BEST LOSS TRACKING ═══
            if lm_loss.item() < best_loss:
                best_loss = lm_loss.item()

            # ═══ SAVE CHECKPOINT ═══
            if steps_since_save >= TRAINING_CONFIG["save_every"]:
                ckpt_path = trainer.save_checkpoint(versioned=True)
                # Salva estado dos órgãos
                state = {
                    "curiosity": curiosity.state.to_dict(),
                    "expert_weights": expert_router.get_all_weights(),
                    "loss_history": loss_history[-100:],
                }
                state_path = checkpoint_dir / f"organs_step_{trainer.state.step:07d}.json"
                state_path.write_text(
                    __import__("json").dumps(state, indent=2))
                last_good_checkpoint = ckpt_path
                steps_since_save = 0
                print(f"\n  💾 {ckpt_path.name} | loss={lm_loss.item():.4f} | ghost={loss_dict.get('ghost',0):.4f} | jepa={loss_dict.get('jepa',0):.4f}")

            # ═══ HEARTBEAT ═══
            if current_step % 100 == 0:
                elapsed = time.time() - started_at
                tok_s = (current_step - start_step) * cfg.batch_size * cfg.block_size / max(elapsed, 0.001)
                cpu_rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
                moe_summary = summarize_moe_stats(output.moe_stats)
                vram_gb = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                print(f"[{current_step:>6d}] lm={lm_loss.item():.4f} ghost={loss_dict.get('ghost',0):.4f} jepa={loss_dict.get('jepa',0):.4f} tok/s={tok_s:.0f} vram={vram_gb:.1f}GB ram={cpu_rss:.1f}GB {moe_summary}")

            # ═══ EXPERT ROTATION ═══
            if current_step % 1000 == 0:
                result = expert_router.promote_demote()
                print(f"  🔄 Expert rotation: promoted={result['promoted']}, demoted={result['demoted']}")

    except KeyboardInterrupt:
        print("\n\nInterrompido.")

    # ═══ FINAL ═══
    elapsed = time.time() - started_at
    ckpt_path = trainer.save_checkpoint(versioned=True)
    print(f"\n{'='*60}")
    print(f"  CONCLUÍDO: {trainer.state.step} steps em {elapsed/3600:.1f}h")
    print(f"  Checkpoint final: {ckpt_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
