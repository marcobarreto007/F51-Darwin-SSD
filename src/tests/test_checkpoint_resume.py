from dataclasses import replace
from pathlib import Path

import torch

from f51_darwin.config import DarwinConfig
from f51_darwin.data import CausalLMDataLoader, load_text_documents, tokenize_documents
from f51_darwin.model import F51DarwinModel
from f51_darwin.tokenizer import F51BPETokenizer, MIN_BPE_VOCAB_SIZE
from f51_darwin.training import BaseTrainer, BaseTrainingConfig


def test_checkpoint_save_and_resume(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    documents = load_text_documents(root / "src" / "tests" / "fixtures" / "corpus")
    tokenizer = F51BPETokenizer.train(documents, vocab_size=MIN_BPE_VOCAB_SIZE)
    token_ids = tokenize_documents(documents, tokenizer)

    model_config = DarwinConfig(
        model_name="F51-Darwin-SSD-ResumeTest",
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        d_model=32,
        n_layers=4,
        n_heads=4,
        mlp_ratio=2,
    )
    training_config = BaseTrainingConfig.from_mapping(
        {
            "training": {
                "batch_size": 2,
                "block_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "max_steps": 4,
                "eval_every": 2,
                "save_every": 2,
                "grad_clip": 1.0,
                "seed": 51,
            },
            "replay": {"capacity": 32, "sample_size": 4, "seed_every": 1},
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "run_id": "resume_test",
        }
    )
    device = torch.device("cpu")
    data_loader = CausalLMDataLoader(
        token_ids,
        block_size=training_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        device=device,
    )
    model = F51DarwinModel(model_config)
    trainer = BaseTrainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        data_loader=data_loader,
        device=device,
        tokenizer_path=tmp_path / "tokenizer",
        project_root=tmp_path,
    )
    trainer.seed_replay_from_batch(data_loader.next_batch())
    trainer.train_step(data_loader.next_batch())
    trainer.state.step = 1
    checkpoint_path = trainer.save_checkpoint(versioned=True)
    assert checkpoint_path.exists()
    assert len(trainer.replay_buffer) >= 1

    resumed_model = F51DarwinModel(model_config)
    resumed = BaseTrainer.resume_from_checkpoint(
        checkpoint_path,
        project_root=tmp_path,
        model=resumed_model,
        model_config=model_config,
        training_config=training_config,
        data_loader=data_loader,
        device=device,
        tokenizer_path=tmp_path / "tokenizer",
    )
    assert resumed.state.step == trainer.state.step
    assert len(resumed.replay_buffer) == len(trainer.replay_buffer)

    resumed.training_config = replace(training_config, max_steps=trainer.state.step + 2)
    summary = resumed.run(max_steps=trainer.state.step + 2, eval_every=1, save_every=2)
    assert summary["steps_logged"] >= 1
    assert (tmp_path / "checkpoints" / "latest.json").exists()
