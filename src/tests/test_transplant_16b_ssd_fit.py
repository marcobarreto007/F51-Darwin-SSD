from __future__ import annotations

import torch

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.layers import SSDMixerOnly
from f51_darwin.transplant_16b.ssd_fit import (
    SSDFitConfig,
    TransitionBatch,
    fit_ssd_block,
)


def test_fit_beats_paired_random_initialization() -> None:
    torch.manual_seed(19)
    config = DarwinXConfig(
        d_model=16,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=64,
        fine_experts=2,
        shared_experts=1,
        experts_per_token=1,
        fine_expert_hidden_dim=8,
        shared_expert_hidden_dim=8,
        ssm_state=4,
        scan_chunk_size=8,
    )
    teacher = SSDMixerOnly(config)
    inputs = torch.randn(8, 12, 16)
    with torch.no_grad():
        targets = teacher(inputs)
    candidate = SSDMixerOnly(config)

    report = fit_ssd_block(
        candidate,
        TransitionBatch(
            inputs[:6],
            targets[:6],
            inputs[6:],
            targets[6:],
        ),
        SSDFitConfig(steps=40, learning_rate=1e-2, seed=19),
    )

    assert report.holdout_mse < report.random_holdout_mse
    assert report.train_mse < report.initial_train_mse
    assert report.finite
