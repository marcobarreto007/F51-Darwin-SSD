from __future__ import annotations

import torch

from f51_darwin.transfer.ssd_mixer import DiscreteSSDMixer


def test_discrete_ssd_shape_and_causality() -> None:
    torch.manual_seed(7)
    mixer = DiscreteSSDMixer(d_model=16, n_heads=4, dropout=0.0)
    mixer.eval()
    x = torch.randn(2, 6, 16)
    y = mixer(x)
    changed = x.clone()
    changed[:, -1] += 100.0
    y_changed = mixer(changed)
    assert y.shape == x.shape
    torch.testing.assert_close(y[:, :-1], y_changed[:, :-1], atol=1e-5, rtol=1e-5)


def test_recurrent_and_materialized_mixer_are_equivalent() -> None:
    torch.manual_seed(11)
    mixer = DiscreteSSDMixer(d_model=12, n_heads=3, dropout=0.0)
    mixer.eval()
    x = torch.randn(2, 5, 12)
    normalized = mixer.input_norm(x)
    recurrent = mixer.mix_recurrent(normalized)
    matrix = mixer.materialize_mixer(normalized)
    value = mixer.project_values(normalized)
    materialized = torch.einsum("bhij,bhjd->bhid", matrix, value)
    torch.testing.assert_close(recurrent, materialized, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        recurrent,
        mixer.mix_parallel(normalized),
        atol=1e-5,
        rtol=1e-5,
    )


def test_projection_transfer_is_exact() -> None:
    torch.manual_seed(13)
    mixer = DiscreteSSDMixer(d_model=8, n_heads=2, dropout=0.0)
    qkv_weight = torch.randn(8, 24)
    qkv_bias = torch.randn(24)
    out_weight = torch.randn(8, 8)
    out_bias = torch.randn(8)
    mixer.load_gpt2_attention_projections(
        qkv_weight=qkv_weight,
        qkv_bias=qkv_bias,
        out_weight=out_weight,
        out_bias=out_bias,
    )
    q_weight, k_weight, v_weight = qkv_weight.split(8, dim=1)
    torch.testing.assert_close(mixer.c_proj.weight, q_weight.T)
    torch.testing.assert_close(mixer.b_proj.weight, k_weight.T)
    torch.testing.assert_close(mixer.x_proj.weight, v_weight.T)
    torch.testing.assert_close(mixer.out_proj.weight, out_weight.T)
