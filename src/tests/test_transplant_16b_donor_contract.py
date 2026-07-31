"""Regressoes do contrato de doador do transplante 1.6B.

Cada teste aqui corresponde a uma causa confirmada da falha do build de
2026-07-28 (bpb 8.935 contra random 4.908). Nenhuma delas aparecia em
shape de tensor, cobertura de ledger ou hash de checkpoint -- o build
inteiro passou por elas em silencio e so falhou no gate de ponta a ponta,
depois de pago o custo total.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.nn import functional as F

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.layers import SSDMixerOnly
from f51_darwin.transplant_16b.assembly import (
    _assert_donor_runtime_contract,
    _initialize_ssd,
)
from f51_darwin.transplant_16b.ssd_fit import (
    SSDFitConfig,
    TransitionBatch,
    fit_ssd_block,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET_CONFIG = ROOT / "src/configs/darwin_x_1.6b_smol_transplant.yaml"

# src/f51_darwin/ssm_core.py:239-240 -- oficial Mamba
DT_MIN, DT_MAX = 0.001, 0.1


def _target_config() -> DarwinXConfig:
    return DarwinXConfig.from_mapping(
        yaml.safe_load(TARGET_CONFIG.read_text(encoding="utf-8"))
    )


def test_target_config_matches_the_llama_family_donor() -> None:
    config = _target_config()
    # SmolLM2 e Llama-family: split-half. O default "interleaved" e GPT-J.
    assert config.rope_style == "llama"
    # block.py:66,84 escalam cada sub-camada por residual_scale. Um
    # transplante zero-shot precisa de 1.0; 1.4/sqrt(16) = 0.35 entregava
    # 35% do delta para o qual os pesos foram derivados.
    assert math.isclose(config.residual_scale, 1.0)
    # config.json do doador declara rms_norm_eps 1e-5.
    assert config.rms_norm_eps == 1e-5
    assert config.rms_norm_fp32
    assert config.rope_base_train == config.rope_base_infer == 130_000.0


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"rope_style": "interleaved"}, "rope_style"),
        ({"residual_scale_multiplier": 1.4}, "residual_scale"),
        ({"rms_norm_eps": 1e-6}, "rms_norm_eps"),
        ({"rms_norm_fp32": False}, "rms_norm_fp32"),
    ],
)
def test_donor_contract_fails_closed_on_each_field(
    override: dict[str, object], expected: str
) -> None:
    raw = yaml.safe_load(TARGET_CONFIG.read_text(encoding="utf-8"))
    raw.update(override)
    with pytest.raises(ValueError, match=expected):
        _assert_donor_runtime_contract(DarwinXConfig.from_mapping(raw))


def _tiny_ssd() -> tuple[SimpleNamespace, DarwinXConfig]:
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
        ssm_state=16,
        scan_chunk_size=8,
    )
    return SimpleNamespace(ssd=SSDMixerOnly(config)), config


def test_ssd_initialization_keeps_dt_inside_the_canonical_range() -> None:
    """dt e constante de tempo (A_bar = exp(dt*A)), nao escala de canal.

    A heuristica anterior (channel_scale.log().clamp(-4,4), com
    channel_scale normalizado para media 1) punha a maioria dos canais em
    softplus(0) = 0.6931. Medido no checkpoint de 2026-07-28: 100.0% dos
    46.080 canais dos 12 blocos SSD acima do teto 0.1, mediana 0.6650.
    """
    torch.manual_seed(5)
    block, config = _tiny_ssd()
    dim = config.d_model
    q, k, v, o = (torch.randn(dim, dim) for _ in range(4))

    _initialize_ssd(block, q, k, v, o)

    dt = F.softplus(block.ssd.state_dict()["ssm.dt_proj.bias"].float())
    assert dt.min() >= DT_MIN * 0.99
    assert dt.max() <= DT_MAX * 1.01


def test_conv_bias_is_not_the_dt_formula() -> None:
    """No checkpoint de 2026-07-28 os dois tensores eram byte-identicos."""
    torch.manual_seed(7)
    block, config = _tiny_ssd()
    dim = config.d_model
    q, k, v, o = (torch.randn(dim, dim) for _ in range(4))

    _initialize_ssd(block, q, k, v, o)

    state = block.ssd.state_dict()
    assert not torch.allclose(
        state["ssm.conv1d.bias"].float(),
        state["ssm.dt_proj.bias"].float(),
    )


def test_fit_gate_compares_against_the_supplied_baseline() -> None:
    """Sem `baseline`, o gate compara o candidato contra ele mesmo.

    No call site de producao o candidato ja chega mutado in-place por
    _initialize_ssd, entao o "random" do gate era o erro da propria
    inicializacao heuristica que ele deveria julgar.
    """
    torch.manual_seed(11)
    _, config = _tiny_ssd()
    inputs = torch.randn(8, 12, config.d_model)
    with torch.no_grad():
        targets = SSDMixerOnly(config)(inputs)
    transitions = TransitionBatch(
        inputs[:6], targets[:6], inputs[6:], targets[6:]
    )
    fit_config = SSDFitConfig(steps=5, learning_rate=1e-2, seed=11)

    candidate = SSDMixerOnly(config)
    without = fit_ssd_block(copy.deepcopy(candidate), transitions, fit_config)
    with_baseline = fit_ssd_block(
        copy.deepcopy(candidate),
        transitions,
        fit_config,
        baseline=SSDMixerOnly(config),
    )

    assert without.random_holdout_mse != with_baseline.random_holdout_mse
