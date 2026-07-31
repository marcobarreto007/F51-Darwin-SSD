import pytest

from f51_darwin.config import DEFAULT_MODULE_STATES, DarwinConfig


def test_default_config_contract() -> None:
    config = DarwinConfig()
    assert config.init == "random"
    assert config.ssd_attention_ratio == "3:1"
    assert config.module_states == DEFAULT_MODULE_STATES
    assert config.attention_layer_indices == (3, 7)
    assert config.ssd_layer_indices == (0, 1, 2, 4, 5, 6)


def test_config_rejects_non_random_init() -> None:
    with pytest.raises(ValueError, match="random initialization"):
        DarwinConfig(init="pretrained")


def test_config_requires_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DarwinConfig(d_model=30, n_heads=8)


def test_from_mapping_coerces_string_values() -> None:
    """Bug #3 — checkpoints serialise everything as strings.

    ``from_mapping`` must coerce back to the dataclass's native types so the
    ``__post_init__`` validators see real ints/floats/bools.
    """
    raw = {
        "vocab_size": "58162",
        "context_length": "1024",
        "d_model": "384",
        "n_layers": "8",
        "n_heads": "6",
        "mlp_ratio": "4",
        "dropout": "0.0",
        "weight_tying": "true",
        # strings that are already strings stay as strings
        "model_name": "F51-from-ckpt",
    }

    config = DarwinConfig.from_mapping(raw)

    assert config.vocab_size == 58162 and isinstance(config.vocab_size, int)
    assert config.context_length == 1024 and isinstance(config.context_length, int)
    assert config.d_model == 384 and isinstance(config.d_model, int)
    assert config.n_layers == 8 and isinstance(config.n_layers, int)
    assert config.n_heads == 6 and isinstance(config.n_heads, int)
    assert config.mlp_ratio == 4 and isinstance(config.mlp_ratio, int)
    assert config.dropout == 0.0 and isinstance(config.dropout, float)
    assert config.weight_tying is True and isinstance(config.weight_tying, bool)
    assert config.model_name == "F51-from-ckpt"


def test_from_mapping_coerces_bool_variants() -> None:
    falsy = DarwinConfig.from_mapping({"weight_tying": "false"})
    assert falsy.weight_tying is False

    falsy_zero = DarwinConfig.from_mapping({"weight_tying": "0"})
    assert falsy_zero.weight_tying is False

    truthy_one = DarwinConfig.from_mapping({"weight_tying": "1"})
    assert truthy_one.weight_tying is True


def test_from_mapping_keeps_module_states_tuple() -> None:
    raw = {"module_states": list(DEFAULT_MODULE_STATES)}
    config = DarwinConfig.from_mapping(raw)
    assert config.module_states == DEFAULT_MODULE_STATES
    assert isinstance(config.module_states, tuple)

