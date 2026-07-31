from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_helper():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "train_cloud.py"
    spec = importlib.util.spec_from_file_location("train_cloud_script", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_resolved_token_ids


def test_train_cloud_loads_small_token_bin_in_ram(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    token_path = data / "tokens_chunk_aa"
    np.asarray([1, 2, 3, 4], dtype=np.int32).tofile(token_path)

    token_ids, selected = _load_helper()(tmp_path)

    assert selected == token_path
    assert isinstance(token_ids, np.ndarray)
    assert not isinstance(token_ids, np.memmap)
    assert token_ids.tolist() == [1, 2, 3, 4]


def test_train_cloud_uses_memmap_above_ram_limit(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    token_path = data / "tokens_chunk_aa"
    np.asarray([1, 2, 3, 4], dtype=np.int32).tofile(token_path)

    token_ids, selected = _load_helper()(tmp_path, ram_limit_bytes=4)

    assert selected == token_path
    assert isinstance(token_ids, np.memmap)
    assert token_ids[:4].tolist() == [1, 2, 3, 4]
