from __future__ import annotations

import pytest

from f51_darwin.circuits.identity import canonical_json_bytes, canonical_sha256


def test_canonical_identity_is_order_independent() -> None:
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_identity_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})
