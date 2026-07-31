"""Legacy entrypoint retained without pretending printed output is a test.

The causal assertions live in ``tests/test_neuroendocrine_moe.py``. Keeping this
small pointer avoids breaking old notes that mention the filename while also
preventing an accidental CUDA allocation from a top-level ad-hoc script.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Run: python -m pytest -q tests/test_neuroendocrine_moe.py"
    )


if __name__ == "__main__":
    main()
