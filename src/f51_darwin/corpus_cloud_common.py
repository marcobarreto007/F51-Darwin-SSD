from __future__ import annotations


def safe_batch_name(batch_name: str) -> str:
    """Return a shell/path-safe batch identifier used by cloud corpus stages."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in batch_name)
    if not safe:
        raise ValueError("batch_name cannot be empty")
    return safe
