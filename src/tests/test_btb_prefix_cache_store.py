from __future__ import annotations

from research.btb.prefix_cache_store import SharedPrefixCacheStore, CachedPrefixEntry


class DummyCache:
    """Mock KVCache para testes sem dependencia de torch."""

    def __init__(self, seq_len: int = 128):
        self._seq_len = seq_len

    def checkpoint_state(self) -> dict:
        return {
            "version": 1,
            "seq_len": self._seq_len,
            "batch_size": 1,
            "ssd_states": {},
            "attn_kv": {},
        }

    def restore_state(self, state: dict, target_device=None) -> None:
        self._seq_len = state.get("seq_len", 0)


def test_prefix_cache_store_put_get():
    store = SharedPrefixCacheStore(max_entries=10)
    source = DummyCache(seq_len=256)
    prefix_hash = "abc123hash"

    assert not store.has_prefix(prefix_hash)

    store.store_prefix(prefix_hash, 256, source)
    assert store.has_prefix(prefix_hash)

    target = DummyCache(seq_len=0)
    success = store.restore_to_cache(prefix_hash, target)
    assert success is True
    assert target._seq_len == 256


def test_prefix_cache_store_lru_eviction():
    store = SharedPrefixCacheStore(max_entries=2)
    c1 = DummyCache(100)
    c2 = DummyCache(200)
    c3 = DummyCache(300)

    store.store_prefix("h1", 100, c1)
    store.store_prefix("h2", 200, c2)

    # Acessa h1 para torna-lo mais recente
    t = DummyCache()
    store.restore_to_cache("h1", t)

    # Adiciona h3 -> h2 deve ser despejado (LRU)
    store.store_prefix("h3", 300, c3)

    assert store.has_prefix("h1")
    assert not store.has_prefix("h2")
    assert store.has_prefix("h3")
