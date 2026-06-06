"""Idempotency-Key support for POST /jobs.

A client retrying a POST after a network blip should not create a
second job and double-spend the LLM workflow. The conventional shape is
the Idempotency-Key header: when present, we cache (api_key_hash, key)
to the resulting job_id and return the original job on a subsequent
matching request.

In-memory single-instance store with TTL eviction. For multi-instance
deploys, swap the store for the Redis-backed equivalent gated by
settings.redis_enabled. Today the store is per-process; that's fine
because POSTs route through one instance per session and TTL is short.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Entry:
    job_id: str
    expires_at: float


class IdempotencyStore:
    """Thread-safe LRU+TTL store keyed by (api_key_hash, idempotency_key)."""

    def __init__(self, max_entries: int = 10_000, ttl_seconds: int = 24 * 3600):
        self._max = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()

    @staticmethod
    def _hash_api_key(api_key: str | None) -> str:
        """Hash so we never store the raw key. Empty key gets a stable bucket."""
        if not api_key:
            return "anonymous"
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    def lookup(self, api_key: str | None, idempotency_key: str) -> str | None:
        """Return cached job_id if (api_key_hash, idempotency_key) is fresh."""
        if not idempotency_key:
            return None
        bucket = (self._hash_api_key(api_key), idempotency_key)
        with self._lock:
            entry = self._entries.get(bucket)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                self._entries.pop(bucket, None)
                return None
            # Refresh LRU position
            self._entries.move_to_end(bucket)
            return entry.job_id

    def remember(self, api_key: str | None, idempotency_key: str, job_id: str) -> None:
        """Store the (api_key_hash, idempotency_key) -> job_id mapping."""
        if not idempotency_key:
            return
        bucket = (self._hash_api_key(api_key), idempotency_key)
        with self._lock:
            self._entries[bucket] = _Entry(
                job_id=job_id, expires_at=time.monotonic() + self._ttl
            )
            self._entries.move_to_end(bucket)
            # Evict expired and oldest until under cap
            now = time.monotonic()
            for k in list(self._entries):
                if self._entries[k].expires_at < now:
                    self._entries.pop(k)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def reset(self) -> None:
        """Clear all entries (test-only)."""
        with self._lock:
            self._entries.clear()


_store: IdempotencyStore | None = None


def get_idempotency_store() -> IdempotencyStore:
    """Module-level singleton; lazy-initialized for test isolation."""
    global _store
    if _store is None:
        _store = IdempotencyStore()
    return _store
