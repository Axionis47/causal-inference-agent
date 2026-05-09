"""Tests for the production-auth raise, idempotency store, and rate-limit coverage.

Three audit findings rolled into one PR:
    - 5.2: API_KEY optional in production (now: refuses startup)
    - 6.2: rate limiting only on POST /jobs (now: every route limited)
    - new: Idempotency-Key support on POST /jobs
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.idempotency import IdempotencyStore


class TestProductionAuthGuard:
    """Settings refuses to construct in production without an API_KEY."""

    def test_production_without_api_key_raises(self):
        from src.config.settings import Settings

        with pytest.raises(ValidationError) as excinfo:
            Settings(environment="production", api_key=None)

        # The error message should name the missing knob
        assert "API_KEY" in str(excinfo.value)

    def test_production_with_api_key_ok(self):
        from src.config.settings import Settings

        s = Settings(
            environment="production",
            api_key="real-key-from-secrets",
            kaggle_username="test",
            kaggle_key="test",
        )
        assert s.api_key_value == "real-key-from-secrets"

    def test_development_without_api_key_ok(self):
        from src.config.settings import Settings

        # Dev mode: missing API_KEY is fine; auth is intentionally disabled.
        s = Settings(environment="development", api_key=None)
        assert s.api_key_value is None


class TestIdempotencyStore:
    """Behavioral contract for the in-memory idempotency cache."""

    def test_lookup_miss_returns_none(self):
        store = IdempotencyStore()
        assert store.lookup("api-key", "fresh-key") is None

    def test_remember_then_lookup_hits(self):
        store = IdempotencyStore()
        store.remember("api-key", "k1", "job-abc")
        assert store.lookup("api-key", "k1") == "job-abc"

    def test_different_api_keys_isolated(self):
        store = IdempotencyStore()
        store.remember("alice", "k1", "job-alice")
        store.remember("bob", "k1", "job-bob")
        assert store.lookup("alice", "k1") == "job-alice"
        assert store.lookup("bob", "k1") == "job-bob"

    def test_anonymous_bucket_for_missing_api_key(self):
        store = IdempotencyStore()
        store.remember(None, "k1", "job-anon")
        assert store.lookup(None, "k1") == "job-anon"
        assert store.lookup("alice", "k1") is None

    def test_empty_idempotency_key_is_ignored(self):
        store = IdempotencyStore()
        store.remember("alice", "", "job-x")  # No-op
        assert store.lookup("alice", "") is None

    def test_ttl_expiration(self):
        store = IdempotencyStore(ttl_seconds=0)  # Immediate expiry
        store.remember("alice", "k1", "job-x")
        # Lookups after the TTL must miss
        assert store.lookup("alice", "k1") is None

    def test_lru_eviction_under_cap(self):
        store = IdempotencyStore(max_entries=2)
        store.remember("alice", "k1", "job-1")
        store.remember("alice", "k2", "job-2")
        store.remember("alice", "k3", "job-3")  # Evicts k1
        assert store.lookup("alice", "k1") is None
        assert store.lookup("alice", "k2") == "job-2"
        assert store.lookup("alice", "k3") == "job-3"


class TestRateLimitDecoratorCoverage:
    """Every job route carries a slowapi limit decorator (audit 6.2)."""

    def test_every_route_has_limit_metadata(self):
        from src.api.routes.jobs import router

        # slowapi attaches a `__wrapped__` and a Limit object to decorated handlers.
        # We check that each route's endpoint passes through the limiter.
        unlimited = []
        for route in router.routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                continue
            # slowapi-decorated handlers expose the original function via __wrapped__
            # and the rate-limit metadata via _rate_limit. We inspect for either.
            has_limit = (
                hasattr(endpoint, "__wrapped__")
                or "limiter" in repr(endpoint)
                or hasattr(endpoint, "_rate_limit")
            )
            if not has_limit:
                unlimited.append(route.path)

        assert not unlimited, (
            f"These routes are missing rate-limit decorators: {unlimited}"
        )
