"""Fixtures every suite shares. A test never writes under the repository's .artifacts or data/memory: run artifacts and the
profile cache go under the test's own temporary directory, and the memory store's save is a no-op unless a test asks
for the real one."""

from __future__ import annotations

import pytest

from causal_agent.memory import store


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("PROFILE_CACHE_DIR", str(tmp_path / "profiles"))


@pytest.fixture
def no_memory_writes(monkeypatch):
    """The memory lives in this process: mining and interviewing never touch data/memory/."""
    monkeypatch.setattr(store, "save", lambda *a, **k: None)
