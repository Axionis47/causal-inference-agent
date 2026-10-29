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


@pytest.fixture
def artifact_repository():
    """Unit seam only. Real publication acceptance uses PostgreSQL and private S3."""
    from uuid import uuid4

    from causal_agent.memory import artifacts as A

    class Repository:
        owner = "test-owner"
        conversation = "test-chat"
        context = ""

        def __init__(self):
            self.records = {}
            self.files = {}

        def begin(self, dataset, *args):
            return uuid4().hex, uuid4().hex, "test-version"

        def finish(self, *args):
            pass

        def external_task(self, attempt, task_id):
            self.task_id = task_id

        def publish(self, a, directory):
            self.records[a.id] = a
            self.files[a.id] = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}

        def load(self, dataset, aid):
            a = self.records[aid]
            if a.dataset != dataset:
                raise ValueError("wrong dataset")
            return a

        def list(self, dataset):
            return [a for a in self.records.values() if a.dataset == dataset]

        def materialize(self, a, directory):
            from causal_agent.common import trace

            directory.mkdir(parents=True, exist_ok=True)
            for name, data in self.files[a.id].items():
                trace.write(directory / name, data)

    repo = Repository()
    with A.using(repo):
        yield repo
