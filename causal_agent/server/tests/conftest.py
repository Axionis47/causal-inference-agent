"""A server over a temp root with the desk's scripted model. No Vertex calls, nothing written under data/."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from causal_agent.common import config
from causal_agent.common.llm import set_decider, set_llm
from causal_agent.explore import client as XC
from causal_agent.explore import server as XS
from causal_agent.profile import data as D
from causal_agent.profile import datasets as DSI
from causal_agent.server.app import create_app
from causal_agent.server.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
QUESTION = "Did completing the prep course raise math scores?"


@pytest.fixture
def settings(tmp_path, monkeypatch, artifact_repository) -> Settings:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "datasets.yaml").write_text("# test index\n{}\n")
    monkeypatch.setattr(DSI, "ROOT", tmp_path)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    XC.set_app(XS.app("http://explore/"))  # the exploration agent in process: a test never starts a process or calls a model
    yield Settings(root=tmp_path, run_root=tmp_path / "runs", dist=tmp_path / "dist")
    XC.set_app(None)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    from causal_agent.server.conversations.routes import authorize_run, bind

    app.dependency_overrides[authorize_run] = lambda: None
    app.dependency_overrides[bind] = lambda: {}  # HTTP projection unit seam; ownership uses real storage tests
    with TestClient(app) as c:
        yield c
    app.state.sessions.pool.shutdown(wait=True)
    set_llm(None)
    set_decider(None)
    D.clear()


def pytest_addoption(parser):
    parser.addoption("--ingestion-integration", action="store_true", help="Require real ingestion tests; provision isolated Docker services")
    parser.addoption("--ingestion-mutation", default="", help="Deliberate defect for the ingestion mutation runner only")


def pytest_configure(config):
    config.addinivalue_line("markers", "ingestion: requires harness-owned PostgreSQL and private S3")


def pytest_collection_modifyitems(config, items):
    selected = [item for item in items if item.get_closest_marker("ingestion")]
    if config.getoption("--ingestion-integration"):
        if not selected:
            raise pytest.UsageError("Integration requested but no ingestion integration cases collected")
    else:
        if config.getoption("--ingestion-mutation"):
            raise pytest.UsageError("Mutation runs require --ingestion-integration")
        for item in selected:
            item.add_marker(pytest.mark.skip(reason="Run make test-ingestion to provision isolated storage"))


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--ingestion-integration"):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter and reporter.stats.get("skipped"):
            session.exitstatus = 1  # A required integration run may never turn green through skips.


@pytest.fixture(scope="session")
def ingestion_cluster():
    from causal_agent.server.tests.ingestion_resources import cluster

    with cluster() as settings:
        yield settings


@pytest.fixture
def storage(ingestion_cluster, monkeypatch, request):
    from causal_agent.server.tests.ingestion_mutations import apply_mutation
    from causal_agent.server.tests.ingestion_resources import database_and_bucket

    with database_and_bucket(ingestion_cluster, monkeypatch) as service:
        apply_mutation(request.config.getoption("--ingestion-mutation"), service, monkeypatch)
        yield service


@pytest.fixture
def isolated_client(storage, settings, monkeypatch):
    from uuid import uuid4

    from causal_agent.common import llm
    from causal_agent.lane import episode
    from causal_agent.profile import profiler
    from causal_agent.server.app import create_app
    from causal_agent.server.conversations.manager import Manager
    from causal_agent.server.ingestion.routes import store

    def forbidden(*args, **kwargs):
        raise AssertionError("Ingestion invoked a profiler, desk or model entrypoint")

    for module, name in [(llm, "decide"), (episode, "run_episode"), (profiler, "profile"), (Manager, "send")]:
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("DEV_USER_ID", str(uuid4()))
    app = create_app(settings)
    app.dependency_overrides[store] = lambda: storage  # real isolated storage, identity is NOT overridden
    with TestClient(app) as client:
        yield client
    app.state.sessions.pool.shutdown(wait=True)


@pytest.fixture
def ingestion_client(isolated_client, monkeypatch):
    from causal_agent.common import llm

    def forbidden(*args, **kwargs):
        raise AssertionError("Ingestion invoked a model entrypoint")

    monkeypatch.setattr(llm, "structured", forbidden)
    return isolated_client
