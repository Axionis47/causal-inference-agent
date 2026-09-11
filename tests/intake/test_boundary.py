"""Intake's independent entry point and per-invocation ownership."""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
from dataclasses import replace
from importlib.util import resolve_name
from pathlib import Path
from typing import Any

from causal.intake.catalog import CatalogStore
from causal.shared.events import EventEmitter
from causal.shared.persistence import ArtifactCommitter, ObjectStore, ProductStore
from tests.infrastructure import requires_docker
from tests.intake.conftest import FrozenKaggleClient
from tests.intake.test_coordinator import CLASSES, NOW, REGISTRY, make_coordinator, submission

ROOT = Path(__file__).resolve().parents[2]
INTAKE = ROOT / "src" / "causal" / "intake"
OTHER_STAGES = tuple(f"causal.{name}" for name in (
    "cli", "design", "analysis", "preparation", "presentation", "runtime"))
MODEL_MODULES = ("causal.shared.gateway", "langgraph", "google.genai")


def test_intake_source_imports_only_its_own_stage_and_shared_services() -> None:
    violations: list[str] = []
    for path in sorted(INTAKE.rglob("*.py")):
        package = ".".join(path.relative_to(ROOT / "src").parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = resolve_name("." * node.level + module, package)
                names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".")
                       for prefix in OTHER_STAGES + MODEL_MODULES):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "Intake imports another stage or a model:\n" + "\n".join(violations)


def test_public_entry_imports_without_downstream_stages_or_model_modules() -> None:
    """A fresh interpreter cannot conceal coupling behind modules pytest already loaded."""
    program = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
blocked = {OTHER_STAGES + MODEL_MODULES!r}
class RejectStageImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('intake imported ' + fullname)
        return None
sys.meta_path.insert(0, RejectStageImport())
from causal.intake.entry import IntakeDeps, run_intake
assert callable(run_intake)
assert IntakeDeps.__module__ == 'causal.intake.entry'
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False,
        cwd=ROOT, timeout=30)
    assert result.returncode == 0, result.stderr


class CountingKaggleClient(FrozenKaggleClient):
    def __init__(self) -> None:
        super().__init__()
        self.captures = 0

    def dataset_status(self, owner: str, slug: str) -> dict[str, object]:
        self.captures += 1
        return super().dataset_status(owner, slug)


@requires_docker
def test_public_entry_runs_with_intake_dependencies_and_publishes_handoff(
    conn: Any, object_store: ObjectStore,
) -> None:
    from causal.intake.entry import IntakeDeps, run_intake
    from causal.intake.outcome import open_handoff

    client = CountingKaggleClient()
    emitter = EventEmitter(io.StringIO())
    products = ProductStore(conn)
    catalog = CatalogStore(conn)
    deps = IntakeDeps(
        client=client, committer=ArtifactCommitter(object_store, products, REGISTRY, emitter),
        products=products, catalog=catalog, objects=object_store, registry=REGISTRY,
        field_classes=CLASSES, emitter=emitter, clock=lambda: NOW)

    result = run_intake(deps, submission("public-entry"))
    assert result.status == "usable" and not result.replayed
    assert result.outcome_artifact_id is not None
    assert client.captures == 1
    manifest = open_handoff(
        catalog, products, result.analysis_id, result.outcome_artifact_id,
        "sr:receiving-design", lambda: NOW)
    assert manifest.producing_stage_run_id == result.stage_run_id
    assert manifest.originating_outcome == "usable"
    assert len(manifest.entries) == 1
    assert manifest.entries[0].artifact_id == result.outcome_artifact_id
    assert conn.execute("SELECT DISTINCT stage FROM causal.stage_runs").fetchall() == [("intake",)]

    def unavailable_client() -> FrozenKaggleClient:
        raise AssertionError("replaying committed intake must not construct the source client")

    replayed = run_intake(replace(deps, client=unavailable_client), submission("public-entry"))
    assert replayed.replayed and client.captures == 1
    assert (replayed.analysis_id, replayed.stage_run_id, replayed.outcome_artifact_id) == (
        result.analysis_id, result.stage_run_id, result.outcome_artifact_id)
    assert conn.execute("SELECT count(*) FROM causal.stage_runs").fetchone() == (1,)


@requires_docker
def test_reused_coordinator_isolates_runs_and_replays_without_recapturing(
    conn: Any, object_store: ObjectStore,
) -> None:
    client = CountingKaggleClient()
    coordinator, sink = make_coordinator(conn, object_store, client)
    first_submission = submission("separate-one", "Does training raise earnings?")
    second_submission = submission("separate-two", "Does training affect employment?")
    results = []
    for request in (first_submission, second_submission):
        offset = len(sink.getvalue())
        result = coordinator.run(request)
        results.append(result)
        events = [json.loads(line) for line in sink.getvalue()[offset:].splitlines()]
        assert events and events[0]["event_name"] == "stage.started"
        assert events[-1]["event_name"] == "stage.completed"
        assert {event["analysis_id"] for event in events} == {result.analysis_id}
        assert {event["stage_run_id"] for event in events} == {result.stage_run_id}
        assert len({event["event_id"] for event in events}) == len(events)
        artifact_owners = conn.execute(
            "SELECT DISTINCT analysis_id, stage_run_id FROM causal.artifacts"
            " WHERE analysis_id = %s", (result.analysis_id,)).fetchall()
        assert artifact_owners == [(result.analysis_id, result.stage_run_id)]
        question_row = conn.execute(
            "SELECT question_artifact_id FROM catalog.runs WHERE analysis_id = %s",
            (result.analysis_id,)).fetchone()
        assert question_row is not None
        question = ProductStore(conn).load_envelope(question_row[0])
        assert json.loads(object_store.get(question.payload_locator))["question_text"] == (
            request.question_text)

    first, second = results
    assert first.analysis_id != second.analysis_id
    assert first.stage_run_id != second.stage_run_id
    assert first.outcome_artifact_id != second.outcome_artifact_id
    before_replay = sink.getvalue()
    replayed = coordinator.run(first_submission)
    assert replayed.replayed and client.captures == 2
    assert (replayed.analysis_id, replayed.stage_run_id, replayed.outcome_artifact_id) == (
        first.analysis_id, first.stage_run_id, first.outcome_artifact_id)
    assert sink.getvalue() == before_replay
    assert conn.execute("SELECT count(*) FROM causal.stage_runs").fetchone() == (2,)


@requires_docker
def test_source_client_construction_failure_finishes_refused_and_can_replay(
    conn: Any, object_store: ObjectStore,
) -> None:
    from causal.intake.entry import IntakeDeps, run_intake

    attempts = 0

    def unavailable_client() -> FrozenKaggleClient:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("authentication-detail-canary")

    sink = io.StringIO()
    emitter = EventEmitter(sink)
    products = ProductStore(conn)
    deps = IntakeDeps(
        client=unavailable_client,
        committer=ArtifactCommitter(object_store, products, REGISTRY, emitter),
        products=products, catalog=CatalogStore(conn), objects=object_store, registry=REGISTRY,
        field_classes=CLASSES, emitter=emitter, clock=lambda: NOW)
    request = submission("unavailable-source-client")
    result = run_intake(deps, request)
    assert result.status == "refused" and result.outcome_artifact_id is not None
    assert conn.execute(
        "SELECT run_state FROM causal.stage_runs WHERE stage_run_id = %s",
        (result.stage_run_id,)).fetchone() == ("completed",)
    events = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert any(event["event_name"] == "tool.failed" and event["error_code"] == "fetch_failed"
               for event in events)
    assert events[-1]["event_name"] == "stage.completed"
    assert "authentication-detail-canary" not in sink.getvalue()
    replayed = run_intake(deps, request)
    assert replayed.replayed and replayed.status == "refused" and attempts == 1
    assert replayed.outcome_artifact_id == result.outcome_artifact_id
