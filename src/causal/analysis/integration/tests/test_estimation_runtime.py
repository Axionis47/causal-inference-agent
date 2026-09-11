# Runtime dispatch into PRD-004, its broad-exception guard, and the CLI render (T-029 §1.1,
# §1.2). The prepared frame every case starts from is built by the real PRD-003 coordinator.

from __future__ import annotations

import io
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import psycopg
import pytest

from causal.analysis.integration import nodes
from causal.analysis.integration.tests.support.coordinator import Gateway, prepared_outcome
from causal.cli import render
from causal.cli.main import CliResultV1
from causal.runtime import failures
from causal.shared import events, persistence
from tests.infrastructure import MIGRATIONS, requires_docker
from tests.preparation.support import REGISTRY, ROOT

pytestmark = requires_docker

ANALYSIS = "an-dispatch"
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
RUN_STATE = "SELECT state FROM estimation.estimation_runs WHERE stage_run_id = %s"


class Fixture:
    """One database, one prepared RCT frame, and the design deps the dispatch reads through."""

    def __init__(self, conn: Any, objects: Any) -> None:
        self.conn, self.sink = conn, io.StringIO()
        emitter = events.EventEmitter(self.sink)
        products = persistence.ProductStore(conn)
        self.gateway = Gateway()
        self.deps = cast(Any, SimpleNamespace(
            conn=conn, products=products, objects=objects, registry=REGISTRY, emitter=emitter,
            clock=lambda: NOW, gateway=self.gateway,
            committer=persistence.ArtifactCommitter(objects, products, REGISTRY, emitter)))
        self.prepared = prepared_outcome(conn, objects, self.sink, ANALYSIS, {})

    def est(self) -> Any:
        return failures.estimation_deps(self.deps, ROOT / "registries", ROOT)

    def estimate(self, analysis_id: str = ANALYSIS) -> Any:
        return failures.estimate(self.deps, self.est(), analysis_id, design_revision=1)


@pytest.fixture()
def fixture(postgres_dsn: str, minio_s3: dict[str, Any]) -> Iterator[Fixture]:
    admin = psycopg.connect(postgres_dsn, autocommit=True)
    name = f"test_{uuid.uuid4().hex[:10]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    admin.close()
    conn = psycopg.connect(postgres_dsn.rsplit("/", 1)[0] + f"/{name}", autocommit=True)
    persistence.apply_migrations(conn, MIGRATIONS)
    yield Fixture(conn, persistence.ObjectStore(minio_s3["client"], minio_s3["bucket"]))
    conn.close()


def test_the_registered_adapter_map_binds_every_estimation_pack() -> None:
    """D-089a: the four PRD-004 adapters are runtime dependencies, not test fixtures."""
    assert set(failures.ADAPTERS) == {"randomized_experiment", "aipw", "did", "sharp_rdd"}


def test_a_prepared_frame_dispatches_estimation_and_then_reports_it(fixture: Fixture) -> None:
    """T-029 §1.1: `prepared` opens revision 1, and a finished revision is reported, not re-run."""
    done = fixture.estimate()
    assert (done.status, done.error_code) == ("complete", None)
    assert done.stage_run_id == f"es:{ANALYSIS}:1" and done.handoff_id is not None
    assert fixture.conn.execute(RUN_STATE, (done.stage_run_id,)).fetchone() == ("completed",)
    calls = len(fixture.gateway.calls)
    again = fixture.estimate()
    assert (again.stage_run_id, again.status) == (done.stage_run_id, done.status)
    assert len(fixture.gateway.calls) == calls  # D-035: nothing was estimated a second time


def test_an_analysis_without_a_prepared_frame_is_refused(fixture: Fixture) -> None:
    """§4: PRD-004 opens on a recorded prepared handoff and on nothing else."""
    done = fixture.estimate("an-nothing")
    assert (done.status, done.error_code) == ("failed", "preparation_not_prepared")
    raised = [line for line in fixture.sink.getvalue().splitlines()
              if '"event_name":"blocker.raised"' in line and "an-nothing" in line]
    assert len(raised) == 1 and '"preparation_not_prepared"' in raised[0]


def poisoned(self: Any, state: Any) -> dict[str, Any]:
    raise RuntimeError("a node died with an untyped exception")


def test_a_crashing_node_returns_a_typed_exit_and_one_blocker(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-069: a broad exception inside PRD-004 leaves a typed result, never a traceback."""
    monkeypatch.setattr(nodes.EstimationNodes, "entry_node", poisoned)
    done = fixture.estimate()
    assert (done.status, done.error_code) == ("failed", "internal_error")
    assert fixture.conn.execute(RUN_STATE, (done.stage_run_id,)).fetchone() == ("failed",)
    raised = [line for line in fixture.sink.getvalue().splitlines()
              if '"event_name":"blocker.raised"' in line and '"internal_error"' in line]
    assert len(raised) == 1 and '"detail":"RuntimeError"' in raised[0]
    assert "untyped exception" not in raised[0]


def test_a_crashed_live_revision_is_replayed_as_the_next_one(fixture: Fixture) -> None:
    """D-069a: a row left `running` by a killed process is re-entered at revision n+1 (D-035)."""
    first = fixture.estimate()
    fixture.conn.execute("UPDATE estimation.estimation_runs SET state = 'running'"
                         " WHERE stage_run_id = %s", (first.stage_run_id,))
    second = fixture.estimate()
    assert second.stage_run_id == f"es:{ANALYSIS}:2" and second.status == first.status
    assert fixture.conn.execute(
        "SELECT count(*) FROM estimation.estimation_runs WHERE analysis_id = %s",
        (ANALYSIS,)).fetchone() == (2,)


@pytest.mark.parametrize(("status", "code", "expected"), [
    ("complete", None, ["Estimation: complete", "Next command: causal presentation an-1"]),
    ("invalidated", "not_reportable", ["Estimation: invalidated", "Reason: not_reportable"]),
    ("design_conflict", "capacity_exceeded",
     ["Estimation: design_conflict", "Conflict: capacity_exceeded",
      "Next command: re-run design as revision 2"])])
def test_the_cli_renders_each_estimation_terminal_and_its_next_command(
    status: str, code: str | None, expected: list[str]
) -> None:
    """T-029 §1.2: the one result document names the outcome, its reason, and what follows."""
    printed = render.render_human(CliResultV1(
        cli_invocation_id="cli:1", command_name="run", analysis_id="an-1",
        stage_run_id="es:an-1:1", status="completed", message_key="stage.finished",
        error_code=code, message_args={"design_revision": 1, "design_status": status}))
    assert all(line in printed.splitlines() for line in expected), printed
