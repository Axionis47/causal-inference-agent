"""Runtime dispatch into PRD-003, the broad-exception guard, and the CLI render (T-019 §1.4)."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest

from causal.cli import render
from causal.cli.main import CliResultV1, main
from causal.design.contracts import ApprovalDecision
from causal.preparation import nodes
from causal.runtime import composition, failures
from tests.conftest import requires_docker
from tests.runtime.test_composition import binding, runtime, start  # noqa: F401

PREPARATION_ROW = "SELECT state FROM preparation.preparation_runs WHERE analysis_id = %s"


def approve(built: composition.CausalRuntime) -> tuple[str, str]:
    """Intake, design, and the approval that makes `causal run` a preparation command."""
    made, opened = start(built)
    done = built.approve_design(made.analysis_id, binding(opened), ApprovalDecision.APPROVED,
                                "k-approve")
    assert done.status == "approved"
    return made.analysis_id, opened.stage_run_id


@requires_docker
def test_run_after_approval_dispatches_preparation_and_status_follows(
    runtime: composition.CausalRuntime, conn: Any  # noqa: F811
) -> None:
    """D-069b: an approved design names `run`, which starts PRD-003's first revision."""
    analysis_id, stage_run = approve(runtime)
    assert runtime.status(analysis_id) == composition.StatusView(
        analysis_id=analysis_id, stage="design", state="completed", next_command="run")
    done = runtime.run(analysis_id, expected_stage_run=stage_run, idempotency_key="k-prepare")
    started = failures.latest_preparation_run(conn, analysis_id)
    assert started is not None and started.stage_run_id == f"pr:{analysis_id}:1"
    # T-013 Amendment 9 (D-077): a live design now binds its capacity check and parents its
    # pre-repair report, so the §4 gate admits the handoff and PRD-003 reaches its own terminal.
    outcome = failures.payload(runtime.deps, str(started.outcome_artifact_id))
    assert (started.state, outcome["status"]) == ("completed", "prepared")
    # T-029: one `run` does not stop at `prepared` — the same command opens PRD-004's revision.
    assert done.stage_run_id == f"es:{analysis_id}:1"
    assert runtime.status(analysis_id).stage == "estimation"


def poisoned(self: Any, state: Any) -> dict[str, Any]:
    raise RuntimeError("a node died with an untyped exception")


@requires_docker
def test_a_crashing_node_returns_a_typed_exit_and_one_blocker(
    runtime: composition.CausalRuntime, conn: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """D-069: a broad exception inside PRD-003 leaves a typed result, never a traceback."""
    analysis_id, stage_run = approve(runtime)
    monkeypatch.setattr(nodes.PreparationNodes, "entry_node", poisoned)
    printed = StringIO()
    code = main(["run", analysis_id, "--expected-stage-run", stage_run,
                 "--idempotency-key", "k-crash"], lambda: runtime, out=printed)
    assert code == 4 and "Traceback" not in printed.getvalue()
    assert conn.execute(PREPARATION_ROW, (analysis_id,)).fetchone() == ("failed",)
    raised = [line for line in runtime.config.event_log.read_text(encoding="utf-8").splitlines()
              if '"event_name":"blocker.raised"' in line]
    assert len(raised) == 1 and '"error_code":"internal_error"' in raised[0]
    assert "untyped exception" not in raised[0] and '"detail":"RuntimeError"' in raised[0]


def test_the_cli_renders_a_preparation_conflict_and_its_next_command() -> None:
    """§16 answers a conflict with a design revision, so the CLI names that command."""
    printed = render.render_human(CliResultV1(
        cli_invocation_id="cli:1", command_name="run", analysis_id="an-1",
        stage_run_id="pr:an-1:1", status="completed", message_key="stage.finished",
        error_code="unresolved_conflict",
        message_args={"design_revision": 1, "design_status": "design_conflict"}))
    assert "Preparation: design_conflict" in printed
    assert "Conflict: unresolved_conflict" in printed
    assert "Next command: re-run design as revision 2" in printed
