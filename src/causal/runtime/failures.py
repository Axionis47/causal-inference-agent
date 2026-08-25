"""The committed design-run row and its terminal failures (T-014 Amendment 3; D-062).

SC §1.1 lets no failure escape as an untyped traceback and §7.1 pairs every one with a
blocker and a terminal stage state. `GatewayError` and `ObservabilityError` cross the
harness boundary as plain exceptions, so the runtime converts them here.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Final, NamedTuple

from psycopg import Connection

from causal.design import graph
from causal.shared import events, gateway, tracing

__all__ = ["COMPONENT", "FAILED", "FAILED_OBSERVABILITY", "VERSION", "DesignRun",
           "emit_blocker", "guard", "latest_design_run"]

COMPONENT, VERSION = "cli-runtime", "cli-runtime.v1"
FAILED, FAILED_OBSERVABILITY = "failed", "failed_observability"
_LATEST_RUN: Final = (
    "SELECT stage_run_id, graph_thread_id, design_revision, state, outcome_artifact_id"
    " FROM design.design_runs WHERE analysis_id = %s ORDER BY design_revision DESC LIMIT 1")
_TERMINAL_ROW: Final = ("UPDATE design.design_runs SET state = %s, updated_at = %s"
                        " WHERE stage_run_id = %s")


class DesignRun(NamedTuple):
    """One `design.design_runs` row: the committed boundary a command acts on."""

    stage_run_id: str
    thread_id: str
    revision: int
    state: str
    outcome_artifact_id: str | None


def latest_design_run(conn: Connection[Any], analysis_id: str) -> DesignRun | None:
    """The newest revision's row, or None when the analysis has no design run yet."""
    row = conn.execute(_LATEST_RUN, (analysis_id,)).fetchone()
    return None if row is None else DesignRun(
        str(row[0]), str(row[1]), int(row[2]), str(row[3]),
        None if row[4] is None else str(row[4]))


def emit_blocker(deps: graph.DesignDeps, command: str, analysis_id: str, code: str) -> None:
    """One `blocker.raised` per refusal or terminal failure, with its stable code (SC §1.1)."""
    deps.emitter.emit(events.build_event(
        occurred_at_utc=deps.clock(), severity=events.Severity.ERROR,
        event_name="blocker.raised", event_id=f"evt:cli:{uuid.uuid4().hex}",
        analysis_id=analysis_id, stage=events.Stage.SYSTEM, error_code=code,
        stage_run_id=f"sr:cli:{command}", component_id=COMPONENT, component_version=VERSION,
        safe_dimensions={"blocked_operation": command}))


def guard(deps: graph.DesignDeps, command: str, analysis_id: str,
          call: Callable[[], graph.DesignRunResult]) -> graph.DesignRunResult:
    """Run one coordinator call, converting a terminal harness-boundary error to a result.

    The T-010 gateway emits `retry.scheduled` and `retry.exhausted` only and never a
    blocker, so the one raised here is the whole failure's single `blocker.raised`.
    """
    try:
        return call()
    except gateway.GatewayError as error:
        emit_blocker(deps, command, analysis_id, error.code)
        return _terminal(deps, analysis_id, FAILED, error.code)
    except tracing.ObservabilityError as error:
        # SC §10.2 keeps its own terminal state; committed artifacts stay committed.
        return _terminal(deps, analysis_id, FAILED_OBSERVABILITY, error.code)


def _terminal(deps: graph.DesignDeps, analysis_id: str, state: str,
              code: str) -> graph.DesignRunResult:
    """Close the design run and its stage run in `state`, then report that as the result."""
    # Both rows exist before any model or tracer call, so the fallback identity only names
    # a revision that failed before `run_design` recorded it.
    found = latest_design_run(deps.conn, analysis_id) or DesignRun(
        f"dr:{analysis_id}:1", "", 1, state, None)
    deps.conn.execute(_TERMINAL_ROW, (state, deps.clock(), found.stage_run_id))
    deps.products.transition_stage_run(found.stage_run_id, state)
    return graph.DesignRunResult(
        status=state, analysis_id=analysis_id, stage_run_id=found.stage_run_id,
        thread_id=found.thread_id, design_revision=found.revision,
        outcome_artifact_id=found.outcome_artifact_id, error_code=code)
