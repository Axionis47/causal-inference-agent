"""The committed stage-run rows, the PRD-003 dispatch, and every terminal failure.

T-014 Amendment 3 and T-019 §1.4. SC §1.1 lets no failure escape as an untyped traceback
and §7.1 pairs every one with a blocker and a terminal stage state, so the errors that
cross a harness boundary as plain exceptions are converted here — D-069 adds the
last-resort broad catch, so a crash inside any node still leaves a typed result behind.
The preparation stage keeps its wiring here too: `composition` sits at its module ceiling
and both stages read the same two run tables.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Final, NamedTuple

from psycopg import Connection

from causal.design import graph
from causal.preparation.diagnostics import load_preparation_diagnostics
from causal.preparation.harness import PreparationDeps, PreparationRunResult
from causal.preparation.nodes import run_preparation
from causal.preparation.operations import load_operation_registry
from causal.preparation.plans import load_preparation_packs
from causal.preparation.validators import load_validation_rules
from causal.shared import events, frames, gateway, persistence, tracing

__all__ = ["COMPONENT", "FAILED", "FAILED_OBSERVABILITY", "INTERNAL_ERROR", "LIVE", "VERSION",
           "DesignRun", "emit_blocker", "guard", "guard_preparation", "latest_design_run",
           "latest_preparation_run", "payload", "preparation_deps", "prepare"]

COMPONENT, VERSION = "cli-runtime", "cli-runtime.v1"
FAILED, FAILED_OBSERVABILITY = "failed", "failed_observability"
INTERNAL_ERROR: Final = "internal_error"
# Every stage-run state a command can still move; anything else is that stage's end (D-069b).
LIVE: Final = frozenset({"created", "tracing_preflight", "running", "waiting_for_user",
                         "stabilizing", "frozen", "preparing", "committing"})
_LATEST_RUN: Final = (
    "SELECT stage_run_id, graph_thread_id, design_revision, state, outcome_artifact_id"
    " FROM design.design_runs WHERE analysis_id = %s ORDER BY design_revision DESC LIMIT 1")
_TERMINAL_ROW: Final = ("UPDATE design.design_runs SET state = %s, updated_at = %s"
                        " WHERE stage_run_id = %s")
_LATEST_PREPARATION: Final = (
    "SELECT stage_run_id, graph_thread_id, preparation_revision, state, outcome_artifact_id"
    " FROM preparation.preparation_runs WHERE analysis_id = %s"
    " ORDER BY preparation_revision DESC LIMIT 1")
_PREPARATION_ROW: Final = ("UPDATE preparation.preparation_runs SET state = %s, updated_at = %s"
                           " WHERE stage_run_id = %s")


class DesignRun(NamedTuple):
    """One stage-run row — design or preparation — as the boundary a command acts on."""

    stage_run_id: str
    thread_id: str
    revision: int
    state: str
    outcome_artifact_id: str | None


def _stage_run(conn: Connection[Any], sql: str, analysis_id: str) -> DesignRun | None:
    row = conn.execute(sql, (analysis_id,)).fetchone()
    return None if row is None else DesignRun(
        str(row[0]), str(row[1]), int(row[2]), str(row[3]),
        None if row[4] is None else str(row[4]))


def latest_design_run(conn: Connection[Any], analysis_id: str) -> DesignRun | None:
    """The newest revision's row, or None when the analysis has no design run yet."""
    return _stage_run(conn, _LATEST_RUN, analysis_id)


def latest_preparation_run(conn: Connection[Any], analysis_id: str) -> DesignRun | None:
    """The newest preparation revision's row; both stage tables expose these five columns."""
    return _stage_run(conn, _LATEST_PREPARATION, analysis_id)


def payload(deps: graph.DesignDeps, artifact_id: str) -> dict[str, Any]:
    """One committed payload, read through the stores every stage commits through."""
    body: dict[str, Any] = json.loads(
        deps.objects.get(deps.products.load_envelope(artifact_id).payload_locator))
    return body


def preparation_deps(deps: graph.DesignDeps, registries: Path,
                     repo_root: Path) -> PreparationDeps:
    """PRD-003 makes no model call, so these are the design deps minus gateway and saver."""
    return PreparationDeps(
        conn=deps.conn, products=deps.products, objects=deps.objects, committer=deps.committer,
        registry=deps.registry, emitter=deps.emitter, clock=deps.clock,
        packs=load_preparation_packs(registries / "method-pack-preparation.v1.json",
                                     registries / "method-packs.v1.json"),
        operations=load_operation_registry(registries / "repair-operations.v1.json"),
        diagnostic_registry=load_preparation_diagnostics(
            registries / "preparation-diagnostics.v1.json"),
        rules=load_validation_rules(registries / "preparation-validation-rules.v1.json"),
        frames=frames.FrameStore(objects=deps.objects),
        method_packs_path=registries / "method-packs.v1.json", repo_root=repo_root)


def prepare(deps: graph.DesignDeps, prep: PreparationDeps, analysis_id: str,
            design: DesignRun) -> graph.DesignRunResult:
    """Run the PRD-003 revision the approved design's recorded handoff opens (T-019 §1.4)."""
    started = latest_preparation_run(deps.conn, analysis_id)
    if started is not None and started.state not in LIVE:
        return _reported(deps, analysis_id, design.revision, started)
    # D-069a: `run` holds the analysis lock, so a live row is a crashed attempt. Preparation
    # keeps no checkpoint, so the next attempt replays the committed artifacts from the top;
    # `preparation_runs` holds one row per revision, so that attempt is the next revision.
    revision = 1 if started is None else started.revision + 1
    stage_run_id = f"pr:{analysis_id}:{revision}"
    return _view(design.revision, guard_preparation(
        deps, "run", analysis_id, stage_run_id, revision, lambda: run_preparation(
            prep, analysis_id=analysis_id, stage_run_id=stage_run_id,
            design_outcome_artifact_id=str(design.outcome_artifact_id),
            preparation_revision=revision)))


def _reported(deps: graph.DesignDeps, analysis_id: str, design_revision: int,
              found: DesignRun) -> graph.DesignRunResult:
    """A finished revision is reported from its committed outcome, never re-run (D-035)."""
    body = {} if found.outcome_artifact_id is None else payload(deps, found.outcome_artifact_id)
    conflict = body.get("design_conflict") or {}
    return _view(design_revision, PreparationRunResult(
        status=str(body.get("status") or FAILED), analysis_id=analysis_id,
        stage_run_id=found.stage_run_id, thread_id=found.thread_id,
        preparation_revision=found.revision, outcome_artifact_id=found.outcome_artifact_id,
        conflict_code=None if not conflict else str(
            payload(deps, str(conflict["artifact_id"]))["conflict_code"]),
        error_code=body.get("error_code")))


def _view(design_revision: int, run: PreparationRunResult) -> graph.DesignRunResult:
    """One PRD-003 terminal in the single result shape the CLI renders (T-019 §1.4)."""
    # PRD-003 §16 answers a `design_conflict` with a new design revision, so the conflict
    # code travels as the refusal code and the revision named is the design one.
    return graph.DesignRunResult(
        status=run.status, analysis_id=run.analysis_id, stage_run_id=run.stage_run_id,
        thread_id=run.thread_id, design_revision=design_revision,
        outcome_artifact_id=run.outcome_artifact_id, handoff_id=run.handoff_id,
        refusal_code=run.conflict_code, error_code=run.error_code)


def emit_blocker(deps: graph.DesignDeps, command: str, analysis_id: str, code: str,
                 detail: str = "") -> None:
    """One `blocker.raised` per refusal or terminal failure, with its stable code (SC §1.1)."""
    # `detail` is the one dimension a blocker adds: the identity a stale command should have
    # named, or a crash's exception class name — never a message body (D-069, D-071).
    deps.emitter.emit(events.build_event(
        occurred_at_utc=deps.clock(), severity=events.Severity.ERROR,
        event_name="blocker.raised", event_id=f"evt:cli:{uuid.uuid4().hex}",
        analysis_id=analysis_id, stage=events.Stage.SYSTEM, error_code=code,
        stage_run_id=f"sr:cli:{command}", component_id=COMPONENT, component_version=VERSION,
        safe_dimensions={"blocked_operation": command} | ({"detail": detail} if detail else {})))


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
    except Exception as error:  # noqa: BLE001 -- D-069: no traceback ever escapes a command
        emit_blocker(deps, command, analysis_id, INTERNAL_ERROR, type(error).__name__)
        return _terminal(deps, analysis_id, FAILED, INTERNAL_ERROR)


def guard_preparation(deps: graph.DesignDeps, command: str, analysis_id: str, stage_run_id: str,
                      revision: int,
                      call: Callable[[], PreparationRunResult]) -> PreparationRunResult:
    """The same conversion for PRD-003, whose own nodes already type every other failure."""
    try:
        return call()
    except tracing.ObservabilityError as error:
        return _preparation_terminal(deps, analysis_id, stage_run_id, revision,
                                     FAILED_OBSERVABILITY, error.code)
    except Exception as error:  # noqa: BLE001 -- D-069: no traceback ever escapes a command
        emit_blocker(deps, command, analysis_id, INTERNAL_ERROR, type(error).__name__)
        return _preparation_terminal(deps, analysis_id, stage_run_id, revision, FAILED,
                                     INTERNAL_ERROR)


def _preparation_terminal(deps: graph.DesignDeps, analysis_id: str, stage_run_id: str,
                          revision: int, state: str, code: str) -> PreparationRunResult:
    """Close the crashed revision's rows; SC §4 leaves no run row live after a failure."""
    deps.conn.execute(_PREPARATION_ROW, (state, deps.clock(), stage_run_id))
    with suppress(persistence.PersistenceError):  # a crash before the stage run was opened
        deps.products.transition_stage_run(stage_run_id, state)
    return PreparationRunResult(
        status=state, analysis_id=analysis_id, stage_run_id=stage_run_id,
        thread_id=f"pt:{analysis_id}:{revision}", preparation_revision=revision, error_code=code)


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
