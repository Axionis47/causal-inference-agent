"""The committed stage-run rows, the PRD-003 and PRD-004 dispatch, and every terminal failure.

T-014 Amendment 3, T-019 §1.4, and T-029 §1.1. SC §1.1 lets no failure escape as an untyped
traceback and §7.1 pairs every one with a blocker and a terminal stage state, so the errors
that cross a harness boundary as plain exceptions are converted here — D-069 adds the
last-resort broad catch, so a crash inside any node still leaves a typed result behind.
Both later stages keep their wiring here too, dependency construction included: `composition`
sits at its module ceiling and all three stages read the same five-column run rows.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Final, NamedTuple

from psycopg import Connection

from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import RESOURCE_ROOT as ANALYSIS_RESOURCES
from causal.analysis.integration.harness import EstimationDeps, EstimationRunResult
from causal.analysis.integration.nodes import run_estimation
from causal.analysis.integration.packs import load_estimation_packs
from causal.analysis.integration.walls import load_validation_rules as load_estimation_rules
from causal.analysis.methods.aipw import estimation as aipw
from causal.analysis.methods.did import estimation as did
from causal.analysis.methods.randomized import estimation as rct
from causal.analysis.methods.rdd import estimation as rdd
from causal.design import graph
from causal.preparation.diagnostics import load_preparation_diagnostics
from causal.preparation.harness import PreparationDeps, PreparationRunResult
from causal.preparation.nodes import run_preparation
from causal.preparation.operations import load_operation_registry
from causal.preparation.plans import load_preparation_packs
from causal.preparation.validators import load_validation_rules
from causal.shared import events, frames, gateway, persistence, tracing
from causal.shared.contracts import ArtifactRef

__all__ = ["ADAPTERS", "COMPONENT", "FAILED", "FAILED_OBSERVABILITY", "INTERNAL_ERROR", "LIVE",
           "NOT_PREPARED", "PREPARED", "VERSION", "DesignRun", "emit_blocker", "estimate",
           "estimation_deps", "guard", "guarded", "latest_design_run", "latest_estimation_run",
           "latest_preparation_run", "latest_stage_run", "payload", "preparation_deps",
           "prepare", "view"]

COMPONENT, VERSION = "cli-runtime", "cli-runtime.v1"
FAILED, FAILED_OBSERVABILITY = "failed", "failed_observability"
INTERNAL_ERROR: Final = "internal_error"
# PRD-004 opens only on a preparation revision that reached this terminal status (§4).
PREPARED, NOT_PREPARED = "prepared", "preparation_not_prepared"
# The four registered PRD-004 estimator adapters, bound by method id: §19 forbids a generic
# estimator tool and any adapter search at run time, so the map is fixed here.
ADAPTERS: Final[dict[str, Callable[[ArtifactRef], engine.EstimatorAdapter]]] = {
    "randomized_experiment": rct.RandomizedExperimentAdapter,
    "aipw": aipw.ObservationalAipwAdapter, "did": did.DifferenceInDifferencesAdapter,
    "sharp_rdd": rdd.SharpRegressionDiscontinuityAdapter}
# Every stage-run state a command can still move; anything else is that stage's end (D-069b).
LIVE: Final = frozenset({"created", "tracing_preflight", "running", "waiting_for_user",
                         "stabilizing", "frozen", "preparing", "committing"})
_LATEST_RUN: Final = (
    "SELECT stage_run_id, graph_thread_id, design_revision, state, outcome_artifact_id"
    " FROM design.design_runs WHERE analysis_id = %s ORDER BY design_revision DESC LIMIT 1")
_LATEST_PREPARATION: Final = (
    "SELECT stage_run_id, graph_thread_id, preparation_revision, state, outcome_artifact_id"
    " FROM preparation.preparation_runs WHERE analysis_id = %s"
    " ORDER BY preparation_revision DESC LIMIT 1")
_LATEST_ESTIMATION: Final = (
    "SELECT stage_run_id, graph_thread_id, estimation_revision, state, outcome_artifact_id"
    " FROM estimation.estimation_runs WHERE analysis_id = %s"
    " ORDER BY estimation_revision DESC LIMIT 1")
# Every stage table closes a crashed revision the same way; only the table name differs.
_STATE_ROW: Final = "UPDATE {} SET state = %s, updated_at = %s WHERE stage_run_id = %s"
DESIGN_TABLE, PREPARATION_TABLE = "design.design_runs", "preparation.preparation_runs"
ESTIMATION_TABLE: Final = "estimation.estimation_runs"
# The two later stages `status` reads, newest revision first; the earlier one is design.
_STAGE_SQL: Final = {"preparation": _LATEST_PREPARATION, "estimation": _LATEST_ESTIMATION}


class DesignRun(NamedTuple):
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
    return _stage_run(conn, _LATEST_RUN, analysis_id)


def latest_preparation_run(conn: Connection[Any], analysis_id: str) -> DesignRun | None:
    return _stage_run(conn, _LATEST_PREPARATION, analysis_id)


def latest_estimation_run(conn: Connection[Any], analysis_id: str) -> DesignRun | None:
    return _stage_run(conn, _LATEST_ESTIMATION, analysis_id)


def latest_stage_run(conn: Connection[Any], stage: str, analysis_id: str) -> DesignRun | None:
    return _stage_run(conn, _STAGE_SQL[stage], analysis_id)


def payload(deps: graph.DesignDeps, artifact_id: str) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(
        deps.objects.get(deps.products.load_envelope(artifact_id).payload_locator))
    return body


def preparation_deps(deps: graph.DesignDeps, registries: Path,
                     repo_root: Path) -> PreparationDeps:
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


def estimation_deps(deps: graph.DesignDeps, registries: Path,
                    repo_root: Path) -> EstimationDeps:
    return EstimationDeps(
        conn=deps.conn, products=deps.products, objects=deps.objects, committer=deps.committer,
        registry=deps.registry, emitter=deps.emitter, clock=deps.clock,
        packs=load_estimation_packs(ANALYSIS_RESOURCES / "method-pack-estimation.v1.json",
                                    registries / "method-packs.v1.json"),
        rules=load_estimation_rules(ANALYSIS_RESOURCES / "estimation-validation-rules.v1.json"),
        frames=frames.FrameStore(objects=deps.objects), registries=registries,
        repo_root=repo_root, adapters=ADAPTERS)


def estimate(deps: graph.DesignDeps, est: EstimationDeps, analysis_id: str,
             design_revision: int) -> graph.DesignRunResult:
    # D-037 rebuilds the handoff from committed payloads rather than storing it, so the recorded
    # PRD-004 handoff is exactly a preparation revision that ended `prepared` with its bundle.
    prepared = latest_preparation_run(deps.conn, analysis_id)
    body = {} if prepared is None or prepared.outcome_artifact_id is None else payload(
        deps, prepared.outcome_artifact_id)
    if prepared is None or prepared.state in LIVE or body.get("status") != PREPARED:
        emit_blocker(deps, "run", analysis_id, NOT_PREPARED)
        return graph.DesignRunResult(
            status=FAILED, analysis_id=analysis_id, design_revision=design_revision,
            stage_run_id="" if prepared is None else prepared.stage_run_id,
            thread_id="" if prepared is None else prepared.thread_id, error_code=NOT_PREPARED)
    started = latest_estimation_run(deps.conn, analysis_id)
    if started is not None and started.state not in LIVE:
        return _finished(deps, analysis_id, design_revision, started)
    # D-069a: `run` holds the analysis lock, so a live row is a crashed attempt. PRD-004 keeps no
    # checkpoint either, so the next attempt replays the committed artifacts from the top at the
    # next revision, and `estimation_runs` holds one row per revision (D-035, §26.1).
    revision = 1 if started is None else started.revision + 1
    stage_run_id = f"es:{analysis_id}:{revision}"
    return guarded(
        deps, ESTIMATION_TABLE, analysis_id, stage_run_id, design_revision,
        f"et:{analysis_id}:{revision}", lambda: view(design_revision, run_estimation(
            est, analysis_id=analysis_id, stage_run_id=stage_run_id,
            preparation_outcome_artifact_id=str(prepared.outcome_artifact_id),
            estimation_revision=revision)))


def _finished(deps: graph.DesignDeps, analysis_id: str, design_revision: int,
              found: DesignRun) -> graph.DesignRunResult:
    # PRD-003 §16 and PRD-004 §5.2 answer a `design_conflict` with a new design revision, so the
    # conflict code travels as the refusal code and the revision named is the design one.
    body = {} if found.outcome_artifact_id is None else payload(deps, found.outcome_artifact_id)
    conflict = body.get("design_conflict") or {}
    return graph.DesignRunResult(
        status=str(body.get("status") or FAILED), analysis_id=analysis_id,
        stage_run_id=found.stage_run_id, thread_id=found.thread_id,
        design_revision=design_revision, outcome_artifact_id=found.outcome_artifact_id,
        error_code=(str(payload(deps, str(conflict["artifact_id"]))["conflict_code"])
                    if conflict else body.get("error_code")))


def prepare(deps: graph.DesignDeps, prep: PreparationDeps, analysis_id: str,
            design: DesignRun) -> graph.DesignRunResult:
    started = latest_preparation_run(deps.conn, analysis_id)
    if started is not None and started.state not in LIVE:
        return _finished(deps, analysis_id, design.revision, started)
    # D-069a: `run` holds the analysis lock, so a live row is a crashed attempt. Preparation
    # keeps no checkpoint, so the next attempt replays the committed artifacts from the top;
    # `preparation_runs` holds one row per revision, so that attempt is the next revision.
    revision = 1 if started is None else started.revision + 1
    stage_run_id = f"pr:{analysis_id}:{revision}"
    return guarded(
        deps, PREPARATION_TABLE, analysis_id, stage_run_id, design.revision,
        f"pt:{analysis_id}:{revision}", lambda: view(design.revision, run_preparation(
            prep, analysis_id=analysis_id, stage_run_id=stage_run_id,
            design_outcome_artifact_id=str(design.outcome_artifact_id),
            preparation_revision=revision)))


def view(design_revision: int,
         run: PreparationRunResult | EstimationRunResult) -> graph.DesignRunResult:
    return graph.DesignRunResult(
        status=run.status, analysis_id=run.analysis_id, stage_run_id=run.stage_run_id,
        thread_id=run.thread_id, design_revision=design_revision,
        outcome_artifact_id=run.outcome_artifact_id, handoff_id=run.handoff_id,
        error_code=run.conflict_code or run.error_code)


def emit_blocker(deps: graph.DesignDeps, command: str, analysis_id: str, code: str,
                 detail: str = "") -> None:
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


def guarded(deps: graph.DesignDeps, table: str, analysis_id: str, stage_run_id: str,
            design_revision: int, thread_id: str,
            call: Callable[[], graph.DesignRunResult]) -> graph.DesignRunResult:
    # Those stages' own nodes already type every other failure. SC §10.2 keeps its own terminal
    # state and committed artifacts stay committed; SC §4 leaves no run row live after a crash.
    try:
        return call()
    except tracing.ObservabilityError as error:
        state, code = FAILED_OBSERVABILITY, error.code
    except Exception as error:  # noqa: BLE001 -- D-069: no traceback ever escapes a command
        emit_blocker(deps, "run", analysis_id, INTERNAL_ERROR, type(error).__name__)
        state, code = FAILED, INTERNAL_ERROR
    _close_rows(deps, table, stage_run_id, state)
    return graph.DesignRunResult(
        status=state, analysis_id=analysis_id, stage_run_id=stage_run_id, thread_id=thread_id,
        design_revision=design_revision, error_code=code)


def _close_rows(deps: graph.DesignDeps, table: str, stage_run_id: str, state: str) -> None:
    deps.conn.execute(_STATE_ROW.format(table), (state, deps.clock(), stage_run_id))
    with suppress(persistence.PersistenceError):  # a crash before the stage run was opened
        deps.products.transition_stage_run(stage_run_id, state)


def _terminal(deps: graph.DesignDeps, analysis_id: str, state: str,
              code: str) -> graph.DesignRunResult:
    # Both rows exist before any model or tracer call, so the fallback identity only names
    # a revision that failed before `run_design` recorded it.
    found = latest_design_run(deps.conn, analysis_id) or DesignRun(
        f"dr:{analysis_id}:1", "", 1, state, None)
    deps.conn.execute(_STATE_ROW.format(DESIGN_TABLE), (state, deps.clock(), found.stage_run_id))
    deps.products.transition_stage_run(found.stage_run_id, state)
    return graph.DesignRunResult(
        status=state, analysis_id=analysis_id, stage_run_id=found.stage_run_id,
        thread_id=found.thread_id, design_revision=found.revision,
        outcome_artifact_id=found.outcome_artifact_id, error_code=code)
