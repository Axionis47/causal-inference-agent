# The PRD-005 stage dispatch: its dependencies, the revision-bump rules `causal run` follows
# after a `complete` estimation revision, the typed terminal conversion, and the one committed
# bundle the §20 delivery command opens. Domain logic stays in `causal.presentation` (SC §14.1).

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

from psycopg import Connection

from causal.design import graph
from causal.estimation.harness import EstimationDeps
from causal.estimation.nodes import open_presentation_handoff
from causal.presentation.catalog import load_visualization_catalog
from causal.presentation.harness import PresentationDeps, PresentationRunResult
from causal.presentation.nodes import run_presentation
from causal.runtime import failures
from causal.shared import persistence
from causal.shared.contracts import ArtifactRef

# PRD-005 opens only on an estimation revision that reached this terminal status (§5).
COMPLETE, NOT_COMPLETE = "complete", "estimation_not_complete"
BUNDLE_UNAVAILABLE: Final = "presentation_bundle_unavailable"
_LATEST: Final = (
    "SELECT stage_run_id, '', presentation_revision, state, bundle_artifact_id"
    " FROM presentation.runs WHERE analysis_id = %s ORDER BY presentation_revision DESC LIMIT 1")
_VIEW: Final = ("SELECT artifact_id FROM design.causal_graph_views WHERE analysis_id = %s"
                " ORDER BY design_revision DESC LIMIT 1")


def latest_stage_run(conn: Connection[Any], stage: str, analysis_id: str) -> Any:
    # The newest row of one later stage for `causal status`; all four expose the same columns.
    return (failures._stage_run(conn, _LATEST, analysis_id) if stage == "presentation"
            else failures.latest_stage_run(conn, stage, analysis_id))


def after(state: str, stage: str) -> str | None:
    # D-069b: a live stage row names `run`; only a finished PRD-005 leaves a delivery command.
    return "run" if state in failures.LIVE else (stage if stage == "presentation" else None)


def presentation_deps(deps: graph.DesignDeps, registries: Path, repo_root: Path,
                      render_root: Path | None = None) -> PresentationDeps:
    # PRD-005's dependencies: the immutable catalog and the one visualization-curator receiver.
    return PresentationDeps(
        conn=deps.conn, products=deps.products, objects=deps.objects, committer=deps.committer,
        registry=deps.registry, emitter=deps.emitter, clock=deps.clock, repo_root=repo_root,
        catalog=load_visualization_catalog(registries / "visualization-catalog.v1.json"),
        render_root=render_root or repo_root / "renders", gateway=cast(Any, deps.gateway))


def _held(deps: graph.DesignDeps, found: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=found, content_hash=str(deps.products.find_artifact_hash(found)))


def estimate_and_present(deps: graph.DesignDeps, est: EstimationDeps, pres: PresentationDeps,
                         analysis_id: str, design_revision: int) -> graph.DesignRunResult:
    # §22: a `complete` estimation revision opens PRD-005 inside the same `causal run` step.
    found = failures.estimate(deps, est, analysis_id, design_revision)
    return present(deps, est, pres, analysis_id, design_revision) if (
        found.status == COMPLETE) else found


def present(deps: graph.DesignDeps, est: EstimationDeps, pres: PresentationDeps, analysis_id: str,
            design_revision: int) -> graph.DesignRunResult:
    # D-037 rebuilds the handoff from committed payloads rather than storing it, and D-035 reports
    # a finished revision from its own row instead of re-running it. `run` holds the analysis
    # lock, so a live row is a crashed attempt whose replay is simply the next revision.
    started = failures._stage_run(deps.conn, _LATEST, analysis_id)
    if started is not None and started.state not in failures.LIVE:
        return _view(design_revision, PresentationRunResult(
            status=started.state, analysis_id=analysis_id, presentation_revision=started.revision,
            stage_run_id=started.stage_run_id,
            presentation_bundle_id=started.outcome_artifact_id))
    done = failures.latest_estimation_run(deps.conn, analysis_id)
    outcome = None if done is None else done.outcome_artifact_id
    if outcome is None or failures.payload(deps, outcome).get("status") != COMPLETE:
        failures.emit_blocker(deps, "run", analysis_id, NOT_COMPLETE)
        return graph.DesignRunResult(status=failures.FAILED, analysis_id=analysis_id,
                                     thread_id="", stage_run_id="", refusal_code=NOT_COMPLETE,
                                     design_revision=design_revision)
    revision = 1 if started is None else started.revision + 1
    stage_run_id = f"ps:{analysis_id}:{revision}"
    view = deps.conn.execute(_VIEW, (analysis_id,)).fetchone()
    return failures.guarded(
        deps, "presentation.runs", analysis_id, stage_run_id, design_revision, "",
        lambda: _view(design_revision, run_presentation(
            pres, analysis_id=analysis_id, stage_run_id=stage_run_id,
            presentation_revision=revision, estimation_outcome=_held(deps, outcome),
            handoff_manifest=open_presentation_handoff(est, analysis_id, outcome, stage_run_id),
            approved_graph_view=_held(deps, str(view[0]) if view else ""))))


def _view(design_revision: int, run: PresentationRunResult) -> graph.DesignRunResult:
    # One PRD-005 terminal in the single result shape the CLI renders. A typed §17.2 status
    # carries its own destination as the refusal code, so the printed result names who owns the
    # next move without inventing a new CLI status.
    return graph.DesignRunResult(
        status=run.status, analysis_id=run.analysis_id, stage_run_id=run.stage_run_id,
        thread_id="", design_revision=design_revision, refusal_code=run.destination,
        outcome_artifact_id=run.presentation_bundle_id, error_code=run.error_code)


def bundle_view(deps: graph.DesignDeps, bundle_id: str, expected_hash: str) -> dict[str, Any]:
    # §20: the one completed bundle, opened by exact artifact id and expected content hash —
    # never an implicit latest revision and never a recomputation. Each render's already
    # committed object paths and hashes are resolved beside the frozen summary.
    found = deps.products.load_envelope(bundle_id)
    if found.artifact_type != "PresentationBundle" or found.content_hash != expected_hash:
        raise persistence.PersistenceError(f"no delivered bundle {bundle_id!r}", BUNDLE_UNAVAILABLE)
    body = failures.payload(deps, bundle_id)
    return body | {"figures": [failures.payload(deps, str(row["artifact_id"]))
                               for row in body["renders"]]}
