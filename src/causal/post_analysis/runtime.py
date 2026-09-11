"""CLI dispatch and exact delivery for the single post-analysis implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from causal.analysis.integration.nodes import open_post_analysis_handoff
from causal.design import graph
from causal.post_analysis.entry import run_post_analysis
from causal.post_analysis.store import PostAnalysisDeps, Store
from causal.runtime import failures
from causal.shared.contracts import ArtifactRef
from causal.shared.persistence import PersistenceError

_LATEST = ("SELECT stage_run_id, '', presentation_revision, state, bundle_artifact_id "
           "FROM presentation.runs WHERE analysis_id=%s ORDER BY presentation_revision DESC LIMIT 1")


def latest_stage_run(conn: Any, stage: str, analysis_id: str) -> Any:
    return failures._stage_run(conn, _LATEST, analysis_id) if stage == "presentation" else (
        failures.latest_stage_run(conn, stage, analysis_id))


def after(state: str, stage: str) -> str | None:
    return "run" if state in failures.LIVE else (
        "presentation" if stage == "presentation" and state == "complete" else None)


def presentation_deps(deps: graph.DesignDeps, registries: Path, repo_root: Path,
                      render_root: Path | None = None) -> PostAnalysisDeps:
    return PostAnalysisDeps(conn=deps.conn, products=deps.products, objects=deps.objects,
        committer=deps.committer, registry=deps.registry, emitter=deps.emitter, clock=deps.clock,
        gateway=deps.gateway, checkpointer=deps.checkpointer, tracer=deps.tracer,
        render_root=render_root or repo_root / "renders" / "post_analysis")


def _ref(deps: Any, artifact_id: str) -> ArtifactRef:
    envelope = deps.products.load_envelope(artifact_id)
    return ArtifactRef(artifact_id=artifact_id, content_hash=envelope.content_hash)


def estimate_and_present(deps: Any, est: Any, pres: PostAnalysisDeps, analysis_id: str,
                         design_revision: int) -> graph.DesignRunResult:
    result = failures.estimate(deps, est, analysis_id, design_revision)
    receipt = failures.latest_estimation_run(deps.conn, analysis_id)
    return present(deps, est, pres, analysis_id, design_revision) if (
        receipt is not None and receipt.outcome_artifact_id is not None
        and result.status in {"complete", "not_estimable", "invalidated", "failed"}) else result


def present(deps: Any, est: Any, pres: PostAnalysisDeps, analysis_id: str,
            design_revision: int) -> graph.DesignRunResult:
    prior = latest_stage_run(deps.conn, "presentation", analysis_id)
    if prior is not None and prior.state not in failures.LIVE:
        return graph.DesignRunResult(status=prior.state, analysis_id=analysis_id,
            stage_run_id=prior.stage_run_id, thread_id=f"post:{prior.stage_run_id}",
            design_revision=design_revision, outcome_artifact_id=prior.outcome_artifact_id)
    done = failures.latest_estimation_run(deps.conn, analysis_id)
    if done is None or done.outcome_artifact_id is None:
        raise PersistenceError("no authoritative analysis outcome", "analysis_outcome_missing")
    revision = prior.revision if prior else 1
    stage_run_id = prior.stage_run_id if prior else f"ps:{analysis_id}:{revision}"
    outcome = _ref(deps, done.outcome_artifact_id)
    if prior is not None:
        record = deps.conn.execute("SELECT run_record FROM presentation.runs WHERE stage_run_id=%s",
                                    (stage_run_id,)).fetchone()[0]
        outcome = ArtifactRef.model_validate(record["outcome"])
    return _run(deps, est, pres, analysis_id, design_revision, revision, stage_run_id, outcome)


def _run(deps: Any, est: Any, pres: PostAnalysisDeps, analysis_id: str, design_revision: int,
         revision: int, stage_run_id: str, outcome: ArtifactRef,
         counters: dict[str, int] | None = None) -> graph.DesignRunResult:
    handoff = open_post_analysis_handoff(est, analysis_id, outcome.artifact_id, stage_run_id)
    result = run_post_analysis(pres, analysis_id=analysis_id, stage_run_id=stage_run_id,
        revision=revision, outcome=outcome, handoff=handoff, previous_counters=counters)
    return graph.DesignRunResult(status=result.status, analysis_id=analysis_id,
        stage_run_id=stage_run_id, thread_id=result.thread_id, design_revision=design_revision,
        error_code=result.error_code, outcome_artifact_id=result.bundle.artifact_id if result.bundle else None)


def recover_presentation(deps: Any, est: Any, pres: PostAnalysisDeps, analysis_id: str,
                         design_revision: int, *, failed_stage_run_id: str) -> graph.DesignRunResult:
    prior = latest_stage_run(deps.conn, "presentation", analysis_id)
    if prior is None or prior.stage_run_id != failed_stage_run_id or prior.state not in {"incomplete", "failed"}:
        raise PersistenceError("recovery requires the latest incomplete post-analysis attempt",
                               "presentation_recovery_unavailable")
    record = deps.conn.execute("SELECT run_record FROM presentation.runs WHERE stage_run_id=%s",
                              (failed_stage_run_id,)).fetchone()[0]
    if not record.get("thread_id") or not record.get("outcome"):
        raise PersistenceError("historical presentation requires an explicit handoff migration",
                               "presentation_recovery_unavailable")
    if prior.state != "incomplete":
        raise PersistenceError("historical presentation requires an explicit handoff migration",
                               "presentation_recovery_unavailable")
    counters = {key: int(record.get(key, 0)) for key in ("calls", "reviews")}
    if counters["calls"] >= pres.max_calls or counters["reviews"] >= pres.max_reviews:
        raise PersistenceError("the original reporting budget is exhausted", "post_analysis_budget_exhausted")
    revision = prior.revision + 1
    return _run(deps, est, pres, analysis_id, design_revision, revision,
                f"ps:{analysis_id}:{revision}", ArtifactRef.model_validate(record["outcome"]), counters)


def bundle_view(deps: Any, bundle_id: str, expected_hash: str) -> dict[str, Any]:
    """Exact immutable read, including historical bundles; never run authoring on delivery."""
    envelope = deps.products.load_envelope(bundle_id)
    if envelope.content_hash != expected_hash:
        raise PersistenceError("bundle hash differs", "presentation_bundle_unavailable")
    pres = presentation_deps(deps, Path(), Path())
    store = Store(pres, envelope.analysis_id, envelope.stage_run_id)
    body = store.read(ArtifactRef(artifact_id=bundle_id, content_hash=expected_hash))
    if envelope.artifact_type == "PresentationBundle":
        return body | {"figures": [store.read(row) for row in body["renders"]]}
    if envelope.artifact_type != "PostAnalysisBundle":
        raise PersistenceError("unsupported delivered artifact", "presentation_bundle_unavailable")
    row = deps.conn.execute("SELECT state,bundle_artifact_id FROM presentation.runs WHERE stage_run_id=%s",
                            (envelope.stage_run_id,)).fetchone()
    if not row or row[0] != "complete" or row[1] != bundle_id:
        raise PersistenceError("bundle has no completed observable release", "presentation_bundle_unavailable")
    from causal.post_analysis.review import verify_objects
    exported = store.read(body["export"])
    verify_objects(store, exported)
    draft = store.read(body["draft"])
    summary = "\n\n".join(statement["text"] for section in draft["sections"]
                          for statement in section["statements"])
    selected = {key for section in draft["sections"] for key in section["visual_ids"]}
    figures = [store.read(body["visuals"][key]) for key in sorted(selected)]
    directory = pres.render_root / "delivery" / expected_hash
    for index, artifact in enumerate([exported, *figures]):
        verify_objects(store, artifact)
        paths = {}
        for name, locator in artifact["objects"].items():
            path = directory / str(index) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pres.objects.get(locator))
            paths[name] = str(path.resolve())
        artifact["objects"] = paths
    return body | {"report": draft, "summary": summary, "delivery": exported, "figures": figures}
