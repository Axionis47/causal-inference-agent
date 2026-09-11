"""Intake's application boundary and invocation-local artifact lifecycle."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from causal.intake.archive import ArchiveSafety
from causal.intake.catalog import CatalogStore
from causal.intake.contracts import IntakeSubmissionV1
from causal.intake.fields import FieldClasses
from causal.intake.kaggle import KaggleClientProtocol
from causal.intake.outcome import IDEMPOTENCY_CONFLICT, IntakeError, IntakeResult
from causal.intake.workflow import execute_intake
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef
from causal.shared.events import EventEmitter, OperationalEventV1, Severity, Stage, build_event
from causal.shared.persistence import ArtifactCommitter, ObjectStore, ProductStore, build_envelope
from causal.shared.registry import ArtifactTypeRegistry


@dataclass(frozen=True)
class IntakeDeps:
    client: KaggleClientProtocol | Callable[[], KaggleClientProtocol]
    committer: ArtifactCommitter
    products: ProductStore
    catalog: CatalogStore
    objects: ObjectStore
    registry: ArtifactTypeRegistry
    field_classes: FieldClasses
    emitter: EventEmitter
    clock: Callable[[], datetime]
    safety: ArchiveSafety | None = None
    component_version: str = "intake-coordinator.v1"


@dataclass
class IntakeSession:
    """One invocation's identities and effects; never retained on the reusable service."""

    deps: IntakeDeps
    analysis_id: str
    stage_run_id: str
    event_count: int = 0

    def event(self, name: str, **overrides: object) -> OperationalEventV1:
        self.event_count += 1
        event_id = str(overrides.pop("event_id", f"evt:{self.stage_run_id}:{self.event_count}"))
        return build_event(
            occurred_at_utc=self.deps.clock(), event_name=name, event_id=event_id,
            analysis_id=self.analysis_id, stage=Stage.INTAKE, stage_run_id=self.stage_run_id,
            component_id="intake-coordinator", component_version=self.deps.component_version,
            versions={"registry": "artifact-types.v1"}, **overrides)

    def commit(self, kind: str, payload: dict[str, object],
               parents: tuple[ArtifactEnvelopeV1, ...]) -> ArtifactEnvelopeV1:
        envelope = build_envelope(
            self.deps.registry, kind, payload, analysis_id=self.analysis_id,
            stage_run_id=self.stage_run_id, producer_version=self.deps.component_version,
            parents=parents, created_at_utc=self.deps.clock())
        return self.deps.committer.commit(envelope, payload, self.event(
            "artifact.committed", status="committed", required_eval_ids=("EV-P1-005",),
            artifact_refs=(ArtifactRef(artifact_id=envelope.artifact_id,
                                      content_hash=envelope.content_hash),)))

    def resource(self, dataset_id: str, kind: str, name: str, sha: str, key: str,
                 media: str | None, size: int, status: str, reason: str | None) -> None:
        self.deps.catalog.insert_resource(
            f"res:{dataset_id}:{name}", dataset_id, kind, name, sha, key, media, size,
            status, reason)

    def finish(self, payload: dict[str, object],
               parents: tuple[ArtifactEnvelopeV1, ...]) -> IntakeResult:
        outcome = self.commit("IntakeOutcome", payload, parents)
        status = str(payload["status"])
        self.deps.products.transition_stage_run(self.stage_run_id, "committing")
        self.deps.catalog.finalize_run(self.analysis_id, status, outcome.artifact_id)
        self.deps.products.transition_stage_run(self.stage_run_id, "completed")
        self.deps.emitter.emit(self.event(
            "stage.completed", status=status,
            artifact_refs=(ArtifactRef(artifact_id=outcome.artifact_id,
                                      content_hash=outcome.content_hash),)))
        return IntakeResult(self.analysis_id, self.stage_run_id, status, outcome.artifact_id, False)


def run_intake(deps: IntakeDeps, submission: IntakeSubmissionV1) -> IntakeResult:
    """Run deterministic intake without constructing a CLI, model, or design graph."""
    digest = content_hash(submission.model_dump(mode="json"))
    analysis_id = "an-" + hashlib.sha256(submission.idempotency_key.encode()).hexdigest()[:16]
    existing = deps.catalog.find_run(submission.idempotency_key)
    if existing is not None:
        if existing.submission_hash != digest:
            session = IntakeSession(deps, analysis_id, existing.stage_run_id)
            deps.emitter.emit(session.event(
                "blocker.raised", severity=Severity.ERROR, error_code=IDEMPOTENCY_CONFLICT,
                event_id=f"evt:{existing.stage_run_id}:conflict:{uuid.uuid4().hex}",
                safe_dimensions={"blocked_operation": "intake.run"}))
            raise IntakeError(
                f"idempotency key {submission.idempotency_key!r} reused with a different submission",
                IDEMPOTENCY_CONFLICT)
        if existing.intake_outcome_artifact_id is not None:
            return IntakeResult(analysis_id, existing.stage_run_id, str(existing.intake_status),
                                existing.intake_outcome_artifact_id, True)
    attempt = deps.catalog.count_stage_runs(analysis_id) + 1
    session = IntakeSession(deps, analysis_id, f"sr:{analysis_id}:{attempt}")
    deps.products.create_stage_run(session.stage_run_id, analysis_id, "intake")
    for state in ("tracing_preflight", "running"):
        deps.products.transition_stage_run(session.stage_run_id, state)
    deps.emitter.emit(session.event("stage.started"))
    question = session.commit("QuestionRecord", {
        "schema_version": "question-record.v1", "question_text": submission.question_text,
        "context_text": submission.context_text, "kaggle_ref": submission.kaggle_ref,
        "submission_schema_version": submission.schema_version}, ())
    if existing is None:
        deps.catalog.create_run(analysis_id, session.stage_run_id, question.artifact_id,
                                submission.idempotency_key, digest, deps.clock())
    else:
        deps.catalog.reassign_run(analysis_id, session.stage_run_id)
    return execute_intake(session, submission, question)
