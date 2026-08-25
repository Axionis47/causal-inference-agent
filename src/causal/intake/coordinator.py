"""The deterministic intake coordinator (PRD-001 §4, §11–§13; T-008; D-031..D-035)."""

from __future__ import annotations

import hashlib
import io
import posixpath
import zipfile
from collections.abc import Callable
from datetime import datetime
from typing import Final

from causal.intake.archive import ArchiveAdmission, ArchiveSafety
from causal.intake.catalog import CatalogStore
from causal.intake.contracts import IntakeSubmissionV1
from causal.intake.fields import FieldClasses, classify_capture
from causal.intake.kaggle import KaggleClientProtocol, KaggleError, capture
from causal.intake.outcome import (
    IDEMPOTENCY_CONFLICT,
    IntakeError,
    IntakeResult,
    open_handoff,
    outcome_payload,
)
from causal.intake.profiler import profile_table
from causal.intake.semantic import (
    build_evidence_bundle,
    build_semantic_map,
    measured_index_rows,
    slot_index_rows,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.events import EventEmitter, OperationalEventV1, Severity, Stage, build_event
from causal.shared.persistence import (
    ArtifactCommitter,
    ObjectStore,
    ProductStore,
    build_envelope,
)
from causal.shared.registry import ArtifactTypeRegistry

__all__ = ["IntakeCoordinator", "IntakeError", "IntakeResult"]

PROFILER_VERSION: Final = "profiler.v1"
TABLE_MEDIA: Final = {".csv": "csv", ".tsv": "tsv", ".parquet": "parquet"}


class IntakeCoordinator:
    def __init__(
        self,
        client: KaggleClientProtocol,
        committer: ArtifactCommitter,
        products: ProductStore,
        catalog: CatalogStore,
        objects: ObjectStore,
        registry: ArtifactTypeRegistry,
        field_classes: FieldClasses,
        emitter: EventEmitter,
        clock: Callable[[], datetime],
        safety: ArchiveSafety | None = None,
        component_version: str = "intake-coordinator.v1",
    ) -> None:
        self._client = client
        self._committer = committer
        self._products = products
        self._catalog = catalog
        self._objects = objects
        self._registry = registry
        self._field_classes = field_classes
        self._emitter = emitter
        self._clock = clock
        self._safety = safety or ArchiveSafety()
        self._component_version = component_version
        self._analysis_id = ""
        self._stage_run_id = ""
        self._event_count = 0

    # -- run entry -------------------------------------------------------

    def run(self, submission: IntakeSubmissionV1) -> IntakeResult:
        submission_hash = content_hash(submission.model_dump(mode="json"))
        self._analysis_id = (
            "an-" + hashlib.sha256(submission.idempotency_key.encode()).hexdigest()[:16]
        )
        existing = self._catalog.find_run(submission.idempotency_key)
        if existing is not None:
            self._stage_run_id = existing.stage_run_id
            if existing.submission_hash != submission_hash:
                self._emitter.emit(self._event(
                    "blocker.raised", severity=Severity.ERROR,
                    error_code=IDEMPOTENCY_CONFLICT,
                    safe_dimensions={"blocked_operation": "intake.run"}))
                raise IntakeError(
                    f"idempotency key {submission.idempotency_key!r} reused with a"
                    " different submission", IDEMPOTENCY_CONFLICT)
            if existing.intake_outcome_artifact_id is not None:
                return IntakeResult(
                    self._analysis_id, existing.stage_run_id,
                    str(existing.intake_status), existing.intake_outcome_artifact_id,
                    replayed=True)
        attempt = self._catalog.count_stage_runs(self._analysis_id) + 1
        self._stage_run_id = f"sr:{self._analysis_id}:{attempt}"
        self._event_count = 0
        self._products.create_stage_run(self._stage_run_id, self._analysis_id, "intake")
        self._products.transition_stage_run(self._stage_run_id, "tracing_preflight")
        self._products.transition_stage_run(self._stage_run_id, "running")  # D-020
        self._emitter.emit(self._event("stage.started"))
        question = self._commit("QuestionRecord", {
            "schema_version": "question-record.v1",
            "question_text": submission.question_text,
            "context_text": submission.context_text,
            "kaggle_ref": submission.kaggle_ref,
            "submission_schema_version": submission.schema_version}, ())
        if existing is None:
            self._catalog.create_run(
                self._analysis_id, self._stage_run_id, question.artifact_id,
                submission.idempotency_key, submission_hash, self._clock())
        else:
            self._catalog.reassign_run(self._analysis_id, self._stage_run_id)
        return self._execute(submission, question)

    def open_handoff(
        self, analysis_id: str, intake_outcome_artifact_id: str,
        receiving_stage_run_id: str,
    ) -> HandoffManifestV1:
        return open_handoff(
            self._catalog, self._products, analysis_id, intake_outcome_artifact_id,
            receiving_stage_run_id, self._clock)

    # -- pipeline --------------------------------------------------------

    def _execute(
        self, submission: IntakeSubmissionV1, question: ArtifactEnvelopeV1
    ) -> IntakeResult:
        capture_dims = {"operation": "kaggle.capture"}
        self._emitter.emit(self._event(
            "tool.started", required_eval_ids=("EV-P1-002",),
            safe_dimensions=capture_dims))
        try:
            captured = capture(self._client, submission.kaggle_ref)
        except KaggleError as error:
            self._emitter.emit(self._event(
                "tool.failed", severity=Severity.ERROR, error_code=error.code,
                required_eval_ids=("EV-P1-002",), safe_dimensions=capture_dims))
            return self._refuse(question, None, f"capture failed: {error.code}")
        self._emitter.emit(self._event(
            "tool.completed", required_eval_ids=("EV-P1-002",), status="completed",
            safe_dimensions=capture_dims))
        cap = self._commit("KaggleCapture", captured.payload, (question,))
        dataset = captured.payload["dataset"]
        assert isinstance(dataset, dict)
        dataset_id = str(dataset["dataset_id"])
        self._catalog.upsert_dataset(dataset, cap.artifact_id)
        self._catalog.set_run_dataset(self._analysis_id, dataset_id)
        archive_sha = hashlib.sha256(captured.archive_bytes).hexdigest()
        archive_key = self._objects.put_if_absent(archive_sha, captured.archive_bytes)
        self._resource(dataset_id, "api_capture", "kaggle_capture", cap.content_hash,
                       cap.payload_locator, "json", 0, "parsed", None)
        admission = self._safety.admit(captured.archive_bytes)
        if not admission.safe:
            reason = admission.refusal_reason or "archive refused"
            self._resource(dataset_id, "archive", "source_archive", archive_sha,
                           archive_key, "zip", len(captured.archive_bytes), "unsafe", reason)
            return self._refuse(question, dataset_id, f"archive refused: {reason}")
        self._resource(dataset_id, "archive", "source_archive", archive_sha, archive_key,
                       "zip", len(captured.archive_bytes), "parsed", None)
        return self._process(question, cap, captured.payload, captured.archive_bytes,
                             dataset_id, admission)

    def _process(
        self, question: ArtifactEnvelopeV1, cap: ArtifactEnvelopeV1,
        capture_payload: dict[str, object], archive_bytes: bytes, dataset_id: str,
        admission: ArchiveAdmission,
    ) -> IntakeResult:
        sizes = {
            info.filename: info.file_size
            for info in zipfile.ZipFile(io.BytesIO(archive_bytes)).infolist()
        }
        extracted = self._safety.extract_admitted(archive_bytes, admission)
        archive_sha = str(capture_payload["archive_sha256"])
        manifest_entries: list[dict[str, object]] = []
        for decision in admission.decisions:
            data = extracted.get(decision.name)
            sha = hashlib.sha256(data).hexdigest() if data is not None else archive_sha
            if data is not None:
                self._objects.put_if_absent(sha, data)
            manifest_entries.append({
                "logical_name": decision.name,
                "classification": decision.classification, "sha256": sha,
                "byte_size": len(data) if data is not None else sizes.get(decision.name, 0),
                "media_type": posixpath.splitext(decision.name)[1].lstrip(".").lower(),
                "reason": decision.reason})
        manifest = self._commit("SourceManifest", {
            "schema_version": "source-manifest.v1", "dataset_id": dataset_id,
            "resources": manifest_entries}, (cap,))
        self._catalog.set_source_manifest(dataset_id, manifest.artifact_id)
        profiles: dict[str, dict[str, object]] = {}
        documents: dict[str, str] = {}
        counts = {"excluded": 0, "unreadable": 0, "failed": 0}
        for decision, entry in zip(admission.decisions, manifest_entries, strict=True):
            status, reason = self._process_entry(decision.name, decision.classification,
                                                 decision.reason, extracted,
                                                 profiles, documents)
            counts[status] = counts.get(status, 0) + 1
            kind = (decision.classification
                    if decision.classification in ("table", "document", "metadata")
                    else "other")
            sha = str(entry["sha256"])
            size = entry["byte_size"]
            assert isinstance(size, int)
            self._resource(dataset_id, kind, decision.name, sha,
                           ObjectStore.locator_for(sha), str(entry["media_type"]),
                           size, status, reason)
        profile_envs = tuple(
            self._commit("TableProfile", dict(profiles[name]) | {"logical_name": name},
                         (manifest,))
            for name in sorted(profiles)
        )
        return self._finish(question, cap, manifest, profile_envs, profiles, documents,
                            capture_payload, dataset_id, counts)

    def _process_entry(
        self, name: str, classification: str, decision_reason: str | None,
        extracted: dict[str, bytes], profiles: dict[str, dict[str, object]],
        documents: dict[str, str],
    ) -> tuple[str, str | None]:
        extension = posixpath.splitext(name)[1].lower()
        status, reason = "parsed", None
        if classification == "table":
            try:
                profiles[name] = profile_table(
                    extracted[name], TABLE_MEDIA[extension], PROFILER_VERSION
                )
            except Exception as error:  # noqa: BLE001 -- any parse failure is terminal
                status, reason = "failed", f"profiling failed: {error}"
        elif classification in ("document", "metadata"):
            if extension in (".yaml", ".yml"):
                status, reason = "excluded", "yaml parsing deferred (PRD-001 §5.5)"
            else:
                try:
                    documents[name] = extracted[name].decode("utf-8")
                except UnicodeDecodeError:
                    status, reason = "failed", "not valid utf-8"
        elif classification == "unreadable":
            status, reason = "unreadable", decision_reason
        else:  # withheld (unsafe never reaches here)
            status, reason = "excluded", decision_reason
        self._emitter.emit(self._event(
            "task.completed" if status == "parsed" else "task.failed",
            required_eval_ids=("EV-P1-003",), status=status,
            error_code=None if status == "parsed" else status,
            safe_dimensions={"resource": name, "parse_status": status}))
        return status, reason

    def _finish(
        self, question: ArtifactEnvelopeV1, cap: ArtifactEnvelopeV1,
        manifest: ArtifactEnvelopeV1, profile_envs: tuple[ArtifactEnvelopeV1, ...],
        profiles: dict[str, dict[str, object]], documents: dict[str, str],
        capture_payload: dict[str, object], dataset_id: str, counts: dict[str, int],
    ) -> IntakeResult:
        evidence_payload = build_evidence_bundle(
            capture_payload, documents, self._field_classes)
        evidence = self._commit(
            "EvidenceBundle", evidence_payload, (cap, manifest, *profile_envs))
        semantic_map = build_semantic_map(capture_payload, profiles)
        map_env = self._commit("SemanticMap", semantic_map, (evidence,))
        slot_rows = slot_index_rows(semantic_map)
        measured = measured_index_rows(
            profiles,
            {name: env.artifact_id
             for name, env in zip(sorted(profiles), profile_envs, strict=True)})
        rows = classify_capture(capture_payload, self._field_classes) + slot_rows + measured
        self._catalog.replace_field_rows(
            dataset_id, cap.artifact_id, map_env.artifact_id, rows)
        if not profiles:
            return self._refuse(question, dataset_id, "no supported table profiled",
                                extra_parents=(manifest, evidence, map_env))
        degraded = any(counts[key] for key in ("excluded", "unreadable", "failed"))
        status = "usable" if evidence_payload["items"] and not degraded else "partial"
        payload = outcome_payload(
            status=status, analysis_id=self._analysis_id,
            stage_run_id=self._stage_run_id, dataset_id=dataset_id, question=question,
            manifest=manifest, profile_envs=profile_envs, evidence=evidence,
            map_env=map_env, tables=tuple(sorted(profiles)), slot_rows=slot_rows,
            counts=counts, refusal_reason=None,
        )
        outcome = self._commit(
            "IntakeOutcome", payload,
            (question, manifest, *profile_envs, evidence, map_env))
        self._finalize(status, outcome)
        return IntakeResult(self._analysis_id, self._stage_run_id, status,
                            outcome.artifact_id, replayed=False)

    # -- refusal and finalization ---------------------------------------

    def _refuse(
        self, question: ArtifactEnvelopeV1, dataset_id: str | None, reason: str,
        extra_parents: tuple[ArtifactEnvelopeV1, ...] = (),
    ) -> IntakeResult:
        payload = outcome_payload(
            status="refused", analysis_id=self._analysis_id,
            stage_run_id=self._stage_run_id, dataset_id=dataset_id, question=question,
            manifest=None, profile_envs=(), evidence=None, map_env=None, tables=(),
            slot_rows=(), counts={}, refusal_reason=reason,
        )
        outcome = self._commit("IntakeOutcome", payload, (question, *extra_parents))
        self._finalize("refused", outcome)
        return IntakeResult(self._analysis_id, self._stage_run_id, "refused",
                            outcome.artifact_id, replayed=False)

    def _finalize(self, status: str, outcome: ArtifactEnvelopeV1) -> None:
        self._products.transition_stage_run(self._stage_run_id, "committing")
        self._catalog.finalize_run(self._analysis_id, status, outcome.artifact_id)
        self._products.transition_stage_run(self._stage_run_id, "completed")
        self._emitter.emit(self._event(
            "stage.completed", status=status,
            artifact_refs=(ArtifactRef(artifact_id=outcome.artifact_id,
                                       content_hash=outcome.content_hash),)))

    # -- helpers ---------------------------------------------------------

    def _commit(
        self, artifact_type: str, payload: dict[str, object],
        parents: tuple[ArtifactEnvelopeV1, ...],
    ) -> ArtifactEnvelopeV1:
        envelope = build_envelope(
            self._registry, artifact_type, payload,
            analysis_id=self._analysis_id, stage_run_id=self._stage_run_id,
            producer_version=self._component_version, parents=parents,
            created_at_utc=self._clock())
        event = self._event(
            "artifact.committed", status="committed",
            required_eval_ids=("EV-P1-005",),
            artifact_refs=(ArtifactRef(artifact_id=envelope.artifact_id,
                                       content_hash=envelope.content_hash),))
        return self._committer.commit(envelope, payload, event)

    def _event(self, name: str, **overrides: object) -> OperationalEventV1:
        self._event_count += 1
        return build_event(
            occurred_at_utc=self._clock(), event_name=name,
            event_id=f"evt:{self._stage_run_id}:{self._event_count}",
            analysis_id=self._analysis_id, stage=Stage.INTAKE,
            stage_run_id=self._stage_run_id, component_id="intake-coordinator",
            component_version=self._component_version,
            versions={"registry": "artifact-types.v1"}, **overrides)

    def _resource(
        self, dataset_id: str, kind: str, logical_name: str, sha: str, key: str,
        media_type: str | None, byte_size: int, parse_status: str, reason: str | None,
    ) -> None:
        self._catalog.insert_resource(
            f"res:{dataset_id}:{logical_name}", dataset_id, kind, logical_name,
            sha, key, media_type, byte_size, parse_status, reason)
