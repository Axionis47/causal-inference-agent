"""Deterministic capture, resource fan-in, and publication of intake evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from causal.intake.archive import ArchiveSafety
from causal.intake.contracts import IntakeSubmissionV1
from causal.intake.fields import classify_capture
from causal.intake.kaggle import KaggleError, capture
from causal.intake.outcome import IntakeResult, outcome_payload
from causal.intake.resources import ResourceResult, inventory_archive, process_resource
from causal.intake.semantic import (
    build_evidence_bundle,
    build_semantic_map,
    measured_index_rows,
    slot_index_rows,
)
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.events import Severity
from causal.shared.persistence import ObjectStore

if TYPE_CHECKING:
    from causal.intake.entry import IntakeSession


@dataclass(frozen=True)
class ProfiledSource:
    capture: ArtifactEnvelopeV1
    capture_payload: dict[str, object]
    manifest: ArtifactEnvelopeV1
    resources: tuple[ResourceResult, ...]


def execute_intake(session: IntakeSession, submission: IntakeSubmissionV1,
                   question: ArtifactEnvelopeV1) -> IntakeResult:
    deps = session.deps
    capture_dims = {"operation": "kaggle.capture"}
    deps.emitter.emit(session.event(
        "tool.started", required_eval_ids=("EV-P1-002",), safe_dimensions=capture_dims))
    try:
        captured = capture(deps.client, submission.kaggle_ref)
    except KaggleError as error:
        deps.emitter.emit(session.event(
            "tool.failed", severity=Severity.ERROR, error_code=error.code,
            required_eval_ids=("EV-P1-002",), safe_dimensions=capture_dims))
        return _refuse(session, question, None, f"capture failed: {error.code}")
    deps.emitter.emit(session.event(
        "tool.completed", required_eval_ids=("EV-P1-002",), status="completed",
        safe_dimensions=capture_dims))
    cap = session.commit("KaggleCapture", captured.payload, (question,))
    dataset = cast(dict[str, object], captured.payload["dataset"])
    dataset_id = str(dataset["dataset_id"])
    deps.catalog.upsert_dataset(dataset, cap.artifact_id)
    deps.catalog.set_run_dataset(session.analysis_id, dataset_id)
    sha = str(captured.payload["archive_sha256"])
    key = deps.objects.put_if_absent(sha, captured.archive_bytes)
    session.resource(dataset_id, "api_capture", "kaggle_capture", cap.content_hash,
                     cap.payload_locator, "json", 0, "parsed", None)
    safety = deps.safety or ArchiveSafety()
    admission = safety.admit(captured.archive_bytes)
    session.resource(dataset_id, "archive", "source_archive", sha, key, "zip",
                     len(captured.archive_bytes), "parsed" if admission.safe else "unsafe",
                     admission.refusal_reason)
    inventory = inventory_archive(captured.archive_bytes, admission, safety)
    if not admission.safe:
        for resource in inventory.entries:
            session.resource(dataset_id, "other", resource.name, sha, key, resource.media_type,
                             resource.byte_size,
                             "unsafe" if resource.classification == "unsafe" else "excluded",
                             resource.reason or admission.refusal_reason or "archive refused")
        return _refuse(session, question, dataset_id,
                       f"archive refused: {admission.refusal_reason or 'archive refused'}")
    for resource in inventory.entries:
        if resource.data is not None:
            deps.objects.put_if_absent(resource.sha256, resource.data)
    manifest = session.commit("SourceManifest", inventory.manifest_payload(dataset_id), (cap,))
    deps.catalog.set_source_manifest(dataset_id, manifest.artifact_id)
    results = tuple(process_resource(resource) for resource in inventory.entries)
    for result in results:
        _record_resource(session, dataset_id, result)
    return _publish(session, submission, question, dataset_id,
                    ProfiledSource(cap, captured.payload, manifest, results))


def _record_resource(session: IntakeSession, dataset_id: str, result: ResourceResult) -> None:
    resource = result.resource
    kind = resource.classification if resource.classification in (
        "table", "document", "metadata") else "other"
    session.deps.emitter.emit(session.event(
        "task.completed" if result.status == "parsed" else "task.failed",
        required_eval_ids=("EV-P1-003",), status=result.status,
        error_code=None if result.status == "parsed" else result.status,
        safe_dimensions={"resource": resource.name, "parse_status": result.status}))
    session.resource(dataset_id, kind, resource.name, resource.sha256,
                     ObjectStore.locator_for(resource.sha256), resource.media_type,
                     resource.byte_size, result.status, result.reason)


def _publish(session: IntakeSession, submission: IntakeSubmissionV1,
             question: ArtifactEnvelopeV1, dataset_id: str, source: ProfiledSource) -> IntakeResult:
    profiles = {r.resource.name: r.profile for r in source.resources if r.profile is not None}
    documents = {r.resource.name: r.document for r in source.resources if r.document is not None}
    counts = dict(Counter(r.status for r in source.resources))
    profile_envs = tuple(session.commit(
        "TableProfile", profiles[name] | {"logical_name": name}, (source.manifest,))
        for name in sorted(profiles))
    evidence_payload = build_evidence_bundle(
        source.capture_payload, documents, session.deps.field_classes)
    degraded = any(counts.get(key, 0) for key in ("excluded", "unreadable", "failed"))
    status = "usable" if evidence_payload["items"] and not degraded else "partial"
    items = cast(list[dict[str, object]], evidence_payload["items"])
    for field, value in (("question", submission.question_text),
                         ("context", submission.context_text)):
        if value:
            items.append({"evidence_id": f"ua:{field}/text", "scope_kind": "dataset",
                          "table_name": None, "column_name": None,
                          "source_field": f"{field}_text", "value": value})
    evidence = session.commit("EvidenceBundle", evidence_payload,
                              (source.capture, source.manifest, *profile_envs))
    semantic_map = build_semantic_map(source.capture_payload, profiles)
    map_env = session.commit("SemanticMap", semantic_map, (evidence,))
    slot_rows = slot_index_rows(semantic_map)
    measured = measured_index_rows(profiles, {
        name: env.artifact_id for name, env in zip(sorted(profiles), profile_envs, strict=True)})
    rows = classify_capture(source.capture_payload, session.deps.field_classes) + slot_rows + measured
    session.deps.catalog.replace_field_rows(
        dataset_id, source.capture.artifact_id, map_env.artifact_id, rows)
    if not profiles:
        return _refuse(session, question, dataset_id, "no supported table profiled",
                       (source.manifest, evidence, map_env))
    payload = outcome_payload(
        status=status, analysis_id=session.analysis_id, stage_run_id=session.stage_run_id,
        dataset_id=dataset_id, question=question, manifest=source.manifest,
        profile_envs=profile_envs, evidence=evidence, map_env=map_env, tables=tuple(sorted(profiles)),
        slot_rows=slot_rows, counts=counts, refusal_reason=None)
    return session.finish(payload, (question, source.manifest, *profile_envs, evidence, map_env))


def _refuse(session: IntakeSession, question: ArtifactEnvelopeV1, dataset_id: str | None,
            reason: str, parents: tuple[ArtifactEnvelopeV1, ...] = ()) -> IntakeResult:
    payload = outcome_payload(
        status="refused", analysis_id=session.analysis_id, stage_run_id=session.stage_run_id,
        dataset_id=dataset_id, question=question, manifest=None, profile_envs=(), evidence=None,
        map_env=None, tables=(), slot_rows=(), counts={}, refusal_reason=reason)
    return session.finish(payload, (question, *parents))
