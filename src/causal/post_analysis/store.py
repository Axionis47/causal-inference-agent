"""Post-analysis artifact access over the shared commit and operational services."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from causal.shared import events, persistence
from causal.shared.contracts import ArtifactRef
from causal.shared.tracing import TracerProtocol

COMPONENT = "post-analysis"
VERSION = "post-analysis.v1"
LOGGER = logging.getLogger(__name__)
EVALUATIONS = {
    "PostAnalysisContext": "EV-P5-001", "PostAnalysisDraft": "EV-P5-003",
    "PostAnalysisVisual": "EV-P5-004", "PostAnalysisExport": "EV-P5-005",
    "PostAnalysisReview": "EV-P5-005", "PostAnalysisBundle": "EV-P5-006",
}


@dataclass(frozen=True)
class PostAnalysisDeps:
    conn: Any
    products: Any
    objects: Any
    committer: Any
    registry: Any
    emitter: Any
    clock: Callable[[], datetime]
    gateway: Any
    checkpointer: Any
    tracer: TracerProtocol | None
    render_root: Path
    require_tracing: bool = True
    max_calls: int = 24
    max_reviews: int = 3


class Store:
    def __init__(self, deps: PostAnalysisDeps, analysis_id: str, stage_run_id: str) -> None:
        self.deps, self.analysis_id, self.stage_run_id = deps, analysis_id, stage_run_id
        self.thread_id = f"post:{stage_run_id}"

    def event(self, name: str, **fields: Any) -> Any:
        import uuid
        return events.build_event(
            occurred_at_utc=self.deps.clock(), event_name=name,
            event_id=f"event:{self.stage_run_id}:{uuid.uuid4()}", analysis_id=self.analysis_id,
            stage=events.Stage.PRESENTATION, stage_run_id=self.stage_run_id,
            graph_thread_id=self.thread_id, component_id=COMPONENT, component_version=VERSION,
            required_eval_ids=fields.pop("required_eval_ids", ("EV-P5-006",)), **fields)

    def read(self, ref: ArtifactRef | dict[str, str]) -> dict[str, Any]:
        held = ArtifactRef.model_validate(ref)
        envelope = self.deps.products.load_envelope(held.artifact_id)
        raw = self.deps.objects.get(envelope.payload_locator)
        if (envelope.analysis_id != self.analysis_id or envelope.content_hash != held.content_hash
                or hashlib.sha256(raw).hexdigest() != held.content_hash):
            raise persistence.PersistenceError("source changed or belongs to another analysis",
                                               "artifact_hash_mismatch")
        return cast(dict[str, Any], json.loads(raw))

    def commit(self, kind: str, body: dict[str, Any], parents: tuple[ArtifactRef, ...]) -> ArtifactRef:
        envelopes = tuple(self.deps.products.load_envelope(ref.artifact_id) for ref in parents)
        registration = self.deps.registry.lookup(kind)
        body = {"schema_version": registration.schema_version, **body}
        built = persistence.build_envelope(
            self.deps.registry, kind, body, analysis_id=self.analysis_id,
            stage_run_id=self.stage_run_id, producer_version=VERSION,
            producer_component=COMPONENT, parents=envelopes, created_at_utc=self.deps.clock())
        ref = ArtifactRef(artifact_id=built.artifact_id, content_hash=built.content_hash)
        self.deps.committer.commit(built, body, self.event(
            "artifact.committed", status="committed", artifact_refs=(ref,),
            required_eval_ids=(EVALUATIONS[kind],)))
        return ref

    def freeze_files(self, rendered: dict[str, Any]) -> dict[str, Any]:
        """Every delivered/preview byte lives in the shared content-addressed store."""
        result = dict(rendered)
        objects = {}
        for key, path in rendered["objects"].items():
            raw = Path(path).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != rendered["object_hashes"][key]:
                raise ValueError("render changed before commit")
            objects[key] = self.deps.objects.put_if_absent(digest, raw)
        result["objects"] = objects
        return result

    def reserve_call(self, role: str) -> int:
        """Durable reservation BEFORE a provider request; a crash cannot refund spend."""
        with self.deps.conn.transaction():
            row = self.deps.conn.execute(
                "SELECT run_record FROM presentation.runs WHERE stage_run_id=%s FOR UPDATE",
                (self.stage_run_id,)).fetchone()
            record = dict(row[0])
            count = int(record.get("calls", 0))
            reviews = int(record.get("reviews", 0))
            if count >= self.deps.max_calls or (
                    role == "review" and reviews >= self.deps.max_reviews):
                raise ValueError("post_analysis_budget_exhausted")
            record.update(calls=count + 1, reviews=reviews + (role == "review"))
            self.deps.conn.execute(
                "UPDATE presentation.runs SET run_record=%s::jsonb,updated_at=%s WHERE stage_run_id=%s",
                (json.dumps(record), self.deps.clock(), self.stage_run_id))
        return count + 1
