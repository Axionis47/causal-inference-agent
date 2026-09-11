"""One model invocation per graph step; no hidden autonomous loop."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal.post_analysis.contracts import Action, Review
from causal.post_analysis.store import Store
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import AgentTaskEnvelopeV1, TaskBudgets, TaskStatus
from causal.shared.gateway import GatewayImage
from causal.shared.tracing import trace_span

RESOURCES = Path(__file__).parent / "resources"


def invoke(store: Store, role: str, context: ArtifactRef, payload: dict[str, Any],
           images: tuple[GatewayImage, ...] = ()) -> Action | Review:
    number = store.reserve_call(role)
    task_id = f"{store.stage_run_id}:{role}:{number}"
    output = Action if role == "author" else Review
    envelope = AgentTaskEnvelopeV1(
        envelope_id=f"env:{task_id}", schema_version="agent-task-envelope.v1",
        analysis_id=store.analysis_id, stage_run_id=store.stage_run_id,
        task_id=task_id, attempt_id=f"attempt:{task_id}", context_manifest=context,
        task_kind=f"post_analysis_{role}", scope_kind="report", scope_ids=(store.analysis_id,),
        parent_artifacts=(context,), allowed_evidence_ids=tuple(payload.get("evidence", {})),
        allowed_retrieval_ids=(), allowed_tool_ids=(
            "read_evidence", "render_visual", "render_dag", "write_report", "submit", "stop"
        ) if role == "author" else (),
        output_schema_version=f"post-analysis-{role}.v1", validator_version="post-analysis.v1",
        prompt_version=f"post-analysis-{role}.v1", model_profile_version="vertex-model-profile.v1",
        budgets=TaskBudgets(token_budget=16000, tool_call_budget=1 if role == "author" else 0,
                            transient_attempt_budget=1, correction_budget=0),
        allowed_stopping_states=(TaskStatus.COMPLETE,), error_vocabulary=("invalid_action",),
        forbidden_payload_classes=("credentials",), payload_type="post-analysis-context.v1",
        payload=payload)
    prompt = (RESOURCES / f"{role}.v1.txt").read_text() + "\n" + json.dumps(payload, ensure_ascii=False)
    with trace_span(store.deps.tracer, f"post_analysis.{role}", inputs={"prompt": prompt},
                    metadata={"analysis_id": store.analysis_id, "task_id": task_id}) as span:
        event_fields = {"task_id": task_id, "attempt_id": envelope.attempt_id,
                        "required_eval_ids": ("EV-P5-002" if role == "author" else "EV-P5-005",)}
        store.deps.emitter.emit(store.event("task.started", **event_fields))
        kwargs = {"images": images} if images else {}
        result = store.deps.gateway.invoke(envelope, prompt, output.model_json_schema(), **kwargs)
        parsed = output.model_validate_json(result.text)
        store.deps.emitter.emit(store.event("task.completed", status="complete", **event_fields))
        span.finish({"output": parsed.model_dump(mode="json"),
                     "decision_summary": parsed.decision_summary})
        return parsed
