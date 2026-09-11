"""Content-rich traces preserve application evidence and parentage without credentials."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import pytest

from causal.shared.events import EventEmitter
from causal.shared.gateway import VERTEX_PROFILE_V1, VertexGateway
from causal.shared.tracing import LangSmithTracer, ObservabilityError, TraceRedactorV1, trace_span
from tests.shared.test_gateway import NOW, SCHEMA, FakeTransport, make_envelope, ok
from tests.shared.test_tracing import FakeTracer


class RecordingClient:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.tracing_queue: Any = None

    def create_run(self, name: str, inputs: Any, run_type: str, **kwargs: Any) -> None:
        self.created.append({"name": name, "inputs": inputs, "run_type": run_type} | kwargs)

    def update_run(self, run_id: Any, **kwargs: Any) -> None:
        self.updated.append({"run_id": run_id} | kwargs)

    def flush(self, **kwargs: Any) -> None:
        pass


def tracer_for(client: RecordingClient) -> LangSmithTracer:
    return LangSmithTracer("causal-test", "test", TraceRedactorV1(),
                           client_factory=lambda callback: client)


def test_default_client_preserves_sanitized_content_and_disables_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: dict[str, Any] = {}

    def build(**kwargs: Any) -> RecordingClient:
        options.update(kwargs)
        return RecordingClient()

    monkeypatch.setattr("langsmith.Client", build)
    tracer = LangSmithTracer("causal-test", "test", TraceRedactorV1())
    tracer._client()
    assert options["hide_inputs"] is False
    assert options["hide_outputs"] is False
    assert options["tracing_sampling_rate"] == 1.0
    assert callable(options["tracing_error_callback"])


def test_nested_node_tool_gateway_records_complete_content_and_parentage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = RecordingClient()
    tracer = tracer_for(client)
    prompt = "Review all evidence: " + "study observation; " * 10000
    answer = ok().model_copy(update={"reasoning": "The provider returned this summary."})
    gateway = VertexGateway(FakeTransport([answer]), VERTEX_PROFILE_V1,
                            EventEmitter(io.StringIO()), lambda: NOW, tracer)
    with (
        caplog.at_level(logging.INFO, logger="causal.shared.tracing"),
        trace_span(tracer, "post_analysis", inputs={"question": "A causal effect?"}) as root,
    ):
        with tracer.span("review", inputs={"figure_ids": ["effect"]}) as node:
            with tracer.span("evidence.read", run_type="tool", inputs={"id": "effect"}) as tool:
                tool.finish({"rows": [{"group": "treated", "effect": 1.25}],
                             "credentials": {"api_key": "never-send-this"}})
            gateway.invoke(make_envelope(), prompt, SCHEMA)
            node.finish({"decision_summary": "The interval warrants qualification."})
        root.finish({"status": "complete"})
    assert [run["run_type"] for run in client.created] == ["chain", "chain", "tool", "llm"]
    root_run, node_run, tool_run, model_run = client.created
    assert root_run["parent_run_id"] is None
    assert node_run["parent_run_id"] == root_run["id"]
    assert tool_run["parent_run_id"] == model_run["parent_run_id"] == node_run["id"]
    assert all(run["trace_id"] == root_run["id"] for run in client.created)
    assert model_run["dotted_order"].startswith(node_run["dotted_order"] + ".")
    assert model_run["inputs"]["prompt"] == prompt
    assert model_run["inputs"]["messages"] == [{"role": "user", "content": prompt}]
    assert model_run["inputs"]["response_schema"] == SCHEMA
    assert len(client.updated) == 4  # explicit finish + context exit never double-updates
    outputs = {run["run_id"]: run["outputs"] for run in client.updated}
    assert outputs[tool_run["id"]]["rows"] == [{"group": "treated", "effect": 1.25}]
    assert outputs[model_run["id"]]["model_output"] == answer.text
    assert outputs[model_run["id"]]["provider_summary"] == answer.reasoning
    assert outputs[node_run["id"]]["decision_summary"].startswith("The interval")
    assert "never-send-this" not in json.dumps(client.updated, default=str)
    assert len(caplog.records) == 8
    assert all(record.message in {"trace.span.started", "trace.span.finished"}
               for record in caplog.records)
    assert prompt not in caplog.text


def test_recursive_redaction_keeps_study_fields_and_masks_nested_credentials() -> None:
    original = {"raw_rows": [{"patient": "Ada", "outcome": 2.0, "token_count": 4}],
                "settings": {"Authorization": "actual-credential", "password": "hunter2"},
                "response": '{"api_key": "do-not-send", "answer": "study data"}',
                "prompt": "Bearer abc.def", "decision_summary": "Use a qualified claim."}
    safe = TraceRedactorV1().redact_content(original)
    assert safe["raw_rows"] == original["raw_rows"]
    assert safe["decision_summary"] == original["decision_summary"]
    assert original["settings"]["password"] == "hunter2"  # type: ignore[index]
    serialized = json.dumps(safe)
    assert all(value not in serialized for value in
               ("actual-credential", "hunter2", "do-not-send", "abc.def"))


def test_opaque_image_base64_is_not_corrupted_by_text_credential_patterns() -> None:
    image = {"mime_type": "image/png", "base64": "AKIA" + "A" * 16}
    assert TraceRedactorV1().redact_content(image) == image


def test_failed_span_closes_and_restores_parent() -> None:
    client = RecordingClient()
    tracer = tracer_for(client)
    with tracer.span("root"):
        with (
            pytest.raises(ValueError, match="invalid"),
            tracer.span("failing-tool", run_type="tool"),
        ):
            raise ValueError("invalid")
        with tracer.span("next-tool", run_type="tool"):
            pass
    assert client.updated[0]["error"] == "ValueError"
    assert client.created[2]["parent_run_id"] == client.created[0]["id"]
    with tracer.span("separate-root"):
        pass
    assert client.created[3]["parent_run_id"] is None


def test_concurrent_tasks_do_not_adopt_each_others_parents() -> None:
    client = RecordingClient()
    tracer = tracer_for(client)

    async def branch(name: str) -> None:
        with tracer.span(name):
            await asyncio.sleep(0)
            with tracer.span(name + ".tool", run_type="tool"):
                pass

    async def run() -> None:
        await asyncio.gather(branch("a"), branch("b"))

    asyncio.run(run())
    runs = {run["name"]: run for run in client.created}
    for name in ("a", "b"):
        assert runs[name]["parent_run_id"] is None
        assert runs[name + ".tool"]["parent_run_id"] == runs[name]["id"]


@pytest.mark.parametrize("tracer", [None, FakeTracer()])
def test_optional_and_legacy_tracers_remain_compatible(tracer: Any) -> None:
    with trace_span(tracer, "optional", inputs={"value": 1}) as span:
        span.finish({"decision_summary": "Application summary."})


def test_nested_delivery_failure_propagates_and_context_is_restored() -> None:
    class RejectingClient(RecordingClient):
        def update_run(self, run_id: Any, **kwargs: Any) -> None:
            raise RuntimeError("token=never-send-this")

    client = RejectingClient()
    tracer = tracer_for(client)
    with pytest.raises(ObservabilityError) as failure, tracer.span("root"):
        pass
    assert failure.value.code == "flush_unacknowledged"
    assert "never-send-this" not in str(failure.value)
    assert tracer._active.get() is None


def test_flush_still_detects_rejected_batches_and_unfinished_queue() -> None:
    client = RecordingClient()
    tracer = tracer_for(client)
    tracer._record_delivery_error(RuntimeError("batch rejected"))
    with pytest.raises(ObservabilityError) as failure:
        tracer.flush()
    assert failure.value.code == "flush_unacknowledged"
    client.tracing_queue = type("Queue", (), {"unfinished_tasks": 1})()
    with pytest.raises(ObservabilityError) as failure:
        tracer.flush()
    assert failure.value.code == "flush_unacknowledged"
