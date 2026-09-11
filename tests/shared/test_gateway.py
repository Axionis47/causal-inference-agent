"""Tests for the shared Vertex gateway (T-010, EV-SYS-003 unit layer)."""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime
from typing import Any

import pytest
from google.genai import types

from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import AgentTaskEnvelopeV1, TaskBudgets, TaskStatus
from causal.shared.events import EventEmitter
from causal.shared.gateway import (
    MODEL_OUTPUT_TRUNCATED,
    VERTEX_PROFILE_V1,
    GatewayError,
    GenerationSettingsV1,
    TransportError,
    TransportResponse,
    VertexGateway,
    _provider_schema,
    _to_transport_response,
    derive_seed,
)

NOW = datetime(2026, 8, 24, 14, 0, 0, tzinfo=UTC)
HASH = "a" * 64
SCHEMA: dict[str, object] = {"type": "object", "properties": {"answer": {"type": "string"}}}
USAGE = {"input": 11, "output": 22, "thinking": 33, "total": 66}


def make_envelope(task_id: str = "task-1", attempts: int = 3) -> AgentTaskEnvelopeV1:
    ref = ArtifactRef(artifact_id="art-1", content_hash=HASH)
    return AgentTaskEnvelopeV1(
        envelope_id="env-1", schema_version="agent-task-envelope.v1", analysis_id="an-1",
        stage_run_id="run-1", task_id=task_id, attempt_id="attempt-1", context_manifest=ref,
        task_kind="column_card", scope_kind="column", scope_ids=("stores.promo_flag",),
        parent_artifacts=(ref,), allowed_evidence_ids=("ev-1",), allowed_retrieval_ids=(),
        allowed_tool_ids=(), output_schema_version="column-semantic-card.v1",
        validator_version="column-semantic-card-validator.v1", prompt_version="column-card.v1",
        model_profile_version="vertex-model-profile.v1",
        budgets=TaskBudgets(token_budget=8000, tool_call_budget=0, transient_attempt_budget=attempts),
        allowed_stopping_states=(TaskStatus.COMPLETE,), error_vocabulary=("SCHEMA_INVALID",),
        forbidden_payload_classes=("raw_rows",), payload_type="column-card-request.v1",
        payload={"column_name": "promo_flag"})


class FakeTransport:
    """Records every settings object it is handed and replays a scripted outcome list."""

    def __init__(self, outcomes: list[TransportResponse | TransportError]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str, GenerationSettingsV1]] = []

    def generate(
        self, model_id: str, prompt: str, settings: GenerationSettingsV1
    ) -> TransportResponse:
        self.calls.append((model_id, prompt, settings))
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, TransportError):
            raise outcome
        return outcome


class Span:
    def __init__(self) -> None:
        self.finished: list[tuple[dict[str, object], str | None]] = []

    def finish(self, outputs: Any = None, error_code: str | None = None) -> None:
        self.finished.append((dict(outputs or {}), error_code))


class Tracer:
    def __init__(self) -> None:
        self.metadata: dict[str, object] = {}
        self.span = Span()

    def preflight(self) -> None: pass
    def flush(self) -> None: pass

    def start_gateway_span(self, metadata: Any) -> Span:
        self.metadata = dict(metadata)
        return self.span


def ok(text: str = '{"answer": "yes"}') -> TransportResponse:
    return TransportResponse(text=text, token_usage=dict(USAGE), finish_reason="STOP")


def transient() -> TransportError:
    return TransportError("503 UNAVAILABLE", "model_unavailable", retryable=True)


def build(
    outcomes: list[TransportResponse | TransportError],
) -> tuple[VertexGateway, FakeTransport, io.StringIO]:
    transport = FakeTransport(outcomes)
    sink = io.StringIO()
    gateway = VertexGateway(transport, VERTEX_PROFILE_V1, EventEmitter(sink), lambda: NOW)
    return gateway, transport, sink


def events(sink: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in sink.getvalue().splitlines()]


class TestProfileKnobs:
    def test_every_knob_lands_on_the_transport(self) -> None:
        gateway, transport, _ = build([ok()])
        gateway.invoke(make_envelope(), "hello", SCHEMA)
        settings = transport.calls[0][2]
        assert transport.calls[0][0] == "gemini-2.5-flash"
        assert settings.temperature == 0.0
        assert settings.candidate_count == 1
        assert settings.thinking_budget_tokens == 4096
        assert settings.max_output_tokens == 16384
        assert settings.response_mime_type == "application/json"
        assert settings.automatic_function_calling is False
        assert settings.response_schema == SCHEMA

    def test_profile_pins_sdk_and_surface(self) -> None:
        assert VERTEX_PROFILE_V1.sdk == "google-genai==2.19.0"
        assert (VERTEX_PROFILE_V1.api_surface, VERTEX_PROFILE_V1.location) == ("v1", "us-central1")
        assert VERTEX_PROFILE_V1.authentication == "adc"
        assert VERTEX_PROFILE_V1.sdk_retries is False


class TestSeed:
    def test_stable_for_one_task(self) -> None:
        assert derive_seed("task-1") == derive_seed("task-1")

    def test_distinct_across_tasks(self) -> None:
        assert len({derive_seed(f"task-{n}") for n in range(64)}) == 64

    @pytest.mark.parametrize("task_id", ["task-1", "", "an-1:column:promo_flag", "é"])
    def test_within_non_negative_31_bit_range(self, task_id: str) -> None:
        assert 0 <= derive_seed(task_id) < 2**31

    def test_value_above_signed_int32_is_masked(self) -> None:
        # sha256("an-1:column:promo_flag")[:8] is 0x8a001a3a, past the signed INT32 ceiling.
        assert derive_seed("an-1:column:promo_flag") == 0x0A001A3A
        settings = GenerationSettingsV1(
            temperature=0.0, candidate_count=1, seed=derive_seed("an-1:column:promo_flag"),
            thinking_budget_tokens=8192, max_output_tokens=16384,
            response_mime_type="application/json", response_schema=SCHEMA,
            automatic_function_calling=False)
        assert settings.seed == 0x0A001A3A

    def test_result_carries_the_task_seed(self) -> None:
        gateway, transport, _ = build([ok()])
        result = gateway.invoke(make_envelope(), "hello", SCHEMA)
        assert result.seed == derive_seed("task-1") == transport.calls[0][2].seed

    def test_evaluation_seed_key_stabilizes_runs_with_different_task_ids(self) -> None:
        gateway, transport, _ = build([ok(), ok()])
        for task_id in ("random-task-1", "random-task-2"):
            envelope = make_envelope(task_id).model_copy(update={"payload": {
                "evaluation": {"seed_key": "fixed-case:intent:scope"}}})
            assert gateway.invoke(envelope, "hello", SCHEMA).seed == derive_seed(
                "fixed-case:intent:scope")
        assert transport.calls[0][2].seed == transport.calls[1][2].seed


class TestRetries:
    def test_two_transient_failures_then_success(self) -> None:
        gateway, transport, sink = build([transient(), transient(), ok()])
        result = gateway.invoke(make_envelope(), "hello", SCHEMA)
        assert result.attempts == 3
        assert len(transport.calls) == 3
        scheduled = [e for e in events(sink) if e["event_name"] == "retry.scheduled"]
        assert len(scheduled) == 2
        assert [e["attempt_number"] for e in scheduled] == [1, 2]
        assert scheduled[0]["stage"] == "system"
        assert scheduled[0]["component_id"] == "vertex-gateway"
        assert scheduled[0]["required_eval_ids"] == ["EV-SYS-003"]
        assert scheduled[0]["retryable"] is True
        assert scheduled[0]["task_id"] == "task-1"
        assert scheduled[0]["event_id"] == "evt:env-1:retry.scheduled:1"

    def test_exhaustion_emits_once_and_raises(self) -> None:
        gateway, transport, sink = build([transient()])
        with pytest.raises(GatewayError) as excinfo:
            gateway.invoke(make_envelope(attempts=3), "hello", SCHEMA)
        assert excinfo.value.code == "transient_exhausted"
        assert len(transport.calls) == 3
        names = [e["event_name"] for e in events(sink)]
        assert names == ["retry.scheduled", "retry.scheduled", "retry.exhausted"]

    def test_budget_of_one_makes_a_single_attempt(self) -> None:
        gateway, transport, sink = build([transient()])
        with pytest.raises(GatewayError):
            gateway.invoke(make_envelope(attempts=1), "hello", SCHEMA)
        assert len(transport.calls) == 1
        assert [e["event_name"] for e in events(sink)] == ["retry.exhausted"]

    def test_non_retryable_error_maps_one_to_one(self) -> None:
        failure = TransportError("401", "invalid_authentication", retryable=False)
        gateway, transport, sink = build([failure])
        with pytest.raises(GatewayError) as excinfo:
            gateway.invoke(make_envelope(), "hello", SCHEMA)
        assert excinfo.value.code == "invalid_authentication"
        assert len(transport.calls) == 1
        assert events(sink) == []


class TestResult:
    def test_valid_json_object_is_parsed(self) -> None:
        gateway, _, _ = build([ok()])
        result = gateway.invoke(make_envelope(), "hello", SCHEMA)
        assert result.parsed == {"answer": "yes"}
        assert result.attempts == 1
        assert result.finish_reason == "STOP"

    @pytest.mark.parametrize("text", ["not json at all", "", "[1, 2]", '"answer"', "{"])
    def test_non_object_text_parses_to_none(self, text: str) -> None:
        gateway, _, _ = build([ok(text)])
        result = gateway.invoke(make_envelope(), "hello", SCHEMA)
        assert result.parsed is None
        assert result.text == text

    def test_token_usage_passes_through(self) -> None:
        gateway, _, _ = build([ok()])
        assert gateway.invoke(make_envelope(), "hello", SCHEMA).token_usage == USAGE

    def test_result_is_frozen(self) -> None:
        gateway, _, _ = build([ok()])
        result = gateway.invoke(make_envelope(), "hello", SCHEMA)
        with pytest.raises(ValueError, match="frozen"):
            result.attempts = 9


def test_max_tokens_is_a_clear_nonretryable_transport_failure() -> None:
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason=types.FinishReason.MAX_TOKENS,
        content=types.Content(parts=[types.Part(text="{")]))])
    with pytest.raises(TransportError) as excinfo:
        _to_transport_response(response)
    assert excinfo.value.code == MODEL_OUTPUT_TRUNCATED
    assert excinfo.value.retryable is False


def test_provider_schema_preserves_compact_reference_enums() -> None:
    schema = {"type": "object", "x-causal-reference-kind": "column", "properties": {
        "value": {"type": "string", "enum": ["margin"],
                  "x-causal-reference-role": "reference"}}}
    assert _provider_schema(schema) == {"type": "object", "properties": {
        "value": {"type": "string", "enum": ["margin"]}}}


def test_provider_lowering_keeps_names_shape_and_enums_and_never_mutates_local_schema() -> None:
    schema = {"type": "object", "additionalProperties": False,
              "required": ["maximum", "title", "values"], "properties": {
        "maximum": {"type": "number", "minimum": 0, "maximum": 1},
        "title": {"type": "string", "minLength": 1, "maxLength": 200,
                  "pattern": "^[a-z]+$", "enum": ["a", "b"]},
        "values": {"type": "array", "minItems": 1, "maxItems": 12,
                   "items": {"anyOf": [{"type": "string", "format": "date-time"},
                                       {"type": "null"}]}},
        "forbidden": {"type": "array", "maxItems": 0, "items": {"type": "string"}}}}
    original = json.dumps(schema, sort_keys=True)
    lowered = _provider_schema(schema)
    assert json.dumps(schema, sort_keys=True) == original
    assert lowered["required"] == schema["required"]
    assert lowered["additionalProperties"] is False
    assert lowered["properties"] == {
        "maximum": {"type": "number", "minimum": 0, "maximum": 1},
        "title": {"type": "string", "enum": ["a", "b"]},
        "values": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"anyOf": [
            {"type": "string", "format": "date-time"}, {"type": "null"}]}},
        "forbidden": {"type": "array", "maxItems": 0, "items": {"type": "string"}}}


def test_gateway_span_captures_rendered_prompt_response_and_exposed_provider_summary() -> None:
    response = ok().model_copy(update={"reasoning": "The evidence supports a qualified answer."})
    transport, sink, tracer = FakeTransport([response]), io.StringIO(), Tracer()
    gateway = VertexGateway(transport, VERTEX_PROFILE_V1, EventEmitter(sink), lambda: NOW, tracer)
    envelope = make_envelope().model_copy(update={"payload": {
        "column_name": "promo_flag", "evaluation": {
            "run_id": "eval-1", "case_id": "case-1", "mode": "gate"}}})
    gateway.invoke(envelope, "Complete rendered application prompt", SCHEMA)
    assert tracer.metadata["evaluation_case_id"] == "case-1"
    assert {"prompt_hash", "envelope_hash", "response_schema_hash"} <= set(tracer.metadata)
    assert tracer.metadata["prompt"] == transport.calls[0][1]
    assert tracer.metadata["response_schema"] == SCHEMA
    assert tracer.span.finished == [({"physical_attempts": 1, "input_tokens": 11,
                                      "output_tokens": 22, "thinking_tokens": 33,
                                      "total_tokens": 66, "finish_reason": "STOP",
                                      "model_output": response.text,
                                      "provider_summary": response.reasoning}, None)]


def test_emitted_events_carry_the_gateway_identity() -> None:
    gateway, _, sink = build([transient(), ok()])
    gateway.invoke(make_envelope(), "hello", SCHEMA)
    emitted = events(sink)
    assert emitted and all(
        e["component_version"] == "vertex-gateway.v1"
        and e["versions"] == {"model": "gemini-2.5-flash"}
        and e["analysis_id"] == "an-1"
        and e["stage_run_id"] == "run-1"
        and e["attempt_id"] == "attempt-1"
        and e["safe_dimensions"] == {"operation": "vertex.generate"}
        for e in emitted
    )


def test_unexpected_transport_failure_closes_the_trace_without_changing_exception() -> None:
    class BrokenTransport:
        def generate(self, model_id: str, prompt: str,
                     settings: GenerationSettingsV1) -> TransportResponse:
            raise RuntimeError("unexpected adapter failure")

    tracer = Tracer()
    gateway = VertexGateway(BrokenTransport(), VERTEX_PROFILE_V1, EventEmitter(io.StringIO()),
                            lambda: NOW, tracer)
    with pytest.raises(RuntimeError, match="unexpected adapter failure"):
        gateway.invoke(make_envelope(), "Review", SCHEMA)
    assert tracer.span.finished == [({"physical_attempts": 1}, "RuntimeError")]


@pytest.mark.skipif(not os.environ.get("RUN_LIVE_VERTEX"), reason="RUN_LIVE_VERTEX is unset")
def test_live_vertex_structured_output() -> None:
    from causal.shared.gateway import GenAiTransport

    schema: dict[str, object] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    gateway = VertexGateway(
        GenAiTransport(), VERTEX_PROFILE_V1, EventEmitter(io.StringIO()),
        lambda: datetime.now(UTC))
    result = gateway.invoke(
        make_envelope(task_id="live-smoke"),
        "Reply with a JSON object whose 'answer' field is the word yes.", schema)
    assert isinstance(result.parsed, dict)
    assert isinstance(result.parsed["answer"], str)
    assert result.token_usage["total"] > 0
