"""Documented temporary-capacity 429s use only the existing same-identity retry bound."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors, types

from causal.shared import gateway as gw
from causal.shared.events import EventEmitter
from tests.shared.test_gateway import NOW, SCHEMA, events, make_envelope

CAPACITY = "Resource exhausted, please try again later."


def api_error(message: str = CAPACITY, *, details: list[dict[str, Any]] | None = None,
              status: str = "RESOURCE_EXHAUSTED", code: int = 429) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "status": status, "message": message,
                                          "details": details or []}})


def build(monkeypatch: pytest.MonkeyPatch, failures: list[errors.APIError]) -> tuple[Any, ...]:
    calls, waits = [], []
    monkeypatch.setattr("time.sleep", waits.append)

    def generate(**kwargs: Any) -> types.GenerateContentResponse:
        calls.append(kwargs)
        if failures:
            raise failures.pop(0)
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text='{"answer":"yes"}')]), finish_reason="STOP")])

    transport = gw.GenAiTransport()
    transport._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))  # type: ignore[assignment]
    sink = io.StringIO()
    gateway = gw.VertexGateway(transport, gw.VERTEX_PROFILE_V1, EventEmitter(sink), lambda: NOW)
    return gateway, calls, waits, sink


def test_explicit_capacity_then_success_reuses_exact_request_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, calls, waits, sink = build(monkeypatch, [api_error(), api_error()])
    envelope = make_envelope()
    result = gateway.invoke(envelope, "exact stable prompt", SCHEMA)
    assert result.parsed == {"answer": "yes"}
    assert result.attempts == len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert calls[0]["model"] == gw.VERTEX_PROFILE_V1.model_id
    assert calls[0]["config"].seed == gw.derive_seed(envelope.task_id)
    assert waits == [1, 2]
    assert [row["event_name"] for row in events(sink)] == ["retry.scheduled"] * 2
    assert all(row["task_id"] == envelope.task_id and row["attempt_id"] == envelope.attempt_id
               for row in events(sink))
    assert CAPACITY not in sink.getvalue()  # Provider bodies do not enter logs.


@pytest.mark.parametrize("declared_budget", (1, 3, 99))
def test_capacity_exhaustion_honors_declared_budget_and_three_attempt_ceiling(
    monkeypatch: pytest.MonkeyPatch, declared_budget: int,
) -> None:
    gateway, calls, waits, sink = build(monkeypatch, [api_error() for _ in range(4)])
    with pytest.raises(gw.GatewayError) as raised:
        gateway.invoke(make_envelope(attempts=declared_budget), "stable", SCHEMA)
    assert raised.value.code == gw.TRANSIENT_EXHAUSTED
    assert len(calls) == min(3, declared_budget)
    assert waits == ([1, 2] if declared_budget > 1 else [])
    assert events(sink)[-1]["event_name"] == "retry.exhausted"
    assert CAPACITY not in sink.getvalue()


@pytest.mark.parametrize("failure", (
    api_error("Quota exceeded for requests per day."),
    api_error("Unknown resource exhaustion"),
    api_error(details=[{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"subject": "quota", "description": "daily limit"}]}]),
    api_error(details=[{"unrecognized": "provider details require explicit classification"}]),
    api_error(status="PERMISSION_DENIED"),
))
def test_quota_unknown_or_structured_429_remains_one_attempt(
    monkeypatch: pytest.MonkeyPatch, failure: errors.APIError,
) -> None:
    gateway, calls, waits, sink = build(monkeypatch, [failure])
    with pytest.raises(gw.GatewayError) as raised:
        gateway.invoke(make_envelope(), "stable", SCHEMA)
    assert raised.value.code == (gw.PERMISSION_DENIED if failure.status == "PERMISSION_DENIED"
                                 else gw.QUOTA_EXHAUSTED)
    assert len(calls) == 1
    assert waits == []
    assert events(sink) == []
