"""Tests for OperationalEventV1 and EventEmitter (T-003)."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from causal.shared.events import (
    EVENT_NAMES_V1,
    EventEmitter,
    EventEmitterError,
    OperationalEventV1,
    Severity,
    Stage,
)

NOW = datetime(2026, 8, 24, 14, 0, 0, 500000, tzinfo=UTC)


def event_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "operational-event.v1",
        "occurred_at_utc": NOW,
        "severity": Severity.INFO,
        "event_name": "stage.started",
        "event_id": "ev-1",
        "parent_event_id": None,
        "analysis_id": "an-1",
        "stage": Stage.INTAKE,
        "stage_run_id": "run-1",
        "graph_thread_id": None,
        "task_id": None,
        "attempt_id": None,
        "attempt_number": None,
        "component_id": "intake-coordinator",
        "component_version": "0.1.0",
        "versions": {},
        "status": None,
        "error_code": None,
        "retryable": None,
        "duration_ms": None,
        "token_usage": {},
        "cost": None,
        "artifact_refs": (),
        "required_eval_ids": (),
        "evaluation_run_id": None,
        "evaluation_case_id": None,
        "evaluation_fixture_hash": None,
        "evaluator_version": None,
        "evaluation_gate_status": None,
        "exception_class": None,
        "exception_fingerprint": None,
        "safe_dimensions": {},
    }
    base.update(overrides)
    return base


def make_event(**overrides: object) -> OperationalEventV1:
    return OperationalEventV1(**event_kwargs(**overrides))  # type: ignore[arg-type]


class TestModel:
    def test_registry_has_28_names(self) -> None:
        assert len(EVENT_NAMES_V1) == 28

    def test_valid_event_roundtrip(self) -> None:
        event = make_event()
        assert OperationalEventV1.model_validate(event.model_dump()) == event

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_event(surprise="x")

    @pytest.mark.parametrize("bad", ["", "noperiods", "Bad.Case", "trailing."])
    def test_bad_event_name_shape_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_event(event_name=bad)

    def test_unknown_version_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_event(versions={"flavor": "x"})

    def test_known_version_key_accepted(self) -> None:
        assert make_event(versions={"model": "gemini-2.5-flash"}).versions

    def test_unknown_token_usage_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_event(token_usage={"cached": 5})

    def test_negative_token_usage_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_event(token_usage={"input": -1})


class TestEmitter:
    def _emit(self, event: OperationalEventV1) -> str:
        sink = io.StringIO()
        EventEmitter(sink).emit(event)
        return sink.getvalue()

    def test_emits_one_parseable_line(self) -> None:
        output = self._emit(make_event())
        assert output.endswith("\n") and output.count("\n") == 1
        parsed = json.loads(output)
        assert parsed["event_name"] == "stage.started"
        assert parsed["occurred_at_utc"] == "2026-08-24T14:00:00.500000Z"

    def test_emission_is_deterministic(self) -> None:
        assert self._emit(make_event()) == self._emit(make_event())

    def test_unregistered_name_rejected(self) -> None:
        event = make_event(event_name="custom.thing")
        with pytest.raises(EventEmitterError) as excinfo:
            EventEmitter(io.StringIO()).emit(event)
        assert excinfo.value.code == "unregistered_event_name"

    def test_extra_registered_names_allowed(self) -> None:
        emitter = EventEmitter(io.StringIO(), EVENT_NAMES_V1 | {"custom.thing"})
        emitter.emit(make_event(event_name="custom.thing"))

    @pytest.mark.parametrize(
        "name", ["task.started", "agent.started", "tool.denied", "handoff.accepted"]
    )
    def test_eval_ids_required_for_gated_prefixes(self, name: str) -> None:
        with pytest.raises(EventEmitterError) as excinfo:
            EventEmitter(io.StringIO()).emit(make_event(event_name=name))
        assert excinfo.value.code == "missing_required_eval_ids"

    def test_gated_prefix_with_eval_ids_emits(self) -> None:
        output = self._emit(make_event(event_name="task.started", required_eval_ids=("EV-SYS-001",)))
        assert json.loads(output)["required_eval_ids"] == ["EV-SYS-001"]

    def test_stage_event_without_eval_ids_emits(self) -> None:
        assert json.loads(self._emit(make_event()))["required_eval_ids"] == []
