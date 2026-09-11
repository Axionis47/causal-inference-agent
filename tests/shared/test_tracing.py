"""Trace redactor, tracer protocol, and the commit flush gate (T-010)."""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from typing import Any, Final

import psycopg
import pytest

from causal.shared.events import EventEmitter
from causal.shared.persistence import ArtifactCommitter, ObjectStore, ProductStore
from causal.shared.registry import ArtifactTypeRegistry
from causal.shared.tracing import (
    API_KEY_ENV_VARS,
    REDACTION_PATTERNS_V1,
    SAFE_METADATA_KEYS_V1,
    LangSmithTracer,
    ObservabilityError,
    TraceRedactorV1,
    TracerProtocol,
)
from tests.infrastructure import requires_docker
from tests.shared.test_persistence import committed_event, envelope_for, registration

# One canary per pattern class: (label, text, the secret that must not survive).
CANARIES: Final[tuple[tuple[str, str, str], ...]] = (
    ("aws_key", "rotate AKIAIOSFODNN7EXAMPLE now", "AKIAIOSFODNN7EXAMPLE"),
    ("signed_url", "https://bucket/obj?X-Amz-Signature=deadbeefcafe", "deadbeefcafe"),
    ("bearer", "Authorization header Bearer abc.def", "abc.def"),
    ("authority_credentials", "postgres://user:hunter2@host/db", "hunter2"),
    ("api_key_assignment", "api_key=sk-123", "sk-123"),
    ("api_key_assignment_colon", "token: xyz", "xyz"),
)

CLEAN_TEXT: Final = "The design harness selected regression discontinuity for task t-9."


class FakeTracer:
    """Minimal TracerProtocol double recording calls and raising on demand."""

    def __init__(
        self,
        *,
        preflight_error: ObservabilityError | None = None,
        flush_error: ObservabilityError | None = None,
    ) -> None:
        self.preflights = 0
        self.flushes = 0
        self._preflight_error = preflight_error
        self._flush_error = flush_error

    def preflight(self) -> None:
        self.preflights += 1
        if self._preflight_error is not None:
            raise self._preflight_error

    def flush(self) -> None:
        self.flushes += 1
        if self._flush_error is not None:
            raise self._flush_error

    def start_gateway_span(self, metadata: Any) -> Any:
        raise AssertionError("the persistence tests never create model spans")


def make_traced_committer(
    object_store: ObjectStore,
    conn: psycopg.Connection[Any],
    tracer: TracerProtocol | None,
) -> tuple[ArtifactCommitter, ProductStore]:
    products = ProductStore(conn)
    products.create_stage_run("run-1", "an-1", "intake")
    committer = ArtifactCommitter(
        object_store, products, ArtifactTypeRegistry((registration(),)),
        EventEmitter(io.StringIO()), tracer=tracer,
    )
    return committer, products


class TestTraceRedactor:
    @pytest.mark.parametrize(("label", "text", "secret"), CANARIES)
    def test_canary_never_survives(self, label: str, text: str, secret: str) -> None:
        redacted = TraceRedactorV1().redact_text(text)
        assert secret not in redacted
        assert "[REDACTED:" in redacted

    def test_every_pattern_class_has_a_canary(self) -> None:
        classes = {name for name, _ in REDACTION_PATTERNS_V1}
        assert classes <= {label.split("_colon")[0] for label, _, _ in CANARIES}

    def test_clean_text_passes_unchanged(self) -> None:
        assert TraceRedactorV1().redact_text(CLEAN_TEXT) == CLEAN_TEXT

    def test_policy_version_is_pinned(self) -> None:
        assert TraceRedactorV1().redaction_policy_version == "trace-redaction.v1"

    def test_metadata_allowlist_and_scalar_only(self) -> None:
        safe = TraceRedactorV1().redact_metadata(
            {
                "environment": "production",
                "analysis_id": "an-1",
                "correction_attempt_count": 2,
                "final_outcome": "usable",
                "validation_status": "failed api_key=sk-123",
                "csv_rows": ["a", "b"],
                "user_prompt": "postgres://user:hunter2@host/db",
                "task_id": {"nested": 1},
            }
        )
        assert safe == {
            "environment": "production",
            "analysis_id": "an-1",
            "correction_attempt_count": 2,
            "final_outcome": "usable",
            "validation_status": "failed [REDACTED:api_key_assignment]",
        }

    def test_allowlist_covers_the_prd_safe_keys(self) -> None:
        assert {"environment", "graph_thread_id", "prompt_version", "final_outcome"} <= (
            SAFE_METADATA_KEYS_V1
        )
        assert "csv_rows" not in SAFE_METADATA_KEYS_V1


class TestTracerProtocol:
    def test_fake_tracer_satisfies_the_protocol(self) -> None:
        tracer: TracerProtocol = FakeTracer()
        tracer.preflight()
        tracer.flush()

    @pytest.mark.parametrize(("method", "kwargs", "code"), (
        ("preflight", {"preflight_error": ObservabilityError("down", "preflight_failed")},
         "preflight_failed"),
        ("flush", {"flush_error": ObservabilityError("lost", "flush_unacknowledged")},
         "flush_unacknowledged")))
    def test_failure_code(self, method: str, kwargs: dict[str, ObservabilityError], code: str) -> None:
        tracer = FakeTracer(**kwargs)
        with pytest.raises(ObservabilityError) as excinfo:
            getattr(tracer, method)()
        assert excinfo.value.code == code


class TestLangSmithTracer:
    def test_preflight_without_api_key_never_touches_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in API_KEY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        built: list[str] = []

        def factory(callback: Callable[[Exception], None]) -> Any:
            built.append("client")
            return object()

        tracer = LangSmithTracer(
            "causal-test", "test", TraceRedactorV1(), client_factory=factory
        )
        with pytest.raises(ObservabilityError) as excinfo:
            tracer.preflight()
        assert excinfo.value.code == "preflight_failed"
        assert built == []

    def test_preflight_call_failure_is_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-not-a-real-key")

        class RefusingClient:
            def has_project(self, project_name: str) -> bool:
                raise RuntimeError("401 unauthorized")

        tracer = LangSmithTracer(
            "causal-test", "test", TraceRedactorV1(),
            client_factory=lambda callback: RefusingClient(),
        )
        with pytest.raises(ObservabilityError) as excinfo:
            tracer.preflight()
        assert excinfo.value.code == "preflight_failed"

    def test_gateway_span_sends_only_allowlisted_scalars(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.created: dict[str, Any] = {}
                self.updated: dict[str, Any] = {}

            def create_run(self, name: str, inputs: Any, run_type: str, **kwargs: Any) -> None:
                self.created = {"name": name, "inputs": inputs, "run_type": run_type} | kwargs

            def update_run(self, run_id: Any, **kwargs: Any) -> None:
                self.updated = {"run_id": run_id} | kwargs

        client = Client()
        tracer = LangSmithTracer("causal-test", "test", TraceRedactorV1(),
                                 client_factory=lambda callback: client)
        span = tracer.start_gateway_span({"analysis_id": "an-1", "prompt_hash": "abc",
                                          "raw_rows": [1], "gold_labels": "forbidden"})
        span.finish({"total_tokens": 4, "reasoning": "hidden"})
        assert client.created["inputs"] == {
            "analysis_id": "an-1", "prompt_hash": "abc", "environment": "test"}
        assert client.updated["outputs"] == {"total_tokens": 4}

    def test_gateway_span_finish_failure_is_fail_closed(self) -> None:
        class Client:
            def create_run(self, *args: Any, **kwargs: Any) -> None:
                pass

            def update_run(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("trace rejected")

        tracer = LangSmithTracer("causal-test", "test", TraceRedactorV1(),
                                 client_factory=lambda callback: Client())
        span = tracer.start_gateway_span({"analysis_id": "an-1"})
        with pytest.raises(ObservabilityError) as excinfo:
            span.finish({"total_tokens": 4})
        assert excinfo.value.code == "flush_unacknowledged"


@requires_docker
class TestCommitFlushGate:
    def test_no_tracer_is_the_default(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        products = ProductStore(conn)
        committer = ArtifactCommitter(
            object_store, products, ArtifactTypeRegistry((registration(),)),
            EventEmitter(io.StringIO()),
        )
        assert committer._tracer is None

    def test_commit_flushes_an_attached_tracer_once(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        tracer = FakeTracer()
        committer, products = make_traced_committer(object_store, conn, tracer)
        payload: dict[str, object] = {"k": "v"}
        envelope = envelope_for(payload)
        assert committer.commit(envelope, payload, committed_event()) == envelope
        assert tracer.flushes == 1
        assert products.load_envelope("art-1") == envelope

    def test_flush_failure_raises_but_preserves_the_artifact(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        tracer = FakeTracer(
            flush_error=ObservabilityError("batch lost", "flush_unacknowledged")
        )
        committer, products = make_traced_committer(object_store, conn, tracer)
        payload: dict[str, object] = {"k": "v"}
        envelope = envelope_for(payload)
        with pytest.raises(ObservabilityError) as excinfo:
            committer.commit(envelope, payload, committed_event())
        assert excinfo.value.code == "flush_unacknowledged"
        assert products.load_envelope("art-1") == envelope
        assert object_store.get(envelope.payload_locator) == b'{"k":"v"}'


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_LANGSMITH"), reason="RUN_LIVE_LANGSMITH is unset"
)
def test_live_preflight_flush() -> None:
    tracer = LangSmithTracer(
        os.environ.get("LANGSMITH_PROJECT", "causal-dev"), "development", TraceRedactorV1()
    )
    tracer.preflight()
    tracer.flush()
