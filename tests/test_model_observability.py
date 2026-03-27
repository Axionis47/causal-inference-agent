"""Provider response bytes and metadata survive parsing without entering model schemas."""

from hashlib import sha256
from types import SimpleNamespace

import pytest
from google.genai import types
from pydantic import BaseModel, ValidationError

from backend import model_access
from backend.analysis import provider


class Reply(BaseModel):
    count: int


def response(raw, *, reason="STOP", usage=True):
    return types.GenerateContentResponse(
        response_id="provider-response-123", model_version="resolved-model-version",
        candidates=[types.Candidate(
            index=0, finish_reason=reason, finish_message="Provider stop detail",
            content=types.Content(parts=[types.Part(text=raw)] if raw is not None else []),
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30,
            cached_content_token_count=50, total_token_count=150,
        ) if usage else None,
    )


def client(monkeypatch, provider_response):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["http_options"].retry_options.attempts == 1
            self.models = SimpleNamespace(generate_content=self.generate)

        def generate(self, **kwargs):
            calls.append(kwargs)
            if isinstance(provider_response, Exception):
                raise provider_response
            return provider_response

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(model_access.genai, "Client", Client)
    return calls


def test_success_preserves_exact_text_and_metadata_outside_typed_model(monkeypatch):
    raw = ' \n { "count" : 7 } \n\t'
    calls = client(monkeypatch, response(raw))
    value = model_access.structured(model_access.prepare(Reply, "A typed test", {}, "requested-model", max_output_tokens=321))
    assert type(value) is Reply and value.count == 7
    assert value.model_dump() == {"count": 7}
    assert "_provider_observation" not in value.model_dump_json()
    assert "_provider_observation" not in str(value.model_json_schema())
    saved_raw, metrics = model_access.response_artifacts(value)
    assert saved_raw == raw and saved_raw != value.model_dump_json()
    assert metrics["provider_response_id"] == "provider-response-123"
    assert metrics["provider_model_version"] == "resolved-model-version"
    assert metrics["requested_model"] == "requested-model"
    assert metrics["finish_reason"] == "STOP" and metrics["finish_message"] == "Provider stop detail"
    assert metrics["input_tokens"] == 100 and metrics["output_tokens"] == 20
    assert metrics["reasoning_tokens"] == 30 and metrics["total_tokens"] == 150
    assert metrics["cached_input_tokens"] == 50
    assert metrics["output_limit"] == calls[0]["config"].max_output_tokens == 321
    assert calls[0]["config"].thinking_config is None
    assert metrics["thinking_config"] is None and metrics["thinking_limit_kind"] == "provider_default"
    assert metrics["response_text_hash"] == sha256(raw.encode()).hexdigest()
    assert metrics["response_bytes"] == len(raw.encode())
    assert len(calls) == 1
    metrics["usage"]["prompt_token_count"] = -1
    assert model_access.response_artifacts(value)[1]["usage"]["prompt_token_count"] == 100


def test_invalid_json_keeps_finish_reason_usage_and_exact_original_text(monkeypatch):
    raw = '\n{"count": '
    calls = client(monkeypatch, response(raw, reason="MAX_TOKENS"))
    with pytest.raises(model_access.InvalidModelOutput) as caught:
        model_access.structured(model_access.prepare(Reply, "A typed test", {}, "requested-model", max_output_tokens=50))
    saved_raw, metrics = model_access.response_artifacts(caught.value)
    assert caught.value.raw == saved_raw == raw
    assert metrics["finish_reason"] == "MAX_TOKENS" and metrics["output_limit"] == 50
    assert metrics["usage"]["thoughts_token_count"] == 30
    assert "json_invalid" in str(caught.value) and len(calls) == 1


@pytest.mark.parametrize("raw,reason", [(None, "SAFETY"), ('{"count":' + "0" * (2 * 1024 * 1024), "MAX_TOKENS")])
def test_nonparseable_provider_responses_keep_observation_before_local_rejection(monkeypatch, raw, reason):
    calls = client(monkeypatch, response(raw, reason=reason, usage=False))
    with pytest.raises(model_access.ProviderResponseError) as caught:
        model_access.structured(model_access.prepare(Reply, "A typed test", {}, "requested-model"))
    saved_raw, metrics = model_access.response_artifacts(caught.value)
    assert saved_raw == raw and metrics["finish_reason"] == reason
    assert metrics["usage"] is None and metrics["input_tokens"] is None
    assert len(calls) == 1


def test_transport_failure_does_not_invent_provider_response_or_retry(monkeypatch):
    calls = client(monkeypatch, OSError("Transport did not return a response"))
    with pytest.raises(OSError) as caught:
        model_access.structured(model_access.prepare(Reply, "A typed test", {}, "requested-model"))
    assert model_access.response_artifacts(caught.value) == (None, {})
    assert len(calls) == 1


def test_plain_model_and_legacy_error_mocks_keep_existing_fallback_contract():
    value = Reply(count=7)
    assert model_access.response_artifacts(value) == (value.model_dump_json(), {})
    try:
        Reply.model_validate_json('{"count":')
    except ValidationError as error:
        invalid = model_access.InvalidModelOutput('{"count":', error)
    assert model_access.response_artifacts(invalid) == ('{"count":', {})


@pytest.mark.parametrize("limit,budget", [(4096, 1024), (8192, 2048), (16384, 2048)])
def test_explicit_analysis_allocation_matches_provider_config_and_observation(monkeypatch, limit, budget):
    configured = provider.thinking_config("gemini-2.5-flash", limit)
    assert configured == {"thinking_budget": budget, "include_thoughts": False}
    calls = client(monkeypatch, response('{"count":7}'))
    value = model_access.structured(model_access.prepare(Reply, "A typed test", {}, "gemini-2.5-flash",
                                    max_output_tokens=limit, thinking_budget=configured["thinking_budget"]))
    sent = calls[0]["config"]
    assert sent.max_output_tokens == limit
    assert sent.thinking_config.model_dump(exclude_none=True) == configured
    metrics = model_access.response_artifacts(value)[1]
    assert metrics["thinking_config"] == configured and metrics["thinking_limit_kind"] == "soft"
    assert metrics["output_limit"] == limit and len(calls) == 1


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-2.5-flash-lite", "publishers/google/models/gemini-2.5-flash"])
def test_allocation_only_recognizes_supported_text_models(model):
    assert provider.thinking_config(model, 8192) == {"thinking_budget": 2048, "include_thoughts": False}


@pytest.mark.parametrize("model", ["gemini-3-flash", "gemini-2.5-flash-image", "unconfigured-model"])
def test_other_model_families_receive_no_automatic_thinking_allocation(model):
    assert provider.thinking_config(model, 8192) is None


@pytest.mark.parametrize("model,limit,budget", [("gemini-3-flash", 8192, 2048), ("gemini-2.5-flash", 1024, 1024)])
def test_invalid_explicit_allocation_never_dispatches(monkeypatch, model, limit, budget):
    calls = client(monkeypatch, response('{"count":7}'))
    with pytest.raises(ValueError, match="Explicit thinking budgets"):
        model_access.structured(model_access.prepare(Reply, "A typed test", {}, model, max_output_tokens=limit, thinking_budget=budget))
    assert not calls
