"""The one Vertex model gateway: profile, seed, transport, transient retries (SC §10.4; T-010)."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Annotated, Final, Literal, Protocol

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from causal.shared.canonical import content_hash
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.events import EventEmitter, Severity, Stage, build_event
from causal.shared.tracing import TracerProtocol, TraceSpanProtocol

__all__ = [
    "GATEWAY_ERROR_CODES",
    "MODEL_OUTPUT_TRUNCATED",
    "UNSUPPORTED_IMAGE_INPUT",
    "VERTEX_PROFILE_V1",
    "GatewayError",
    "GatewayImage",
    "GatewayResultV1",
    "GenAiTransport",
    "GenerationSettingsV1",
    "ModelTransportProtocol",
    "TransportError",
    "TransportResponse",
    "VertexGateway",
    "VertexModelProfileV1",
    "derive_seed",
]

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)

COMPONENT_ID: Final = "vertex-gateway"
COMPONENT_VERSION: Final = "vertex-gateway.v1"
GATEWAY_EVAL_IDS: Final[tuple[str, ...]] = ("EV-SYS-003",)

PROVIDER_SAFETY_REJECTION: Final = "provider_safety_rejection"
MODEL_UNAVAILABLE: Final = "model_unavailable"
INVALID_AUTHENTICATION: Final = "invalid_authentication"
PERMISSION_DENIED: Final = "permission_denied"
QUOTA_EXHAUSTED: Final = "quota_exhausted"
UNSUPPORTED_STRUCTURED_OUTPUT: Final = "unsupported_structured_output"
MODEL_OUTPUT_TRUNCATED: Final = "model_output_truncated"
UNSUPPORTED_IMAGE_INPUT: Final = "unsupported_image_input"
TRANSIENT_EXHAUSTED: Final = "transient_exhausted"

GATEWAY_ERROR_CODES: Final[frozenset[str]] = frozenset({
    PROVIDER_SAFETY_REJECTION, MODEL_UNAVAILABLE, INVALID_AUTHENTICATION,
    PERMISSION_DENIED, QUOTA_EXHAUSTED, UNSUPPORTED_STRUCTURED_OUTPUT,
    MODEL_OUTPUT_TRUNCATED, UNSUPPORTED_IMAGE_INPUT, TRANSIENT_EXHAUSTED,
})

_STATUS_CODES: Final[dict[str, str]] = {
    "UNAUTHENTICATED": INVALID_AUTHENTICATION,
    "PERMISSION_DENIED": PERMISSION_DENIED,
    "RESOURCE_EXHAUSTED": QUOTA_EXHAUSTED,
    "INVALID_ARGUMENT": UNSUPPORTED_STRUCTURED_OUTPUT,
    "FAILED_PRECONDITION": UNSUPPORTED_STRUCTURED_OUTPUT,
}
_HTTP_CODES: Final[dict[int, str]] = {
    401: INVALID_AUTHENTICATION, 403: PERMISSION_DENIED, 429: QUOTA_EXHAUSTED,
    400: UNSUPPORTED_STRUCTURED_OUTPUT, 404: MODEL_UNAVAILABLE,
}
_SAFETY_FINISH_REASONS: Final[frozenset[str]] = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
)


class VertexModelProfileV1(BaseModel):
    model_config = _MODEL_CONFIG

    profile_version: Literal["vertex-model-profile.v1"] = "vertex-model-profile.v1"
    sdk: str = "google-genai==2.19.0"
    api_surface: str = "v1"
    model_id: str = "gemini-2.5-flash"
    location: str = "us-central1"
    authentication: str = "adc"
    temperature: float = 0.0
    candidate_count: int = 1
    thinking_budget_tokens: int = 4096
    max_output_tokens: int = 16384
    response_mime_type: str = "application/json"
    automatic_function_calling: bool = False
    sdk_retries: bool = False


VERTEX_PROFILE_V1: Final = VertexModelProfileV1()


def derive_seed(task_id: str) -> int:
    """Deterministic non-negative 31-bit seed: first 8 hex chars of sha256(task_id), masked.

    Vertex `generation_config.seed` is a signed INT32, so the mask keeps the value inside
    the range the provider accepts (D-061).
    """
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFF_FFFF


class GenerationSettingsV1(BaseModel):
    model_config = _MODEL_CONFIG

    temperature: float
    candidate_count: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0, lt=2**31)]
    thinking_budget_tokens: Annotated[int, Field(ge=0)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    response_mime_type: str
    response_schema: dict[str, object]
    automatic_function_calling: bool


class TransportResponse(BaseModel):
    model_config = _MODEL_CONFIG

    text: str
    reasoning: str = ""
    token_usage: dict[str, Annotated[int, Field(ge=0)]]
    finish_reason: str


class GatewayImage(BaseModel):
    """The exact image bytes sent alongside the rendered textual prompt."""

    model_config = _MODEL_CONFIG

    mime_type: Annotated[str, Field(pattern=r"^image/[A-Za-z0-9.+-]+$")]
    data: Annotated[bytes, Field(min_length=1)]


class TransportError(Exception):
    def __init__(self, message: str, code: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class GatewayError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class ModelTransportProtocol(Protocol):
    def generate(
        self, model_id: str, prompt: str, settings: GenerationSettingsV1
    ) -> TransportResponse: ...


class GatewayResultV1(BaseModel):
    model_config = _MODEL_CONFIG

    text: str
    reasoning: str = ""
    parsed: dict[str, object] | None
    token_usage: dict[str, Annotated[int, Field(ge=0)]]
    attempts: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0, lt=2**31)]
    finish_reason: str = "UNKNOWN"


def _parse_json(text: str) -> dict[str, object] | None:
    try:
        value: object = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class VertexGateway:
    def __init__(
        self,
        transport: ModelTransportProtocol,
        profile: VertexModelProfileV1,
        emitter: EventEmitter,
        clock: Callable[[], datetime],
        tracer: TracerProtocol | None = None,
    ) -> None:
        self._transport = transport
        self._profile = profile
        self._emitter = emitter
        self._clock = clock
        self._tracer = tracer

    def invoke(
        self, envelope: AgentTaskEnvelopeV1, prompt: str, response_schema: dict[str, object], *,
        images: tuple[GatewayImage, ...] = (),
    ) -> GatewayResultV1:
        evaluation = envelope.payload.get("evaluation")
        seed_key = evaluation.get("seed_key") if isinstance(evaluation, Mapping) else None
        seed = derive_seed(str(seed_key or envelope.task_id))
        settings = self._settings(seed, response_schema)
        span = self._span(envelope, prompt, response_schema, seed, images)
        image_transport = getattr(self._transport, "generate_with_images", None)
        if images and not callable(image_transport):
            if span:
                span.finish({"physical_attempts": 0}, UNSUPPORTED_IMAGE_INPUT)
            raise GatewayError("model transport does not support images", UNSUPPORTED_IMAGE_INPUT)
        budget = min(3, max(1, envelope.budgets.transient_attempt_budget))
        attempt = 0
        while True:
            attempt += 1
            try:
                if images:
                    assert callable(image_transport)
                    response = image_transport(self._profile.model_id, prompt, settings, images)
                else:
                    response = self._transport.generate(self._profile.model_id, prompt, settings)
            except TransportError as error:
                if not error.retryable:
                    if span:
                        span.finish({"physical_attempts": attempt}, error.code)
                    raise GatewayError(str(error), error.code) from error
                if attempt >= budget:
                    self._emit(envelope, "retry.exhausted", attempt, error, Severity.ERROR)
                    if span:
                        span.finish({"physical_attempts": attempt}, TRANSIENT_EXHAUSTED)
                    raise GatewayError(
                        f"transient attempt budget of {budget} exhausted: {error}",
                        TRANSIENT_EXHAUSTED,
                    ) from error
                self._emit(envelope, "retry.scheduled", attempt, error, Severity.WARNING)
                time.sleep(min(2 ** (attempt - 1), 2))
                continue
            except Exception as error:
                if span:
                    span.finish({"physical_attempts": attempt}, type(error).__name__)
                raise
            if span:
                usage = response.token_usage
                span.finish({
                    "physical_attempts": attempt, "input_tokens": usage.get("input", 0),
                    "output_tokens": usage.get("output", 0),
                    "thinking_tokens": usage.get("thinking", 0),
                    "total_tokens": usage.get("total", 0),
                    "finish_reason": response.finish_reason,
                    "model_output": response.text,
                    "provider_summary": response.reasoning})
            return GatewayResultV1(
                text=response.text, reasoning=response.reasoning,
                parsed=_parse_json(response.text),
                token_usage=dict(response.token_usage), attempts=attempt, seed=seed,
                finish_reason=response.finish_reason)

    def _span(self, envelope: AgentTaskEnvelopeV1, prompt: str,
              schema: Mapping[str, object], seed: int,
              images: tuple[GatewayImage, ...] = ()) -> TraceSpanProtocol | None:
        if self._tracer is None:
            return None
        evaluation = envelope.payload.get("evaluation")
        ids = dict(evaluation) if isinstance(evaluation, Mapping) else {}
        return self._tracer.start_gateway_span({
            "analysis_id": envelope.analysis_id, "stage_run_id": envelope.stage_run_id,
            "task_id": envelope.task_id, "attempt_id": envelope.attempt_id,
            "correction_attempt_count": int(envelope.attempt_id.rsplit(":", 1)[-1])
            if envelope.attempt_id.rsplit(":", 1)[-1].isdigit() else 0,
            "envelope_id": envelope.envelope_id,
            "envelope_hash": content_hash(envelope.canonical_payload()),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_characters": len(prompt),
            "prompt": prompt, "response_schema": dict(schema),
            "images": [{"mime_type": image.mime_type,
                        "base64": base64.b64encode(image.data).decode("ascii"),
                        "sha256": hashlib.sha256(image.data).hexdigest(),
                        "byte_length": len(image.data)} for image in images],
            "response_schema_hash": content_hash(dict(schema)),
            "model_profile_version": self._profile.profile_version,
            "model_id": self._profile.model_id, "seed": seed,
            "evaluation_run_id": ids.get("run_id", ""),
            "evaluation_case_id": ids.get("case_id", ""),
            "evaluation_mode": ids.get("mode", "")})

    def _settings(self, seed: int, response_schema: dict[str, object]) -> GenerationSettingsV1:
        profile = self._profile
        return GenerationSettingsV1(
            temperature=profile.temperature, candidate_count=profile.candidate_count,
            seed=seed, thinking_budget_tokens=profile.thinking_budget_tokens,
            max_output_tokens=profile.max_output_tokens,
            response_mime_type=profile.response_mime_type, response_schema=response_schema,
            automatic_function_calling=profile.automatic_function_calling)

    def _emit(
        self, envelope: AgentTaskEnvelopeV1, name: str, attempt: int,
        error: TransportError, severity: Severity,
    ) -> None:
        self._emitter.emit(build_event(
            occurred_at_utc=self._clock(), severity=severity, event_name=name,
            event_id=f"evt:{envelope.envelope_id}:{name}:{attempt}",
            analysis_id=envelope.analysis_id, stage=Stage.SYSTEM,
            stage_run_id=envelope.stage_run_id, task_id=envelope.task_id,
            attempt_id=envelope.attempt_id, attempt_number=attempt,
            component_id=COMPONENT_ID, component_version=COMPONENT_VERSION,
            versions={"model": self._profile.model_id}, error_code=error.code,
            retryable=error.retryable, required_eval_ids=GATEWAY_EVAL_IDS,
            safe_dimensions={"operation": "vertex.generate"}))


def _classify(error: errors.APIError) -> tuple[str, bool]:
    code = int(error.code or 0)
    if code >= 500:
        return MODEL_UNAVAILABLE, True
    # Google's documented PayGo capacity signal is transient, unlike quota denial.
    # Any structured details (including QuotaFailure) keep the conservative denial path.
    body = error.details.get("error", error.details) if isinstance(error.details, Mapping) else {}
    if (code == 429 and str(error.status or "").upper() == "RESOURCE_EXHAUSTED"
            and error.message == "Resource exhausted, please try again later."
            and isinstance(body, Mapping) and not body.get("details")):
        return MODEL_UNAVAILABLE, True
    mapped = _STATUS_CODES.get(str(error.status or "").upper())
    return mapped or _HTTP_CODES.get(code, MODEL_UNAVAILABLE), False


def _token_usage(usage: types.GenerateContentResponseUsageMetadata | None) -> dict[str, int]:
    if usage is None:
        return {"input": 0, "output": 0, "thinking": 0, "total": 0}
    return {
        "input": usage.prompt_token_count or 0,
        "output": usage.candidates_token_count or 0,
        "thinking": usage.thoughts_token_count or 0,
        "total": usage.total_token_count or 0,
    }


class GenAiTransport:
    def __init__(self, profile: VertexModelProfileV1 = VERTEX_PROFILE_V1) -> None:
        self._profile = profile
        self._client: genai.Client | None = None

    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                vertexai=True, location=self._profile.location,
                http_options=types.HttpOptions(
                    api_version=self._profile.api_surface,
                    # attempts=1 means the single original request: SDK-level retries off.
                    retry_options=types.HttpRetryOptions(attempts=1)))
        return self._client

    def generate(
        self, model_id: str, prompt: str, settings: GenerationSettingsV1
    ) -> TransportResponse:
        return self._generate(model_id, prompt, settings)

    def generate_with_images(
        self, model_id: str, prompt: str, settings: GenerationSettingsV1,
        images: tuple[GatewayImage, ...],
    ) -> TransportResponse:
        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt), *(
            types.Part.from_bytes(data=image.data, mime_type=image.mime_type) for image in images)])
        return self._generate(model_id, content, settings)

    def _generate(
        self, model_id: str, contents: str | types.Content, settings: GenerationSettingsV1,
    ) -> TransportResponse:
        try:
            config = types.GenerateContentConfig(
                temperature=settings.temperature,
                candidate_count=settings.candidate_count,
                seed=settings.seed,
                max_output_tokens=settings.max_output_tokens,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=settings.thinking_budget_tokens, include_thoughts=True),
                response_mime_type=settings.response_mime_type,
                response_schema=_provider_schema(settings.response_schema),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=not settings.automatic_function_calling,
                    maximum_remote_calls=None))
            response = self.client().models.generate_content(
                model=model_id, contents=contents, config=config)
        except ValidationError as error:
            raise TransportError(
                str(error), UNSUPPORTED_STRUCTURED_OUTPUT, retryable=False) from error
        except errors.APIError as error:
            code, retryable = _classify(error)
            raise TransportError(str(error), code, retryable=retryable) from error
        except (TimeoutError, ConnectionError, OSError) as error:
            raise TransportError(str(error), MODEL_UNAVAILABLE, retryable=True) from error
        return _to_transport_response(response)


def _provider_schema(value: dict[str, object], *, _reference: bool = False) -> dict[str, object]:
    """Lower decoder complexity, without mutating the application's strict schema.

    Vertex expands reused definitions. Repeated string matchers can exceed its
    serving-state limit. Keep shape, required fields, types, nullability, decision enums,
    numeric ranges, array bounds and compact reference catalogs. Local walls enforce
    string limits and long evidence/artifact catalogs repeated across semantic slots.
    """
    local_only = {"title", "default", "minLength", "maxLength", "pattern"}
    reference = _reference or (value.get("x-causal-reference-role") == "reference"
                              and value.get("x-causal-reference-kind") in {"evidence", "artifact"})
    lowered: dict[str, object] = {}
    for key, item in value.items():
        if key.startswith("x-causal-") or key in local_only or (reference and key == "enum"):
            continue
        if key in {"properties", "$defs", "definitions"} and isinstance(item, dict):
            lowered[key] = {name: _provider_schema(child) for name, child in item.items()}
        elif key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(item, list):
            lowered[key] = [_provider_schema(child, _reference=reference) for child in item]
        elif isinstance(item, dict):
            lowered[key] = _provider_schema(item, _reference=reference)
        else:
            lowered[key] = item
    return lowered


def _split_parts(response: types.GenerateContentResponse) -> tuple[str, str]:
    """Answer text and the provider's exposed thought summary, not private reasoning.

    `include_thoughts` puts both in `parts`, and
    `response.text` concatenates them, so the thought parts must be separated here or they
    reach the JSON parser (SC §10.4, D-097)."""
    candidates = response.candidates or ()
    content = candidates[0].content if candidates else None
    parts = (content.parts if content is not None else None) or ()
    answer = "".join(part.text for part in parts if part.text and not part.thought)
    reasoning = "".join(part.text for part in parts if part.text and part.thought)
    return (answer or (response.text or "")) if not reasoning else answer, reasoning


def _to_transport_response(response: types.GenerateContentResponse) -> TransportResponse:
    candidates = response.candidates or ()
    finish = candidates[0].finish_reason if candidates else None
    reason = finish.value if finish is not None else "FINISH_REASON_UNSPECIFIED"
    if reason in _SAFETY_FINISH_REASONS:
        raise TransportError(
            f"provider stopped generation: {reason}", PROVIDER_SAFETY_REJECTION, retryable=False)
    if reason == "MAX_TOKENS":
        raise TransportError(
            "provider stopped generation at the output-token limit",
            MODEL_OUTPUT_TRUNCATED, retryable=False)
    answer, reasoning = _split_parts(response)
    return TransportResponse(
        text=answer, reasoning=reasoning, token_usage=_token_usage(response.usage_metadata),
        finish_reason=reason)
