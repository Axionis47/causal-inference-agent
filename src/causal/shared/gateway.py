"""The one Vertex model gateway: profile, seed, transport, transient retries (SC §10.4; T-010)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Final, Literal, Protocol

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field

from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.events import EventEmitter, Severity, Stage, build_event

__all__ = [
    "GATEWAY_ERROR_CODES",
    "VERTEX_PROFILE_V1",
    "GatewayError",
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
TRANSIENT_EXHAUSTED: Final = "transient_exhausted"

GATEWAY_ERROR_CODES: Final[frozenset[str]] = frozenset({
    PROVIDER_SAFETY_REJECTION, MODEL_UNAVAILABLE, INVALID_AUTHENTICATION,
    PERMISSION_DENIED, QUOTA_EXHAUSTED, UNSUPPORTED_STRUCTURED_OUTPUT, TRANSIENT_EXHAUSTED,
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
    """The frozen V1 model profile; every call is made with exactly these knobs."""

    model_config = _MODEL_CONFIG

    profile_version: Literal["vertex-model-profile.v1"] = "vertex-model-profile.v1"
    sdk: str = "google-genai==2.19.0"
    api_surface: str = "v1"
    model_id: str = "gemini-2.5-flash"
    location: str = "us-central1"
    authentication: str = "adc"
    temperature: float = 0.0
    candidate_count: int = 1
    thinking_budget_tokens: int = 8192
    max_output_tokens: int = 16384
    response_mime_type: str = "application/json"
    automatic_function_calling: bool = False
    sdk_retries: bool = False


VERTEX_PROFILE_V1: Final = VertexModelProfileV1()


def derive_seed(task_id: str) -> int:
    """Deterministic unsigned 32-bit seed for a task: first 8 hex chars of sha256(task_id)."""
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16)


class GenerationSettingsV1(BaseModel):
    """The exact per-call knobs a transport must apply, with nothing implied."""

    model_config = _MODEL_CONFIG

    temperature: float
    candidate_count: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    thinking_budget_tokens: Annotated[int, Field(ge=0)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    response_mime_type: str
    response_schema: dict[str, object]
    automatic_function_calling: bool


class TransportResponse(BaseModel):
    """One provider response, already reduced to the fields the gateway uses."""

    model_config = _MODEL_CONFIG

    text: str
    token_usage: dict[str, Annotated[int, Field(ge=0)]]
    finish_reason: str


class TransportError(Exception):
    """A provider failure already translated to a stable code and a retry verdict."""

    def __init__(self, message: str, code: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class GatewayError(Exception):
    """A terminal model-call failure; `code` is one of GATEWAY_ERROR_CODES."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class ModelTransportProtocol(Protocol):
    """The only surface the gateway calls; provider types never leak past it."""

    def generate(
        self, model_id: str, prompt: str, settings: GenerationSettingsV1
    ) -> TransportResponse: ...


class GatewayResultV1(BaseModel):
    """What one completed model call yields; schema conformance is the harness's job."""

    model_config = _MODEL_CONFIG

    text: str
    parsed: dict[str, object] | None
    token_usage: dict[str, Annotated[int, Field(ge=0)]]
    attempts: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0)]


def _parse_json(text: str) -> dict[str, object] | None:
    """json.loads of the response text, or None when it is not a JSON object."""
    try:
        value: object = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class VertexGateway:
    """Applies the frozen profile, derives the seed, and owns the transient retry rule."""

    def __init__(
        self,
        transport: ModelTransportProtocol,
        profile: VertexModelProfileV1,
        emitter: EventEmitter,
        clock: Callable[[], datetime],
    ) -> None:
        self._transport = transport
        self._profile = profile
        self._emitter = emitter
        self._clock = clock

    def invoke(
        self, envelope: AgentTaskEnvelopeV1, prompt: str, response_schema: dict[str, object]
    ) -> GatewayResultV1:
        """One model call with up to the envelope's transient attempt budget of physical tries."""
        seed = derive_seed(envelope.task_id)
        settings = self._settings(seed, response_schema)
        budget = max(1, envelope.budgets.transient_attempt_budget)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._transport.generate(self._profile.model_id, prompt, settings)
            except TransportError as error:
                if not error.retryable:
                    raise GatewayError(str(error), error.code) from error
                if attempt >= budget:
                    self._emit(envelope, "retry.exhausted", attempt, error, Severity.ERROR)
                    raise GatewayError(
                        f"transient attempt budget of {budget} exhausted: {error}",
                        TRANSIENT_EXHAUSTED,
                    ) from error
                self._emit(envelope, "retry.scheduled", attempt, error, Severity.WARNING)
                continue
            return GatewayResultV1(
                text=response.text, parsed=_parse_json(response.text),
                token_usage=dict(response.token_usage), attempts=attempt, seed=seed)

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
    """Provider exception to (stable code, retryable); only 5xx responses are retryable."""
    code = int(error.code or 0)
    if code >= 500:
        return MODEL_UNAVAILABLE, True
    mapped = _STATUS_CODES.get(str(error.status or "").upper())
    return mapped or _HTTP_CODES.get(code, MODEL_UNAVAILABLE), False


def _token_usage(usage: types.GenerateContentResponseUsageMetadata | None) -> dict[str, int]:
    """The D-015 token keys; counters the provider omits read as zero."""
    if usage is None:
        return {"input": 0, "output": 0, "thinking": 0, "total": 0}
    return {
        "input": usage.prompt_token_count or 0,
        "output": usage.candidates_token_count or 0,
        "thinking": usage.thoughts_token_count or 0,
        "total": usage.total_token_count or 0,
    }


class GenAiTransport:
    """Live google-genai 2.19.0 adapter; the project comes from ADC and is never logged."""

    def __init__(self, profile: VertexModelProfileV1 = VERTEX_PROFILE_V1) -> None:
        self._profile = profile
        self._client: genai.Client | None = None

    def client(self) -> genai.Client:
        """The lazily built Vertex-mode client; project and credentials resolve from ADC."""
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
        """One structured-output generate_content call with every settings knob applied."""
        config = types.GenerateContentConfig(
            temperature=settings.temperature,
            candidate_count=settings.candidate_count,
            seed=settings.seed,
            max_output_tokens=settings.max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=settings.thinking_budget_tokens, include_thoughts=False),
            response_mime_type=settings.response_mime_type,
            response_schema=dict(settings.response_schema),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=not settings.automatic_function_calling, maximum_remote_calls=None))
        try:
            response = self.client().models.generate_content(
                model=model_id, contents=prompt, config=config)
        except errors.APIError as error:
            code, retryable = _classify(error)
            raise TransportError(str(error), code, retryable=retryable) from error
        except (TimeoutError, ConnectionError, OSError) as error:
            raise TransportError(str(error), MODEL_UNAVAILABLE, retryable=True) from error
        return _to_transport_response(response)


def _to_transport_response(response: types.GenerateContentResponse) -> TransportResponse:
    """Reduce an SDK response, raising on a provider safety stop before any text is read."""
    candidates = response.candidates or ()
    finish = candidates[0].finish_reason if candidates else None
    reason = finish.value if finish is not None else "FINISH_REASON_UNSPECIFIED"
    if reason in _SAFETY_FINISH_REASONS:
        raise TransportError(
            f"provider stopped generation: {reason}", PROVIDER_SAFETY_REJECTION, retryable=False)
    return TransportResponse(
        text=response.text or "", token_usage=_token_usage(response.usage_metadata),
        finish_reason=reason)
