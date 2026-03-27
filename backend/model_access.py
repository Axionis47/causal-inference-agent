"""One typed Vertex boundary; retries and call budgets belong to the caller."""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from google import genai
from google.genai import types
from langsmith import traceable
from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.config import settings

_THINKING_MODELS = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"}


def supports_thinking(model: str) -> bool:
    return model.rsplit("/", 1)[-1] in _THINKING_MODELS


@dataclass(frozen=True)
class PreparedModelRequest[T: BaseModel]:
    """Detached immutable wire inputs; parsed views are fresh copies for inspection."""
    response_model: type[T]
    system: str
    payload_json: str
    model: str
    schema_json: str
    schema_format: str
    output_limit: int
    thinking_budget: int | None
    images: tuple[bytes, ...]
    reference_decoder: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    preflight: Callable[[], None] | None = None

    @property
    def payload(self) -> dict[str, object]:
        return dict(json.loads(self.payload_json))

    @property
    def schema(self) -> dict[str, Any]:
        return dict(json.loads(self.schema_json))


def prepare[T: BaseModel](
    schema: type[T], system: str, payload: dict[str, object], model: str, *,
    images: list[bytes] | None = None, max_output_tokens: int = 32768,
    thinking_budget: int | None = None, encoded_schema: dict[str, Any] | None = None,
    schema_format: str = "response_schema",
    reference_decoder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    preflight: Callable[[], None] | None = None,
) -> PreparedModelRequest[T]:
    return PreparedModelRequest(schema, system, json.dumps(payload, ensure_ascii=False), model,
        json.dumps(encoded_schema if encoded_schema is not None else provider_schema(schema)),
        schema_format, max_output_tokens, thinking_budget, tuple(images or []), reference_decoder, preflight)


@dataclass(frozen=True)
class ProviderObservation:
    raw: str | None
    metrics: dict[str, Any]


class ProviderResponseError(ValueError):
    def __init__(self, message: str, observation: ProviderObservation):
        self._provider_observation = observation
        super().__init__(message)


class InvalidModelOutput(ValueError):
    def __init__(self, raw: str, error: ValidationError, observation: ProviderObservation | None = None):
        self.raw = raw
        self._provider_observation = observation
        errors = error.errors(include_url=False, include_input=False)
        super().__init__(json.dumps(errors, ensure_ascii=False, default=str))


def response_artifacts(value: object) -> tuple[str | None, dict[str, Any]]:
    """Read private provider observations without changing typed model contracts.

    Controlled callers may return ordinary models or legacy validation errors;
    keep their existing serialization fallback and never invent provider usage.
    """
    observation = getattr(value, "_provider_observation", None)
    if isinstance(observation, ProviderObservation):
        return observation.raw, deepcopy(observation.metrics)
    if isinstance(value, InvalidModelOutput):
        return value.raw, {}
    return (value.model_dump_json(), {}) if isinstance(value, BaseModel) else (None, {})


def _observe(
    response: types.GenerateContentResponse, model: str, limit: int,
    thinking_config: types.ThinkingConfig | None = None,
    schema_format: str = "response_schema",
    raw_content: str | None = None,
) -> ProviderObservation:
    raw = raw_content if raw_content is not None else response.text
    usage = response.usage_metadata.model_dump(mode="json", exclude_none=True) if response.usage_metadata else None
    candidates = [
        candidate.model_dump(mode="json", include={"index", "finish_reason", "finish_message"})
        for candidate in response.candidates or []
    ]
    first = candidates[0] if candidates else {}
    return ProviderObservation(raw, {
        "provider_response_id": response.response_id,
        "provider_model_version": response.model_version,
        "requested_model": model,
        "output_limit": limit,
        "provider_schema_format": schema_format,
        "thinking_config": thinking_config.model_dump(mode="json", exclude_none=True) if thinking_config else None,
        "thinking_limit_kind": "soft" if thinking_config else "provider_default",
        "finish_reason": first.get("finish_reason"),
        "finish_message": first.get("finish_message"),
        "candidate_finishes": candidates,
        "usage": usage,
        "input_tokens": (usage or {}).get("prompt_token_count"),
        "output_tokens": (usage or {}).get("candidates_token_count"),
        "reasoning_tokens": (usage or {}).get("thoughts_token_count"),
        "total_tokens": (usage or {}).get("total_token_count"),
        "cached_input_tokens": (usage or {}).get("cached_content_token_count"),
        "response_bytes": len(raw.encode()) if raw is not None else None,
        "response_text_hash": sha256(raw.encode()).hexdigest() if raw is not None else None,
    })


def provider_schema(schema: type[BaseModel]) -> dict[str, Any]:
    return encode_schema(schema.model_json_schema())


def encode_schema(
    native: dict[str, Any], *, choices: dict[str, list[str]] | None = None,
    schema_format: str = "response_schema", explicit_fields: bool = False,
    definition_transform: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Encode provider-supported JSON constructs without domain policy."""
    definitions = native.get("$defs", {})
    shared_definitions = schema_format == "response_json_schema"
    transform = definition_transform or (lambda name, value: value)

    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            if shared_definitions:
                # The JSON Schema transport accepts standalone local references.
                return {"$ref": value["$ref"]}
            name = value["$ref"].split("/")[-1]
            return transform(name, convert(definitions[name]))
        omitted = {
            "title",
            "additionalProperties",
            "default",
            "format",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "maxItems",
        }
        if not shared_definitions:
            omitted.add("$defs")
        converted = {
            key: convert(item) for key, item in value.items() if key not in omitted and key != "const"
        }
        if "const" in value:
            converted["enum"] = [value["const"]]
        if "$defs" in converted:
            converted["$defs"] = {name: transform(name, item) for name, item in converted["$defs"].items()}
        return converted

    result: dict[str, Any] = convert(native)
    def constrain(value: Any) -> None:
        if isinstance(value, dict):
            if explicit_fields and value.get("type") == "object" and "properties" in value:
                # Complex optional-property combinations can exceed Vertex's
                # native decoding limit. Presence is explicit; nullable values
                # and all field constraints retain their original meaning.
                value["required"] = list(value["properties"])
            for name, field in value.get("properties", {}).items():
                if choices is not None and name in choices and not choices[name] and field.get("type") == "array":
                    field["maxItems"] = 0
                if choices and name in choices and choices[name]:
                    selected = {"type": "string", "enum": choices[name]}
                    if field.get("type") == "array":
                        field["items"] = selected
                    elif field.get("type") == "string":
                        field.update(selected)
            for item in value.values():
                constrain(item)
        elif isinstance(value, list):
            for item in value:
                constrain(item)
    constrain(result)
    return result


@traceable(name="structured-model", run_type="llm")
def structured[T: BaseModel](request: PreparedModelRequest[T]) -> T:
    schema, system, model = request.response_model, request.system, request.model
    max_output_tokens, thinking_budget = request.output_limit, request.thinking_budget
    reference_decoder = request.reference_decoder
    if request.preflight is not None:
        request.preflight()
    if thinking_budget is not None and (
        model.rsplit("/", 1)[-1] not in _THINKING_MODELS or not 0 <= thinking_budget < max_output_tokens
    ):
        raise ValueError("Explicit thinking budgets require a supported Gemini 2.5 text model and room within the total output limit.")
    thinking = types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=False) if thinking_budget is not None else None
    schema_format = request.schema_format
    encoded_schema = request.schema
    config = settings()
    with genai.Client(
        vertexai=True,
        project=config.vertex_project or None,
        location=config.vertex_location,
        http_options=types.HttpOptions(
            api_version="v1",
            timeout=config.model_timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=request.payload_json),
                        *[types.Part.from_bytes(data=data, mime_type="image/png") for data in request.images],
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_schema=encoded_schema if schema_format == "response_schema" else None,
                response_json_schema=encoded_schema if schema_format == "response_json_schema" else None,
                thinking_config=thinking,
            ),
        )
    observation = _observe(response, model, max_output_tokens, thinking, schema_format)
    if not observation.raw:
        raise ProviderResponseError("The model returned no structured response.", observation)
    if len(observation.raw.encode()) > 2 * 1024 * 1024:
        raise ProviderResponseError("The model result exceeded the 2 MiB report limit.", observation)
    try:
        value = schema.model_validate(reference_decoder(TypeAdapter(dict[str, Any]).validate_json(observation.raw))) if reference_decoder else schema.model_validate_json(observation.raw)
    except ValidationError as error:
        raise InvalidModelOutput(observation.raw, error, observation) from error
    except ValueError as error:
        raise ProviderResponseError("Invalid reference selection: " + str(error), observation) from error
    # Undeclared private attributes are absent from Pydantic schemas and dumps.
    # Each response owns its observation; no process-global request state exists.
    object.__setattr__(value, "_provider_observation", observation)
    return value


@dataclass(frozen=True)
class PreparedToolRequest:
    """Exact native function declarations and opaque conversation parts for one turn."""
    model: str
    system: str
    contents_json: str
    functions_json: str
    output_limit: int = 8192
    thinking_budget: int = 2048
    function_calling_mode: str = "VALIDATED"

    def record(self) -> dict[str, Any]:
        return {"model": self.model, "system": self.system,
                "contents": json.loads(self.contents_json), "functions": json.loads(self.functions_json),
                "output_limit": self.output_limit, "thinking_budget": self.thinking_budget,
                "schema_format": "native_function_calling", "function_calling_mode": self.function_calling_mode}


@dataclass(frozen=True)
class ToolTurn:
    content: dict[str, Any]
    raw: str
    metrics: dict[str, Any]


@traceable(name="agent-model-turn", run_type="llm")
def invoke_tools(request: PreparedToolRequest) -> ToolTurn:
    """One physical provider invocation. The caller owns all tools, retries and budgets."""
    config = settings()
    thinking = types.ThinkingConfig(thinking_budget=request.thinking_budget, include_thoughts=False) if supports_thinking(request.model) else None
    functions = [types.FunctionDeclaration.model_validate(item) for item in json.loads(request.functions_json)]
    if not functions or not 0 <= request.thinking_budget < request.output_limit or request.function_calling_mode not in {"ANY", "VALIDATED"}:
        raise ValueError("A native tool turn requires tools and a valid output allowance.")
    with genai.Client(vertexai=True, project=config.vertex_project or None, location=config.vertex_location,
                     http_options=types.HttpOptions(api_version="v1", timeout=config.model_timeout_seconds * 1000,
                                                    retry_options=types.HttpRetryOptions(attempts=1))) as client:
        response = client.models.generate_content(
            model=request.model,
            contents=[types.Content.model_validate(item) for item in json.loads(request.contents_json)],
            config=types.GenerateContentConfig(system_instruction=request.system, temperature=0,
                max_output_tokens=request.output_limit, thinking_config=thinking,
                tools=[types.Tool(function_declarations=functions)],
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode(request.function_calling_mode), allowed_function_names=[str(function.name) for function in functions])),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
        )
    raw = response.model_dump_json(exclude_none=True)
    observation = _observe(response, request.model, request.output_limit, thinking, "native_function_calling", raw)
    candidates = response.candidates or []
    if not candidates or candidates[0].content is None:
        raise ProviderResponseError("The agent returned no conversation content.", observation)
    if len(raw.encode()) > 2 * 1024 * 1024:
        raise ProviderResponseError("The agent response exceeded the 2 MiB limit.", observation)
    # Keep the complete content, including provider-owned opaque thought signatures.
    # Never rebuild it from parsed function names and arguments.
    content = candidates[0].content.model_dump(mode="json", exclude_none=True)
    return ToolTurn(content=content, raw=raw, metrics=observation.metrics)
