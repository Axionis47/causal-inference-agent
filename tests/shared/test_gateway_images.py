"""Image review sends actual image bytes and traces the same bytes without live APIs."""

from __future__ import annotations

import base64
import hashlib
import io
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types
from pydantic import ValidationError

from causal.shared.events import EventEmitter
from causal.shared.gateway import (
    UNSUPPORTED_IMAGE_INPUT,
    VERTEX_PROFILE_V1,
    GatewayError,
    GatewayImage,
    GenAiTransport,
    VertexGateway,
)
from tests.shared.test_gateway import NOW, SCHEMA, FakeTransport, make_envelope, ok
from tests.shared.test_tracing_content import RecordingClient, tracer_for

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j7uoAAAAASUVORK5CYII=")


def test_provider_and_trace_receive_identical_image_bytes_and_prompt() -> None:
    calls: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> types.GenerateContentResponse:
        calls.append(kwargs)
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text='{"answer":"legible"}')]),
            finish_reason="STOP")])

    transport = GenAiTransport()
    transport._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))  # type: ignore[assignment]
    client = RecordingClient()
    gateway = VertexGateway(transport, VERTEX_PROFILE_V1, EventEmitter(io.StringIO()),
                            lambda: NOW, tracer_for(client))
    image = GatewayImage(mime_type="image/png", data=PNG)
    result = gateway.invoke(make_envelope(), "Inspect the rendered figure", SCHEMA, images=(image,))
    assert result.parsed == {"answer": "legible"}
    sent = calls[0]["contents"]
    assert sent.role == "user"
    assert sent.parts[0].text == "Inspect the rendered figure"
    assert sent.parts[1].inline_data.data == PNG
    assert sent.parts[1].inline_data.mime_type == image.mime_type
    recorded = client.created[0]["inputs"]
    assert recorded["prompt"] == sent.parts[0].text
    assert recorded["images"][0]["sha256"] == hashlib.sha256(PNG).hexdigest()
    assert recorded["images"][0]["byte_length"] == len(PNG)
    assert base64.b64decode(recorded["images"][0]["base64"]) == PNG
    message_image = recorded["messages"][0]["content"][1]
    assert message_image["type"] == "image"
    assert base64.b64decode(message_image["base64"]) == sent.parts[1].inline_data.data


def test_text_only_transports_fail_explicitly_before_an_image_call() -> None:
    transport, client = FakeTransport([ok()]), RecordingClient()
    gateway = VertexGateway(transport, VERTEX_PROFILE_V1, EventEmitter(io.StringIO()),
                            lambda: NOW, tracer_for(client))
    with pytest.raises(GatewayError) as failure:
        gateway.invoke(make_envelope(), "Review", SCHEMA,
                       images=(GatewayImage(mime_type="image/png", data=PNG),))
    assert failure.value.code == UNSUPPORTED_IMAGE_INPUT
    assert transport.calls == []
    assert client.updated[0]["error"] == UNSUPPORTED_IMAGE_INPUT
    assert client.updated[0]["outputs"]["physical_attempts"] == 0


@pytest.mark.parametrize("mime,data", [("text/plain", PNG), ("image/png", b"")])
def test_images_reject_empty_bytes_and_non_image_media(mime: str, data: bytes) -> None:
    with pytest.raises(ValidationError):
        GatewayImage(mime_type=mime, data=data)
