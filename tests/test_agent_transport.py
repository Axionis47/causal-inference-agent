import json

import pytest
from google.genai import types
from test_model_observability import client

from backend import model_access


@pytest.fixture(autouse=True)
def isolated():
    yield


@pytest.fixture(autouse=True)
def prevent_unmocked_agent_calls():
    # This transport test stubs the SDK client instead of the transport function.
    yield


def test_native_turn_preserves_function_calls_and_opaque_signatures(monkeypatch):
    content = types.Content(role="model", parts=[types.Part(
        function_call=types.FunctionCall(name="read_context", args={"keys": ["source.a"]}, id="call-a"),
        thought_signature=b"\xff\x00opaque-provider-signature")])
    response = types.GenerateContentResponse(candidates=[types.Candidate(content=content, finish_reason="STOP")])
    requests = client(monkeypatch, response)
    prior = content.model_dump(mode="json", exclude_none=True)
    request = model_access.PreparedToolRequest("gemini-2.5-flash", "Test", json.dumps([
        prior, {"role": "user", "parts": [{"function_response": {
            "id": "call-a", "name": "read_context", "response": {"ok": True}}}]}]),
        json.dumps([{"name": "read_context", "description": "Read a frozen source.",
                     "parameters_json_schema": {"type": "object", "properties": {"keys": {"type": "array", "items": {"type": "string"}}}, "required": ["keys"]}}]))
    result = model_access.invoke_tools(request)
    assert types.Content.model_validate(result.content) == content
    assert requests[0]["contents"][0] == content
    assert requests[0]["config"].automatic_function_calling.disable
    assert requests[0]["config"].response_schema is None
    assert json.loads(result.raw)["candidates"][0]["content"] == prior
    assert result.metrics["provider_schema_format"] == "native_function_calling"

