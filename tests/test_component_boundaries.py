"""Enforced ownership and ordered public/provider compatibility for the refactor."""

import importlib
import json
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest

from backend import model_access
from backend.analysis import provider
from backend.main import app
from backend.responses import Reply
from scripts.check_dependencies import ROOT, check, module_name

FIXTURES = Path(__file__).parent / "fixtures" / "separation"


def test_repository_component_boundaries():
    sources = {module_name(p, ROOT): p.read_text() for p in (ROOT / "backend").rglob("*.py")}
    assert check(sources) == []


@pytest.mark.parametrize("code", [
    "from backend.analysis import artifacts",
    "def read():\n from backend.analysis.artifacts import read",
    "import importlib as loader\nloader.import_module('backend.analysis.artifacts')",
    "from importlib import import_module as load\nload('backend.analysis.artifacts')",
    "__import__('backend.analysis.artifacts')",
    "from importlib import import_module as load\nload('.analysis.artifacts', package='backend')",
    "import importlib\nimportlib.import_module('.analysis', package=variable)",
    "from backend.models import AnalysisRun",
    "from backend import models",
    "import backend.models as records",
    "from backend.models import *",
    "import importlib\nimportlib.import_module(variable)",
    "from backend.shared import artifacts",
])
def test_disallowed_dependencies_include_local_dynamic_and_reexport(code):
    assert check({"backend.chat.example": code,
                  "backend.shared": "from backend.analysis import artifacts"})


def test_shared_transport_and_runner_cannot_depend_on_domains():
    assert check({"backend.model_access": "from backend.context.contracts import ContextContent"})
    assert check({"backend.execution.runner": "from backend.chat import exploration"})
    assert check({"backend.analysis.service": "from backend.models import Conversation"})


def test_retired_execution_is_unavailable_to_production():
    assert importlib.util.find_spec("backend.analysis.graph") is None
    assert importlib.util.find_spec("backend.analysis.work") is None
    assert check({"backend.analysis.example": "from historical_analysis import runtime"})
    assert check({"backend.analysis.example": "from tests.historical_analysis import runtime"})


@pytest.mark.parametrize("entry", json.loads((FIXTURES / "schemas.json").read_text()),
                         ids=lambda e: e["module"] + "." + e["name"] + str(e["choices"]))
def test_ordered_provider_schema_parity(entry):
    module, name = entry["module"], entry["name"]
    if module == "backend.chat.contracts" and name in {"Contract", "ReplyNumber"}:
        module = "backend.validation"
    if module == "backend.chat.contracts" and name == "Reply":
        module = "backend.responses"
    schema = getattr(importlib.import_module(module), name)
    if module.startswith(("backend.analysis", "historical_analysis")):
        actual = provider.provider_schema(schema, entry["choices"])
        assert provider.provider_schema_format(entry["choices"]) == entry["format"]
    else:
        actual = model_access.provider_schema(schema)
    assert sha256(json.dumps(actual, ensure_ascii=False).encode()).hexdigest() == entry["sha256"]


def test_public_api_and_sandbox_runtime_are_unchanged():
    expected = json.loads((FIXTURES / "public-and-runtime.json").read_text())
    assert sha256(json.dumps(app.openapi(), sort_keys=True).encode()).hexdigest() == expected["openapi_sha256"]
    for path, digest in expected["runtime"].items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_prepared_request_is_detached_and_keeps_wire_order():
    payload = {"z": "μ", "a": [1, 2]}
    schema = {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "number"}}}
    request = model_access.prepare(Reply, "unchanged", payload, "test", encoded_schema=schema)
    payload["a"].append(3)
    schema["properties"].clear()
    request.payload["z"] = "other"
    request.schema["properties"].clear()
    assert request.payload_json == '{"z": "μ", "a": [1, 2]}'
    assert list(request.schema["properties"]) == ["z", "a"]
    with pytest.raises(FrozenInstanceError):
        request.system = "changed"
