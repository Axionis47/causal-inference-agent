"""Empty diagnostic output stays visible without an invalid zero-byte upload."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from causal_agent.common import trace


@pytest.mark.parametrize("content", ["", "diagnostic result"])
def test_artifact_records_empty_output_without_attaching_an_empty_payload(tmp_path, monkeypatch, content):
    recorded = {}

    @contextmanager
    def capture(name, **kwargs):
        recorded.update(kwargs)
        yield SimpleNamespace(end=lambda **outputs: recorded.update(outputs))

    monkeypatch.setattr(trace, "on", lambda: True)
    monkeypatch.setattr(trace, "_trace", capture)
    path = trace.write(tmp_path / "stdout.txt", content)
    assert path.read_text() == content
    assert recorded["outputs"]["bytes"] == len(content.encode())
    assert recorded["outputs"]["text"] == content
    assert bool(recorded["attachments"]) == bool(content)
    if content:
        assert recorded["attachments"]["stdout_txt"][1] == content.encode()
