"""The trace mirrors the graph. One place that opens LangSmith spans, so a judgement, an episode, a tool call, a gate, a file
written and a process boundary each show in the trace as what they are, under the node that did them.

Every helper is a no-op when tracing is off, and none of them touches the model object: a span is opened around a call, never
set as a config on a runnable, so a fake with a bare `invoke` is never asked for `with_config`. The chokepoints that call these
are the ones every new piece of the graph passes through anyway (`llm.structured`, `lane.episode`, `lane.tools`, the compile
helpers, `desk.pipeline`); `common/tests/test_trace_policy.py` holds that."""

from __future__ import annotations

import json
import mimetypes
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from langsmith import get_current_run_tree, tracing_context
from langsmith import trace as _trace
from langsmith.run_trees import RunTree, get_cached_client
from langsmith.utils import tracing_is_enabled

PARENT_ENV = "CAUSAL_TRACE_PARENT"  # the parent run's headers, JSON, for a child process
THREAD_ENV = "CAUSAL_THREAD_ID"  # the conversation's thread id, so a child process groups under the same thread
MAX_ATTACHMENT = 15 * 1024 * 1024
MAX_TEXT = 20_000  # a prompt or a file shown inline is cut here; the full text is in the attachment or on disk


def on() -> bool:
    """Whether spans are sent: LANGSMITH_TRACING=true, or already inside a trace."""
    return tracing_is_enabled() is True


class Span:
    """What a span's caller holds: `end(outputs=...)` records the result; an exception leaving the block records the error."""

    def __init__(self, run: RunTree | None):
        self.run = run

    def end(self, *, outputs: Mapping[str, Any] | None = None, error: str | None = None) -> None:
        if self.run is not None:
            self.run.end(outputs=dict(outputs or {}), error=error)

    def note(self, **metadata: Any) -> None:
        if self.run is not None:
            self.run.add_metadata(metadata)


@contextmanager
def span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Span]:
    """A span under the current run (a graph node, an outer span, or a parent handed across a process); a root when there is none."""
    if not on():
        yield Span(None)
        return
    with _trace(name, run_type=run_type, inputs=_safe(inputs), metadata=dict(metadata or {}), tags=tags) as run:  # type: ignore[arg-type]
        yield Span(run)


def gate(errors: list[str], *, name: str = "gate", **inputs: Any) -> list[str]:
    """A gate's verdict as its own span: the errors it found, or none. Returns the errors unchanged so a caller can wrap in place."""
    with span(name, run_type="tool", inputs=inputs) as s:
        s.end(outputs={"passed": not errors, "errors": list(errors)})
    return errors


def artifact(path: str | Path, *, name: str | None = None, note: str = "") -> Path:
    """A file the node left behind, as a tool span with the file attached; small text is also shown inline."""
    p = Path(path)
    label = name or p.name
    if not on() or not p.exists():
        return p
    size = p.stat().st_size
    mime = mimetypes.guess_type(p.name)[0] or (
        "text/plain" if p.suffix in {".md", ".jsonl", ".yaml", ".yml", ".py", ".txt", ".csv"} else "application/octet-stream"
    )
    key = label.replace(".", "_")  # LangSmith refuses a period in an attachment's name; the run's name keeps the file's
    attachments: dict[str, Any] = {key: (mime, p.read_bytes())} if 0 < size <= MAX_ATTACHMENT else {}
    with _trace(f"artifact:{label}", run_type="tool", inputs={"path": str(p), "note": note}, attachments=attachments) as run:
        out: dict[str, Any] = {"path": str(p), "bytes": size, "mime": mime}
        if mime.startswith("text/") or mime in {"application/json"}:
            out["text"] = _cut(p.read_text(errors="replace"))
        run.end(outputs=out)
    return p


def write(path: str | Path, content: str | bytes, *, note: str = "") -> Path:
    """Write a file and attach it to the current span. The one way a node writes under a design, run or viz folder."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content)
    return artifact(p, note=note)


def headers() -> dict[str, str] | None:
    """The current run as headers a child process can continue under; None outside a trace."""
    if not on():
        return None
    rt = get_current_run_tree()
    if rt is None:
        try:
            rt = RunTree.from_runnable_config(None)
        except Exception:
            rt = None
    return dict(rt.to_headers()) if rt is not None else None


def child_env(env: dict[str, str], thread_id: str | None = None) -> dict[str, str]:
    """The environment for a child process: the parent run's headers and the thread id, when there are any."""
    h = headers()
    if h:
        env[PARENT_ENV] = json.dumps(h)
    if thread_id:
        env[THREAD_ENV] = thread_id
    return env


def parent(headers: dict[str, str] | None = None) -> Any:
    """In a child process: the context that nests everything under the parent run the environment names; nothing otherwise."""
    if headers:
        return tracing_context(parent=headers)
    raw = os.environ.get(PARENT_ENV)
    if not raw:
        return nullcontext()
    try:
        h = json.loads(raw)
    except ValueError:
        return nullcontext()
    return tracing_context(parent=h)


def thread_id() -> str | None:
    return os.environ.get(THREAD_ENV) or None


def flush(timeout: float = 10) -> None:
    """Finish queued child-process spans before its owner tears down the process."""
    if on():
        get_cached_client().flush(timeout=timeout)


# ------------------------------------------------------------------ shaping what a span shows


def _cut(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + f"\n… [{len(text) - MAX_TEXT} more characters]"


def _safe(inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inputs as the trace shows them: a pydantic record dumped, long text cut, anything else as is."""
    out: dict[str, Any] = {}
    for k, v in (inputs or {}).items():
        if isinstance(v, type):  # a schema class: its name, never an unbound model_dump
            out[k] = v.__name__
        elif hasattr(v, "model_dump"):
            out[k] = v.model_dump(mode="json")
        elif isinstance(v, str):
            out[k] = _cut(v)
        else:
            out[k] = v
    return out


def record(v: Any) -> Any:
    """A typed record as the trace shows it."""
    return v.model_dump(mode="json") if hasattr(v, "model_dump") and not isinstance(v, type) else v
