"""A2A execution and current desk publication adapter. Whole-data exploration and transformations are separate contracts."""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import proto_helpers as H
from a2a.types import Role, SendMessageRequest, Task, TaskState

from causal_agent.common import config, trace
from causal_agent.common.contracts import Decline, Thought
from causal_agent.explore import project as PJ
from causal_agent.explore import sandbox
from causal_agent.explore.contracts import ArtifactInput, DatasetInput, Exploration, ExploreRequest, TransformRequest
from causal_agent.explore.inputs import digest, safe_file
from causal_agent.memory import artifacts as VA
from causal_agent.memory import versions as VS
from causal_agent.memory.artifacts import Artifact
from causal_agent.memory.versions import Version

CARD_WAIT = 30.0
_app: Any = None


def set_app(app: Any) -> None:
    """Tests install the server's Starlette app here; the task is then sent in process. None restores the process."""
    global _app
    _app = app


def explore_data(
    question: str,
    dataset_version: DatasetInput,
    context: str,
    artifact_refs: list[ArtifactInput] | None = None,
    *,
    out_dir: Path,
    conversation_id: str,
    request_id: str,
    attempt_id: str,
    thread_id: str | None = None,
) -> Task:
    """Execution-only integration surface. The caller owns resource resolution and publication."""
    req = ExploreRequest(
        question=question,
        dataset_version=dataset_version,
        context=context,
        artifact_refs=artifact_refs or [],
        out_dir=out_dir,
        conversation_id=conversation_id,
        request_id=request_id,
        attempt_id=attempt_id,
    )
    with trace.span("explore_data", inputs=req.model_dump(mode="json")) as span:
        try:
            task = ask_agent(req, thread_id)
            span.end(outputs={"task": task.id, "status": str(task.status.state)})
            return task
        finally:
            shutil.rmtree(out_dir / "input", ignore_errors=True)


def explore(
    *,
    dataset: str,
    ask: str,
    context: str,
    csv: Path,
    design: int | None,
    memory_version: int,
    data_version: int,
    artifact_refs: list[ArtifactInput] | None = None,
    thread_id: str | None = None,
    root: Path | None = None,
) -> tuple[Artifact | None, Decline | None, list[Thought]]:
    """Desk adapter using the integration service's bound conversation publication repository."""
    artifact_id = VA.new_id()
    out_dir = VA.folder(dataset, artifact_id, root)
    out_dir.mkdir(parents=True, exist_ok=False)
    publisher = VA.repository()
    context = publisher.context + "\n\nCHAT CONTEXT\n" + context
    request_id, attempt_id, version_id = publisher.begin(dataset, ask, context, digest(csv), artifact_refs or [])
    version = DatasetInput(id=version_id, path=csv, sha256=digest(csv))
    try:
        task = explore_data(
            ask,
            version,
            context,
            artifact_refs,
            out_dir=out_dir,
            conversation_id=publisher.conversation,
            request_id=request_id,
            attempt_id=attempt_id,
            thread_id=thread_id,
        )
        publisher.external_task(attempt_id, task.id)
        found = {a.name: a for a in task.artifacts}
        thoughts = [Thought.model_validate(t) for t in (H.get_data_parts(found["thoughts"].parts)[0].get("thoughts", []) if "thoughts" in found else [])]
        if "trail" in found:
            trace.write(out_dir / "trail.json", json.dumps(H.get_data_parts(found["trail"].parts)[0], indent=2))
        if task.status.state != TaskState.TASK_STATE_COMPLETED or "exploration" not in found:
            why = H.get_message_text(task.status.message) if task.status.HasField("message") else "the agent returned no result"
            publisher.finish(attempt_id, "failed", why)
            return None, Decline(stage="explore", kind="declined", about="artifact", check="explore.failed", reason=why), thoughts
        rec = Exploration.model_validate(H.get_data_parts(found["exploration"].parts)[0])
        if rec.status == "needs_input":
            trace.write(out_dir / "result.json", rec.model_dump_json(indent=2))
            publisher.finish(attempt_id, "needs_input", rec.model_dump_json())
            return (
                None,
                Decline(stage="explore", kind="declined", about="input", check="explore.needs_input", reason=rec.summary + " " + "; ".join(rec.needs)),
                thoughts,
            )
        facts = {str(k): float(v) for k, v in H.get_data_parts(found["facts"].parts)[0].items()}
        trace.write(out_dir / "facts.json", json.dumps(facts, indent=2))
        trace.write(out_dir / "code.py", H.get_text_parts(found["code"].parts)[0])
        trace.write(out_dir / "result.json", rec.model_dump_json(indent=2))
        trace.write(out_dir / "thoughts.json", json.dumps([t.model_dump(mode="json") for t in thoughts], indent=2))
        trace.write(
            out_dir / "request.json",
            json.dumps(
                {
                    "question": ask,
                    "dataset_version": version.model_dump(mode="json"),
                    "context": context,
                    "artifact_refs": [a.model_dump(mode="json") for a in artifact_refs or []],
                    "request_id": request_id,
                    "attempt_id": attempt_id,
                    "conversation_id": publisher.conversation,
                },
                indent=2,
            ),
        )
        if rec.figure:
            trace.write(out_dir / "figure.png", H.get_raw_parts(found["figure"].parts)[0])
        if rec.table:
            if Path(rec.table).name != rec.table or rec.table in {
                "artifact.json",
                "result.json",
                "request.json",
                "code.py",
                "facts.json",
                "trail.json",
                "thoughts.json",
            }:
                raise ValueError("invalid result table filename")
            trace.write(out_dir / rec.table, H.get_raw_parts(found["table"].parts)[0])
        a = Artifact(
            id=artifact_id,
            dataset=dataset,
            owner_id=publisher.owner,
            dataset_version_id=version.id,
            origin="exploration",
            format="figure" if rec.figure else "table" if rec.table else "result",
            design=design,
            memory_version=memory_version,
            data_version=data_version,
            ask=ask,
            summary=rec.summary,
            method=rec.method,
            findings=rec.findings,
            limitations=rec.limitations,
            tags=rec.tags,
            source_sha256=version.sha256,
            parent_refs=[i.id for i in artifact_refs or []],
            conversation_id=publisher.conversation,
            request_id=request_id,
            attempt_id=attempt_id,
            facts=facts,
            files=sorted(p.name for p in out_dir.iterdir() if p.is_file()) + [VA.FILE],
            table=rec.table,
            figure=rec.figure,
            sandbox=H.get_data_parts(found["trail"].parts)[0].get("sandbox", "not_run"),
            made_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        )
        if digest(csv) != version.sha256:
            raise ValueError("Dataset changed during exploration")
        VA.save(a, root)
        return a, None, thoughts
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        publisher.finish(attempt_id, "failed", str(exc))
        trace.write(out_dir / "failure.json", json.dumps({"request_id": request_id, "attempt_id": attempt_id, "error": str(exc)}))
        return None, Decline(stage="explore", kind="declined", about="task", check="explore.failed", reason=str(exc)), []


def artifact_inputs(dataset: str, ids: list[str], root: Path | None = None) -> list[ArtifactInput]:
    """Resolve explicitly selected artifacts from the bound conversation."""
    result = []
    for artifact_id in ids:
        if len(artifact_id) != 32 or any(c not in "0123456789abcdef" for c in artifact_id):
            raise ValueError("invalid artifact identity")
        artifact = VA.load(dataset, artifact_id, root)
        folder = VA.folder(dataset, artifact.id, root)
        for name in ("facts.json", "result.json", artifact.table, "figure.png" if artifact.figure else None):
            if not name or name not in artifact.files:
                continue
            path = safe_file(folder / name, folder, 5 * 1024 * 1024)
            result.append(ArtifactInput(id=f"{artifact.address}/{name}", path=path, sha256=digest(path)))
    return result


def transform(
    *,
    dataset: str,
    ask: str,
    context: str,
    columns: dict[str, str],
    csv: Path,
    outcome: str | None,
    treatment: str | None,
    frozen: bool,
    memory_version: int,
    data_version: int,
    thread_id: str | None = None,
    root: Path | None = None,
) -> tuple[Version | None, Decline | None, list[Thought]]:
    """A new version of the table, from the same agent asked for a table instead of a look: the table in play is projected
    with its row ids (and one of the pair held back before the freeze, as for a look), the agent writes table.csv, and what
    comes back is re-joined with the column held back by row id, put in the file's column order, and saved as version v
    beside its parent. Nothing is changed in place. One span; a refusal is a Decline and no folder."""
    with trace.span("transform", inputs={"ask": ask, "parent": data_version, "frozen": frozen, "outcome": outcome, "treatment": treatment}) as s:
        v, d, thoughts = _transform(dataset, ask, context, columns, csv, outcome, treatment, frozen, memory_version, data_version, thread_id, root)
        s.end(outputs={"version": trace.record(v), "decline": trace.record(d)})
        return v, d, thoughts


def _transform(dataset, ask, context, columns, csv, outcome, treatment, frozen, memory_version, data_version, thread_id, root):
    drop = PJ.choose_drop(ask, outcome, treatment, frozen)
    if drop == "both":
        reason = f"the ask names both {outcome!r} and {treatment!r}; nothing may relate the outcome to the treatment before the design has run"
        return None, Decline(stage="transform", kind="declined", about="pair", check="explore.no_join", reason=reason), []
    v = VS.next_version(dataset, root)
    out_dir = VS.folder(dataset, v, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped_name = outcome if drop == "outcome" else None
    orig = pd.read_csv(csv)
    table = PJ.project(csv, out_dir, drop=dropped_name, row_id=True)
    sent = {k: n for k, n in columns.items() if n != dropped_name}
    keep = [c for c in (outcome, treatment) if c and c != dropped_name and c in orig.columns]
    req = TransformRequest(
        dataset=dataset,
        ask=ask,
        context=context,
        columns=sent,
        csv=table,
        out_dir=out_dir,
        outcome=outcome,
        treatment=treatment,
        dropped=drop,  # type: ignore[arg-type]
        frozen=frozen,
        design=None,
        memory_version=memory_version,
        data_version=data_version,
        product="table",
        rows=int(len(orig)),
        keep=keep,
    )
    try:
        task = ask_agent(req, thread_id)
    finally:
        shutil.rmtree(out_dir / "input", ignore_errors=True)
    found = {a.name: a for a in task.artifacts}
    thoughts = [Thought.model_validate(t) for t in (H.get_data_parts(found["thoughts"].parts)[0].get("thoughts", []) if "thoughts" in found else [])]
    if task.status.state != TaskState.TASK_STATE_COMPLETED or "exploration" not in found or "table" not in found:
        why = H.get_message_text(task.status.message) if task.status.HasField("message") else "the agent returned no table"
        shutil.rmtree(out_dir, ignore_errors=True)
        return None, Decline(stage="transform", kind="declined", about="table", check="transform.failed", reason=why), thoughts
    rec = Exploration.model_validate(H.get_data_parts(found["exploration"].parts)[0])
    new = pd.read_csv(io.BytesIO(H.get_raw_parts(found["table"].parts)[0]))
    if dropped_name:  # the column held back comes back by row id, so the version carries the whole file
        back = orig[[dropped_name]].copy()
        back[PJ.ROW] = range(len(orig))
        new = new.merge(back, on=PJ.ROW, how="left")
    new = new.sort_values(PJ.ROW).drop(columns=[PJ.ROW])
    new = new[[c for c in orig.columns if c in new.columns] + [c for c in new.columns if c not in orig.columns]]
    trace.write(out_dir / VS.TABLE, new.to_csv(index=False))
    trace.write(out_dir / "code.py", H.get_text_parts(found["code"].parts)[0])
    trace.write(out_dir / "thoughts.json", json.dumps([t.model_dump(mode="json") for t in thoughts], indent=2))
    trace.write(out_dir / "request.json", json.dumps(req.model_dump(mode="json", exclude={"context"}), indent=2))
    trail = H.get_data_parts(found["trail"].parts)[0] if "trail" in found else {}
    version = Version(
        v=v,
        dataset=dataset,
        parent=data_version,
        ask=ask,
        summary=rec.summary,
        rows=int(len(new)),
        columns=[str(c) for c in new.columns],
        added=[str(c) for c in new.columns if c not in orig.columns],
        dropped=[str(c) for c in orig.columns if c not in new.columns],
        memory_version=memory_version,
        sandbox=str(trail.get("sandbox") or "subprocess"),
        made_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    VS.save(version, root)
    return version, None, thoughts


# ------------------------------------------------------------------ the wire


def ask_agent(req: ExploreRequest | TransformRequest, thread_id: str | None = None) -> Task:
    """Send one task and wait for its end: in process when a test installed the app, else to a process started for the call."""
    req = req.model_copy(update={"trace_headers": trace.headers() or {}})
    if _app is not None:
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app), base_url="http://explore", timeout=900)
        return asyncio.run(_send(req, http, "http://explore"))
    if url := config.get().sandbox.agent_url:
        return asyncio.run(_send(req, httpx.AsyncClient(timeout=900), url))
    with process(thread_id, req.attempt_id if isinstance(req, ExploreRequest) else uuid4().hex) as url:
        return asyncio.run(_send(req, httpx.AsyncClient(timeout=900), url))


async def _send(req: ExploreRequest | TransformRequest, http: httpx.AsyncClient, url: str) -> Task:
    async with http:
        card = await A2ACardResolver(http, url).get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)
        msg = H.new_data_message(req.model_dump(mode="json"), role=Role.ROLE_USER)
        task: Task | None = None
        artifacts: dict[str, Any] = {}
        async for resp in client.send_message(SendMessageRequest(message=msg)):
            if resp.HasField("task"):
                task = resp.task
            elif resp.HasField("artifact_update"):
                artifacts[resp.artifact_update.artifact.name] = resp.artifact_update.artifact
            elif resp.HasField("status_update") and task is not None:
                task.status.CopyFrom(resp.status_update.status)
        if task is None:
            raise RuntimeError("the agent answered with no task")
        if artifacts and not task.artifacts:
            task.artifacts.extend(artifacts.values())
        if task.status.state not in (TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED):
            task = await client.get_task(_get(task.id))
        return task


def _get(task_id: str):
    from a2a.types import GetTaskRequest

    return GetTaskRequest(id=task_id)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def process(thread_id: str | None = None, attempt_id: str | None = None) -> Iterator[str]:
    """The agent's process for one call: started under the caller's run and thread, its card waited for, stopped after."""
    from causal_agent.profile.datasets import ROOT

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    attempt_id = attempt_id or uuid4().hex
    env = trace.child_env({**os.environ, "EXPLORE_ATTEMPT_ID": attempt_id}, thread_id)
    proc = subprocess.Popen(
        [sys.executable, "-m", "causal_agent.explore.server", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    errors = [""]

    def drain_errors() -> None:
        if proc.stderr is not None:
            with proc.stderr:
                while chunk := proc.stderr.read(4096):
                    errors[0] = (errors[0] + chunk)[-2000:]

    reader = threading.Thread(target=drain_errors, daemon=True)
    reader.start()
    try:
        _wait_for_card(url, proc, errors)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        reader.join(timeout=1)
        sandbox.cancel_attempt(attempt_id)


def _wait_for_card(url: str, proc: subprocess.Popen, errors: list[str], timeout: float = CARD_WAIT) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"the exploration agent's process exited {proc.returncode} before serving its card:\n{errors[0]}")
        try:
            if httpx.get(url + "/.well-known/agent-card.json", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"the exploration agent's process served no card within {timeout:g}s")
