"""How the desk reaches the exploration agent. One call: the artifact's folder is allocated, the table is projected when the
design has not run, the task is sent over A2A, and what came back is written under the folder and saved to the artifacts
shelf. The agent runs in its own process, started for the call and stopped after, under the caller's run and thread
(`trace.child_env`), as the lane does; a test installs the server's app in process with `set_app`."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import proto_helpers as H
from a2a.types import Role, SendMessageRequest, Task, TaskState

from causal_agent.common import trace
from causal_agent.common.contracts import Decline, Thought
from causal_agent.explore import project as PJ
from causal_agent.explore.contracts import Exploration, ExploreRequest
from causal_agent.memory import artifacts as VA
from causal_agent.memory.artifacts import Artifact

CARD_WAIT = 30.0
_app: Any = None


def set_app(app: Any) -> None:
    """Tests install the server's Starlette app here; the task is then sent in process. None restores the process."""
    global _app
    _app = app


def explore(
    *,
    dataset: str,
    ask: str,
    context: str,
    columns: dict[str, str],
    csv: Path,
    outcome: str | None,
    treatment: str | None,
    frozen: bool,
    design: int | None,
    memory_version: int,
    data_version: int,
    thread_id: str | None = None,
    root: Path | None = None,
) -> tuple[Artifact | None, Decline | None, list[Thought]]:
    """One ask to the exploration agent. The columns are the memory's (key -> the file's own name); the one held back is
    removed from what the agent is told. An ask naming both the outcome and the treatment before the freeze is declined
    here, with no process started. One span; a refusal is a Decline and no folder."""
    with trace.span("explore", inputs={"ask": ask, "design": design, "frozen": frozen, "outcome": outcome, "treatment": treatment}) as s:
        a, d, thoughts = _explore(dataset, ask, context, columns, csv, outcome, treatment, frozen, design, memory_version, data_version, thread_id, root)
        s.end(outputs={"artifact": trace.record(a), "decline": trace.record(d)})
        return a, d, thoughts


def _explore(dataset, ask, context, columns, csv, outcome, treatment, frozen, design, memory_version, data_version, thread_id, root):
    drop = PJ.choose_drop(ask, outcome, treatment, frozen)
    if drop == "both":
        reason = f"the ask names both {outcome!r} and {treatment!r}; nothing may relate the outcome to the treatment before the design has run"
        return None, Decline(stage="explore", kind="declined", about="pair", check="explore.no_join", reason=reason), []
    artifact_id = VA.new_id()
    out_dir = VA.folder(dataset, artifact_id, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped_name = {"outcome": outcome, "treatment": treatment}.get(drop or "")
    table = PJ.project(csv, out_dir, drop=dropped_name) if dropped_name else csv
    sent = {k: v for k, v in columns.items() if v != dropped_name}
    req = ExploreRequest(
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
        design=design,
        memory_version=memory_version,
        data_version=data_version,
    )
    try:
        task = ask_agent(req, thread_id)
    finally:
        shutil.rmtree(out_dir / "input", ignore_errors=True)
    found = {a.name: a for a in task.artifacts}
    thoughts = [Thought.model_validate(t) for t in (H.get_data_parts(found["thoughts"].parts)[0].get("thoughts", []) if "thoughts" in found else [])]
    if task.status.state != TaskState.TASK_STATE_COMPLETED or "exploration" not in found:
        why = H.get_message_text(task.status.message) if task.status.HasField("message") else "the agent returned no record"
        shutil.rmtree(out_dir, ignore_errors=True)
        return None, Decline(stage="explore", kind="declined", about="artifact", check="explore.failed", reason=why), thoughts
    rec = Exploration.model_validate(H.get_data_parts(found["exploration"].parts)[0])
    facts = {str(k): float(v) for k, v in H.get_data_parts(found["facts"].parts)[0].items()}
    trace.write(out_dir / "facts.json", json.dumps(facts, indent=2))
    trace.write(out_dir / "code.py", H.get_text_parts(found["code"].parts)[0])
    trace.write(out_dir / "thoughts.json", json.dumps([t.model_dump(mode="json") for t in thoughts], indent=2))
    trace.write(out_dir / "request.json", json.dumps(req.model_dump(mode="json", exclude={"context"}), indent=2))
    if "figure" in found:
        trace.write(out_dir / "figure.png", H.get_raw_parts(found["figure"].parts)[0])
    if rec.table and "table" in found:
        trace.write(out_dir / rec.table, H.get_raw_parts(found["table"].parts)[0])
    trail = H.get_data_parts(found["trail"].parts)[0] if "trail" in found else {}
    a = Artifact(
        id=artifact_id,
        dataset=dataset,
        kind="picture" if rec.figure else "exploration",
        design=design,
        memory_version=memory_version,
        data_version=data_version,
        ask=ask,
        summary=rec.summary,
        facts=facts,
        files=sorted(p.name for p in out_dir.iterdir() if p.is_file()) + [VA.FILE],
        table=rec.table,
        figure=rec.figure,
        sandbox=str(trail.get("sandbox") or "subprocess"),
        made_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    VA.save(a, root)
    return a, None, thoughts


# ------------------------------------------------------------------ the wire


def ask_agent(req: ExploreRequest, thread_id: str | None = None) -> Task:
    """Send one task and wait for its end: in process when a test installed the app, else to a process started for the call."""
    if _app is not None:
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app), base_url="http://explore", timeout=None)
        return asyncio.run(_send(req, http, "http://explore"))
    with process(thread_id) as url:
        return asyncio.run(_send(req, httpx.AsyncClient(timeout=None), url))


async def _send(req: ExploreRequest, http: httpx.AsyncClient, url: str) -> Task:
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
def process(thread_id: str | None = None) -> Iterator[str]:
    """The agent's process for one call: started under the caller's run and thread, its card waited for, stopped after."""
    from causal_agent.profile.datasets import ROOT

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = trace.child_env(dict(os.environ), thread_id)
    proc = subprocess.Popen(
        [sys.executable, "-m", "causal_agent.explore.server", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_card(url, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_for_card(url: str, proc: subprocess.Popen, timeout: float = CARD_WAIT) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            err = (proc.stderr.read() if proc.stderr else "")[-2000:]
            raise RuntimeError(f"the exploration agent's process exited {proc.returncode} before serving its card:\n{err}")
        try:
            if httpx.get(url + "/.well-known/agent-card.json", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"the exploration agent's process served no card within {timeout:g}s")
