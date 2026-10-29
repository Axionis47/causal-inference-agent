"""The exploration agent as an A2A server: a card, one skill, and an executor that runs one exploration per task. The task's
message carries an `ExploreRequest` as its data part; the task's artifacts carry the record, the facts, the summary, the
picture, the table, the script, the thoughts and the trail. A refused exploration is a failed task whose status says why.

    python -m causal_agent.explore.server --port 8765

The process opens under the parent run the environment names (`trace.parent()`), so the episode nests under the desk's
`explore` span in the trace. Compose keeps the service running and passes per-request trace headers. Local development starts one per call (`client.py`)."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from pathlib import Path

from a2a.helpers import proto_helpers as H
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Part, TaskState
from starlette.applications import Starlette

from causal_agent.common import trace
from causal_agent.explore import agent
from causal_agent.explore.contracts import ExploreRequest, Trail, TransformRequest

NAME = "causal-desk-explore"
SKILL = AgentSkill(
    id="explore",
    name="explore",
    description="Explores the whole supplied CSV and optional result, image or text artifacts. Returns computed facts, tables or plots with methods, findings and limitations.",
    tags=["data", "table", "picture"],
    input_modes=["application/json"],
    output_modes=["application/json", "text/plain", "image/png", "text/csv"],
)


def card(url: str) -> AgentCard:
    return AgentCard(
        name=NAME,
        description="A read-only look at a CSV table: facts, tables and pictures on request, made by scripts in a sandbox, with every number kept beside them.",
        version="1",
        default_input_modes=["application/json"],
        default_output_modes=["application/json", "text/plain", "image/png", "text/csv"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[SKILL],
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC")],
    )


def parts_of(trail: Trail, out_dir: Path) -> list[tuple[str, list[Part]]]:
    """The task's artifacts, by name: the record, the facts, the summary, the picture, the table, the script, the thoughts, the trail."""
    rec = trail.record
    assert rec is not None
    if rec.status == "needs_input":
        return [("exploration", [H.new_data_part(rec.model_dump(mode="json"))]), ("trail", [H.new_data_part(trail.model_dump(mode="json"))])]
    step = out_dir / f"step_{rec.step}"
    facts = {n: v for n, v in (trail.steps[rec.step - 1].facts or {}).items() if n in rec.facts}
    out: list[tuple[str, list[Part]]] = [
        ("exploration", [H.new_data_part(rec.model_dump(mode="json"))]),
        ("facts", [H.new_data_part(facts)]),
        ("summary", [H.new_text_part(rec.summary)]),
    ]
    if rec.figure:
        out.append(("figure", [H.new_raw_part((step / "figure.png").read_bytes(), media_type="image/png", filename="figure.png")]))
    if rec.table:
        media = "text/csv" if rec.table.endswith(".csv") else "application/json"
        out.append(("table", [H.new_raw_part((step / rec.table).read_bytes(), media_type=media, filename=rec.table)]))
    out.append(("code", [H.new_text_part((step / "code.py").read_text(), media_type="text/x-python")]))
    out.append(("thoughts", [H.new_data_part({"thoughts": [t.model_dump(mode="json") for t in trail.thoughts]})]))
    out.append(
        (
            "trail",
            [H.new_data_part(trail.model_dump(mode="json"))],
        )
    )
    return out


def _explore(req: ExploreRequest | TransformRequest, cancel: threading.Event) -> Trail:
    try:
        with trace.parent(req.trace_headers):  # per-request parent for the persistent agent; process mode also supports trace.parent()
            return agent.transform(req, cancel) if isinstance(req, TransformRequest) else agent.explore(req, cancel)
    finally:
        trace.flush()


class Executor(AgentExecutor):
    def __init__(self):
        self.cancellations: dict[str, threading.Event] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        assert msg is not None
        task = context.current_task
        if task is None:
            task = H.new_task_from_user_message(msg)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        data = H.get_data_parts(msg.parts)
        if not data:
            await updater.failed(updater.new_agent_message([H.new_text_part("the message carries no request as a data part")]))
            return
        cancellation = threading.Event()
        self.cancellations[task.id] = cancellation
        try:
            req = TransformRequest.model_validate(data[0]) if data[0].get("product") == "table" else ExploreRequest.model_validate(data[0])
            trail = await asyncio.to_thread(_explore, req, cancellation)
        except asyncio.CancelledError:
            cancellation.set()
            raise
        except Exception as exc:
            await updater.failed(updater.new_agent_message([H.new_text_part(f"exploration failed: {type(exc).__name__}: {exc}")]))
            return
        finally:
            self.cancellations.pop(task.id, None)
        if cancellation.is_set():
            await updater.failed(updater.new_agent_message([H.new_text_part("exploration cancelled")]))
            return
        if trail.record is None:
            await updater.add_artifact([H.new_data_part({"thoughts": [t.model_dump(mode="json") for t in trail.thoughts]})], name="thoughts")
            await updater.add_artifact(
                [H.new_data_part(trail.model_dump(mode="json"))],
                name="trail",
            )
            await updater.failed(updater.new_agent_message([H.new_text_part(trail.why or "no record passed the gate")]))
            return
        for name, parts in parts_of(trail, req.out_dir):
            await updater.add_artifact(parts, name=name)
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is not None and task.id in self.cancellations:
            self.cancellations[task.id].set()


class BoundedTaskStore(InMemoryTaskStore):
    """A2A delivery cache only. Published history belongs to the integration store."""

    def __init__(self):
        super().__init__()
        self.finished: dict[str, ServerCallContext] = {}

    async def save(self, task, context):
        await super().save(task, context)
        if task.status.state in (TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED):
            self.finished[task.id] = context
            while len(self.finished) > 8:
                oldest = next(iter(self.finished))
                old_context = self.finished.pop(oldest)
                await super().delete(oldest, old_context)


def app(url: str) -> Starlette:
    """The server: the card at its well-known path and the JSON-RPC endpoint at the root."""
    c = card(url)
    handler = DefaultRequestHandler(agent_executor=Executor(), task_store=BoundedTaskStore(), agent_card=c)
    return Starlette(routes=[*create_agent_card_routes(c), *create_jsonrpc_routes(handler, rpc_url="/")])


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    url = f"http://{args.host}:{args.port}/"
    print(json.dumps({"card": url + ".well-known/agent-card.json"}), flush=True)
    uvicorn.run(app(url), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
