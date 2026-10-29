"""Single-application execution with PostgreSQL receipts and SQLite checkpoints."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import func, select

from causal_agent.desk import graph as chat
from causal_agent.memory import artifacts
from causal_agent.memory.context.review import RevisionConflict
from causal_agent.server.conversations import schema as S
from causal_agent.server.conversations.commands import Busy, Repository
from causal_agent.server.conversations.runtime import Runtime
from causal_agent.server.models import ChatOverview


class Manager:
    def __init__(self, settings):
        self.s = settings
        settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        self.graph = chat.compile_with(self.saver)
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reviewed-chat")
        self.repositories = {}
        self.futures = {}
        self.lock = threading.RLock()

    def bind(self, name, storage, owner):
        with self.lock:
            if name in self.repositories:
                if self.repositories[name].owner != owner:
                    raise ValueError("Conversation owner mismatch.")
                return
            repo = Repository(storage, owner, name)
            state = repo.state()
            if state["status"] == "running":
                repo.interrupt(state["active_operation"], "The application restarted during this turn. Resume explicitly to recover its recorded work.")
            self.repositories[name] = repo

    def view(self, name):
        repo = self.repositories[name]
        state = repo.state()
        focus_label = None
        if state["focus"]:
            focus = state["focus"]
            if focus["kind"] == "column":
                focus_label = next(c.name for c in repo.context.read(state["revision_id"]).knowledge.columns if c.column_id == focus["id"])
            else:
                from causal_agent.server.conversations.publication import Repository as Products

                product = Products(repo.storage, repo.owner, name).load(name, focus["id"])
                focus_label = product.summary
                if focus.get("element"):
                    focus_label += ": " + next(e.label for e in product.elements if e.id == focus["element"])
        with repo.storage.engine.connect() as db:
            count = db.scalar(select(func.count()).select_from(S.messages).where(S.messages.c.conversation_id == name, S.messages.c.owner_id == repo.owner))
            questions = list(
                db.scalars(
                    select(S.questions.c.body).where(
                        S.questions.c.id.in_(state["pending"]), S.questions.c.owner_id == repo.owner, S.questions.c.conversation_id == name
                    )
                )
            )
        return ChatOverview(
            name=name,
            dataset_id=repo.chat["dataset_id"],
            title=repo.chat["title"],
            question=state["question"],
            version_id=state["version_id"],
            revision_id=state["revision_id"],
            status=state["status"],
            prompt=state["prompt"],
            focus=state["focus"],
            focus_label=focus_label,
            fast_track=state["fast_track"],
            questions=questions,
            analysis_review=state["preparation"],
            message_count=count,
            error=state["error"],
            readiness=state["readiness"],
        )

    def send(self, name, command):
        with self.lock:
            repo = self.repositories[name]
            _, fresh = repo.accept(command)
            if fresh:
                self.futures[name] = self.pool.submit(copy_context().run, self.step, name, command.submission_id)

    def resume(self, name):
        with self.lock:
            repo = self.repositories[name]
            with repo.storage.engine.begin() as db:
                state = repo.state(db, lock=True)
                if state["status"] != "interrupted" or not state["active_operation"]:
                    raise Busy("There is no interrupted turn to resume.")
                repo.update(db, status="running", error=None)
            self.futures[name] = self.pool.submit(copy_context().run, self.step, name, state["active_operation"])

    def step(self, name, operation_id):
        repo = self.repositories[name]
        runtime = Runtime(repo, self.s)
        config = {
            "configurable": {"thread_id": name},
            "run_name": "chat:" + name,
            "metadata": {"conversation_id": name, "operation_id": operation_id},
            "tags": ["desk", "reviewed-chat"],
        }
        try:
            with chat.using(runtime), artifacts.using(runtime.artifacts):
                snapshot = self.graph.get_state(config)
                inp = None if snapshot.next and snapshot.values.get("operation_id") == operation_id else {"operation_id": operation_id}
                self.graph.invoke(inp, config)
        except RevisionConflict:
            repo.finish(
                operation_id,
                "The dataset context changed in another chat. Your correction was not applied. Review the latest context and explicitly refresh this chat before trying again.",
            )
        except ValueError as exc:
            if repo.operation_error_is_input(operation_id):
                repo.finish(operation_id, "The requested operation could not be applied: " + str(exc))
            else:
                repo.interrupt(operation_id, f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            repo.interrupt(operation_id, f"{type(exc).__name__}: {exc}")
