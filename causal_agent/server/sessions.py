"""One conversation at a time per dataset, any number in turn: the desk graph on a persistent checkpointer, stepped in a
worker thread. A new analysis is a new thread over the same memory; the designs every thread ran stay on disk and stay
listed. The manager drives; it never judges. The transcript on disk is transcript.py; the projection for the page is
views.py."""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from causal_agent.desk import graph as desk_graph
from causal_agent.server import datasets as DS
from causal_agent.server import transcript as T
from causal_agent.server import views as V
from causal_agent.server.models import SessionView, Turn
from causal_agent.server.settings import Settings
from causal_agent.server.views import run_view as run_view  # the projection of one run, as the tests import it


class SessionBusy(Exception):
    pass


class SessionEnded(Exception):
    pass


class SessionState(Exception):
    """The requested transition does not apply in this stage."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass
class Session:
    name: str
    thread_id: str
    busy: bool = False
    future: Future | None = None
    activity: tuple[str, str] | None = None
    last_payload: dict | None = None
    ended: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def cfg(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "tags": [f"dataset:{self.name}", "desk", "web"], "metadata": {"dataset": self.name}}


class SessionManager:
    def __init__(self, settings: Settings, workers: int = 4):
        self.s = settings
        self.s.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.s.checkpoint_db), check_same_thread=False)
        self.saver = SqliteSaver(conn, serde=desk_graph.serde)
        self.graph = desk_graph.compile_with(self.saver)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="desk")
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lookup

    def get(self, name: str) -> Session:
        with self._lock:
            if name in self.sessions:
                return self.sessions[name]
            meta = DS.read_meta(self.s, name)
            if meta is None:
                raise DS.NotFound(f"no dataset named {name!r}")
            sess = Session(name=name, thread_id=meta.get("thread_id") or "", last_payload=meta.get("last_prompt"), ended=bool(meta.get("ended")))
            self.sessions[name] = sess
            return sess

    def forget(self, name: str) -> None:
        with self._lock:
            self.sessions.pop(name, None)

    # ------------------------------------------------------------------ transitions

    def start(self, name: str) -> Session:
        """First conversation for a dataset the server created."""
        meta = DS.read_meta(self.s, name)
        if meta is None:
            raise DS.NotFound(f"no dataset named {name!r}")
        sess = self.get(name)
        with sess.lock:
            if sess.busy:
                raise SessionBusy(name)
            self._append(name, Turn(role="system", text="Conversation started.", at=_now()))
            self._begin(sess, meta)
        return sess

    def new_analysis(self, name: str) -> Session:
        """A new thread over the same memory, whenever the desk is not working: mid-interview, after a run, after the end, or
        before any thread. The thread so far is listed under the meta's analyses and its designs stay on disk."""
        meta = DS.read_meta(self.s, name)
        if meta is None:
            raise DS.NotFound(f"no dataset named {name!r}")
        sess = self.get(name)
        with sess.lock:
            if sess.busy:
                raise SessionBusy(name)
            if sess.thread_id:
                past = list(meta.get("analyses") or [])
                past.append({"thread_id": sess.thread_id, "question": meta.get("question"), "started_at": meta.get("started_at"), "ended": True})
                meta["analyses"] = past
            meta["question"] = None
            self._append(name, Turn(role="system", kind="divider", text="New analysis.", at=_now()))
            self._begin(sess, meta)
        return sess

    def _begin(self, sess: Session, meta: dict) -> None:
        """A fresh thread for the session, under its lock."""
        sess.thread_id = str(uuid.uuid4())
        sess.ended, sess.error, sess.last_payload = False, None, None
        meta.update({"thread_id": sess.thread_id, "ended": False, "last_prompt": None, "started_at": _now()})
        DS.write_meta(self.s, sess.name, meta)
        self._launch(sess, {"dataset": sess.name})

    def send(self, name: str, text: str) -> Session:
        sess = self.get(name)
        with sess.lock:
            if sess.busy:
                raise SessionBusy(name)
            if sess.ended or not sess.thread_id:
                raise SessionEnded(name)
            if self._pending_without_interrupt(sess):
                raise SessionState("the last step did not finish; resume it first")
            phase = (sess.last_payload or {}).get("phase") or "before"
            self._append(name, Turn(role="user", text=text, phase=phase, at=_now()))
            self._launch(sess, Command(resume=text))
        return sess

    def resume(self, name: str) -> Session:
        """Continue a checkpoint whose worker died with the process."""
        sess = self.get(name)
        with sess.lock:
            if sess.busy:
                raise SessionBusy(name)
            if sess.ended or not sess.thread_id:
                raise SessionEnded(name)
            sess.error = None
            self._launch(sess, None)
        return sess

    def restart(self, name: str) -> Session:
        """The old name for a new analysis."""
        return self.new_analysis(name)

    def threads(self, name: str) -> list[str]:
        """Every thread the dataset has had: the ones listed as past analyses, then the current one."""
        meta = DS.read_meta(self.s, name) or {}
        past = [str(a.get("thread_id")) for a in meta.get("analyses") or [] if a.get("thread_id")]
        cur = self.get(name).thread_id
        return past + ([cur] if cur and cur not in past else [])

    def delete(self, name: str) -> list[str]:
        """Forget the session and every thread's checkpoints; return the run directories the dataset's designs produced."""
        sess = self.get(name)
        if sess.busy:
            raise SessionBusy(name)
        run_dirs = [r.run_dir for r in V.disk_records(self.s, name) if r.run_dir]
        if sess.thread_id:
            try:
                vals = self._values(sess)
                run_dirs += [r.run_dir for r in vals.get("runs") or [] if getattr(r, "run_dir", None) and r.run_dir not in run_dirs]
            except Exception:
                pass
        for tid in self.threads(name):
            try:
                self.saver.delete_thread(tid)
            except Exception:
                pass
        self.forget(name)
        return run_dirs

    def _pending_without_interrupt(self, sess: Session) -> bool:
        """A checkpoint whose next node never ran to an interrupt: the worker died with the process."""
        try:
            snap = self._snapshot(sess)
        except Exception:
            return False
        return bool(snap.next) and not any(getattr(t, "interrupts", None) for t in snap.tasks or ())

    # ------------------------------------------------------------------ stepping

    def _launch(self, sess: Session, inp: Any) -> None:
        sess.busy, sess.error, sess.activity = True, None, ("starting", _now())
        sess.future = self.pool.submit(self._step, sess, inp)

    def _step(self, sess: Session, inp: Any) -> None:
        payload: dict | None = None
        try:
            for _ns, mode, chunk in self.graph.stream(inp, sess.cfg, stream_mode=["updates", "tasks"], subgraphs=True):
                if not isinstance(chunk, dict):
                    continue
                if mode == "tasks":
                    # a task start names the node now running; the parent's "interview" wraps the subgraph's own nodes
                    if "triggers" in chunk and chunk.get("name") is not None:
                        sess.activity = (str(chunk["name"]), _now())
                    continue
                if "__interrupt__" in chunk:
                    ints = chunk["__interrupt__"]
                    if ints:
                        payload = dict(ints[0].value or {})
            meta = DS.read_meta(self.s, sess.name) or {}
            if payload is None:
                sess.ended, sess.last_payload = True, None
                meta.update({"ended": True, "last_prompt": None})
                self._append(sess.name, Turn(role="system", text="Conversation ended.", phase=self._phase(sess), at=_now()))
            else:
                sess.last_payload = payload
                meta["last_prompt"] = payload
                try:
                    q = (self._values(sess) or {}).get("question")
                    if q and not (self._values(sess) or {}).get("invalid"):
                        meta["question"] = q
                except Exception:
                    pass
                self._append(
                    sess.name,
                    Turn(role="assistant", text=payload.get("text") or "", phase=payload.get("phase") or "before", at=_now(), figure=payload.get("figure")),
                )
            DS.write_meta(self.s, sess.name, meta)
        except Exception as e:  # the graph's own retries are inside; this is what got through
            sess.error = f"{type(e).__name__}: {e}"
            self._append(sess.name, Turn(role="system", text=f"The step failed: {sess.error}", at=_now(), kind="error"))
        finally:
            sess.activity, sess.busy = None, False

    # ------------------------------------------------------------------ transcript and projection (see transcript.py, views.py)

    def _append(self, name: str, turn: Turn) -> None:
        T.append(self.s, name, turn)

    def transcript(self, name: str) -> list[Turn]:
        return T.read(self.s, name)

    def _snapshot(self, sess: Session):
        return self.graph.get_state(sess.cfg, subgraphs=True)

    def _values(self, sess: Session) -> dict:
        """Parent values overlaid with the interview subgraph's while it is the one interrupted."""
        snap = self._snapshot(sess)
        v = dict(snap.values or {})
        for t in snap.tasks or ():
            sub = getattr(t, "state", None)
            if sub is not None and hasattr(sub, "values"):
                v.update(sub.values or {})
        return v

    def _phase(self, sess: Session) -> str:
        return (sess.last_payload or {}).get("phase") or "before"

    def view(self, name: str) -> SessionView:
        return V.session_view(self, name)
