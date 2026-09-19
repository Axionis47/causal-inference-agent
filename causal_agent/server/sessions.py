"""One conversation per dataset: the desk graph on a persistent checkpointer, stepped in a worker thread, and a
projection of its state for the page. The manager drives; it never judges."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from causal_agent.desk import graph as desk_graph
from causal_agent.desk.contracts import RunRecord
from causal_agent.memory import store as MS
from causal_agent.server import datasets as DS
from causal_agent.server.models import (Activity, CheckView, ClaimView, EstimateView, InterpretationView, Prompt, QuestionView, RefutationView, RunView,
                                        SessionView, StatusView, Turn)
from causal_agent.server.settings import Settings


class SessionBusy(Exception):
    pass


class SessionEnded(Exception):
    pass


class SessionState(Exception):
    """The requested transition does not apply in this stage."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


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
            sess.thread_id = str(uuid.uuid4())
            sess.ended, sess.error, sess.last_payload = False, None, None
            meta.update({"thread_id": sess.thread_id, "ended": False, "last_prompt": None})
            DS.write_meta(self.s, name, meta)
            inp = {"dataset": name}
            self._append(name, Turn(role="system", text="Conversation started.", at=_now()))
            self._launch(sess, inp)
        return sess

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
        """A new thread after the conversation ended; same description and question."""
        sess = self.get(name)
        if sess.busy:
            raise SessionBusy(name)
        if not sess.ended and sess.thread_id:
            raise SessionState("the conversation has not ended")
        self._append(name, Turn(role="system", text="New conversation.", at=_now()))
        return self.start(name)

    def delete(self, name: str) -> list[str]:
        """Forget the session and its checkpoints; return the run directories the conversation produced."""
        sess = self.get(name)
        if sess.busy:
            raise SessionBusy(name)
        run_dirs: list[str] = []
        if sess.thread_id:
            try:
                vals = self._values(sess)
                run_dirs = [r.run_dir for r in vals.get("runs") or [] if getattr(r, "run_dir", None)]
            except Exception:
                pass
            try:
                self.saver.delete_thread(sess.thread_id)
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
            for ns, mode, chunk in self.graph.stream(inp, sess.cfg, stream_mode=["updates", "tasks"], subgraphs=True):
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
                self._append(sess.name, Turn(role="assistant", text=payload.get("text") or "", phase=payload.get("phase") or "before", at=_now(), figure=payload.get("figure")))
            DS.write_meta(self.s, sess.name, meta)
        except Exception as e:  # the graph's own retries are inside; this is what got through
            sess.error = f"{type(e).__name__}: {e}"
            self._append(sess.name, Turn(role="system", text=f"The step failed: {sess.error}", at=_now(), kind="error"))
        finally:
            sess.activity, sess.busy = None, False

    # ------------------------------------------------------------------ transcript

    def _transcript_path(self, name: str) -> Path:
        return self.s.web_root / name / "transcript.jsonl"

    def _append(self, name: str, turn: Turn) -> None:
        p = self._transcript_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(turn.model_dump_json() + "\n")

    def transcript(self, name: str) -> list[Turn]:
        p = self._transcript_path(name)
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    out.append(Turn.model_validate_json(line))
                except Exception:
                    continue
        return out

    # ------------------------------------------------------------------ projection

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
        sess = self.get(name)
        meta = DS.read_meta(self.s, name) or {}
        title, question = meta.get("title") or name, meta.get("question")
        if not sess.thread_id:
            return SessionView(name=name, title=title, question=question, stage="new", transcript=self.transcript(name))
        snap = self._snapshot(sess)
        values = dict(snap.values or {})
        interrupted = False
        for t in snap.tasks or ():
            if getattr(t, "interrupts", None):
                interrupted = True
            sub = getattr(t, "state", None)
            if sub is not None and hasattr(sub, "values"):
                values.update(sub.values or {})
        prompt_raw = sess.last_payload
        if prompt_raw is None:
            for t in snap.tasks or ():
                if getattr(t, "interrupts", None):
                    prompt_raw = dict(t.interrupts[0].value or {})
                    break
        if prompt_raw is None:
            prompt_raw = meta.get("last_prompt")
        if sess.busy:
            stage = "busy"
        elif sess.error:
            stage = "error"
        elif sess.ended or (not snap.next and values):
            stage = "ended"
        elif interrupted:
            stage = "waiting"
        elif snap.next:
            stage = "stale"  # a node was pending when the process died; Resume continues it
        else:
            stage = "new"
        phase = values.get("phase") or (prompt_raw or {}).get("phase") or "before"
        prompt = Prompt(text=(prompt_raw or {}).get("text") or "", status=(prompt_raw or {}).get("status") or "", ready=bool((prompt_raw or {}).get("ready")),
                        open=list((prompt_raw or {}).get("open") or []), runs=int((prompt_raw or {}).get("runs") or 0), phase=(prompt_raw or {}).get("phase") or "before",
                        kind=(prompt_raw or {}).get("kind")) if prompt_raw else None
        questions: list[QuestionView] = []
        ask = values.get("ask")
        if phase == "before" and ask is not None and stage == "waiting" and (prompt_raw or {}).get("kind") == "ask":
            questions = [QuestionView(keys=list(ask.addresses), field=None, kind=ask.kind, text=ask.text, options=list(ask.options or []), evidence_cites=list(ask.evidence or []), because=list(ask.because or []))]
        claims: list[ClaimView] = []
        if MS.exists(name, self.s.root):
            try:
                table = MS.load(name, self.s.root).to_claims()
                claims = [ClaimView(**c.model_dump()) for c in table.claims.values() if c.status != "empty"]
            except Exception:
                claims = []
        st = values.get("status")
        status = StatusView(**st.model_dump()) if st is not None else None
        runs = [run_view(r) for r in values.get("runs") or []]
        activity = Activity(node=sess.activity[0], since=sess.activity[1]) if sess.activity else None
        return SessionView(name=name, title=title, question=values.get("question") or question, stage=stage, phase=phase, activity=activity,
                           ready=bool(prompt and prompt.ready) if phase == "before" else True, prompt=prompt, questions=questions, claims=claims, status=status,
                           runs=runs, brief=values.get("brief") or "", transcript=self.transcript(name), written=None, error=sess.error)


def run_view(r: RunRecord) -> RunView:
    sr = r.specialist_result or {}
    design = sr.get("design") or {}
    results = (design.get("checks") or {}).get("results") or [] if isinstance(design, dict) else []
    checks = [CheckView(contrast=c.get("contrast"), name=str(c.get("name")), level=c.get("level"), value=c.get("value"), threshold=c.get("threshold"), detail=c.get("detail")) for c in results]
    flags = [c for c in checks if c.level != "pass"]
    refs = [RefutationView(contrast=x.get("contrast"), refuter=str(x.get("refuter")), kind=x.get("kind"), passed=x.get("passed"), p_value=x.get("p_value"), new_effect=x.get("new_effect"), detail=x.get("detail"))
            for x in (sr.get("refutations") or r.artifacts.get("refutations") or [])]
    interps = [InterpretationView(contrast=i.get("contrast"), answer=str(i.get("answer") or ""), caveats=list(i.get("caveats") or []), cites=list(i.get("cites") or []))
               for i in (sr.get("interpretations") or r.artifacts.get("interpretations") or [])]
    ests = [EstimateView(contrast=e.get("contrast"), method=e.get("method"), value=e.get("value"), ci_low=e.get("ci_low"), ci_high=e.get("ci_high"), n=e.get("n"),
                         n_treated=e.get("n_treated"), n_control=e.get("n_control"), secondary=bool(e.get("secondary")), error=e.get("error"))
            for e in (sr.get("estimates") or r.artifacts.get("estimates") or [])]
    run_id = Path(r.run_dir).name if r.run_dir else None
    files = sorted(p.name for p in Path(r.run_dir).iterdir() if p.is_file()) if r.run_dir and Path(r.run_dir).is_dir() else []
    return RunView(index=r.index, question=r.question, family=r.family, specialist=r.specialist, status=r.status, run_id=run_id, effect=r.effect, ci_low=r.ci_low,
                   ci_high=r.ci_high, estimator=r.estimator, decision=dict(r.decision or {}), decision_record=r.decision_record or "", flags=flags, checks=checks,
                   refutations=refs, interpretations=interps, estimates=ests, feasibility=sr.get("feasibility") or r.artifacts.get("feasibility"), files=files,
                   what_if=dict(r.what_if or {}), differs=list(r.differs or []), figures=list(r.figures or []))
