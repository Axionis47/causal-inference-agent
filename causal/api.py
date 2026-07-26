"""Six endpoints and a live tape.

    GET  /health
    GET  /datasets                 the eight shipped datasets
    POST /jobs                     start a run; it parks at the design gate
    GET  /jobs/{id}                where it got to, and the menu
    POST /jobs/{id}/design         choose a design; the run continues
    GET  /jobs/{id}/stream         SSE, replayed from the start on connect
    GET  /jobs/{id}/result         the estimate and the readout

There is no job database. LangGraph's checkpoint is the store: a job id is a
thread id, and the state of a run is whatever the checkpointer holds. One
source of truth rather than two that can disagree.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .graph import build
from .kaggle import KaggleError, fetch

DATA = Path(__file__).parent.parent / "data"

# The event names the UI depends on. Renaming one breaks the tape silently,
# so they are pinned by a test.
EVENTS = ("stage_started", "stage_done", "waiting_for_you", "completed", "failed")

app = FastAPI(title="causal-engine")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
graph = build()


class NewJob(BaseModel):
    """One of `dataset` (a bundled name) or `kaggle` (a URL or owner/name)."""

    question: str
    dataset: str = ""
    kaggle: str = ""
    context: str = ""


class Choice(BaseModel):
    lane: str
    kwargs: dict = {}


def _config(job_id: str) -> dict:
    return {"configurable": {"thread_id": job_id}}


def _snapshot(job_id: str):
    snap = graph.get_state(_config(job_id))
    if not snap.values:
        raise HTTPException(404, f"no job {job_id}")
    return snap


def _status(snap) -> str:
    if snap.values.get("error"):
        return "failed"
    if snap.values.get("narrative"):
        return "completed"
    if snap.next and "gate" in snap.next:
        return "waiting_for_you"
    return "running"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/datasets")
def datasets() -> list[dict]:
    """What a person can analyse without bringing their own file."""
    import pandas as pd

    out = []
    for path in sorted(DATA.glob("*.csv")):
        head = pd.read_csv(path, nrows=200)
        out.append({
            "name": path.stem,
            "columns": [str(c) for c in head.columns],
            "n_columns": len(head.columns),
        })
    return out


@app.post("/jobs", status_code=201)
async def create_job(body: NewJob, background: BackgroundTasks) -> dict:
    note = ""
    source = ""
    if body.kaggle:
        # the download is the slow part, so do it before the job exists rather
        # than inside the graph: a bad URL should be a 400, not a failed run
        try:
            got = await asyncio.to_thread(fetch, body.kaggle)
        except KaggleError as exc:
            raise HTTPException(400, str(exc)) from exc
        path, note, source = got.csv, got.note, got.slug
    elif body.dataset:
        path = DATA / f"{body.dataset}.csv"
        source = body.dataset
        if not path.exists():
            raise HTTPException(400, f"no dataset '{body.dataset}'")
    else:
        raise HTTPException(400, "give either a bundled dataset or a kaggle url")

    job_id = uuid.uuid4().hex[:8]

    def run() -> None:
        graph.invoke(
            {"csv_path": str(path), "question": body.question,
             "context": body.context, "source": source, "source_note": note},
            _config(job_id),
        )

    background.add_task(asyncio.to_thread, run)
    return {"id": job_id, "status": "running", "source": source, "note": note}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    snap = _snapshot(job_id)
    v = snap.values
    return {
        "id": job_id,
        "status": _status(snap),
        "question": v.get("question"),
        "source": v.get("source"),
        "source_note": v.get("source_note"),
        "n_rows": v.get("n_rows"),
        "columns": v.get("columns", []),
        "intake": v.get("intake"),
        "menu": v.get("menu", []),
        "roles": v.get("roles", {}),
        "recommendation": v.get("recommendation", {}),
        "suggestions": v.get("suggestions", {}),
        "error": v.get("error"),
    }


@app.post("/jobs/{job_id}/design")
async def choose_design(job_id: str, body: Choice, background: BackgroundTasks) -> dict:
    snap = _snapshot(job_id)
    if _status(snap) != "waiting_for_you":
        raise HTTPException(409, f"job {job_id} is {_status(snap)}, not waiting")

    def resume() -> None:
        graph.invoke(Command(resume=body.model_dump()), _config(job_id))

    background.add_task(asyncio.to_thread, resume)
    return {"id": job_id, "status": "running", "lane": body.lane}


@app.get("/jobs/{job_id}/stream")
async def stream(job_id: str) -> EventSourceResponse:
    async def events():
        # A client opens the tape immediately after POST /jobs, often before the
        # background task has written anything. Wait for the run to appear
        # rather than 404 on a job that is about to exist.
        for _ in range(50):
            if graph.get_state(_config(job_id)).values:
                break
            await asyncio.sleep(0.2)
        else:
            yield {"event": "failed", "data": json.dumps({"reason": f"no job {job_id}"})}
            return

        sent = 0
        while True:
            snap = graph.get_state(_config(job_id))
            log = snap.values.get("events", [])
            for item in log[sent:]:
                yield {"event": item["event"], "data": json.dumps(item)}
            sent = len(log)

            status = _status(snap)
            if status == "waiting_for_you":
                yield {"event": "waiting_for_you",
                       "data": json.dumps({"menu": snap.values.get("menu", [])})}
                return
            if status in ("completed", "failed"):
                return
            await asyncio.sleep(0.4)

    return EventSourceResponse(events())


@app.get("/jobs/{job_id}/result")
def result(job_id: str) -> dict:
    snap = _snapshot(job_id)
    status = _status(snap)
    if status not in ("completed", "failed"):
        raise HTTPException(409, f"job {job_id} is {status}")
    v = snap.values
    return {
        "id": job_id,
        "status": status,
        "lane": (v.get("choice") or {}).get("lane"),
        "estimate": v.get("estimate"),
        "strength": v.get("strength"),
        "headline": v.get("headline"),
        "narrative": v.get("narrative"),
        "error": v.get("error"),
    }
