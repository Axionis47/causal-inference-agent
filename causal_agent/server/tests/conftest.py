"""A server over a temp root with the desk's scripted model. No Vertex calls, nothing written under data/."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from causal_agent.common.llm import set_llm
from causal_agent.desk import pipeline
from causal_agent.desk.contracts import RunRecord
from causal_agent.profile import data as D
from causal_agent.profile import datasets as DSI
from causal_agent.server.app import create_app
from causal_agent.server.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
QUESTION = "Did completing the prep course raise math scores?"


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "datasets.yaml").write_text("# test index\n{}\n")
    monkeypatch.setattr(DSI, "ROOT", tmp_path)
    monkeypatch.setenv("PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    return Settings(root=tmp_path, run_root=tmp_path / "runs", dist=tmp_path / "dist")


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    app.state.sessions.pool.shutdown(wait=True)
    set_llm(None)
    D.clear()


def wait_idle(client: TestClient, name: str, timeout: float = 60.0) -> dict:
    t0 = time.time()
    while True:
        v = client.get(f"/api/sessions/{name}").json()
        if v["stage"] != "busy":
            return v
        if time.time() - t0 > timeout:
            raise AssertionError(f"still busy after {timeout}s: {v.get('activity')}")
        time.sleep(0.05)


def upload(client: TestClient, path: Path = STUDENTS) -> dict:
    r = client.post("/api/profile", files={"file": (path.name, path.read_bytes(), "text/csv")})
    assert r.status_code == 201, r.text
    return r.json()


def create_request(prof: dict, name: str = "students_web") -> dict:
    return {"name": name, "title": "Students web", "upload_id": prof["upload_id"]}


def canned_run(path, n, dataset, question, decision=None, decision_record="", effect: float = 5.618) -> RunRecord:
    sr = {"status": "done", "design": {"estimator": "linear_regression", "contrast": {"treated": "completed", "control": "none"},
                                       "checks": {"results": [{"contrast": "completed_vs_none", "name": "overlap", "level": "pass", "value": 0.9, "threshold": 0.1, "detail": "fine"}]}},
          "estimates": [{"contrast": "completed_vs_none", "method": "linear_regression", "value": effect, "ci_low": effect - 2, "ci_high": effect + 2, "secondary": False, "error": None}],
          "refutations": [], "interpretations": [], "feasibility": None}
    return RunRecord(index=n, dataset=dataset, question=question, family="adjustment", specialist="dowhy", status="done", effect=effect, ci_low=effect - 2, ci_high=effect + 2,
                     estimator="linear_regression", decision=decision or {}, decision_record=decision_record, specialist_result=sr, design_dir=str(Path(path).parent))


def patch_pipeline(monkeypatch, fn=canned_run):
    monkeypatch.setattr(pipeline, "run", fn)


# the answers a person gives to a bare file, one field per turn, in the desk's own address words
ANSWERS = {
    "claim:grain.row_is": "one student's exam results", "claim:grain.panel": "false", "claim:sampling.how": "whole",
    "claim:change.what": "a six-week prep course", "claim:change.to_whom": "students at the school", "claim:change.when": "the six weeks before the exam",
    "claim:assignment.kind": "own_choice", "claim:assignment.rule": "offered first by lunch and parental education, then anyone who asked",
    "claim:assignment.treated_level": "completed", "claim:unobserved.exists": "false", "claim:spillover.possible": "false", "claim:exclusion.exists": "false",
}
WHEN = {"math score": "after", "test preparation course": "at"}


def reply_for(view: dict) -> str:
    """The person's answer to the one question the page shows (the session view's `questions`)."""
    q = (view.get("questions") or [{}])[0]
    addrs = q.get("keys") or []
    if q.get("kind") == "confirm":
        return "yes, all right"
    if q.get("kind") == "columns":
        return "cols"
    if addrs and addrs[0] in ANSWERS:
        return f"{addrs[0]} = {ANSWERS[addrs[0]]}"
    return "I don't know"


def infer_columns(msg, addrs, human):
    """The scripted model's answer to the columns tick: what each in-play column records and when it was set."""
    from causal_agent.desk.contracts import FieldUpdate, Inference

    if msg != "cols":
        return None
    ups = []
    for a in addrs:
        col = a[4:].split(".")[0].replace("_", " ")
        name = next((n for n in ("math score", "test preparation course", "lunch", "parental level of education") if n.replace(" ", "_") == a[4:].split(".")[0]), col)
        if a.endswith(".meaning"):
            ups.append(FieldUpdate(address=a, value=f"{name} as recorded", said=msg))
        elif a.endswith(".when"):
            ups.append(FieldUpdate(address=a, value=WHEN.get(name, "before"), said=msg))
    return Inference(updates=ups)
