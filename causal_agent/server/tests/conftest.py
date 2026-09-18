"""A server over a temp root with the interview and desk fakes. No Vertex calls, nothing written under data/."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from causal_agent.common.llm import set_llm
from causal_agent.intake import datasets as DSI
from causal_agent.intake.interview import data as D
from causal_agent.intake.interview import writer as W
from causal_agent.server.app import create_app
from causal_agent.server.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IT = _load("interview_tests", "causal_agent/intake/interview/tests/test_interview.py")
CT = _load("chat_tests", "causal_agent/chat/tests/test_chat.py")


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "datasets.yaml").write_text("# test index\n{}\n")
    monkeypatch.setattr(W, "ROOT", tmp_path)
    monkeypatch.setattr(DSI, "ROOT", tmp_path)
    return Settings(root=tmp_path, run_root=tmp_path / "runs", dist=tmp_path / "dist")


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
    app.state.sessions.pool.shutdown(wait=True)
    set_llm(None)
    D.clear()


def wait_idle(client: TestClient, name: str, timeout: float = 30.0) -> dict:
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


def create_request(prof: dict, name: str = "students_web", question: str = "Did completing the prep course raise math scores?") -> dict:
    lines = {
        "gender": "student's gender, recorded at enrolment",
        "race/ethnicity": "the school's grouping, recorded at enrolment",
        "parental level of education": "declared at enrolment",
        "lunch": "standard or free/reduced, set by the district before the exam",
        "test preparation course": "completed or none, decided by offer and uptake before the exam",
        "math score": "the exam mark, 0 to 100, recorded at the sitting",
        "reading score": "the exam mark, 0 to 100, recorded at the sitting",
        "writing score": "the exam mark, 0 to 100, recorded at the sitting",
    }
    return {"name": name, "title": "Students web", "upload_id": prof["upload_id"], "question": question,
            "about": "Each row is one student's results from the May 2026 exam at one school; every student who sat is included",
            "changed": "The school ran a six-week prep course before the exam; places were offered first to free-lunch students and to those whose parents hold no degree, then to anyone who asked. Completion was recorded by the counsellor",
            "columns": [{"name": c["name"], "description": lines.get(c["name"], "")} for c in prof["columns"]]}
