"""The desk's side: the folder is allocated on the shelf, the table is projected, the task is sent, and what came back is
written under the folder and saved as an artifact; a decline leaves no folder; an ask naming both the outcome and the treatment
before the freeze is declined with no call; the real process serves its card and stops."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from causal_agent.common import config
from causal_agent.common.llm import set_llm
from causal_agent.explore import client as XC
from causal_agent.explore import server as S
from causal_agent.explore.contracts import Exploration
from causal_agent.explore.tests.fakes import FakeLLM
from causal_agent.explore.tests.test_agent import PICTURE, TABLE
from causal_agent.memory import artifacts as VA

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "subprocess")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    XC.set_app(S.app("http://explore/"))
    yield tmp_path
    XC.set_app(None)
    set_llm(None)


def _ask(ask: str, frozen: bool = False, design: int | None = None):
    return XC.explore(
        dataset="students",
        ask=ask,
        context="[col:lunch.note] standard or free/reduced\n[col:reading_score.note] the reading mark",
        columns={"lunch": "lunch", "reading_score": "reading score", "math_score": "math score", "test_preparation_course": "test preparation course"},
        csv=CSV,
        outcome="math score",
        treatment="test preparation course",
        frozen=frozen,
        design=design,
        memory_version=4,
        data_version=0,
    )


def test_an_exploration_lands_on_the_shelf_with_its_products(tmp_path):
    fake = FakeLLM([[TABLE], []], [Exploration(summary="Standard lunch reads 71.65 on average.", step=1, facts=["reading_standard"], table="by_lunch.csv")])
    set_llm(fake)
    a, d, thoughts = _ask("mean reading score by lunch, as a table")
    assert d is None and a is not None and a.kind == "exploration" and a.design is None and a.table == "by_lunch.csv" and not a.figure
    folder = tmp_path / "data" / "memory" / "students" / "artifacts" / a.id
    assert folder == VA.folder("students", a.id) and not (folder / "input").exists()  # the projected table is removed
    assert {"artifact.json", "facts.json", "code.py", "thoughts.json", "request.json", "by_lunch.csv", "step_1"} <= {p.name for p in folder.iterdir()}
    assert json.loads((folder / "facts.json").read_text()) == {"reading_standard": 71.65} and a.facts == {"reading_standard": 71.65}
    assert "math_score" not in json.loads((folder / "request.json").read_text())["columns"]  # the held-back column was not named
    assert VA.load("students", a.id) == a and [x.id for x in VA.list_artifacts("students")] == [a.id]
    assert any(t.text == "looking" for t in thoughts)


def test_a_picture_is_a_picture_and_is_tagged_with_its_design(tmp_path):
    set_llm(FakeLLM([[PICTURE], []], [Exploration(summary="Two bars.", step=1, figure=True)]))
    a, d, _ = _ask("reading score by lunch, as a picture", frozen=True, design=2)
    assert (
        d is None and a is not None and a.kind == "picture" and a.figure and a.design == 2 and (VA.folder("students", a.id) / "figure.png").stat().st_size > 0
    )


def test_a_refusal_is_a_decline_and_leaves_no_folder(tmp_path):
    bad = Exploration(summary="x", step=1, facts=["reading_standard"])
    set_llm(FakeLLM([[], [], []], [bad, bad, bad]))
    a, d, _ = _ask("anything")
    assert a is None and d is not None and d.check == "explore.failed" and "no script has run yet" in d.reason
    assert (
        VA.list_artifacts("students") == []
        and not (tmp_path / "data" / "memory" / "students" / "artifacts").exists()
        or not list((tmp_path / "data" / "memory" / "students" / "artifacts").iterdir())
    )


def test_an_ask_naming_both_before_the_freeze_is_declined_without_a_call():
    fake = FakeLLM([], [])
    set_llm(fake)
    a, d, _ = _ask("math score by test preparation course")
    assert a is None and d is not None and d.check == "explore.no_join" and fake.humans == [] and fake.bound == []


def test_the_process_serves_its_card_and_stops():
    XC.set_app(None)
    with XC.process() as url:
        card = httpx.get(url + "/.well-known/agent-card.json", timeout=5).json()
        assert card["name"] == S.NAME
