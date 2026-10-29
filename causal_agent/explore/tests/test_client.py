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
from causal_agent.explore.inspection import ExploreAction
from causal_agent.explore.tests.fakes import FakeLLM
from causal_agent.explore.tests.test_agent import ANSWER, PICTURE, TABLE
from causal_agent.memory import artifacts as VA

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch, artifact_repository):
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
        csv=CSV,
        design=design,
        memory_version=4,
        data_version=0,
    )


def test_an_exploration_lands_on_the_shelf_with_its_products(tmp_path):
    fake = FakeLLM([[TABLE], []], [ANSWER])
    set_llm(fake)
    a, d, thoughts = _ask("mean reading score by lunch, as a table")
    assert d is None and a is not None and a.origin == "exploration" and a.format == "table" and a.design is None and a.table == "by_lunch.csv" and not a.figure
    folder = tmp_path / "runs" / "explorations" / "students" / a.id
    assert folder == VA.folder("students", a.id) and not (folder / "input").exists()  # the projected table is removed
    assert {"artifact.json", "facts.json", "code.py", "thoughts.json", "request.json", "by_lunch.csv", "step_1"} <= {p.name for p in folder.iterdir()}
    assert json.loads((folder / "facts.json").read_text()) == {"reading_standard": 71.65} and a.facts == {"reading_standard": 71.65}
    assert json.loads((folder / "request.json").read_text())["dataset_version"]["sha256"] == a.source_sha256
    assert a.method == ANSWER.method and a.findings == ANSWER.findings
    assert VA.load("students", a.id) == a and [x.id for x in VA.list_artifacts("students")] == [a.id]
    assert any(t.text == "looking" for t in thoughts)


def test_a_picture_is_a_picture_and_is_tagged_with_its_design(tmp_path):
    set_llm(FakeLLM([[PICTURE, ExploreAction(action="inspect_output", reference="step_1/figure.png")]], [ANSWER.model_copy(update={"figure": True})]))
    a, d, _ = _ask("reading score by lunch, as a picture", frozen=True, design=2)
    assert (
        d is None
        and a is not None
        and a.origin == "exploration"
        and a.format == "figure"
        and a.figure
        and a.design == 2
        and (VA.folder("students", a.id) / "figure.png").stat().st_size > 0
    )


def test_a_refusal_is_a_decline_and_leaves_no_folder(tmp_path):
    bad = Exploration(summary="x", step=1, facts=["reading_standard"])
    set_llm(FakeLLM([[], [], []], [bad, bad, bad]))
    a, d, _ = _ask("anything")
    assert a is None and d is not None and d.check == "explore.failed" and "no script has run yet" in d.reason
    assert VA.list_artifacts("students") == []
    failures = list((tmp_path / "runs/explorations/students").glob("*/trail.json"))
    assert len(failures) == 1


def test_the_process_serves_its_card_and_stops():
    XC.set_app(None)
    with XC.process() as url:
        card = httpx.get(url + "/.well-known/agent-card.json", timeout=5).json()
        assert card["name"] == S.NAME


def test_published_explanation_and_files_feed_a_followup_visualization(tmp_path):
    from causal_agent.desk.material import Material, add_artifacts

    set_llm(FakeLLM([[TABLE]], [ANSWER]))
    first, decline, _ = _ask("mean reading score by lunch")
    assert first is not None and decline is None
    inputs = XC.artifact_inputs("students", [first.id])
    assert {item.path.name for item in inputs} == {"facts.json", "result.json", "by_lunch.csv"}
    saved = {item.id: item.sha256 for item in inputs}
    code = """import os, json
from pathlib import Path
import matplotlib.pyplot as plt
root = Path(os.environ['INPUTS_DIR'])
manifest = json.loads((root / 'manifest.json').read_text())
facts_path = next(root / name for name, meta in manifest.items() if meta['id'].endswith('/facts.json'))
facts = json.loads(facts_path.read_text())
plt.bar(['standard lunch'], [facts['reading_standard']])
plt.ylabel('Saved mean reading score')
plt.tight_layout()
plt.savefig('figure.png')
Path('facts.json').write_text(json.dumps(facts))
"""
    second_answer = ANSWER.model_copy(update={"table": None, "figure": True, "method": "Plot the saved mean without recomputing it."})
    set_llm(FakeLLM([[code, ExploreAction(action="inspect_output", reference="step_1/figure.png")]], [second_answer]))
    second, decline, _ = XC.explore(
        dataset="students",
        ask="Plot the earlier result",
        context="Use the explicitly supplied saved result.",
        csv=CSV,
        design=None,
        memory_version=4,
        data_version=0,
        artifact_refs=inputs,
        thread_id="same-chat",
    )
    assert second is not None and decline is None
    assert second.facts == first.facts and second.parent_refs == list(saved)
    assert {item.id: item.sha256 for item in XC.artifact_inputs("students", [first.id])} == saved
    material = Material()
    add_artifacts(material, "students", None)
    assert material.pull(f"{second.address}.method").text == second.method
    assert material.pull(f"{first.address}.reading_standard").value == 71.65
    assert (VA.folder("students", second.id) / "result.json").is_file()
