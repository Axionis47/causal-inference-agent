"""The agent end to end on the students table with a scripted model and the real sandbox: a good step then a record; each gate
refusal re-prompts with its reason; three refusals leave no record."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_agent.common.llm import set_llm
from causal_agent.explore import agent as AG
from causal_agent.explore import project as PJ
from causal_agent.explore.contracts import Exploration, ExploreRequest
from causal_agent.explore.tests.fakes import FakeLLM

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"

TABLE = """import os, json, pandas as pd
df = pd.read_csv(os.environ["VIZ_CSV"])
t = df.groupby("lunch")["reading score"].mean().round(2)
print(t.to_string())
json.dump({"reading_standard": float(t["standard"]), "reading_free_reduced": float(t["free/reduced"])}, open("facts.json", "w"))
t.rename_axis("lunch").reset_index(name="mean reading score").to_csv("by_lunch.csv", index=False)
"""
PICTURE = (
    TABLE
    + """import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t.plot.bar(); plt.savefig("figure.png")
"""
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "subprocess")
    yield
    set_llm(None)


def request(tmp_path, ask="mean reading score by lunch, as a table") -> ExploreRequest:
    out = tmp_path / "art"
    path = PJ.project(CSV, out, drop="math score")
    return ExploreRequest(
        dataset="students",
        ask=ask,
        context="[col:lunch.note] standard or free/reduced\n[col:reading_score.note] the reading mark, 0 to 100",
        columns={"lunch": "lunch", "reading_score": "reading score", "test_preparation_course": "test preparation course"},
        csv=path,
        out_dir=out,
        outcome="math score",
        treatment="test preparation course",
        dropped="outcome",
        frozen=False,
    )


def test_a_good_step_and_a_record_make_a_trail(tmp_path):
    fake = FakeLLM(
        [[TABLE], []],
        [
            Exploration(
                summary="Students on standard lunch read 71.65 on average, against 64.65.",
                step=1,
                facts=["reading_standard", "reading_free_reduced"],
                table="by_lunch.csv",
            )
        ],
    )
    set_llm(fake)
    trail = AG.explore(request(tmp_path))
    assert trail.record is not None and trail.why == "" and trail.calls == 1 and [s.ok for s in trail.steps] == [True]
    assert trail.steps[0].facts == {"reading_standard": 71.65, "reading_free_reduced": 64.65} and trail.sandbox == "subprocess"
    assert fake.bound == ["run_code"] and "step 1 ran" in fake.tool_texts[0] and "FACTS YOU ASKED FOR" in fake.humans[0]
    assert "held back until the design has run" in fake.humans[0] and "'math score'" in fake.humans[0]  # the model is told what it cannot see
    assert any(t.text == "looking" for t in trail.thoughts)


def test_each_refusal_is_re_prompted_with_its_reason_and_three_leave_no_record(tmp_path):
    bad = [
        Exploration(summary="x", step=1, facts=["reading_standard"]),  # before any script ran
        Exploration(summary="x", step=1, facts=["not_there"], figure=True, table="nope.csv"),  # after one: facts missing, no figure, no table
        Exploration(
            summary="math score rose with the test preparation course, by 5.78 points.", step=1, facts=["reading_standard"]
        ),  # the join, and a loose number
    ]
    fake = FakeLLM([[], [TABLE], []], bad)
    set_llm(fake)
    trail = AG.explore(request(tmp_path))
    assert trail.record is None and trail.calls == 1
    assert "no script has run yet" in fake.humans[1] and "lacks: not_there" in fake.humans[2] and "wrote no figure.png" in fake.humans[2]
    assert "did not write" in fake.humans[2]
    assert "nothing may relate the outcome to the treatment" in trail.why and "5.78" in trail.why


def test_a_picture_is_a_product_too_and_a_record_without_any_product_is_refused(tmp_path):
    fake = FakeLLM([[PICTURE], []], [Exploration(summary="Two bars.", step=1), Exploration(summary="Two bars.", step=1, figure=True)])
    set_llm(fake)
    trail = AG.explore(request(tmp_path, ask="reading score by lunch, as a picture"))
    assert trail.record is not None and trail.record.figure and "names no product" in fake.humans[1]
    assert (tmp_path / "art" / "step_1" / "figure.png").stat().st_size > 0
