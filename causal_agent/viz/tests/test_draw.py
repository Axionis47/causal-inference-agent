"""The drawing tool with a scripted model: the scripts it hands back run in the sandbox on the students file, under a temp
root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from causal_agent.common import config
from causal_agent.common.llm import set_llm
from causal_agent.viz import draw as D
from causal_agent.viz import sandbox, store

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"

GOOD = """
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(os.environ["VIZ_CSV"])
means = df.groupby("lunch")["math score"].mean()
fig, ax = plt.subplots()
ax.bar(means.index, means.values)
ax.set_ylabel("mean math score")
fig.savefig("figure.png")
json.dump({"mean_standard": float(means["standard"]), "mean_free_reduced": float(means["free/reduced"])}, open("facts.json", "w"))
"""
RAISES = "raise RuntimeError('boom: no such column')\n"
EXTRA = GOOD + "open('extra.txt', 'w').write('scratch')\n"
STRING_FACT = GOOD + "json.dump({'mean_standard': 'high', 'mean_free_reduced': 58.9}, open('facts.json', 'w'))\n"


class FakeLLM:
    """Answers each call with the next scripted DrawCode and keeps the prompts it saw."""

    def __init__(self, *codes: str, facts=("mean_standard", "mean_free_reduced")):
        self.codes, self.facts, self.prompts = list(codes), list(facts), []

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                fake.prompts.append(messages[1][1])
                code = fake.codes.pop(0)
                raw = AIMessage(
                    content=[{"type": "thinking", "thinking": "t"}, "{}"], usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
                )
                return {
                    "raw": raw,
                    "parsed": D.DrawCode(code=code, caption="Standard-lunch students score higher in math.", facts=fake.facts),
                    "parsing_error": None,
                }

        return R()


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    yield tmp_path
    set_llm(None)


def request(moment="pre", design=None) -> D.DrawRequest:
    return D.DrawRequest(
        dataset="students",
        moment=moment,
        design=design,
        memory_version=2,
        ask="mean math score by lunch",
        context="[col:lunch.note] standard or free/reduced\n[col:math_score.note] the exam score",
        csv=CSV,
        columns={"lunch": "lunch", "math_score": "math score"},
    )


def test_a_good_script_becomes_an_artifact(tmp_path):
    fake = FakeLLM(GOOD)
    set_llm(fake)
    a, decline, thoughts = D.draw(request())
    assert decline is None and a is not None and len(thoughts) == 1 and thoughts[0].node == "draw"
    folder = store.folder("students", "pre", None, a.id)
    assert folder.parent == tmp_path / "data" / "memory" / "students" / "viz" / "pre"
    assert sorted(p.name for p in folder.iterdir()) == ["artifact.json", "code.py", "facts.json", "figure.png", "request.json"] == a.files
    assert set(a.facts) == {"mean_standard", "mean_free_reduced"} and all(isinstance(v, float) for v in a.facts.values())
    assert a.facts["mean_standard"] > a.facts["mean_free_reduced"]
    assert len(a.addresses()) == 3 and a.render().count("[artifact:") == 3
    assert [x.id for x in store.list_artifacts("students", "pre")] == [a.id]
    req = json.loads((folder / "request.json").read_text())
    assert req["ask"] == "mean math score by lunch" and "context" not in req
    assert "use the file's own name" in fake.prompts[0] and "math score" in fake.prompts[0] and "PREVIOUS ATTEMPT" not in fake.prompts[0]


def test_a_post_artifact_lives_under_its_design(tmp_path):
    set_llm(FakeLLM(GOOD))
    a, decline, _ = D.draw(request(moment="post", design=2))
    assert decline is None and a is not None and a.design == 2
    assert store.folder("students", "post", 2, a.id).parent == tmp_path / "data" / "memory" / "students" / "designs" / "2" / "viz"
    assert [x.id for x in store.list_artifacts("students", "post", 2)] == [a.id]
    assert store.list_artifacts("students", "pre") == []


def test_a_failed_script_is_retried_with_its_error():
    fake = FakeLLM(RAISES, GOOD)
    set_llm(fake)
    a, decline, thoughts = D.draw(request())
    assert decline is None and a is not None and len(fake.prompts) == 2 and len(thoughts) == 2
    assert "PREVIOUS ATTEMPT FAILED" in fake.prompts[1] and "boom: no such column" in fake.prompts[1]
    assert len(store.list_artifacts("students", "pre")) == 1
    assert (store.folder("students", "pre", None, a.id) / "code.py").read_text() == GOOD


def test_three_failures_are_a_decline_and_no_folder(tmp_path):
    fake = FakeLLM(RAISES, RAISES, RAISES)
    set_llm(fake)
    a, decline, thoughts = D.draw(request())
    assert a is None and decline is not None and len(fake.prompts) == 3 and len(thoughts) == 3
    assert decline.stage == "draw" and decline.check == "draw.failed" and "boom" in decline.reason
    assert store.list_artifacts("students", "pre") == []
    assert not (tmp_path / "data" / "memory" / "students" / "viz").exists()


def test_an_extra_file_is_swept():
    set_llm(FakeLLM(EXTRA))
    a, decline, _ = D.draw(request())
    assert decline is None and a is not None
    folder = store.folder("students", "pre", None, a.id)
    assert not (folder / "extra.txt").exists() and not (folder / sandbox.MPL_DIR).exists()
    assert "extra.txt" not in a.files


def test_a_string_fact_fails_the_check():
    fake = FakeLLM(STRING_FACT, GOOD)
    set_llm(fake)
    a, decline, _ = D.draw(request())
    assert decline is None and a is not None and len(fake.prompts) == 2
    assert "not numbers: mean_standard" in fake.prompts[1]


def test_the_sandbox_times_out(tmp_path):
    out = sandbox.run("import time\ntime.sleep(5)\n", CSV, tmp_path / "slow", timeout=0.5)
    assert not out.ok and out.stderr == "timed out after 0.5s" and out.files == ["code.py"]
