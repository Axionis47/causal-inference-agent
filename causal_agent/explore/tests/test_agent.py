"""Whole-data exploration through the real isolated runtime, with deterministic model records."""

from pathlib import Path

import pytest

from causal_agent.common.llm import set_llm
from causal_agent.explore import agent as AG
from causal_agent.explore.contracts import DatasetInput, Exploration, ExploreRequest
from causal_agent.explore.inputs import digest
from causal_agent.explore.inspection import ExploreAction
from causal_agent.explore.tests.fakes import FakeLLM

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
TABLE = """import os, json, pandas as pd
df = pd.read_csv(os.environ["DATA_CSV"])
assert "math score" in df and "test preparation course" in df
t = df.groupby("lunch")["reading score"].mean().round(2)
json.dump({"reading_standard": float(t["standard"]), "reading_free_reduced": float(t["free/reduced"])}, open("facts.json", "w"))
t.rename_axis("lunch").reset_index(name="mean reading score").to_csv("by_lunch.csv", index=False)
"""
PICTURE = TABLE + '\nimport matplotlib.pyplot as plt\nt.plot.bar(); plt.tight_layout(); plt.savefig("figure.png")\n'
ANSWER = Exploration(
    summary="Standard lunch reads 71.65 on average.",
    step=1,
    facts=["reading_standard"],
    table="by_lunch.csv",
    method="Compare observed group means.",
    findings=["The mean is 71.65 [reading_standard]."],
    limitations=["Descriptive association only."],
    tags=["group-means"],
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    yield
    set_llm(None)


def request(tmp_path, ask="mean reading score by lunch"):
    return ExploreRequest(
        question=ask,
        dataset_version=DatasetInput(id="students:v0", path=CSV, sha256=digest(CSV)),
        context="lunch is standard or free/reduced",
        out_dir=tmp_path / "art",
        request_id="request",
        attempt_id="attempt",
        conversation_id="chat",
    )


def test_whole_csv_is_unchanged_and_answer_has_evidence(tmp_path):
    before = digest(CSV)
    set_llm(FakeLLM([[TABLE]], [ANSWER]))
    trail = AG.explore(request(tmp_path))
    assert trail.record == ANSWER, trail.why
    assert digest(CSV) == before == digest(tmp_path / "art/input/dataset.csv")
    assert trail.sandbox == "docker" and len(trail.steps) == 1


def test_repair_then_actual_image_inspection(tmp_path):
    fake = FakeLLM(
        [["raise ValueError('repair me')", PICTURE, ExploreAction(action="inspect_output", reference="step_2/figure.png")]],
        [ANSWER.model_copy(update={"step": 2, "figure": True})],
    )
    set_llm(fake)
    trail = AG.explore(request(tmp_path))
    assert trail.record is not None, trail.why
    assert [step.ok for step in trail.steps] == [False, True]
    assert any("[image]" in prompt for prompt in fake.humans["ExploreAction"])


def test_needs_input_returns_no_invented_artifact(tmp_path):
    answer = Exploration(
        summary="The requested uncertainty draws were not supplied.", step=0, status="needs_input", needs=["Supply the saved bootstrap draws."]
    )
    set_llm(FakeLLM([[ExploreAction(action="inspect_inputs")]], [answer]))
    trail = AG.explore(request(tmp_path))
    assert trail.record == answer and not trail.steps


def test_numbers_from_question_cannot_be_passed_off_as_results(tmp_path):
    bad = ANSWER.model_copy(update={"summary": "The mean is 999."})
    set_llm(FakeLLM([[TABLE], [], []], [bad, bad, bad]))
    trail = AG.explore(request(tmp_path, "Is the mean 999?"))
    assert trail.record is None and "numbers missing from the selected facts" in trail.why


def test_successful_plot_pixels_are_delivered_without_an_extra_model_action(tmp_path):
    answer = ANSWER.model_copy(update={"figure": True})
    fake = FakeLLM([[PICTURE]], [answer])
    set_llm(fake)
    trail = AG.explore(request(tmp_path))
    assert trail.record == answer and trail.calls == 1
    assert "[image]" in fake.humans["ExploreAction"][-1]
    assert "[image]" in fake.humans["Exploration"][-1]


def test_uninspected_picture_is_rejected(tmp_path, monkeypatch):
    limits = {**AG.checks(), "inspections": 0}
    monkeypatch.setattr(AG, "checks", lambda: limits)
    bad = ANSWER.model_copy(update={"figure": True})
    set_llm(FakeLLM([[PICTURE], [], []], [bad, bad, bad]))
    trail = AG.explore(request(tmp_path))
    assert trail.record is None and "inspect the generated figure" in trail.why


def test_supplied_saved_estimate_can_be_plotted_without_refitting(tmp_path):
    import json

    from causal_agent.explore.contracts import ArtifactInput

    saved = tmp_path / "result.json"
    saved.write_text(json.dumps({"engine": "dowhy", "estimates": [{"value": 2.5, "ci_low": 1.0, "ci_high": 4.0, "n": 120}]}))
    before = saved.read_bytes()
    req = request(tmp_path, "Plot the supplied saved estimate and uncertainty.")
    req.artifact_refs = [ArtifactInput(id="design:1:result", path=saved, sha256=digest(saved))]
    code = """import os, json
from pathlib import Path
import matplotlib.pyplot as plt
r = json.loads((Path(os.environ['INPUTS_DIR']) / 'artifact_1.json').read_text())['estimates'][0]
plt.errorbar([r['value']], [0], xerr=[[r['value']-r['ci_low']], [r['ci_high']-r['value']]], fmt='o')
plt.xlabel('Saved estimate and confidence interval')
plt.tight_layout()
plt.savefig('figure.png')
Path('facts.json').write_text(json.dumps(r))
"""
    answer = Exploration(
        summary="The saved estimate is 2.5.",
        step=1,
        figure=True,
        facts=["value", "ci_low", "ci_high", "n"],
        method="Render the saved estimate and uncertainty without refitting.",
        findings=["The estimate is 2.5 [value]."],
        limitations=["Validity depends on the assumptions in the original saved analysis."],
    )
    set_llm(FakeLLM([[code, ExploreAction(action="inspect_output", reference="step_1/figure.png")]], [answer]))
    trail = AG.explore(req)
    assert trail.record == answer, trail.why
    assert trail.steps[0].facts == {"value": 2.5, "ci_low": 1.0, "ci_high": 4.0, "n": 120}
    assert saved.read_bytes() == before


def test_missing_input_response_cannot_claim_uncomputed_numbers(tmp_path):
    bad = Exploration(summary="The correlation is -0.197 but data are missing.", step=0, status="needs_input", needs=["Supply observations."])
    set_llm(FakeLLM([[ExploreAction(action="inspect_inputs")], [], []], [bad, bad, bad]))
    trail = AG.explore(request(tmp_path))
    assert trail.record is None and "uncomputed numerical findings" in trail.why
