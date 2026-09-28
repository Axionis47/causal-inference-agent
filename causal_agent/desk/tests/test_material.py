"""The after-run material: with a memory and a pack in the design dir, the outcome's and the treatment's scale are lines with
numbers, so "is 5.7 points a lot" can be answered against the outcome's range."""

from __future__ import annotations

from causal_agent.common.contracts import RunRecord
from causal_agent.desk import material as M
from causal_agent.desk.handoff import forced
from causal_agent.memory import store


def _run(design_dir) -> RunRecord:
    return RunRecord(index=1, dataset="students3", question="q", family="adjustment", specialist="dowhy", status="done", design_dir=str(design_dir))


def test_the_outcome_and_the_treatment_scale_come_from_the_pack_with_their_numbers(tmp_path):
    m = store.migrate("students3", write=False)
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", ["lunch"], memory=m)
    h.column("math score").facts.bounds = ["0", "100"]
    (tmp_path / "handoff.json").write_text(h.model_dump_json(indent=2))
    mat = M.render(_run(tmp_path), m)
    assert mat.by_address["col:math_score.profile.bounds"] == "0 to 100"
    assert mat.numbers["col:math_score.profile.bounds.low"] == 0.0 and mat.numbers["col:math_score.profile.bounds.high"] == 100.0
    line = mat.by_address["col:math_score.profile.numeric"]
    assert line.startswith("min 0, p50 66, max 100, mean 66.089")
    assert mat.numbers["col:math_score.profile.numeric.max"] == 100.0 and abs(mat.numbers["col:math_score.profile.numeric.mean"] - 66.089) < 1e-6
    assert "col:math_score.profile.numeric" in mat.addresses and "col:math_score.profile.numeric.p50" in mat.addresses
    assert not [a for a in mat.addresses if a.startswith("col:test_preparation_course.profile.")]  # a level column has no scale


def test_without_a_memory_or_a_pack_no_scale_line_is_written(tmp_path):
    m = store.migrate("students3", write=False)
    assert not [a for a in M.render(_run(tmp_path), m).addresses if ".profile." in a]  # no pack in the design dir
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", ["lunch"], memory=m)
    (tmp_path / "handoff.json").write_text(h.model_dump_json(indent=2))
    assert not [a for a in M.render(_run(tmp_path)).addresses if ".profile." in a]  # no memory given
