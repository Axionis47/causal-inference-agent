"""Estimator selection executes diagnostics on a source-preserving, outcome-free projection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from causal_agent.common import trace
from causal_agent.common.addresses import key
from causal_agent.families.adjustment.lane.selection import diagnostic_tools
from causal_agent.lane import episode

ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


@pytest.fixture
def selection(tmp_path):
    source_bytes = CSV.read_bytes()
    table = pd.read_csv(CSV).rename(columns=key)
    table_path = tmp_path / "loaded.csv"
    table.to_csv(table_path, index=False)
    state = {
        "handoff": SimpleNamespace(treatment="test preparation course", outcome="math score"),
        "table_path": str(table_path),
        "run_dir": str(tmp_path / "run"),
    }
    candidates = [
        SimpleNamespace(estimand=SimpleNamespace(adjustment_set=["lunch"])),
        SimpleNamespace(estimand=SimpleNamespace(adjustment_set=["parental_level_of_education", "lunch"])),
    ]
    return SimpleNamespace(state=state, candidates=candidates, table=table, source_bytes=source_bytes)


def test_diagnostics_use_only_the_admitted_treatment_and_adjustment_columns(selection):
    tools = diagnostic_tools(selection.state, selection.candidates)
    projected = pd.read_csv(tools.csv)
    expected = ["test_preparation_course", "lunch", "parental_level_of_education"]
    pd.testing.assert_frame_equal(projected, selection.table[expected])
    assert len(projected) == 1000
    assert tools.dropped_name == "math_score"
    assert not {"math_score", "reading_score", "writing_score", "gender", "race_ethnicity"}.intersection(projected.columns)
    assert CSV.read_bytes() == selection.source_bytes
    pd.testing.assert_frame_equal(pd.read_csv(selection.state["table_path"]), selection.table)


def test_real_selection_code_has_addressed_output_and_cannot_read_the_full_table(selection, monkeypatch):
    tools = diagnostic_tools(selection.state, selection.candidates)
    projection_bytes = tools.csv.read_bytes()
    loaded_bytes = Path(selection.state["table_path"]).read_bytes()
    written = set()
    original_write = trace.write

    def capture_write(path, content, **kwargs):
        written.add(Path(path))
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(trace, "write", capture_write)
    script = f"""import os, json
from pathlib import Path
import pandas as pd
table = pd.read_csv(os.environ['DATA_CSV'])
try:
    Path({selection.state["table_path"]!r}).read_text()
except OSError:
    print('full table read blocked')
else:
    raise AssertionError('the full analysis table was readable')
counts = table['test_preparation_course'].value_counts()
print(counts.to_string())
counts.rename_axis('arm').reset_index(name='rows').to_csv('arms.csv', index=False)
Path('facts.json').write_text(json.dumps({{'rows': len(table), 'completed': int(counts['completed'])}}))
"""
    log = episode.EpisodeLog(node="pick_estimator", visit="selection-test")
    response = episode._run(tools, "run_code", {"code": script}, 2, log)
    step = tools.steps[0]
    assert step.ok, step.stderr
    address = "probe:pick_estimator.selection-test.1"
    assert response.startswith(f"[{address}] run_code(")
    assert "full table read blocked" in response and "completed" in response
    expected = {"rows": 1000.0, "completed": float((selection.table.test_preparation_course == "completed").sum())}
    assert step.facts == expected and json.loads((tools.out_dir / "step_1/facts.json").read_text()) == expected
    assert log.resolve(address) and log.facts[0].tool == "run_code"
    assert tools.evidence_errors([address], log) == []
    assert {tools.out_dir / "step_1" / name for name in ("code.py", "stdout.txt", "facts.json", "arms.csv")} <= written
    assert tools.csv.read_bytes() == projection_bytes
    assert Path(selection.state["table_path"]).read_bytes() == loaded_bytes
    assert CSV.read_bytes() == selection.source_bytes
