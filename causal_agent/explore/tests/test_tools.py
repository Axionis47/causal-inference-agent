"""The one tool: each script runs in its own step folder on the table given; prose is refused without a run; a script naming the
held-back column is refused; the result text carries what was printed, the files and the facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_agent.explore.tools import CodeTools
from causal_agent.lane.tools import Refused

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"

COUNTS = """import os, json, pandas as pd
df = pd.read_csv(os.environ["DATA_CSV"])
counts = df["lunch"].value_counts()
print(counts.to_string())
json.dump({"n_standard": int(counts["standard"]), "n_free_reduced": int(counts["free/reduced"])}, open("facts.json", "w"))
counts.rename_axis("lunch").reset_index(name="n").to_csv("by_lunch.csv", index=False)
"""


def _tools(tmp_path, dropped=None) -> CodeTools:
    return CodeTools(csv=CSV, out_dir=tmp_path / "art", timeout=60, stdout_tail=4000, dropped_name=dropped)


def test_each_script_runs_in_its_own_step_folder_and_the_text_carries_what_came_back(tmp_path, monkeypatch):
    t = _tools(tmp_path)
    r = t.call("run_code", {"code": COUNTS})
    assert "step 1 ran (docker)." in r.text and "standard" in r.text and "files: by_lunch.csv, code.py, facts.json" in r.text
    assert "facts.json: n_standard=645, n_free_reduced=355" in r.text
    assert (tmp_path / "art" / "step_1" / "stdout.txt").read_text().startswith("lunch")
    r2 = t.call("run_code", {"code": "print('again')"})
    assert "step 2 ran" in r2.text and "again" in r2.text and "facts.json" not in r2.text.split("files:")[1]
    assert [s.n for s in t.steps] == [1, 2] and t.step(1).facts == {"n_standard": 645.0, "n_free_reduced": 355.0} and t.step(2).facts is None
    assert t.step(3) is None and (tmp_path / "art" / "step_2").is_dir()


def test_a_failing_script_is_a_failed_step_with_its_error(tmp_path, monkeypatch):
    t = _tools(tmp_path)
    r = t.call("run_code", {"code": "import pandas as pd\nraise ValueError('boom: no such column')\n"})
    assert "step 1 failed" in r.text and "boom: no such column" in r.text and not t.step(1).ok


def test_code_outputs_cannot_make_the_host_read_linked_files(tmp_path, monkeypatch):
    outside = tmp_path / "outside.json"
    outside.write_text('{"withheld_outcome": 123}')
    t = _tools(tmp_path)
    t.call("run_code", {"code": f"import os\nos.symlink({str(outside)!r}, 'facts.json')\nos.link({str(outside)!r}, 'linked.json')\n"})
    assert not t.step(1).ok and t.step(1).facts is None
    assert t.step(1).files == ["code.py", "stdout.txt"]
    assert outside.read_text() == '{"withheld_outcome": 123}' and outside.stat().st_nlink == 1


def test_prose_and_the_held_back_column_are_refused_without_a_run(tmp_path):
    t = _tools(tmp_path, dropped="math score")
    with pytest.raises(Refused, match="only the Python script"):
        t.call("run_code", {"code": "Here is the script:\nimport pandas"})
    with pytest.raises(Refused, match="held back until the design has run"):
        t.call("run_code", {"code": "import pandas as pd\nprint(df['math score'].mean())\n"})
    with pytest.raises(Refused, match="no tool named"):
        t.call("describe", {"column": "lunch"})
    with pytest.raises(Refused, match="could not run with"):
        t.call("run_code", {"script": "x"})
    assert t.steps == [] and not (tmp_path / "art").exists()
