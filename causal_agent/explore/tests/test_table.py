"""The agent as the data agent: asked for a table, it writes a new version of the whole file, every row with the id it was
given; the client re-joins the column held back by row id, keeps the file's column order, and saves the version beside its
parent; a table that loses the row id, invents a row or drops a column of the pair is refused with the reason."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causal_agent.common import config
from causal_agent.common.llm import set_llm
from causal_agent.explore import agent as AG
from causal_agent.explore import client as XC
from causal_agent.explore import project as PJ
from causal_agent.explore import server as S
from causal_agent.explore.contracts import Exploration, TransformRequest
from causal_agent.explore.tests.fakes import FakeLLM
from causal_agent.memory import versions as VS

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
COLUMNS = {"lunch": "lunch", "reading_score": "reading score", "math_score": "math score", "test_preparation_course": "test preparation course"}

NEW_TABLE = """import os, json, pandas as pd
df = pd.read_csv(os.environ["DATA_CSV"])
df["high reading"] = (df["reading score"] >= 70).astype(int)
df = df[df["lunch"] == "standard"]
df.to_csv("table.csv", index=False)
json.dump({"rows": int(len(df)), "columns": int(df.shape[1])}, open("facts.json", "w"))
print(len(df))
"""
NO_ROW = NEW_TABLE.replace('df.to_csv("table.csv", index=False)', 'df.drop(columns=["__row"]).to_csv("table.csv", index=False)')
ADDED_ROW = NEW_TABLE.replace('df.to_csv("table.csv", index=False)', 'pd.concat([df, df.iloc[:1].assign(__row=5000)]).to_csv("table.csv", index=False)')
DROPPED_PAIR = NEW_TABLE.replace('df.to_csv("table.csv", index=False)', 'df.drop(columns=["test preparation course"]).to_csv("table.csv", index=False)')
ANSWER = Exploration(summary="Kept the standard-lunch rows and added high reading.", step=1, facts=["rows"], table="table.csv")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    XC.set_app(S.app("http://explore/"))
    yield tmp_path
    XC.set_app(None)
    set_llm(None)


def test_the_projection_puts_the_row_id_first(tmp_path):
    path = PJ.project(CSV, tmp_path / "v", drop="math score", row_id=True)
    df = pd.read_csv(path)
    assert list(df.columns)[0] == PJ.ROW and list(df[PJ.ROW]) == list(range(1000)) and "math score" not in df.columns
    whole = pd.read_csv(PJ.project(CSV, tmp_path / "w", drop=None, row_id=True))
    assert whole.shape == (1000, 9) and "math score" in whole.columns


def test_a_new_version_lands_beside_its_parent_with_the_held_back_column_re_joined(tmp_path):
    set_llm(FakeLLM([[NEW_TABLE], []], [ANSWER]))
    v, d, thoughts = XC.transform(
        dataset="students",
        ask="keep only the standard lunch rows and add a column saying whether reading score is 70 or more",
        context="[col:lunch.note] standard or free/reduced",
        columns=COLUMNS,
        csv=CSV,
        outcome="math score",
        treatment="test preparation course",
        frozen=False,
        memory_version=4,
        data_version=0,
    )
    assert d is None and v is not None and v.v == 1 and v.parent == 0 and v.address == "version:1" and v.summary == ANSWER.summary
    folder = tmp_path / "data" / "memory" / "students" / "versions" / "1"
    assert folder == VS.folder("students", 1) and {"table.csv", "code.py", "version.json", "request.json", "thoughts.json"} <= {
        p.name for p in folder.iterdir()
    }
    assert not (folder / "input").exists()
    new, orig = pd.read_csv(folder / "table.csv"), pd.read_csv(CSV)
    assert PJ.ROW not in new.columns and list(new.columns) == list(orig.columns) + ["high reading"]
    assert v.rows == len(new) == int((orig["lunch"] == "standard").sum()) and v.added == ["high reading"] and v.dropped == []
    kept = orig[orig["lunch"] == "standard"].reset_index(drop=True)
    assert new["math score"].tolist() == kept["math score"].tolist(), "the outcome, held back from the agent, is back on every kept row"
    assert VS.load("students", 1) == v and VS.next_version("students") == 2
    assert "math score" not in (folder / "code.py").read_text()


def test_a_table_that_loses_the_row_id_invents_a_row_or_drops_the_pair_is_refused(tmp_path):
    out = tmp_path / "v"
    path = PJ.project(CSV, out, drop="math score", row_id=True)
    req = TransformRequest(
        dataset="students",
        ask="keep only the standard lunch rows",
        context="",
        columns={k: n for k, n in COLUMNS.items() if n != "math score"},
        csv=path,
        out_dir=out,
        outcome="math score",
        treatment="test preparation course",
        dropped="outcome",
        frozen=False,
        product="table",
        rows=1000,
        keep=["test preparation course"],
    )
    fake = FakeLLM(
        [[NO_ROW], [ADDED_ROW], [DROPPED_PAIR]],
        [ANSWER.model_copy(update={"step": 1}), ANSWER.model_copy(update={"step": 2}), ANSWER.model_copy(update={"step": 3})],
    )
    set_llm(fake)
    trail = AG.transform(req)
    assert trail.record is None and trail.calls == 3
    answers = fake.humans["Exploration"]
    assert "has no __row column" in answers[1] and "outside 0 to 999" in answers[2] and "must keep 'test preparation course'" in trail.why


def test_a_number_the_ask_itself_states_is_grounded(tmp_path):
    out = tmp_path / "v"
    path = PJ.project(CSV, out, drop="math score", row_id=True)
    req = TransformRequest(
        dataset="students",
        ask="add a column saying whether reading score is 70 or more",
        context="",
        columns={k: n for k, n in COLUMNS.items() if n != "math score"},
        csv=path,
        out_dir=out,
        outcome="math score",
        treatment="test preparation course",
        dropped="outcome",
        frozen=False,
        product="table",
        rows=1000,
        keep=["test preparation course"],
    )
    set_llm(FakeLLM([[NEW_TABLE], []], [ANSWER.model_copy(update={"summary": "Added high reading, 1 when reading score is 70 or more."})]))
    trail = AG.transform(req)
    assert trail.record is not None and trail.why == ""
