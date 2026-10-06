"""The artifacts shelf on disk under a temp root: one folder per artifact under the dataset, the addresses, the listing by
kind and by design."""

from __future__ import annotations

import pytest

from causal_agent.common import config
from causal_agent.memory import artifacts as A
from causal_agent.memory.artifacts import Artifact


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def _artifact(**over) -> Artifact:
    base = dict(
        id=A.new_id(),
        dataset="students",
        kind="picture",
        design=None,
        memory_version=3,
        data_version=0,
        ask="mean math score by lunch",
        summary="Students on standard lunch score higher in math on average.",
        facts={"mean_standard": 70.0341, "mean_free_reduced": 58.9211},
        files=["artifact.json", "code.py", "facts.json", "figure.png", "request.json"],
        figure=True,
        made_at="2026-09-28T10:00:00+00:00",
    )
    return Artifact(**{**base, **over})


def test_folders_hang_off_the_dataset_whatever_the_artifact_is_about(tmp_path):
    assert A.folder("students", "abcd1234") == tmp_path / "data" / "memory" / "students" / "artifacts" / "abcd1234"


def test_new_ids_are_short_and_distinct():
    ids = {A.new_id() for _ in range(50)}
    assert len(ids) == 50 and all(len(i) == 8 and i.isalnum() for i in ids)


def test_addresses_and_render():
    a = _artifact(id="abcd1234")
    assert a.address == "artifact:abcd1234"
    assert a.addresses() == {"artifact:abcd1234", "artifact:abcd1234.mean_standard", "artifact:abcd1234.mean_free_reduced", "artifact:abcd1234.figure"}
    lines = a.render().splitlines()
    assert lines[0] == "[artifact:abcd1234] Students on standard lunch score higher in math on average. (picture; before any run)"
    assert lines[1] == "  [artifact:abcd1234.mean_standard] mean_standard: 70.03"
    assert lines[-1] == "  [artifact:abcd1234.figure] a picture"
    t = _artifact(id="ef015678", kind="exploration", design=2, figure=False, table="by_lunch.csv")
    assert (
        "(exploration; design 2)" in t.render().splitlines()[0]
        and "artifact:ef015678.table" in t.addresses()
        and "artifact:ef015678.figure" not in t.addresses()
    )


def test_save_load_and_list_by_kind_by_design_sorted_by_time():
    late = _artifact(made_at="2026-09-28T11:00:00+00:00")
    early = _artifact(made_at="2026-09-28T09:00:00+00:00")
    d = A.save(late)
    A.save(early)
    assert (d / "artifact.json").is_file()
    assert A.load("students", late.id) == late
    assert [a.id for a in A.list_artifacts("students")] == [early.id, late.id]
    assert A.list_artifacts("students", design=1) == [] and A.list_artifacts("students", kind="exploration") == []
    p = _artifact(kind="exploration", design=1, figure=False, table="t.csv", made_at="2026-09-28T12:00:00+00:00")
    A.save(p)
    assert [a.id for a in A.list_artifacts("students", design=1)] == [p.id]
    assert [a.id for a in A.list_artifacts("students", kind="exploration")] == [p.id]
    assert [a.id for a in A.about("students", None)] == [early.id, late.id]  # before any run: nothing about a design
    assert [a.id for a in A.about("students", 1)] == [early.id, late.id, p.id]  # after run 1: the data's and run 1's
    assert [a.id for a in A.about("students", 2)] == [early.id, late.id]
    assert A.list_artifacts("nobody") == []
