"""Artifacts on disk under a temp root: the folder per moment, the addresses, the listing."""

from __future__ import annotations

import pytest

from causal_agent.common import config
from causal_agent.viz import store
from causal_agent.viz.store import Artifact


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def _artifact(**over) -> Artifact:
    base = dict(
        id=store.new_id(),
        dataset="students",
        moment="pre",
        design=None,
        memory_version=3,
        ask="mean math score by lunch",
        caption="Students on standard lunch score higher in math on average.",
        facts={"mean_standard": 70.0341, "mean_free_reduced": 58.9211},
        made_at="2026-09-28T10:00:00+00:00",
        files=["artifact.json", "code.py", "facts.json", "figure.png", "request.json"],
    )
    return Artifact(**{**base, **over})


def test_folders_hang_off_the_memory_and_the_design(tmp_path):
    pre = store.folder("students", "pre", None, "abcd1234")
    post = store.folder("students", "post", 2, "abcd1234")
    assert pre == tmp_path / "data" / "memory" / "students" / "viz" / "pre" / "abcd1234"
    assert post == tmp_path / "data" / "memory" / "students" / "designs" / "2" / "viz" / "abcd1234"
    with pytest.raises(ValueError):
        store.folder("students", "post", None, "abcd1234")


def test_new_ids_are_short_and_distinct():
    ids = {store.new_id() for _ in range(50)}
    assert len(ids) == 50 and all(len(i) == 8 and i.isalnum() for i in ids)


def test_addresses_and_render():
    a = _artifact(id="abcd1234")
    assert a.address == "artifact:abcd1234"
    assert a.addresses() == {"artifact:abcd1234", "artifact:abcd1234.mean_standard", "artifact:abcd1234.mean_free_reduced"}
    lines = a.render().splitlines()
    assert lines[0] == "[artifact:abcd1234] Students on standard lunch score higher in math on average. (before the run)"
    assert lines[1] == "  [artifact:abcd1234.mean_standard] mean_standard: 70.03"
    assert "(design 2)" in _artifact(moment="post", design=2).render().splitlines()[0]


def test_save_load_and_list_sorted_by_time():
    late = _artifact(made_at="2026-09-28T11:00:00+00:00")
    early = _artifact(made_at="2026-09-28T09:00:00+00:00")
    d = store.save(late)
    store.save(early)
    assert (d / "artifact.json").is_file()
    assert store.load("students", "pre", None, late.id) == late
    assert [a.id for a in store.list_artifacts("students", "pre")] == [early.id, late.id]
    assert store.list_artifacts("students", "post", 1) == []
    p = _artifact(moment="post", design=1)
    store.save(p)
    assert [a.id for a in store.list_artifacts("students", "post", 1)] == [p.id]
    assert [a.id for a in store.list_artifacts("students", "pre")] == [early.id, late.id]
