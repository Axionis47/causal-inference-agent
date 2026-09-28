"""The matrix as a record: each cell with what set it and when, a diff between two states, and lines the chat can cite."""

from __future__ import annotations

import json

import pandas as pd

from causal_agent.families import registry as R
from causal_agent.memory import ops
from causal_agent.memory.matrix import CellChange, Matrix
from causal_agent.memory.records import Memory
from causal_agent.memory.store import load_claims
from causal_agent.profile.datasets import ROOT, dataset_entries
from causal_agent.profile.profiler import Profile

STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


def _students3() -> tuple[Memory, list]:
    e = dataset_entries()["students3"]
    prof = Profile.model_validate(json.loads((ROOT / e["profile"]).read_text()))
    table, _ = load_claims(ROOT / e["claims"])
    m = Memory.from_claims("students3", table, profile=prof, csv=e["csv"])
    return m, ops.probe(m, pd.read_csv(STUDENTS), R.REGISTRY.values())


def test_update_fills_every_cell_with_what_set_it_and_the_version():
    m, probes = _students3()
    mx = Matrix().update(m, probes, R.needs())
    assert mx.memory_version == m.version and mx.ready and "adjustment" in mx.surviving
    # a fits cell decided by a `fits` field carries that field; one with no such field carries the claim itself
    kind = mx.cell("adjustment", "assignment")
    assert kind is not None and kind.value == "fits" and kind.set_by == "claim:assignment.kind" and kind.at == f"v{m.version}"
    grain = mx.cell("adjustment", "grain")
    assert grain is not None and grain.value == "fits" and grain.set_by == "claim:grain"
    # a per-column cell names the columns that decided it
    measured = mx.cell("adjustment", "measured")
    assert measured is not None and measured.value == "fits" and "col:lunch" in (measured.set_by or "")
    # a kind the family does not need is not needed, and set by nothing
    assert all(c.value == "not_needed" and c.set_by is None for fam, row in mx.cells.items() for k, c in row.items() if k not in R.needs()[fam].requires)


def test_a_struck_cell_names_the_field_that_failed_and_the_family_is_out():
    m, probes = _students3()
    mx = Matrix().update(m, probes, R.needs())
    struck = mx.cell("discontinuity", "assignment")
    assert struck is not None and struck.value == "does_not_fit" and struck.set_by == "claim:assignment.kind"
    assert mx.struck["discontinuity"] == "assignment does not fit" and "discontinuity" not in mx.surviving


def test_diff_lists_only_the_cells_that_moved_and_a_moved_cell_takes_the_new_version():
    m, probes = _students3()
    first = Matrix().update(m, probes, R.needs())
    v_first = m.version
    same = first.update(m, probes, R.needs())
    assert same.diff(first) == [] and same.cell("adjustment", "assignment").at == first.cell("adjustment", "assignment").at
    m.set("claim:unobserved.exists", None, status="empty", source=None)  # the belief goes vague again
    second = same.update(m, probes, R.needs())
    changed = second.diff(same)
    adjustment = CellChange(family="adjustment", kind="unobserved", before="fits", after="unknown", set_by="claim:unobserved")
    assert adjustment in changed and all(c.kind == "unobserved" and (c.before, c.after) == ("fits", "unknown") for c in changed)  # every family that needs it
    assert adjustment.line() == "adjustment.unobserved: fits -> unknown (claim:unobserved)"
    assert second.cell("adjustment", "unobserved").at == f"v{m.version}" and m.version > v_first
    assert second.cell("adjustment", "assignment").at == f"v{v_first}"  # untouched: keeps the version it was set at
    assert not second.ready
    # against the empty record every cell in play moved; a cell not needed never counts as a move
    from_nothing = first.diff(Matrix())
    assert all(c.before is None and c.after != "not_needed" for c in from_nothing) and len(from_nothing) > 5


def test_render_gives_one_addressed_line_per_cell_in_play_then_the_struck_and_the_verdict():
    m, probes = _students3()
    lines = Matrix().update(m, probes, R.needs()).render()
    assert f"[matrix:adjustment.assignment] fits · set by claim:assignment.kind · v{m.version}" in lines
    assert "[matrix:discontinuity.struck] assignment does not fit" in lines
    assert not any(" not_needed" in ln for ln in lines)
    assert lines[-1].startswith("[matrix] ready · in play: ") and lines[-1].endswith(f"v{m.version}")
