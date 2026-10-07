"""The documentation's pictures come from the checked-in records, the same way twice, and say what the records say."""

from __future__ import annotations

import json

from causal_agent.evals import docfigs as D

TH = D.THEMES["light"]


def test_every_picture_is_written_twice_the_same(tmp_path):
    first = {p.name: p.read_text() for p in D.write_all(tmp_path / "a")}
    second = {p.name: p.read_text() for p in D.write_all(tmp_path / "b")}
    assert set(first) == {
        f"{stem}-{theme}.svg"
        for stem in ("flow", "lane-swimlane", "matrix-diff", "judgement-relate", "desk-graph", "draw-sequence", "decisions")
        for theme in ("light", "dark")
    }
    assert first == second
    assert all(s.startswith("<svg ") and s.rstrip().endswith("</svg>") for s in first.values())


def test_the_matrix_diff_undoes_the_journals_fit_steps_back_to_the_story():
    matrix = json.loads((D.DEMO / "interview" / "matrix.json").read_text())
    journal = [json.loads(line) for line in (D.DEMO / "interview" / "journal.jsonl").read_text().splitlines() if line.strip()]
    before = D._before_grid(matrix, journal, 35)
    after = {fam: {kind: c["value"] for kind, c in row.items()} for fam, row in matrix["cells"].items()}
    # a cell the record says was set after v35 is the one the journal moved; the rest stand
    for fam, row in matrix["cells"].items():
        for kind, c in row.items():
            moved = int(c["at"][1:]) > 35
            assert (before[fam][kind] != after[fam][kind]) == moved, (fam, kind)
    struck_before, surviving_before, ready_before = D._verdict(before)
    struck_after, surviving_after, ready_after = D._verdict(after)
    assert "instrument" in surviving_before and "instrument" in struck_after
    assert surviving_after == ["adjustment"] and ready_after and not ready_before
    svg = D.matrix_diff(D.DEMO / "interview" / "matrix.json", D.DEMO / "interview" / "journal.jsonl", 35, TH)
    assert "instrument.exclusion: unknown → does_not_fit" in svg and "ready · in play: adjustment" in svg


def test_the_opened_judgement_shows_the_real_prompt_answer_and_a_passing_gate():
    svg = D.opened_judgement(D.DEMO / "run", "gender", TH)
    assert "THE COLUMNS TO PLACE" in svg and "[col:gender.when]" in svg
    assert "affects_outcome" in svg and "cites  col:gender.when" in svg
    assert "rejected" not in svg and "passed: the graph takes it" in svg
