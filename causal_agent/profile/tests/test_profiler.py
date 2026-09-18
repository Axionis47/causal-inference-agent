from pathlib import Path

import pandas as pd

from causal_agent.profile.profiler import profile

ROOT = Path(__file__).resolve().parents[3]
STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


def test_students_profile_shape():
    p = profile(STUDENTS)
    assert p.dataset.rows == 1000
    assert p.dataset.columns == 8
    assert p.dataset.duplicate_rows == 0
    by = {c.name: c for c in p.columns}
    assert by["math score"].kind == "numeric"
    assert by["test preparation course"].kind == "categorical"
    assert by["test preparation course"].distinct == 2
    assert by["lunch"].top_values is not None
    assert all(c.nulls == 0 for c in p.columns)


def test_deterministic():
    a = profile(STUDENTS).model_dump()
    b = profile(STUDENTS).model_dump()
    assert a == b


def test_panel_varies_over_and_switch(tmp_path):
    rows = []
    for store in (1, 2, 3):
        for week in range(6):
            day = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * week)).date()
            rows.append({"store": store, "week": str(day), "on": int(store == 1 and week >= 3), "area": store * 100, "sales": 10 + week + store, "same": 10 + week})
    f = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    p = profile(f, entity_columns=["store"], time_column="week")
    by = {c.name: c for c in p.columns}
    assert by["area"].varies_over == "entity"
    assert by["on"].varies_over == "both"
    assert by["on"].switch is not None
    assert by["on"].switch.entities_never_on == 2
    assert by["on"].switch.first_on == "2024-01-22"
    assert by["sales"].varies_over == "both"
    assert by["same"].varies_over == "time"
    assert p.dataset.grain == ["store", "week"]
    assert p.dataset.time_coverage is not None and p.dataset.time_coverage.inferred_frequency == "weekly"


def test_sentinels_and_whitespace(tmp_path):
    # note: pandas reads "N/A" as null on load, so it shows as a null, not a sentinel; "?" survives as text
    df = pd.DataFrame({" status": [" Approved", "Rejected ", " Approved", "Rejected", "?"], "score": [700, -1, 650, 620, -1], "x": range(5)})
    f = tmp_path / "messy.csv"
    df.to_csv(f, index=False)
    p = profile(f)
    by = {c.name: c for c in p.columns}
    assert "values have leading or trailing whitespace" in by[" status"].format_issues
    assert "header has leading or trailing whitespace" in by[" status"].format_issues
    assert any(s.value == "-1" and s.count == 2 for s in by["score"].observed_sentinels)
    assert any("?" in s.value for s in by[" status"].observed_sentinels)
