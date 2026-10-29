"""Arbitrary DiD diagnostics receive only admitted metadata and globally pre-adoption values."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causal_agent.common.contracts import ColumnBrief, Handoff, Provenance
from causal_agent.explore.diagnostics import tools_for
from causal_agent.families.diff_in_diff.design import DidDesign
from causal_agent.families.diff_in_diff.lane.contracts import Groups, Periods, ProbeReading
from causal_agent.families.diff_in_diff.lane.knowledge import ENDPOINTS
from causal_agent.families.diff_in_diff.lane.reason import nodes
from causal_agent.families.diff_in_diff.lane.reason.diagnostics import diagnostic_table, record_errors
from causal_agent.families.diff_in_diff.lane.reason.ledger import commit
from causal_agent.families.diff_in_diff.lane.shape import canonical
from causal_agent.families.diff_in_diff.tests.test_policy import panel
from causal_agent.lane import episode
from causal_agent.lane.endpoints import admit, validate_context
from causal_agent.lane.ledger import Entry, Ledger


def context() -> Handoff:
    confirmed = Provenance(status="confirmed", source="user:turn:1", said="Recorded before treatment")
    return Handoff.model_construct(
        outcome="y",
        treatment="treated",
        design=DidDesign(before_column="earlier_outcome", after_column="later_outcome"),
        columns=[
            ColumnBrief(name="baseline", key="baseline", when="before", provenance={"when": confirmed}),
            ColumnBrief(name="unaffected", key="unaffected", when="at", moved_by_change=False, provenance={"when": confirmed, "moved_by_change": confirmed}),
            ColumnBrief(name="unknown", key="unknown", when="unknown"),
            ColumnBrief(name="draft", key="draft", when="before", provenance={"when": Provenance(status="proposed")}),
            ColumnBrief(name="affected", key="affected", when="before", moved_by_change=True, provenance={"when": confirmed}),
            ColumnBrief(name="proxy", key="proxy", when="before", measures_outcome=True, provenance={"when": confirmed}),
            ColumnBrief(name="after", key="after", when="after", moved_by_change=False, provenance={"when": confirmed, "moved_by_change": confirmed}),
            ColumnBrief(name="other_outcome", key="other_outcome", role="outcome", when="before", provenance={"when": confirmed}),
            ColumnBrief(name="earlier_outcome", key="earlier_outcome", when="before", provenance={"when": confirmed}),
            ColumnBrief(name="later_outcome", key="later_outcome", when="before", provenance={"when": confirmed}),
        ],
    )


def input_panel() -> pd.DataFrame:
    p = panel(cohorts=(4, 7), units=2)
    for i, card in enumerate(context().columns):
        p[card.key] = p.index + i / 10
    # Leave a real metadata gap; projection must preserve it, not complete or drop rows.
    return p.loc[~((p.unit == "0") & (p.time_index == 2))].copy()


def test_projection_preserves_metadata_and_hides_all_later_measurements():
    p = input_panel()
    projected = diagnostic_table(context(), p)
    metadata = ["unit", "time", "time_index", "treated", "post", "treat", "rel_time", "cohort", "cluster", "row_id"]
    assert list(projected) == [*metadata, "y", "baseline", "unaffected"]
    pd.testing.assert_frame_equal(projected[metadata], p[metadata])
    values = ["y", "baseline", "unaffected"]
    pd.testing.assert_frame_equal(projected.loc[p.post == 0, values], p.loc[p.post == 0, values])
    assert projected.loc[p.post != 0, values].isna().all().all()
    assert projected.loc[(p.cohort == 7) & p.time_index.between(4, 6), "y"].isna().all()
    assert projected.loc[(p.cohort == 0) & (p.post == 1), "y"].isna().all()
    assert len(projected) == len(p)


def test_projection_bytes_are_invariant_to_post_outcomes_covariates_and_missingness():
    p = input_panel()
    expected = diagnostic_table(context(), p).to_csv(index=False)
    changed = p.copy()
    for column in ["y", "baseline", "unaffected", "proxy", "later_outcome"]:
        changed.loc[p.post == 1, column] = 9e20
    assert diagnostic_table(context(), changed).to_csv(index=False) == expected
    changed.loc[p.post == 1, ["y", "baseline", "unaffected", "proxy", "later_outcome"]] = None
    assert diagnostic_table(context(), changed).to_csv(index=False) == expected


@pytest.mark.parametrize("role", ["unit", "time", "assignment", "adoption", "cluster"])
def test_outcome_proxies_cannot_supply_metadata(role):
    h = context()
    updates = {"unit": "proxy"} if role == "unit" else {"time": "proxy"} if role == "time" else {"cluster_level": "proxy"} if role == "cluster" else {}
    if role in {"assignment", "adoption"}:
        updates["treated_group"] = {"column" if role == "assignment" else "cohort_column": "proxy"}
    h.design = h.design.model_copy(update=updates)
    with pytest.raises(ValueError, match="metadata cannot expose an outcome"):
        diagnostic_table(h, input_panel())


def test_another_outcome_role_cannot_supply_metadata_even_without_measurement_flag():
    h = context()
    h.design = h.design.model_copy(update={"cluster_level": "other_outcome"})
    with pytest.raises(ValueError, match="metadata cannot expose an outcome"):
        diagnostic_table(h, input_panel())


@pytest.mark.parametrize("role", ["unit", "time", "cluster_level"])
def test_treatment_affected_identities_cannot_supply_metadata(role):
    h = context()
    h.design = h.design.model_copy(update={role: "affected"})
    with pytest.raises(ValueError, match="identity cannot be treatment-affected"):
        diagnostic_table(h, input_panel())


@pytest.mark.parametrize("column", ["y", "unit", "time", "time_index", "treated", "post", "treat", "rel_time", "cohort", "cluster", "row_id"])
def test_reserved_candidate_cannot_overwrite_diagnostic_metadata(column):
    h = context()
    h.columns.append(ColumnBrief(name=column, key=column, when="after", measures_outcome=True))
    with pytest.raises(ValueError, match="candidate collides with canonical diagnostic metadata"):
        diagnostic_table(h, input_panel())


def test_declared_long_unit_time_and_cluster_sources_allow_candidate_briefs():
    raw = pd.DataFrame(
        [{"unit": f"u{u}", "cluster": f"c{u}", "time": t, "group": "yes" if u < 2 else "no", "outcome": float(u + t)} for u in range(4) for t in (1, 2, 3)]
    )
    groups = Groups(column="group", treated_level="yes", reason="declared assignment", cites=[])
    periods = Periods(kind="long", time_column="time", first_post="3", reason="declared time", cites=[])
    p, _ = canonical(raw, groups, periods, "outcome", ["unit", "time", "cluster"], unit_column="unit", cluster_column="cluster")
    h = Handoff.model_construct(
        outcome="outcome",
        treatment="group",
        design=DidDesign(unit="unit", time="time", cluster_level="cluster", treated_group={"column": "group", "level": "yes"}),
        columns=[ColumnBrief(name=c, key=c, role="candidate") for c in ("unit", "time", "cluster")],
    )
    projected = diagnostic_table(h, p)
    pd.testing.assert_frame_equal(projected[["unit", "time", "cluster"]], p[["unit", "time", "cluster"]])
    assert projected.unit.tolist() == raw.unit.tolist() and projected.cluster.tolist() == raw.cluster.tolist()
    assert projected.loc[p.post == 1, "y"].isna().all()


@pytest.mark.parametrize("cluster_source", ["later_outcome", "state"])
def test_actual_wide_outcome_cluster_alias_returns_deficiency_without_writing_projection(tmp_path, cluster_source):
    raw = pd.DataFrame(
        {"group": ["yes", "yes", "no", "no"], "state": [1, 1, 2, 2], "earlier_outcome": [1.0, 2.0, 1.0, 2.0], "later_outcome": [8.0, 7.0, 3.0, 4.0]}
    )
    raw["cluster"] = raw.later_outcome
    groups = Groups(column="group", treated_level="yes", reason="declared group", cites=[])
    periods = Periods(kind="wide", before_column="earlier_outcome", after_column="later_outcome", reason="declared waves", cites=[])
    candidates = ["cluster"] if cluster_source == "state" else []
    p, _ = canonical(raw, groups, periods, "later_outcome", candidates, cluster_column=cluster_source)
    assert p.cluster.astype(float).tolist() == [8.0, 7.0, 3.0, 4.0] * 2
    h = context()
    h.outcome = "later_outcome"
    h.design = h.design.model_copy(update={"treated_group": {"column": "group", "level": "yes"}, "cluster_level": cluster_source})
    if candidates:
        h.columns.append(ColumnBrief(name="cluster", key="cluster", role="candidate", when="after", measures_outcome=True))
    path = tmp_path / "panel.csv"
    p.to_csv(path, index=False)
    run_dir = tmp_path / "run"
    entry = Entry(id="rel:anticipation.uncertain", kind="anticipation", value=1, status="proposed", reason="an unresolved validation-only question")
    command = nodes.probe({"handoff": h, "panel_path": str(path), "run_dir": str(run_dir), "ledger": Ledger(entries={entry.id: entry})})
    assert command.goto == "feasibility"
    assert command.update["feasibility"].terminal_kind == "context_deficiency"
    assert command.update["context_deficiency"]["addresses"] == ["col:cluster" if candidates else "col:later_outcome"]
    assert not run_dir.exists()


def test_admitted_wide_outcome_cluster_alias_fails_before_load_writes_or_model_calls(tmp_path, monkeypatch):
    handoff_path = Path(__file__).parents[1] / "evals/handoffs/card_krueger_forced.json"
    h = Handoff.model_validate_json(handoff_path.read_text())
    h.design = h.design.model_copy(update={"cluster_level": h.design.after_column})
    h = admit(h, ENDPOINTS)
    assert validate_context(h, ENDPOINTS) is None

    def forbidden(*_args, **_kwargs):
        raise AssertionError("malformed metadata must stop before intake writes or any model episode")

    monkeypatch.setenv("RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(nodes.intake, "load", forbidden)
    monkeypatch.setattr(nodes, "run_episode", forbidden)
    command = nodes.load({"handoff": h})
    assert command.goto == "feasibility"
    assert command.update["feasibility"].terminal_kind == "context_deficiency"
    assert command.update["context_deficiency"]["addresses"] == [f"col:{h.design.after_column}"]
    assert not list(tmp_path.rglob("*"))


def test_code_reads_only_projection_and_preserves_addressed_numeric_provenance(tmp_path, monkeypatch):
    p = input_panel()
    source = tmp_path / "full_panel.csv"
    p.to_csv(source, index=False)
    source_bytes = source.read_bytes()
    projection = diagnostic_table(context(), p)
    tools = tools_for(projection, tmp_path / "run", required=False)
    untouched = episode.EpisodeLog(node="probe", visit="did-code")
    assert tools.completion_errors(untouched) == []
    assert tools.evidence_errors([], untouched) == []
    script = f"""import os, json
from pathlib import Path
import numpy as np
import pandas as pd
table = pd.read_csv(os.environ['DATA_CSV'])
try:
    Path({str(source)!r}).read_text()
except OSError:
    pass
else:
    raise AssertionError('the full panel was readable')
assert table.loc[table.post == 1, ['y', 'baseline', 'unaffected']].isna().all().all()
facts = {{'panel_rows': len(table), 'pre_outcome_rows': int(table.y.notna().sum()), 'pre_covariate_rank': int(np.linalg.matrix_rank(table.loc[table.post == 0, ['baseline', 'unaffected']].to_numpy()))}}
Path('facts.json').write_text(json.dumps(facts))
print(facts)
"""
    response = episode._run(tools, "run_code", {"code": script}, 3, untouched)
    assert tools.steps[0].ok, tools.steps[0].stderr
    address = "probe:probe.did-code.1"
    assert response.startswith(f"[{address}]") and untouched.resolve(address)
    assert tools.completion_errors(untouched) == []
    assert tools.evidence_errors([address], untouched) == []
    assert tools.evidence_errors([], untouched)
    assert tools.steps[0].facts["pre_outcome_rows"] == float((p.post == 0).sum())
    reading = ProbeReading(notes="The pre-period rank diagnostic leaves anticipation unresolved.", diagnostic_cites=[address], operations=[])
    assert record_errors(reading, untouched, tools, untouched.resolve) == []
    unknown = reading.model_copy(update={"diagnostic_cites": ["probe:probe.wrong-visit.1"]})
    assert any("unresolved diagnostic citation" in error for error in record_errors(unknown, untouched, tools, untouched.resolve))
    entry = Entry(id="rel:anticipation.uncertain", kind="anticipation", value=1, status="proposed", reason="a conditional uncertainty")
    ledger = Ledger(entries={entry.id: entry})
    unchanged, errors = commit(ledger, reading, "probe", untouched.resolve, handoff=context(), log=untouched)
    assert not errors and unchanged.model_dump() == ledger.model_dump()
    assert source.read_bytes() == source_bytes
    assert tools.csv.read_bytes() == projection.to_csv(index=False).encode()
