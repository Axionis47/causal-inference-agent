"""Cutoff diagnostic inputs preserve the assignment rule and withhold target outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from causal_agent.common import trace
from causal_agent.common.addresses import key
from causal_agent.common.contracts import Cited, Handoff
from causal_agent.explore import gates
from causal_agent.families.discontinuity.lane.contracts import CovariateRelation, CovariateRoles, Covariates, Ladder, Score
from causal_agent.families.discontinuity.lane.reason.diagnostics import diagnostic_table, diagnostic_tools
from causal_agent.families.discontinuity.lane.reason.nodes import line
from causal_agent.families.discontinuity.lane.reason.nodes import score as score_node
from causal_agent.lane import episode

ROOT = Path(__file__).resolve().parents[4]
HANDOFF = ROOT / "causal_agent/families/discontinuity/evals/handoffs/senate_context.json"


def role(column, **updates):
    return CovariateRelation(
        column=column,
        predetermined=True,
        affected_by_treatment=False,
        is_outcome_measure=False,
        reasons=[Cited(reason="the recorded pre-assignment account", cites=[f"col:{column}.when"])],
    ).model_copy(update=updates)


@pytest.fixture
def diagnostic_input(tmp_path):
    h = Handoff.model_validate_json(HANDOFF.read_text())
    source = ROOT / h.csv
    table = pd.read_csv(source).rename(columns=key)
    path = tmp_path / "loaded.csv"
    table.to_csv(path, index=False)
    roles = CovariateRoles(
        items=[
            role("population"),
            role("demvoteshlag1"),
            role("dopen", affected_by_treatment=True),
            role("presdemvoteshlag1", predetermined=False),
            role("demvoteshfor1"),
            role("year"),
            role("margin"),
            role("vote"),
            role("state"),
        ]
    )
    allowed = [r.column for r in roles.items if r.column != "year"]
    state = {
        "handoff": h,
        "table_path": str(path),
        "run_dir": str(tmp_path / "run"),
        "score": Score(column="margin", cutoff=0, treated_side="above", cutoff_value_treated=True, reason="admitted assignment rule", cites=[]),
        "cluster_column": "state",
        "covariates": Covariates(balance_tested=allowed, adjusted=allowed),
        "ladder": Ladder(covariates=roles),
    }
    return SimpleNamespace(state=state, table=table, source=source, source_bytes=source.read_bytes())


def test_projection_includes_only_rule_receipt_cluster_and_approved_predetermined_values(diagnostic_input):
    projected, metadata = diagnostic_table(diagnostic_input.state)
    assert list(projected) == ["x", "side", "cluster", "cov_population", "cov_demvoteshlag1"]
    assert len(projected) == len(diagnostic_input.table) == 1390
    assert diagnostic_input.table.vote.isna().sum() == 93
    pd.testing.assert_series_equal(projected.x, diagnostic_input.table.margin, check_names=False)
    np.testing.assert_array_equal(projected.side, (diagnostic_input.table.margin >= 0).astype(float))
    pd.testing.assert_series_equal(projected.cov_population, diagnostic_input.table.population, check_names=False)
    assert "original cutoff 0.0" in metadata and "equality at the original cutoff is treated" in metadata
    assert "including rows with missing outcomes" in metadata
    assert diagnostic_input.source.read_bytes() == diagnostic_input.source_bytes


def test_diagnostic_projection_does_not_change_with_outcome_values_or_missingness(diagnostic_input):
    before, metadata = diagnostic_table(diagnostic_input.state)
    altered = diagnostic_input.table.copy()
    altered["vote"] = np.where(np.arange(len(altered)) % 3, 1e9, np.nan)
    altered["demvoteshfor1"] = -1e9
    altered.to_csv(diagnostic_input.state["table_path"], index=False)
    after, changed_metadata = diagnostic_table(diagnostic_input.state)
    pd.testing.assert_frame_equal(before, after)
    assert metadata == changed_metadata


def test_strict_below_rule_and_receipt_follow_the_admitted_levels(diagnostic_input):
    state = diagnostic_input.state
    raw = diagnostic_input.table.iloc[:5].copy()
    raw["margin"] = [-2, 0, 1, -1, np.nan]
    raw["dopen"] = [1, 1, 0, np.nan, 0]
    raw.to_csv(state["table_path"], index=False)
    state["score"] = state["score"].model_copy(update={"treated_side": "below", "cutoff_value_treated": False, "takeup_column": "dopen", "takeup_level": "1"})
    projected, metadata = diagnostic_table(state)
    np.testing.assert_allclose(projected.x, [1.5, -0.5, -1.5, 0.5, np.nan], equal_nan=True)
    np.testing.assert_allclose(projected.side, [1, 0, 0, 1, np.nan], equal_nan=True)
    np.testing.assert_allclose(projected.t, [1, 1, 0, np.nan, 0], equal_nan=True)
    assert "cov_dopen" not in projected
    assert "effective-cutoff shift -0.5" in metadata and "equality at the original cutoff is control" in metadata


@pytest.mark.parametrize("field", ["column", "takeup_column", "cluster_column"])
def test_outcome_cannot_enter_through_a_rule_or_cluster_alias(diagnostic_input, field):
    state = diagnostic_input.state
    if field == "cluster_column":
        state[field] = "vote"
    else:
        state["score"] = state["score"].model_copy(update={field: "vote"})
    with pytest.raises(ValueError, match="target outcome"):
        diagnostic_table(state)


@pytest.mark.parametrize("field", ["column", "takeup_column"])
def test_outcome_proxy_as_score_or_receipt_returns_deficiency_before_any_projection_write(diagnostic_input, field):
    state = diagnostic_input.state
    state["score"] = state["score"].model_copy(update={field: "demvoteshfor1"})
    result = line(state)
    assert result.goto == "feasibility"
    assert result.update["feasibility"].terminal_kind == "context_deficiency"
    assert "outcome measure" in result.update["context_deficiency"].reason
    assert not Path(state["run_dir"]).exists()


@pytest.mark.parametrize("field", ["column", "takeup_column", "cluster_column"])
def test_secondary_outcome_role_blocks_essential_alias_without_measurement_flag(diagnostic_input, field):
    state = diagnostic_input.state
    h = state["handoff"]
    state["handoff"] = h.model_copy(
        update={"columns": [c.model_copy(update={"role": "outcome", "measures_outcome": None}) if c.key == "demvoteshfor1" else c for c in h.columns]}
    )
    if field == "cluster_column":
        state[field] = "demvoteshfor1"
    else:
        state["score"] = state["score"].model_copy(update={field: "demvoteshfor1"})
    result = line(state)
    assert result.update["feasibility"].terminal_kind == "context_deficiency"
    assert "outcome measure" in result.update["context_deficiency"].reason
    assert not Path(state["run_dir"]).exists()


def test_secondary_outcome_role_excludes_a_covariate_without_measurement_flag(diagnostic_input):
    state = diagnostic_input.state
    unchanged, _ = diagnostic_table(state)
    h = state["handoff"]
    state["handoff"] = h.model_copy(
        update={"columns": [c.model_copy(update={"role": "outcome", "measures_outcome": None}) if c.key == "population" else c for c in h.columns]}
    )
    projected, _ = diagnostic_table(state)
    assert "cov_population" not in projected
    pd.testing.assert_frame_equal(projected, unchanged.drop(columns="cov_population"))


@pytest.mark.parametrize("role", ["score", "receipt", "cluster"])
def test_outcome_sources_stop_at_score_before_fixed_reasoning_tools(diagnostic_input, role):
    state = diagnostic_input.state
    h = state["handoff"]
    source = "margin" if role == "score" else "demvoteshfor1"
    state["handoff"] = h.model_copy(
        update={"columns": [c.model_copy(update={"role": "outcome", "measures_outcome": None}) if c.key == source else c for c in h.columns]}
    )
    if role == "receipt":
        state["handoff"].design = h.design.model_copy(update={"takeup": {"column": source, "level": "1"}})
    elif role == "cluster":
        state["cluster_column"] = source
    Path(state["table_path"]).unlink()
    result = score_node(state)
    assert result.goto == "feasibility"
    assert result.update["feasibility"].terminal_kind == "context_deficiency"
    assert result.update["context_deficiency"].stage == "score"
    assert "outcome measure" in result.update["context_deficiency"].reason
    assert not Path(state["run_dir"]).exists()


def test_treatment_affected_score_is_blocked_but_receipt_is_allowed(diagnostic_input):
    state = diagnostic_input.state
    h = state["handoff"]
    state["handoff"] = h.model_copy(update={"columns": [c.model_copy(update={"moved_by_change": True}) if c.key == "margin" else c for c in h.columns]})
    result = line(state)
    assert result.update["feasibility"].terminal_kind == "context_deficiency"
    assert "score cannot be treatment-affected" in result.update["context_deficiency"].reason
    assert not Path(state["run_dir"]).exists()
    state["handoff"] = h.model_copy(update={"columns": [c.model_copy(update={"moved_by_change": True}) if c.key == "dopen" else c for c in h.columns]})
    state["score"] = state["score"].model_copy(update={"takeup_column": "dopen", "takeup_level": "1"})
    projected, _ = diagnostic_table(state)
    np.testing.assert_allclose(projected.t, diagnostic_input.table.dopen, equal_nan=True)


@pytest.mark.parametrize("outcome", ["x", "side", "t", "cluster", "cov_population"])
def test_raw_outcome_name_cannot_disable_a_distinct_safe_projection_alias(diagnostic_input, outcome):
    state = diagnostic_input.state
    h = state["handoff"]
    state["handoff"] = h.model_copy(
        update={"outcome": outcome, "columns": [c.model_copy(update={"key": outcome, "name": outcome}) if c.key == "vote" else c for c in h.columns]}
    )
    diagnostic_input.table.rename(columns={"vote": outcome}).to_csv(state["table_path"], index=False)
    state["score"] = state["score"].model_copy(update={"takeup_column": "dopen", "takeup_level": "1"})
    tools, _ = diagnostic_tools(state)
    projected = pd.read_csv(tools.csv)
    assert tools.dropped_name is None
    assert gates.names_dropped_column(f"table[{outcome!r}].notna().sum()", tools.dropped_name) == ""
    assert outcome in projected
    assert not projected[outcome].equals(diagnostic_input.table.vote)
    assert "vote" not in projected


def test_code_sees_only_the_projection_and_retains_addressed_numerical_artifacts(diagnostic_input, monkeypatch):
    tools, _ = diagnostic_tools(diagnostic_input.state)
    assert not tools.required and tools.dropped_name == "vote"
    written = set()
    original_write = trace.write

    def capture_write(path, content, **kwargs):
        written.add(Path(path))
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(trace, "write", capture_write)
    script = f"""import os, json
from pathlib import Path
import pandas as pd
data = pd.read_csv(os.environ['DATA_CSV'])
assert set(data) == {{'x', 'side', 'cluster', 'cov_population', 'cov_demvoteshlag1'}}
for source in [{diagnostic_input.state["table_path"]!r}, {str(diagnostic_input.source)!r}]:
    try:
        Path(source).read_text()
    except OSError:
        pass
    else:
        raise AssertionError('full outcome table was readable')
near = data.loc[data.x.abs() <= 2]
facts = {{'rows': len(data), 'near_left': int((near.side == 0).sum()), 'near_right': int((near.side == 1).sum())}}
Path('facts.json').write_text(json.dumps(facts))
near.side.value_counts().to_csv('support.csv')
print(facts)
"""
    log = episode.EpisodeLog(node="line", visit="rd-diagnostics-test")
    assert tools.completion_errors(log) == []
    response = episode._run(tools, "run_code", {"code": script}, 3, log)
    step = tools.steps[0]
    assert step.ok, step.stderr
    address = "probe:line.rd-diagnostics-test.1"
    assert response.startswith(f"[{address}] run_code(") and log.resolve(address)
    assert tools.completion_errors(log) == [] and tools.evidence_errors([address], log) == []
    assert tools.evidence_errors([], log)
    assert step.facts == {"rows": 1390.0, "near_left": 50.0, "near_right": 52.0}
    assert json.loads((tools.out_dir / "step_1/facts.json").read_text()) == step.facts
    assert {tools.out_dir / "step_1" / name for name in ("code.py", "stdout.txt", "facts.json", "support.csv")} <= written
    assert diagnostic_input.source.read_bytes() == diagnostic_input.source_bytes
