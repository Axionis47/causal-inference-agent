from __future__ import annotations

import zipfile
from io import BytesIO

import pandas as pd
from langgraph.types import Command

from causal.checks import CHECKS
from causal.studio_export import build_bundle
from causal.monitoring import (
    append_interaction,
    read_analysis_runs,
    read_design_contracts,
    read_interactions,
    save_analysis_run,
    save_design_contract,
)
from causal.preparation_agent import run as run_preparation_agent
from causal.studio_eda import (
    apply_cohort,
    column_summary,
    grouped_trends,
    kaplan_meier_frame,
    standardized_differences,
)
from causal.studio_policy import evaluate
from causal.studio_prep import (
    apply_repairs,
    build_data_version,
    propose_repairs,
    quality_summary,
    read_bundle,
)
from causal.studio_protocols import (
    build_contract,
    contract_hash,
    protocol_as_dict,
    run_preflight,
)
from causal.studio_workflow import build_in_memory, node_estimate
from fixtures import cases


def context(**overrides):
    value = {
        "question": "Did treatment change the outcome?",
        "description": "Synthetic randomized verification data.",
        "unit": "one participant",
        "assignment": "randomized",
        "timing": "treatment before outcome",
        "population": "test participants",
        "intended_use": "software verification",
        "outcome": "y",
        "treatment": "t",
        "high_impact": False,
    }
    value.update(overrides)
    return value


def approved_design(frame, lane, kwargs, ctx):
    data_version = build_data_version(frame, frame, cohort=None, repairs=[])
    data_version["revision"] = 1
    contract = build_contract(
        dataset_id="test-data", lane=lane, kwargs=kwargs, context=ctx, cohort=None,
        data_version=data_version,
        answers={"test assumption": "confirmed for software verification"},
    )
    contract["configuration_hash"] = contract_hash(build_contract(
        dataset_id="test-data", lane=lane, kwargs=kwargs, context=ctx, cohort=None,
        data_version=data_version,
    ))
    contract["contract_hash"] = contract_hash(contract)
    contract["revision"] = 1
    contract["revision_timing"] = "pre_estimation"
    contract["approval"] = {
        "approved": True,
        "reviewer": "test reviewer",
        "role_ledger_and_map_confirmed": True,
    }
    return {
        "data_version": data_version,
        "design_contract": contract,
        "design_approval": contract["approval"],
        "preflight": run_preflight(frame, lane, kwargs, ctx),
        "protocol": protocol_as_dict(lane),
    }


def test_repairs_are_proposed_then_applied_to_a_copy():
    raw = pd.DataFrame({" treatment ": [0, 0, 1, 1], "y": ["1", " ", "3", "3"]})
    raw = pd.concat([raw, raw.iloc[[3]]], ignore_index=True)
    plan = propose_repairs(raw)
    ids = {item["id"] for item in plan}
    assert "normalize_column_names" in ids
    assert "blank_strings_to_null" in ids
    assert "drop_exact_duplicates" in ids

    clean, audit = apply_repairs(raw, plan, list(ids))
    assert " treatment " in raw.columns  # immutable source
    assert "treatment" in clean.columns
    assert len(clean) == 3
    assert clean["y"].isna().sum() == 1
    assert {item["id"] for item in audit} == ids


def test_interactive_eda_is_deterministic_and_cohort_is_explicit():
    frame = pd.DataFrame({
        "person_id": range(6),
        "date": pd.date_range("2024-01-01", periods=6).astype(str),
        "treatment": [0, 0, 0, 1, 1, 1],
        "age": [20, 22, 24, 40, 42, 44],
        "outcome": [1, 2, 3, 5, 6, 7],
        "event": [0, 1, 0, 1, 1, 0],
        "period": [0, 1, 0, 1, 0, 1],
    })
    summary = column_summary(frame).set_index("column")
    assert "possible identifier" in summary.loc["person_id", "signals"]
    assert "date candidate" in summary.loc["date", "signals"]

    spec = {"kind": "numeric_range", "column": "age", "low": 22, "high": 42}
    cohort = apply_cohort(frame, spec)
    assert cohort.age.tolist() == [22, 24, 40, 42]
    assert len(frame) == 6  # preview never mutates the uploaded frame

    balance = standardized_differences(frame, "treatment", ["age"])
    assert balance.iloc[0]["standardised difference"] > 0.25
    curve = kaplan_meier_frame(frame, "age", "event", "treatment")
    assert not curve.empty
    assert curve.groupby("group").survival.apply(lambda values: values.is_monotonic_decreasing).all()
    trends = grouped_trends(frame, "period", "treatment", "outcome")
    assert {"mean outcome", "rows"} <= set(trends.columns)


def test_server_interaction_events_are_chained_sanitized_and_persistent(tmp_path):
    path = tmp_path / "events.sqlite"
    first = append_interaction(
        session_id="session-1",
        dataset_id="dataset-1",
        kind="eda_distribution_changed",
        stage="data_understanding",
        payload={"column": "age", "email": "person@example.com"},
        path=path,
    )
    second = append_interaction(
        session_id="session-1",
        dataset_id="dataset-1",
        kind="analysis_population_committed",
        stage="data_understanding",
        payload={"rows_before": 100, "rows_after": 80},
        parent_event_id=first["event_id"],
        path=path,
    )
    events = read_interactions("session-1", path=path)
    assert [item["kind"] for item in events] == [
        "eda_distribution_changed", "analysis_population_committed"
    ]
    assert events[0]["payload"]["email"] == "[redacted]"
    assert events[1]["parent_event_id"] == first["event_id"]
    assert second["schema_version"] == "1.0.0"


def test_all_eight_protocols_run_preflight_before_any_effect_is_estimated():
    for case in cases():
        ctx = context(
            outcome=case.kwargs.get("outcome", case.kwargs.get("event", "outcome")),
            treatment=case.kwargs.get("treatment", ""),
            timing="all selected roles were reviewed in temporal order",
        )
        protocol = protocol_as_dict(case.name)
        findings = run_preflight(case.frame, case.name, case.kwargs, ctx)
        assert {item["check"] for item in findings} == set(protocol["pre_checks"])
        assert not [item for item in findings if item["verdict"] == "fail"]
        assert set(protocol["post_checks"]) <= set(CHECKS)
        assert all(item["phase"] == "pre" for item in findings)


def test_contract_hash_changes_when_scientific_configuration_changes():
    frame = pd.DataFrame({"t": [0, 1] * 20, "y": range(40), "age": range(20, 60)})
    ctx = context()
    without_age = build_contract(
        dataset_id="d1", lane="observational",
        kwargs={"outcome": "y", "treatment": "t", "covariates": []},
        context=ctx, cohort=None,
    )
    with_age = build_contract(
        dataset_id="d1", lane="observational",
        kwargs={"outcome": "y", "treatment": "t", "covariates": ["age"]},
        context=ctx, cohort=None,
    )
    assert contract_hash(without_age) != contract_hash(with_age)
    assert {item["role"] for item in with_age["role_ledger"]} >= {
        "treatment", "outcome", "confounder"
    }


def test_prepared_data_version_changes_with_repairs_or_content():
    raw = pd.DataFrame({"t": [0, 1, 1], "y": [1.0, 2.0, 2.0]})
    repaired = raw.drop_duplicates().copy()
    original = build_data_version(raw, raw, cohort=None, repairs=[])
    changed = build_data_version(
        raw,
        repaired,
        cohort=None,
        repairs=[{"id": "drop_exact_duplicates"}],
    )
    assert original["version_id"] != changed["version_id"]
    assert original["prepared_fingerprint"] != changed["prepared_fingerprint"]
    assert original["manifest_hash"] != changed["manifest_hash"]


def test_frozen_contract_revisions_persist_by_dataset_fingerprint(tmp_path):
    path = tmp_path / "memory.sqlite"
    contract = {
        "dataset_id": "bundle:table.csv",
        "revision": 1,
        "contract_hash": "abc123",
        "lane": "observational",
        "approval": {"approved": True, "reviewer": "test"},
    }
    save_design_contract(contract, path=path)
    save_design_contract(contract, path=path)  # immutable hash is idempotent
    assert read_design_contracts("bundle:table.csv", path=path) == [contract]
    assert read_design_contracts("different:table.csv", path=path) == []


def test_run_lineage_persists_without_storing_dataset_rows(tmp_path):
    path = tmp_path / "memory.sqlite"
    save_analysis_run({
        "run_id": "run-2", "dataset_id": "bundle:table.csv",
        "data_version_id": "data-v2", "contract_hash": "contract-v2",
        "parent_run_id": "run-1", "status": "estimated",
    }, path=path)
    runs = read_analysis_runs("bundle:table.csv", path=path)
    assert runs[0] | {"created_at": "ignored"} == {
        "run_id": "run-2", "dataset_id": "bundle:table.csv",
        "data_version_id": "data-v2", "contract_hash": "contract-v2",
        "parent_run_id": "run-1", "status": "estimated",
        "created_at": "ignored",
    }
    assert read_analysis_runs("different:table.csv", path=path) == []


def test_policy_blocks_missing_context_and_reviews_high_impact():
    blocked = evaluate({"context": context(question=""), "estimate": {"value": 1}})
    assert blocked["decision"] == "block"

    frame = pd.DataFrame({"t": [0, 1], "y": [0.0, 1.0]})
    version = build_data_version(frame, frame, cohort=None, repairs=[])
    reviewed = evaluate({
        "context": context(high_impact=True),
        "estimate": {"value": 1},
        "diagnostics": [],
        "lane": "observational",
        "data_quality": {},
        "data_version": version,
        "design_contract": {"contract_hash": "test", "data_version": version},
        "design_approval": {
            "approved": True,
            "role_ledger_and_map_confirmed": True,
        },
        "preflight": [],
    })
    assert reviewed["decision"] == "review"
    assert any(item["rule"] == "use.high_impact" for item in reviewed["findings"])


def test_execution_guard_blocks_a_changed_artifact_before_lane_dispatch(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "t": [0, 1] * 40,
        "y": [0.1, 2.1] * 40,
        "age": list(range(20, 100)),
    })
    path = tmp_path / "analysis.csv"
    frame.to_csv(path, index=False)
    ctx = context()
    kwargs = {"outcome": "y", "treatment": "t", "covariates": ["age"]}
    state = {
        "csv_path": str(path), "lane": "observational", "kwargs": kwargs,
        "context": ctx, "cohort": None, "events": [],
    } | approved_design(frame, "observational", kwargs, ctx)
    frame.assign(y=frame.y + 100).to_csv(path, index=False)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("lane must not be called")

    module = __import__("causal.studio_workflow", fromlist=["LANE_FUNCTIONS"])
    monkeypatch.setitem(module.LANE_FUNCTIONS, "observational", forbidden)
    result = node_estimate(state)
    assert not called
    assert "changed after its version was approved" in result["error"]
    assert result["events"][-1]["stage"] == "execution_guard"


def test_unknown_bundle_is_investigated_with_versioned_tools(monkeypatch):
    monkeypatch.setenv("PREPARATION_OFFLINE", "true")
    archive_bytes = BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(
            "customers.csv",
            "customer_id,treatment,outcome,age\n1,0,0,20\n2,1,1,25\n3,0,0,30\n",
        )
        archive.writestr("notes.csv", "customer_id,note\n1,a\n2,b\n")
    tables = read_bundle([("kaggle.zip", archive_bytes.getvalue())])
    assert set(tables) == {"kaggle/customers.csv", "kaggle/notes.csv"}

    # Use a fuller table so lane structural checks have enough rows.
    tables["kaggle/customers.csv"] = pd.DataFrame({
        "customer_id": range(80),
        "treatment": [0, 1] * 40,
        "outcome": [0.1, 2.1] * 40,
        "age": range(20, 100),
    })
    plan = run_preparation_agent(
        tables,
        "Did treatment change outcome?",
        "Unknown Kaggle customer export.",
    ).to_dict()
    assert plan["primary_table"] == "kaggle/customers.csv"
    assert plan["prompt"]["prompt_id"] == "preparation-agent"
    assert plan["prompt"]["version"] == "1.1.0"
    assert plan["prompt"]["sha256"]
    assert plan["provider"] == "vertex-ai"
    assert plan["location"]
    assert "observational" in plan["eligible_lanes"]
    assert "matching" in plan["eligible_lanes"]
    assert plan["recommended_lane"] == "observational"
    assert {call["tool"] for call in plan["trace"]} >= {
        "list_tables", "inspect_table", "select_primary_table",
        "draft_context", "check_lane_readiness", "recommend_lane", "finalize_plan",
    }
    assert all(call["provider"] == "vertex-ai" for call in plan["trace"])
    assert any("joining" in question for question in plan["unresolved_questions"])


def test_graph_pauses_for_human_then_exports_executable_bundle(tmp_path):
    frame = pd.DataFrame({
        "t": [0, 1] * 40,
        "y": [0.1, 2.1] * 40,
        "age": list(range(20, 100)),
    })
    csv_path = tmp_path / "analysis_data.csv"
    frame.to_csv(csv_path, index=False)
    graph = build_in_memory()
    config = {"configurable": {"thread_id": "review-case"}}
    ctx = context(high_impact=True)
    kwargs = {"outcome": "y", "treatment": "t", "covariates": ["age"]}
    graph.invoke({
        "run_id": "review-case",
        "csv_path": str(csv_path),
        "source": "test.csv",
        "lane": "observational",
        "kwargs": kwargs,
        "context": ctx,
        "repairs": [],
        "data_quality": quality_summary(frame),
        "preparation": {"mode": "deterministic", "tables_seen": 1, "trace": []},
        "prompt_versions": {},
        "events": [],
    } | approved_design(frame, "observational", kwargs, ctx), config=config)

    paused = graph.get_state(config)
    assert paused.next == ("human_gate",)
    assert paused.values["policy"]["decision"] == "review"
    assert not paused.values.get("report")

    graph.invoke(
        Command(resume={"approved": True, "reviewer": "test", "note": "checked"}),
        config=config,
    )
    done = graph.get_state(config)
    assert done.next == ()
    assert done.values["policy"]["decision"] == "approved"
    assert "Publication policy" in done.values["report"]
    assert done.values["monitoring"]["policy_decision"] == "approved"

    payload = build_bundle(dict(done.values))
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = set(archive.namelist())
        assert {
            "analysis.ipynb", "analysis_data.csv", "policy.json", "run.json",
            "monitoring.json", "design_contract.json", "data_version.json", "preflight.json",
        } <= names
        assert {"causal/lanes.py", "causal/prep.py", "requirements.txt"} <= names
