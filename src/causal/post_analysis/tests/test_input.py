"""Scientific provenance, requested evidence and role semantics at the post boundary."""
from __future__ import annotations

import pytest

from causal.post_analysis.contracts import InputError
from causal.post_analysis.tests.input_support import handoff


@pytest.mark.parametrize("method", ["randomized_experiment", "aipw",
                                   "did", "sharp_rdd"])
def test_exact_method_roles_and_approved_dag_without_upstream_plotting(method: str) -> None:
    sources = handoff(method)
    sources.put("CausalContext", {"frame": {}, "concept_ids": ["unapproved"], "edges": []}, key="decoy")
    packet = sources.packet()
    assert packet.evidence["plan"]["method_id"] == method
    assert packet.diagram is not None
    assert packet.diagram.source == sources.refs["CausalContext"]
    assert {node.node_id for node in packet.diagram.nodes} == {"offer", "completion"}
    assert "required_visual_evidence" not in packet.evidence["design"]
    assert "figure_builder_ids" not in packet.evidence["plan"]
    assert "svg" not in repr(packet.evidence)
    assert "diagnostic:future_unknown_diagnostic" in packet.required_evidence
    for kind in ("DesignApproval", "DesignReviewBundle", "GraphViewSet", "CausalGraphView",
                 "PreparedFrame", "ColumnSemanticCard", "AnalysisContributionMask"):
        assert sources.refs[kind] in packet.sources.values()
    assert sources.refs["decoy"] not in packet.sources.values()


def test_supplied_intervals_and_comparison_rows_retain_exact_sources() -> None:
    sources = handoff()
    packet = sources.packet()
    table = packet.tables["comparison:0"]
    assert table.source == sources.refs["NumericalBundle"]
    assert [row.source for row in table.row_sources] == [sources.refs["PrimaryAnalysisResult"],
                                                       sources.refs["SensitivityResult"]]
    assert [row.selector for row in table.row_sources] == ["/primary_items/0", "/result"]
    columns = {column.name: column for column in table.columns}
    assert columns["estimate"].quantity == columns["interval_lower"].quantity == columns["interval_upper"].quantity
    assert table.rows[0][1:4] == (0.2, 0.1, 0.3)
    assert table.rows[1][1:4] == (0.25, 0.1, 0.3)
    diagnostic = packet.tables["diagnostic:future_unknown_diagnostic"]
    assert diagnostic.columns[1].units is None
    assert diagnostic.rows == (("recorded_metric", 0.3),)
    assert any("No branch-specific diagnostic evidence" in limitation for limitation in packet.limitations)
    assert packet.tables["support:balance"].row_sources[0].selector == "/measurements/balance/smd~1age"


def test_mismatched_effect_units_cannot_join_the_primary_comparison() -> None:
    packet = handoff(incompatible_branch=True).packet()
    assert "comparison:0" not in packet.tables
    assert "sensitivity:alternative" in packet.tables


@pytest.mark.parametrize("coverage", ["missing", "duplicate"])
def test_every_requested_result_is_present_exactly_once(coverage: str) -> None:
    with pytest.raises(InputError) as failed:
        handoff(coverage=coverage).packet()
    issue = failed.value.issues[0]
    assert issue.owner == "analysis" and issue.path == "/results"


def test_explicit_failed_diagnostic_is_preserved_without_manufacturing_a_pass() -> None:
    packet = handoff(coverage="failed").packet()
    assert packet.evidence["diagnostic:future_unknown_diagnostic"]["execution_status"] == "failed"
    assert "diagnostic:future_unknown_diagnostic" in packet.required_evidence


def test_payload_bytes_cannot_be_modified_even_without_changing_json_values() -> None:
    sources = handoff()
    ref = sources.refs["PrimaryAnalysisResult"]
    sources.objects[ref.artifact_id] += b"\n"
    with pytest.raises(InputError) as failed:
        sources.packet()
    assert failed.value.issues[0].path == "/payload_bytes"
    assert failed.value.issues[0].source == ref


def test_actual_outcome_role_must_match_the_frozen_request_context() -> None:
    with pytest.raises(InputError) as failed:
        handoff(wrong_role=True).packet()
    assert failed.value.issues[0].path == "/role_columns"


def test_no_response_requires_exact_attempt_and_never_invents_individual_results() -> None:
    sources = handoff(failed=True)
    packet = sources.packet()
    assert packet.evidence["failure"]["status"] == "not_estimable"
    assert packet.sources["plan"] == sources.refs["EstimationPlan"]
    assert packet.required_evidence == ("design", "plan", "failure")
    assert packet.tables == {}
    assert not any(key.startswith("diagnostic:") for key in packet.evidence)
    assert "future_unknown_diagnostic" in packet.evidence["plan"]["required_diagnostics"]
    with pytest.raises(InputError) as failed:
        handoff(failed=True, attempted_plan=False).packet()
    assert failed.value.issues[0].code == "attempt_request_unavailable"


def test_unreachable_storage_is_not_misreported_as_faulty_science() -> None:
    sources = handoff()
    sources.objects.pop(sources.refs["CausalContext"].artifact_id)
    with pytest.raises(InputError) as failed:
        sources.packet()
    assert failed.value.issues[0].owner == "storage"
    assert failed.value.issues[0].code == "source_unreachable"


def test_valid_dag_exceeding_display_capacity_is_a_receiver_support_gap() -> None:
    sources = handoff(concept_count=41)
    with pytest.raises(InputError) as failed:
        sources.packet()
    issue = failed.value.issues[0]
    assert issue.owner == "post_analysis"
    assert issue.code == "reader_unsupported"
    assert issue.source == sources.refs["CausalContext"]
