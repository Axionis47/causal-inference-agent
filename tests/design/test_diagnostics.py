"""Read-only pre-repair diagnostics and the two inspection handlers (T-012 §4, §6)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from causal.design.diagnostics import (
    DIAGNOSTIC_SPECS,
    IMPLEMENTATION_VERSION,
    BytesFrameSource,
    CsvObjectFrameSource,
    DiagnosticError,
    make_diagnostic_handlers,
    run_diagnostic,
)
from causal.design.frame import DiagnosticResultV1, DiagnosticStatus
from causal.design.packs import PREREPAIR_DIAGNOSTIC_IDS, load_method_packs
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import AgentTaskEnvelopeV1, TaskBudgets, TaskStatus

ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64
REF = ArtifactRef(artifact_id="csv-1", content_hash=HASH)
PACKS = load_method_packs(ROOT / "registries" / "method-packs.v1.json")
# The golden frame: one null cluster, two missing outcomes, one unparsable dose, a repeated key.
CSV = (
    b"unit_id,arm,cluster,period,outcome,score,running,dose\n"
    b"u1,control,c1,2020,1.0,10,4.5,10\n"
    b"u2,control,c1,2020,,11,4.0,na\n"
    b"u3,treated,c2,2020,3.0,12,5.5,12\n"
    b"u4,treated,c2,2021,4.0,,6.0,13\n"
    b"u5,control,c1,2021,5.0,10,4.0,10\n"
    b"u6,treated,,2021,,13,5.0,10\n"
    b"u7,treated,c2,2021,7.0,14,5.0,14\n"
    b"u8,control,c1,2020,8.0,10,3.0,10\n"
)

ENVELOPE = AgentTaskEnvelopeV1(
    envelope_id="env-1", schema_version="agent-task-envelope.v1", analysis_id="an-1",
    stage_run_id="run-1", task_id="task-1", attempt_id="attempt-1", context_manifest=REF,
    task_kind="method_design", scope_kind="design", scope_ids=("design",),
    parent_artifacts=(REF,), allowed_evidence_ids=(),
    allowed_retrieval_ids=("intake_inventory",),
    allowed_tool_ids=("run_preflight_diagnostic", "preview_eligibility_impact"),
    output_schema_version="experiment-design.v1",
    validator_version="experiment-design-validator.v1", prompt_version="method-design.v1",
    model_profile_version="vertex-model-profile.v1",
    budgets=TaskBudgets(token_budget=8000, tool_call_budget=4),
    allowed_stopping_states=(TaskStatus.COMPLETE,), error_vocabulary=("SCHEMA_INVALID",),
    forbidden_payload_classes=("raw_rows",), payload_type="method-design-request.v1",
    payload={"method_id": "randomized_experiment"})


class RecordingSource:
    """A FrameSource that counts reads and exposes no writer of any kind."""

    def __init__(self, data: bytes = CSV) -> None:
        self.data = data
        self.reads = 0

    def csv_ref(self) -> ArtifactRef:
        return REF

    def frame(self) -> pl.DataFrame:
        self.reads += 1
        return pl.read_csv(io.BytesIO(self.data))


class FakeObjects:
    """The one ObjectStore method `CsvObjectFrameSource` may use."""

    def __init__(self) -> None:
        self.locators: list[str] = []

    def get(self, locator: str) -> bytes:
        self.locators.append(locator)
        return CSV


def run(spec_id: str, **params: Any) -> DiagnosticResultV1:
    return run_diagnostic(spec_id, RecordingSource(), params)


def handlers(source: RecordingSource) -> dict[str, Any]:
    return make_diagnostic_handlers(lambda: source, PACKS)


class TestPrimitives:
    def test_arm_counts_report_group_sizes(self) -> None:
        result = run("arm_counts", columns=["arm"])
        assert result.status is DiagnosticStatus.COMPUTED
        assert (result.total_rows, result.used_rows) == (8, 8)
        assert result.values["count:control"] == 4
        assert result.values["count:treated"] == 4
        assert result.columns_read == ("arm",)
        assert result.implementation_version == IMPLEMENTATION_VERSION

    def test_cluster_sizes_exclude_null_keys_from_the_denominator(self) -> None:
        result = run("cluster_sizes", columns=["cluster"])
        assert (result.total_rows, result.used_rows) == (8, 7)
        assert result.unused_reason_counts == {"null_excluded": 1}
        assert result.values["count:c1"] == 4
        assert result.values["count:c2"] == 3
        assert result.row_set_hash is not None

    def test_group_time_cells_report_expected_and_observed_cells(self) -> None:
        result = run("panel_completeness", columns=["arm", "period"])
        assert result.values["group_count"] == 4
        assert result.values["expected_cells"] == 4
        assert result.values["cell_completeness"] == 1.0
        assert result.values["count:control|2020"] == 3

    def test_cutoff_side_counts_use_the_derived_side_column(self) -> None:
        result = run("cutoff_side_counts", running_column="running", cutoff=5.0)
        assert result.values["count:below"] == 4
        assert result.values["count:at_or_above"] == 4
        assert result.columns_read == ("running",)

    def test_missing_share_reports_overall_group_and_cutoff_side_shares(self) -> None:
        overall = run("outcome_missingness", target="outcome")
        assert (overall.values["missing_count"], overall.values["missing_share"]) == (2, 0.25)
        assert overall.used_rows == 8
        grouped = run("missingness_by_group_time", target="outcome", by=["arm"])
        assert grouped.values["missing_count:control"] == 1
        assert grouped.values["missing_share:treated"] == 0.25
        sided = run("missingness_by_side_and_distance", target="outcome",
                    running_column="running", cutoff=5.0)
        assert sided.values["missing_count:below"] == 1
        assert sided.values["missing_count:at_or_above"] == 1

    def test_uniqueness_counts_repeated_keys(self) -> None:
        unique = run("assignment_unit_uniqueness", key_columns=["unit_id"])
        assert unique.values["distinct_key_count"] == 8
        assert unique.values["duplicate_row_count"] == 0
        assert unique.values["is_unique"] is True
        repeated = run("unit_period_uniqueness", key_columns=["arm", "period"])
        assert repeated.values["distinct_key_count"] == 4
        assert repeated.values["duplicate_row_count"] == 4
        assert repeated.warnings == ("4 rows repeat a key",)

    def test_level_profile_reports_level_counts_and_sparsity(self) -> None:
        result = run("level_sparsity", column="cluster", min_level_count=4)
        assert result.values["distinct_count"] == 2
        assert result.values["sparse_level_count"] == 1
        assert result.values["level:c1"] == 4
        assert (result.used_rows, result.unused_reason_counts) == (7, {"null_excluded": 1})

    def test_numeric_support_reports_range_mass_points_and_cutoff_bands(self) -> None:
        result = run("distance_to_cutoff_support", column="running", cutoff=5.0)
        assert (result.values["min"], result.values["max"]) == (3.0, 6.0)
        assert result.values["q50"] == 4.75
        assert result.values["mass_point:4.0"] == 0.25
        assert result.values["below_count"] == 4
        assert result.values["at_or_above_count"] == 4
        assert result.values["band:0"] == 3

    def test_numeric_support_counts_parse_failures_separately(self) -> None:
        result = run("mass_points", column="dose")
        assert (result.total_rows, result.used_rows) == (8, 7)
        assert result.unused_reason_counts == {"parse_failed": 1}
        assert result.values["max"] == 14.0

    def test_availability_uses_every_physical_row(self) -> None:
        result = run("baseline_availability", columns=["score", "outcome"])
        assert result.values["non_null_count:score"] == 7
        assert result.values["non_null_share:outcome"] == 0.75
        assert (result.used_rows, result.row_set_hash) == (8, None)


class TestStatuses:
    def test_a_missing_column_computes_the_rest_and_warns(self) -> None:
        result = run("baseline_availability", columns=["score", "nope"])
        assert result.status is DiagnosticStatus.PARTIAL
        assert result.columns_read == ("score",)
        assert "nope" in result.warnings[0]
        assert result.values["non_null_count:score"] == 7

    def test_absent_required_columns_or_parameters_are_not_computable(self) -> None:
        result = run("baseline_availability", columns=["nope"])
        assert result.status is DiagnosticStatus.NOT_COMPUTABLE
        assert (result.used_rows, result.values, result.columns_read) == (0, {}, ())
        assert result.total_rows == 8
        assert run("arm_counts").status is DiagnosticStatus.NOT_COMPUTABLE

    def test_an_unknown_diagnostic_fails_closed(self) -> None:
        with pytest.raises(DiagnosticError) as error:
            run("not_a_diagnostic")
        assert error.value.code == "unknown_diagnostic"


class TestDeterminism:
    def test_identical_input_gives_an_identical_model(self) -> None:
        first = run("cluster_sizes", columns=["cluster"])
        second = run("cluster_sizes", columns=["cluster"])
        assert first == second
        assert first.row_set_hash == second.row_set_hash
        assert len(first.row_set_hash or "") == 64

    def test_a_hash_appears_only_when_rows_were_dropped(self) -> None:
        assert run("arm_counts", columns=["arm"]).row_set_hash is None
        assert run("level_sparsity", column="cluster").row_set_hash is not None


class TestSpecTable:
    def test_every_pack_diagnostic_id_has_a_spec_row_with_a_known_primitive(self) -> None:
        assert set(DIAGNOSTIC_SPECS) == set(PREREPAIR_DIAGNOSTIC_IDS)
        primitives = {"count_by", "missing_share", "uniqueness", "level_profile",
                      "numeric_support", "availability"}
        for pack in PACKS.all():
            assert set(pack.allowed_prerepair_diagnostic_ids) <= set(DIAGNOSTIC_SPECS)
        for spec_id, spec in DIAGNOSTIC_SPECS.items():
            assert spec.diagnostic_id == spec_id
            assert spec.primitive in primitives


class TestFrameSources:
    def test_sources_only_read_and_every_reader_yields_the_same_frame(self) -> None:
        source = RecordingSource()
        run_diagnostic("arm_counts", source, {"columns": ["arm"]})
        run_diagnostic("arm_counts", source, {"columns": ["arm"]})
        assert source.reads == 2
        assert set(dir(source)) & {"write_csv", "sink_csv", "write_parquet"} == set()
        objects = FakeObjects()
        stored = CsvObjectFrameSource(objects, f"objects/{HASH}", REF)
        assert stored.frame().equals(BytesFrameSource("t.csv", CSV, REF).frame())
        assert (objects.locators, stored.csv_ref()) == ([f"objects/{HASH}"], REF)


class TestHandlers:
    def test_preflight_returns_one_serialized_result(self) -> None:
        bound = handlers(RecordingSource())
        result = bound["run_preflight_diagnostic"](
            ENVELOPE, {"diagnostic_id": "arm_counts", "params": {"columns": ["arm"]},
                       "method_id": "randomized_experiment"})
        assert result["status"] == "computed"
        assert result["values"]["count:treated"] == 4
        assert result["csv_artifact"]["artifact_id"] == "csv-1"

    def test_preflight_enforces_the_pack_allowed_list_and_the_spec_table(self) -> None:
        bound = handlers(RecordingSource())
        with pytest.raises(DiagnosticError) as outside_pack:
            bound["run_preflight_diagnostic"](
                ENVELOPE, {"diagnostic_id": "arm_counts", "method_id": "did"})
        assert outside_pack.value.code == "diagnostic_not_allowed"
        with pytest.raises(DiagnosticError) as unlisted:
            bound["run_preflight_diagnostic"](ENVELOPE, {"diagnostic_id": "invented"})
        assert unlisted.value.code == "unknown_diagnostic"

    def test_eligibility_preview_counts_kept_and_excluded_rows(self) -> None:
        bound = handlers(RecordingSource())
        result = bound["preview_eligibility_impact"](ENVELOPE, {"rules": [
            {"column": "outcome", "op": "not_null"},
            {"column": "running", "op": "ge", "value": 4.0}], "by": "arm"})
        assert (result["total_rows"], result["kept_rows"], result["excluded_rows"]) == (8, 5, 3)
        assert result["kept_by"] == {"control": 2, "treated": 3}
        assert result["excluded_by"] == {"control": 2, "treated": 1}

    def test_eligibility_preview_supports_the_closed_grammar(self) -> None:
        bound = handlers(RecordingSource())
        for rule, kept in (
            ({"column": "arm", "op": "in", "value": ["treated"]}, 4),
            ({"column": "arm", "op": "eq", "value": "control"}, 4),
            ({"column": "arm", "op": "ne", "value": "control"}, 4),
            ({"column": "running", "op": "le", "value": 4.0}, 3),
        ):
            result = bound["preview_eligibility_impact"](ENVELOPE, {"rules": [rule]})
            assert result["kept_rows"] == kept

    def test_eligibility_preview_fails_closed_on_unknown_columns_and_ops(self) -> None:
        bound = handlers(RecordingSource())
        with pytest.raises(DiagnosticError) as unknown_column:
            bound["preview_eligibility_impact"](
                ENVELOPE, {"rules": [{"column": "nope", "op": "not_null"}]})
        assert unknown_column.value.code == "unknown_column"
        with pytest.raises(DiagnosticError) as unknown_op:
            bound["preview_eligibility_impact"](
                ENVELOPE, {"rules": [{"column": "arm", "op": "matches", "value": "x"}]})
        assert unknown_op.value.code == "unsupported_rule"
