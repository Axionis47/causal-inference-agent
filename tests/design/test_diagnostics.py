"""Read-only pre-repair diagnostics (T-012 §4)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from causal.design import v2
from causal.design.contracts import DiagnosticResultV1, DiagnosticStatus
from causal.design.diagnostics import (
    DIAGNOSTIC_ASSESSMENT_MISMATCH,
    DIAGNOSTIC_NOT_ALLOWED,
    DIAGNOSTIC_SPECS,
    DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED,
    IMPLEMENTATION_VERSION,
    BytesFrameSource,
    CsvObjectFrameSource,
    DiagnosticError,
    diagnostic_result_id,
    run_diagnostic,
    run_requested_diagnostics,
)
from causal.design.packs import PREREPAIR_DIAGNOSTIC_IDS, load_method_packs
from causal.shared.contracts import ArtifactRef

ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64
REF = ArtifactRef(artifact_id="csv-1", content_hash=HASH)
PACKS = load_method_packs(ROOT / "registries" / "method-packs.v1.json")
# The golden frame: one null cluster, two missing outcomes, one unparsable dose, a repeated key.
CSV = (
    b"unit_id,arm,cluster,period,outcome,score,running,dose\n"
    b"u1,control,c1,2020,1.0,10,4.5,10\nu2,control,c1,2020,,11,4.0,na\n"
    b"u3,treated,c2,2020,3.0,12,5.5,12\nu4,treated,c2,2021,4.0,,6.0,13\n"
    b"u5,control,c1,2021,5.0,10,4.0,10\nu6,treated,,2021,,13,5.0,10\n"
    b"u7,treated,c2,2021,7.0,14,5.0,14\nu8,control,c1,2020,8.0,10,3.0,10\n"
)

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


ARM_RESULT = run("arm_counts", columns=["arm"])
ARM_RESULT_ID = diagnostic_result_id(ARM_RESULT)


PLAN = v2.DiagnosticPlanV2(selected_csv=REF, candidate_method_id="randomized_experiment", issues=(),
    items=(v2.DiagnosticPlanItemV2(diagnostic_id="arm_counts", primitive="count_by", required_for_eligibility=True,
        inputs=(v2.BoundDiagnosticInputV2(parameter="columns", source_kind="role", source_id="treatment", columns=("arm",)),)),))


def decision(requested: tuple[str, ...], assessed: tuple[str, ...] = ()) -> v2.AgentDesignProposalV2:
    return v2.AgentDesignProposalV2.model_construct(requested_diagnostic_ids=requested,
        diagnostic_assessments=tuple(v2.DiagnosticAssessmentV2(
            diagnostic_result_id=name, judgment="not_decisive") for name in assessed))


def test_model_diagnostic_request_runs_only_the_compiler_bound_read_only_plan() -> None:
    assert (result := run_requested_diagnostics(decision(("arm_counts",)), PLAN, RecordingSource()))[0].diagnostic_id == "arm_counts" and result[0].values["count:treated"] == 4


@pytest.mark.parametrize(("requested", "assessed", "observed", "code"), [
    (("not_registered",), (), (), DIAGNOSTIC_NOT_ALLOWED),
    (("arm_counts", "arm_counts"), (), (), DIAGNOSTIC_ASSESSMENT_MISMATCH),
    ((), (), (ARM_RESULT,), DIAGNOSTIC_ASSESSMENT_MISMATCH),
    (("a", "b", "c", "d"), (ARM_RESULT_ID,), (ARM_RESULT,),
     DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED)])
def test_model_diagnostic_turn_fails_closed(requested: tuple[str, ...], assessed: tuple[str, ...], observed: tuple[DiagnosticResultV1, ...], code: str) -> None:
    with pytest.raises(DiagnosticError) as error:
        run_requested_diagnostics(decision(requested, assessed), PLAN, RecordingSource(), observed)
    assert error.value.code == code


class TestPrimitives:
    def test_aliased_group_and_assignment_roles_count_one_actual_column(self) -> None:
        repeated = run("adoption_cohorts", columns=["arm", "arm"])
        expected = run("adoption_cohorts", columns=["arm"])
        assert repeated == expected
        assert repeated.values["expected_cells"] == repeated.values["group_count"] == 2
        assert repeated.used_rows == 8 and repeated.columns_read == ("arm",)

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
        result = run("cutoff_side_counts", running_column="running", target="arm", cutoff=5.0)
        assert result.values["below_count"] == 4
        assert result.values["at_or_above_count"] == 4
        assert result.values["assignment_direction"] == "above"
        assert result.values["contradiction_count"] == 0
        assert result.columns_read == ("arm", "running")

    def test_zero_cutoff_is_computable(self) -> None:
        result = run("cutoff_side_counts", running_column="running", target="arm", cutoff=0.0)
        assert result.status is DiagnosticStatus.COMPUTED
        assert result.values["at_or_above_count"] == 8

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
        assert set(DIAGNOSTIC_SPECS) == set(PREREPAIR_DIAGNOSTIC_IDS) and all(set(pack.allowed_prerepair_diagnostic_ids) <= set(DIAGNOSTIC_SPECS) for pack in PACKS.all())


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
