"""Row identity and the ordered disposition engine (T-016 §1.1; PRD-003 §8, §9, §24.4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causal.preparation.contracts import (
    DiagnosticStatus,
    PreparationContextManifestV1,
    RowDisposition,
    StabilizationRecordV1,
)
from causal.preparation.impact import DimensionKind, DimensionSpecV1, dimension_impact
from causal.preparation.packs import (
    PreparationPackV1,
    eligibility_vocabulary,
    load_preparation_packs,
)
from causal.preparation.stabilize import (
    EXACT_DUPLICATE_RECORD,
    REQUIRED_ROLE_MISSING,
    UNREGISTERED_RULE,
    UNREPRESENTABLE_CELL,
    ColumnParseSpecV1,
    DuplicatePolicyV1,
    RequiredRoleRuleV1,
    RowRuleV1,
    RuleEvaluator,
    StabilizationError,
    StabilizationResult,
    parse_source_csv,
    row_content_hash,
    stabilization_summaries,
    stabilize,
)
from tests.conftest import MemoryObjects
from tests.preparation.test_contracts import MANIFEST, REF

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS_PATH = REGISTRIES / "method-packs.v1.json"
AIPW = load_preparation_packs(REGISTRIES / "method-pack-preparation.v1.json", PACKS_PATH).get(
    "aipw", "aipw-pack.v1")
VOCABULARY = eligibility_vocabulary(PACKS_PATH, "aipw")
CSV_HASH = "c" * 64

COLUMNS = (
    ColumnParseSpecV1(column_name="unit_id", dtype="String", parse_critical=True),
    ColumnParseSpecV1(column_name="treat", dtype="Int64", parse_critical=True),
    ColumnParseSpecV1(column_name="y", dtype="Float64", parse_critical=False),
    ColumnParseSpecV1(column_name="signup", dtype="Date", parse_critical=False),
)
HEADER = "unit_id,treat,y,signup\n"
CONTEXT = PreparationContextManifestV1(**{
    **MANIFEST, "key_columns": ("unit_id",),
    "column_roles": {"unit_id": "unit_identifier", "treat": "treatment", "y": "outcome"},
    "unusable_row_rule_ids": ("corrupt_record", "missing_observed_treatment"),
})
POPULATION = RowRuleV1(
    rule_id="target_population_filter", rule_kind="target_population_filter",
    evaluator=RuleEvaluator.NUMERIC_RANGE, column="y", minimum=0.0,
    disposition=RowDisposition.NOT_ELIGIBLE_POPULATION)
WINDOW = RowRuleV1(
    rule_id="observation_window", rule_kind="observation_window",
    evaluator=RuleEvaluator.TIMEFRAME_WINDOW, column="signup", minimum="2020-01-01",
    disposition=RowDisposition.NOT_ELIGIBLE_TIMEFRAME)
NO_DUPLICATES = DuplicatePolicyV1()
TREATMENT_OBSERVED = RequiredRoleRuleV1(
    role="treatment", column="treat", observed_rule_id="treatment_observed",
    disposition_rule_id="missing_observed_treatment")


def run(body: str, *, rules: tuple[RowRuleV1, ...] = (POPULATION, WINDOW),
        role_rules: tuple[RequiredRoleRuleV1, ...] = (TREATMENT_OBSERVED,),
        duplicates: DuplicatePolicyV1 = NO_DUPLICATES,
        manifest: PreparationContextManifestV1 = CONTEXT,
        pack: PreparationPackV1 = AIPW) -> StabilizationResult:
    """Parse one CSV body under the pinned profile and run the §9.1 engine over it."""
    parsed = parse_source_csv((HEADER + body).encode(), COLUMNS)
    return stabilize(parsed, csv_hash=CSV_HASH, manifest=manifest, pack=pack, rules=rules,
                     role_rules=role_rules, duplicates=duplicates, vocabulary=VOCABULARY)


CELL = st.none() | st.text(alphabet="ab", max_size=3)
ROWS = st.lists(st.tuples(st.text(alphabet="uv", min_size=1, max_size=2), st.integers(0, 1),
                          st.integers(0, 9)), min_size=1, max_size=6)


def body_of(rows: Sequence[tuple[str, int, int]]) -> str:
    return "".join(f"{unit},{treat},{y},2020-06-01\n" for unit, treat, y in rows)


@given(ROWS)
@settings(max_examples=25, deadline=None)
def test_identical_bytes_give_identical_identities_and_row_set_hash(
    rows: Sequence[tuple[str, int, int]]
) -> None:
    body = body_of(rows)
    first, second = run(body), run(body)
    assert [row.source_row_id for row in first.rows] == [row.source_row_id for row in second.rows]
    units = sum(first.retained_mask())
    freeze = stabilization_summaries(MemoryObjects(), parse_source_csv(
        (HEADER + body).encode(), COLUMNS), first, (POPULATION,), units)[3]
    other = stabilization_summaries(MemoryObjects(), parse_source_csv(
        (HEADER + body).encode(), COLUMNS), second, (POPULATION,), units)[3]
    assert freeze.row_set_hash == other.row_set_hash


@given(st.lists(CELL, min_size=1, max_size=4), st.lists(CELL, min_size=1, max_size=4))
@settings(max_examples=50, deadline=None)
def test_any_value_change_changes_the_row_content_hash(
    left: list[str | None], right: list[str | None]
) -> None:
    assert (row_content_hash(left) == row_content_hash(right)) is (left == right)


@given(ROWS)
@settings(max_examples=25, deadline=None)
def test_every_row_carries_exactly_one_primary_disposition(
    rows: Sequence[tuple[str, int, int]]
) -> None:
    result = run(body_of(rows))
    assert sum(count.row_count for count in result.counts()) == len(result.rows)
    assert len({row.row_number for row in result.rows}) == len(rows)


def test_identical_duplicate_rows_stay_distinct() -> None:
    result = run("u1,1,5,2020-06-01\nu1,1,5,2020-06-01\n")
    assert result.rows[0].row_content_hash == result.rows[1].row_content_hash
    assert result.rows[0].source_row_id != result.rows[1].source_row_id
    assert [row.disposition for row in result.rows] == [RowDisposition.RETAINED] * 2
    assert result.rows[1].warning_codes == (EXACT_DUPLICATE_RECORD,)


ORDERED = (
    ("u1,1,5,2020-06-01\n", RowDisposition.RETAINED, ()),
    ("u2,x,5,2020-06-01\n", RowDisposition.UNUSABLE_CORRUPT_RECORD, (REQUIRED_ROLE_MISSING,)),
    ("u3,1,-1,2019-06-01\n", RowDisposition.NOT_ELIGIBLE_POPULATION, ("observation_window",)),
    (",1,5,2020-06-01\n", RowDisposition.UNUSABLE_REQUIRED_IDENTITY, ()),
    ("u5,,5,2020-06-01\n", RowDisposition.UNUSABLE_REQUIRED_ROLE, ()),
    ("u6,1,5,2019-06-01\n", RowDisposition.NOT_ELIGIBLE_TIMEFRAME, ()),
    ("u7,1,z,2020-06-01\n", RowDisposition.NOT_ELIGIBLE_POPULATION, ()),
)


@pytest.mark.parametrize(("body", "disposition", "warnings"), ORDERED)
def test_the_first_applicable_rule_is_primary_and_later_hits_are_warnings(
    body: str, disposition: RowDisposition, warnings: tuple[str, ...]
) -> None:
    row = run(body).rows[0]
    assert row.disposition is disposition
    assert set(warnings) <= set(row.warning_codes)
    assert disposition.value not in row.warning_codes


def test_an_optional_cell_the_parser_cannot_represent_is_a_warning_not_a_corrupt_record() -> None:
    parsed = parse_source_csv((HEADER + "u1,1,z,2020-06-01\n").encode(), COLUMNS)
    assert parsed.corrupt_rows == frozenset()
    assert parsed.parse_warning_counts == {UNREPRESENTABLE_CELL: 1}


def test_a_source_the_pinned_parser_cannot_represent_fails_closed() -> None:
    with pytest.raises(StabilizationError):
        parse_source_csv((HEADER + "u1,1,5,2020-06-01,extra\n").encode(), COLUMNS)


@pytest.mark.parametrize(("field", "value"), [("rule_kind", "invented_kind"),
                                              ("rule_id", "invented_rule")])
def test_an_unregistered_rule_raises_instead_of_being_skipped(field: str, value: str) -> None:
    with pytest.raises(StabilizationError) as error:
        run("u1,1,5,2020-06-01\n", rules=(POPULATION.model_copy(update={field: value}),))
    assert error.value.code == UNREGISTERED_RULE


ROLE_CASES = (
    (TREATMENT_OBSERVED, RowDisposition.UNUSABLE_REQUIRED_ROLE),
    (TREATMENT_OBSERVED.model_copy(update={"disposition_rule_id": None}),
     RowDisposition.UNRESOLVED_CONFLICT),
    (TREATMENT_OBSERVED.model_copy(update={"disposition_rule_id": "propensity_trimming"}),
     RowDisposition.UNRESOLVED_CONFLICT),
    (TREATMENT_OBSERVED.model_copy(update={"observed_rule_id": "not_registered"}),
     RowDisposition.RETAINED_WITH_MISSINGNESS),
    (RequiredRoleRuleV1(role="outcome", column="y", observed_rule_id="outcome_observed"),
     RowDisposition.UNRESOLVED_CONFLICT),
)


@pytest.mark.parametrize(("rule", "disposition"), ROLE_CASES)
def test_the_four_condition_test_decides_a_missing_required_value(
    rule: RequiredRoleRuleV1, disposition: RowDisposition
) -> None:
    outcome = rule.role == "outcome"
    body = "u1,1,,2020-06-01\n" if outcome else "u1,,5,2020-06-01\n"
    rules = (WINDOW,) if outcome else (POPULATION, WINDOW)
    assert run(body, rules=rules, role_rules=(rule,)).rows[0].disposition is disposition


def test_a_role_rule_naming_a_column_without_that_role_fails_closed() -> None:
    stray = RequiredRoleRuleV1(role="cluster", column="treat", observed_rule_id="treatment_observed")
    with pytest.raises(StabilizationError):
        run("u1,1,5,2020-06-01\n", role_rules=(stray,))


DUPLICATES = (
    (NO_DUPLICATES, "u1,1,5,2020-06-01\nu1,1,6,2020-06-01\n",
     RowDisposition.UNRESOLVED_CONFLICT, RowDisposition.UNRESOLVED_CONFLICT),
    (DuplicatePolicyV1(conflict_resolution_rule_id="unit_grain_collision"),
     "u1,1,5,2020-06-01\nu1,1,6,2020-06-01\n",
     RowDisposition.RETAINED, RowDisposition.UNUSABLE_GRAIN_VIOLATION),
    (DuplicatePolicyV1(exact_duplicate_rule_id="unit_grain_collision"),
     "u1,1,5,2020-06-01\nu1,1,5,2020-06-01\n",
     RowDisposition.RETAINED, RowDisposition.UNUSABLE_GRAIN_VIOLATION),
)


@pytest.mark.parametrize(("policy", "body", "first", "second"), DUPLICATES)
def test_duplicate_and_collision_rules(
    policy: DuplicatePolicyV1, body: str, first: RowDisposition, second: RowDisposition
) -> None:
    rows = run(body, duplicates=policy).rows
    assert (rows[0].disposition, rows[1].disposition) == (first, second)
    assert rows[1].rule_id == (policy.exact_duplicate_rule_id or
                               policy.conflict_resolution_rule_id)


def test_the_summaries_assemble_a_stabilization_record() -> None:
    objects = MemoryObjects()
    body = "u1,1,5,2020-06-01\nu2,0,-1,2020-06-01\nu3,,5,2020-06-01\n"
    parsed = parse_source_csv((HEADER + body).encode(), COLUMNS)
    result = run(body)
    index, eligibility, ledger, freeze = stabilization_summaries(
        objects, parsed, result, (POPULATION, WINDOW), 1)
    impact = dimension_impact(parsed.frame, result.retained_mask(), (
        DimensionSpecV1(dimension_id="overall", kind=DimensionKind.OVERALL),
        DimensionSpecV1(dimension_id="treatment_group", kind=DimensionKind.COLUMN_LEVELS,
                        column="treat")))
    record = StabilizationRecordV1(
        context_manifest=REF, source_row_index=index, eligibility=eligibility,
        dispositions=ledger, impact=impact, method_structure_status=DiagnosticStatus.PASS,
        method_structure_codes=(), freeze=freeze, pre_stabilization_diagnostics=(),
        post_stabilization_diagnostics=(), versions={"preparation": "t-016"})
    assert record.freeze.retained_row_count == 1
    assert eligibility.rule_counts == {"target_population_filter": 1, "observation_window": 0}
    assert objects.data[freeze.retained_row_object.object_locator].endswith(b'"]}')
    assert index.index_object.content_hash != ledger.ledger_object.content_hash


def test_a_frozen_row_set_hash_moves_when_a_retained_row_leaves() -> None:
    objects: Any = MemoryObjects()
    kept = "u1,1,5,2020-06-01\nu2,1,6,2020-06-01\n"
    lost = "u1,1,5,2020-06-01\nu2,1,-6,2020-06-01\n"
    hashes = {
        stabilization_summaries(objects, parse_source_csv((HEADER + body).encode(), COLUMNS),
                                run(body), (POPULATION,), 1)[3].row_set_hash
        for body in (kept, lost)
    }
    assert len(hashes) == 2
