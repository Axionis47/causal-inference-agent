"""The six §24.3 walls over the declarative rows: table-driven pass and failure (T-018 §1.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from causal.preparation import contracts as ct
from causal.preparation import impact as im
from causal.preparation import plans as pl
from causal.preparation import validators as vl
from causal.preparation.diagnostics import load_preparation_diagnostics
from causal.preparation.plancompile import FrameGapV1
from causal.shared import contracts as sc
from causal.shared import receipts as rc
from causal.shared.registry import RegistryError
from causal.shared.validation import ValidationRuleV1
from tests.preparation.test_contracts import DIAGNOSTIC, HASH, OTHER_HASH, RECORD, REF
from tests.preparation.test_plancompile import AIPW, CONTEXT, OPERATIONS, REGISTRIES, item

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
RULES = vl.load_validation_rules(REGISTRIES / "preparation-validation-rules.v1.json")
DIAGNOSTIC_ROWS = load_preparation_diagnostics(REGISTRIES / "preparation-diagnostics.v1.json")
STABILIZATION = ct.StabilizationRecordV1(**RECORD)
SOURCE = sc.ArtifactRef(artifact_id="frame-0", content_hash=HASH)
OUTPUT = sc.ArtifactRef(artifact_id="frame-1", content_hash=OTHER_HASH)
READABLE = dict.fromkeys(tuple(next(
    rule for rule in RULES if rule.rule_id == "prep.final.handoff_readable"
).params["conditions"]), True)
RUNNABLE = im.MethodStructureResultV1(verdict=im.StructureVerdict.RUNNABLE, codes=())
LOST = im.MethodStructureResultV1(verdict=im.StructureVerdict.NOT_RUNNABLE,
                                  codes=("below_minimum_rows",))


def diagnostic(status: ct.DiagnosticStatus) -> ct.PreparationDiagnosticV1:
    return ct.PreparationDiagnosticV1(**{**DIAGNOSTIC, "status": status})


def executed(after: str = HASH, source: sc.ArtifactRef = SOURCE,
             status: ct.DiagnosticStatus = ct.DiagnosticStatus.PASS) -> tuple[vl.ReceiptCheck, ...]:
    """One receipt over (source → output) as the §17.6 gate sees it."""
    shape = rc.FrameShapeV1(row_count=100, column_count=3)
    receipt = rc.ExecutionReceiptV1(
        stage_run_id="run-1", plan_artifact_id="plan-1", plan_item_id="i-1", parameters_hash=HASH,
        operation_id="type_conversion", operation_version="v1", implementation_version="v1",
        input_ref=source, output_ref=OUTPUT, shape_before=shape, shape_after=shape,
        row_set_hash_before=HASH, row_set_hash_after=after, examined_count=100, changed_count=1,
        derived_count=0, imputed_count=0, warning_codes=(), error_codes=(), attempt_id="a-1",
        idempotency_key="k-1", status=rc.ReceiptStatus.SUCCEEDED, started_at_utc=NOW,
        finished_at_utc=NOW)
    return (vl.ReceiptCheck(receipt, OUTPUT, diagnostic(status)),)


def plan(*items: pl.PlanItemV1) -> pl.PreparationPlanV1:
    return pl.PreparationPlanV1(
        plan_revision=1, phase=pl.PlanPhase.PREPARATION, items=items, eligibility_rule_ids=(),
        unusable_row_rule_ids=(), recipes=(), groups=(), context_manifest=REF,
        stabilized_frame=None, versions={})


ROWS_OK: dict[str, Any] = {"record": STABILIZATION, "manifest": CONTEXT,
                           "evaluated_rule_ids": ("target_population_filter", "corrupt_record")}
FREEZE_OK: dict[str, Any] = {**ROWS_OK, "structure": RUNNABLE, "row_set_hash": HASH}
PLAN_OK: dict[str, Any] = {
    "manifest": CONTEXT, "pack": AIPW, "operations": OPERATIONS,
    "plan": plan(item("i-1", "type_conversion", ("age",), output="age_num")),
    "gaps": (FrameGapV1(gap_code="type_mismatch", column="age"),)}
PLAN_BAD: dict[str, Any] = {
    **PLAN_OK, "gaps": (FrameGapV1(gap_code="type_mismatch", column="weight"),),
    "plan": plan(item("i-1", "category_normalization", ("treat",)),
                 item("i-2", "numeric_median_imputation", ("age",),
                      phase=pl.ItemPhase.IMPUTATION),
                 item("i-3", "invented", ("signup",)))}
FINAL_OK: dict[str, Any] = {
    "diagnostics": (diagnostic(ct.DiagnosticStatus.PASS),), "readability": READABLE,
    "required_diagnostic_ids": ("missingness",), "diagnostic_registry": DIAGNOSTIC_ROWS,
    "implemented_diagnostic_ids": frozenset({"missingness"}),
    "required_schema": {"age": "Float64"}, "prepared_schema": {"age": "Float64"}}
EXECUTION_OK: dict[str, Any] = {"source_ref": SOURCE, "row_set_hash": HASH,
                                "receipts": executed()}


@pytest.mark.parametrize(("number", "extra", "expected"), [
    (1, {"handoff_accepted": True}, set()),
    (1, {"entry_codes": ("csv_hash_mismatch",)},
     {"handoff_not_accepted", "entry_validation_failed"}),
    (2, ROWS_OK, set()),
    (2, {**ROWS_OK, "evaluated_rule_ids": ("invented_rule",),
         "record": ct.StabilizationRecordV1(**{
             **RECORD, "source_row_index": RECORD["source_row_index"].model_copy(
                 update={"row_count": 106})})},
     {"unapproved_rule_evaluated", "row_disposition_incomplete"}),
    (3, FREEZE_OK, set()),
    (3, {**FREEZE_OK, "structure": LOST, "row_set_hash": OTHER_HASH, "manifest": CONTEXT
         .model_copy(update={"deletion_impact_dimensions": ("overall", "subgroup")})},
     {"method_support_lost", "row_set_hash_unstable", "impact_counts_unreconciled"}),
    (4, PLAN_OK, set()),
    (4, PLAN_BAD, {"unregistered_operation", "operation_not_permitted", "gap_not_covered",
                   "protected_role_target", "illegal_fit_scope"}),
    (5, EXECUTION_OK, set()),
    (5, {**EXECUTION_OK, "receipts": executed(OTHER_HASH, OUTPUT, ct.DiagnosticStatus.FAIL)},
     {"tri_agreement_failed", "lineage_chain_broken", "row_set_not_invariant"}),
    (6, FINAL_OK, set()),
    (6, {**FINAL_OK, "implemented_diagnostic_ids": frozenset(), "prepared_schema": {"age": "S"},
         "readability": {**READABLE, "receipts_complete": False}},
     {"required_diagnostic_unhandled", "estimator_schema_unmet", "handoff_not_readable"}),
    # A registered-but-unimplemented method diagnostic passes only as approved `not_computable`.
    (6, {**FINAL_OK, "implemented_diagnostic_ids": frozenset(),
         "approved_handling": frozenset({"not_computable"}),
         "diagnostics": (diagnostic(ct.DiagnosticStatus.NOT_COMPUTABLE),)}, set()),
])
def test_each_wall_passes_and_fails_on_its_own_registered_rows(
    number: int, extra: dict[str, Any], expected: set[str]
) -> None:
    report = vl.wall(number, vl.WallContext(rules=RULES, **extra))
    assert {issue.code for issue in report.issues} == expected
    assert not any(issue.user_resolvable for issue in report.issues)  # PRD-003 never asks


def test_an_unknown_check_or_wall_fails_closed() -> None:
    rogue = ValidationRuleV1(rule_id="prep.entry.rogue", wall=1, kind="entry", code="rogue",
                             params={"check": "guess"}, allowed_actions=("revise_field",),
                             user_resolvable=False)
    for number, rules, expected in ((1, (rogue,), vl.UNKNOWN_RULE_CHECK),
                                    (vl.MAX_WALL + 1, RULES, vl.UNKNOWN_WALL)):
        with pytest.raises(RegistryError) as error:
            vl.wall(number, vl.WallContext(rules=rules, handoff_accepted=True))
        assert error.value.code == expected


def test_the_first_failing_wall_stops_the_run_and_is_never_waived() -> None:
    passing = {**FREEZE_OK, **PLAN_OK, **FINAL_OK, **EXECUTION_OK, "handoff_accepted": True}
    lost = vl.validate(vl.MAX_WALL, vl.WallContext(rules=RULES, **{**passing, "structure": LOST}))
    assert (lost.wall, lost.issues[0].code) == (3, "method_support_lost")
    clean = vl.validate(vl.MAX_WALL, vl.WallContext(rules=RULES, **passing))
    assert (clean.wall, clean.passed) == (vl.MAX_WALL, True)
    # A later wall's failure never reaches the report while an earlier wall is still open.
    early = vl.WallContext(rules=RULES, **{**passing, **PLAN_BAD, "handoff_accepted": False})
    assert vl.validate(vl.MAX_WALL, early).wall == 1
