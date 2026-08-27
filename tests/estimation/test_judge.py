# The §16 claim judgment: bounded context assembly, the ONE claim-review call and its two
# targeted corrections, the deterministic §16.3 validator behind wall 13, and the
# not_estimable path that never builds a model call (T-025 §2; PRD-004 §16, §20.3).

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from causal.estimation import contracts as ec
from causal.estimation import judge as ej
from causal.estimation import walls as ew
from causal.shared import persistence
from causal.shared.agenttask import TaskRunner
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.gateway import GatewayResultV1
from causal.shared.registry import load_artifact_type_registry
from causal.shared.validation import ValidationReport
from tests.estimation.test_plancompile import REGISTRIES, ref
from tests.estimation.test_walls import CEILING, MANIFEST, PLAN, context, diagnostic, sensitivity
from tests.estimation.test_walls import result as primary

REPO = REGISTRIES.parent
REGISTRY = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
NOW = datetime(2026, 8, 26, tzinfo=UTC)
CONTRAST = PLAN.contrast_ids[0]
DIAGNOSTICS = tuple(diagnostic(name) for name in PLAN.required_diagnostics)
SENSITIVITIES = tuple(sensitivity(name) for name in PLAN.required_sensitivity_ids)
# Evidence id to its committed artifact id; the value set is the closed citation allowlist.
REFS = {"plan": "art:plan", "ceiling": "art:ceiling", "result": "art:result"} | {
    row.diagnostic_id: f"art:{row.diagnostic_id}" for row in DIAGNOSTICS} | {
    row.branch_id: f"art:{row.branch_id}" for row in SENSITIVITIES}
LINEAGE: dict[str, Any] = {
    "parents": [ref("plan").model_dump(mode="json")],
    "judgment_ceiling": ref("ceiling").model_dump(mode="json"),
    "primary_result": ref("result").model_dump(mode="json"),
    "supporting_refs": [ref("ceiling").model_dump(mode="json")]}
# The same ceiling one notch down, and the ceiling a run with no primary result reaches.
CAPPED = CEILING.model_copy(update={
    "items": (CEILING.items[0].model_copy(update={"ceiling": "reportable_with_qualifications"}),),
    "overall_ceiling": "reportable_with_qualifications"})
NO_RESULT = CEILING.model_copy(update={
    "items": (CEILING.items[0].model_copy(update={"ceiling": "not_estimable"}),),
    "overall_ceiling": "not_estimable", "primary_result": None})
# A frozen upstream design payload carrying exactly what §16.2 forbids beside what it allows.
FORBIDDEN = ("raw_rows", "dataframe", "predictions", "residuals", "weights", "figure_data",
             "credentials")
POISONED: dict[str, Any] = {
    "causal_question": "does the offer raise completion?", "assumptions": ["random assignment"],
    "alternative_graphs": ["selection into wave two"], "unresolved_uncertainty": [],
    "raw_rows": [{"uid": 1, "y": 1.0}], "dataframe": "s3://bucket/frame.parquet",
    "predictions": [0.4, 0.6], "residuals": [0.1], "weights": [1.0, 2.0],
    "figure_data": {"points": [{"x": 1, "y": 2}]}, "credentials": "token"}
# §16.3's mapping table, transcribed from the PRD row by row.
MAPPING = (("reportable", "complete"), ("reportable_with_qualifications", "complete"),
           ("not_reportable", "invalidated"), ("not_estimable", "not_estimable"),
           ("failed", "failed"))


def review(row: Any = ..., ceiling: Any = CEILING) -> ej.ClaimReviewContextV1:
    return ej.review_context(MANIFEST, ceiling, primary() if row is ... else row, POISONED,
                             DIAGNOSTICS, SENSITIVITIES, refs=REFS,
                             population_summary={"row": 100, "contributing": 90})


def draft(item: Any = None, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "contrast_id": CONTRAST, "status": "reportable", "ceiling": "reportable",
        "estimate_units": "proportion", "population_id": MANIFEST.population_id,
        "comparator_id": "control", "timeframe_id": MANIFEST.timeframe_id, "estimate": 0.2,
        "effect_statement": "the offer raised completion by 20 percentage points",
        "interval_lower": 0.1, "interval_upper": 0.3, "confidence_level": 0.95,
        "cited_artifact_ids": [REFS["result"]]} | dict(item or {})
    return {"causal_question": POISONED["causal_question"], "estimand_id": MANIFEST.estimand_id,
            "status": "reportable", "overall_ceiling": "reportable", "items": [row],
            "cited_artifact_ids": [REFS["ceiling"]]} | over


def qualified(**over: Any) -> dict[str, Any]:
    # The same draft rewritten to sit at the capped ceiling, as a correction must.
    return draft(item={"status": "reportable_with_qualifications",
                       "ceiling": "reportable_with_qualifications"}) | {
        "status": "reportable_with_qualifications",
        "overall_ceiling": "reportable_with_qualifications"} | over


class Gateway:
    # Canned `AgentTaskResultV1` bodies; every construction is counted, so "zero calls" is
    # an assertion about this list and not about a mock's internals.

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[AgentTaskEnvelopeV1] = []
        self.prompts: list[str] = []

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        self.prompts.append(prompt)
        body = {"envelope_id": envelope.envelope_id, "schema_version": "agent-task-result.v1",
                "task_id": envelope.task_id, "status": "complete",
                "artifact_type": ej.ARTIFACT_TYPE, "artifact_schema_version": ej.SCHEMA_VERSION,
                "parent_artifact_ids": [], "payload": self.payloads.pop(0), "claims": [],
                "missing_requirements": [], "conflicts": [], "warnings": [], "evidence_ids": [],
                "tool_receipts": [], "output_hash": None, "validation_target": "v"}
        return GatewayResultV1(text=json.dumps(body), parsed=body, token_usage={}, attempts=1,
                               seed=1)


class Harness:
    # The injected hooks a real estimation coordinator supplies to the shared TaskRunner.

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.gateway = Gateway(payloads)
        self.committed: list[tuple[str, dict[str, Any]]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.exhausted_issues: list[Any] = []
        self.state: dict[str, Any] = {
            "analysis_id": "an-1", "stage_run_id": "es:an-1:1", "design_revision": 1,
            "corrections": {}, "open_requirement_ids": []}
        self.runner = TaskRunner(
            gateway=self.gateway, tasks={ej.TASK_KIND: ej.CLAIM_TASK}, tools={ej.TASK_KIND: ()},
            evals={ej.TASK_KIND: ("EV-P4-008",)}, prompts_root=REPO,
            envelope=ej.build_claim_envelope, prompt=ej.render_claim_prompt,
            validate=self.never, context=lambda state: None,
            evidence=lambda state: frozenset(REFS.values()),
            manifest=lambda state: ref("context_manifest"), parents=lambda state, *kinds: (),
            commit=self.commit, emit=self.emit, record=self.record, exhausted=self.exhausted,
            upsert=lambda *args: None)

    def never(self, *args: Any) -> ValidationReport:
        raise AssertionError("the judge must bind its own claim validator")

    def commit(self, state: Any, kind: str, payload: Any,
               parents: tuple[Any, ...]) -> ArtifactEnvelopeV1:
        self.committed.append((kind, dict(payload)))
        return persistence.build_envelope(
            REGISTRY, kind, dict(payload), analysis_id=state["analysis_id"],
            stage_run_id=state["stage_run_id"], producer_version="test", parents=parents,
            created_at_utc=NOW)

    def emit(self, state: Any, name: str, evals: Any, **over: Any) -> None:
        self.events.append((name, over))

    def record(self, *args: Any) -> None:
        self.events.append(("task.recorded", {}))

    def exhausted(self, state: Any, task_id: str, kind: str, issues: Any) -> None:
        self.exhausted_issues.append(issues)
        self.emit(state, "blocker.raised", (), error_code=ej.CORRECTION_EXHAUSTED)

    def judge(self, ctx: ew.WallContext,
              seen: ej.ClaimReviewContextV1) -> ej.ClaimJudgmentOutcome:
        return ej.judge(self.runner, self.state, seen, ctx=ctx, lineage=LINEAGE,
                        run_walls=lambda highest, walls: ew.validate(highest, walls))


def keys_of(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {found for row in node.values() for found in keys_of(row)}
    if isinstance(node, list):
        return {found for row in node for found in keys_of(row)}
    return set()


def test_a_golden_draft_commits_one_claim_judgment_under_harness_owned_lineage() -> None:
    harness = Harness([draft()])
    found = harness.judge(context(), review())
    assert found.error_code is None and found.judgment is not None
    assert found.judgment.status == "reportable" and found.judgment.outcome_status == "complete"
    assert [kind for kind, _ in harness.committed] == [ej.ARTIFACT_TYPE]
    assert len(harness.gateway.calls) == 1
    envelope = harness.gateway.calls[0]
    assert envelope.allowed_tool_ids == () and envelope.budgets.tool_call_budget == 0
    assert envelope.budgets.correction_budget == ej.CORRECTION_BUDGET
    assert envelope.allowed_stopping_states == ("complete",)
    # D-071: lineage and the five pinned versions are the harness's, never the model's echo.
    assert dict(found.judgment.versions) == ej.PINNED_VERSIONS
    assert found.judgment.judgment_ceiling.artifact_id == "ceiling"
    assert harness.committed[0][1]["schema_version"] == ej.SCHEMA_VERSION


def test_the_prompt_hydrates_the_closed_vocabularies_the_model_must_echo() -> None:
    harness = Harness([draft()])
    harness.judge(context(), review())
    prompt = harness.gateway.prompts[0]
    for marker in ("## allowed_evidence", "## parent_artifacts", "## contrasts",
                   "## allowed_artifact_ids", "## overall_ceiling", CONTRAST, REFS["result"]):
        assert marker in prompt
    for status, _ in MAPPING:
        assert status in prompt


def test_a_ceiling_exceeding_draft_is_corrected_once_and_the_compliant_retry_is_accepted() -> None:
    harness = Harness([draft(), qualified()])
    found = harness.judge(context(ceiling=CAPPED), review(ceiling=CAPPED))
    assert found.judgment is not None and found.judgment.status == "reportable_with_qualifications"
    assert len(harness.gateway.calls) == 2
    assert harness.state["corrections"] == {f"{ej.ARTIFACT_TYPE}:claim_exceeds_ceiling": 1}
    correction = harness.gateway.calls[1].payload["correction"]
    issue = dict(correction["issues"][0])  # type: ignore[index, call-overload]
    assert issue["code"] == "claim_exceeds_ceiling" and issue["rule_id"] == "est.claim.within_ceiling"
    assert issue["artifact_ids"] == [
        f"{CONTRAST}:ceiling_restated:reportable!=reportable_with_qualifications",
        f"{CONTRAST}:status_above_ceiling:reportable>reportable_with_qualifications",
        "overall_ceiling_restated:reportable",
        "status_above_ceiling:reportable>reportable_with_qualifications"]


def test_two_corrections_exhaust_into_one_typed_failure_and_exactly_one_blocker() -> None:
    harness = Harness([draft(), draft(), draft()])
    found = harness.judge(context(ceiling=CAPPED), review(ceiling=CAPPED))
    assert found.judgment is None and found.error_code == ej.CORRECTION_EXHAUSTED
    assert ej.OUTCOME_BY_STATUS["failed"] == "failed" and harness.committed == []
    assert len(harness.gateway.calls) == 1 + ej.CORRECTION_BUDGET
    assert [name for name, _ in harness.events].count("blocker.raised") == 1
    assert len(harness.exhausted_issues) == 1
    assert harness.exhausted_issues[0][0].code == "claim_exceeds_ceiling"


def test_a_citation_outside_the_committed_set_is_rejected() -> None:
    bad = draft(item={"cited_artifact_ids": ["art:invented"]})
    harness = Harness([bad, bad, bad])
    found = harness.judge(context(), review())
    assert found.judgment is None and found.error_code == ej.CORRECTION_EXHAUSTED
    assert "uncommitted_artifact_id:art:invented" in harness.exhausted_issues[0][0].artifact_ids


def test_context_assembly_leaves_every_forbidden_class_structurally_absent() -> None:
    seen = review()
    rendered = seen.model_dump(mode="json")
    assert not keys_of(rendered) & set(FORBIDDEN)
    assert not set(ej.ClaimReviewContextV1.model_fields) & set(FORBIDDEN)
    assert set(ej.FORBIDDEN_PAYLOAD_CLASSES) == set(FORBIDDEN)
    text = json.dumps(rendered)
    for poison in ("s3://bucket/frame.parquet", "token", "0.4", "uid"):
        assert poison not in text
    # What the receives-list DOES permit still crosses, and the citation allowlist is closed.
    assert rendered["causal_question"] == POISONED["causal_question"]
    assert rendered["assumptions"] == ["random assignment"] and rendered["method_id"]
    assert seen.allowed_artifact_ids == tuple(sorted(set(REFS.values())))
    assert {row["evidence_id"] for row in rendered["evidence"]} == {
        *PLAN.required_diagnostics, *PLAN.required_sensitivity_ids}


@pytest.mark.parametrize("row", [None, primary(complete=False)])
def test_the_not_estimable_path_commits_with_zero_gateway_constructions(row: Any) -> None:
    harness = Harness([])
    found = harness.judge(context(primary_result=row, ceiling=NO_RESULT),
                          review(row=row, ceiling=NO_RESULT))
    assert harness.gateway.calls == [] and harness.gateway.prompts == []
    assert found.judgment is not None and found.error_code is None
    assert found.judgment.status == "not_estimable"
    assert found.judgment.outcome_status == "not_estimable"
    assert [kind for kind, _ in harness.committed] == [ej.ARTIFACT_TYPE]
    item = found.judgment.items[0]
    assert item.contrast_id == CONTRAST and item.status == "not_estimable"
    assert item.estimate is None and item.interval_lower is None
    assert found.judgment.cannot_conclude and found.judgment.primary_result is not None


def test_wall_thirteen_delegates_to_the_deterministic_claim_validator() -> None:
    known = frozenset(REFS.values())
    passing = ej.claim_validator(draft(), known, ())
    assert passing(context()) == ()
    assert ew.wall(13, context(claim_validator=passing)).passed
    report = ew.wall(13, context(claim_validator=ej.claim_validator(
        draft(items=[]), known, ())))
    assert not report.passed and report.issues[0].code == "claim_exceeds_ceiling"
    assert report.issues[0].artifact_ids == ("items_not_the_plan_contrasts:(none)",)


@pytest.mark.parametrize(("payload", "code"), [
    (draft(qualifications=[]), "qualification_missing:overlap:qualify.v1"),
    (draft(item={"estimate": None}), f"{CONTRAST}:reportable_without_estimate"),
    (draft(overall_ceiling="not_reportable"), "overall_ceiling_restated:not_reportable"),
    ({"causal_question": "x"}, "draft_shape_invalid")])
def test_each_substantive_defect_returns_its_own_detail_code(payload: Any, code: str) -> None:
    found = ej.claim_validator(payload, frozenset(REFS.values()),
                               ("overlap:qualify.v1",))(context())
    assert any(row.startswith(code) for row in found), found


def test_the_status_and_outcome_mapping_is_exactly_the_prd_table() -> None:
    assert ej.OUTCOME_BY_STATUS == dict(MAPPING)
    assert tuple(ej.OUTCOME_BY_STATUS) == ec.JUDGMENT_ORDER[::-1]
    assert ej.HANDOFF_READABLE == {"reportable", "reportable_with_qualifications"}
    # §16.3: model-reported confidence is never stored; the interval carries the uncertainty.
    fields = set(ej.ClaimJudgmentDraftV1.model_fields) | set(ej.ClaimItemV1.model_fields)
    assert not {name for name in fields if "confidence" in name} - {"confidence_level"}
    assert set(ej.ClaimJudgmentV1.model_fields) - set(ej.ClaimJudgmentDraftV1.model_fields) == {
        "schema_version", "judgment_ceiling", "primary_result", "outcome_status",
        "supporting_refs", "parents", "versions"}


def test_the_overall_status_is_the_most_restrictive_item_status() -> None:
    known = frozenset(REFS.values())
    found = ej.claim_validator(qualified(status="reportable"), known, ())(context(ceiling=CAPPED))
    assert "overall_status_not_most_restrictive:reportable" in found
    assert ej.claim_validator(qualified(), known, ())(context(ceiling=CAPPED)) == ()
