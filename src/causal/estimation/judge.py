# The §16 claim judgment: the bounded ClaimReviewContext, the ONE claim-review model call
# through the shared TaskRunner, the deterministic §16.3 claim validator wall 13 delegates to,
# and the deterministic not_estimable path (PRD-004 §16, §20.3; SC §5.4 PRD-004 row).
# This module never imports `causal.design` or `causal.preparation`, and it adds no gateway
# code: the frozen shared model profile reaches it through the injected TaskRunner.

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import Field, ValidationError

from causal.estimation import contracts as ec
from causal.estimation import walls as ew
from causal.shared.agenttask import TaskRunner
from causal.shared.contracts import ArtifactRef, Identity
from causal.shared.envelope import AgentTaskEnvelopeV1, TaskBudgets, TaskStatus
from causal.shared.validation import ValidationReport, collect_ids, parse_strict

TASK_KIND, SCOPE_KIND = "claim_review", "contrast"
PROMPT_PATH: Final = "prompts/estimation/claim-review.v1.txt"
ARTIFACT_TYPE, SCHEMA_VERSION = "ClaimJudgment", "claim-judgment.v1"
CLAIM_WALL, CORRECTION_BUDGET, TOKEN_BUDGET = 13, 2, 24576
CORRECTION_EXHAUSTED: Final = "correction_exhausted"
PARENT_KINDS: Final = ("EstimationPlan", "PrimaryAnalysisResult", "JudgmentCeiling")
NOT_ESTIMABLE_PARENTS: Final = ("EstimationPlan", "JudgmentCeiling")
# The five versions §16.3 pins on every judgment; the harness owns them, never the model.
PINNED_VERSIONS: Final = {
    "prompt": "estimation-claim-review-prompt.v1", "schema": SCHEMA_VERSION,
    "model_profile": "vertex-model-profile.v1", "validator": "estimation-validators.v1",
    "policy": "claim-judgment-policy.v1"}
# §16.3's mechanical map; V1 opens no partial handoff, so only these two statuses are readable.
OUTCOME_BY_STATUS: Final[dict[str, ec.EstimationOutcomeStatus]] = {
    "reportable": "complete", "reportable_with_qualifications": "complete",
    "not_reportable": "invalidated", "not_estimable": "not_estimable", "failed": "failed"}
HANDOFF_READABLE: Final = frozenset({"reportable", "reportable_with_qualifications"})
# §16.2/SC §5.4: this row's tool allowlist is empty, and no field below could carry one of these
# payload classes even if an upstream artifact held it.
FORBIDDEN_PAYLOAD_CLASSES: Final = ("raw_rows", "dataframe", "predictions", "residuals",
                                    "weights", "figure_data", "credentials")
# The only keys lifted out of the approved upstream design payload (§16.2 receives-list).
DESIGN_ALLOWLIST: Final = ("causal_question", "assumptions", "alternative_graphs",
                           "unresolved_uncertainty")


# One primary or ceiling-capped contrast, bounded to the numbers §16.2 permits.
class ClaimContrastSummaryV1(ec._Row):
    contrast_id: Identity
    ceiling: ec.JudgmentStatus
    estimate: ec.Finite | None = None
    interval_lower: ec.Finite | None = None
    interval_upper: ec.Finite | None = None
    confidence_level: ec._Level | None = None
    estimate_units: Identity | None = None
    comparator_id: Identity | None = None
    result_artifact_ids: tuple[Identity, ...] = ()


# One diagnostic or sensitivity branch: its statuses and its interpreted bounded values.
class ClaimEvidenceSummaryV1(ec._Row):
    evidence_id: Identity
    kind: ec.EvidenceKind
    execution_status: ec.ExecutionStatus
    policy_result: ec.PolicyResult
    comparison_result: Identity
    values: ec.ValueMap
    artifact_id: Identity


# Everything the §16.2 claim-review agent receives, and structurally nothing else.
class ClaimReviewContextV1(ec._Row):
    causal_question: str
    estimand_id: Identity
    method_id: Identity
    population_id: Identity
    timeframe_id: Identity
    assumptions: tuple[str, ...]
    alternative_graphs: tuple[str, ...]
    unresolved_uncertainty: tuple[str, ...]
    contrasts: tuple[ClaimContrastSummaryV1, ...]
    evidence: tuple[ClaimEvidenceSummaryV1, ...]
    population_summary: ec.CountMap
    overall_ceiling: ec.JudgmentStatus
    required_qualifications: tuple[str, ...]
    allowed_artifact_ids: tuple[Identity, ...]


# One ordered §16.3 claim item per primary contrast; no model-reported confidence is stored.
class ClaimItemV1(ec._Row):
    contrast_id: Identity
    status: ec.JudgmentStatus
    ceiling: ec.JudgmentStatus
    estimate_units: Identity
    population_id: Identity
    comparator_id: Identity
    timeframe_id: Identity
    cited_artifact_ids: Annotated[tuple[Identity, ...], Field(min_length=1)]
    effect_statement: Annotated[str, Field(max_length=600)] = ""
    estimate: ec.Finite | None = None
    interval_lower: ec.Finite | None = None
    interval_upper: ec.Finite | None = None
    confidence_level: ec._Level | None = None


# What the claim-review model returns: the §16.3 record without a single lineage fact.
class ClaimJudgmentDraftV1(ec._Row):
    causal_question: str
    estimand_id: Identity
    status: ec.JudgmentStatus
    overall_ceiling: ec.JudgmentStatus
    items: tuple[ClaimItemV1, ...]
    assumption_statements: tuple[str, ...] = ()
    diagnostic_findings: tuple[str, ...] = ()
    diagnostic_non_findings: tuple[str, ...] = ()
    sensitivity_summary: str = ""
    qualifications: tuple[str, ...] = ()
    alternative_explanations: tuple[str, ...] = ()
    unresolved_uncertainty: tuple[str, ...] = ()
    cannot_conclude: tuple[str, ...] = ()
    cited_artifact_ids: tuple[Identity, ...] = ()


# The committed artifact: the revalidated draft under lineage the harness alone owns (D-071).
class ClaimJudgmentV1(ClaimJudgmentDraftV1, ec._Lineage):
    schema_version: Literal["claim-judgment.v1"] = "claim-judgment.v1"
    judgment_ceiling: ArtifactRef
    primary_result: ArtifactRef | None
    outcome_status: ec.EstimationOutcomeStatus
    supporting_refs: tuple[ArtifactRef, ...]


# The qualifications §16.3 makes mandatory; a judgment that drops one fails wall 13.
def required_qualifications(diagnostics: Sequence[ec.DiagnosticResultV1],
                            sensitivities: Sequence[ec.SensitivityResultV1]) -> tuple[str, ...]:
    rows = {f"{row.diagnostic_id}:{row.interpreting_rule_id}" for row in diagnostics
            if row.severity == "qualification_guard" and row.policy_result != "acceptable"}
    return tuple(sorted(rows | {f"{row.branch_id}:{rule}" for row in sensitivities
                                for rule in row.qualification_rule_ids}))


# A contrast with no result of its own (§16.1 row 1) leaves the numbers absent, never guessed.
def _contrast(cap: ec.CeilingItemV1,
              row: ec.PrimaryContrastResultV1 | None) -> ClaimContrastSummaryV1:
    over: dict[str, Any] = {} if row is None else {
        "estimate": row.estimate, "interval_lower": row.interval_lower,
        "interval_upper": row.interval_upper, "confidence_level": row.confidence_level,
        "estimate_units": row.estimate_units, "comparator_id": row.comparator_id}
    return ClaimContrastSummaryV1(
        contrast_id=cap.contrast_id, ceiling=cap.ceiling,
        result_artifact_ids=tuple(ref.artifact_id for ref in cap.evidence), **over)


def review_context(manifest: ec.EstimationContextManifestV1, ceiling: ec.JudgmentCeilingV1,
                   result: ec.PrimaryAnalysisResultV1 | None, design: Mapping[str, Any],
                   diagnostics: Sequence[ec.DiagnosticResultV1],
                   sensitivities: Sequence[ec.SensitivityResultV1], *, refs: Mapping[str, str],
                   population_summary: Mapping[str, int]) -> ClaimReviewContextV1:
    # §16.2: assembled from the frozen committed artifacts through this allowlist and nothing
    # else — there is no field here that could carry a row, a prediction, or a figure payload.
    text: dict[str, Any] = {DESIGN_ALLOWLIST[0]: str(design.get(DESIGN_ALLOWLIST[0], ""))} | {
        key: tuple(str(row) for row in design.get(key, ()) or ())
        for key in DESIGN_ALLOWLIST[1:]}
    found = {row.contrast_id: row for row in (result.primary_items if result else ())}
    rows: list[tuple[Any, str, ec.EvidenceKind, str]] = [
        (row, row.diagnostic_id, "diagnostic", row.interpreting_rule_id) for row in diagnostics]
    rows += [(row, row.branch_id, "sensitivity", row.comparison_result) for row in sensitivities]
    return ClaimReviewContextV1(
        **text, estimand_id=manifest.estimand_id, method_id=manifest.method_id,
        population_id=manifest.population_id, timeframe_id=manifest.timeframe_id,
        contrasts=tuple(_contrast(cap, found.get(cap.contrast_id)) for cap in ceiling.items),
        evidence=tuple(ClaimEvidenceSummaryV1(
            evidence_id=name, kind=kind, comparison_result=comparison, values=dict(row.values),
            execution_status=row.execution_status, policy_result=row.policy_result,
            artifact_id=refs.get(name, name)) for row, name, kind, comparison in rows),
        population_summary=dict(population_summary), overall_ceiling=ceiling.overall_ceiling,
        required_qualifications=required_qualifications(diagnostics, sensitivities),
        allowed_artifact_ids=tuple(sorted(set(refs.values()))))


def _item_codes(item: ClaimItemV1, cap: ec.CeilingItemV1) -> tuple[str, ...]:
    rows = ((item.ceiling != cap.ceiling, f"ceiling_restated:{item.ceiling}!={cap.ceiling}"),
            (ec.most_restrictive((item.status, cap.ceiling)) != item.status,
             f"status_above_ceiling:{item.status}>{cap.ceiling}"),
            (item.status in HANDOFF_READABLE and item.estimate is None,
             "reportable_without_estimate"))
    return tuple(f"{item.contrast_id}:{code}" for bad, code in rows if bad)


def claim_codes(draft: ClaimJudgmentDraftV1, ceiling: ec.JudgmentCeilingV1,
                contrast_ids: Sequence[str], committed: frozenset[str],
                qualifications: Sequence[str]) -> tuple[str, ...]:
    # §16.3 rechecked over what the model returned: one ordered item per primary contrast, every
    # status at or below its own ceiling, the overall status the most restrictive item, every
    # cited id a committed ref, and every required qualification still carried.
    caps = {cap.contrast_id: cap for cap in ceiling.items}
    if (order := tuple(row.contrast_id for row in draft.items)) != tuple(contrast_ids):
        return (f"items_not_the_plan_contrasts:{','.join(order) or '(none)'}",)
    codes = tuple(code for row in draft.items for code in _item_codes(row, caps[row.contrast_id]))
    if draft.status != ec.most_restrictive(tuple(row.status for row in draft.items)):
        codes += (f"overall_status_not_most_restrictive:{draft.status}",)
    if draft.overall_ceiling != ceiling.overall_ceiling:
        codes += (f"overall_ceiling_restated:{draft.overall_ceiling}",)
    if ec.most_restrictive((draft.status, ceiling.overall_ceiling)) != draft.status:
        codes += (f"status_above_ceiling:{draft.status}>{ceiling.overall_ceiling}",)
    cited = collect_ids(draft.model_dump(mode="json"), ("cited_artifact_ids",))
    codes += tuple(f"uncommitted_artifact_id:{found}" for found in sorted(cited - committed))
    return codes + tuple(f"qualification_missing:{row}" for row in
                         sorted(set(qualifications) - set(draft.qualifications)))


def claim_validator(payload: Mapping[str, object], committed: frozenset[str],
                    qualifications: Sequence[str]) -> ew.ClaimValidator:
    # Wall 13's callable. It reads only the context the coordinator handed the wall, and every
    # failure — shape or substance — returns under the wall's one stable code (D-067 detail).
    def check(ctx: ew.WallContext) -> tuple[str, ...]:
        if ctx.ceiling is None or ctx.plan is None:
            return ("judgment_ceiling_missing",)
        try:
            draft = parse_strict(ClaimJudgmentDraftV1, payload)
        except ValidationError as bad:
            return (f"draft_shape_invalid:{str(bad)[:160]}",)
        return claim_codes(draft, ctx.ceiling, ctx.plan.contrast_ids, committed, qualifications)
    return check


@dataclass(frozen=True)
class ClaimTaskSpecV1:
    # SC §5.4's one estimation row: no tools, one initial response, two targeted corrections.
    task_kind: str = TASK_KIND
    wall: int = CLAIM_WALL
    correction_budget: int = CORRECTION_BUDGET


CLAIM_TASK: Final = ClaimTaskSpecV1()


def build_claim_envelope(spec: ClaimTaskSpecV1, *, task_id: str, attempt_id: str,
                         manifest_ref: ArtifactRef, scope_ids: Sequence[str],
                         parent_artifacts: Sequence[ArtifactRef], allowed_tool_ids: Sequence[str],
                         allowed_evidence_ids: Sequence[str], **rest: Any) -> AgentTaskEnvelopeV1:
    # §16.2: one bounded envelope with an empty tool allowlist and one stopping state.
    return AgentTaskEnvelopeV1(
        envelope_id=f"env:{task_id}:{attempt_id}", schema_version="agent-task-envelope.v1",
        task_id=task_id, attempt_id=attempt_id, context_manifest=manifest_ref,
        task_kind=spec.task_kind, scope_ids=tuple(scope_ids), allowed_retrieval_ids=(),
        parent_artifacts=tuple(parent_artifacts), allowed_tool_ids=tuple(allowed_tool_ids),
        allowed_evidence_ids=tuple(allowed_evidence_ids), output_schema_version=SCHEMA_VERSION,
        prompt_version=PINNED_VERSIONS["prompt"], validator_version=PINNED_VERSIONS["validator"],
        model_profile_version=PINNED_VERSIONS["model_profile"],
        budgets=TaskBudgets(token_budget=TOKEN_BUDGET, tool_call_budget=0,
                            correction_budget=spec.correction_budget),
        allowed_stopping_states=(TaskStatus.COMPLETE,),
        forbidden_payload_classes=FORBIDDEN_PAYLOAD_CLASSES,
        error_vocabulary=("schema_invalid", "claim_exceeds_ceiling", CORRECTION_EXHAUSTED), **rest)


def render_claim_prompt(spec: ClaimTaskSpecV1, prompts_root: Path,
                        sections: Mapping[str, object]) -> str:
    # D-064..D-071: the template plus every closed vocabulary it must echo, hydrated as JSON.
    return (prompts_root / PROMPT_PATH).read_text(encoding="utf-8") + "".join(
        f"\n\n## {key}\n{json.dumps(sections[key], indent=1, sort_keys=True)}"
        for key in sorted(sections))


@dataclass(frozen=True)
class ClaimJudgmentOutcome:
    # The committed §16.3 judgment, or a typed failure carrying its one stable code.
    judgment: ClaimJudgmentV1 | None
    artifact_id: str | None = None
    error_code: str | None = None


def sealed_judgment(payload: Mapping[str, Any], lineage: Mapping[str, Any]) -> ClaimJudgmentV1:
    # Nothing the model returned is trusted: lineage, the §16.3 outcome mapping, and the five
    # pinned versions are set here, and the whole record re-parses strictly before it commits.
    body = dict(payload) | dict(lineage)
    body["outcome_status"] = OUTCOME_BY_STATUS.get(str(body.get("status")), "failed")
    body["schema_version"], body["versions"] = SCHEMA_VERSION, dict(
        body.get("versions") or {}) | PINNED_VERSIONS
    return parse_strict(ClaimJudgmentV1, body)


def not_estimable_judgment(context: ClaimReviewContextV1,
                           lineage: Mapping[str, Any]) -> ClaimJudgmentV1:
    # §16.3's last rule: with no complete primary result the terminal judgment is deterministic.
    cited = list(context.allowed_artifact_ids[:1])
    return sealed_judgment({
        "causal_question": context.causal_question, "estimand_id": context.estimand_id,
        "status": "not_estimable", "overall_ceiling": context.overall_ceiling,
        "qualifications": list(context.required_qualifications), "cited_artifact_ids": cited,
        "unresolved_uncertainty": list(context.unresolved_uncertainty),
        "cannot_conclude": ["no complete valid primary analysis result is available"],
        "items": [{"contrast_id": row.contrast_id, "ceiling": row.ceiling, "population_id":
                   context.population_id, "status": "not_estimable", "cited_artifact_ids": cited,
                   "timeframe_id": context.timeframe_id,
                   "estimate_units": row.estimate_units or context.estimand_id,
                   "comparator_id": row.comparator_id or context.population_id}
                  for row in context.contrasts]}, lineage)


def judge(runner: TaskRunner, state: Any, context: ClaimReviewContextV1, *,
          ctx: ew.WallContext, run_walls: Callable[[int, ew.WallContext], ValidationReport],
          lineage: Mapping[str, Any]) -> ClaimJudgmentOutcome:
    # The ONE §16.2 call — or none at all: with no complete primary result the not_estimable
    # judgment is built and committed here, and no envelope, prompt, or gateway is constructed.
    result = ctx.primary_result
    if result is None or not result.complete:
        judgment = not_estimable_judgment(context, lineage)
        built = runner.commit(state, ARTIFACT_TYPE, judgment.model_dump(mode="json"),
                              runner.parents(state, *NOT_ESTIMABLE_PARENTS))
        return ClaimJudgmentOutcome(judgment, built.artifact_id)
    known, quals = frozenset(context.allowed_artifact_ids), context.required_qualifications
    bound = replace(runner, commit=lambda inner, kind, body, parents: runner.commit(
        inner, kind, sealed_judgment(body, lineage).model_dump(mode="json"), parents),
        validate=lambda highest, kind, model, answer, walls: run_walls(highest, replace(
            walls, claim_validator=claim_validator(answer.payload, known, quals))))
    done = bound.run(state, TASK_KIND, ClaimJudgmentDraftV1, scope_kind=SCOPE_KIND,
                     scope_ids=[row.contrast_id for row in context.contrasts],
                     parent_kinds=PARENT_KINDS, payload=context.model_dump(mode="json"),
                     commits=ARTIFACT_TYPE, ctx=ctx)
    if done is None:
        return ClaimJudgmentOutcome(None, None, CORRECTION_EXHAUSTED)
    draft, built = done[-1]
    return ClaimJudgmentOutcome(sealed_judgment(draft.model_dump(mode="json"), lineage),
                                built.artifact_id)
