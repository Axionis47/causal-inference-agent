# The §9 visualization curator: the bounded curator context, the one allowlisted layout-fact
# operation, the ONE curator call through the shared TaskRunner, and the deterministic §9.2/§10
# plan validator that gate 2 delegates to (PRD-005 §7.1, §9, §10, §11.4, §14).
# This module imports `causal.shared` only; every upstream payload arrives here as data.

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from causal.presentation import contracts as pc
from causal.shared.agenttask import TaskRunner
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import AgentTaskEnvelopeV1, TaskBudgets, TaskStatus
from causal.shared.validation import ValidationReport, make_issue, parse_strict

TASK_KIND, SCOPE_KIND = "figure_plan", "visual_evidence"
PROMPT_PATH: Final = "prompts/presentation/curator.v1.txt"
ARTIFACT_TYPE, SCHEMA_VERSION = "FigurePlan", "figure-plan.v1"
PLAN_GATE, CORRECTION_BUDGET, TOKEN_BUDGET = 2, 2, 24576
PARENT_KINDS: Final = ("PresentationContextManifest", "ClaimJudgment", "FigureDataArtifact")
# §9.3: the one allowlisted tool operation, and the one call it is ever permitted.
TOOL_ID, TOOL_CALL_BUDGET = "resolve_registered_layout_facts", 1
CORRECTION_EXHAUSTED, TOOL_DENIED = "correction_exhausted", "layout_tool_budget_exceeded"
# The gate-2 family. Each code is stable and names the figure or question that failed it.
UNCOVERED, UNAPPROVED = "required_evidence_uncovered", "unapproved_evidence"
UNREGISTERED_TEMPLATE, INCOMPATIBLE_TEMPLATE = "unregistered_template", "template_not_compatible"
UNREGISTERED_CHOICE, INCOMPATIBLE_PANEL = "unregistered_choice", "incompatible_panel_quantities"
OVER_CAPACITY, DROPPED_ENCODING = "capacity_exceeded", "mandatory_encoding_dropped"
MISPLACED_QUALIFICATION, INACCESSIBLE = "qualification_misplaced", "accessibility_text_missing"
SHAPE_INVALID: Final = "draft_shape_invalid"
# §9.3: a code that survives its two corrections becomes one of these two terminal statuses.
NEEDS_TEMPLATE_CODES: Final = frozenset({
    UNCOVERED, UNAPPROVED, UNREGISTERED_TEMPLATE, INCOMPATIBLE_TEMPLATE, INCOMPATIBLE_PANEL,
    DROPPED_ENCODING, SHAPE_INVALID})
# §10: the versions the harness pins on every accepted plan; the curator owns none of them.
PINNED_VERSIONS: Final = {
    "prompt": "presentation-curator-prompt.v1", "schema": SCHEMA_VERSION,
    "model_profile": "vertex-model-profile.v1", "validators": "presentation-validators.v1"}
# §9.2: no field below could carry one of these even if an upstream artifact held it.
FORBIDDEN_PAYLOAD_CLASSES: Final = ("rows", "raw_rows", "observations", "frame",
                                    "computed_statistics", "vega_lite_spec", "credentials")
# §11.4: the profile marks an effect display by mandating a null reference on it.
PRIMARY_MARKER: Final = "null_effect"
INTERVAL_FIELDS: Final = frozenset({"interval_lower", "interval_upper"})


def curator_context(manifest: pc.PresentationContextManifestV1,
                    catalog: pc.VisualizationCatalogV1,
                    manifest_ref: ArtifactRef) -> pc.PresentationCuratorContextV1:
    # §7.1: the approved claim surface, the bounded evidence facts the manifest already froze,
    # and the compatible catalog entries. No field of this model can hold a row or an array.
    ids = {name for row in manifest.evidence for name in row.compatible_template_ids}
    return pc.PresentationCuratorContextV1(
        claim_status=manifest.claim_status, statement_ids=manifest.statement_ids,
        qualification_ids=manifest.qualification_ids, evidence=manifest.evidence,
        manifest=manifest_ref, display_profile=manifest.display_profile, limits=catalog.limits,
        templates=tuple(row for row in catalog.templates if row.template_id in ids))


def layout_facts(figure_data: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    # Bounded counts over the frozen figure-data payloads: how much there is to lay out, never
    # what it says. No entry here is a measured value, a label string, or a coordinate.
    facts: dict[str, dict[str, int]] = {}
    for name, payload in figure_data.items():
        points = tuple(payload.get("points") or ())
        labels = tuple(str(row) for row in (payload.get("labels") or {}).values())
        facts[name] = {
            "series": len({str(row.get("series_id")) for row in points}), "points": len(points),
            "labels": len(labels), "label_characters": max((len(row) for row in labels), default=0),
            "intervals": sum(1 for row in points if row.get("interval_lower") is not None),
            "denominators": sum(1 for row in points if row.get("denominator") is not None)}
    return facts


@dataclass
class LayoutFactTool:
    # §9.3: the bounded lookup is one allowlisted operation inside the curator task, not a
    # second agent and not a new planning call. The second call is denied here, in code.
    facts: Mapping[str, Mapping[str, int]]
    calls: int = 0

    def resolve(self, evidence_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        self.calls += 1
        if self.calls > TOOL_CALL_BUDGET:
            raise pc.PresentationError(f"{TOOL_ID} is a one-call operation", TOOL_DENIED)
        return {name: dict(self.facts[name]) for name in evidence_ids if name in self.facts}


def primary_evidence_ids(profile: pc.MethodProfileV1) -> tuple[str, ...]:
    return tuple(name for name, ids in profile.mandatory_encodings.items() if PRIMARY_MARKER in ids)


def _coverage_codes(draft: pc.FigurePlanDraftV1, profile: pc.MethodProfileV1,
                    known: Mapping[str, pc.EvidenceEntryV1]) -> tuple[str, ...]:
    # §9.2: no required question is omitted, doubled, or answered by unapproved evidence.
    counts = Counter(name for row in draft.figures for name in row.visual_evidence_ids)
    codes = [f"{UNCOVERED}:{name}" for name in profile.required_evidence_ids
             if counts[name] != 1]
    return tuple(codes + [f"{UNAPPROVED}:{name}" for name in sorted(set(counts) - set(known))])


def _kept_encodings(entry: pc.FigureEntryV1, template: pc.TemplateV1,
                    fact: Mapping[str, int]) -> set[str]:
    # What one figure still carries: the frozen fields its template maps and its data holds,
    # plus the one enumerated reference line it selected.
    fields = set(INTERVAL_FIELDS) if fact.get("intervals") else set()
    fields |= {"denominator"} if fact.get("denominators") else set()
    mapped = set(template.field_mappings.values())
    return {name for name in fields if name in mapped} | {entry.choices.get("reference_lines", "")}


def _capacity_codes(entry: pc.FigureEntryV1, bounds: pc.TemplateBoundsV1,
                    facts: Mapping[str, Mapping[str, int]]) -> tuple[str, ...]:
    # §8: every declared ceiling, measured against the frozen counts rather than the draft's
    # own account of itself.
    def total(key: str) -> int:
        return sum(facts.get(name, {}).get(key, 0) for name in entry.visual_evidence_ids)

    def peak(key: str) -> int:
        return max((facts.get(name, {}).get(key, 0) for name in entry.visual_evidence_ids),
                   default=0)
    checks = ((not bounds.min_panels <= len(entry.panel_groups) <= bounds.max_panels, "panels"),
              (total("series") > bounds.max_series_per_figure, "series_per_figure"),
              (total("labels") > bounds.max_labels, "labels"),
              (peak("label_characters") > bounds.max_label_characters, "label_characters"),
              (len(entry.annotation_ids) > bounds.max_annotations_per_figure, "annotations"))
    return tuple(f"{OVER_CAPACITY}:{entry.figure_id}:{name}" for bad, name in checks if bad)


def _panel_codes(entry: pc.FigureEntryV1, bounds: pc.TemplateBoundsV1,
                 known: Mapping[str, pc.EvidenceEntryV1],
                 facts: Mapping[str, Mapping[str, int]]) -> tuple[str, ...]:
    # §11.1: series share a panel only under one quantity meaning, unit, and denominator.
    codes: list[str] = []
    for group in entry.panel_groups:
        rows = [known[name] for name in group if name in known]
        shapes = {(tuple(sorted(row.quantities.items())), tuple(sorted(row.units.items())),
                   bool(facts.get(row.visual_evidence_id, {}).get("denominators")))
                  for row in rows}
        series = sum(facts.get(name, {}).get("series", 0) for name in group)
        if len(shapes) > 1 or set(group) - set(entry.visual_evidence_ids):
            codes.append(f"{INCOMPATIBLE_PANEL}:{entry.figure_id}:{'+'.join(group)}")
        if not bounds.min_series_per_panel <= series <= bounds.max_series_per_panel:
            codes.append(f"{OVER_CAPACITY}:{entry.figure_id}:series_per_panel")
    return tuple(codes)


def _figure_codes(entry: pc.FigureEntryV1, template: pc.TemplateV1 | None,
                  profile: pc.MethodProfileV1, known: Mapping[str, pc.EvidenceEntryV1],
                  facts: Mapping[str, Mapping[str, int]]) -> tuple[str, ...]:
    # §9.2 per figure: a registered compatible template, enumerated choices only, and every
    # mandatory encoding still drawn.
    if template is None:
        return (f"{UNREGISTERED_TEMPLATE}:{entry.figure_id}:{entry.template_id}",)
    codes = [f"{INCOMPATIBLE_TEMPLATE}:{entry.figure_id}:{name}"
             for name in entry.visual_evidence_ids
             if name not in template.visual_evidence_ids
             or entry.template_id not in profile.templates_by_evidence.get(name, ())]
    codes += [f"{UNREGISTERED_CHOICE}:{entry.figure_id}:{key}:{value}"
              for key, value in sorted(entry.choices.items())
              if value not in template.choices.get(key, ())]
    codes += [f"{DROPPED_ENCODING}:{entry.figure_id}:{name}"
              for evidence_id in entry.visual_evidence_ids
              for name in profile.mandatory_encodings.get(evidence_id, ())
              if name not in _kept_encodings(entry, template, facts.get(evidence_id, {}))]
    return (tuple(codes) + _capacity_codes(entry, template.bounds, facts)
            + _panel_codes(entry, template.bounds, known, facts))


def _text_codes(entry: pc.FigureEntryV1, profile: pc.MethodProfileV1) -> tuple[str, ...]:
    # §14: a title phrased as the visual question, three distinct texts, and a description
    # that names the uncertainty whenever the profile makes an interval mandatory.
    needed = {name for evidence_id in entry.visual_evidence_ids
              for name in profile.mandatory_encodings.get(evidence_id, ())}
    described = entry.text["accessible_description"].lower()
    bad = (len(set(entry.text.values())) != len(pc.TEXT_KEYS) or "?" not in entry.text["title"]
           or (bool(needed & INTERVAL_FIELDS) and "interval" not in described))
    return (f"{INACCESSIBLE}:{entry.figure_id}",) if bad else ()


def _qualification_codes(draft: pc.FigurePlanDraftV1, context: pc.PresentationCuratorContextV1,
                         profile: pc.MethodProfileV1) -> tuple[str, ...]:
    # §11.4: every approved qualification sits beside the primary result, in the placement the
    # profile fixed, and is named in that figure's caption. A footnote alone is not enough.
    wanted, primary = set(context.qualification_ids), set(primary_evidence_ids(profile))
    if not wanted:
        return ()
    return tuple(f"{MISPLACED_QUALIFICATION}:{row.figure_id}" for row in draft.figures
                 if primary & set(row.visual_evidence_ids)
                 and (wanted - set(row.qualification_ids)
                      or row.choices.get("qualifications") != profile.qualification_placement
                      or any(name not in row.text["caption"] for name in wanted)))


def plan_codes(draft: pc.FigurePlanDraftV1, context: pc.PresentationCuratorContextV1,
               profile: pc.MethodProfileV1,
               facts: Mapping[str, Mapping[str, int]]) -> tuple[str, ...]:
    # Gate 2 over one draft: coverage, template compatibility, panel and capacity bounds,
    # qualification placement, and accessibility. A typed inability is not a failed plan (§9.3).
    if draft.inability_code is not None:
        return ()
    known = {row.visual_evidence_id: row for row in context.evidence}
    templates = {row.template_id: row for row in context.templates}
    codes = _coverage_codes(draft, profile, known)
    for entry in draft.figures:
        codes += _figure_codes(entry, templates.get(entry.template_id), profile, known, facts)
        codes += _text_codes(entry, profile)
    if len(draft.figures) > context.limits.max_figures:
        codes += (f"{OVER_CAPACITY}:plan:figures",)
    return codes + _qualification_codes(draft, context, profile)


def plan_report(codes: Sequence[str]) -> ValidationReport:
    # `code` is what the §9.3 correction budget counts; `detail` is the near-miss text the
    # curator is sent back with (D-067 idiom).
    return ValidationReport(wall=PLAN_GATE, issues=tuple(
        make_issue(code.split(":", 1)[0], "$.figures", "presentation-plan-validator.v1",
                   ("revise_figure_plan",), detail=code) for code in codes))


def plan_validator(context: pc.PresentationCuratorContextV1, profile: pc.MethodProfileV1,
                   facts: Mapping[str, Mapping[str, int]]) -> Any:
    def check(highest: int, kind: str, model: Any, answer: Any, walls: Any) -> ValidationReport:
        try:
            draft = parse_strict(pc.FigurePlanDraftV1, answer.payload)
        except ValidationError as bad:
            return plan_report((f"{SHAPE_INVALID}:{str(bad)[:120]}",))
        return plan_report(plan_codes(draft, context, profile, facts))
    return check


@dataclass(frozen=True)
class CuratorTaskSpecV1:
    # §9.3: one initial response, at most two targeted corrections, one bounded tool operation.
    task_kind: str = TASK_KIND
    wall: int = PLAN_GATE
    correction_budget: int = CORRECTION_BUDGET


CURATOR_TASK: Final = CuratorTaskSpecV1()


def build_curator_envelope(spec: CuratorTaskSpecV1, *, task_id: str, attempt_id: str,
                           manifest_ref: ArtifactRef, scope_ids: Sequence[str],
                           parent_artifacts: Sequence[ArtifactRef],
                           allowed_tool_ids: Sequence[str], allowed_evidence_ids: Sequence[str],
                           **rest: Any) -> AgentTaskEnvelopeV1:
    # §9: one bounded envelope. The allowlist is filtered to the one registered layout-fact
    # operation here, so a widened caller configuration never reaches the model.
    return AgentTaskEnvelopeV1(
        envelope_id=f"env:{task_id}:{attempt_id}", schema_version="agent-task-envelope.v1",
        task_id=task_id, attempt_id=attempt_id, context_manifest=manifest_ref,
        task_kind=spec.task_kind, scope_ids=tuple(scope_ids), allowed_retrieval_ids=(),
        parent_artifacts=tuple(parent_artifacts),
        allowed_tool_ids=tuple(name for name in allowed_tool_ids if name == TOOL_ID),
        allowed_evidence_ids=tuple(allowed_evidence_ids), output_schema_version=SCHEMA_VERSION,
        prompt_version=PINNED_VERSIONS["prompt"], validator_version=PINNED_VERSIONS["validators"],
        model_profile_version=PINNED_VERSIONS["model_profile"],
        budgets=TaskBudgets(token_budget=TOKEN_BUDGET, tool_call_budget=TOOL_CALL_BUDGET,
                            correction_budget=spec.correction_budget),
        allowed_stopping_states=(TaskStatus.COMPLETE,),
        forbidden_payload_classes=FORBIDDEN_PAYLOAD_CLASSES,
        error_vocabulary=(SHAPE_INVALID, UNCOVERED, UNREGISTERED_TEMPLATE, UNREGISTERED_CHOICE,
                          OVER_CAPACITY, TOOL_DENIED, CORRECTION_EXHAUSTED), **rest)


def render_curator_prompt(spec: CuratorTaskSpecV1, prompts_root: Path,
                          sections: Mapping[str, object]) -> str:
    # The template plus every closed vocabulary it must echo, hydrated as JSON (judge idiom).
    return (prompts_root / PROMPT_PATH).read_text(encoding="utf-8") + "".join(
        f"\n\n## {key}\n{json.dumps(sections[key], indent=1, sort_keys=True)}"
        for key in sorted(sections))


@dataclass(frozen=True)
class CurationOutcome:
    # The accepted plan, or a typed presentation status and the stable codes behind it.
    plan: pc.FigurePlanV1 | None
    artifact_id: str | None = None
    status: pc.PresentationOutcomeStatus | None = None
    error_code: str | None = None
    detail_codes: tuple[str, ...] = ()


def sealed_plan(payload: Mapping[str, Any], lineage: Mapping[str, Any],
                coverage: Mapping[str, str],
                figure_data: Mapping[str, ArtifactRef]) -> pc.FigurePlanV1:
    # Nothing the curator returned is trusted with lineage: parents, the coverage map, the
    # frozen figure-data refs, and the pinned versions are set here and re-parsed strictly.
    body = dict(payload) | dict(lineage) | {
        "schema_version": SCHEMA_VERSION, "coverage": dict(coverage),
        "figure_data": {name: ref.model_dump(mode="json") for name, ref in figure_data.items()},
        "versions": dict(lineage.get("versions") or {}) | PINNED_VERSIONS}
    return parse_strict(pc.FigurePlanV1, body)


def terminal_status(codes: Sequence[str]) -> pc.PresentationOutcomeStatus:
    # §9.3: repeated failure becomes a typed stage status, never a quietly reduced figure set.
    return ("needs_template" if any(code.split(":", 1)[0] in NEEDS_TEMPLATE_CODES
                                    for code in codes) else "needs_layout_revision")


def curate(runner: TaskRunner, state: Any, context: pc.PresentationCuratorContextV1, *,
           profile: pc.MethodProfileV1, facts: Mapping[str, Mapping[str, int]],
           lineage: Mapping[str, Any], coverage: Mapping[str, str],
           figure_data: Mapping[str, ArtifactRef]) -> CurationOutcome:
    # The ONE §9 curator call. Every draft passes gate 2 before anything commits, and the
    # compiler is typed on FigurePlanV1, so a draft that never sealed cannot reach it.
    check, seen = plan_validator(context, profile, facts), list[str]()

    def validate(*args: Any) -> ValidationReport:
        report: ValidationReport = check(*args)
        seen.extend(issue.detail for issue in report.issues)
        return report

    def commit(inner: Any, kind: str, body: Mapping[str, Any], parents: Any) -> Any:
        sealed = (sealed_plan(body, lineage, coverage, figure_data).model_dump(mode="json")
                  if body.get("figures") else dict(body))
        return runner.commit(inner, kind, sealed, parents)
    done = replace(runner, validate=validate, commit=commit).run(
        state, TASK_KIND, pc.FigurePlanDraftV1, scope_kind=SCOPE_KIND, commits=ARTIFACT_TYPE,
        scope_ids=[row.visual_evidence_id for row in context.evidence],
        parent_kinds=PARENT_KINDS, payload=context.model_dump(mode="json"))
    if done is None:
        return CurationOutcome(None, None, terminal_status(seen), CORRECTION_EXHAUSTED,
                               tuple(sorted(set(seen))))
    draft, built = done[-1]
    if draft.inability_code is not None:
        return CurationOutcome(None, built.artifact_id, "needs_template", draft.inability_code,
                               draft.implicated_evidence_ids)
    return CurationOutcome(sealed_plan(draft.model_dump(mode="json"), lineage, coverage,
                                       figure_data), built.artifact_id)
