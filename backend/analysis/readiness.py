"""Deterministic preflight over applied context, not a scientific identification verdict."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from backend.context.contracts import (
    BriefClaim,
    BriefVariable,
    ContextContent,
    Contract,
    DatasetStatement,
    Relationship,
    Statement,
    statement_entries,
)
from backend.context.service import fingerprint

# Bump with either readiness rules or the launch scientific-authority policy.
POLICY_VERSION = "frozen-context-19"


class ReadinessCheck(Contract):
    field: str
    passed: bool
    detail: str
    source_keys: list[str] = Field(default_factory=list)


class AnalysisReadiness(Contract):
    status: Literal["needs_context", "ready_with_assumptions", "ready"]
    question: str
    fingerprint: str
    context_hash: str
    brief_hash: str | None
    dataset_version_id: UUID
    context_revision_id: UUID | None
    source_sha256: str | None
    policy_version: str
    checks: list[ReadinessCheck]
    missing: list[ReadinessCheck]
    uncertainties: list[BriefClaim]
    optional_unknowns: list[str]


def assess(
    content: ContextContent,
    dataset_version_id: UUID,
    context_revision_id: UUID | None,
    *, question: str | None = None,
    policy_version: str = POLICY_VERSION,
    source_sha256: str | None = None,
) -> AnalysisReadiness:
    """Check essential bindings and reference eligibility without interpreting prose.

    This establishes that selected, reviewed source records exist in suitable
    fields. It cannot prove their entailment, identification, or computational
    adequacy. Optional unknown column roles never become a universal questionnaire.
    """
    brief = content.analysis_brief
    entries = dict(statement_entries(content))
    checks: list[ReadinessCheck] = []
    uncertainties: list[BriefClaim] = []

    def check(field: str, passed: bool, detail: str, keys: list[str] | None = None) -> None:
        checks.append(ReadinessCheck(field=field, passed=passed, detail=detail, source_keys=keys or []))

    def eligible(key: str, statement: Statement, purpose: str, columns: set[str]) -> bool:
        parts = key.split(".")
        if purpose == "population":
            return parts[:2] == ["dataset", "population"]
        if purpose == "unit":
            return parts[:2] in (["dataset", "observations"], ["dataset", "organization"])
        if purpose == "comparison":
            return parts[:2] == ["dataset", "question"]
        if purpose == "timing":
            return (
                parts[:2] in (["dataset", "measurement"], ["dataset", "study_process"])
                or len(parts) == 4 and parts[0] == "columns" and parts[1] in columns
                and parts[2] in {"collection_time", "reference_period"}
                or isinstance(statement, Relationship) and statement.kind == "temporal_order"
                and any(p.kind == "column" and p.id in columns for p in statement.participants)
            )
        if len(parts) == 4 and parts[0] == "columns" and parts[1] in columns:
            fields = {"definition", "concept", "construction", "derivation"}
            return parts[2] in (fields | {"codes", "determined_by"} if purpose == "change" else fields)
        if isinstance(statement, DatasetStatement):
            if purpose == "meaning":
                return parts[:2] == ["dataset", "measurement"] and bool(columns & set(statement.column_ids))
            return parts[:2] == ["dataset", "study_process"] and (not statement.column_ids or bool(columns & set(statement.column_ids)))
        return purpose == "change" and isinstance(statement, Relationship) and statement.kind == "assignment_rule" and any(
            p.kind == "column" and p.id in columns for p in statement.participants
        )

    def support(field: str, claim: BriefClaim, purpose: str, columns: set[str]) -> None:
        keys = claim.source_keys
        invalid = [key for key in keys if key not in entries or entries[key].status != "reviewed"
                   or entries[key].basis != "source_statement" or not eligible(key, entries[key], purpose, columns)]
        passed = bool(keys) and len(keys) == len(set(keys)) and not invalid
        check(field, passed,
              "Selected reviewed source statements are eligible for this field; their scientific meaning still requires analysis review."
              if passed else "Select current reviewed source statements in the context fields for this essential meaning. Missing, unreviewed, interpreted, withdrawn or unrelated references cannot establish readiness.", keys)
        if passed:
            qualified = [key for key in keys if entries[key].qualifications.strip()]
            if qualified:
                uncertainties.append(BriefClaim(source_keys=qualified))
            collections = {key.rsplit(".", 1)[0] for key in keys}
            unresolved = sorted(key for key, record in entries.items() if key.rsplit(".", 1)[0] in collections
                                and record.status in {"proposed", "unresolved"})
            check(field + ".unresolved", not unresolved,
                  "No unreviewed or unresolved competing records remain in these essential context fields."
                  if not unresolved else "Resolve the remaining proposed or unresolved records in this essential context field before declaring its meaning settled.", unresolved)
            conflicts = sorted({identity for key, record in entries.items() if record.status != "withdrawn"
                                for other in record.conflicts_with or [] if entries[other].status != "withdrawn"
                                and (key in keys or other in keys) for identity in (key, other)})
            check(field + ".conflicts", not conflicts,
                  "No live declared conflict affects these selected essential statements." if not conflicts
                  else "Resolve the declared context conflict affecting this essential meaning before preparation.", conflicts)

    def binding(field: str, value: BriefVariable | None) -> set[str]:
        columns = set(value.column_ids) if value else set()
        valid = bool(columns) and value is not None and len(columns) == len(value.column_ids) and columns <= content.columns.keys()
        check(field + ".column_ids", valid, "Exact source column bindings are present." if valid else "Bind the variable to its exact current source columns, without duplicates or inferred names.")
        support(field + ".meaning", value.meaning if value else BriefClaim(), "meaning", columns)
        if value and columns:
            covered = {cid for cid in columns if any(
                key.startswith(f"columns.{cid}.") or isinstance(entries.get(key), DatasetStatement)
                and cid in entries[key].column_ids  # type: ignore[attr-defined]
                for key in value.meaning.source_keys
            )}
            check(field + ".meaning_columns", covered == columns,
                  "Every bound source column has a selected meaning record." if covered == columns else "Supply meaning references for every column in this variable binding.", value.meaning.source_keys)
        return columns

    check("context_revision", context_revision_id is not None, "An exact applied context revision is selected." if context_revision_id else "Select an applied context revision.")
    check("analysis_brief", brief is not None, "An analysis brief was explicitly applied." if brief else "Discuss and apply an analysis brief before preparing analysis.")
    if brief:
        check("question", bool(brief.question.strip()) and (question is None or question == brief.question),
              "The launch question is the exact applied brief question." if brief.question.strip() and (question is None or question == brief.question) else "Finalize a nonempty brief question; a separate or changed launch question requires a new applied brief.")
        columns = binding("target", brief.target)
        ids = [item.id for item in brief.treatments]
        check("treatments", bool(ids) and len(ids) == len(set(ids)), "Distinct intervention variables are explicitly bound." if ids and len(ids) == len(set(ids)) else "Bind at least one intervention variable with a distinct identity for each treatment.")
        assigned = set(columns)
        for index, treatment in enumerate(brief.treatments):
            field = f"treatments[{index}]"
            selected = binding(field, treatment)
            check(field + ".distinct_binding", not bool(assigned & selected),
                  "The target and intervention variables have distinct source bindings." if not assigned & selected else "Overlapping target or treatment source bindings are ambiguous in this brief. Clarify distinct variables before preparation.")
            assigned.update(selected)
            columns.update(selected)
            support(field + ".change", treatment.change, "change", selected)
        for field in ("comparison", "population", "unit", "timing"):
            support(field, getattr(brief, field), field, columns)
        for index, assumption in enumerate(brief.assumptions):
            valid = bool(assumption.source_keys) and len(set(assumption.source_keys)) == len(assumption.source_keys) and all(
                key in entries and entries[key].status == "reviewed" for key in assumption.source_keys
            )
            check(f"assumptions[{index}]", valid, "Conditional context references remain explicit and do not fill missing essential meanings." if valid else "Select existing current statement references for this uncertainty.", assumption.source_keys)
            if valid:
                uncertainties.append(assumption)
    missing = [item for item in checks if not item.passed]
    unique = {tuple(item.source_keys): item for item in uncertainties}
    context_hash = fingerprint(content.model_dump(mode="json"))
    brief_hash = fingerprint(brief.model_dump(mode="json")) if brief else None
    frozen = {"dataset_version_id": str(dataset_version_id), "context_revision_id": str(context_revision_id) if context_revision_id else None,
              "source_sha256": source_sha256, "context_hash": context_hash, "brief_hash": brief_hash,
              "question": brief.question if brief else "", "policy_version": policy_version}
    return AnalysisReadiness(
        status="needs_context" if missing else "ready_with_assumptions" if unique else "ready",
        question=frozen["question"] or "", fingerprint=fingerprint(frozen), context_hash=context_hash, brief_hash=brief_hash,
        dataset_version_id=dataset_version_id, context_revision_id=context_revision_id, source_sha256=source_sha256,
        policy_version=policy_version, checks=checks, missing=missing, uncertainties=list(unique.values()),
        optional_unknowns=brief.optional_unknowns if brief else [],
    )
