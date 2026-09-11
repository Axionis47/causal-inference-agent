"""Accepted-run assessment wraps the shared evaluator and complete fixed snapshot."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ValidationError

from causal.analysis.common.evaluation import evaluate_candidate
from causal.analysis.common.models import Category, Issue
from causal.analysis.contracts import (
    AnalysisSpecification,
    Assessment,
    FixedCandidate,
    FixedDesign,
    fingerprint,
)


def validation_issues(error: ValidationError) -> tuple[Issue, ...]:
    issues = []
    for item in error.errors(include_url=False):
        field = ".".join(str(part) for part in item["loc"])
        missing = item["type"] == "missing"
        category: Category = ("missing_data" if missing and field.startswith("dataset") else
                    "missing_context" if missing else
                    "unsupported_capability" if item["type"] in (
                        "literal_error", "extra_forbidden", "union_tag_invalid",
                        "less_than", "less_than_equal", "greater_than", "greater_than_equal") else
                    "contradictory_configuration")
        issues.append(Issue(category=category, field=field or "specification",
                            finding=item["msg"], requirement="Use the retrieved typed input schema.",
                            explanation="This proposal cannot yet describe a supported run.",
                            resolutions=("Supply the missing context or use a supported choice.",)))
    return tuple(issues)


def _fixed_issue(field: str, finding: str) -> Issue:
    return Issue(category="contradictory_configuration", field=field, finding=finding,
                 requirement="Use the complete accepted candidate under its current reference and hash.",
                 explanation="Changing any scientific choice or computation requires a new accepted candidate revision.")


def _status(issues: tuple[Issue, ...]) -> Literal["ready", "needs_information", "rejected"]:
    if not issues:
        return "ready"
    return ("needs_information" if all(issue.category in ("missing_context", "missing_data")
                                       for issue in issues) else "rejected")


def assess(design: FixedCandidate | FixedDesign | Mapping[str, Any],
           draft: AnalysisSpecification | Mapping[str, Any]) -> Assessment:
    raw_design = design.model_dump() if isinstance(design, FixedCandidate | FixedDesign) else dict(design)
    raw = draft.model_dump() if isinstance(draft, AnalysisSpecification) else dict(draft)
    if raw_design.get("schema_version") != "analysis-fixed-design.v2":
        return Assessment(status="rejected", capability_version=None, specification_hash=None,
                          specification=None, issues=(_fixed_issue("design.schema_version",
                              "Historical FixedDesign records do not freeze the complete candidate and cannot certify a new run."),))
    try:
        fixed = FixedCandidate.model_validate(raw_design)
    except ValidationError as error:
        return Assessment(status=_status(validation_issues(error)), capability_version=None,
                          specification_hash=None, specification=None,
                          issues=validation_issues(error))
    evaluation = evaluate_candidate(fixed.candidate)
    issues = evaluation.issues
    if evaluation.capability_version != fixed.capability_version:
        issues += (_fixed_issue("design.capability_version",
                               "The fixed candidate was accepted under a different capability version."),)
    if raw.get("schema_version", "analysis-specification.v2") != "analysis-specification.v2":
        issues += (_fixed_issue("schema_version", "Historical specifications cannot certify the current fixed-design boundary."),)
    if issues or evaluation.status != "design_ready":
        status = _status(issues)
        if status == "ready":
            status = "rejected" if evaluation.status == "rejected" else "needs_information"
        return Assessment(status=status, capability_version=evaluation.capability_version,
                          specification_hash=None, specification=None, issues=issues,
                          evaluation=evaluation)
    raw.setdefault("design", fixed)
    try:
        spec = AnalysisSpecification.model_validate(raw)
    except ValidationError as error:
        invalid = validation_issues(error)
        return Assessment(status=_status(invalid), capability_version=evaluation.capability_version,
                          specification_hash=None, specification=None, issues=invalid,
                          evaluation=evaluation)
    if spec.design != fixed:
        issues += (_fixed_issue("design", "The proposed specification differs from the supplied fixed candidate."),)
    return Assessment(status=_status(issues), capability_version=evaluation.capability_version,
                      specification_hash=fingerprint(spec) if not issues else None,
                      specification=spec if not issues else None, issues=issues,
                      evaluation=evaluation)
