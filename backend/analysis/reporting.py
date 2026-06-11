"""A closed report contract: references are selected, never invented addresses."""

from enum import StrEnum
from typing import Any, cast

from pydantic import Field, create_model

from backend.analysis.contracts import AnalysisReport, Conclusion
from backend.validation import ReplyNumber


def contract(references: set[str], numeric: set[str]) -> type[Conclusion]:
    # String enums serialize to the same stable identities as domain records. This
    # constrains the provider's schema as well as local Pydantic validation.
    def choices(name: str, values: set[str]) -> type[StrEnum]:
        return cast(type[StrEnum], StrEnum(name, {f"item_{i}": value for i, value in enumerate(sorted(values))}))

    refs = choices("ReportReference", references)
    numbers = choices("NumericalReference", numeric)
    scalar = create_model("ReportNumber", __base__=ReplyNumber, source=(numbers, ...))
    report = create_model(
        "GroundedReport", __base__=AnalysisReport,
        numbers=(list[scalar], Field(default_factory=list, max_length=200)),  # type: ignore[valid-type]
    )
    fields: dict[str, Any] = {
        "citations": (list[refs], Field(default_factory=list, max_length=60)),  # type: ignore[valid-type]
        "report": (report, ...),
    }
    return create_model("GroundedConclusion", __base__=Conclusion, **fields)
