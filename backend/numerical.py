"""Pure numerical citation checks, with no feature or persistence dependencies."""

import ast
import math
import re
from collections.abc import Mapping
from typing import Any, Protocol, Self

from backend.validation import ReplyNumber


class NumericalText(Protocol):
    text: str
    numbers: list[ReplyNumber]

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self: ...


NUMBER = re.compile(
    r"(?<![\w:.\-])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.:\-]*[a-zA-Z_])"
)


def numbers(text: str) -> list[float]:
    return [float(token.replace(",", "")) for token in NUMBER.findall(text)]


def source_values(value: Any) -> list[float]:
    """The exact numeric operands a source can support, without interpreting words."""
    return numbers(value) if isinstance(value, str) else [float(value)]


def program_literals(source: str, code: str) -> dict[str, Any]:
    """Expose constants in a selected program as code evidence, never computed facts."""
    result: dict[str, Any] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result
    signed = {
        node.operand: node
        for node in ast.walk(tree)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            value = float(node.value)
            if node in signed and isinstance(signed[node].op, ast.USub):
                value = -value
            if math.isfinite(value):
                result[f"{source}.line_{node.lineno}_{node.col_offset}"] = {
                    "basis": "program_literal_not_result",
                    "value": value,
                    "line": node.lineno,
                }
    return result


def bind_literals[T: NumericalText](reply: T, prespecified: dict[str, str]) -> T:
    """Attach exact request literals without inventing computed evidence or rounding.

    Callers supply only immutable requests already authorized for this decision/turn.
    Model-selected numerical references take precedence and still undergo validation.
    """
    reply = reply.model_copy(deep=True)
    missing = set(numbers(reply.text)) - {item.value for item in reply.numbers}
    for source, text in sorted(prespecified.items()):
        matches = missing & set(numbers(text))
        reply.numbers.extend(ReplyNumber(value=value, source=source) for value in sorted(matches))
        missing -= matches
    return reply


def render_handles[T: NumericalText](reply: T) -> T:
    """Render a selected handle using its explicit value, then require validation.

    This never repairs a source choice or calculates a number. Call validate
    after canonical reference resolution to check every rendered value.
    """
    def replace(match: re.Match[str]) -> str:
        source = match[1]
        values = {entry.value for entry in reply.numbers if entry.source == source}
        if len(values) != 1 or not math.isfinite(next(iter(values))):
            raise ValueError(f"Numeric placeholder <{source}> needs exactly one finite displayed value in numbers metadata. Use literal numbers when a source supports several displayed values.")
        return format(next(iter(values)), ".15g")
    return reply.model_copy(update={"text": re.sub(r"<(n\d+)>", replace, reply.text)})


def validate(reply: NumericalText, evidence: dict[str, Any]) -> None:
    written = set(numbers(reply.text))
    registered = {entry.value for entry in reply.numbers}
    if written != registered:
        raise ValueError(
            f"Every digit-written number needs matching numbers metadata, with no unused entries. Written: {sorted(written)}; registered: {sorted(registered)}. Avoid numbered lists; use prose or bullets."
        )
    for entry in reply.numbers:
        if entry.source not in evidence:
            raise ValueError(f"Unknown numerical evidence source: {entry.source}.")
        value = evidence[entry.source]["value"]
        known = source_values(value)
        if not any(
            math.isclose(entry.value, n, rel_tol=1e-8, abs_tol=1e-9)
            or any(round(n, places) == entry.value for places in range(7))
            for n in known
        ):
            raise ValueError(
                f"Number {entry.value} does not match its selected source {entry.source}. Use the source supporting that specific claim or omit the number."
            )
