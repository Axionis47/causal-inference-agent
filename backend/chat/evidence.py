"""Chat-owned collection of supplied, historical, and exploratory evidence."""

from typing import Any

from backend import numerical
from backend.context.contracts import ContextContent, statement_entries


def numeric_options(sources: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Each display handle selects one source value, including request literals."""
    aliases: dict[str, str] = {}
    options: dict[str, Any] = {}
    for key, source in sorted(sources.items()):
        for value in sorted(set(numerical.source_values(source["value"]))):
            alias = f"n{len(aliases) + 1}"
            aliases[alias] = key
            options[alias] = {**source, "value": value, "reference": key, "supported_values": [value]}
    return aliases, options


def sources(
    context: dict[str, Any],
    artifacts: list[dict[str, Any]],
    current: dict[str, Any] | None,
    history: list[dict[str, Any]],
    message: str,
    rows: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset.rows": {"basis": "source_shape", "value": rows},
        "user.current": {"basis": "user_statement_not_computed", "value": message},
    }
    for turn in history:
        result[f"user.turn.{turn['turn']}"] = {"basis": "user_statement_not_computed", "value": turn["user"]}
    if context:
        for key, statement in statement_entries(ContextContent.model_validate(context["content"])):
            if statement.status != "withdrawn":
                result[f"context.{key}"] = {
                    "basis": statement.basis,
                    "status": statement.status,
                    "value": statement.value + " " + statement.qualifications,
                }
    for artifact in artifacts:
        for name, value in artifact["facts"].items():
            result[f"artifact.{artifact['id']}.{name}"] = {"basis": "computed", "value": value}
    if current:
        for name, value in current["facts"].items():
            result[f"exploration.{name}"] = {"basis": "computed", "value": value}
    return result


