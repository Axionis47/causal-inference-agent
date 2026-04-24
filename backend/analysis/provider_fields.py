"""Short wire keys for the large Treatments grammar, with exact reversible decoding."""

from copy import deepcopy
from typing import Any


def compact(schema: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            # Visit child schemas before changing the properties map.
            for item in list(node.values()):
                visit(item)
            if node.get("type") == "object" and "properties" in node:
                names = {name: f"p{index}" for index, name in enumerate(node["properties"])}
                properties = {}
                for name, field in node["properties"].items():
                    field = deepcopy(field)
                    # Repeating arbitrary evidence hashes throughout this grammar
                    # exceeds the provider's structured-output compiler limits.
                    # Keep these finite selections in the packet and enforce the
                    # canonical enum while decoding, before scientific admission.
                    if name in {"citations", "references", "prerequisites", "column_ids", "history_variables",
                                "unit_variable_id", "time_variable_id", "adoption_variable_id",
                                "assignment_variable_id", "running_variable_id"}:
                        field.pop("enum", None)
                        field.get("items", {}).pop("enum", None)
                    # A $ref stays standalone; put its explanation on a wrapper.
                    if "$ref" in field:
                        field = {"anyOf": [field]}
                    field["description"] = f"{name}. " + field.get("description", "")
                    properties[names[name]] = field
                node["properties"] = properties
                if "required" in node:
                    node["required"] = [names[name] for name in node["required"]]
    visit(result)
    result["description"] = (
        "Use the compact property keys exactly as shown. Each field description starts "
        "with its full application field name. Interpret instructions using those full names."
    )
    return result


def expand(value: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Reject unknown/missing wire keys; never repair names or rewrite literal values."""
    def visit(item: Any, node: dict[str, Any]) -> Any:
        if "enum" in node and item not in node["enum"]:
            raise ValueError("A compact field selected a value outside its canonical choices.")
        if "$ref" in node:
            return visit(item, schema["$defs"][node["$ref"].removeprefix("#/$defs/")])
        if "anyOf" in node:
            variants = node["anyOf"]
            if item is None and any(branch.get("type") == "null" for branch in variants):
                return None
            nonnull = [branch for branch in variants if branch.get("type") != "null"]
            if len(nonnull) != 1:
                raise ValueError("Compact fields require one non-null schema branch.")
            return visit(item, nonnull[0])
        if node.get("type") == "object":
            if not isinstance(item, dict):
                raise ValueError("Compact object fields require an object.")
            names = {f"p{index}": name for index, name in enumerate(node["properties"])}
            if set(item) - names.keys():
                raise ValueError("Unknown compact field; canonical or invented field names are not accepted.")
            present = {names[key] for key in item}
            if set(node.get("required", [])) - present:
                raise ValueError("A required compact field is missing.")
            return {names[key]: visit(child, node["properties"][names[key]]) for key, child in item.items()}
        if node.get("type") == "array":
            if not isinstance(item, list):
                raise ValueError("Compact array fields require a list.")
            return [visit(child, node["items"]) for child in item]
        return deepcopy(item)

    decoded = visit(value, schema)
    if not isinstance(decoded, dict):
        raise ValueError("The compact response must be an object.")
    return decoded
