"""Make plain JSON-Schema tool definitions safe for Vertex AI.

Vertex AI's function-calling `Schema` proto accepts only a fixed subset of
JSON-Schema keywords. Tool schemas authored as ordinary JSON Schema (e.g.
a `dict[str, str]` field becomes `additionalProperties`, Pydantic emits
`$defs`/`$ref`, unions emit `oneOf`/`allOf`) make the `FunctionDeclaration`
parser raise `ParseError`. The Gemini API and Claude accept those keys; only
Vertex is this strict, so the quirk is handled here in the Vertex adapter
rather than by rewriting every agent's tool schema.

The supported-key allowlist below is taken verbatim from the Vertex error
message ("Available Fields ..."), so we prune to exactly what the proto
accepts rather than guessing.
"""

from __future__ import annotations

from typing import Any

# Keys Vertex's Schema proto accepts. Everything else is dropped.
VERTEX_SCHEMA_KEYS = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "default",
        "items",
        "minItems",
        "maxItems",
        "enum",
        "properties",
        "propertyOrdering",
        "required",
        "minProperties",
        "maxProperties",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "example",
        "anyOf",
    }
)


def sanitize_vertex_schema(node: Any) -> Any:
    """Return a copy of a JSON-Schema node with Vertex-unsupported keys removed.

    Recurses through the structural keys that hold nested schemas
    (`properties` values, `items`, `anyOf` entries). Keys whose values are
    literal data, not schemas (`enum`, `required`, `default`, `example`), are
    kept verbatim. Non-dict input is returned unchanged so callers can pass a
    whole parameters block without pre-checking its shape.
    """
    if isinstance(node, list):
        return [sanitize_vertex_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key not in VERTEX_SCHEMA_KEYS:
            # additionalProperties, $defs, $ref, const, oneOf, allOf, ...
            continue
        if key == "properties" and isinstance(value, dict):
            # Keys here are caller-defined field names, not schema keywords,
            # so keep every key and sanitize only the schema values.
            cleaned[key] = {k: sanitize_vertex_schema(v) for k, v in value.items()}
        elif key == "items":
            cleaned[key] = sanitize_vertex_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            cleaned[key] = [sanitize_vertex_schema(v) for v in value]
        else:
            cleaned[key] = value
    return cleaned
