"""Tests for the Vertex tool-schema sanitizer."""

from src.llm.schema_sanitize import sanitize_vertex_schema


def test_strips_additional_properties_but_keeps_named_properties():
    # Shape mirrors the eda finalize tool's plot_captions field, which broke
    # the live run: an object carrying both explicit properties and the
    # additionalProperties key Vertex rejects.
    schema = {
        "type": "object",
        "description": "captions",
        "additionalProperties": {"type": "string"},
        "properties": {
            "distribution": {"type": "string", "description": "a caption"},
        },
    }

    cleaned = sanitize_vertex_schema(schema)

    assert "additionalProperties" not in cleaned
    assert cleaned["properties"]["distribution"] == {
        "type": "string",
        "description": "a caption",
    }
    assert cleaned["type"] == "object"
    assert cleaned["description"] == "captions"


def test_strips_pydantic_ref_and_union_keywords_recursively():
    schema = {
        "type": "object",
        "$defs": {"Inner": {"type": "string"}},
        "properties": {
            "items_field": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "union_field": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "oneOf": [{"type": "string"}],
            },
        },
    }

    cleaned = sanitize_vertex_schema(schema)

    assert "$defs" not in cleaned
    assert "additionalProperties" not in cleaned["properties"]["items_field"]["items"]
    union = cleaned["properties"]["union_field"]
    assert "oneOf" not in union
    assert union["anyOf"] == [{"type": "string"}, {"type": "integer"}]


def test_preserves_property_named_like_a_schema_keyword():
    # A field literally named "enum" or "type" is a property key, not a schema
    # keyword. The sanitizer must not prune it from properties.
    schema = {
        "type": "object",
        "properties": {
            "enum": {"type": "string"},
            "type": {"type": "integer"},
        },
    }

    cleaned = sanitize_vertex_schema(schema)

    assert set(cleaned["properties"]) == {"enum", "type"}


def test_keeps_enum_and_required_values_verbatim():
    schema = {
        "type": "string",
        "enum": ["ready", "needs_attention"],
        "required": ["a", "b"],
    }

    cleaned = sanitize_vertex_schema(schema)

    assert cleaned["enum"] == ["ready", "needs_attention"]
    assert cleaned["required"] == ["a", "b"]


def test_non_dict_input_passes_through():
    assert sanitize_vertex_schema("x") == "x"
    assert sanitize_vertex_schema(None) is None
    assert sanitize_vertex_schema({}) == {}
