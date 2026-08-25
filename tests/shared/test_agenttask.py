"""Tests for the model-task response schema (T-013 Amendment 2, D-063)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, create_model

from causal.shared.agenttask import result_schema
from causal.shared.envelope import AgentTaskResultV1, ClaimV1


class Anchor(BaseModel):
    label: str


class Draft(BaseModel):
    """A draft payload: one definition shared with the result schema, one of its own."""

    reason: str
    claim: ClaimV1
    anchor: Anchor


BARE: dict[str, Any] = AgentTaskResultV1.model_json_schema()


class TestResultSchema:
    def test_the_payload_property_becomes_the_draft(self) -> None:
        merged: Any = result_schema(Draft)
        assert set(merged["properties"]) == set(BARE["properties"])
        assert set(merged["properties"]["payload"]["properties"]) == {"reason", "claim", "anchor"}
        assert "$defs" not in merged["properties"]["payload"]

    def test_the_defs_union_keeps_both_sides(self) -> None:
        defs: Any = result_schema(Draft)["$defs"]
        assert set(defs) == set(BARE["$defs"]) | {"Anchor"}
        assert defs["ClaimV1"] == BARE["$defs"]["ClaimV1"]

    def test_a_colliding_definition_raises(self) -> None:
        clash = create_model("Clash", claim=(create_model("ClaimV1", note=(str, ...)), ...))
        with pytest.raises(ValueError, match="redefines"):
            result_schema(clash)

    def test_memoized_per_draft_and_the_source_schema_is_untouched(self) -> None:
        assert result_schema(Draft) is result_schema(Draft)
        assert result_schema(Draft) is not result_schema(Anchor)
        assert AgentTaskResultV1.model_json_schema() == BARE
