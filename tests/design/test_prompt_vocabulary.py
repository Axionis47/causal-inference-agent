"""Prompt requirement vocabulary matches the registry (T-013 Amendment 3; D-064)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = json.loads(
    (ROOT / "registries" / "context-requirements.v1.json").read_text(encoding="utf-8"))
REGISTERED: set[str] = {str(row["requirement_id"]) for row in _REGISTRY["requirements"]}
SCHEMAS: dict[str, str] = {str(row["requirement_id"]): str(row["expected_answer_schema"])
                           for row in _REGISTRY["requirements"]}
MENTIONED = re.compile(r"\b(?:design|dataset|column)\.[a-z_]+\b")
LISTED = re.compile(r"`((?:design|dataset|column)\.[a-z_]+)` — [^`]*answer `([a-z-]+\.v1)`")


def test_prompt_requirement_ids_are_registered() -> None:
    for template in sorted((ROOT / "prompts" / "design").glob("*.v1.txt")):
        found = set(MENTIONED.findall(template.read_text(encoding="utf-8")))
        assert found <= REGISTERED, f"{template.name}: {sorted(found - REGISTERED)}"
        assert found, f"{template.name} names no registered requirement id"


def test_prompt_answer_schemas_match_the_registry() -> None:
    """The constants a model must author blind: registry version and each id's schema (D-065)."""
    for template in sorted((ROOT / "prompts" / "design").glob("*.v1.txt")):
        text = template.read_text(encoding="utf-8")
        assert "`context-requirements.v1`" in text, f"{template.name} states no registry version"
        listed = LISTED.findall(text)
        assert listed, f"{template.name} lists no answer schema"
        for requirement_id, schema in listed:
            assert schema == SCHEMAS[requirement_id], f"{template.name}: {requirement_id}"
