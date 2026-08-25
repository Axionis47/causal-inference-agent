"""Prompt requirement vocabulary matches the registry (T-013 Amendment 3; D-064)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = json.loads(
    (ROOT / "registries" / "context-requirements.v1.json").read_text(encoding="utf-8"))
REGISTERED: set[str] = {str(row["requirement_id"]) for row in _REGISTRY["requirements"]}
MENTIONED = re.compile(r"\b(?:design|dataset|column)\.[a-z_]+\b")


def test_prompt_requirement_ids_are_registered() -> None:
    for template in sorted((ROOT / "prompts" / "design").glob("*.v1.txt")):
        found = set(MENTIONED.findall(template.read_text(encoding="utf-8")))
        assert found <= REGISTERED, f"{template.name}: {sorted(found - REGISTERED)}"
        assert found, f"{template.name} names no registered requirement id"
