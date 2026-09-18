"""Loaders for the interview's knowledge: the claim kinds, the family needs, and the thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_HERE = Path(__file__).parent


class FieldSpec(BaseModel):
    type: str  # text | choice | bool | number | column | columns
    options: list[Any] = Field(default_factory=list)
    about: dict[str, str] = Field(default_factory=dict)  # what each option means, in world terms
    hint: str = ""  # what to put in the field, in world terms
    optional: bool = False


class ClaimKind(BaseModel):
    name: str
    about: str
    fields: dict[str, FieldSpec]
    check: str = "none"
    frame: str
    cues: str = ""
    order: int = 99
    per_column: bool = False
    uncheckable: bool = False

    def legal(self, field: str) -> list[Any]:
        return list(self.fields[field].options) if field in self.fields else []


class FamilyNeeds(BaseModel):
    name: str
    requires: list[str]
    fits: dict[str, list[Any]] = Field(default_factory=dict)


class Catalogue(BaseModel):
    kinds: dict[str, ClaimKind]
    families: dict[str, FamilyNeeds]
    method_words: list[str]

    def ordered(self) -> list[ClaimKind]:
        return sorted(self.kinds.values(), key=lambda k: k.order)


def load_catalogue(path: str | Path | None = None) -> Catalogue:
    raw = yaml.safe_load(Path(path or _HERE / "claims.yaml").read_text())
    kinds = {k: ClaimKind(name=k, **v) for k, v in raw["kinds"].items()}
    fams = {k: FamilyNeeds(name=k, **v) for k, v in raw["family_needs"].items()}
    return Catalogue(kinds=kinds, families=fams, method_words=list(raw.get("method_words") or []))


def load_thresholds(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or _HERE / "checks.yaml").read_text())
