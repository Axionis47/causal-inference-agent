"""The field catalogue: the kinds of claim, their fields, the checks, the family needs. Loaded from fields.yaml and checks.yaml.

Loaders for the interview's knowledge: the claim kinds, the family needs, and the thresholds."""

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
    required_when: dict[str, list[Any]] = Field(default_factory=dict, description="field -> values of a sibling field under which this optional field is required")


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

    def required(self, values: dict[str, Any] | None = None) -> list[str]:
        """The fields that must be settled, given the sibling values known so far: the always-required ones, plus any optional
        field whose `required_when` condition the known values meet."""
        values = values or {}
        out = []
        for name, spec in self.fields.items():
            if not spec.optional:
                out.append(name)
            elif spec.required_when and all(
                (values.get(k) is not None) if allowed == ["*"] else (str(values.get(k)).lower() in {str(v).lower() for v in allowed})
                for k, allowed in spec.required_when.items()
            ):
                out.append(name)
        return out


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
    raw = yaml.safe_load(Path(path or _HERE / "fields.yaml").read_text())
    kinds = {k: ClaimKind(name=k, **v) for k, v in raw["kinds"].items()}
    fams = {k: FamilyNeeds(name=k, **v) for k, v in raw["family_needs"].items()}
    return Catalogue(kinds=kinds, families=fams, method_words=list(raw.get("method_words") or []))


def load_thresholds(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or _HERE / "checks.yaml").read_text())
