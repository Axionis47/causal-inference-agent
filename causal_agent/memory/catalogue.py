"""The field catalogue: the kinds of claim, their fields, the checks, and the thresholds. Loaded from fields.yaml and
checks.yaml. FamilyNeeds is the shape of what a family asks of the catalogue; the families themselves say which."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_HERE = Path(__file__).parent


class FieldSpec(BaseModel):
    type: str  # text | choice | bool | number | column | columns | column_or_none
    options: list[Any] = Field(default_factory=list)
    about: dict[str, str] = Field(default_factory=dict)  # what each option means, in world terms
    hint: str = ""  # what to put in the field, in world terms
    optional: bool = False
    required_when: dict[str, list[Any]] = Field(
        default_factory=dict, description="field -> values of a sibling field under which this optional field is required"
    )
    asked_when: dict[str, list[Any]] = Field(
        default_factory=dict, description="field -> values of a sibling field under which a decision that rests on this field has it asked"
    )

    def holds(self, condition: dict[str, list[Any]], values: dict[str, Any]) -> bool:
        return all(
            (values.get(k) is not None) if allowed == ["*"] else (str(values.get(k)).lower() in {str(v).lower() for v in allowed})
            for k, allowed in condition.items()
        )


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
            elif spec.required_when and spec.holds(spec.required_when, values):
                out.append(name)
        return out


    def asked(self, field: str, values: dict[str, Any] | None = None) -> bool:
        """Whether a decision that rests on this field has it asked, given the sibling values known so far: always, unless the
        field says `asked_when` and the condition does not hold."""
        spec = self.fields.get(field)
        if spec is None:
            return False
        values = values or {}
        if spec.required_when and not spec.holds(spec.required_when, values):
            return False  # a cutoff is asked once the rule is a cutoff rule, not before
        return not spec.asked_when or spec.holds(spec.asked_when, values)

class FamilyNeeds(BaseModel):
    """The claim kinds a family requires, and for some fields which values fit."""

    name: str
    requires: list[str]
    fits: dict[str, list[Any]] = Field(default_factory=dict)

    asks: list[str] = Field(
        default_factory=list,
        description="the field addresses the family's decisions rest on, as patterns; <column> stands for any column in play. The interview asks them because a decision needs them",
    )

class Catalogue(BaseModel):
    kinds: dict[str, ClaimKind]
    method_words: list[str]

    def ordered(self) -> list[ClaimKind]:
        return sorted(self.kinds.values(), key=lambda k: k.order)


def load_catalogue(path: str | Path | None = None) -> Catalogue:
    raw = yaml.safe_load(Path(path or _HERE / "fields.yaml").read_text())
    kinds = {k: ClaimKind(name=k, **v) for k, v in raw["kinds"].items()}
    return Catalogue(kinds=kinds, method_words=list(raw.get("method_words") or []))


def load_thresholds(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or _HERE / "checks.yaml").read_text())
