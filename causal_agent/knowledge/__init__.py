from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

_HERE = Path(__file__).parent


class Family(BaseModel):
    name: str
    applies_to: list[str]
    answers: str
    needs: list[str]
    look_for: str
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    convince: str = Field(default="", description="the point a figure makes at the ready moment, in the question's words")
    specialist: str
    status: Literal["built", "declared"]

    def render(self) -> str:
        needs = "\n".join(f"    - {n}" for n in self.needs)
        return (
            f"family: {self.name}  (status: {self.status})\n"
            f"  applies to questions of kind: {', '.join(self.applies_to)}\n"
            f"  answers: {self.answers}\n"
            f"  needs:\n{needs}\n"
            f"  where the evidence usually lives: {self.look_for}\n"
            f"  assumes: {self.assumes}\n"
            f"  weak when: {self.weak_when}"
        )


def load_registry(path: str | Path | None = None) -> list[Family]:
    raw = yaml.safe_load(Path(path or _HERE / "families.yaml").read_text())
    return [Family(name=k, **v) for k, v in raw.items()]


def family_names(registry: list[Family]) -> list[str]:
    return [f.name for f in registry]


def render_preferences(registry: list[Family]) -> str:
    lines = []
    for f in registry:
        for other, why in f.prefer_over.items():
            lines.append(f"- prefer {f.name} over {other}: {why}")
    return "\n".join(lines) or "(none recorded)"
