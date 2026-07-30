"""Immutable prompt loading and lineage records."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).parent.parent / "prompts"


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    version: str
    sha256: str
    path: str
    text: str

    def lineage(self) -> dict[str, str]:
        value = asdict(self)
        value.pop("text")
        return value


def load(prompt_id: str, version: str) -> PromptRecord:
    path = ROOT / prompt_id / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt {prompt_id}@{version}: {path}")
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    return PromptRecord(prompt_id, version, digest, str(path), text)
