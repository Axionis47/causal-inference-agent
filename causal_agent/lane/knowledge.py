"""A lane's knowledge files, read once and rendered for the model. A lane declares its entry models and binds a Knowledge
to its folder; the loaders, the lookups by name, the first applicable inference, and the preference lines are shared."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


class Knowledge:
    def __init__(
        self,
        directory: Path,
        *,
        estimator: type[Any],
        inference: type[Any] | None = None,
        placebo: type[Any] | None = None,
        refuter: type[Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self._cls = {"estimators": estimator, "inference": inference, "placebos": placebo, "refuters": refuter}
        self._cache: dict[str, Any] = {}

    def _entries(self, file: str) -> list[Any]:
        """The file's entries as its model, keyed by name, in file order (file order is precedence)."""
        if file not in self._cache:
            cls = self._cls[file]
            if cls is None:
                raise LookupError(f"this lane declares no {file}")
            raw = yaml.safe_load((self.directory / f"{file}.yaml").read_text())
            self._cache[file] = [cls(name=k, **v) for k, v in raw.items()]
        return list(self._cache[file])

    def _yaml(self, file: str) -> dict[str, Any]:
        if file not in self._cache:
            self._cache[file] = yaml.safe_load((self.directory / f"{file}.yaml").read_text()) or {}
        return self._cache[file]

    # ------------------------------------------------------------------ the files

    def estimators(self) -> list[Any]:
        return sorted(self._entries("estimators"), key=lambda e: e.rank)

    def inference(self) -> list[Any]:
        return self._entries("inference")

    def placebos(self) -> list[Any]:
        return self._entries("placebos")

    def refuters(self) -> list[Any]:
        return self._entries("refuters")

    def checks(self) -> dict[str, Any]:
        return self._yaml("checks")

    def beliefs(self) -> dict[str, Any]:
        """What the person's beliefs, the unknowns, and the contradictions mean to this lane (see lane.case)."""
        return self._yaml("beliefs")

    # ------------------------------------------------------------------ lookups

    def estimator(self, name: str) -> Any:
        return self._by_name(self.estimators(), name)

    def placebo(self, name: str) -> Any:
        return self._by_name(self.placebos(), name)

    def refuter(self, name: str) -> Any:
        return self._by_name(self.refuters(), name)

    @staticmethod
    def _by_name(entries: Sequence[Any], name: str) -> Any:
        for e in entries:
            if e.name == name:
                return e
        raise KeyError(name)

    def pick_inference(self, **facts: Any) -> Any:
        """The first inference entry whose applies() holds for the facts; file order is precedence."""
        for entry in self.inference():
            if entry.applies(**facts):
                return entry
        raise LookupError("no inference entry applies")

    @staticmethod
    def render_preferences(entries: Sequence[Any]) -> str:
        """The estimators' stated preferences over one another, among the entries given."""
        names = {e.name for e in entries}
        lines = [f"- prefer {e.name} over {o}: {why}" for e in entries for o, why in e.prefer_over.items() if o in names]
        return "\n".join(lines) or "(none recorded among these)"
