"""Isolated diagnostic code and numerical evidence gates shared by the lanes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from uuid import uuid4

import pandas as pd

from causal_agent.common import trace
from causal_agent.explore.contracts import Step
from causal_agent.explore.tools import CodeTools
from causal_agent.lane.episode import EpisodeLog

DIAGNOSTIC_RULE = """You may run arbitrary Python for an unresolved numerical diagnostic using run_code. Existing evidence can suffice without code. Read only os.environ['DATA_CSV']; the supplied projection is the entire permitted data scope. Use the actual projected column names, not original source names from the study account. Each script is a complete fresh Python program beginning with its imports: preserve actual newlines and indentation, without surrounding quotes, prose or markdown fences, and use explicit observed category labels. Inspecting schema alone is not diagnostic evidence. If you use code, compute a relevant diagnostic and WRITE facts.json as a nonempty JSON object containing only finite numbers; printing JSON alone is insufficient. Convert pandas/NumPy scalar results to Python int or float before json.dump; do not construct JSON manually. Cite the successful returned probe: address in your record. Failed scripts and missing, mixed-text or nonfinite facts cannot support a decision. Repair within the declared budget or stop. Do not estimate target effects, choose by significance, inspect other files, change the data, or reinterpret a diagnostic as proof of causal assumptions. New probe: addresses returned by this episode are permitted citations even though they were absent from the initial view."""


@dataclass
class DiagnosticTools(CodeTools):
    required: bool = True
    input_columns: tuple[str, ...] = ()

    def input_summary(self) -> str:
        columns = ", ".join(repr(column) for column in self.input_columns)
        return f"DATA_CSV actual column names: {columns}. Use exactly these names in Python; study-account source names are not extra CSV aliases."

    @staticmethod
    def usable(step: Step) -> bool:
        facts = step.facts
        return step.ok and facts is not None and bool(facts) and all(isfinite(value) for value in facts.values())

    def completion_errors(self, log: EpisodeLog) -> list[str]:
        if not self.required and not log.calls:
            return []
        return [] if any(self.usable(step) for step in self.steps) else ["no successful numerical diagnostic: run_code must write nonempty finite facts.json"]

    def run_code(self, code: str):
        result = super().run_code(code)
        result = result._replace(text=result.text + "\n" + self.input_summary())
        step = self.steps[-1]
        folder = self.out_dir / f"step_{step.n}"
        for name in step.files:
            path = folder / name
            if name not in {"code.py", "stdout.txt"} and path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1:
                trace.write(path, path.read_bytes())
        if not self.usable(step):
            return result._replace(
                text=result.text
                + "\nNot admissible diagnostic evidence. The next script must compute a relevant diagnostic and write finite numerical facts.json."
            )
        return result

    def evidence_errors(self, cites: list[str], log: EpisodeLog) -> list[str]:
        successful: set[str] = set()
        failed: set[str] = set()
        for fact, step in zip(log.facts, self.steps, strict=True):
            (successful if self.usable(step) else failed).add(fact.address)
        errors = []
        if (self.required or log.calls) and not successful.intersection(cites):
            errors.append(
                "cite a successful run_code diagnostic with finite numerical facts.json; permitted supporting probes: "
                + (", ".join(sorted(successful)) or "none")
            )
        errors.extend(f"failed diagnostic or missing finite numerical facts at {address} cannot support the record" for address in failed.intersection(cites))
        return errors


def tools_for(table: pd.DataFrame, root: Path, *, dropped_name: str | None = None, required: bool = True) -> DiagnosticTools:
    """The caller supplies its permitted projection; only that CSV is exposed to code."""
    folder = root / uuid4().hex[:12]
    csv = folder / "input" / "table.csv"
    trace.write(csv, table.to_csv(index=False))
    return DiagnosticTools(
        csv=csv,
        out_dir=folder,
        timeout=30,
        stdout_tail=4000,
        dropped_name=dropped_name,
        required=required,
        input_columns=tuple(str(c) for c in table.columns),
    )
