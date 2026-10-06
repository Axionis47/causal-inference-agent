"""The one tool the exploration agent has: run a script in the sandbox. Each call is a step in its own folder under the
artifact's; what the script printed, the files it left and the numbers in its facts.json come back as the tool's text, and the
episode gives that text an address the record may cite."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from causal_agent.common import trace
from causal_agent.explore import gates
from causal_agent.explore.contracts import Step
from causal_agent.lane.tools import Refused, Result
from causal_agent.viz import sandbox

RUN_CODE = (
    "Run one complete Python script on the table at the path in VIZ_CSV, in a fresh folder of its own. Back comes the tail of "
    "what it printed, the files it wrote, and the numbers in facts.json when it wrote one. pandas, numpy, scipy, statsmodels and "
    "matplotlib (Agg) are there; no network; the table is read-only; nothing of an earlier step is on disk."
)


class RunCodeStep(BaseModel):
    """One step of looking: the next script to run, or done. Asked for as a record, not a function call, because a model that
    carries a script inside a function call mangles it."""

    done: bool = Field(default=False, description="true when you have what was asked for and will answer; the code is then empty")
    code: str = Field(default="", description="the complete Python script to run next, and nothing else: it starts with its first import")


@dataclass
class CodeTools:
    """The tool set over one table, for `run_episode`: one tool, `run_code`, looked at one step at a time as a record, and the
    steps it ran."""

    csv: Path
    out_dir: Path
    timeout: float
    stdout_tail: int
    dropped_name: str | None = None
    steps: list[Step] = field(default_factory=list)

    def names(self) -> tuple[str, ...]:
        return ("run_code",)

    def step_schema(self) -> type[BaseModel]:
        return RunCodeStep

    def brief_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The script by its length in the fact's line; the script itself is on disk and in the tool's span."""
        code = str(args.get("code", ""))
        return {"code": f"a script of {len(code.splitlines())} lines"}

    def schemas(self) -> list[Any]:
        return [StructuredTool.from_function(func=self.run_code, name="run_code", description=RUN_CODE)]

    def call(self, name: str, args: dict[str, Any]) -> Result:
        if name not in self.names():
            raise Refused(f"refused: no tool named {name!r}; the only tool is run_code")
        try:
            return self.run_code(**args)
        except TypeError as e:
            raise Refused(f"refused: run_code could not run with {args!r}: {e}") from e

    def run_code(self, code: str) -> Result:
        """Run one complete Python script on the table; back comes what it printed, the files it wrote, and its facts.json."""
        why = gates.not_a_script(code) or gates.names_dropped_column(code, self.dropped_name)
        if why:
            raise Refused("refused: " + why)
        n = len(self.steps) + 1
        out = self.out_dir / f"step_{n}"
        out.mkdir(parents=True, exist_ok=True)
        ran = sandbox.run(code, self.csv, out, timeout=self.timeout)
        trace.write(out / "stdout.txt", ran.stdout)
        step = Step(n=n, ok=ran.ok, stdout=ran.stdout, stderr=ran.stderr, files=sorted(p.name for p in out.iterdir()), facts=_facts(out), kind=ran.kind)
        self.steps.append(step)
        return Result(self.render(step), None)

    def render(self, step: Step) -> str:
        head = f"step {step.n} {'ran' if step.ok else 'failed'} ({step.kind})."
        out = step.stdout[-self.stdout_tail :].strip()
        parts = [head, f"stdout (last {self.stdout_tail} chars): {out or '(nothing printed)'}"]
        if not step.ok:
            parts.append(f"stderr: {step.stderr[-self.stdout_tail :].strip() or '(nothing)'}")
        parts.append("files: " + (", ".join(f for f in step.files if f != "stdout.txt") or "(none)"))
        if step.facts is not None:
            parts.append("facts.json: " + ", ".join(f"{k}={v:.6g}" for k, v in step.facts.items()))
        return "\n".join(parts)

    def step(self, n: int) -> Step | None:
        return self.steps[n - 1] if 1 <= n <= len(self.steps) else None


def _facts(out: Path) -> dict[str, float] | None:
    """The numbers in a step's facts.json, or None when there is none or it is not an object of numbers."""
    p = out / "facts.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or any(isinstance(v, bool) or not isinstance(v, int | float) for v in doc.values()):
        return None
    return {str(k): float(v) for k, v in doc.items()}
