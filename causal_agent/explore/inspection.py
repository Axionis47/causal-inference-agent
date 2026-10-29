"""Task-scoped code and inspection actions, including image pixels for the model."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field

from causal_agent.common import trace
from causal_agent.explore import sandbox
from causal_agent.explore.contracts import Step
from causal_agent.explore.inputs import safe_file
from causal_agent.explore.tools import CodeTools
from causal_agent.lane.tools import Refused, Result


class ExploreAction(BaseModel):
    """One action to execute now, or completion after the tool evidence answers the question."""

    done: bool = Field(default=False, description="False to execute this action. True stops tool use without executing code or inspecting anything.")
    action: Literal["inspect_inputs", "run_code", "inspect_output"] = Field(
        default="inspect_inputs",
        description="inspect_inputs opens supplied dataset/artifact filenames. run_code executes Python. inspect_output opens only step_N/filename generated in this task.",
    )
    code: str = Field(default="", description="For run_code: complete Python source to execute now. Set done=false. Otherwise empty.")
    reference: str = Field(default="", description="input filename or step_N/filename from the returned manifest")


@dataclass
class ExplorationTools(CodeTools):
    manifest: dict[str, dict[str, str]] = field(default_factory=dict)
    inspections: int = 0
    executions: int = 0
    code_budget: int = 8
    inspection_budget: int = 16
    inspected: set[str] = field(default_factory=set)
    _media: list[dict[str, Any]] = field(default_factory=list)

    def names(self) -> tuple[str, ...]:
        return ("explore_action",)

    def step_schema(self) -> type[BaseModel]:
        return ExploreAction

    def media(self) -> list[dict[str, Any]]:
        return self._media

    def brief_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("action") == "run_code":
            return {"action": "run_code", **super().brief_args(name, args)}
        return {"action": args.get("action"), "reference": args.get("reference", "")}

    def render(self, step: Step) -> str:
        text = super().render(step)
        if step.ok:
            text += "\nOutput references: " + ", ".join(f"step_{step.n}/{name}" for name in step.files)
        return text

    def call(self, name: str, args: dict[str, Any]) -> Result:
        if name != "explore_action":
            raise Refused("only task-scoped exploration actions are available")
        if self.cancel is not None and self.cancel.is_set():
            raise Refused("task cancelled")
        self._media = []
        action = ExploreAction.model_validate(args)
        if action.action == "run_code":
            if self.executions >= self.code_budget:
                raise Refused("the code execution budget is exhausted")
            self.executions += 1
            result = self.run_code(action.code)
            step = self.steps[-1]
            # Only successful bounded files become read-only inputs for later scripts.
            if step.ok and self.inputs is not None:
                for name in step.files:
                    path = self.out_dir / f"step_{step.n}" / name
                    if name not in {"code.py", "stdout.txt"}:
                        safe_file(path, path.parent, sandbox.MAX_FILE)
                        trace.write(self.inputs / f"step_{step.n}" / name, path.read_bytes())
            if step.ok and "figure.png" in step.files and self.inspections < self.inspection_budget:
                self.inspections += 1
                try:
                    reference = f"step_{step.n}/figure.png"
                    with trace.span("inspect_output", run_type="tool", inputs={"reference": reference, "automatic": True}) as span:
                        inspected = self.inspect(reference, output=True)
                        span.end(outputs={"text": inspected.text, "inspections_used": self.inspections})
                    return Result(result.text + "\n" + inspected.text + " Review the attached plot now. Do not regenerate it unless it needs correction.")
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    return Result(result.text + f"\nPlot inspection failed: {exc}")
            return result
        if self.inspections >= self.inspection_budget:
            raise Refused("the inspection budget is exhausted")
        self.inspections += 1
        if action.action == "inspect_inputs" and not action.reference:
            return Result(json.dumps(self.manifest, indent=2))
        try:
            return self.inspect(action.reference, output=action.action == "inspect_output")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            if action.action == "inspect_output" and action.reference in self.manifest:
                raise Refused(
                    f"inspect_output was refused: {action.reference!r} is a supplied input, not a generated output. "
                    f"Next use action='inspect_inputs', reference={action.reference!r}, done=false, code=''."
                ) from exc
            raise Refused(f"cannot inspect this task resource: {exc}") from exc

    def inspect(self, reference: str, *, output: bool) -> Result:
        if output:
            parts = Path(reference).parts
            if len(parts) != 2 or not parts[0].startswith("step_"):
                raise ValueError("use step_N/filename from a successful execution")
            step = self.step(int(parts[0][5:]))
            if step is None or not step.ok or parts[1] not in step.files:
                raise ValueError("not an output from a successful step")
            root = self.out_dir / parts[0]
            path = safe_file(root / parts[1], root, sandbox.MAX_FILE)
        else:
            if reference not in self.manifest or self.inputs is None:
                raise ValueError("not a supplied input filename")
            root = self.inputs
            path = safe_file(root / reference, root, 100 * 1024 * 1024)
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as im:
                if im.width * im.height > 20_000_000:
                    raise ValueError("image exceeds inspection pixel limit")
                im.load()
                im.thumbnail((1600, 1600))
                buffer = io.BytesIO()
                im.convert("RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            self._media = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]
            text = "Image pixels attached for inspection."
        elif suffix in {".csv", ".tsv"}:
            df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=20)
            text = f"Preview only, first {len(df)} rows; dtypes: {df.dtypes.to_dict()}\n{df.to_string(index=False)}"
        elif suffix in {".json", ".txt", ".py"}:
            with path.open() as stream:
                text = stream.read(8001)
            if len(text) > 8000:
                text = text[:8000] + "\n[preview truncated; use code to read the supplied file]"
        else:
            raise ValueError("unsupported inspection format")
        self.inspected.add(reference)
        return Result(f"{reference}: {text[:8500]}")
