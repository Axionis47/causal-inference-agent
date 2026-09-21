"""What the runner needs to know about a family's evals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalSpec:
    family: str
    dataset: str  # the LangSmith dataset name
    description: str
    prefix: str  # the experiment prefix, and the lane tag
    cases_dir: Path  # holds cases.yaml and handoffs/
    lane_graph: Callable[[], Any]  # the family's lane, compiled with a local checkpointer
    summarise: Callable[[dict], dict]  # the lane's result as the graded summary
    evaluators: list[Callable[..., dict]]
    skip_keys: tuple[str, ...] = ("report", "design")  # streamed keys too long to print
