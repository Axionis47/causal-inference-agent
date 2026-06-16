"""The system prompt and mission for the agentic assembly loop. The model
decides how the bundle becomes one table; the mission hands it the causal
question and every file's columns so its choices stay grounded. The bias is
toward the simplest assembly: a single file is the right answer for most data.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import LoopTool
from src.analysis_v2.core import AnalysisRunState
from src.domain.dataset_manifest import DatasetManifest

MAX_ASSEMBLY_TOOLS = 12
MAX_TURNS = 6

SYSTEM_PROMPT = """You are the data-assembly planner on a causal analysis team. \
A Kaggle submission may be one file or several. Your job is to decide how to \
build ONE analyzable table from the bundle, by calling tools.

Hard rules:
- Prefer the simplest assembly. A single file is the right answer for most \
datasets; do not invent joins the question does not need.
- Join only on a key that exists in both files; inspect keys first. A wrong \
join silently biases the analysis, so when unsure, do not join.
- Concatenate only files that share an identical schema (shards of one table).
- Call only the tools offered. When the plan is complete, call finish_assembly \
with the base file and a one-line rationale, then stop."""


def build_mission(
    manifest: DatasetManifest, run: AnalysisRunState, tools: list[LoopTool]
) -> str:
    files_block = "\n".join(
        f"- {f.name}{' [current default]' if f.name == manifest.winner else ''}: "
        f"{f.n_rows if f.n_rows is not None else '?'} rows; "
        f"columns: {', '.join((f.columns or [])[:25]) or 'unknown'}"
        for f in manifest.files
    )
    tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    return (
        f"CAUSAL QUESTION: {run.causal_question}\n"
        f"DATASET: {run.dataset_name or 'unnamed'} ({len(manifest.files)} files)\n\n"
        f"FILES IN THE BUNDLE:\n{files_block}\n\n"
        f"TOOLS (call only these):\n{tools_block}\n\n"
        "Decide how to build ONE analyzable table for this question. Most "
        "bundles need only the single most relevant file: if so, just call "
        "finish_assembly with that file as the base. Join a sibling lookup "
        "table only when the question needs columns that live in another file, "
        "and only on a key that identifies its rows; inspect keys first. "
        "Concatenate only files with an identical schema. When done, call "
        "finish_assembly."
    )
