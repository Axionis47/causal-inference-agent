"""System prompt placeholder for dataset_inspector.

The inspector itself is procedural: it dispatches data_profiler per
candidate file and runs a deterministic scorer. There is no inner LLM
reasoning step on the inspector's own behalf — the LLM work happens
inside the data_profiler invocations it fans out to. This file exists
to match the folder shape required by CLAUDE.md §2.3 so future
maintainers find the same layout across every specialist.
"""
from __future__ import annotations

SYSTEM_PROMPT = ""
