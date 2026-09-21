"""How a lane's graph is compiled: as a node inside the desk's graphs, or standalone for tests and the command line. The
retry policy every judgement node shares lives here too."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from causal_agent.common.llm import RETRY as RETRY  # the policy every judgement node shares


def compile_subgraph(build: Callable[[], StateGraph]) -> Any:
    """As a node inside the desk's graphs: no checkpointer of its own, no interrupts."""
    return build().compile(checkpointer=False)


def compile_local(build: Callable[[], StateGraph]) -> Any:
    """Standalone, for tests and the command line: an in-memory checkpointer, a thread_id per run."""
    return build().compile(checkpointer=InMemorySaver())
