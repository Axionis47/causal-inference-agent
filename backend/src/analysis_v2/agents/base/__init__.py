"""Shared agent machinery: the contract, the dispatch context, the loop."""
from .agent import AgentResult, AnalysisAgent
from .context import AgentCtx
from .loop import LoopResult, LoopTool, react_loop

__all__ = [
    "AgentResult",
    "AnalysisAgent",
    "AgentCtx",
    "LoopResult",
    "LoopTool",
    "react_loop",
]
