"""Base agent module."""

from .agent import BaseAgent
from .errors import AgentError, ToolExecutionError
from .context_tools import ContextTools
from .react_agent import ReActAgent, ReActStep, ToolResult, ToolResultStatus
from .state import (
    AgentTrace,
    AnalysisDecision,
    AnalysisState,
    CausalDAG,
    CausalEdge,
    CausalPair,
    CritiqueDecision,
    CritiqueFeedback,
    DataProfile,
    DatasetInfo,
    EDAResult,
    JobStatus,
    SensitivityResult,
    TreatmentEffectResult,
    TreatmentEncoding,
)
__all__ = [
    # Base agents
    "BaseAgent",
    "AgentError",
    "ToolExecutionError",
    # ReAct agent
    "ReActAgent",
    "ReActStep",
    "ToolResult",
    "ToolResultStatus",
    # Context tools mixin
    "ContextTools",
    # State
    "AgentTrace",
    "AnalysisDecision",
    "AnalysisState",
    "CausalDAG",
    "CausalEdge",
    "CausalPair",
    "CritiqueDecision",
    "CritiqueFeedback",
    "DataProfile",
    "DatasetInfo",
    "EDAResult",
    "JobStatus",
    "SensitivityResult",
    "TreatmentEffectResult",
    "TreatmentEncoding",
]
