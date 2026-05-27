"""Base agent module."""

from .agent import BaseAgent
from .errors import AgentError, ToolExecutionError
from .context_tools import ContextTools
from .react_agent import ReActAgent, ReActStep, ToolResult, ToolResultStatus
from src.analysis.agents.causal_discovery import CausalDAG, CausalEdge, CausalPair
from src.analysis.agents.eda import EDAResult
from src.analysis.agents.effect_estimator import TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst import SensitivityResult
from .state import (
    AgentTrace,
    AnalysisDecision,
    AnalysisState,
    CritiqueDecision,
    CritiqueFeedback,
    DataProfile,
    DatasetInfo,
    FileEntry,
    JobStatus,
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
    "FileEntry",
    "JobStatus",
    "SensitivityResult",
    "TreatmentEffectResult",
    "TreatmentEncoding",
]
