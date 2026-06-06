"""Agents module - The agentic system for causal inference."""

from .base import (
    AgentTrace,
    AnalysisState,
    BaseAgent,
    CausalDAG,
    CausalEdge,
    CritiqueDecision,
    CritiqueFeedback,
    DataProfile,
    DatasetInfo,
    EDAResult,
    FileEntry,
    JobStatus,
    ReActAgent,
    ReActStep,
    SensitivityResult,
    ToolResult,
    ToolResultStatus,
    TreatmentEffectResult,
)
from .critique.agent import CritiqueAgent
from .specialists import (
    CausalDiscoveryAgent,
    ConfounderDiscoveryAgent,
    DataProfilerAgent,
    DataRepairAgent,
    DomainKnowledgeAgent,
    EDAAgent,
    EffectEstimatorAgent,
    NotebookGeneratorAgent,
    PSDiagnosticsAgent,
    SensitivityAnalystAgent,
)
from .effect_estimator.react import EffectEstimatorReActAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentTrace",
    "AnalysisState",
    "CausalDAG",
    "CausalEdge",
    "CritiqueDecision",
    "CritiqueFeedback",
    "DataProfile",
    "DatasetInfo",
    "EDAResult",
    "FileEntry",
    "JobStatus",
    "SensitivityResult",
    "TreatmentEffectResult",
    # ReAct framework
    "ReActAgent",
    "ReActStep",
    "ToolResult",
    "ToolResultStatus",
    # Agents
    "DataProfilerAgent",
    "DataRepairAgent",
    "DomainKnowledgeAgent",
    "EDAAgent",
    "CausalDiscoveryAgent",
    "ConfounderDiscoveryAgent",
    "EffectEstimatorAgent",
    "EffectEstimatorReActAgent",
    "PSDiagnosticsAgent",
    "SensitivityAnalystAgent",
    "NotebookGeneratorAgent",
    "CritiqueAgent",
]
