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
from .critique import CritiqueAgent
from .orchestrator import OrchestratorAgent, ReActOrchestrator, StandardOrchestrator
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
from .specialists.effect_estimator_react import EffectEstimatorReActAgent

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
    "OrchestratorAgent",
    "ReActOrchestrator",
    "StandardOrchestrator",
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
