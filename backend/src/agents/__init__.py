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
    JobStatus,
    ReActAgent,
    ReActStep,
    SensitivityResult,
    ToolResult,
    ToolResultStatus,
    TreatmentEffectResult,
)
from .critique import CritiqueAgent
from .orchestrator import OrchestratorAgent
from .orchestrator.react_orchestrator import ReActOrchestrator
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
