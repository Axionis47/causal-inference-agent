"""Stage -> agent registry: which worker completes each spine stage.

Stages without an entry are the build frontier; the spine stops there
honestly (see graph.py) instead of pretending the run finished. S0, S6,
and S12 never have agents: S0 is the pickup point, S6 is the orchestrator-
owned approval transition, S12 is terminal.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import AnalysisAgent
from src.analysis_v2.agents.claim_critic import ClaimCriticAgent
from src.analysis_v2.agents.design_detection import DesignDetectionAgent
from src.analysis_v2.agents.diagnostics import DiagnosticsSensitivityAgent
from src.analysis_v2.agents.intake import IntakeAgent
from src.analysis_v2.agents.investigator import InvestigatorAgent
from src.analysis_v2.agents.method_lane import MethodLaneAgent
from src.analysis_v2.agents.notebook_verify import NotebookVerificationAgent
from src.analysis_v2.agents.plan_critic import PlanCriticAgent
from src.analysis_v2.agents.profiling import ProfilingAgent
from src.analysis_v2.agents.readiness import ReadinessAgent
from src.analysis_v2.agents.report import ReportNotebookAgent
from src.analysis_v2.agents.targeted_eda import TargetedEDAAgent
from src.analysis_v2.core import AnalysisStage


def build_agents() -> dict[AnalysisStage, AnalysisAgent]:
    """Fresh agent instances per run; no cross-job shared state."""
    agents: list[AnalysisAgent] = [
        IntakeAgent(),
        ProfilingAgent(),
        InvestigatorAgent(),
        DesignDetectionAgent(),
        TargetedEDAAgent(),
        PlanCriticAgent(),
        ReadinessAgent(),
        MethodLaneAgent(),
        DiagnosticsSensitivityAgent(),
        ClaimCriticAgent(),
        ReportNotebookAgent(),
        NotebookVerificationAgent(),
    ]
    return {agent.stage: agent for agent in agents}
