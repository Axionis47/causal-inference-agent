"""Typed cross-agent artifact schemas: the contracts agents exchange.

Agents communicate only through these models, committed to the run state
by the orchestrator. No agent reads another agent's internals.
"""
from .causal_spec import CausalSpec, Confidence, QuestionType, VariableRef
from .critique import (
    ClaimCritique,
    ClaimStrength,
    ConfirmationCard,
    ConfirmationItem,
    PlanCritique,
    PlanGateStatus,
)
from .dag import CausalDAG, CausalEdge, CausalNode, EdgeProvenance
from .design import (
    DesignCandidate,
    EligibilityState,
    LaneEligibility,
    MethodLane,
    MethodPlan,
    ToolEligibility,
)
from .diagnostics import (
    CheckStatus,
    DiagnosticCheck,
    DiagnosticsResult,
    RobustnessStatus,
    SensitivityResult,
)
from .dossier import (
    BANNED_ADJUSTMENT_ROLES,
    ColumnRole,
    ContextLedgerItem,
    DatasetDossier,
    LedgerStatus,
    RoleLabel,
)
from .eda import EDACheck, EDACheckStatus, EDAPlan, EDASummary
from .estimate import EffectEstimate, EstimateResult
from .profile import ColumnProfile, ProfileSummary
from .readiness import ReadinessResult
from .report import NotebookBuildResult, NotebookStatus, NotebookVerificationResult

__all__ = [
    "CausalSpec",
    "Confidence",
    "QuestionType",
    "VariableRef",
    "CausalDAG",
    "CausalEdge",
    "CausalNode",
    "EdgeProvenance",
    "ClaimCritique",
    "ClaimStrength",
    "ConfirmationCard",
    "ConfirmationItem",
    "PlanCritique",
    "PlanGateStatus",
    "DesignCandidate",
    "EligibilityState",
    "LaneEligibility",
    "MethodLane",
    "MethodPlan",
    "ToolEligibility",
    "BANNED_ADJUSTMENT_ROLES",
    "ColumnRole",
    "ContextLedgerItem",
    "DatasetDossier",
    "LedgerStatus",
    "RoleLabel",
    "CheckStatus",
    "DiagnosticCheck",
    "DiagnosticsResult",
    "RobustnessStatus",
    "SensitivityResult",
    "EDACheck",
    "EDACheckStatus",
    "EDAPlan",
    "EDASummary",
    "EffectEstimate",
    "EstimateResult",
    "ColumnProfile",
    "ProfileSummary",
    "ReadinessResult",
    "NotebookBuildResult",
    "NotebookStatus",
    "NotebookVerificationResult",
]
