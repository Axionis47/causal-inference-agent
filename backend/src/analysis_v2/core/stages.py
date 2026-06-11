"""The linear spine of the analysis workflow.

Stages are the analysis slice's internal progression, recorded on the run
state. The public job status stays coarse (running / waiting_for_user /
completed); stages are what the orchestrator, the state-event log, and the
frontend tiles reason about. Forward by default; backward only on the
enumerated hard-gate failures (docs/analysis-slice/00-architecture.md §1).
"""
from __future__ import annotations

from enum import StrEnum


class AnalysisStage(StrEnum):
    S0_DATASET_SAVED = "s0_dataset_saved"
    S1_INTAKE_PARSED = "s1_intake_parsed"
    S2_PROFILE_CREATED = "s2_profile_created"
    S3_DESIGN_CANDIDATES_CREATED = "s3_design_candidates_created"
    S4_TARGETED_EDA_COMPLETE = "s4_targeted_eda_complete"
    S5_PLAN_CRITIQUED = "s5_plan_critiqued"
    S6_USER_CONFIRMED_OR_AUTO_APPROVED = "s6_user_confirmed_or_auto_approved"
    S7_METHOD_EXECUTED = "s7_method_executed"
    S8_DIAGNOSTICS_SENSITIVITY_COMPLETE = "s8_diagnostics_sensitivity_complete"
    S9_CLAIM_CRITIQUED = "s9_claim_critiqued"
    S10_REPORT_NOTEBOOK_CREATED = "s10_report_notebook_created"
    S11_NOTEBOOK_VERIFIED = "s11_notebook_verified"
    S12_JOB_COMPLETE = "s12_job_complete"


# The one allowed order. Position in this tuple defines "earlier than".
SPINE: tuple[AnalysisStage, ...] = tuple(AnalysisStage)

# Which agent owns the work that completes each stage. S0 is the pickup
# point (no agent); S6 is the human/auto approval transition owned by the
# orchestrator itself; S12 is terminal.
STAGE_AGENT: dict[AnalysisStage, str | None] = {
    AnalysisStage.S0_DATASET_SAVED: None,
    AnalysisStage.S1_INTAKE_PARSED: "intake",
    AnalysisStage.S2_PROFILE_CREATED: "profiling",
    AnalysisStage.S3_DESIGN_CANDIDATES_CREATED: "design_detection",
    AnalysisStage.S4_TARGETED_EDA_COMPLETE: "targeted_eda",
    AnalysisStage.S5_PLAN_CRITIQUED: "plan_critic",
    AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED: None,
    AnalysisStage.S7_METHOD_EXECUTED: "method_lane",
    AnalysisStage.S8_DIAGNOSTICS_SENSITIVITY_COMPLETE: "diagnostics_sensitivity",
    AnalysisStage.S9_CLAIM_CRITIQUED: "claim_critic",
    AnalysisStage.S10_REPORT_NOTEBOOK_CREATED: "report_notebook",
    AnalysisStage.S11_NOTEBOOK_VERIFIED: "notebook_verification",
    AnalysisStage.S12_JOB_COMPLETE: None,
}

TERMINAL_STAGE = AnalysisStage.S12_JOB_COMPLETE


def stage_index(stage: AnalysisStage) -> int:
    return SPINE.index(stage)


def next_stage(stage: AnalysisStage) -> AnalysisStage | None:
    """The stage after `stage` on the spine, or None at the terminal."""
    i = stage_index(stage)
    if i + 1 >= len(SPINE):
        return None
    return SPINE[i + 1]


def is_earlier(candidate: AnalysisStage, than: AnalysisStage) -> bool:
    return stage_index(candidate) < stage_index(than)
