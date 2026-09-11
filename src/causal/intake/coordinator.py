"""Reusable immutable intake service; execution state belongs to each invocation."""

from causal.intake.contracts import IntakeSubmissionV1
from causal.intake.entry import IntakeDeps, run_intake
from causal.intake.outcome import IntakeError, IntakeResult, open_handoff
from causal.shared.contracts import HandoffManifestV1

__all__ = ["IntakeCoordinator", "IntakeError", "IntakeResult"]


class IntakeCoordinator(IntakeDeps):
    """Retains the existing dependency constructor, run, and open_handoff API."""

    def run(self, submission: IntakeSubmissionV1) -> IntakeResult:
        return run_intake(self, submission)

    def open_handoff(self, analysis_id: str, intake_outcome_artifact_id: str,
                     receiving_stage_run_id: str) -> HandoffManifestV1:
        return open_handoff(self.catalog, self.products, analysis_id,
                            intake_outcome_artifact_id, receiving_stage_run_id, self.clock)
