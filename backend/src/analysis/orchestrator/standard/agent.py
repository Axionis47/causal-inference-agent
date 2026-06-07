"""Standard orchestrator: a deterministic spine of agent dispatches.

The orchestrator walks a fixed stage plan (see spine.py). Each specialist keeps
its own ReAct loop, but the order is fixed by the agents' declared dependencies,
not chosen by an LLM each turn. After every dispatch the readiness gate reads the
agent's sealed brief and stops the run if it refused, failed, or raised a fatal
flag. The data-review gate parks the job after profiling; on a critique ITERATE
the estimation tail re-runs, bounded by max_iterations; then the notebook is
sealed and the run completes.

There is no orchestrator-level LLM. The intelligence lives inside the specialist
agents and in the briefs they seal; the orchestrator is the deterministic
conductor that runs them in order and verifies each hand-back. The agent-running
half (dispatch, readiness gate, parallel merge, progress) lives in dispatch.py as
DispatchMixin, split out only to respect the file-size cap.
"""

from src.analysis.agents.base import (
    AnalysisState,
    BaseAgent,
    CritiqueDecision,
    JobStatus,
)
from src.analysis.orchestrator.base import (
    park_for_approval,
    park_for_dag_approval,
    should_pause_for_approval,
    should_pause_for_dag_approval,
)
from src.analysis.orchestrator.standard.dispatch import DispatchMixin
from src.analysis.orchestrator.standard.spine import SPINE, Stage
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

_TERMINAL = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class StandardOrchestrator(DispatchMixin, BaseAgent):
    """Conductor that runs the specialist pool along a deterministic spine."""

    AGENT_NAME = "orchestrator"

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        super().__init__()
        self._specialist_agents: dict[str, BaseAgent] = {}
        self._status_callback = None
        # Start from baseline (DispatchMixin); agent-declared metadata overrides
        # at registration time.
        self.AGENT_WRITES: dict[str, list[str]] = dict(self._DEFAULT_AGENT_WRITES)

    def set_status_callback(self, callback) -> None:
        """Set a callback for persisting status updates (e.g. to Firestore)."""
        self._status_callback = callback

    def register_specialist(self, name: str, agent: BaseAgent) -> None:
        """Register a specialist agent.

        Agent-declared WRITES_STATE_FIELDS overrides the baseline mapping,
        keeping a single source of truth for which state fields each agent writes.
        """
        self._specialist_agents[name] = agent
        # Prefer agent-declared metadata over static baseline
        declared = getattr(agent, "WRITES_STATE_FIELDS", None)
        if declared:
            self.AGENT_WRITES[name] = list(declared)
        self.logger.info("specialist_registered", agent_name=name)

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Run the analysis along the deterministic spine.

        Walks SPINE in order, parks at the data-review gate after profiling,
        runs the bounded critique/iterate loop, then seals the notebook. Returns
        the updated state; in-pipeline failures mark it FAILED rather than
        raising, and the data-review gate returns with AWAITING_APPROVAL, which
        is a yield (the worker persists and later resumes), not a finish.
        """
        self.logger.info(
            "orchestration_start",
            job_id=state.job_id,
            current_status=state.status.value,
            iteration=state.iteration_count,
        )

        state = await self._run_forward_spine(state)
        if state.status in (JobStatus.AWAITING_APPROVAL, JobStatus.FAILED, JobStatus.CANCELLED):
            return state

        # The forward spine could not estimate anything to review.
        if not state.treatment_effects:
            state.mark_failed(
                "Analysis ended without producing treatment effect estimates",
                self.AGENT_NAME,
            )
            return state

        state = await self._run_critique_loop(state)
        if state.status in (JobStatus.FAILED, JobStatus.CANCELLED):
            return state

        state = await self._finalize_run(state)
        self.logger.info(
            "orchestration_complete",
            job_id=state.job_id,
            final_status=state.status.value,
        )
        return state

    async def _park_if_gate_due(self, state: AnalysisState) -> AnalysisState | None:
        """Park at whichever human gate is due, or return None to continue.

        The data gate fires after profiling; the DAG gate fires after dag_expert
        refines the DAG. Their predicates are mutually exclusive in normal flow
        (the DAG gate needs refined_dag, which implies the data gate already
        passed), so checking both in order is safe.
        """
        if should_pause_for_approval(state):
            return await park_for_approval(state, self._status_callback)
        if should_pause_for_dag_approval(state):
            return await park_for_dag_approval(state, self._status_callback)
        return None

    async def _run_forward_spine(self, state: AnalysisState) -> AnalysisState:
        """Run SPINE in order until terminal, a human gate, or the tail is done.

        A stage is skipped when it already ran (resume after a gate) or when its
        skip predicate says the data does not call for it. The human gates (data
        review after profiling, DAG review after dag_expert) are checked at entry
        (a job that arrives already at a gate parks at once) and after each stage.
        """
        parked = await self._park_if_gate_due(state)
        if parked is not None:
            return parked

        for stage in SPINE:
            if stage.already_done(state):
                continue
            if stage.skip_when is not None and stage.skip_when(state):
                self.logger.info("stage_skipped", stage=stage.name)
                continue

            state = await self._run_stage(state, stage)
            if state.status in _TERMINAL:
                return state

            parked = await self._park_if_gate_due(state)
            if parked is not None:
                return parked

        return state

    async def _run_stage(self, state: AnalysisState, stage: Stage) -> AnalysisState:
        """Dispatch a stage's agents (in parallel when it names more than one)."""
        if stage.parallel:
            return await self._dispatch_parallel(state, {
                "agents": [
                    {"agent_name": name, "task_description": stage.name}
                    for name in stage.agents
                ],
                "reasoning": f"spine stage: {stage.name}",
            })
        return await self._dispatch_to_agent(state, {
            "agent_name": stage.agents[0],
            "task_description": stage.name,
            "reasoning": f"spine stage: {stage.name}",
        })

    async def _run_critique_loop(self, state: AnalysisState) -> AnalysisState:
        """Critique, with a bounded deterministic iterate.

        REJECT fails the run; APPROVE (or no critique) returns for finalize;
        ITERATE re-runs the estimation tail with the critique on record, up to
        max_iterations, then returns for a best-effort finalize.
        """
        while True:
            state = await self._run_critique(state)
            if state.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                return state

            latest = state.get_latest_critique()
            if latest is None or latest.decision != CritiqueDecision.ITERATE:
                return state  # APPROVE or no critique -> finalize

            if state.iteration_count >= state.max_iterations:
                self.logger.info(
                    "iteration_cap_reached",
                    iteration=state.iteration_count,
                    max_iterations=state.max_iterations,
                )
                return state  # best effort

            state.iteration_count += 1
            state.status = JobStatus.ITERATING
            self.logger.info("iterating", iteration=state.iteration_count)

            state = await self._run_estimation_tail(state)
            if state.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                return state

    async def _run_critique(self, state: AnalysisState) -> AnalysisState:
        """Run the critique agent once; a REJECT marks the run failed."""
        state.status = JobStatus.CRITIQUE_REVIEW

        critique_agent = self._specialist_agents.get("critique")
        if critique_agent is None:
            self.logger.error("critique_agent_not_found")
            return state

        trace = self.create_trace(
            action="request_critique",
            reasoning="deterministic critique review",
        )
        state.add_trace(trace)

        state.push_sse_event("agent_started", {"agent_name": "critique"})
        try:
            state = await critique_agent.execute_with_tracing(state)
            state.push_sse_event("agent_completed", {"agent_name": "critique", "success": True})
        except Exception as e:
            self.logger.error("critique_failed", error=str(e))
            state.push_sse_event("agent_completed", {"agent_name": "critique", "success": False})
            return state

        latest_critique = state.get_latest_critique()
        if latest_critique:
            feedback_summary = "; ".join(latest_critique.issues[:3]) if latest_critique.issues else "No issues"
            state.push_decision(
                agent="orchestrator",
                decision_type="iteration_decision",
                choice=latest_critique.decision.value,
                reason=f"Critique review: {feedback_summary}",
            )

        # Critique rejected the analysis: stop the pipeline. No notebook, no
        # further iteration. Surface the critique reasoning in the failure.
        if state.is_rejected():
            reason = (
                latest_critique.reasoning
                if latest_critique and latest_critique.reasoning
                else "Critique agent rejected the analysis"
            )
            self.logger.warning(
                "critique_rejected_analysis",
                iteration=state.iteration_count,
                issues=latest_critique.issues if latest_critique else [],
            )
            state.mark_failed(f"rejected_by_critique: {reason}", "critique")

        return state

    async def _run_estimation_tail(self, state: AnalysisState) -> AnalysisState:
        """Re-run estimation after a critique ITERATE.

        effect_estimator overwrites treatment_effects, so re-running it produces
        fresh estimates the next critique reviews. ps_diagnostics and sensitivity
        follow under their normal skip rules.
        """
        tail = ("estimate", "ps_diagnostics", "sensitivity")
        for stage in SPINE:
            if stage.name not in tail:
                continue
            if stage.skip_when is not None and stage.skip_when(state):
                self.logger.info("stage_skipped", stage=stage.name)
                continue
            state = await self._run_stage(state, stage)
            if state.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                return state
        return state

    async def _finalize_run(self, state: AnalysisState) -> AnalysisState:
        """Seal the notebook and mark the run complete."""
        notebook_agent = self._specialist_agents.get("notebook_generator")
        if notebook_agent and state.notebook_path is None:
            state.status = JobStatus.GENERATING_NOTEBOOK
            try:
                state = await notebook_agent.execute_with_tracing(state)
            except Exception as e:
                # A notebook failure does not lose the estimates already produced.
                self.logger.error("notebook_generation_failed", error=str(e))

        state.mark_completed()
        return state
