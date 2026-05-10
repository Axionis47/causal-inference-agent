"""ReAct Orchestrator - Fully autonomous LLM-driven orchestration.

This orchestrator uses the ReAct paradigm to autonomously:
1. Observe the current analysis state
2. Reason about what needs to be done
3. Dispatch to specialist agents
4. Observe results and adapt
5. Handle errors and iterate as needed

NO HARDCODED WORKFLOW - the LLM decides everything.
"""

import asyncio
import time

from src.agents.base import (
    AgentTrace,
    AnalysisState,
    BaseAgent,
    CritiqueDecision,
    JobStatus,
    ReActAgent,
    ToolResult,
    ToolResultStatus,
)
from src.agents.orchestrator.common import (
    AGENT_STATUS_MAP,
    validate_required_fields,
)
from src.config.settings import get_settings
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class ReActOrchestrator(ReActAgent):
    """Fully autonomous orchestrator using ReAct paradigm.

    This orchestrator:
    1. Does NOT follow a fixed workflow
    2. Reasons about what to do based on current state
    3. Adapts to errors and unexpected situations
    4. Knows when to iterate vs. finalize
    """

    AGENT_NAME = "react_orchestrator"
    MAX_STEPS = 20  # More steps for complex orchestration

    SYSTEM_PROMPT = """You are an autonomous causal inference orchestrator.
Your job is to coordinate a team of specialist agents to perform rigorous causal analysis.

AVAILABLE SPECIALISTS:
- domain_knowledge: Reads dataset metadata (description, column names, tags) to extract causal hints (treatment/outcome candidates, immutable variables, temporal ordering). Run this FIRST when metadata is available; downstream specialists pull from its output via ask_domain_knowledge.
- data_profiler: Analyzes dataset, identifies treatment/outcome candidates
- eda_agent: Exploratory data analysis (distributions, correlations, outliers)
- causal_discovery: Learns causal graph structure (PC, GES, NOTEARS algorithms)
- dag_expert: Refines the DAG using domain expertise, enforces forbidden edges, computes adjustment sets. Run AFTER causal_discovery.
- confounder_discovery: Identifies confounders through statistical tests and causal reasoning. Run AFTER data_profiler.
- data_repair: Diagnoses and repairs data quality issues (missing data, outliers, encoding). Run AFTER data_profiler.
- effect_estimator: Estimates treatment effects (PSM, IPW, AIPW, DiD, etc.)
- ps_diagnostics: Validates propensity score models (overlap, balance, calibration). Run AFTER effect_estimator.
- sensitivity_analyst: Tests robustness (Rosenbaum bounds, E-values)
- critique_agent: Reviews analysis quality, identifies issues
- notebook_generator: Creates reproducible Jupyter notebook

YOUR TOOLS:
- check_state: See current analysis state
- dispatch_agent: Send task to a specialist
- request_critique: Get analysis reviewed
- handle_feedback: Process critique feedback
- generate_notebook: Create final notebook
- finish: Complete the analysis

PRINCIPLES:
1. If dataset metadata is available, dispatch domain_knowledge FIRST so downstream agents can pull semantic context (immutable vars, temporal ordering, treatment/outcome hints).
2. After domain_knowledge (or first, if no metadata), dispatch data_profiler to understand the data.
3. EDA is crucial - don't skip it.
4. Choose methods based on data characteristics, not defaults.
5. Request critique before finalizing.
6. If critique says ITERATE, address the feedback.
7. Maximum 3 iterations, then finalize with best effort.
8. Explain your reasoning at every step.

BE AUTONOMOUS:
- Don't just follow a script - reason about what's needed
- Adapt if something fails
- Skip unnecessary steps if data doesn't support them
- Be skeptical and thorough
"""

    def __init__(self) -> None:
        """Initialize the ReAct orchestrator."""
        super().__init__()
        self._specialists: dict[str, BaseAgent] = {}
        self._iteration_count = 0
        self._status_callback = None
        self._register_orchestration_tools()

    def set_status_callback(self, callback) -> None:
        """Set a callback for persisting status updates (e.g. to Firestore)."""
        self._status_callback = callback

    def register_specialist(self, name: str, agent: BaseAgent) -> None:
        """Register a specialist agent."""
        self._specialists[name] = agent
        self.logger.info("specialist_registered", agent_name=name)

    def _validate_required_fields(self, agent_name: str, state: AnalysisState) -> list[str]:
        """Check if state has the fields this agent requires. Returns list of missing fields."""
        return validate_required_fields(
            self._specialists, agent_name, state, self.logger
        )

    def _register_orchestration_tools(self) -> None:
        """Register orchestration-specific tools."""

        # Check state tool. The "all" aspect was deliberately removed:
        # it was an escape hatch back to push-based context (the LLM
        # could pull everything in one call, defeating the point of
        # focused queries). Forcing aspect-specific calls keeps each
        # observation small and lets the loop budget go further.
        self.register_tool(
            name="check_state",
            description=(
                "Check a specific aspect of the current analysis state. "
                "Call this multiple times with different aspects rather "
                "than asking for everything at once."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "enum": ["profile", "results", "critique", "progress"],
                        "description": "Which aspect of state to check",
                    },
                },
                "required": ["aspect"],
            },
            handler=self._check_state,
        )

        # Dispatch agent tool
        self.register_tool(
            name="dispatch_agent",
            description="Dispatch a task to a specialist agent.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": [
                            "domain_knowledge",
                            "data_profiler",
                            "data_repair",
                            "eda_agent",
                            "causal_discovery",
                            "dag_expert",
                            "confounder_discovery",
                            "effect_estimator",
                            "ps_diagnostics",
                            "sensitivity_analyst",
                        ],
                        "description": "Which specialist to dispatch to",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "What the agent should focus on",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this agent is needed now",
                    },
                },
                "required": ["agent_name", "reasoning"],
            },
            handler=self._dispatch_agent,
        )

        # Request critique tool
        self.register_tool(
            name="request_critique",
            description="Request the critique agent to review the current analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of analysis for the critic",
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Areas to focus critique on",
                    },
                },
                "required": ["summary"],
            },
            handler=self._request_critique,
        )

        # Handle feedback tool
        self.register_tool(
            name="handle_feedback",
            description="Process critique feedback and decide how to address issues.",
            parameters={
                "type": "object",
                "properties": {
                    "action_plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps to address critique feedback",
                    },
                },
                "required": ["action_plan"],
            },
            handler=self._handle_feedback,
        )

        # Generate notebook tool
        self.register_tool(
            name="generate_notebook",
            description="Generate the final Jupyter notebook documenting the analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key recommendations to include",
                    },
                },
                "required": ["recommendations"],
            },
            handler=self._generate_notebook,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Get initial observation from state."""
        return f"""Starting causal analysis orchestration.

Job ID: {state.job_id}
Dataset: {state.dataset_info.name or state.dataset_info.url}
Status: {state.status.value}
Iteration: {state.iteration_count}/{state.max_iterations}

Registered specialists: {list(self._specialists.keys())}

You need to coordinate these specialists to perform a rigorous causal analysis.
Start by understanding the current state, then decide what to do.
"""

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if orchestration is complete."""
        return state.status in [JobStatus.COMPLETED, JobStatus.FAILED]

    async def _check_state(
        self,
        state: AnalysisState,
        aspect: str = "progress",
        **kwargs,
    ) -> ToolResult:
        """Check a specific aspect of analysis state.

        Each branch returns only the fields relevant to that aspect, so
        the LLM gets focused observations rather than a state dump.
        `aspect` defaults to "progress" so a stripped tool call still
        returns something useful instead of erroring; **kwargs absorbs
        spurious params (e.g. agent_name, reasoning) the model attaches.
        """
        output: dict = {}

        if aspect == "progress":
            output["status"] = state.status.value
            output["iteration"] = f"{state.iteration_count}/{state.max_iterations}"
            output["has_profile"] = state.data_profile is not None
            output["has_eda"] = state.eda_result is not None
            output["has_dag"] = state.proposed_dag is not None
            output["n_effects"] = len(state.treatment_effects)
            output["n_sensitivity"] = len(state.sensitivity_results)
            output["is_approved"] = state.is_approved()

        elif aspect == "profile" and state.data_profile:
            output["profile"] = {
                "n_samples": state.data_profile.n_samples,
                "n_features": state.data_profile.n_features,
                "treatment_candidates": state.data_profile.treatment_candidates[:5],
                "outcome_candidates": state.data_profile.outcome_candidates[:5],
                "has_time": state.data_profile.has_time_dimension,
                "has_instruments": bool(state.data_profile.potential_instruments),
            }

        elif aspect == "results":
            if state.treatment_effects:
                # Cap at 8 (out of 12 possible methods) so a long run
                # does not blow up the observation.
                output["effects"] = [
                    {
                        "method": e.method,
                        "estimate": f"{e.estimate:.4f}",
                        "ci": f"[{e.ci_lower:.4f}, {e.ci_upper:.4f}]",
                    }
                    for e in state.treatment_effects[:8]
                ]
                if len(state.treatment_effects) > 8:
                    output["effects_truncated"] = (
                        f"{len(state.treatment_effects) - 8} more"
                    )
            if state.sensitivity_results:
                output["sensitivity"] = [
                    {"method": s.method, "robustness": f"{s.robustness_value:.2f}"}
                    for s in state.sensitivity_results[:8]
                ]

        elif aspect == "critique":
            latest = state.get_latest_critique()
            if latest:
                output["critique"] = {
                    "decision": latest.decision.value,
                    "issues": latest.issues[:5],
                    "improvements": latest.improvements[:5],
                    "issues_total": len(latest.issues),
                    "improvements_total": len(latest.improvements),
                }
            else:
                output["critique"] = "No critique yet"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=output,
        )

    async def _dispatch_agent(
        self,
        state: AnalysisState,
        agent_name: str,
        reasoning: str = "",
        task_description: str = "",
        **kwargs,
    ) -> ToolResult:
        """Dispatch to a specialist agent.

        reasoning is encouraged in the schema but not required — the run
        should not abort if the LLM omits it.
        """
        specialist = self._specialists.get(agent_name)
        if not specialist:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Specialist '{agent_name}' not registered. Available: {list(self._specialists.keys())}",
            )

        # Warn-only: check if required state fields are populated
        self._validate_required_fields(agent_name, state)

        # Update status based on agent (shared mapping with the standard orchestrator)
        state.status = AGENT_STATUS_MAP.get(agent_name, state.status)

        self.logger.info(
            "dispatching_specialist",
            agent=agent_name,
            reasoning=reasoning[:100],
        )

        settings = get_settings()

        # Execute the specialist with timeout
        try:
            start_time = time.time()
            try:
                # INT1: Deep-copy state so agent mutations don't bypass
                # WRITES_STATE_FIELDS whitelist merge
                state_copy = state.model_copy(deep=True)
                updated_state = await asyncio.wait_for(
                    specialist.execute_with_tracing(state_copy),
                    timeout=settings.agent_timeout_seconds,
                )
            except TimeoutError:
                logger.warning(
                    "agent_timeout",
                    agent=agent_name,
                    timeout=settings.agent_timeout_seconds,
                )
                state.add_trace(AgentTrace(
                    agent_name=agent_name,
                    action="timeout",
                    reasoning=f"Agent {agent_name} timed out after {settings.agent_timeout_seconds}s",
                ))
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output=None,
                    error=f"Agent {agent_name} timed out after {settings.agent_timeout_seconds}s",
                )

            duration_ms = int((time.time() - start_time) * 1000)

            # Merge only the fields declared by the agent's WRITES_STATE_FIELDS
            valid_fields = set(state.model_fields)
            fields_to_merge = getattr(specialist, "WRITES_STATE_FIELDS", None) or []
            if not fields_to_merge:
                self.logger.warning(
                    "agent_no_writes_declared",
                    agent=agent_name,
                    note="Agent does not declare WRITES_STATE_FIELDS; no fields will be merged",
                )
            for field_name in fields_to_merge:
                if field_name not in valid_fields:
                    self.logger.warning(
                        "invalid_merge_field", agent=agent_name, field=field_name
                    )
                    continue
                value = getattr(updated_state, field_name, None)
                if value is not None:
                    setattr(state, field_name, value)
            # Always merge traces, decisions, status, timestamps. Decisions
            # are append-only audit data, not contested writes, so they never
            # need to be in WRITES_STATE_FIELDS — same treatment as traces.
            # Without this merge, every push_decision() call inside a
            # specialist is silently dropped by the deep-copy isolation
            # boundary, leaving state.decisions empty across react runs.
            state.agent_traces = updated_state.agent_traces
            state.decisions = updated_state.decisions
            state.updated_at = updated_state.updated_at
            state.status = updated_state.status
            if updated_state.error_message:
                state.error_message = updated_state.error_message
                state.error_agent = updated_state.error_agent

            # Audit trail: record this dispatch as an orchestrator decision so
            # it shows up in the notebook's Methodology Decisions section and
            # /jobs/{id}/results.decision_log alongside specialist decisions.
            state.push_decision(
                agent="orchestrator",
                decision_type="agent_dispatched",
                choice=agent_name,
                reason=(reasoning or task_description or "")[:200] or f"Dispatched {agent_name}",
            )

            # Build result summary based on what changed
            result_summary = {"agent": agent_name, "duration_ms": duration_ms}

            if agent_name == "data_profiler" and state.data_profile:
                result_summary["profile"] = {
                    "samples": state.data_profile.n_samples,
                    "features": state.data_profile.n_features,
                    "treatment_candidates": len(state.data_profile.treatment_candidates),
                }
            elif agent_name == "eda_agent" and state.eda_result:
                result_summary["eda"] = {
                    "quality_score": state.eda_result.data_quality_score,
                    "issues": len(state.eda_result.data_quality_issues),
                }
            elif agent_name == "effect_estimator":
                result_summary["effects"] = len(state.treatment_effects)
            elif agent_name == "sensitivity_analyst":
                result_summary["sensitivity_results"] = len(state.sensitivity_results)

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=result_summary,
            )

        except Exception as e:
            self.logger.error("specialist_failed", agent=agent_name, error=str(e))
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Agent {agent_name} failed: {str(e)}",
            )

    async def _request_critique(
        self,
        state: AnalysisState,
        summary: str = "",
        focus_areas: list[str] | None = None,
        **kwargs,
    ) -> ToolResult:
        """Request critique of the analysis.

        summary is encouraged but optional so the loop survives a tool
        call that omits it; the critic doesn't actually consume it
        directly anyway. focus_areas is reserved for future hinting.
        **kwargs swallows extras the LLM may attach (e.g. `reasoning`).
        """
        critique_agent = self._specialists.get("critique")
        if not critique_agent:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Critique agent not registered",
            )

        state.status = JobStatus.CRITIQUE_REVIEW

        try:
            state = await critique_agent.execute_with_tracing(state)

            latest = state.get_latest_critique()
            if latest:
                # Short-circuit on REJECT: don't let the LLM reason its way
                # past a critique that explicitly failed the analysis.
                if state.is_rejected():
                    reason = latest.reasoning or "Critique agent rejected the analysis"
                    self.logger.warning(
                        "critique_rejected_analysis",
                        iteration=state.iteration_count,
                        issues=latest.issues,
                    )
                    state.mark_failed(f"rejected_by_critique: {reason}", "critique")
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        output={
                            "decision": "REJECT",
                            "issues": latest.issues,
                            "improvements": latest.improvements,
                            "reasoning": reason,
                            "note": "Analysis rejected; job marked FAILED. Stop the loop.",
                        },
                        error=f"rejected_by_critique: {reason}",
                    )

                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "decision": latest.decision.value,
                        "issues": latest.issues,
                        "improvements": latest.improvements,
                        "confidence": latest.scores.get("overall", 0),
                    },
                )
            else:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"decision": "APPROVE", "note": "No specific feedback"},
                )

        except Exception as e:
            self.logger.error("critique_failed", error=str(e))
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=str(e),
            )

    async def _handle_feedback(
        self,
        state: AnalysisState,
        action_plan: list[str] | None = None,
        **kwargs,
    ) -> ToolResult:
        """Handle critique feedback.

        action_plan defaults to an empty list so an LLM that calls this
        tool without one still progresses (no-iteration branch returns).
        """
        action_plan = action_plan or []
        latest = state.get_latest_critique()

        # Defensive: REJECT should already have short-circuited in
        # _request_critique. If we got here anyway, refuse to proceed.
        if state.is_rejected():
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output={
                    "decision": "REJECT",
                    "note": "Cannot iterate past a REJECT. Stop the loop.",
                },
                error="rejected_by_critique",
            )

        if latest and latest.decision == CritiqueDecision.ITERATE:
            state.iteration_count += 1
            state.status = JobStatus.ITERATING

            if state.iteration_count >= state.max_iterations:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "note": "Max iterations reached - proceeding to finalize",
                        "action_plan": action_plan,
                    },
                )

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "iteration": state.iteration_count,
                    "action_plan": action_plan,
                    "feedback_to_address": latest.issues,
                },
            )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"note": "No iteration needed", "action_plan": action_plan},
        )

    async def _generate_notebook(
        self,
        state: AnalysisState,
        recommendations: list[str] | None = None,
        **kwargs,
    ) -> ToolResult:
        """Generate the final notebook.

        `recommendations` is declared required in the tool schema for prompt
        clarity, but the model regularly omits it; making it optional here
        prevents the entire run from failing on the last step. **kwargs
        absorbs any extra params the LLM may attach.
        """
        notebook_agent = self._specialists.get("notebook_generator")
        if not notebook_agent:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Notebook generator not registered",
            )

        state.status = JobStatus.GENERATING_NOTEBOOK
        state.recommendations = recommendations or []

        try:
            state = await notebook_agent.execute_with_tracing(state)

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "notebook_generated": state.notebook_path is not None,
                    "path": state.notebook_path,
                },
            )

        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=str(e),
            )

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the ReAct orchestration loop."""
        self.logger.info(
            "react_orchestration_start",
            job_id=state.job_id,
        )

        # Run the ReAct loop
        state = await super().execute(state)

        # INT2: Mark completed only if pipeline produced meaningful results
        if state.status == JobStatus.FAILED:
            pass  # already failed
        elif not state.treatment_effects:
            state.mark_failed(
                "Pipeline completed but produced no treatment effect estimates.",
                "react_orchestrator",
            )
        else:
            state.mark_completed()

        self.logger.info(
            "react_orchestration_complete",
            job_id=state.job_id,
            status=state.status.value,
        )

        return state
