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
- data_profiler: Analyzes dataset, identifies treatment/outcome candidates
- eda_agent: Exploratory data analysis (distributions, correlations, outliers)
- causal_discovery: Learns causal graph structure (PC, GES, NOTEARS algorithms)
- effect_estimator: Estimates treatment effects (PSM, IPW, AIPW, DiD, etc.)
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
1. ALWAYS start with data_profiler - you need to understand the data first
2. EDA is crucial - don't skip it
3. Choose methods based on data characteristics, not defaults
4. Request critique before finalizing
5. If critique says ITERATE, address the feedback
6. Maximum 3 iterations, then finalize with best effort
7. Explain your reasoning at every step

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
        self._register_orchestration_tools()

    def register_specialist(self, name: str, agent: BaseAgent) -> None:
        """Register a specialist agent."""
        self._specialists[name] = agent
        self.logger.info("specialist_registered", agent_name=name)

    def _register_orchestration_tools(self) -> None:
        """Register orchestration-specific tools."""

        # Check state tool
        self.register_tool(
            name="check_state",
            description="Check the current analysis state to understand progress and decide next steps.",
            parameters={
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "enum": ["all", "profile", "results", "critique", "progress"],
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
                            "eda_agent",
                            "causal_discovery",
                            "effect_estimator",
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
        aspect: str,
    ) -> ToolResult:
        """Check current analysis state."""
        output = {}

        if aspect in ["all", "progress"]:
            output["status"] = state.status.value
            output["iteration"] = f"{state.iteration_count}/{state.max_iterations}"
            output["has_profile"] = state.data_profile is not None
            output["has_eda"] = state.eda_result is not None
            output["has_dag"] = state.proposed_dag is not None
            output["n_effects"] = len(state.treatment_effects)
            output["n_sensitivity"] = len(state.sensitivity_results)
            output["is_approved"] = state.is_approved()

        if aspect in ["all", "profile"] and state.data_profile:
            output["profile"] = {
                "n_samples": state.data_profile.n_samples,
                "n_features": state.data_profile.n_features,
                "treatment_candidates": state.data_profile.treatment_candidates[:5],
                "outcome_candidates": state.data_profile.outcome_candidates[:5],
                "has_time": state.data_profile.has_time_dimension,
                "has_instruments": bool(state.data_profile.potential_instruments),
            }

        if aspect in ["all", "results"]:
            if state.treatment_effects:
                output["effects"] = [
                    {
                        "method": e.method,
                        "estimate": f"{e.estimate:.4f}",
                        "ci": f"[{e.ci_lower:.4f}, {e.ci_upper:.4f}]",
                    }
                    for e in state.treatment_effects
                ]
            if state.sensitivity_results:
                output["sensitivity"] = [
                    {"method": s.method, "robustness": f"{s.robustness_value:.2f}"}
                    for s in state.sensitivity_results
                ]

        if aspect in ["all", "critique"]:
            latest = state.get_latest_critique()
            if latest:
                output["critique"] = {
                    "decision": latest.decision.value,
                    "issues": latest.issues,
                    "improvements": latest.improvements,
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
        reasoning: str,
        task_description: str = "",
    ) -> ToolResult:
        """Dispatch to a specialist agent."""
        specialist = self._specialists.get(agent_name)
        if not specialist:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Specialist '{agent_name}' not registered. Available: {list(self._specialists.keys())}",
            )

        # Update status based on agent
        status_map = {
            "data_profiler": JobStatus.PROFILING,
            "eda_agent": JobStatus.EXPLORATORY_ANALYSIS,
            "causal_discovery": JobStatus.DISCOVERING_CAUSAL,
            "effect_estimator": JobStatus.ESTIMATING_EFFECTS,
            "sensitivity_analyst": JobStatus.SENSITIVITY_ANALYSIS,
        }
        state.status = status_map.get(agent_name, state.status)

        self.logger.info(
            "dispatching_specialist",
            agent=agent_name,
            reasoning=reasoning[:100],
        )

        # Define which fields each agent is allowed to update
        AGENT_OUTPUT_FIELDS = {
            "data_profiler": ["data_profile", "dataframe_path", "dataset_info", "treatment_variable", "outcome_variable"],
            "eda_agent": ["eda_result"],
            "causal_discovery": ["proposed_dag"],
            "effect_estimator": ["treatment_effects", "analyzed_pairs"],
            "sensitivity_analyst": ["sensitivity_results"],
            "notebook_generator": ["notebook_path"],
            "critique": ["critique_history", "debate_history"],
            "domain_knowledge": ["domain_knowledge"],
        }

        settings = get_settings()

        # Execute the specialist with timeout
        try:
            start_time = time.time()
            try:
                updated_state = await asyncio.wait_for(
                    specialist.execute_with_tracing(state),
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

            # Merge only the relevant fields (not full __dict__)
            fields_to_merge = AGENT_OUTPUT_FIELDS.get(agent_name, [])
            for field_name in fields_to_merge:
                value = getattr(updated_state, field_name, None)
                if value is not None:
                    setattr(state, field_name, value)
            # Always merge traces, status, timestamps
            state.agent_traces = updated_state.agent_traces
            state.updated_at = updated_state.updated_at
            state.status = updated_state.status
            if updated_state.error_message:
                state.error_message = updated_state.error_message
                state.error_agent = updated_state.error_agent

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
        summary: str,
        focus_areas: list[str] | None = None,
    ) -> ToolResult:
        """Request critique of the analysis."""
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
        action_plan: list[str],
    ) -> ToolResult:
        """Handle critique feedback."""
        latest = state.get_latest_critique()

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
        recommendations: list[str],
    ) -> ToolResult:
        """Generate the final notebook."""
        notebook_agent = self._specialists.get("notebook_generator")
        if not notebook_agent:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Notebook generator not registered",
            )

        state.status = JobStatus.GENERATING_NOTEBOOK
        state.recommendations = recommendations

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

        # Mark as completed if not already failed
        if state.status != JobStatus.FAILED:
            state.mark_completed()

        self.logger.info(
            "react_orchestration_complete",
            job_id=state.job_id,
            status=state.status.value,
        )

        return state
