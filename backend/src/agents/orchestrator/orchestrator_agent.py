"""Orchestrator Agent - The brain of the causal inference system.

This agent coordinates the entire analysis pipeline using LLM reasoning.
CRITICAL: All decisions are made through Gemini reasoning - NO hardcoded if-else logic.
"""

import time
from typing import Any

from src.agents.base import (
    AnalysisState,
    BaseAgent,
    CritiqueDecision,
    JobStatus,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """Central orchestrator that coordinates the causal inference pipeline.

    This agent uses Gemini to reason about:
    1. What analysis steps are needed
    2. Which specialist agents to dispatch
    3. How to handle critique feedback
    4. When to iterate vs. finalize

    NO HARDCODED IF-ELSE LOGIC for routing decisions.
    """

    AGENT_NAME = "orchestrator"

    SYSTEM_PROMPT = """You are an expert causal inference orchestrator. Your role is to coordinate
a team of specialist agents to perform rigorous causal analysis on datasets.

You have access to the following specialist agents:
- domain_knowledge: Extracts domain knowledge from dataset metadata (treatment/outcome hints, temporal ordering, immutable variables). Run this FIRST if metadata is available.
- data_profiler: Analyzes dataset characteristics, identifies treatment/outcome candidates. Uses domain knowledge if available.
- eda_agent: Performs comprehensive exploratory data analysis (correlations, outliers, distributions, covariate balance, multicollinearity). Uses domain knowledge for context.
- causal_discovery: Learns causal graph structure from data using algorithms like PC, GES, NOTEARS. Uses domain knowledge for constraints.
- effect_estimator: Estimates treatment effects (ATE, ATT, CATE) using appropriate methods
- sensitivity_analyst: Performs robustness checks (Rosenbaum bounds, E-values, placebo tests)
- notebook_generator: Creates reproducible Jupyter notebooks documenting the analysis

You also interact with:
- critique_agent: Reviews analysis quality and provides feedback for iteration

Your workflow:
1. If metadata is available, FIRST dispatch to domain_knowledge to extract causal hints
2. Then dispatch to data_profiler to understand the dataset (it will query domain knowledge)
3. ALWAYS dispatch to eda_agent for comprehensive EDA (correlations, outliers, balance checks)
4. Based on the profile and EDA, reason about what causal methods are appropriate
5. Dispatch to causal_discovery to learn the causal graph structure
6. Dispatch to effect_estimator and sensitivity_analyst
7. Request critique review before finalizing
8. If critique says ITERATE, address the feedback and re-dispatch relevant agents
9. Once APPROVED, dispatch to notebook_generator

CONTEXT ENGINEERING:
- Domain knowledge agent builds understanding from metadata (variable descriptions, tags, domain)
- All downstream agents (profiler, EDA, discovery) can QUERY domain knowledge through tools
- This is PULL-based context - agents request what they need, not push-based dumps
- If domain knowledge has high confidence hypotheses, profiler/discovery should leverage them

CRITICAL RULES:
- If metadata exists, dispatch domain_knowledge FIRST to build context
- NEVER skip the data profiling step
- NEVER skip the EDA step - it's crucial for identifying data quality issues
- ALWAYS reason explicitly about WHY you're choosing each method
- Consider assumptions required for each method
- Address ALL critique feedback before proceeding
- Maximum 3 iterations before finalizing with best effort

When making decisions, output your reasoning step-by-step, then specify which agent to dispatch."""

    TOOLS = [
        {
            "name": "dispatch_to_agent",
            "description": "Dispatch a task to a specialist agent",
            "parameters": {
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
                            "notebook_generator",
                        ],
                        "description": "The specialist agent to dispatch to",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Detailed description of what the agent should do",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning for choosing this agent and task",
                    },
                },
                "required": ["agent_name", "task_description", "reasoning"],
            },
        },
        {
            "name": "request_critique",
            "description": "Request the critique agent to review the current analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific areas to focus the critique on",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary of the analysis so far for the critique agent",
                    },
                },
                "required": ["focus_areas", "summary"],
            },
        },
        {
            "name": "finalize_analysis",
            "description": "Mark the analysis as complete and ready for notebook generation",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Final summary of the causal analysis",
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key recommendations based on the analysis",
                    },
                },
                "required": ["summary", "recommendations"],
            },
        },
    ]

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        super().__init__()
        self._specialist_agents: dict[str, BaseAgent] = {}

    def register_specialist(self, name: str, agent: BaseAgent) -> None:
        """Register a specialist agent.

        Args:
            name: Name to register the agent under
            agent: The specialist agent instance
        """
        self._specialist_agents[name] = agent
        self.logger.info("specialist_registered", agent_name=name)

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the orchestration logic.

        This method drives the entire analysis pipeline by:
        1. Assessing current state
        2. Using LLM to decide next action
        3. Dispatching to specialists
        4. Handling critique feedback
        5. Iterating until complete

        Args:
            state: Current analysis state

        Returns:
            Updated analysis state
        """
        self.logger.info(
            "orchestration_start",
            job_id=state.job_id,
            current_status=state.status.value,
            iteration=state.iteration_count,
        )

        # Build context prompt based on current state
        context_prompt = self._build_context_prompt(state)

        # Orchestration loop
        max_decisions = 10  # Safety limit
        decisions_made = 0

        while decisions_made < max_decisions:
            decisions_made += 1
            start_time = time.time()

            # Get next action from LLM
            reasoning_result = await self.reason(
                prompt=context_prompt,
                context=self._get_state_context(state),
            )

            # Process the LLM's decision
            if reasoning_result.get("pending_calls"):
                # Execute the function calls
                for call in reasoning_result["pending_calls"]:
                    state = await self._execute_tool_call(state, call)

                    # Check if we should continue or break
                    if state.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        return state

                    # If we dispatched to an agent, update context for next iteration
                    context_prompt = self._build_continuation_prompt(state)

            elif reasoning_result.get("response"):
                # LLM provided a text response, try to extract intent
                response_text = reasoning_result["response"]
                self.logger.debug("llm_text_response", response=response_text[:500])

                # If the response suggests completion, finalize
                if self._should_finalize(state, response_text):
                    state = await self._finalize(state, response_text)
                    return state

                # Otherwise, build a new prompt asking for specific action
                context_prompt = self._build_action_request_prompt(state)

            else:
                self.logger.warning("no_action_from_llm", result=reasoning_result)
                break

            # Record trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action=f"orchestration_decision_{decisions_made}",
                reasoning=str(reasoning_result.get("response", reasoning_result)),
                outputs={"tool_calls": reasoning_result.get("tool_calls", [])},
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

        self.logger.info(
            "orchestration_complete",
            job_id=state.job_id,
            final_status=state.status.value,
            decisions_made=decisions_made,
        )

        return state

    def _build_context_prompt(self, state: AnalysisState) -> str:
        """Build the initial context prompt for the LLM."""
        prompt = f"""Analyze the current state and decide the next action for this causal inference job.

Job ID: {state.job_id}
Dataset: {state.dataset_info.name or state.dataset_info.url}
Current Status: {state.status.value}
Iteration: {state.iteration_count} / {state.max_iterations}

"""
        # Add metadata status
        if state.raw_metadata:
            quality = state.raw_metadata.get("metadata_quality", "unknown")
            prompt += f"\nDataset Metadata: Available (quality: {quality})"
            if not state.domain_knowledge:
                prompt += "\n→ Domain knowledge NOT extracted yet. Consider dispatching domain_knowledge agent."
            prompt += "\n"
        else:
            prompt += "\nDataset Metadata: Not available\n"

        # Add domain knowledge if available
        if state.domain_knowledge:
            dk = state.domain_knowledge
            prompt += "\nDomain Knowledge (extracted from metadata):\n"

            # Show hypotheses
            hypotheses = dk.get("hypotheses", [])
            if hypotheses:
                prompt += "- Key hypotheses:\n"
                for h in hypotheses[:3]:
                    prompt += f"  * {h.get('claim', '')} (confidence: {h.get('confidence', 'unknown')})\n"

            # Show temporal ordering
            if dk.get("temporal_understanding"):
                prompt += f"- Temporal ordering: {dk['temporal_understanding']}\n"

            # Show immutable vars
            if dk.get("immutable_vars"):
                prompt += f"- Immutable variables: {dk['immutable_vars']}\n"

            # Show uncertainties
            if dk.get("uncertainties"):
                prompt += f"- Uncertainties: {len(dk['uncertainties'])} flagged\n"

            prompt += "\n"

        # Add data profile if available
        if state.data_profile:
            prompt += f"""
Dataset Profile:
- Samples: {state.data_profile.n_samples}
- Features: {state.data_profile.n_features}
- Treatment candidates: {state.data_profile.treatment_candidates}
- Outcome candidates: {state.data_profile.outcome_candidates}
- Has time dimension: {state.data_profile.has_time_dimension}
- Potential instruments: {state.data_profile.potential_instruments}
- Discontinuity candidates: {state.data_profile.discontinuity_candidates}

"""
        else:
            prompt += "\nDataset has NOT been profiled yet. This should be the first step.\n"

        # Add EDA results if available
        if state.eda_result:
            prompt += f"""
EDA Results:
- Data Quality Score: {state.eda_result.data_quality_score:.1f}/100
- Quality Issues: {state.eda_result.data_quality_issues}
- High Correlations: {len(state.eda_result.high_correlations)} pairs found
- Multicollinearity Warnings: {state.eda_result.multicollinearity_warnings}
- Covariate Balance: {state.eda_result.balance_summary}
- Columns with Outliers: {len(state.eda_result.outliers)}

"""
        elif state.data_profile:
            prompt += "\nEDA has NOT been performed yet. Dispatch to eda_agent after profiling.\n"

        # Add treatment effect results if available
        if state.treatment_effects:
            prompt += "\nTreatment Effect Results:\n"
            for effect in state.treatment_effects:
                prompt += f"- {effect.method} ({effect.estimand}): {effect.estimate:.4f} "
                prompt += f"[{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]\n"

        # Add critique feedback if available
        if state.critique_history:
            latest = state.get_latest_critique()
            if latest:
                prompt += f"""
Latest Critique:
- Decision: {latest.decision.value}
- Issues: {', '.join(latest.issues)}
- Improvements needed: {', '.join(latest.improvements)}

"""
                if latest.decision == CritiqueDecision.ITERATE:
                    prompt += "You MUST address the critique feedback before proceeding.\n"
                elif latest.decision == CritiqueDecision.APPROVE:
                    prompt += "Analysis has been APPROVED. Proceed to notebook generation.\n"

        prompt += """
Based on the above state, reason about what should be done next and call the appropriate tool.
Think step by step:
1. What information do we have?
2. What information do we need?
3. Which agent should be dispatched?
4. What specific task should they perform?
"""
        return prompt

    def _build_continuation_prompt(self, state: AnalysisState) -> str:
        """Build a continuation prompt after an agent has completed."""
        prompt = self._build_context_prompt(state)
        prompt += "\n\nAn agent just completed. Review the updated state and decide the next action."
        return prompt

    def _build_action_request_prompt(self, state: AnalysisState) -> str:
        """Build a prompt requesting a specific action."""
        prompt = self._build_context_prompt(state)
        prompt += """

Please call one of the available tools to proceed:
- dispatch_to_agent: Send a task to a specialist
- request_critique: Request analysis review
- finalize_analysis: Complete the analysis

Do not just provide text - you MUST call a tool to proceed."""
        return prompt

    def _get_state_context(self, state: AnalysisState) -> dict[str, Any]:
        """Get state context as a dict for the LLM."""
        return {
            "job_id": state.job_id,
            "status": state.status.value,
            "has_metadata": state.raw_metadata is not None,
            "has_domain_knowledge": state.domain_knowledge is not None,
            "has_profile": state.data_profile is not None,
            "has_eda": state.eda_result is not None,
            "has_dag": state.proposed_dag is not None,
            "num_effects": len(state.treatment_effects),
            "num_sensitivity": len(state.sensitivity_results),
            "iteration": state.iteration_count,
            "is_approved": state.is_approved(),
        }

    async def _execute_tool_call(
        self,
        state: AnalysisState,
        call: dict[str, Any],
    ) -> AnalysisState:
        """Execute a tool call from the LLM.

        Args:
            state: Current analysis state
            call: Tool call with name and args

        Returns:
            Updated analysis state
        """
        tool_name = call["name"]
        args = call.get("args", {})

        self.logger.info(
            "executing_tool_call",
            tool=tool_name,
            args_keys=list(args.keys()),
        )

        if tool_name == "dispatch_to_agent":
            return await self._dispatch_to_agent(state, args)
        elif tool_name == "request_critique":
            return await self._request_critique(state, args)
        elif tool_name == "finalize_analysis":
            return await self._finalize_from_tool(state, args)
        else:
            self.logger.warning("unknown_tool_call", tool=tool_name)
            return state

    async def _dispatch_to_agent(
        self,
        state: AnalysisState,
        args: dict[str, Any],
    ) -> AnalysisState:
        """Dispatch to a specialist agent.

        Args:
            state: Current analysis state
            args: Tool call arguments

        Returns:
            Updated analysis state
        """
        agent_name = args["agent_name"]
        task_description = args.get("task_description", "")
        reasoning = args.get("reasoning", "")

        self.logger.info(
            "dispatching_to_specialist",
            agent=agent_name,
            task=task_description[:100],
        )

        # Update status based on agent
        status_map = {
            "domain_knowledge": JobStatus.PROFILING,  # Part of early profiling phase
            "data_profiler": JobStatus.PROFILING,
            "eda_agent": JobStatus.EXPLORATORY_ANALYSIS,
            "causal_discovery": JobStatus.DISCOVERING_CAUSAL,
            "effect_estimator": JobStatus.ESTIMATING_EFFECTS,
            "sensitivity_analyst": JobStatus.SENSITIVITY_ANALYSIS,
            "notebook_generator": JobStatus.GENERATING_NOTEBOOK,
        }
        state.status = status_map.get(agent_name, state.status)

        # Get the specialist agent
        specialist = self._specialist_agents.get(agent_name)
        if specialist is None:
            self.logger.error("specialist_not_found", agent=agent_name)
            state.mark_failed(f"Specialist agent '{agent_name}' not found", self.AGENT_NAME)
            return state

        # Record dispatch trace
        trace = self.create_trace(
            action=f"dispatch_to_{agent_name}",
            reasoning=reasoning,
            inputs={"task": task_description},
        )
        state.add_trace(trace)

        # Execute the specialist
        try:
            state = await specialist.execute_with_tracing(state)
        except Exception as e:
            self.logger.error(
                "specialist_execution_failed",
                agent=agent_name,
                error=str(e),
            )
            state.mark_failed(str(e), agent_name)

        return state

    async def _request_critique(
        self,
        state: AnalysisState,
        args: dict[str, Any],
    ) -> AnalysisState:
        """Request critique of the current analysis.

        Args:
            state: Current analysis state
            args: Tool call arguments

        Returns:
            Updated analysis state with critique feedback
        """
        state.status = JobStatus.CRITIQUE_REVIEW

        # Get the critique agent
        critique_agent = self._specialist_agents.get("critique")
        if critique_agent is None:
            self.logger.error("critique_agent_not_found")
            # Continue without critique if not available
            return state

        # Record request trace
        trace = self.create_trace(
            action="request_critique",
            reasoning=args.get("summary", "Requesting analysis review"),
            inputs={"focus_areas": args.get("focus_areas", [])},
        )
        state.add_trace(trace)

        # Execute critique
        try:
            state = await critique_agent.execute_with_tracing(state)
        except Exception as e:
            self.logger.error("critique_failed", error=str(e))
            # Continue without critique if it fails

        # Check if we need to iterate
        if state.should_iterate():
            state.iteration_count += 1
            state.status = JobStatus.ITERATING
            self.logger.info(
                "iteration_required",
                iteration=state.iteration_count,
                feedback=state.get_latest_critique().issues if state.get_latest_critique() else [],
            )

        return state

    async def _finalize_from_tool(
        self,
        state: AnalysisState,
        args: dict[str, Any],
    ) -> AnalysisState:
        """Finalize the analysis from a tool call.

        Args:
            state: Current analysis state
            args: Tool call arguments

        Returns:
            Updated analysis state
        """
        state.recommendations = args.get("recommendations", [])

        # Dispatch to notebook generator
        notebook_agent = self._specialist_agents.get("notebook_generator")
        if notebook_agent:
            state.status = JobStatus.GENERATING_NOTEBOOK
            try:
                state = await notebook_agent.execute_with_tracing(state)
            except Exception as e:
                self.logger.error("notebook_generation_failed", error=str(e))

        state.mark_completed()
        return state

    async def _finalize(self, state: AnalysisState, response: str) -> AnalysisState:
        """Finalize the analysis based on LLM response.

        Args:
            state: Current analysis state
            response: LLM response text

        Returns:
            Updated analysis state
        """
        # Extract any recommendations from the response
        if "recommendation" in response.lower():
            state.recommendations.append(response)

        # Generate notebook if not already done
        if state.notebook_path is None:
            notebook_agent = self._specialist_agents.get("notebook_generator")
            if notebook_agent:
                state.status = JobStatus.GENERATING_NOTEBOOK
                state = await notebook_agent.execute_with_tracing(state)

        state.mark_completed()
        return state

    def _should_finalize(self, state: AnalysisState, response: str) -> bool:
        """Determine if we should finalize based on state and response.

        Args:
            state: Current analysis state
            response: LLM response text

        Returns:
            True if we should finalize
        """
        # Finalize if approved by critique
        if state.is_approved():
            return True

        # Finalize if max iterations reached
        if state.iteration_count >= state.max_iterations:
            self.logger.warning("max_iterations_reached", finalizing=True)
            return True

        # Check response for finalization signals
        finalize_signals = [
            "analysis complete",
            "ready to finalize",
            "generate notebook",
            "all steps completed",
        ]
        return any(signal in response.lower() for signal in finalize_signals)
