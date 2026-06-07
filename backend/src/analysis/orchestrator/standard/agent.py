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
conductor that runs them in order and verifies each hand-back.
"""

import asyncio
import time
from typing import Any

from src.analysis.agents.base import (
    AnalysisState,
    BaseAgent,
    CritiqueDecision,
    JobStatus,
)
from src.analysis.orchestrator.base import (
    park_for_approval,
    should_pause_for_approval,
)
from src.analysis.orchestrator.common import (
    AGENT_STATUS_MAP,
    classify_brief,
    validate_required_fields,
)
from src.analysis.orchestrator.standard.spine import SPINE, Stage
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

_TERMINAL = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class StandardOrchestrator(BaseAgent):
    """Conductor that runs the specialist pool along a deterministic spine."""

    AGENT_NAME = "orchestrator"

    # Baseline mapping of agent → state fields it writes (for safe parallel merging).
    # Agents that declare WRITES_STATE_FIELDS override their entry at registration time.
    _DEFAULT_AGENT_WRITES: dict[str, list[str]] = {
        "domain_knowledge": ["domain_knowledge"],
        "data_profiler": [
            "data_profile",
            "dataframe_path",
            "treatment_encoding",
            "dataset_info",
            "raw_metadata",
        ],
        "data_repair": ["data_repairs"],
        "eda_agent": ["eda_result"],
        "causal_discovery": ["discovered_dag"],
        "dag_expert": ["refined_dag"],
        "confounder_discovery": ["confounder_discovery"],
        "effect_estimator": ["treatment_effects", "analyzed_pairs", "treatment_binarization_threshold"],
        "effect_estimator_react": ["treatment_effects"],
        "ps_diagnostics": ["ps_diagnostics"],
        "sensitivity_analyst": ["sensitivity_results"],
        "notebook_generator": ["notebook_path"],
        "critique": ["critique_history"],
    }

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        super().__init__()
        self._specialist_agents: dict[str, BaseAgent] = {}
        self._status_callback = None
        # Start from baseline; agent-declared metadata overrides at registration
        self.AGENT_WRITES: dict[str, list[str]] = dict(self._DEFAULT_AGENT_WRITES)

    def set_status_callback(self, callback) -> None:
        """Set a callback for persisting status updates (e.g. to Firestore)."""
        self._status_callback = callback

    def _validate_required_fields(self, agent_name: str, state: AnalysisState) -> list[str]:
        """Check if state has the fields this agent requires. Returns list of missing fields."""
        return validate_required_fields(
            self._specialist_agents, agent_name, state, self.logger
        )

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

    async def _run_forward_spine(self, state: AnalysisState) -> AnalysisState:
        """Run SPINE in order until terminal, the data gate, or the tail is done.

        A stage is skipped when it already ran (resume after the gate) or when its
        skip predicate says the data does not call for it. The data-review gate is
        checked at entry (a job that arrives already profiled parks at once) and
        after each stage (so it fires right after the profiler on a fresh run).
        """
        if should_pause_for_approval(state):
            return await park_for_approval(state, self._status_callback)

        for stage in SPINE:
            if stage.already_done(state):
                continue
            if stage.skip_when is not None and stage.skip_when(state):
                self.logger.info("stage_skipped", stage=stage.name)
                continue

            state = await self._run_stage(state, stage)
            if state.status in _TERMINAL:
                return state

            if should_pause_for_approval(state):
                return await park_for_approval(state, self._status_callback)

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

    async def _dispatch_to_agent(
        self,
        state: AnalysisState,
        args: dict[str, Any],
    ) -> AnalysisState:
        """Dispatch to a specialist agent.

        Args:
            state: Current analysis state
            args: Dispatch arguments (agent_name, task_description, reasoning)

        Returns:
            Updated analysis state
        """
        agent_name = args.get("agent_name")
        task_description = args.get("task_description", "")
        reasoning = args.get("reasoning", "")

        if not agent_name:
            self.logger.error("dispatch_missing_agent_name", args=args)
            return state

        # Warn-only: check if required state fields are populated
        self._validate_required_fields(agent_name, state)

        self.logger.info(
            "dispatching_to_specialist",
            agent=agent_name,
            task=task_description[:100],
        )

        # Update status based on agent (informational phase mapping)
        state.status = AGENT_STATUS_MAP.get(agent_name, state.status)

        # Dynamic progress: count completed agent traces vs expected total
        state.progress_percentage = self._compute_progress(state)

        # Persist status to Firestore so API consumers can track progress
        if self._status_callback:
            try:
                await self._status_callback(state)
            except Exception:
                self.logger.debug("status_callback_failed", exc_info=True)

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
        state.push_sse_event("agent_started", {"agent_name": agent_name})
        try:
            state = await specialist.execute_with_tracing(state)
            # Readiness gate: the specialist sealed a brief into
            # state.agent_briefs. Refuse to advance if it refused/failed or
            # raised a flag that makes downstream work unsafe (a cyclic DAG);
            # a soft quality flag is surfaced but does not stop the run.
            verdict, reason = classify_brief(state.agent_briefs.get(agent_name))
            if verdict == "halt":
                state.push_sse_event("agent_completed", {"agent_name": agent_name, "success": False})
                state.mark_failed(reason, agent_name)
                return state
            state.push_sse_event("agent_completed", {"agent_name": agent_name, "success": True})
            if verdict == "soft":
                state.push_decision(
                    agent="orchestrator",
                    decision_type="readiness_flagged",
                    choice=agent_name,
                    reason=reason,
                )
            state.push_decision(
                agent="orchestrator",
                decision_type="agent_dispatched",
                choice=agent_name,
                reason=f"Dispatched on the deterministic spine: {reasoning[:200]}" if reasoning else "Dispatched on the deterministic spine",
            )
        except Exception as e:
            self.logger.error(
                "specialist_execution_failed",
                agent=agent_name,
                error=str(e),
            )
            state.push_sse_event("agent_completed", {"agent_name": agent_name, "success": False})
            state.mark_failed(str(e), agent_name)

        return state

    async def _dispatch_parallel(
        self,
        state: AnalysisState,
        args: dict[str, Any],
    ) -> AnalysisState:
        """Dispatch multiple agents to run concurrently.

        Uses copy-on-write: each agent gets a deep copy of state,
        then results are merged back using AGENT_WRITES field mapping.
        """
        agent_configs = args.get("agents", [])
        reasoning = args.get("reasoning", "")

        if len(agent_configs) < 2:
            # Fall back to sequential if only one agent
            if agent_configs:
                return await self._dispatch_to_agent(state, {
                    "agent_name": agent_configs[0]["agent_name"],
                    "task_description": agent_configs[0].get("task_description", ""),
                    "reasoning": reasoning,
                })
            return state

        agent_names = [c["agent_name"] for c in agent_configs]
        self.logger.info("parallel_dispatch_start", agents=agent_names)

        # Record dispatch trace
        trace = self.create_trace(
            action=f"parallel_dispatch_{'_'.join(agent_names)}",
            reasoning=reasoning,
            inputs={"agents": agent_names},
        )
        state.add_trace(trace)

        # Create state branches and gather specialist references
        branches: list[AnalysisState] = []
        specialists: list[tuple[str, BaseAgent]] = []
        for config in agent_configs:
            name = config["agent_name"]
            specialist = self._specialist_agents.get(name)
            if specialist is None:
                self.logger.error("parallel_specialist_not_found", agent=name)
                continue
            # Warn-only: check if required state fields are populated
            self._validate_required_fields(name, state)
            branches.append(state.model_copy(deep=True))
            specialists.append((name, specialist))

        # Emit agent_started SSE events for all parallel agents
        for name, _ in specialists:
            state.push_sse_event("agent_started", {"agent_name": name})

        # Execute all agents concurrently
        start_time = time.time()
        tasks = [
            specialist.execute_with_tracing(branch)
            for (_, specialist), branch in zip(specialists, branches, strict=True)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results back into main state
        readiness_halts: list[str] = []
        for (name, _), result in zip(specialists, results, strict=True):
            if isinstance(result, Exception):
                self.logger.error(
                    "parallel_agent_failed", agent=name, error=str(result)
                )
                state.push_sse_event("agent_completed", {"agent_name": name, "success": False})
                error_trace = self.create_trace(
                    action=f"parallel_{name}_failed",
                    reasoning=str(result),
                )
                state.add_trace(error_trace)
                continue

            # Merge the fields this agent writes (validate against model)
            valid_fields = set(state.model_fields)
            for field in self.AGENT_WRITES.get(name, []):
                if field not in valid_fields:
                    self.logger.warning(
                        "invalid_merge_field", agent=name, field=field
                    )
                    continue
                value = getattr(result, field, None)
                if value is not None:
                    setattr(state, field, value)

            # Merge traces from branch (last 5 per agent)
            if result.agent_traces:
                state.agent_traces.extend(result.agent_traces[-5:])

            # Readiness gate: AGENT_WRITES does not carry agent_briefs, so bring
            # the branch's sealed brief into the main state before classifying
            # it. halt stops the whole run after every branch is merged; soft is
            # surfaced; the completion event reflects the verdict.
            branch_brief = result.agent_briefs.get(name)
            if branch_brief is not None:
                state.agent_briefs[name] = branch_brief
            verdict, reason = classify_brief(branch_brief)
            state.push_sse_event(
                "agent_completed",
                {"agent_name": name, "success": verdict != "halt"},
            )
            if verdict == "halt":
                readiness_halts.append(reason)
            elif verdict == "soft":
                state.push_decision(
                    agent="orchestrator",
                    decision_type="readiness_flagged",
                    choice=name,
                    reason=reason,
                )

        # INT3: Update status/progress from branch results
        # These aren't in AGENT_WRITES so whitelist merge misses them
        branch_results = [r for r in results if not isinstance(r, (str, Exception))]
        if branch_results:
            # Take the highest progress from any branch
            branch_progress = [
                getattr(r, 'progress_percentage', 0)
                for r in branch_results
            ]
            if branch_progress:
                state.progress_percentage = max(state.progress_percentage, max(branch_progress))

        duration_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "parallel_dispatch_complete",
            agents=agent_names,
            duration_ms=duration_ms,
        )

        # A fatal brief from any parallel branch stops the run, after every
        # branch has merged so the panel still shows what each agent produced.
        if readiness_halts:
            state.mark_failed("; ".join(readiness_halts), self.AGENT_NAME)

        return state

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

    # Expected number of specialist agents in a full pipeline run
    _EXPECTED_AGENTS = 12

    def _compute_progress(self, state: AnalysisState) -> int:
        """Compute progress dynamically from completed agent traces.

        Progress ranges from 5 (just started) to 95 (notebook phase).
        Final 100% is set by mark_completed().

        Formula: completed_agents / expected_agents * 90 + 5
        """
        # Count unique agents that have completed (traces with "dispatch_to_" prefix)
        completed_agents: set[str] = set()
        for trace in state.agent_traces:
            action = getattr(trace, "action", "") or ""
            if action.startswith("dispatch_to_"):
                agent_name = action[len("dispatch_to_"):]
                completed_agents.add(agent_name)
            elif action.startswith("parallel_dispatch_"):
                # Parallel dispatches encode agent names joined by underscore
                # but individual agent completions are also traced
                pass
            elif action.startswith("parallel_") and action.endswith("_failed"):
                # Still counts as dispatched (attempted)
                agent_name = action[len("parallel_"):-len("_failed")]
                completed_agents.add(agent_name)

        # Also count agents from parallel dispatch completion SSE events
        # by checking state fields that are populated (more reliable)
        field_to_agent = {
            "domain_knowledge": "domain_knowledge",
            "data_profile": "data_profiler",
            "eda_result": "eda_agent",
            "discovered_dag": "causal_discovery",
            "refined_dag": "dag_expert",
            "confounder_discovery": "confounder_discovery",
            "data_repairs": "data_repair",
            "treatment_effects": "effect_estimator",
            "ps_diagnostics": "ps_diagnostics",
            "sensitivity_results": "sensitivity_analyst",
            "notebook_path": "notebook_generator",
        }
        for field, agent_name in field_to_agent.items():
            value = getattr(state, field, None)
            if value is not None:
                # For list fields, check non-empty
                if isinstance(value, list) and not value:
                    continue
                completed_agents.add(agent_name)

        n_completed = len(completed_agents)
        progress = int(n_completed / self._EXPECTED_AGENTS * 90) + 5
        # Clamp to [5, 95] — final 100% is set by mark_completed()
        return max(5, min(95, progress))
