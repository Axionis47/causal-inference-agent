"""Dispatch half of the standard orchestrator: run an agent, gate the hand-back.

DispatchMixin is split out of agent.py only to respect the file-size cap; it is
one class with the conductor in agent.py via inheritance. It shares the host's
state (_specialist_agents, AGENT_WRITES, _status_callback) and BaseAgent helpers
(logger, create_trace, AGENT_NAME).

This is the single place a specialist is actually run. After each run the
readiness gate (classify_brief) halts on a refused/failed brief or a fatal flag
(a cyclic DAG); a soft quality flag is surfaced. Parallel stages run on
deep-copied branches and merge the declared output fields plus each branch's
sealed brief back into the main state, since AGENT_WRITES does not carry
agent_briefs.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from src.analysis.agents.base import AnalysisState, BaseAgent
from src.analysis.orchestrator.common import (
    AGENT_STATUS_MAP,
    classify_brief,
    validate_required_fields,
)


class DispatchMixin:
    """The agent-running half of StandardOrchestrator (see module docstring)."""

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

    # Expected number of specialist agents in a full pipeline run
    _EXPECTED_AGENTS = 12

    def _validate_required_fields(self, agent_name: str, state: AnalysisState) -> list[str]:
        """Check if state has the fields this agent requires. Returns list of missing fields."""
        return validate_required_fields(
            self._specialist_agents, agent_name, state, self.logger
        )

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
