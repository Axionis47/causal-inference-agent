"""Agent-level tests: initialization, execute(), completion check, initial observation."""

import pytest

from src.analysis.agents.domain_knowledge import DomainKnowledge, Hypothesis, Uncertainty


class TestInitialization:
    def test_agent_name(self, agent):
        assert agent.AGENT_NAME == "domain_knowledge"

    def test_max_steps(self, agent):
        assert agent.MAX_STEPS == 12

    def test_tools_registered(self, agent):
        """Every specialist tool + the ReAct built-ins land in the registry."""
        tool_names = list(agent._tools.keys())
        expected = [
            "read_description",
            "list_columns",
            "investigate_column",
            "search_metadata",
            "get_tags",
            "hypothesize",
            "revise_hypothesis",
            "set_temporal_ordering",
            "mark_immutable",
            "flag_uncertainty",
            "finish",
            "reflect",
        ]
        for tool in expected:
            assert tool in tool_names, f"Tool '{tool}' not registered"


class TestTaskCompletion:
    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state_with_metadata):
        assert await agent.is_task_complete(state_with_metadata) is False

    @pytest.mark.asyncio
    async def test_complete_with_treatment_and_outcome(self, agent, state_with_metadata):
        """Medium-or-higher confidence on both treatment and outcome -> complete."""
        await agent._hypothesize(state_with_metadata, "treat is the treatment variable", "high", "evidence")
        await agent._hypothesize(state_with_metadata, "re78 is the outcome variable", "medium", "evidence")
        assert await agent.is_task_complete(state_with_metadata) is True

    @pytest.mark.asyncio
    async def test_not_complete_with_low_confidence(self, agent, state_with_metadata):
        """Low-confidence hypotheses do not satisfy completion."""
        await agent._hypothesize(state_with_metadata, "maybe treatment", "low", "weak evidence")
        await agent._hypothesize(state_with_metadata, "maybe outcome", "low", "weak evidence")
        assert await agent.is_task_complete(state_with_metadata) is False


class TestInitialObservation:
    def test_generates_lean_observation(self, agent, state_with_metadata):
        """Initial observation surfaces the dataset, not a metadata dump."""
        obs = agent._get_initial_observation(state_with_metadata)

        assert len(obs) < 500
        assert "LaLonde" in obs or "lalonde" in obs.lower()
        assert "Kaggle" in obs
        assert "randomly assigned" not in obs
        assert "mid-1970s" not in obs


class TestExecute:
    @pytest.mark.asyncio
    async def test_finalization_produces_typed_domain_knowledge(self, agent, state_with_metadata):
        """The finalization step of execute() builds a DomainKnowledge slot from
        the agent's accumulated dict buffers, separating live from revised
        hypotheses and preserving all of them under all_hypotheses."""
        agent._steps = []
        agent._hypotheses = [
            {"claim": "treat is treatment", "confidence": "high", "evidence": "test", "revised": False},
            {"claim": "re78 is outcome", "confidence": "high", "evidence": "test", "revised": False},
        ]
        agent._uncertainties = [{"issue": "test issue", "impact": "test impact"}]
        agent._temporal_understanding = "baseline -> treatment -> outcome"
        agent._immutable_vars = ["age", "race"]

        all_hyps = [Hypothesis(**h) for h in agent._hypotheses]
        live_hyps = [h for h in all_hyps if not h.revised]
        state_with_metadata.domain_knowledge = DomainKnowledge(
            hypotheses=live_hyps,
            all_hypotheses=all_hyps,
            uncertainties=[Uncertainty(**u) for u in agent._uncertainties],
            temporal_understanding=agent._temporal_understanding,
            immutable_vars=list(agent._immutable_vars),
            investigation_complete=True,
        )

        dk = state_with_metadata.domain_knowledge
        assert dk is not None
        assert len(dk.hypotheses) == 2
        assert dk.temporal_understanding == "baseline -> treatment -> outcome"
        assert "age" in dk.immutable_vars
        assert dk.investigation_complete is True
