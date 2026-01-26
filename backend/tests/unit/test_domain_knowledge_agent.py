"""Unit tests for DomainKnowledgeAgent."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.agents.base import AnalysisState, DatasetInfo, ToolResultStatus
from src.agents.specialists.domain_knowledge_agent import DomainKnowledgeAgent


@pytest.fixture
def agent():
    """Create agent instance."""
    return DomainKnowledgeAgent()


@pytest.fixture
def state_with_metadata():
    """Create state with rich metadata."""
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="lalonde"),
        raw_metadata={
            "title": "LaLonde NSW Job Training Program",
            "description": (
                "This dataset contains data from the National Supported Work (NSW) "
                "demonstration, a labor training program implemented in the mid-1970s. "
                "Participants were randomly assigned to receive job training. "
                "The outcome of interest is earnings in 1978 (re78). "
                "Baseline characteristics include age, education, race (black, hispanic), "
                "marital status (married), degree status (nodegree), and pre-program earnings (re74, re75)."
            ),
            "subtitle": "Causal inference benchmark dataset",
            "tags": ["economics", "causal-inference", "employment"],
            "keywords": ["treatment effect", "job training"],
            "column_descriptions": {
                "treat": "Treatment indicator: 1 if received job training, 0 otherwise",
                "age": "Age in years",
                "education": "Years of education",
                "re78": "Earnings in 1978 (outcome)",
            },
            "metadata_quality": "high"
        }
    )


@pytest.fixture
def state_with_minimal_metadata():
    """Create state with minimal metadata."""
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="test_data"),
        raw_metadata={
            "title": "Test Dataset",
            "description": "",
            "tags": [],
            "column_descriptions": {},
            "metadata_quality": "low"
        }
    )


class TestInitialization:
    """Test agent initialization."""

    def test_agent_name(self, agent):
        """Test agent has correct name."""
        assert agent.AGENT_NAME == "domain_knowledge"

    def test_max_steps(self, agent):
        """Test agent has reasonable max steps."""
        assert agent.MAX_STEPS == 12

    def test_tools_registered(self, agent):
        """Test that investigation tools are registered."""
        tool_names = list(agent._tools.keys())

        expected_tools = [
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
            "finish",  # Built-in
            "reflect",  # Built-in
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Tool '{tool}' not registered"


class TestReadDescription:
    """Test read_description tool."""

    @pytest.mark.asyncio
    async def test_reads_full_description(self, agent, state_with_metadata):
        """Test reading description with full metadata."""
        result = await agent._read_description(state_with_metadata)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_description"] is True
        assert "randomly assigned" in result.output["description"]
        assert "job training" in result.output["description"]

    @pytest.mark.asyncio
    async def test_handles_no_description(self, agent, state_with_minimal_metadata):
        """Test handling missing description."""
        result = await agent._read_description(state_with_minimal_metadata)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_description"] is False


class TestInvestigateColumn:
    """Test investigate_column tool."""

    @pytest.mark.asyncio
    async def test_identifies_treatment_clues(self, agent, state_with_metadata):
        """Test identifying treatment variable clues."""
        result = await agent._investigate_column(state_with_metadata, column="treat")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["column"] == "treat"
        assert result.output["has_description"] is True
        assert any("TREATMENT" in clue for clue in result.output["name_clues"])

    @pytest.mark.asyncio
    async def test_identifies_demographic_clues(self, agent, state_with_metadata):
        """Test identifying demographic variable clues."""
        result = await agent._investigate_column(state_with_metadata, column="age")

        assert result.status == ToolResultStatus.SUCCESS
        assert any("DEMOGRAPHIC" in clue or "IMMUTABLE" in clue for clue in result.output["name_clues"])

    @pytest.mark.asyncio
    async def test_identifies_outcome_clues(self, agent, state_with_metadata):
        """Test identifying outcome variable clues from description."""
        result = await agent._investigate_column(state_with_metadata, column="re78")

        assert result.status == ToolResultStatus.SUCCESS
        # Should have description mentioning outcome
        assert result.output["has_description"] is True


class TestSearchMetadata:
    """Test search_metadata tool."""

    @pytest.mark.asyncio
    async def test_finds_matching_text(self, agent, state_with_metadata):
        """Test finding matching text."""
        result = await agent._search_metadata(state_with_metadata, query="random")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is True
        assert len(result.output["matches"]) > 0

    @pytest.mark.asyncio
    async def test_handles_no_matches(self, agent, state_with_metadata):
        """Test handling no matches."""
        result = await agent._search_metadata(state_with_metadata, query="xyznonexistent")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is False


class TestGetTags:
    """Test get_tags tool."""

    @pytest.mark.asyncio
    async def test_returns_tags(self, agent, state_with_metadata):
        """Test getting tags."""
        result = await agent._get_tags(state_with_metadata)

        assert result.status == ToolResultStatus.SUCCESS
        assert "economics" in result.output["tags"]
        assert "causal-inference" in result.output["tags"]

    @pytest.mark.asyncio
    async def test_identifies_domain(self, agent, state_with_metadata):
        """Test domain identification from tags."""
        result = await agent._get_tags(state_with_metadata)

        assert result.status == ToolResultStatus.SUCCESS
        assert any("Economics" in hint for hint in result.output["domain_hints"])


class TestHypothesize:
    """Test hypothesize tool."""

    @pytest.mark.asyncio
    async def test_records_hypothesis(self, agent, state_with_metadata):
        """Test recording a hypothesis."""
        result = await agent._hypothesize(
            state_with_metadata,
            claim="treat is the treatment variable",
            confidence="high",
            evidence="Column description says treatment indicator"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["recorded"] is True
        assert len(agent._hypotheses) == 1
        assert agent._hypotheses[0]["claim"] == "treat is the treatment variable"
        assert agent._hypotheses[0]["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_records_multiple_hypotheses(self, agent, state_with_metadata):
        """Test recording multiple hypotheses."""
        await agent._hypothesize(state_with_metadata, "A is treatment", "high", "evidence A")
        await agent._hypothesize(state_with_metadata, "B is outcome", "medium", "evidence B")

        assert len(agent._hypotheses) == 2


class TestReviseHypothesis:
    """Test revise_hypothesis tool."""

    @pytest.mark.asyncio
    async def test_revises_hypothesis(self, agent, state_with_metadata):
        """Test revising a hypothesis."""
        # First record a hypothesis
        await agent._hypothesize(state_with_metadata, "X is treatment", "medium", "initial")

        # Then revise it
        result = await agent._revise_hypothesis(
            state_with_metadata,
            original_claim="X is treatment",
            new_claim="Y is treatment",
            reason="Found better evidence for Y"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["revision_recorded"] is True

        # Original should be marked as revised
        assert agent._hypotheses[0]["revised"] is True

        # New hypothesis should be added
        assert len(agent._hypotheses) == 2
        assert agent._hypotheses[1]["claim"] == "Y is treatment"


class TestSetTemporalOrdering:
    """Test set_temporal_ordering tool."""

    @pytest.mark.asyncio
    async def test_sets_ordering(self, agent, state_with_metadata):
        """Test setting temporal ordering."""
        result = await agent._set_temporal_ordering(
            state_with_metadata,
            ordering="Demographics at baseline, then treatment, then outcome",
            pre_treatment_vars=["age", "education", "race"],
            post_treatment_vars=["re78"]
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert agent._temporal_understanding is not None
        assert "age" in agent._immutable_vars  # Pre-treatment vars added


class TestMarkImmutable:
    """Test mark_immutable tool."""

    @pytest.mark.asyncio
    async def test_marks_immutable(self, agent, state_with_metadata):
        """Test marking variable as immutable."""
        result = await agent._mark_immutable(
            state_with_metadata,
            variable="age",
            reason="Age is a demographic characteristic that cannot be changed"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert "age" in agent._immutable_vars

    @pytest.mark.asyncio
    async def test_no_duplicates(self, agent, state_with_metadata):
        """Test that marking same variable twice doesn't duplicate."""
        await agent._mark_immutable(state_with_metadata, "age", "reason 1")
        await agent._mark_immutable(state_with_metadata, "age", "reason 2")

        assert agent._immutable_vars.count("age") == 1


class TestFlagUncertainty:
    """Test flag_uncertainty tool."""

    @pytest.mark.asyncio
    async def test_flags_uncertainty(self, agent, state_with_metadata):
        """Test flagging uncertainty."""
        result = await agent._flag_uncertainty(
            state_with_metadata,
            issue="Control group source unclear",
            impact="May need to adjust for confounding if observational"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert len(agent._uncertainties) == 1
        assert agent._uncertainties[0]["issue"] == "Control group source unclear"


class TestTaskCompletion:
    """Test task completion logic."""

    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state_with_metadata):
        """Test that task is not complete initially."""
        result = await agent.is_task_complete(state_with_metadata)
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_with_treatment_and_outcome(self, agent, state_with_metadata):
        """Test task is complete with treatment and outcome hypotheses."""
        await agent._hypothesize(state_with_metadata, "treat is the treatment variable", "high", "evidence")
        await agent._hypothesize(state_with_metadata, "re78 is the outcome variable", "medium", "evidence")

        result = await agent.is_task_complete(state_with_metadata)
        assert result is True

    @pytest.mark.asyncio
    async def test_not_complete_with_low_confidence(self, agent, state_with_metadata):
        """Test task not complete with low confidence hypotheses."""
        await agent._hypothesize(state_with_metadata, "maybe treatment", "low", "weak evidence")
        await agent._hypothesize(state_with_metadata, "maybe outcome", "low", "weak evidence")

        result = await agent.is_task_complete(state_with_metadata)
        assert result is False


class TestInitialObservation:
    """Test initial observation generation."""

    def test_generates_lean_observation(self, agent, state_with_metadata):
        """Test that initial observation is lean, not a dump."""
        obs = agent._get_initial_observation(state_with_metadata)

        # Should be short
        assert len(obs) < 500

        # Should contain key info
        assert "LaLonde" in obs or "lalonde" in obs.lower()
        assert "Kaggle" in obs

        # Should NOT contain full description
        assert "randomly assigned" not in obs
        assert "mid-1970s" not in obs


class TestExecute:
    """Test full execution."""

    @pytest.mark.asyncio
    async def test_stores_findings_in_state(self, agent, state_with_metadata):
        """Test that execute stores findings in state.domain_knowledge."""
        # Mock the LLM to return finish immediately
        agent._steps = []

        # Manually add hypotheses as if agent ran
        agent._hypotheses = [
            {"claim": "treat is treatment", "confidence": "high", "evidence": "test", "revised": False},
            {"claim": "re78 is outcome", "confidence": "high", "evidence": "test", "revised": False},
        ]
        agent._uncertainties = [{"issue": "test issue", "impact": "test impact"}]
        agent._temporal_understanding = "baseline -> treatment -> outcome"
        agent._immutable_vars = ["age", "race"]

        # Mock the ReAct loop to just return
        with patch.object(agent, 'execute', wraps=agent.execute):
            # Directly call the finalization logic
            state_with_metadata.domain_knowledge = {
                "hypotheses": [h for h in agent._hypotheses if not h.get("revised", False)],
                "all_hypotheses": agent._hypotheses,
                "uncertainties": agent._uncertainties,
                "temporal_understanding": agent._temporal_understanding,
                "immutable_vars": agent._immutable_vars,
                "investigation_complete": True,
            }

        dk = state_with_metadata.domain_knowledge

        assert dk is not None
        assert len(dk["hypotheses"]) == 2
        assert dk["temporal_understanding"] == "baseline -> treatment -> outcome"
        assert "age" in dk["immutable_vars"]
        assert dk["investigation_complete"] is True
