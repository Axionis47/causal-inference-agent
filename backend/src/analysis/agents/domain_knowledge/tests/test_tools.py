"""Tool-handler tests: one class per registered tool."""

import pytest

from src.analysis.agents.base import ToolResultStatus


class TestReadDescription:
    @pytest.mark.asyncio
    async def test_reads_full_description(self, agent, state_with_metadata):
        result = await agent._read_description(state_with_metadata)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_description"] is True
        assert "randomly assigned" in result.output["description"]
        assert "job training" in result.output["description"]

    @pytest.mark.asyncio
    async def test_handles_no_description(self, agent, state_with_minimal_metadata):
        result = await agent._read_description(state_with_minimal_metadata)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_description"] is False


class TestInvestigateColumn:
    @pytest.mark.asyncio
    async def test_identifies_treatment_clues(self, agent, state_with_metadata):
        result = await agent._investigate_column(state_with_metadata, column="treat")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["column"] == "treat"
        assert result.output["has_description"] is True
        assert any("TREATMENT" in clue for clue in result.output["name_clues"])

    @pytest.mark.asyncio
    async def test_identifies_demographic_clues(self, agent, state_with_metadata):
        result = await agent._investigate_column(state_with_metadata, column="age")
        assert result.status == ToolResultStatus.SUCCESS
        assert any("DEMOGRAPHIC" in clue or "IMMUTABLE" in clue for clue in result.output["name_clues"])

    @pytest.mark.asyncio
    async def test_identifies_outcome_clues(self, agent, state_with_metadata):
        result = await agent._investigate_column(state_with_metadata, column="re78")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_description"] is True


class TestSearchMetadata:
    @pytest.mark.asyncio
    async def test_finds_matching_text(self, agent, state_with_metadata):
        result = await agent._search_metadata(state_with_metadata, query="random")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is True
        assert len(result.output["matches"]) > 0

    @pytest.mark.asyncio
    async def test_handles_no_matches(self, agent, state_with_metadata):
        result = await agent._search_metadata(state_with_metadata, query="xyznonexistent")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is False


class TestGetTags:
    @pytest.mark.asyncio
    async def test_returns_tags(self, agent, state_with_metadata):
        result = await agent._get_tags(state_with_metadata)
        assert result.status == ToolResultStatus.SUCCESS
        assert "economics" in result.output["tags"]
        assert "causal-inference" in result.output["tags"]

    @pytest.mark.asyncio
    async def test_identifies_domain(self, agent, state_with_metadata):
        result = await agent._get_tags(state_with_metadata)
        assert result.status == ToolResultStatus.SUCCESS
        assert any("Economics" in hint for hint in result.output["domain_hints"])


class TestHypothesize:
    @pytest.mark.asyncio
    async def test_records_hypothesis(self, agent, state_with_metadata):
        result = await agent._hypothesize(
            state_with_metadata,
            claim="treat is the treatment variable",
            confidence="high",
            evidence="Column description says treatment indicator",
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["recorded"] is True
        assert len(agent._hypotheses) == 1
        assert agent._hypotheses[0]["claim"] == "treat is the treatment variable"
        assert agent._hypotheses[0]["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_records_multiple_hypotheses(self, agent, state_with_metadata):
        await agent._hypothesize(state_with_metadata, "A is treatment", "high", "evidence A")
        await agent._hypothesize(state_with_metadata, "B is outcome", "medium", "evidence B")
        assert len(agent._hypotheses) == 2


class TestReviseHypothesis:
    @pytest.mark.asyncio
    async def test_revises_hypothesis(self, agent, state_with_metadata):
        """Original is flagged revised; new claim is appended; both retained."""
        await agent._hypothesize(state_with_metadata, "X is treatment", "medium", "initial")

        result = await agent._revise_hypothesis(
            state_with_metadata,
            original_claim="X is treatment",
            new_claim="Y is treatment",
            reason="Found better evidence for Y",
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["revision_recorded"] is True
        assert agent._hypotheses[0]["revised"] is True
        assert len(agent._hypotheses) == 2
        assert agent._hypotheses[1]["claim"] == "Y is treatment"


class TestSetTemporalOrdering:
    @pytest.mark.asyncio
    async def test_sets_ordering(self, agent, state_with_metadata):
        """Ordering narrative recorded; pre-treatment vars seed the immutable list."""
        result = await agent._set_temporal_ordering(
            state_with_metadata,
            ordering="Demographics at baseline, then treatment, then outcome",
            pre_treatment_vars=["age", "education", "race"],
            post_treatment_vars=["re78"],
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert agent._temporal_understanding is not None
        assert "age" in agent._immutable_vars


class TestMarkImmutable:
    @pytest.mark.asyncio
    async def test_marks_immutable(self, agent, state_with_metadata):
        result = await agent._mark_immutable(
            state_with_metadata,
            variable="age",
            reason="Age is a demographic characteristic that cannot be changed",
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert "age" in agent._immutable_vars

    @pytest.mark.asyncio
    async def test_no_duplicates(self, agent, state_with_metadata):
        await agent._mark_immutable(state_with_metadata, "age", "reason 1")
        await agent._mark_immutable(state_with_metadata, "age", "reason 2")
        assert agent._immutable_vars.count("age") == 1


class TestFlagUncertainty:
    @pytest.mark.asyncio
    async def test_flags_uncertainty(self, agent, state_with_metadata):
        result = await agent._flag_uncertainty(
            state_with_metadata,
            issue="Control group source unclear",
            impact="May need to adjust for confounding if observational",
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert len(agent._uncertainties) == 1
        assert agent._uncertainties[0]["issue"] == "Control group source unclear"
