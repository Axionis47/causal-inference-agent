"""Contract tests for the domain_knowledge brief.

These tests pin the contract layer only — the frozen CAPABILITY, the
preflight refusal function, and build_brief flag derivation. The ReAct
loop's own behaviour is covered by `test_agent.py`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.domain_knowledge.brief import (
    CAPABILITY,
    build_brief,
    preflight,
)
from src.analysis.agents.domain_knowledge.output import (
    DomainKnowledge,
    Hypothesis,
    Uncertainty,
)
from src.domain.briefs import Flag


# --- fixtures ---------------------------------------------------------------


def _dataset_info_with_metadata() -> DatasetInfo:
    return DatasetInfo(
        url="https://kaggle.com/x",
        kaggle_description="a randomised study of vaccine efficacy",
        kaggle_column_descriptions={"vaccinated": "binary treatment indicator"},
    )


def _dataset_info_empty() -> DatasetInfo:
    return DatasetInfo(url="https://kaggle.com/x")


def _make_state(
    *,
    dataset_info: DatasetInfo | None = None,
    raw_metadata: dict | None = None,
    domain_knowledge: DomainKnowledge | None = None,
) -> AnalysisState:
    return AnalysisState(
        job_id="test-job",
        dataset_info=dataset_info or _dataset_info_with_metadata(),
        raw_metadata=raw_metadata,
        domain_knowledge=domain_knowledge,
    )


def _hyp(claim: str, confidence: str = "medium") -> Hypothesis:
    return Hypothesis(claim=claim, confidence=confidence, evidence="x")


# --- CAPABILITY -------------------------------------------------------------


class TestCapability:
    def test_name_is_domain_knowledge(self):
        assert CAPABILITY.name == "domain_knowledge"

    def test_needs_only_dataset_info(self):
        assert CAPABILITY.needs == ("dataset_info",)

    def test_delivers_lists_domain_knowledge(self):
        assert "domain_knowledge" in CAPABILITY.delivers

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "something_else"

    def test_one_line_starts_with_name(self):
        assert CAPABILITY.one_line().startswith("domain_knowledge: ")

    def test_success_criteria_cover_each_flag_this_agent_raises(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.WEAK_CONFOUNDER_EVIDENCE in flags_with_criteria


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_refuses_when_all_metadata_sources_are_empty(self):
        state = _make_state(dataset_info=_dataset_info_empty(), raw_metadata=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.status == "refused"
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "metadata" in brief.headline.lower()

    def test_passes_when_kaggle_description_present(self):
        info = DatasetInfo(url="u", kaggle_description="a study")
        state = _make_state(dataset_info=info)
        assert preflight(state) is None

    def test_passes_when_column_descriptions_present(self):
        info = DatasetInfo(url="u", kaggle_column_descriptions={"x": "covariate"})
        state = _make_state(dataset_info=info)
        assert preflight(state) is None

    def test_passes_when_kaggle_tags_present(self):
        info = DatasetInfo(url="u", kaggle_tags=["healthcare"])
        state = _make_state(dataset_info=info)
        assert preflight(state) is None

    def test_passes_when_raw_metadata_present(self):
        state = _make_state(
            dataset_info=_dataset_info_empty(),
            raw_metadata={"title": "Vaccine trial"},
        )
        assert preflight(state) is None


# --- build_brief: status ----------------------------------------------------


class TestBuildBriefStatus:
    def test_failed_when_state_domain_knowledge_is_none(self):
        state = _make_state(domain_knowledge=None)
        brief = build_brief(state)
        assert brief.status == "failed"
        assert "no domain knowledge" in brief.headline.lower()
        assert brief.artifact_keys == []

    def test_done_when_treatment_and_outcome_hypotheses_present(self):
        dk = DomainKnowledge(
            hypotheses=[
                _hyp("vaccinated is the treatment"),
                _hyp("infection is the outcome"),
                _hyp("age is a confounder"),
            ],
            investigation_complete=True,
        )
        state = _make_state(domain_knowledge=dk)
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["domain_knowledge"]

    def test_failed_when_treatment_hypothesis_absent(self):
        dk = DomainKnowledge(hypotheses=[_hyp("infection is the outcome")])
        state = _make_state(domain_knowledge=dk)
        brief = build_brief(state)
        assert brief.status == "failed"

    def test_failed_when_outcome_hypothesis_absent(self):
        dk = DomainKnowledge(hypotheses=[_hyp("vaccinated is the treatment")])
        state = _make_state(domain_knowledge=dk)
        brief = build_brief(state)
        assert brief.status == "failed"

    def test_low_confidence_hypothesis_does_not_count(self):
        # Brief's notion of "complete" matches helpers.has_treatment_and_outcome_
        # hypotheses, which gates on confidence in {medium, high}.
        dk = DomainKnowledge(
            hypotheses=[
                _hyp("treatment is X", confidence="low"),
                _hyp("outcome is Y", confidence="low"),
            ]
        )
        state = _make_state(domain_knowledge=dk)
        brief = build_brief(state)
        assert brief.status == "failed"


# --- build_brief: flag derivation ------------------------------------------


class TestBuildBriefFlags:
    def test_raises_weak_confounder_evidence_when_no_confounder_hypothesis(self):
        dk = DomainKnowledge(
            hypotheses=[
                _hyp("vaccinated is the treatment"),
                _hyp("infection is the outcome"),
            ]
        )
        state = _make_state(domain_knowledge=dk)
        brief = build_brief(state)
        assert Flag.WEAK_CONFOUNDER_EVIDENCE in brief.flags

    def test_does_not_raise_weak_confounder_when_confounder_hypothesis_present(self):
        dk = DomainKnowledge(
            hypotheses=[
                _hyp("vaccinated is the treatment"),
                _hyp("infection is the outcome"),
                _hyp("age is a confounder"),
            ]
        )
        state = _make_state(domain_knowledge=dk)
        brief = build_brief(state)
        assert Flag.WEAK_CONFOUNDER_EVIDENCE not in brief.flags

    def test_low_confidence_confounder_does_not_block_flag(self):
        # If the only confounder claim is low-confidence, weak-evidence flag
        # still fires — the agent did not deliver a usable confounder claim.
        dk = DomainKnowledge(
            hypotheses=[
                _hyp("vaccinated is the treatment"),
                _hyp("infection is the outcome"),
                _hyp("age is a confounder", confidence="low"),
            ]
        )
        state = _make_state(domain_knowledge=dk)
        brief = build_brief(state)
        assert Flag.WEAK_CONFOUNDER_EVIDENCE in brief.flags


# --- build_brief: headline --------------------------------------------------


class TestBuildBriefHeadline:
    def test_headline_lists_each_count(self):
        dk = DomainKnowledge(
            hypotheses=[
                _hyp("vaccinated is the treatment"),
                _hyp("infection is the outcome"),
                _hyp("age is a confounder"),
            ],
            uncertainties=[Uncertainty(issue="sparse columns", impact="low")],
        )
        state = _make_state(domain_knowledge=dk)
        headline = build_brief(state).headline
        assert "1 treatment" in headline
        assert "1 outcome" in headline
        assert "1 confounder" in headline
        assert "1 uncertainties" in headline
