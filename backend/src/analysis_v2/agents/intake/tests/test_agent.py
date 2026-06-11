"""IntakeAgent with a stubbed LLM: commit, repair, and failure paths."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.intake import IntakeAgent
from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.core import AnalysisRunState, GateStatus, NextAction
from src.analysis_v2.spec import Confidence, QuestionType


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


class StubLLM:
    """Returns queued drafts (or raises queued exceptions) per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def generate_structured(self, prompt, response_schema, system_instruction=None):
        self.prompts.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ctx(question="Does the training program raise 1978 earnings?") -> AgentCtx:
    frame = pd.DataFrame({"treat": [0, 1, 0], "re78": [0.0, 9930.0, 24909.5]})
    run = AnalysisRunState(job_id="job-intake", causal_question=question)
    return AgentCtx(job_id="job-intake", run=run, frame=frame)


def _good_draft() -> IntakeDraft:
    return IntakeDraft(
        question_type=QuestionType.BINARY_TREATMENT,
        confidence=Confidence.HIGH,
        outcome_column="re78",
        treatment_column="treat",
        reasoning_summary="A binary treatment question: treat against re78.",
    )


def _stub(monkeypatch, llm: StubLLM) -> None:
    monkeypatch.setattr(
        "src.analysis_v2.agents.intake.agent.get_llm_client", lambda: llm
    )


async def test_happy_path_commits_the_spec_and_writes_artifacts(data_dir, monkeypatch):
    llm = StubLLM([_good_draft()])
    _stub(monkeypatch, llm)
    ctx = _ctx()
    agent = IntakeAgent()

    result = await agent.execute(ctx)
    agent.commit(ctx.run, result.output)

    assert result.gate.status == GateStatus.ADVANCE
    assert ctx.run.causal_spec.question_type == QuestionType.BINARY_TREATMENT
    assert ctx.run.causal_spec.outcome.column == "re78"
    assert ctx.run.artifact_registry.get("intake/causal_spec") is not None
    assert (data_dir / "job-intake/analysis/intake/causal_spec.json").exists()
    assert result.public_summary.startswith("A binary treatment question")
    assert len(llm.prompts) == 1


async def test_invented_column_triggers_one_repair_round(data_dir, monkeypatch):
    invented = _good_draft().model_copy(update={"outcome_column": "earnings_78"})
    llm = StubLLM([invented, _good_draft()])
    _stub(monkeypatch, llm)

    result = await IntakeAgent().execute(_ctx())

    assert len(llm.prompts) == 2
    assert "earnings_78" in llm.prompts[1]  # the repair prompt names the invention
    assert result.gate.status == GateStatus.ADVANCE
    assert result.output.outcome.column == "re78"
    assert result.warnings == []  # second draft was clean


async def test_persistent_invention_is_quarantined_not_passed_through(
    data_dir, monkeypatch
):
    invented = _good_draft().model_copy(update={"outcome_column": "earnings_78"})
    llm = StubLLM([invented, invented, invented])
    _stub(monkeypatch, llm)

    result = await IntakeAgent().execute(_ctx())

    assert result.gate.status == GateStatus.ADVANCE
    assert result.output.outcome.column is None
    assert "earnings_78" in (result.output.outcome.clue or "")
    assert result.output.confidence == Confidence.LOW
    assert result.warnings  # the violation is visible on the tile


async def test_three_generation_failures_fail_the_gate(data_dir, monkeypatch):
    llm = StubLLM([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    _stub(monkeypatch, llm)

    result = await IntakeAgent().execute(_ctx())

    assert result.gate.status == GateStatus.FAIL
    assert result.failure is not None
    assert result.failure.next_action == NextAction.FAIL_JOB
    assert result.output is None


async def test_empty_question_fails_without_calling_the_llm(data_dir, monkeypatch):
    llm = StubLLM([])
    _stub(monkeypatch, llm)

    result = await IntakeAgent().execute(_ctx(question="   "))

    assert result.gate.status == GateStatus.FAIL
    assert llm.prompts == []
