"""Investigator: scripted full investigation and the degraded path."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.investigator import InvestigatorAgent
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.core import AnalysisRunState, AgentRun, AnalysisStage, GateStatus
from src.analysis_v2.spec import (
    CausalSpec,
    DatasetDossier,
    QuestionType,
    RoleLabel,
    VariableRef,
)

FRAME = pd.DataFrame(
    {
        "treat": [0, 1] * 30,
        "re78": [float(i) for i in range(60)],
        "age": [20 + i % 30 for i in range(60)],
        "row_id": list(range(60)),
    }
)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    run = AnalysisRunState(job_id="job-inv", causal_question="does treat raise re78?")
    run.causal_spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        treatment=VariableRef(column="treat"),
        outcome=VariableRef(column="re78"),
    )
    run.dataset_profile = build_profile_summary(FRAME)
    run.add_agent_run(AgentRun(agent="investigator", stage=AnalysisStage.S2A_DATASET_INVESTIGATED))
    yield AgentCtx(job_id="job-inv", run=run, frame=FRAME)
    settings_mod.get_settings.cache_clear()


class NoToolsLLM:
    """An older client surface: structured output only."""

    async def generate_structured(self, prompt, response_schema, system_instruction=None):
        raise AssertionError("must not be called on the degraded path")


def _dossier(roles_columns: list[tuple[str, RoleLabel]]) -> dict:
    return DatasetDossier(
        provenance="NSW job training subset",
        roles=[
            {"column": c, "role": r, "reason": "seen in investigation"}
            for c, r in roles_columns
        ],
        summary="A small experimental subset; row_id is an index.",
    ).model_dump(mode="json")


class ScriptedLLM:
    """One investigation turn, then dossier synthesis (with one invention)."""

    def __init__(self):
        self.structured_calls = 0

    def user_message(self, text):
        return {"role": "user", "content": text}

    def tool_results_message(self, results):
        return {"role": "tool_result", "content": [o for _, o in results]}

    async def chat_with_tools(self, messages, system_instruction, tools):
        if len(messages) == 1:
            return {
                "text": "checking the columns",
                "tool_calls": [
                    {"id": "1", "name": "list_columns", "args": {}},
                    {"id": "2", "name": "column_stats", "args": {"column": "row_id"}},
                ],
                "assistant_message": {"role": "assistant", "content": "checking"},
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        return {
            "text": "row_id is an index; treat/re78 as parsed",
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": "done"},
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }

    async def generate_structured_with_usage(self, prompt, response_schema, system_instruction=None):
        self.structured_calls += 1
        roles = [
            ("treat", RoleLabel.TREATMENT),
            ("re78", RoleLabel.OUTCOME),
            ("row_id", RoleLabel.IDENTIFIER),
            ("ghost_column", RoleLabel.PRE_TREATMENT),  # invented, every time
        ]
        return (
            response_schema.model_validate(_dossier(roles)),
            {"input_tokens": 200, "output_tokens": 80},
        )


async def test_scripted_investigation_builds_a_dossier_with_tool_evidence(ctx, monkeypatch):
    llm = ScriptedLLM()
    monkeypatch.setattr(
        "src.analysis_v2.agents.investigator.agent.get_llm_client", lambda: llm
    )
    agent = InvestigatorAgent()
    result = await agent.execute(ctx)

    assert result.gate.status == GateStatus.ADVANCE
    assert len(result.tool_calls) == 2
    assert {r.name for r in result.tool_calls} == {"list_columns", "column_stats"}
    assert all(r.ok for r in result.tool_calls)
    # loop usage + two structured attempts (the invention forces a retry)
    assert llm.structured_calls == 2
    assert result.tokens.total == 100 + 20 + 50 + 30 + 2 * (200 + 80)
    # the invented column was quarantined, the real roles kept
    dossier = result.output
    assert dossier.role_of("ghost_column") is None
    assert dossier.role_of("row_id").role == RoleLabel.IDENTIFIER
    assert any("ghost_column" in w for w in result.warnings)
    assert {"investigator/dossier", "investigator/transcript", "investigator/summary"} <= set(
        result.artifact_ids
    )

    agent.commit(ctx.run, dossier)
    assert ctx.run.dataset_dossier is dossier


async def test_an_llm_without_tools_degrades_to_a_spec_and_profile_dossier(ctx, monkeypatch):
    monkeypatch.setattr(
        "src.analysis_v2.agents.investigator.agent.get_llm_client", lambda: NoToolsLLM()
    )
    result = await InvestigatorAgent().execute(ctx)

    assert result.gate.status == GateStatus.ADVANCE
    dossier = result.output
    assert dossier.investigated is False
    assert dossier.role_of("treat").role == RoleLabel.TREATMENT
    assert dossier.role_of("re78").role == RoleLabel.OUTCOME
    assert any("skipped" in w for w in result.warnings)
    # the assignment-mechanism question always reaches the user
    assert any("assigned" in q for q in dossier.open_questions)


async def test_investigator_refuses_to_run_before_intake_and_profiling(ctx):
    ctx.run.causal_spec = None
    result = await InvestigatorAgent().execute(ctx)
    assert result.gate.status == GateStatus.FAIL
