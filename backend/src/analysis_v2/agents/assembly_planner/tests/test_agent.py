"""AssemblyPlannerAgent (S0A): the agentic loop, fallbacks, and trace."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.assembly_planner.agent import AssemblyPlannerAgent
from src.analysis_v2.agents.assembly_planner.proposer import (
    summarize_plan,
    synthesize_trace,
)
from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.spec import AssemblyJoin, AssemblyPlan
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, write_manifest

JOB_ID = "job-assembly-agent"


@pytest.fixture
def storage_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod
    from src.storage.local_storage import reset_local_storage_client

    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()
    yield tmp_path
    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()


class ToolLLM:
    """Scripts chat_with_tools turns for the assembly loop."""

    def __init__(self, turns):
        self.turns = list(turns)

    def user_message(self, text):
        return {"role": "user", "content": text}

    def tool_results_message(self, results):
        return {"role": "tool_result",
                "content": [{"name": c["name"], "output": o} for c, o in results]}

    async def chat_with_tools(self, messages, system_instruction, specs):
        turn = self.turns.pop(0)
        return {
            "text": turn.get("text"),
            "tool_calls": turn.get("tool_calls", []),
            "assistant_message": {"role": "assistant", "content": turn.get("text") or ""},
            "usage": turn.get("usage", {"input_tokens": 9, "output_tokens": 4}),
        }


def _stub_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "src.analysis_v2.agents.assembly_planner.agent.get_llm_client", lambda: llm
    )


def _stage(frames: dict[str, pd.DataFrame], winner: str) -> None:
    files = []
    for name, df in frames.items():
        df.to_parquet(job_normalized_dir(JOB_ID) / f"{name}.parquet", index=False)
        files.append(
            ManifestFile(
                name=name, relative_path=f"raw/{name}", size_bytes=1, format="csv",
                sha256="0" * 64, used=(name == winner),
                normalized_path=f"normalized/{name}.parquet", tabular=True,
                columns=list(df.columns), n_rows=len(df),
            )
        )
    write_manifest(
        JOB_ID,
        DatasetManifest(
            job_id=JOB_ID, kaggle_url="https://www.kaggle.com/x", raw_dir="raw",
            winner=winner, files=files,
        ),
    )


def _ctx() -> AgentCtx:
    run = AnalysisRunState(job_id=JOB_ID, causal_question="does training raise sales?")
    return AgentCtx(job_id=JOB_ID, run=run, frame=pd.DataFrame({"x": [1]}))


async def test_agentic_run_records_a_join_and_commits_the_plan_and_trace(
    storage_dir, monkeypatch
):
    _stage(
        {
            "facts.csv": pd.DataFrame({"store_id": [1, 1, 2], "sales": [10, 12, 9]}),
            "store.csv": pd.DataFrame({"store_id": [1, 2], "store_type": ["a", "b"]}),
        },
        winner="facts.csv",
    )
    llm = ToolLLM([
        {"tool_calls": [{"name": "inspect_join_keys",
                         "args": {"left_file": "facts.csv", "right_file": "store.csv"}}]},
        {"tool_calls": [
            {"name": "propose_join",
             "args": {"right_file": "store.csv", "on": ["store_id"], "how": "left"}},
            {"name": "finish_assembly",
             "args": {"base_file": "facts.csv", "rationale": "store lookup"}},
        ]},
        {"text": "Assembly complete."},
    ])
    _stub_llm(monkeypatch, llm)
    ctx = _ctx()
    agent = AssemblyPlannerAgent()

    result = await agent.execute(ctx)

    assert result.gate.status == GateStatus.ADVANCE
    assert result.output.base_file == "facts.csv"
    assert result.output.joins[0].right_file == "store.csv"
    assert not result.output.is_trivial
    assert result.tokens.output_tokens > 0  # the loop's usage is aggregated
    agent.commit(ctx.run, result.output)
    assert ctx.run.assembly_plan.joins[0].on == ["store_id"]
    kinds = [c.kind for c in ctx.run.assembly_tool_trace.calls]
    assert "join" in kinds and kinds[-1] == "finish"


async def test_a_bad_join_key_is_rejected_and_the_plan_stays_single_file(
    storage_dir, monkeypatch
):
    _stage(
        {
            "facts.csv": pd.DataFrame({"store_id": [1, 2], "sales": [10, 9]}),
            "store.csv": pd.DataFrame({"other": [1, 2], "store_type": ["a", "b"]}),
        },
        winner="facts.csv",
    )
    llm = ToolLLM([
        {"tool_calls": [{"name": "propose_join",
                         "args": {"right_file": "store.csv", "on": ["store_id"]}}]},
        {"tool_calls": [{"name": "finish_assembly", "args": {"base_file": "facts.csv"}}]},
        {"text": "No safe join; single file."},
    ])
    _stub_llm(monkeypatch, llm)
    ctx = _ctx()

    result = await AssemblyPlannerAgent().execute(ctx)

    assert result.output.is_trivial  # the join was rejected, nothing recorded
    assert result.output.joins == []


async def test_no_tool_support_falls_back_to_the_single_winner(storage_dir, monkeypatch):
    _stage(
        {"a.csv": pd.DataFrame({"x": [1]}), "b.csv": pd.DataFrame({"y": [1]})},
        winner="a.csv",
    )
    _stub_llm(monkeypatch, object())  # no chat_with_tools
    ctx = _ctx()
    agent = AssemblyPlannerAgent()

    result = await agent.execute(ctx)

    assert result.gate.status == GateStatus.ADVANCE
    assert result.output.base_file == "a.csv" and result.output.is_trivial
    agent.commit(ctx.run, result.output)
    assert ctx.run.assembly_tool_trace.calls[-1].kind == "finish"  # synthesized


async def test_a_loop_error_falls_back_to_the_single_winner(storage_dir, monkeypatch):
    _stage(
        {"a.csv": pd.DataFrame({"x": [1]}), "b.csv": pd.DataFrame({"y": [1]})},
        winner="a.csv",
    )

    class BoomLLM:
        def user_message(self, text):
            return {"role": "user", "content": text}

        def tool_results_message(self, results):
            return {"role": "tool_result", "content": []}

        async def chat_with_tools(self, messages, system_instruction, specs):
            raise RuntimeError("provider down")

    _stub_llm(monkeypatch, BoomLLM())
    ctx = _ctx()

    result = await AssemblyPlannerAgent().execute(ctx)

    assert result.output.is_trivial  # degraded, not crashed


async def test_single_file_bundle_skips_the_loop_entirely(storage_dir):
    # One file: no tools, no LLM constructed; exactly today's behavior.
    _stage({"only.csv": pd.DataFrame({"x": [1]})}, winner="only.csv")
    ctx = _ctx()

    result = await AssemblyPlannerAgent().execute(ctx)

    assert result.output.base_file == "only.csv" and result.output.is_trivial


async def test_no_manifest_advances_without_committing_a_plan(storage_dir):
    ctx = _ctx()  # no manifest written

    result = await AssemblyPlannerAgent().execute(ctx)

    assert result.gate.status == GateStatus.ADVANCE
    assert result.output is None  # nothing to commit; run.assembly_plan stays None


def test_synthesize_trace_records_each_concat_and_join_then_finish():
    plan = AssemblyPlan(
        base_file="facts.csv", concat_files=["facts2.csv"],
        joins=[AssemblyJoin(right_file="store.csv", on=["id"])],
    )

    kinds = [c.kind for c in synthesize_trace(plan).calls]

    assert kinds == ["concat", "join", "finish"]


def test_summarize_plan_names_the_join_for_a_non_trivial_plan():
    plan = AssemblyPlan(
        base_file="facts.csv",
        joins=[AssemblyJoin(right_file="store.csv", on=["store_id"])],
    )

    summary = summarize_plan(plan)

    assert "store.csv" in summary and "store_id" in summary
    assert summarize_plan(AssemblyPlan.single_file("a.csv")).startswith("Single dataset")
