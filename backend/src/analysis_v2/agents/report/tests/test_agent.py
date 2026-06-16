"""ReportNotebookAgent S10: the agentic loop, fallbacks, and assembly."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.agents.report.agent import ReportNotebookAgent
from src.analysis_v2.agents.report.notebook import SECTIONS, build_notebook
from src.analysis_v2.agents.report.study import SECTION_TITLES
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.spec import (
    CausalDAG,
    CausalEdge,
    CausalNode,
    CausalSpec,
    ClaimCritique,
    ClaimStrength,
    Confidence,
    DesignCandidate,
    EffectEstimate,
    EstimateResult,
    MethodLane,
    MethodPlan,
    QuestionType,
    RobustnessStatus,
    SensitivityResult,
    VariableRef,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


class ToolLLM:
    """Scripts chat_with_tools turns for the report loop."""

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
            "usage": turn.get("usage", {"input_tokens": 12, "output_tokens": 6}),
        }


def _stub_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "src.analysis_v2.agents.report.agent.get_llm_client", lambda: llm
    )


def _full_run(job_id="job-report-agent"):
    frame = pd.DataFrame(
        {"treat": [0, 1, 0, 1, 0, 1], "age": [20, 30, 40, 25, 35, 45],
         "re78": [1.0, 2, 3, 4, 5, 6]}
    )
    run = AnalysisRunState(job_id=job_id, causal_question="Does training raise earnings?")
    run.causal_spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"), treatment=VariableRef(column="treat"),
    )
    run.dataset_profile = build_profile_summary(frame)
    run.method_plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="re78", treatment="treat",
    )
    run.estimate_result = EstimateResult(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        effects=[EffectEstimate(estimand="ate", estimate=1548.2, std_error=750.0,
                                ci_lower=78.2, ci_upper=3018.2, p_value=0.04,
                                interpretation="adjusted association")],
        n_rows_used=6, outcome="re78", treatment="treat", covariates_used=["age"],
    )
    run.sensitivity_result = SensitivityResult(
        robustness=RobustnessStatus.FRAGILE, confidence_reason="e-value 1.4"
    )
    run.claim_critique = ClaimCritique(
        strength=ClaimStrength.WEAK,
        allowed_language=["suggests", "consistent with"],
        forbidden_language=["proves", "definitely causes"],
        limitations=["unmeasured confounding can bias the estimate"],
        rationale="observational design, fragile sensitivity",
    )
    run.selected_design = DesignCandidate(
        lane=MethodLane.OBSERVATIONAL, design_label="regression adjustment",
        confidence=Confidence.MEDIUM, rationale="backdoor adjustment on age",
    )
    run.causal_dag = CausalDAG(
        nodes=[CausalNode(name="treat"), CausalNode(name="re78"), CausalNode(name="age")],
        edges=[
            CausalEdge(source="age", target="treat", mechanism="age shapes selection"),
            CausalEdge(source="age", target="re78", mechanism="age shapes earnings"),
            CausalEdge(source="treat", target="re78", mechanism="training effect"),
        ],
        treatment="treat", outcome="re78",
    )
    return run, frame


async def _execute(run, frame, llm, monkeypatch):
    _stub_llm(monkeypatch, llm)
    ctx = AgentCtx(job_id=run.job_id, run=run, frame=frame)
    agent = ReportNotebookAgent()
    result = await agent.execute(ctx)
    agent.commit(run, result.output)
    return result


async def test_agentic_run_records_the_trace_and_marks_the_report_agentic(data_dir, monkeypatch):
    run, frame = _full_run()
    llm = ToolLLM([
        {"tool_calls": [{"name": "describe_dag", "args": {}}]},
        {"tool_calls": [
            {"name": "write_section", "args": {
                "section_id": "estimate",
                "prose": "The adjusted gap suggests higher earnings under this design."}},
            {"name": "finish_report", "args": {
                "executive_summary": "A weak but suggestive association under adjustment."}},
        ]},
        {"text": "Report complete."},
    ])
    result = await _execute(run, frame, llm, monkeypatch)

    assert result.gate.status == GateStatus.ADVANCE
    assert "agentic" in result.public_summary
    assert "Executive summary" in result.output.sections
    section_ids = [c.section_id for c in run.report_tool_trace.calls]
    assert "estimate" in section_ids and "summary" in section_ids
    assert result.tokens.output_tokens > 0  # the loop's usage is aggregated


async def test_no_tool_support_falls_back_to_the_deterministic_build(data_dir, monkeypatch):
    run, frame = _full_run()
    result = await _execute(run, frame, object(), monkeypatch)  # no chat_with_tools

    assert result.output.sections == list(SECTIONS)  # the fixed 13-section spine
    assert "agentic" not in result.public_summary
    assert run.report_tool_trace.calls[0].kind == "section"  # synthesized from titles


async def test_a_model_that_writes_nothing_falls_back_to_deterministic(data_dir, monkeypatch):
    run, frame = _full_run()
    llm = ToolLLM([{"text": "Nothing to add."}])  # stops immediately, no tool calls
    result = await _execute(run, frame, llm, monkeypatch)

    assert result.output.sections == list(SECTIONS)
    assert "agentic" not in result.public_summary


async def test_a_loop_error_falls_back_to_the_deterministic_build(data_dir, monkeypatch):
    class BoomLLM:
        def user_message(self, text):
            return {"role": "user", "content": text}

        def tool_results_message(self, results):
            return {"role": "tool_result", "content": []}

        async def chat_with_tools(self, messages, system_instruction, specs):
            raise RuntimeError("provider down")

    run, frame = _full_run()
    result = await _execute(run, frame, BoomLLM(), monkeypatch)

    assert result.output.sections == list(SECTIONS)  # degraded, not crashed
    assert "agentic" not in result.public_summary


def test_assemble_weaves_prose_keeps_verification_and_covers_every_section():
    run, _ = _full_run()
    assembled = ReportNotebookAgent()._assemble(
        run, {"estimate": "The adjusted gap points to higher earnings."},
        "A weak but suggestive read of the data.",
    )
    nb = build_notebook(run, assembled)
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")

    assert "## Executive summary" in md
    assert "A weak but suggestive read" in md
    assert "The adjusted gap points to higher earnings." in md  # model prose woven in
    assert "LANES[MethodLane(plan['lane'])]" in code  # verification cell survives
    assert "assert abs(fresh - stored)" in code
    for title in SECTION_TITLES.values():  # coverage: nothing dropped
        assert title in md
    for cell in nb.cells:  # still no raw dict displayed
        if cell.cell_type == "code":
            last = [line for line in cell.source.splitlines() if line.strip()][-1]
            assert not last.lstrip().startswith("{")


def test_agentic_notebook_keeps_the_exact_deterministic_code_cells_in_order():
    # Markdown is inert at execution; the agentic path only weaves prose. If the
    # code cells (incl. the verification cell) are byte-identical and in the same
    # order as the deterministic notebook, the agentic notebook executes the same.
    run, _ = _full_run()
    assembled = ReportNotebookAgent()._assemble(
        run, {"estimate": "a", "diagnostics": "b", "claim": "c"}, "exec summary"
    )
    agentic_code = [c.source for c in build_notebook(run, assembled).cells if c.cell_type == "code"]
    det_code = [c.source for c in build_notebook(run).cells if c.cell_type == "code"]
    assert agentic_code == det_code


def test_render_report_weaves_prose_and_keeps_limitations():
    run, _ = _full_run()
    md = ReportNotebookAgent()._render_report(
        run, {"estimate": "The adjusted gap suggests a lift."}, "A guarded summary."
    )
    assert "A guarded summary." in md
    assert "The adjusted gap suggests a lift." in md
    assert "unmeasured confounding" in md  # deterministic limitations preserved
    for forbidden in run.claim_critique.forbidden_language:
        assert forbidden not in md.lower()


def test_render_report_states_the_headline_number_even_when_prose_omits_it():
    """Regression: the report writer is told to keep raw numbers out of its prose
    (the figure/table carry them), so the agentic final_report.md used to omit the
    estimate entirely. The headline block is now injected deterministically, so the
    human-facing report always states the estimate, interval, p-value, and strength."""
    run, _ = _full_run()
    md = ReportNotebookAgent()._render_report(
        run, {"estimate": "The program is linked to higher earnings."}, "An adjusted read."
    )
    flat = md.replace(",", "")
    assert "1548" in flat                       # point estimate
    assert "78.2" in flat and "3018" in flat    # 95% interval
    assert "p=0.04" in md                       # p-value
    assert "weak" in md.lower()                 # calibrated claim strength
    assert "fragile" in md.lower()              # robustness verdict
