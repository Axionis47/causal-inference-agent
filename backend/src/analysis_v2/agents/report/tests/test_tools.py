"""Report tools and mission, tested in isolation (no react loop)."""
from __future__ import annotations

from src.analysis_v2.agents.report.prompt import build_mission
from src.analysis_v2.agents.report.tools import build_report_tools, collect_trace
from src.analysis_v2.core import AnalysisRunState
from src.analysis_v2.spec import (
    CausalDAG,
    CausalEdge,
    CausalNode,
    CausalSpec,
    ClaimCritique,
    ClaimStrength,
    Confidence,
    DesignCandidate,
    MethodLane,
    QuestionType,
    VariableRef,
)


def _run() -> AnalysisRunState:
    run = AnalysisRunState(job_id="job-tools", causal_question="does training raise earnings?")
    run.causal_spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        treatment=VariableRef(column="treat"), outcome=VariableRef(column="re78"),
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
    return run


def _by_name(run):
    _, tools = build_report_tools(run)
    return {t.name: t for t in tools}


def test_build_report_tools_offers_read_write_and_finish_tools():
    names = set(_by_name(_run()))
    assert names == {
        "list_available_results", "read_result_summary", "describe_dag",
        "write_section", "finish_report",
    }


def test_describe_dag_unpacks_identifiability_not_a_raw_tuple():
    out = _by_name(_run())["describe_dag"].handler()
    assert "adjustment set" in out
    assert "age" in out  # the backdoor adjustment covariate
    assert "(True," not in out and "(False," not in out  # tuple was unpacked


def test_describe_dag_frames_did_by_its_design_assumption_not_backdoor():
    """Regression: for design-based identification (DiD/RDD/IV) the report writer
    was told 'identified by adjustment: False', which it rendered as 'not causally
    identified / association only' prose that contradicted a 'strong' DiD claim.
    The DAG framing now names the design assumption and explicitly refuses to call
    an open backdoor path a mere association."""
    run = _run()
    run.selected_design = DesignCandidate(
        lane=MethodLane.DID, design_label="difference-in-differences",
        confidence=Confidence.HIGH, rationale="NJ vs PA, before vs after",
    )
    out = _by_name(run)["describe_dag"].handler()
    assert "parallel trends" in out.lower()
    assert "identified by adjustment" not in out  # backdoor test is not the route here
    assert "not the identification route here" in out.lower()  # explicitly steered off it
    _, tools = build_report_tools(run)
    assert "parallel trends" in build_mission(run, tools).lower()


def test_read_result_summary_returns_prose_and_rejects_unknown_slots():
    tool = _by_name(_run())["read_result_summary"]
    assert "fragile sensitivity" in tool.handler(slot="claim")
    assert "backdoor adjustment" in tool.handler(slot="design")
    assert "not available" in tool.handler(slot="eda")  # slot empty in this run
    assert "rejected" in tool.handler(slot="nonsense")


def test_write_section_records_prose_and_an_emitted_id():
    ledger, tools = build_report_tools(_run())
    write = next(t for t in tools if t.name == "write_section")
    out = write.handler(section_id="estimation", prose="The adjusted gap suggests higher earnings.")
    assert "Estimation" in out
    assert ledger.emitted_ids() == {"estimation"}
    assert ledger.prose["estimation"].startswith("The adjusted gap")
    assert ledger.order == ["estimation"]
    assert ledger.calls[0].kind == "write_section" and ledger.calls[0].status == "ok"


def test_write_section_rejects_forbidden_language_without_recording_prose():
    ledger, tools = build_report_tools(_run())
    write = next(t for t in tools if t.name == "write_section")
    out = write.handler(section_id="conclusion", prose="this proves the program works")
    assert "rejected" in out and "proves" in out
    assert ledger.prose == {}  # nothing recorded
    assert ledger.calls[0].status == "rejected"


def test_write_section_rejects_a_non_narrative_section_id():
    ledger, tools = build_report_tools(_run())
    write = next(t for t in tools if t.name == "write_section")
    # 'limitations' stays deterministic; the model may not overwrite it.
    out = write.handler(section_id="limitations", prose="anything")
    assert "rejected" in out
    assert ledger.prose == {}


def test_finish_report_records_the_executive_summary():
    ledger, tools = build_report_tools(_run())
    finish = next(t for t in tools if t.name == "finish_report")
    finish.handler(executive_summary="A weak but suggestive association under adjustment.")
    assert "weak but suggestive" in ledger.executive_summary
    assert ledger.calls[-1].kind == "finish" and ledger.calls[-1].status == "ok"


def test_collect_trace_preserves_call_order():
    ledger, tools = build_report_tools(_run())
    by = {t.name: t for t in tools}
    by["write_section"].handler(section_id="estimation", prose="suggests higher earnings")
    by["write_section"].handler(section_id="diagnostics", prose="overlap looks adequate")
    by["finish_report"].handler(executive_summary="a guarded read")
    trace = collect_trace(ledger)
    assert [c.section_id for c in trace.calls] == ["estimation", "diagnostics", "summary"]


def test_list_available_results_marks_present_and_absent():
    out = _by_name(_run())["list_available_results"].handler()
    assert "claim: present" in out
    assert "diagnostics: absent" in out


def test_mission_states_strength_language_and_the_dag():
    _, tools = build_report_tools(_run())
    mission = build_mission(_run(), tools)
    assert "CLAIM STRENGTH: weak" in mission
    assert "suggests" in mission  # allowed phrasing surfaced
    assert "proves" in mission  # forbidden phrasing surfaced
    assert "adjustment set (backdoor): ['age']" in mission
    assert "write_section" in mission
