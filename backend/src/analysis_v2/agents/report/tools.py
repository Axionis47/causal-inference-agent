"""The report tool registry the ReAct loop offers.

The model grounds itself with read tools (what ran, the slot summaries, the
DAG), writes each section's connective narrative with write_section, and
closes with finish_report. Read tools return prose observations and are not
recorded; the building actions (write_section / finish_report) record the
prose and a ReportToolCall. The prose feeds BOTH the notebook and the report
markdown at assembly, so they cannot drift. Tables and figures are NOT
model-rendered: each emitted section's data cell is attached deterministically
from sections.py, so the report never shows model-shaped JSON. Nothing here
mutates run state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.analysis_v2.agents.base import LoopTool
from src.analysis_v2.core import AnalysisRunState
from src.analysis_v2.spec import ReportToolCall, ReportToolTrace

from .guard import forbidden_hit
from .prompt import dag_identification_lines
from .sections import NARRATIVE_SECTIONS, SECTION_TITLES

# Named run-state slots the model can read prose from.
_SLOTS = ("eda", "diagnostics", "sensitivity", "claim", "design", "dossier", "flow_audit")

_NONE_SCHEMA = {"type": "object", "properties": {}}
_SLOT_SCHEMA = {
    "type": "object",
    "properties": {"slot": {"type": "string", "enum": list(_SLOTS)}},
    "required": ["slot"],
}
_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "section_id": {"type": "string", "enum": list(NARRATIVE_SECTIONS)},
        "prose": {"type": "string", "description": "plain-prose narrative for the section"},
    },
    "required": ["section_id", "prose"],
}
_FINISH_SCHEMA = {
    "type": "object",
    "properties": {"executive_summary": {"type": "string"}},
    "required": ["executive_summary"],
}


@dataclass
class ReportLedger:
    """What the loop built: the per-section narrative prose (keyed by id), the
    order it was written, the recorded calls, and the executive summary. Read
    observations are not recorded here, only the building actions."""

    prose: dict[str, str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    calls: list[ReportToolCall] = field(default_factory=list)
    executive_summary: str = ""

    def emitted_ids(self) -> set[str]:
        return set(self.prose)


def _slot_summary(run: AnalysisRunState, slot: str) -> str | None:
    if slot == "eda":
        return run.eda_summary.story if run.eda_summary else None
    if slot == "diagnostics":
        return run.diagnostics_result.summary if run.diagnostics_result else None
    if slot == "sensitivity":
        s = run.sensitivity_result
        return f"{s.robustness.value}: {s.confidence_reason}" if s else None
    if slot == "claim":
        c = run.claim_critique
        return f"strength {c.strength.value}: {c.rationale}" if c else None
    if slot == "design":
        d = run.selected_design or (run.design_candidates[0] if run.design_candidates else None)
        return d.rationale if d else None
    if slot == "dossier":
        return run.dataset_dossier.summary if run.dataset_dossier else None
    if slot == "flow_audit":
        f = run.flow_audit
        return "; ".join(s.signal for s in f.signals) if f and f.signals else None
    return None


def build_report_tools(run: AnalysisRunState) -> tuple[ReportLedger, list[LoopTool]]:
    """(ledger, tools). Handlers close over run and the ledger; the ledger
    fills as the model reads, writes sections, and finishes."""
    ledger = ReportLedger()

    def list_available_results() -> str:
        present = [
            ("spec", run.causal_spec is not None),
            ("profile", run.dataset_profile is not None),
            ("dag", run.causal_dag is not None),
            ("design", run.selected_design is not None or bool(run.design_candidates)),
            ("eda", run.eda_summary is not None),
            ("estimate", run.estimate_result is not None),
            ("diagnostics", run.diagnostics_result is not None),
            ("sensitivity", run.sensitivity_result is not None),
            ("claim", run.claim_critique is not None),
            ("dossier", run.dataset_dossier is not None),
            ("flow_audit", run.flow_audit is not None),
        ]
        lines = [f"- {name}: {'present' if ok else 'absent'}" for name, ok in present]
        return "available results:\n" + "\n".join(lines)

    def read_result_summary(slot: str) -> str:
        if slot not in _SLOTS:
            return f"rejected: unknown slot '{slot}'; choose one of {list(_SLOTS)}"
        return _slot_summary(run, slot) or f"the {slot} result is not available in this run"

    def describe_dag() -> str:
        return "\n".join(dag_identification_lines(run))

    def write_section(section_id: str, prose: str) -> str:
        title = SECTION_TITLES.get(section_id)
        if section_id not in NARRATIVE_SECTIONS or title is None:
            ledger.calls.append(ReportToolCall(
                name=section_id or "?", section_id=section_id or "",
                kind="write_section", status="rejected", rejected="unknown section"))
            return f"rejected: cannot narrate '{section_id}'; choose one of {list(NARRATIVE_SECTIONS)}"
        hit = forbidden_hit(prose, run.claim_critique)
        if hit:
            ledger.calls.append(ReportToolCall(
                name=title, section_id=section_id, kind="write_section",
                status="rejected", rejected=f"forbidden phrase: {hit}"))
            return f"rejected: drop the phrase '{hit}'; it overclaims. Rewrite within the allowed strength."
        if section_id not in ledger.prose:
            ledger.order.append(section_id)
        ledger.prose[section_id] = prose.strip()
        ledger.calls.append(ReportToolCall(
            name=title, section_id=section_id, kind="write_section", status="ok"))
        return f"wrote the {title} section"

    def finish_report(executive_summary: str) -> str:
        hit = forbidden_hit(executive_summary, run.claim_critique)
        if hit:
            ledger.calls.append(ReportToolCall(
                name="executive_summary", section_id="summary", kind="finish",
                status="rejected", rejected=f"forbidden phrase: {hit}"))
            return f"rejected: drop the phrase '{hit}' from the summary."
        ledger.executive_summary = executive_summary.strip()
        ledger.calls.append(ReportToolCall(
            name="executive_summary", section_id="summary", kind="finish", status="ok"))
        return "executive summary recorded; the report will open with it"

    tools = [
        LoopTool("list_available_results",
                 "Which analysis results and artifacts exist, one line each.",
                 _NONE_SCHEMA, list_available_results),
        LoopTool("read_result_summary",
                 "The existing prose summary for one named result slot.",
                 _SLOT_SCHEMA, read_result_summary),
        LoopTool("describe_dag",
                 "The causal model: adjustment set, identifiability, latent confounding.",
                 _NONE_SCHEMA, describe_dag),
        LoopTool("write_section",
                 "Append one section's connective narrative (its table/figure attaches automatically).",
                 _WRITE_SCHEMA, write_section),
        LoopTool("finish_report",
                 "Close the report with a short executive summary.",
                 _FINISH_SCHEMA, finish_report),
    ]
    return ledger, tools


def collect_trace(ledger: ReportLedger, *, exhausted: bool = False) -> ReportToolTrace:
    return ReportToolTrace(calls=list(ledger.calls), exhausted=exhausted)
