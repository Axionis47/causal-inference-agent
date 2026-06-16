"""System prompt and mission for the agentic report loop.

The model decides the order and emphasis of the report and writes the
connective narrative; the deterministic floor still guarantees every finding
is covered, and the data cells attach automatically. The language rules mirror
the report guard: the model may state the claim at the critic's strength but
never use the forbidden phrasings.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import LoopTool
from src.analysis_v2.core import AnalysisRunState

# The loop only curates narrative on top of a guaranteed floor, so the budget
# just bounds curation; the floor-fill covers anything left unwritten.
MAX_NARRATIVE_TOOLS = 20
MAX_TURNS = 8

SYSTEM_PROMPT = """You are the report writer on a causal analysis team. The \
analysis is finished; your job is to turn its results into a report that reads \
like prose, not a data dump.

How you work:
- Read the results first (list_available_results, read_result_summary, \
describe_dag), then write each section's connective narrative with write_section, \
then close with finish_report.
- Tie the story to the causal model: what the design assumed, what the DAG implies \
for identification, what the estimate means under those assumptions.
- Cover every finding that exists; reorder for narrative flow, but do not skip a \
result that ran.

Language:
- You MAY state the causal claim at the strength the critic permits and use the \
allowed phrasings. You may NOT use the forbidden phrasings; a rejected call tells \
you which phrase to drop. Describe honestly; never overclaim.
- Write plain prose. Never paste JSON or raw number tables into a section; each \
section's table and figure are attached for you."""


def _dag_summary(run: AnalysisRunState) -> str:
    dag = run.causal_dag
    if dag is None:
        return "CAUSAL MODEL: no graph available."
    identifiable, reason = dag.is_identifiable()
    adjustment = sorted(dag.adjustment_set())
    latents = [n.name for n in dag.nodes if not n.observed]
    lines = [
        "CAUSAL MODEL:",
        f"- adjustment set (backdoor): {adjustment or 'empty'}",
        f"- identified by adjustment: {identifiable} ({reason})",
    ]
    if latents:
        lines.append(f"- suspected latent confounders: {latents}")
    return "\n".join(lines)


def build_mission(run: AnalysisRunState, tools: list[LoopTool]) -> str:
    critique = run.claim_critique
    lane = (
        run.selected_design.design_label
        if run.selected_design
        else (run.method_plan.lane.value if run.method_plan else "undecided")
    )
    allowed = (
        ", ".join(critique.allowed_language)
        if critique and critique.allowed_language
        else "(none specified)"
    )
    forbidden = (
        ", ".join(critique.forbidden_language)
        if critique and critique.forbidden_language
        else "(none)"
    )
    tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools) or "- (none)"
    return (
        f"CAUSAL QUESTION: {run.causal_question}\n"
        f"DESIGN: {lane}\n"
        f"CLAIM STRENGTH: {critique.strength.value if critique else 'unknown'}\n"
        f"ALLOWED PHRASINGS: {allowed}\n"
        f"FORBIDDEN PHRASINGS: {forbidden}\n\n"
        f"{_dag_summary(run)}\n\n"
        f"TOOLS:\n{tools_block}\n\n"
        "Read the results, then write the report section by section in an order "
        "that tells the story, and close with a short executive summary. Spend at "
        f"most {MAX_NARRATIVE_TOOLS} narrative calls."
    )
