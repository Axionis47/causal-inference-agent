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
from src.analysis_v2.spec import MethodLane

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


# Designs that identify by a design assumption rather than backdoor adjustment on
# the observed graph. For these, an open backdoor path is NOT grounds to call the
# estimate a mere association; the design assumption carries identification.
_DESIGN_IDENTIFICATION: dict[MethodLane, str] = {
    MethodLane.DID: (
        "parallel trends across groups; time-invariant confounders cancel in the "
        "before/after difference"
    ),
    MethodLane.RDD: (
        "continuity at the cutoff; units just above and below are comparable, so "
        "smoothly-varying confounders do not drive the jump"
    ),
    MethodLane.IV: (
        "the instrument's relevance and the exclusion restriction; the instrument "
        "moves treatment but affects the outcome only through it"
    ),
    MethodLane.TIME_SERIES: "no other change coinciding with the intervention",
    MethodLane.SURVIVAL: (
        "no unmeasured confounding given the covariates, plus proportional hazards"
    ),
}


def _design_lane(run: AnalysisRunState) -> MethodLane | None:
    if run.method_plan is not None:
        return run.method_plan.lane
    if run.selected_design is not None:
        return run.selected_design.lane
    return None


def dag_identification_lines(run: AnalysisRunState) -> list[str]:
    """How to frame identification for the report writer, made design-aware.
    Backdoor-adjustment identifiability is the right test only for observational
    and matching designs; for DiD/RDD/IV/time-series/survival the estimate is
    identified by the design's own assumption, so an open backdoor path on the
    observed graph must NOT be reported as 'an association, not a causal effect'.
    Latent confounders are then framed as threats to that assumption."""
    dag = run.causal_dag
    if dag is None:
        return ["no causal graph available."]
    latents = [n.name for n in dag.nodes if not n.observed]
    basis = _DESIGN_IDENTIFICATION.get(_design_lane(run))
    if basis is not None:
        lines = [
            f"identification rests on the design assumption: {basis}.",
            "backdoor adjustment on the observed graph is NOT the identification "
            "route here; do not call the estimate 'an association, not a causal "
            "effect' because a backdoor path is open.",
        ]
        if latents:
            lines.append(
                f"suspected latent confounders {latents} matter only as threats to "
                "that design assumption; name them as risks to it, not as proof the "
                "effect is unidentified."
            )
        return lines
    identifiable, reason = dag.is_identifiable()
    adjustment = sorted(dag.adjustment_set())
    lines = [
        f"adjustment set (backdoor): {adjustment or 'empty'}",
        f"identified by adjustment: {identifiable} ({reason})",
        f"latent confounding present: {dag.has_latent_confounding()}",
    ]
    if latents:
        lines.append(f"suspected latent confounders: {latents}")
    return lines


def _dag_summary(run: AnalysisRunState) -> str:
    return "CAUSAL MODEL:\n" + "\n".join(
        f"- {line}" for line in dag_identification_lines(run)
    )


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
