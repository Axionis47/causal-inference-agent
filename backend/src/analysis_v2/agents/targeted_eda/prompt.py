"""The system prompt and mission for the agentic EDA loop. The model decides
which offered checks to run; the mission hands it the design, the resolved
columns, and a summary of the causal model so its choices stay grounded. The
causal-language ban mirrors the deterministic story prompt.
"""
from __future__ import annotations

from src.analysis_v2.agents.base import LoopTool
from src.analysis_v2.core import AnalysisRunState

# The budget the model curates within: enough to tell the story, few enough to
# stay readable. The data-health floor runs separately and is not counted here.
MIN_TARGETED_CHECKS = 4
MAX_TARGETED_CHECKS = 7

SYSTEM_PROMPT = """You are the exploratory-data-analysis analyst on a causal \
analysis team. Decide which checks matter for THIS design and question, and run \
them by calling tools.

Hard rules:
- Describe patterns only. Never claim anything causes anything; never call a \
difference an effect; never invent numbers that are not in a tool observation.
- Call only the tools offered. The menu is already constrained to what this \
design and the causal model allow; a rejected call tells you why.
- Start from what the design needs, read each observation, and decide the next \
check. When you have seen enough, stop and write a short plain-text summary of \
the patterns and what the warnings mean in practice."""


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
        f"- testable independencies implied: {len(dag.testable_implications())}",
    ]
    if latents:
        lines.append(f"- suspected latent confounders: {latents}")
    return "\n".join(lines)


def build_mission(run: AnalysisRunState, tools: list[LoopTool]) -> str:
    spec = run.causal_spec
    treatment = spec.treatment.column if spec and spec.treatment else None
    outcome = spec.outcome.column if spec and spec.outcome else None
    profile = run.dataset_profile
    shape = (
        f"{profile.n_rows:,} rows x {profile.n_columns} columns"
        if profile else "shape unknown"
    )
    lane = run.design_candidates[0].lane.value if run.design_candidates else "undecided"
    tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools) or "- (none)"
    return (
        f"CAUSAL QUESTION: {run.causal_question}\n"
        f"QUESTION TYPE: {spec.question_type.value if spec else 'unknown'}\n"
        f"LEADING DESIGN: {lane}\n"
        f"TREATMENT: {treatment or 'not resolved'} | "
        f"OUTCOME: {outcome or 'not resolved'}\n"
        f"DATASET: {run.dataset_name or 'unnamed'} ({shape})\n\n"
        f"{_dag_summary(run)}\n\n"
        f"AVAILABLE CHECKS (call only these):\n{tools_block}\n\n"
        f"Run about {MIN_TARGETED_CHECKS}-{MAX_TARGETED_CHECKS} checks, the "
        "handful that best tell this data's story for this design, then finish "
        "with a short plain-text summary of the patterns and what the warnings "
        "mean. The basic data-health checks run automatically; spend your "
        "budget on what is design-relevant."
    )
