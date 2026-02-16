"""Introduction section renderer."""

from datetime import datetime

from nbformat.v4 import new_markdown_cell

from src.agents.base import AnalysisState

from ..helpers import deduplicate_effects, generate_llm_narrative


async def render_introduction(state: AnalysisState, *, llm=None, system_prompt: str = "") -> list:
    """Generate title and introduction."""
    cells = []

    title = "# Causal Inference Analysis Report\n\n"
    title += f"**Dataset**: {state.dataset_info.name or state.dataset_info.url}\n\n"
    title += f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    title += f"**Job ID**: {state.job_id}\n"
    cells.append(new_markdown_cell(title))

    # Static intro
    intro = "## Introduction\n\n"
    intro += "This notebook reports the results of an automated causal inference analysis "
    intro += f"estimating the effect of **{state.treatment_variable}** on **{state.outcome_variable}**.\n\n"

    if state.data_profile:
        intro += f"The dataset contains **{state.data_profile.n_samples:,}** observations "
        intro += f"and **{state.data_profile.n_features}** variables.\n\n"

    # Pipeline overview
    intro += "### Pipeline Steps Completed\n\n"
    steps = []
    effects = deduplicate_effects(state.treatment_effects)
    if state.data_profile:
        steps.append("Data Profiling")
    if state.domain_knowledge:
        steps.append("Domain Knowledge Extraction")
    if state.data_repairs:
        n_repairs = len(state.data_repairs)
        steps.append(f"Data Repair ({n_repairs} repair{'s' if n_repairs != 1 else ''})")
    if state.eda_result:
        steps.append("Exploratory Data Analysis")
    if state.proposed_dag:
        steps.append(f"Causal Discovery ({state.proposed_dag.discovery_method})")
    if state.ps_diagnostics:
        quality = state.ps_diagnostics.get("model_quality", "unknown")
        steps.append(f"Propensity Score Diagnostics (quality: {quality})")
    if state.treatment_effects:
        methods = [e.method for e in effects]
        steps.append(f"Treatment Effect Estimation ({', '.join(methods)})")
    if state.sensitivity_results:
        steps.append(f"Sensitivity Analysis ({len(state.sensitivity_results)} tests)")
    if state.critique_history:
        steps.append(f"Quality Review ({state.critique_history[-1].decision.value})")

    for i, step in enumerate(steps, 1):
        intro += f"{i}. {step}\n"

    cells.append(new_markdown_cell(intro))

    # LLM-generated context paragraph
    if llm:
        llm_context = {
            "dataset": state.dataset_info.name or state.dataset_info.url or "unknown",
            "treatment": state.treatment_variable,
            "outcome": state.outcome_variable,
            "n_samples": state.data_profile.n_samples if state.data_profile else "unknown",
            "n_methods": len(effects),
        }
        if state.domain_knowledge and state.domain_knowledge.get("hypotheses"):
            top = state.domain_knowledge["hypotheses"][0]
            if isinstance(top, dict):
                llm_context["key_hypothesis"] = top.get("claim", str(top))

        llm_intro = await generate_llm_narrative(llm, system_prompt, "introduction", llm_context)
        if llm_intro:
            cells.append(new_markdown_cell(llm_intro))

    return cells
