"""Conclusions section renderer."""

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from nbformat.v4 import new_markdown_cell

from src.agents.base import AnalysisState

from ..helpers import deduplicate_effects, generate_llm_narrative


async def render_conclusions(state: AnalysisState, *, llm=None, system_prompt: str = "") -> list:
    """Generate conclusions section with LLM narrative."""
    cells = []

    effects = deduplicate_effects(state.treatment_effects)

    # L10: Group by estimand instead of averaging all together
    by_estimand: dict[str, list] = defaultdict(list)
    for e in effects:
        by_estimand[e.estimand].append(e)

    all_positive = all(e.estimate > 0 for e in effects) if effects else False
    all_negative = all(e.estimate < 0 for e in effects) if effects else False
    direction = "positive" if all_positive else "negative" if all_negative else "mixed"

    # Try LLM-generated conclusion
    llm_context = {
        "treatment": state.treatment_variable,
        "outcome": state.outcome_variable,
        "n_methods": len(effects),
        "direction": direction,
    }

    # Per-estimand averages for LLM context
    estimand_summaries = {}
    for estimand, eff_list in by_estimand.items():
        avg = np.mean([e.estimate for e in eff_list])
        estimand_summaries[estimand] = f"{avg:.4f} ({len(eff_list)} methods)"
    llm_context["estimand_summaries"] = str(estimand_summaries)

    if state.sensitivity_results:
        llm_context["sensitivity"] = "; ".join(
            f"{s.method}: {s.robustness_value:.2f}" for s in state.sensitivity_results
        )

    if state.critique_history:
        latest = state.critique_history[-1]
        llm_context["critique_decision"] = latest.decision.value
        if latest.issues:
            llm_context["key_issues"] = "; ".join(latest.issues[:3])

    if state.domain_knowledge and state.domain_knowledge.get("uncertainties"):
        uncerts = state.domain_knowledge["uncertainties"]
        if uncerts:
            first = uncerts[0]
            if isinstance(first, dict):
                llm_context["uncertainty"] = first.get("issue", str(first))
            else:
                llm_context["uncertainty"] = str(first)

    llm_conclusion = ""
    if llm:
        llm_conclusion = await generate_llm_narrative(llm, system_prompt, "conclusions", llm_context)

    # Build conclusions section
    conclusion_md = "## Conclusions\n\n"

    # L5: Important Caveats BEFORE key findings
    conclusion_md += "### Important Caveats\n\n"
    conclusion_md += (
        "- **Observational data**: These results are based on observational data. "
        "They assume that all relevant confounders have been measured and adjusted for. "
        "Unmeasured confounding could bias the estimates in either direction.\n"
    )
    conclusion_md += "- **Model assumptions**: Each estimation method relies on specific assumptions that may not hold.\n"
    conclusion_md += "- **External validity**: Results may not generalize to other populations or settings.\n"
    n_tests = sum(1 for e in effects if e.p_value is not None)
    if n_tests > 1:
        conclusion_md += f"- **Multiple comparisons**: p-values have been adjusted for {n_tests} simultaneous tests (Holm-Bonferroni).\n"
    conclusion_md += "\n"

    if llm_conclusion:
        conclusion_md += llm_conclusion + "\n\n"
    else:
        # Fallback: template conclusion
        conclusion_md += "### Key Findings\n\n"
        conclusion_md += f"The analysis estimated the effect of **{state.treatment_variable}** "
        conclusion_md += f"on **{state.outcome_variable}** using {len(effects)} method(s).\n\n"

        # L10: Report per-estimand instead of one combined average
        for estimand, eff_list in by_estimand.items():
            avg = np.mean([e.estimate for e in eff_list])
            methods = [e.method for e in eff_list]
            conclusion_md += f"- **{estimand}**: {avg:.4f} (from {len(eff_list)} method(s): {', '.join(methods)})\n"
        conclusion_md += f"\n- **Direction consistency**: {'All methods agree on a **' + direction + '** effect.' if direction != 'mixed' else 'Methods show **mixed** directions — interpret with caution.'}\n\n"

    # L11: Robustness assessment from sensitivity results
    if state.sensitivity_results:
        conclusion_md += "### Robustness Assessment\n\n"
        for s in state.sensitivity_results:
            interpretation = getattr(s, "interpretation", None) or f"robustness value = {s.robustness_value:.2f}"
            conclusion_md += f"- **{s.method}**: {interpretation}\n"
        conclusion_md += "\n"

    # Recommendations (from orchestrator)
    if state.recommendations:
        conclusion_md += "### Recommendations\n\n"
        for i, rec in enumerate(state.recommendations, 1):
            conclusion_md += f"{i}. {rec}\n"
        conclusion_md += "\n"

    cells.append(new_markdown_cell(conclusion_md))

    # Reproducibility footer
    repro_md = f"""---

### Reproducibility Information

- **Analysis Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
- **Job ID**: {state.job_id}
- **Dataset**: {state.dataset_info.name or state.dataset_info.url}
- **Treatment Variable**: {state.treatment_variable}
- **Outcome Variable**: {state.outcome_variable}

This notebook was automatically generated by the Causal Inference Orchestrator.
"""
    cells.append(new_markdown_cell(repro_md))

    return cells
