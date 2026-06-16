"""Final report assembly. Every sentence obeys the claim critique."""
from __future__ import annotations

from src.analysis_v2.core import AnalysisRunState

from .guard import forbidden_hit

_STRENGTH_PHRASES = {
    "strong": "the evidence is consistent with a causal effect",
    "moderate": "the evidence suggests an effect under the design's assumptions",
    "weak": "the adjusted association is suggestive but vulnerable to confounding",
    "exploratory": "these are exploratory associations, not causal effect estimates",
    "not_supported": "the analysis cannot support a causal claim on this data",
}


def headline_lines(run: AnalysisRunState) -> list[str]:
    """The load-bearing numbers stated in prose: primary estimand, point
    estimate, interval, p-value, the calibrated claim strength, and robustness.
    Shared by the deterministic report and the agentic report so the human-facing
    markdown always carries the number, not only the attached table or figure."""
    primary = run.estimate_result.primary
    critique = run.claim_critique
    sens = run.sensitivity_result
    interval = ""
    if primary.ci_lower is not None and primary.ci_upper is not None:
        interval = f" (95% interval {primary.ci_lower:.4g} to {primary.ci_upper:.4g})"
    pval = f", p={primary.p_value:.3g}" if primary.p_value is not None else ""
    no_clear = (
        primary.ci_lower is not None
        and primary.ci_upper is not None
        and primary.ci_lower <= 0 <= primary.ci_upper
    )
    strength_line = _STRENGTH_PHRASES[critique.strength.value]
    if no_clear:
        strength_line = (
            "the interval spans zero, so the data do not show a clear effect; "
            + strength_line
        )
    lines = [
        f"**Estimated effect.** {primary.estimand}: {primary.estimate:.4g}{interval}"
        f"{pval}. {primary.interpretation}.",
        "",
        f"**Reading the result.** Claim strength is **{critique.strength.value}**: "
        f"{strength_line}, under these assumptions.",
    ]
    if sens is not None:
        lines += ["", f"**Robustness.** {sens.robustness.value}: {sens.confidence_reason}."]
    return lines


def build_report_markdown(run: AnalysisRunState) -> str:
    spec = run.causal_spec
    result = run.estimate_result
    critique = run.claim_critique

    lines: list[str] = []
    lines.append("# Causal analysis report")
    lines.append("")
    lines.append(f"**Question.** {run.causal_question}")
    lines.append("")
    lines.append(
        f"**Design.** {run.selected_design.design_label if run.selected_design else result.lane.value}"
        f" on {result.n_rows_used:,} rows"
        f" ({spec.question_type.value} question)."
    )
    lines.append("")
    lines += headline_lines(run)
    if run.diagnostics_result is not None and run.diagnostics_result.summary:
        lines.append("")
        lines.append(f"**Diagnostics.** {run.diagnostics_result.summary}.")
    if run.eda_summary is not None and run.eda_summary.story:
        lines.append("")
        lines.append("## What the data look like")
        lines.append("")
        lines.append(run.eda_summary.story)
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for limit in critique.limitations:
        lines.append(f"- {limit}")
    lines.append("")
    lines.append("## Method detail")
    lines.append("")
    lines.append(
        f"Estimator: {result.estimator}; covariates: "
        f"{', '.join(result.covariates_used) or 'none'}."
    )
    for effect in result.effects:
        eff_interval = (
            f", 95% interval {effect.ci_lower:.4g} to {effect.ci_upper:.4g}"
            if effect.ci_lower is not None and effect.ci_upper is not None
            else ""
        )
        lines.append(
            f"- {effect.estimand}: {effect.estimate:.4g}"
            f"{f' (se {effect.std_error:.3g})' if effect.std_error else ''}{eff_interval}"
        )
    text = "\n".join(lines)
    # Deterministic inputs: this assert can never fire here. The agentic path
    # (S10) reuses forbidden_hit to sanitize-or-fall-back per section instead.
    hit = forbidden_hit(text, critique)
    assert hit is None, f"forbidden phrase in report: {hit}"
    return text


def build_dashboard_payload(run: AnalysisRunState) -> dict:
    primary = run.estimate_result.primary
    return {
        "headline": {
            "estimand": primary.estimand,
            "estimate": primary.estimate,
            "ci_lower": primary.ci_lower,
            "ci_upper": primary.ci_upper,
            "claim_strength": run.claim_critique.strength.value,
            "robustness": run.sensitivity_result.robustness.value
            if run.sensitivity_result else None,
        },
        "tiles": [
            {
                "agent": r.agent,
                "stage": r.stage.value,
                "status": r.status.value,
                "headline": (r.public_summary or "").splitlines()[0][:160]
                if r.public_summary else None,
                "warnings": r.warnings,
                "artifact_ids": r.artifact_ids,
                "tokens": r.tokens.model_dump(),
                "cost_usd": r.cost_usd,
            }
            for r in run.agent_runs
        ],
        "artifacts": [
            {"artifact_id": a.artifact_id, "kind": a.kind.value, "title": a.title,
             "agent": a.agent}
            for a in run.artifact_registry.artifacts
        ],
        "costs": {
            "total_input_tokens": run.total_tokens.input_tokens,
            "total_output_tokens": run.total_tokens.output_tokens,
            "total_cost_usd": run.total_cost_usd,
        },
    }
