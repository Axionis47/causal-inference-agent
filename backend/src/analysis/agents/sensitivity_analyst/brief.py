"""Contract surface for the sensitivity_analyst.

The sensitivity analyst takes the estimates produced by
effect_estimator and stress-tests them. The brief reads from the
typed `state.sensitivity_results: list[SensitivityResult]` written
after the ReAct loop and derives two flags from the closed enum:

- WEAK_TO_UNOBSERVED: the E-value result indicates the estimate is
  sensitive to unmeasured confounding (robustness_value < 1.5, which
  is what the agent's own auto-finalize calls "low").
- NOT_ROBUST: at least two sensitivity results flag concerns across
  any method family (spec curve unstable, placebo concerning,
  subgroup heterogeneous, bootstrap unstable, Rosenbaum gamma < 1.5,
  or E-value < 1.5). Two-or-more is the threshold so a lone signal
  does not over-fire; the lone E-value case is already covered by
  WEAK_TO_UNOBSERVED, so NOT_ROBUST adds value only when concerns
  agree across methods.

Preflight refuses with NEEDS_NOT_MET when state.treatment_effects is
empty: nothing to stress-test means the orchestrator should reroute to
effect_estimator before retrying.
"""
from __future__ import annotations

from src.analysis.agents.base import AnalysisState, SensitivityResult
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "sensitivity_analyst"

E_VALUE_WEAK_MAX = 1.5
ROSENBAUM_WEAK_MAX = 1.5
SPEC_CURVE_CV_MAX = 0.4
NOT_ROBUST_MIN_CONCERNS = 2


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "How robust are the treatment effect estimates to assumption "
        "violations and unmeasured confounding?"
    ),
    needs=("treatment_effects",),
    delivers=(
        "sensitivity_results",
        "agent_briefs[sensitivity_analyst]",
    ),
    refuses_when=(
        "state.treatment_effects is empty (effect_estimator has not produced estimates)",
    ),
    success_criteria=(
        Criterion(
            id="sensitivity.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['sensitivity_analyst']"
            ),
        ),
        Criterion(
            id="sensitivity.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "state.treatment_effects is empty"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="sensitivity.flag.weak_to_unobserved",
            description=(
                "raises WEAK_TO_UNOBSERVED when the E-value method's "
                "robustness_value is below 1.5"
            ),
            raises_flag=Flag.WEAK_TO_UNOBSERVED,
        ),
        Criterion(
            id="sensitivity.flag.not_robust",
            description=(
                "raises NOT_ROBUST when at least two sensitivity "
                "results flag concerns across any method family"
            ),
            raises_flag=Flag.NOT_ROBUST,
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse when there are no estimates to stress-test."""
    if not state.treatment_effects:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline=(
                "no treatment_effects to stress-test; "
                "effect_estimator has not produced estimates"
            ),
            issues=["state.treatment_effects is empty"],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from state.sensitivity_results."""
    results = state.sensitivity_results
    if not results:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="sensitivity analyst produced no results (loop or load failure)",
            artifact_keys=[],
        )

    flags: list[Flag] = []
    issues: list[str] = []

    e_value = _find_method(results, "e-value")
    if (
        e_value is not None
        and e_value.robustness_value is not None
        and e_value.robustness_value < E_VALUE_WEAK_MAX
    ):
        flags.append(Flag.WEAK_TO_UNOBSERVED)
        issues.append(
            f"E-value is {e_value.robustness_value:.2f} "
            f"(< {E_VALUE_WEAK_MAX:.1f}); weak unmeasured confounding could explain the effect"
        )

    n_concerns = sum(1 for r in results if _is_concerning(r))
    if n_concerns >= NOT_ROBUST_MIN_CONCERNS:
        flags.append(Flag.NOT_ROBUST)
        issues.append(
            f"{n_concerns} sensitivity result(s) flagged concerns "
            f"(>= {NOT_ROBUST_MIN_CONCERNS}-method threshold)"
        )

    methods = sorted({r.method for r in results})
    headline = (
        f"{len(results)} sensitivity test(s) run "
        f"across {len(methods)} method(s)"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=["sensitivity_results"],
    )


def _find_method(
    results: list[SensitivityResult],
    needle: str,
) -> SensitivityResult | None:
    """First result whose method name contains `needle` (case-insensitive)."""
    target = needle.lower()
    for r in results:
        if target in r.method.lower():
            return r
    return None


def _is_concerning(r: SensitivityResult) -> bool:
    """Per-method concern check matching the agent tools' own labels.

    Each tool writes a labelled interpretation string; we trust those
    labels rather than re-deriving thresholds, except for spec curve
    where the CV detail is the cleaner signal.
    """
    method = r.method.lower()
    interp = r.interpretation.lower()

    if "e-value" in method:
        return (
            r.robustness_value is not None
            and r.robustness_value < E_VALUE_WEAK_MAX
        )
    if "rosenbaum" in method:
        return (
            r.robustness_value is not None
            and r.robustness_value < ROSENBAUM_WEAK_MAX
        )
    if "specification curve" in method:
        cv = r.details.get("cv") if isinstance(r.details, dict) else None
        all_same_sign = (
            r.details.get("all_same_sign") if isinstance(r.details, dict) else True
        )
        return (
            (isinstance(cv, int | float) and cv > SPEC_CURVE_CV_MAX)
            or all_same_sign is False
        )
    if "placebo" in method:
        return "concerning" in interp
    if "subgroup" in method:
        return "heterogeneous" in interp
    if "bootstrap" in method:
        return "unstable" in interp
    return False
