"""The overclaiming guards, per the mission's eval requirements."""
from __future__ import annotations

from src.analysis_v2.agents.claim_critic.rubric import (
    FORBIDDEN_LANGUAGE,
    cap_for_latent_confounding,
    claim_strength,
    limitations,
)
from src.analysis_v2.spec import (
    CheckStatus,
    ClaimStrength,
    DiagnosticCheck,
    EffectEstimate,
    EstimateResult,
    MethodLane,
    QuestionType,
    RobustnessStatus,
    SensitivityResult,
)


def _result(lane=MethodLane.OBSERVATIONAL, ci=(100.0, 3000.0)) -> EstimateResult:
    return EstimateResult(
        lane=lane, estimator="x",
        effects=[EffectEstimate(estimand="ate", estimate=1500.0,
                                ci_lower=ci[0], ci_upper=ci[1])],
        n_rows_used=600, outcome="y", treatment="t",
    )


def _sens(robustness: RobustnessStatus) -> SensitivityResult:
    return SensitivityResult(
        checks=[DiagnosticCheck(name="c", status=CheckStatus.WARNING, detail="d")]
        if robustness == RobustnessStatus.FRAGILE else [],
        robustness=robustness, confidence_reason="r",
    )


def test_failed_diagnostics_pin_not_supported():
    strength, notes = claim_strength(
        QuestionType.IV, _result(MethodLane.IV), _sens(RobustnessStatus.NOT_SUPPORTED)
    )
    assert strength == ClaimStrength.NOT_SUPPORTED
    assert "assumption failed" in notes[0]


def test_driver_analysis_is_always_exploratory_even_when_robust():
    strength, _ = claim_strength(
        QuestionType.DRIVER_ANALYSIS, _result(), _sens(RobustnessStatus.ROBUST)
    )
    assert strength == ClaimStrength.EXPLORATORY


def test_robust_rdd_upgrades_and_fragile_observational_downgrades():
    up, _ = claim_strength(
        QuestionType.RDD, _result(MethodLane.RDD), _sens(RobustnessStatus.ROBUST)
    )
    assert up == ClaimStrength.STRONG
    down, _ = claim_strength(
        QuestionType.BINARY_TREATMENT, _result(), _sens(RobustnessStatus.FRAGILE)
    )
    assert down == ClaimStrength.EXPLORATORY  # weak stepped down


def test_before_after_caps_at_weak_and_warns_about_the_missing_control():
    strength, _ = claim_strength(
        QuestionType.BEFORE_AFTER,
        _result(MethodLane.TIME_SERIES),
        _sens(RobustnessStatus.ROBUST),
    )
    assert strength in (ClaimStrength.WEAK,)
    limits = limitations(
        QuestionType.BEFORE_AFTER, _result(MethodLane.TIME_SERIES), None
    )
    assert any("control group" in entry for entry in limits)


def test_iv_limitation_names_the_untestable_exclusion_restriction():
    limits = limitations(QuestionType.IV, _result(MethodLane.IV), None)
    assert any("exclusion restriction" in entry for entry in limits)


def test_zero_spanning_ci_converts_to_a_no_clear_effect_note():
    _, notes = claim_strength(
        QuestionType.NO_EFFECT, _result(ci=(-200.0, 300.0)), None
    )
    assert any("spans zero" in n for n in notes)


def test_latent_confounding_caps_a_moderate_claim_to_weak():
    capped, note, limitation = cap_for_latent_confounding(
        ClaimStrength.MODERATE, ["market size"]
    )
    assert capped == ClaimStrength.WEAK
    assert note is not None and "not point-identified" in note
    assert "market size" in limitation


def test_latent_confounding_cap_leaves_an_already_weak_claim_unchanged():
    capped, note, _ = cap_for_latent_confounding(ClaimStrength.WEAK, [])
    assert capped == ClaimStrength.WEAK
    assert note is None


def test_forbidden_language_never_leaks_into_allowed():
    from src.analysis_v2.agents.claim_critic.rubric import ALLOWED_LANGUAGE

    assert not set(ALLOWED_LANGUAGE) & set(FORBIDDEN_LANGUAGE)
    assert "proves" in FORBIDDEN_LANGUAGE and "definitely causes" in FORBIDDEN_LANGUAGE
