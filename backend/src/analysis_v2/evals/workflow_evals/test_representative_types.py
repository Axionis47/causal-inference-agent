"""Full-spine evals for the taxonomy types the original six cases missed:
simple_effect, binary_treatment, multi_factor, interaction, rdd,
before_after, heterogeneous_effects, driver_analysis, no_effect, and
mechanism_search. Together with test_representative_workflows.py every
one of the 16 representative question types runs S1..S12 end to end.

Hillstrom/telco cases read the gitignored cache (fixtures/cache.py) and
skip offline. Frames that need encoding (yes/no flags, derived two-arm
filters) do it in frame_fn with the recipe stated, mirroring the
manifest's derived-fixture pattern; doing that inside the spine is the
investigator agent's job and is asserted in its own evals.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.core import AnalysisStage, RunStatus
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.evals.fixtures.cache import cached_path, load
from src.analysis_v2.spec import Confidence, MethodLane, QuestionType
from src.analysis_v2.state import JobStatus

from .harness import DATA, Case, drive, stub_llm

LALONDE_NSW_CONFOUNDERS = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
LALONDE_OBS_CONFOUNDERS = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def _hillstrom_two_arm() -> pd.DataFrame:
    """Womens E-Mail vs No E-Mail with a derived 0/1 treatment flag."""
    frame = load("hillstrom")
    frame = frame[frame["segment"].isin(["Womens E-Mail", "No E-Mail"])].copy()
    frame["received_email"] = (frame["segment"] == "Womens E-Mail").astype(int)
    return frame.drop(columns=["segment", "history_segment", "zip_code", "channel"])


def _hillstrom_null_subgroup() -> pd.DataFrame:
    """Manifest recipe: mens-only buyers, womens campaign vs control."""
    frame = _hillstrom_two_arm()
    frame = frame[(frame["mens"] == 1) & (frame["womens"] == 0)]
    return frame.drop(columns=["mens", "womens"])


def _telco_encoded() -> pd.DataFrame:
    frame = load("telco")
    frame["churn"] = (frame["Churn"] == "Yes").astype(int)
    frame["contract_monthly"] = (frame["Contract"] == "Month-to-month").astype(int)
    frame["monthly_charges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    return frame[["churn", "contract_monthly", "tenure", "monthly_charges", "SeniorCitizen"]]


def _needs(key: str):
    return pytest.mark.skipif(
        cached_path(key) is None,
        reason=f"{key} not cached; run python -m src.analysis_v2.evals.fixtures.cache",
    )


NULL_CASE = Case(
    case_id="hillstrom-null-subgroup",  # type 15: no_effect
    frame_fn=_hillstrom_null_subgroup,
    question="Did the womens campaign raise conversion among mens-only buyers?",
    draft=IntakeDraft(
        question_type=QuestionType.NO_EFFECT, confidence=Confidence.HIGH,
        outcome_column="conversion", treatment_column="received_email",
        reasoning_summary="Randomized arms; a likely null to report honestly.",
    ),
    expected_lane=MethodLane.OBSERVATIONAL,
    estimand="ate",
    truth_band=(-0.003, 0.004),  # ground truth +0.00063, p=0.56
    extra_warning="spans zero",
)


CASES = [
    Case(
        case_id="lalonde-nsw-experimental",  # type 1: simple_effect
        frame_fn=lambda: pd.read_csv(DATA / "lalonde_nsw.csv"),
        question="In the NSW experiment, did job training raise 1978 earnings?",
        draft=IntakeDraft(
            question_type=QuestionType.SIMPLE_EFFECT, confidence=Confidence.HIGH,
            outcome_column="re78", treatment_column="treat",
            candidate_confounders=LALONDE_NSW_CONFOUNDERS,
            reasoning_summary="A randomized training program; covariates are balanced.",
        ),
        expected_lane=MethodLane.OBSERVATIONAL,
        estimand="ate",
        truth_band=(1000.0, 2500.0),  # experimental truth $1,794; adjusted 1676
    ),
    Case(
        case_id="lalonde-observational",  # type 2: binary_treatment
        frame_fn=lambda: pd.read_csv(DATA / "lalonde.csv"),
        question="Does participating in job training (treat) increase 1978 earnings?",
        draft=IntakeDraft(
            question_type=QuestionType.BINARY_TREATMENT, confidence=Confidence.HIGH,
            outcome_column="re78", treatment_column="treat",
            candidate_confounders=LALONDE_OBS_CONFOUNDERS,
            reasoning_summary="Treated vs PSID controls; strong pre-program differences.",
        ),
        expected_lane=MethodLane.OBSERVATIONAL,
        estimand="ate",
        truth_band=(500.0, 3000.0),  # naive diff is -$635; adjusted flips positive
    ),
    Case(
        case_id="student-multi-factor",  # type 4: multi_factor
        frame_fn=lambda: pd.read_csv(DATA / "student_data.csv"),
        question="Do study time, absences, and past failures together affect the final grade?",
        draft=IntakeDraft(
            question_type=QuestionType.MULTI_FACTOR, confidence=Confidence.HIGH,
            outcome_column="G3",
            candidate_factors=["studytime", "absences", "failures"],
            reasoning_summary="Open joint-factor question; G1/G2 are intermediate grades, excluded.",
        ),
        expected_lane=MethodLane.OBSERVATIONAL,
        estimand="assoc_failures",
        truth_band=(-3.0, -1.4),  # joint OLS -2.20 per prior failure
        extra_warning="association",
    ),
    Case(
        case_id="insurance-interaction",  # type 5: interaction
        frame_fn=lambda: pd.read_csv(DATA / "insurance.csv").assign(
            smoker=lambda d: (d["smoker"] == "yes").astype(int)
        ),
        question="Does the effect of smoking on charges depend on BMI?",
        draft=IntakeDraft(
            question_type=QuestionType.INTERACTION, confidence=Confidence.HIGH,
            outcome_column="charges", treatment_column="smoker",
            moderator_column="bmi",
            candidate_confounders=["bmi", "age", "children"],
            reasoning_summary="Effect-modification question: smoking premium by BMI.",
        ),
        expected_lane=MethodLane.OBSERVATIONAL,
        estimand="interaction",
        truth_band=(1100.0, 1800.0),  # verified smoker:bmi OLS coef ~ +1434
    ),
    Case(
        case_id="synthetic-scholarship-rdd",  # type 8: rdd
        frame_fn=lambda: generators.scholarship_rdd().drop(
            columns=["scholarship_fuzzy", "outcome_fuzzy"]
        ),
        question="What is the effect of the scholarship at the eligibility cutoff of 50?",
        draft=IntakeDraft(
            question_type=QuestionType.RDD, confidence=Confidence.HIGH,
            outcome_column="outcome_sharp", treatment_column="scholarship_sharp",
            running_variable_column="score", cutoff_value=50.0,
            reasoning_summary="Sharp assignment at a score threshold.",
        ),
        expected_lane=MethodLane.RDD,
        estimand="itt_jump",  # sharp design: the ITT jump is the effect
        truth_band=(6.0, 10.0),  # constructed jump 8.0
    ),
    Case(
        case_id="website-before-after",  # type 12: before_after
        frame_fn=lambda: generators.website_visitors_step(
            DATA / "daily_website_visitors.csv"
        ),
        question="Compare daily visits before and after the 2019-04-01 change.",
        draft=IntakeDraft(
            question_type=QuestionType.BEFORE_AFTER, confidence=Confidence.HIGH,
            outcome_column="visits", time_column="Date",
            intervention_date="2019-04-01",
            treatment_derived=True, treatment_clue="post-change period",
            reasoning_summary="A plain pre/post comparison around a known date.",
        ),
        expected_lane=MethodLane.TIME_SERIES,
        estimand="level_shift",
        truth_band=(200.0, 1400.0),
        extra_warning="before/after",
    ),
    pytest.param(
        Case(
            case_id="hillstrom-hte",  # type 13: heterogeneous_effects
            frame_fn=_hillstrom_two_arm,
            question="Which customers respond most to the womens email campaign?",
            draft=IntakeDraft(
                question_type=QuestionType.HETEROGENEOUS_EFFECTS,
                confidence=Confidence.HIGH,
                outcome_column="visit", treatment_column="received_email",
                moderator_column="womens",
                candidate_confounders=["womens", "mens", "newbie"],
                reasoning_summary="Randomized campaign; effect varies by purchase history.",
            ),
            expected_lane=MethodLane.OBSERVATIONAL,
            estimand="interaction",
            truth_band=(0.04, 0.085),  # verified interaction +0.062
        ),
        marks=_needs("hillstrom"), id="hillstrom-hte",
    ),
    pytest.param(
        Case(
            case_id="telco-churn-drivers",  # type 14: driver_analysis
            frame_fn=_telco_encoded,
            question="What factors appear to drive customer churn?",
            draft=IntakeDraft(
                question_type=QuestionType.DRIVER_ANALYSIS, confidence=Confidence.MEDIUM,
                outcome_column="churn",
                candidate_factors=["contract_monthly", "tenure", "monthly_charges"],
                reasoning_summary="Open-ended driver question; no designated treatment.",
            ),
            expected_lane=MethodLane.OBSERVATIONAL,
            estimand="assoc_contract_monthly",
            truth_band=(0.12, 0.25),  # joint OLS +0.187
            extra_warning="association",
        ),
        marks=_needs("telco"), id="telco-churn-drivers",
    ),
    pytest.param(
        NULL_CASE,
        marks=_needs("hillstrom"), id="hillstrom-null-subgroup",
    ),
    Case(
        case_id="student-por-mechanism",  # type 16: mechanism_search
        frame_fn=lambda: pd.read_csv(DATA / "student_por.csv"),
        question="Does early-period performance carry the effect of study time on final grades?",
        draft=IntakeDraft(
            question_type=QuestionType.MECHANISM_SEARCH, confidence=Confidence.MEDIUM,
            outcome_column="G3", treatment_column="studytime",
            mediator_column="G2",
            candidate_confounders=["failures", "Medu", "Fedu", "absences"],
            reasoning_summary="Pathway question: G2 as the candidate mediator to G3.",
        ),
        expected_lane=MethodLane.MEDIATION,
        estimand="indirect",
        truth_band=(0.2, 1.2),  # product-of-coefficients ~ 0.585
        extra_warning="timing",
    ),
]


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[c.id if hasattr(c, "id") else c.case_id for c in CASES],
)
async def test_taxonomy_type_runs_the_full_spine_into_its_truth_band(
    case: Case, storage_dir, monkeypatch
):
    stub_llm(monkeypatch, case.draft)
    run, state = await drive(f"wt-{case.case_id[:12]}", case.frame_fn(), case.question)

    assert run.status == RunStatus.COMPLETED, run.error_message
    assert run.current_state == AnalysisStage.S12_JOB_COMPLETE
    assert state.status == JobStatus.COMPLETED

    assert run.estimate_result.lane == case.expected_lane
    effect = next(
        e for e in run.estimate_result.effects if e.estimand == case.estimand
    )
    lo, hi = case.truth_band
    assert lo <= effect.estimate <= hi, (case.case_id, effect.estimate)

    assert run.claim_critique is not None
    assert run.claim_critique.limitations
    if case.extra_warning is not None:
        # honesty language may land in the limitations or the strength rationale
        joined = " ".join(
            [*run.claim_critique.limitations, run.claim_critique.rationale]
        ).lower()
        assert case.extra_warning in joined, joined

    assert run.notebook_verification.notebook_status.value == "verified_running"


async def test_the_null_case_keeps_zero_inside_the_interval(storage_dir, monkeypatch):
    if cached_path("hillstrom") is None:
        pytest.skip("hillstrom not cached")
    stub_llm(monkeypatch, NULL_CASE.draft)
    run, _ = await drive("wt-null-ci", NULL_CASE.frame_fn(), NULL_CASE.question)
    primary = run.estimate_result.primary
    assert primary.ci_lower <= 0 <= primary.ci_upper
