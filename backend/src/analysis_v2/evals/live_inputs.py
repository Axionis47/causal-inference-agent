"""Live-only inputs for the representative audit.

Two things the hermetic suite does not carry:

* EXTRA_CASES wires three real datasets whose lanes the deterministic
  machinery already supports but which were never turned into hermetic
  Case objects (a wide-format DiD, a proximity IV, a sharp RDD). They use
  only existing columns, so no FrameRecipe is needed; the open question
  each one answers is whether live intake tags the special role correctly.
  The draft on each is unused on the live path (the real LLM runs intake)
  and is kept only to satisfy the dataclass and document the expected roles.

* CASE_CONTEXTS is the analyst's prose about the columns, keyed by case_id.
  The live_runner injects it as dataset_info.user_provided_context so the
  run exercises the realistic path where a user describes the data, not the
  bare-question path the hermetic harness uses.
"""
from __future__ import annotations

import pandas as pd

from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.spec import Confidence, MethodLane, QuestionType

from .workflow_evals.harness import DATA, Case

EXTRA_CASES = [
    Case(
        case_id="card-krueger-did",
        frame_fn=lambda: pd.read_csv(DATA / "employment.csv"),
        question=(
            "New Jersey raised its minimum wage in April 1992; Pennsylvania did "
            "not. Using fast-food employment measured before (February) and after "
            "(November), did the increase reduce employment in NJ relative to PA?"
        ),
        draft=IntakeDraft(
            question_type=QuestionType.DID, confidence=Confidence.HIGH,
            group_column="state", treatment_derived=True,
            treatment_clue="New Jersey (state=1) was exposed to the wage increase",
            reasoning_summary="A policy comparison with a wide pre/post outcome.",
        ),
        expected_lane=MethodLane.DID,
        estimand="att",
        truth_band=(1.5, 4.0),  # published DID +2.75
    ),
    Case(
        case_id="card-proximity-iv",
        frame_fn=lambda: pd.read_csv(DATA / "card_schooling_wages.csv"),
        question=(
            "Does an additional year of education causally increase wages? "
            "Education is self-selected and confounded by unobserved ability; use "
            "growing up near a four-year college as an instrument for years of "
            "education on log hourly wage."
        ),
        draft=IntakeDraft(
            question_type=QuestionType.IV, confidence=Confidence.HIGH,
            outcome_column="lwage", treatment_column="educ",
            instrument_column="nearc4",
            candidate_confounders=["exper", "expersq", "black", "south", "smsa"],
            reasoning_summary="An instrumented return-to-schooling question.",
        ),
        expected_lane=MethodLane.IV,
        estimand="late",
        truth_band=(0.08, 0.20),  # 2SLS ~0.13
    ),
    Case(
        case_id="bank-recovery-rdd",
        frame_fn=lambda: pd.read_csv(DATA / "bank_data.csv"),
        question=(
            "The bank assigned debts with expected recovery above $1000 to a more "
            "intensive recovery strategy. Did the more intensive strategy cause "
            "higher actual recovery? Use the $1000 threshold."
        ),
        draft=IntakeDraft(
            question_type=QuestionType.RDD, confidence=Confidence.HIGH,
            outcome_column="actual_recovery_amount",
            treatment_column="recovery_strategy",
            running_variable_column="expected_recovery_amount",
            cutoff_value=1000.0,
            reasoning_summary="A sharp threshold-assignment design.",
        ),
        expected_lane=MethodLane.RDD,
        estimand="itt_jump",
        truth_band=(150.0, 400.0),  # local-linear jump ~278
    ),
]


CASE_CONTEXTS: dict[str, str] = {
    "advertising-dose-response": (
        "Each row is one regional market. 'TV Ad Budget ($)', 'Radio Ad Budget "
        "($)' and 'Newspaper Ad Budget ($)' are the spend on each channel for a "
        "single product in that market (thousands of dollars); 'Sales ($)' is "
        "units sold there (thousands). Spend was set by the marketing team, not "
        "randomized. I want to know whether raising TV spend actually drives "
        "sales, separated from radio and newspaper spend."
    ),
    "lalonde-observational": (
        "Each row is a person. 'treat'=1 means they took part in a job-training "
        "program; 0 is an untreated comparison group drawn from a national "
        "survey, which was not randomized and is richer and older on average. "
        "'re78' is 1978 earnings in dollars (the outcome); 're74' and 're75' are "
        "earnings before the program; 'age','educ','black','hispan','married',"
        "'nodegree' are background characteristics. 'Unnamed: 0' is just a row "
        "index, ignore it. I want the causal effect of training on 1978 "
        "earnings; the treated and control groups differ a lot before treatment."
    ),
    "card-krueger-did": (
        "Each row is a fast-food restaurant. 'state'=1 is New Jersey and 0 is "
        "Pennsylvania. In April 1992 New Jersey raised its minimum wage and "
        "Pennsylvania did not. 'total_emp_feb' is full-time-equivalent "
        "employment at the restaurant in February 1992 (before the change) and "
        "'total_emp_nov' is the same in November 1992 (after). I want the causal "
        "effect of the minimum-wage increase on employment, using Pennsylvania "
        "as the comparison for what New Jersey would otherwise have done."
    ),
    "card-proximity-iv": (
        "Each row is a young man surveyed in 1976. 'educ' is years of schooling "
        "completed; 'lwage' is the log of his hourly wage (the outcome). "
        "'nearc4'=1 means he grew up near a four-year college. Schooling is a "
        "personal choice tangled up with unobserved ability, so a plain "
        "wage-on-schooling regression is biased upward. I want the causal return "
        "to an extra year of schooling, using growing up near a college as an "
        "instrument for how much schooling he got (it shifts schooling but "
        "should not affect wages except through schooling). 'exper','expersq',"
        "'black','south','smsa' and the region dummies are background controls."
    ),
    "bank-recovery-rdd": (
        "Each row is a defaulted debt the bank is collecting. The bank used a "
        "rule: debts with an 'expected_recovery_amount' above $1000 were assigned "
        "to a more intensive, more expensive recovery strategy, and "
        "'recovery_strategy' records which level was used. "
        "'actual_recovery_amount' is how much was actually recovered (the "
        "outcome). 'age' and 'sex' describe the debtor. Because the strategy is "
        "assigned purely by whether expected recovery crosses $1000, I want the "
        "causal effect of the more intensive strategy by comparing debts just "
        "above versus just below the $1000 cutoff."
    ),
    "student-por-mechanism": (
        "Each row is a secondary-school student in a Portuguese-language course. "
        "'studytime' is weekly study time on an ordinal 1-4 scale. The course is "
        "graded in three periods: 'G1' (first), 'G2' (second), and 'G3' (the "
        "final grade, the outcome), so G1 precedes G2 precedes G3 in time. "
        "'failures' is past class failures; 'Medu'/'Fedu' are mother's/father's "
        "education; 'schoolsup' is extra school support; 'absences' is days "
        "missed. I want the pathway: does studying more raise the final grade "
        "mainly by raising the earlier-period grades, or is there a direct effect "
        "left after accounting for them?"
    ),
    "heart-failure-survival": (
        "Each row is a heart-failure patient followed after a clinic visit. "
        "'time' is follow-up time in days and 'DEATH_EVENT'=1 means the patient "
        "died during follow-up while 0 means they were still alive when last seen "
        "(censored, not a survivor forever). 'ejection_fraction' is the "
        "percentage of blood the heart pumps per beat, a key measure of heart "
        "function. 'age','serum_creatinine','serum_sodium','diabetes','anaemia',"
        "'smoking','sex','high_blood_pressure' are other clinical measurements. I "
        "want to know whether a higher ejection fraction delays death, properly "
        "accounting for patients who had not died by the end of follow-up."
    ),
    "website-visitors-its-step": (
        "This is a daily time series for a single website. 'Date' is the calendar "
        "day and 'visits' is the number of unique visitors that day (the "
        "outcome). On 2019-04-01 the site shipped a redesign. There is strong "
        "weekly seasonality (weekday versus weekend) tied to the day of week. I "
        "want the causal effect of the redesign on daily visits, separating the "
        "real step change from the ordinary weekly swings and any pre-existing "
        "trend."
    ),
}
