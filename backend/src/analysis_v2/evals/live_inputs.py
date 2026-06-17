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
        "Each row is one observation with three advertising-spend figures and a "
        "sales figure; the columns are labelled in dollars and I will treat "
        "them at face value rather than assume a different unit. The treatment "
        "is the continuous dose 'TV Ad Budget ($)', and the outcome is 'Sales "
        "($)'. I want the average effect on sales of one more unit of TV spend, "
        "that is the slope of sales on 'TV Ad Budget ($)'. This is "
        "observational rather than an experiment, and the spend channels may "
        "move together, so the TV effect must be estimated while adjusting for "
        "the two other pre-treatment channels, 'Radio Ad Budget ($)' and "
        "'Newspaper Ad Budget ($)'; treat both as confounders, not mediators. "
        "There is no grouping, time, instrument, or cutoff variable here, so "
        "this is an observational continuous dose-response, not a "
        "difference-in-differences, IV, or regression-discontinuity design. The "
        "leading unnamed row index is not a variable; ignore it."
    ),
    "lalonde-observational": (
        "Each row is a person. 'treat' is the treatment: 1 means they enrolled "
        "in the job-training program, 0 is a non-randomized comparison group "
        "pulled from a national survey (CPS) that is richer and older on "
        "average, so the groups are not balanced before treatment. 're78' is "
        "1978 earnings in dollars and is the outcome I want the effect on. I "
        "want the causal effect of training on 're78'. Because assignment was "
        "not randomized, a raw treated-minus-control difference is misleading: "
        "it actually comes out negative simply because the comparison group "
        "earned more to begin with. So this is an observational design that "
        "needs adjustment (matching or regression on covariates). The "
        "pre-treatment confounders to adjust for are 're74' and 're75' "
        "(earnings in 1974 and 1975, before the program), 'age', 'educ', "
        "'black', 'hispan', 'married', and 'nodegree'. All of these are "
        "measured before treatment, so none is a mediator or post-treatment "
        "variable. Exclude 'Unnamed: 0', which is just a row index and carries "
        "no information."
    ),
    "card-krueger-did": (
        "Each row is one fast-food restaurant observed twice in 1992: "
        "'total_emp_feb' is full-time-equivalent employment in February "
        "(before) and 'total_emp_nov' is the same measure in November (after). "
        "'state' is the group flag: 1 = New Jersey, which raised its minimum "
        "wage in April 1992 (the treated group), and 0 = Pennsylvania, which "
        "did not (the comparison group). I want the average effect of the "
        "minimum-wage increase on employment for New Jersey restaurants. This "
        "is a difference-in-differences setup: the outcome is employment "
        "measured at the two time points, 'state' is the treatment-group "
        "indicator, and the pre/post split is February versus November. Compare "
        "the November-minus-February change in New Jersey against the same "
        "change in Pennsylvania, using Pennsylvania's change as the "
        "counterfactual trend. There are no separate covariate columns to "
        "adjust for; 'total_emp_feb' is the baseline outcome, not a confounder, "
        "so do not treat it as a control variable or as the treatment."
    ),
    "card-proximity-iv": (
        "Each row is a young man surveyed in 1976. This is an "
        "instrumental-variables (2SLS) return-to-schooling question. The "
        "treatment is 'educ' (years of schooling completed). The outcome is "
        "'lwage', the log of his hourly wage; do not use the level column "
        "'wage'. The instrument is 'nearc4' = 1 if he grew up near a four-year "
        "college, which strongly and positively raised schooling (relevance) "
        "and plausibly affects wages only through schooling (exclusion); "
        "'nearc2' (near a two-year college) is not the instrument and should be "
        "ignored. Schooling is self-selected and confounded by unobserved "
        "ability, so OLS is biased. Include these exogenous, pre-treatment "
        "controls in BOTH stages so the first stage is well specified: 'exper', "
        "'expersq', 'black', 'south', 'smsa', and the regional dummies "
        "'reg662'..'reg669', 'south66', 'smsa66' (some regional dummies are "
        "collinear and may be dropped, which is fine). Exclude 'id', 'age', "
        "'KWW', 'IQ', 'enroll', 'married', 'fatheduc', 'motheduc', 'weight', "
        "and the family-at-14 variables as ID, ability proxies, or "
        "post-treatment leakage. Expect a borderline first stage near F=10; "
        "report it honestly but keep the full control set."
    ),
    "bank-recovery-rdd": (
        "Each row is one defaulted debt the bank is working, keyed by 'id'. The "
        "bank followed a sharp rule: debts whose 'expected_recovery_amount' "
        "(the dollar amount the bank expected to recover) crossed $1000 were "
        "escalated to a more intensive, costlier recovery strategy. "
        "'recovery_strategy' records which level was actually applied, and it "
        "flips at the $1000 mark. 'actual_recovery_amount' is the dollars "
        "actually collected, my outcome. This is a regression discontinuity "
        "design: 'expected_recovery_amount' is the running variable, $1000 is "
        "the cutoff, and 'recovery_strategy' is the threshold-assigned "
        "treatment. I want the intent-to-treat jump in 'actual_recovery_amount' "
        "right at $1000, estimated locally just above versus just below. Note "
        "'recovery_strategy' has several named levels, but the only break I "
        "care about is the one the $1000 threshold creates, so treat it as "
        "above-versus-below, not a per-level comparison. Do not adjust for "
        "'age' or 'sex'; in an RDD the cutoff identifies the effect, and those "
        "are pre-treatment debtor covariates useful only as placebo balance "
        "checks. Exclude 'id'."
    ),
    "student-por-mechanism": (
        "Each row is a secondary-school student in a Portuguese-language "
        "course. This is a mediation question: I want to decompose the effect "
        "of weekly study time on the final grade into a direct path and an "
        "indirect path through the earlier-period grades. The treatment is "
        "'studytime' (ordinal 1-4). The outcome is 'G3', the final grade. The "
        "course is graded in three periods that are strictly ordered in time, "
        "studytime -> 'G1' (first period) -> 'G2' (second period) -> 'G3', so "
        "G1 and G2 are the mediators: studying earlier feeds into G1 and G2, "
        "which feed into G3. Please separate the indirect effect that runs "
        "through G1 and G2 from the direct effect of studytime on G3. The "
        "pre-treatment confounders to adjust for are 'failures' (past class "
        "failures), 'Medu' and 'Fedu' (parents' education), and 'schoolsup' "
        "(extra school support); these come before studytime and are not "
        "mediators. Do not treat G1 or G2 as confounders, and do not control "
        "them away, since they are the mediating mechanism itself."
    ),
    "heart-failure-survival": (
        "Each row is one heart-failure patient followed after a clinic visit. "
        "This is time-to-event data, so a survival (Cox proportional-hazards) "
        "model fits: 'time' is the follow-up duration in days and is the "
        "survival time, and 'DEATH_EVENT' is the event indicator, where 1 means "
        "the patient died during follow-up and 0 means they were censored "
        "(alive when last seen, not a permanent survivor). The exposure of "
        "interest is 'ejection_fraction', the percentage of blood the heart "
        "pumps per beat; I want its hazard ratio for death. Adjust for these "
        "pre-treatment baseline clinical covariates: 'age', 'serum_creatinine', "
        "'serum_sodium', 'creatinine_phosphokinase', 'platelets', 'diabetes', "
        "'anaemia', 'high_blood_pressure', 'smoking', and 'sex'. Do not treat "
        "'time' or 'DEATH_EVENT' as covariates, since they define the outcome. "
        "There are no ID, post-treatment, or mediator columns to exclude here."
    ),
    "website-visitors-its-step": (
        "This is a daily time series for one website with three columns. Date "
        "is the calendar day and defines the time axis; visits is the integer "
        "count of unique daily visitors and is the outcome; Day is the "
        "day-of-week name (Sunday through Saturday). On 2019-04-01 the site "
        "shipped a change, and I want its causal effect on daily visits, "
        "specifically the level shift in visits at that date. There is no "
        "comparison site or control group here, so treat this as an interrupted "
        "time series: model the pre-period level and trend, then estimate the "
        "jump at 2019-04-01. The series has strong weekly seasonality, with "
        "weekends lower than weekdays, so adjust for Day to separate the real "
        "step change from ordinary weekly swings. Also account for any "
        "pre-existing trend so a gradual drift is not mistaken for the "
        "intervention. There are no IDs, post-treatment variables, or mediators "
        "to exclude; Date, Day, and visits are the only columns."
    ),
}
