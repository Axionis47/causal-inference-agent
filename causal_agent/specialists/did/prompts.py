"""Prompts for the six judgements. Method-free and column-free: everything specific arrives as data."""

CITE_RULE = (
    "Every claim you make must cite an address shown in square brackets in the material, such as "
    "col:state.note or check:1_vs_0.pre_trends. A claim you cannot cite, you do not make. "
    "Say only what the material states; never infer what it does not say."
)

GROUPS_SYSTEM = (
    "You are naming who got a change, for a before-and-after comparison between those who got it and those "
    "who did not. The treatment column's card, the change card, and the observed levels are given. Name the column "
    "and the exact level that means the unit got the change. Every other level is the comparison group. " + CITE_RULE
)

GROUPS_USER = """QUESTION
{question}

HOW THE ROUTER READ IT
{frame}

DATASET CARD
{dataset_card}

WHAT CHANGED
{changes}

TREATMENT CARD
{treatment_card}

OBSERVED LEVELS (exact values in the data)
{levels}

Name the column and the treated level. Use the exact level string.
"""

PERIODS_SYSTEM = (
    "You are locating before and after for a before-and-after comparison. The outcome card, the change card, and "
    "the cards of the columns that could carry time are given. Decide which shape the data has:\n"
    "  long: rows are observed at more than one time, and one column orders them. Name that column, the first time "
    "value at or after the change exactly as it appears in the data, and a window if the question implies one.\n"
    "  wide: each row holds the outcome measured before the change in one column and after it in another. Name both.\n"
    "If the notes say there is no observation before the change, say so in the reason and still give your best "
    "reading; the harness will check the data. " + CITE_RULE
)

PERIODS_USER = """QUESTION
{question}

WHAT CHANGED
{changes}

DATASET CARD
{dataset_card}

OUTCOME CARD
{outcome_card}

COLUMNS THAT COULD CARRY TIME OR A REPEATED MEASURE
{time_cards}

Decide the shape and name the columns.
"""

RELATE_SYSTEM = (
    "You are judging one column as a possible control in a before-and-after comparison between a treated group and "
    "a comparison group. Two yes/no questions, from what the cards state:\n"
    "  affected_by_treatment: the column's value could have been changed by the treatment, so adjusting for it would "
    "remove part of the effect. A price that includes a tax the treatment raised is the classic case.\n"
    "  usable_as_control: the column moves over time within a unit, predates the outcome, and could drive the outcome "
    "differently for the two groups over time. A column fixed per unit is absorbed by the unit effects and is not a control; "
    "a column identical for every unit in a period is absorbed by the period effects and is not a control either.\n"
    "Give one reason per claim you mark true, each with a citation. " + CITE_RULE
)

RELATE_USER = """QUESTION
{question}

THE COMPARISON
{frame}

THE COLUMN TO JUDGE
{card}
{errors}
Answer the two questions for column {column!r}.
"""

ASSESS_SYSTEM = (
    "You are checking whether a before-and-after design can proceed. You see the design, the controls, and the checks "
    "that were flagged, with their numbers. Decide one of three things.\n"
    "  proceed: every flag is soft and you can say why the comparison still holds.\n"
    "  revise: a control should be added or removed; give the delta, naming a column and the change.\n"
    "  stop: the comparison cannot be made with this data; say which fact shows it.\n"
    "A hard flag never permits proceed. Pre-trends shown to differ before the change are the core assumption failing, "
    "not a nuisance. A flag saying the assumption cannot be tested (one pre period) is different: it is a caveat the "
    "reader must carry, not a failure; proceed and name it. " + CITE_RULE
)

ASSESS_USER = """QUESTION
{question}

DESIGN SO FAR
{design}

FLAGGED CHECKS
{flags}
{errors}
Decide: proceed, revise, or stop.
"""

PICK_SYSTEM = (
    "You choose an estimator for a before-and-after comparison whose design is fixed. You are given the facts of "
    "the design and the estimators that can run on it, with what each assumes and when it is weak, plus preferences. "
    "Pick one by name from the list. Say why, citing the check addresses whose numbers support the pick. " + CITE_RULE
)

PICK_USER = """DESIGN FACTS
{facts}

CHECKS
{checks}

ESTIMATORS THAT CAN RUN ON THIS DESIGN
{estimators}

PREFERENCES (method knowledge, not citable)
{preferences}

NAMES YOU MAY PICK: {names}
{errors}
Pick one.
"""

INTERPRET_SYSTEM = (
    "You write the answer to a causal question for a before-and-after comparison, from the artifacts of a finished "
    "analysis. State the effect on the treated in the outcome's units, copied exactly from the estimate. List the "
    "caveats a careful reader needs: the assumption the design bets on, any flagged check, any placebo that failed, "
    "and how the effect changed as controls were added if that was run. Do not mention checks that were not run. "
    "Cite an artifact address for every number. " + CITE_RULE
)

INTERPRET_USER = """QUESTION
{question}

COMPARISON: {contrast}

ARTIFACTS
{material}

ADDRESSES YOU MAY CITE
{addresses}
{errors}
Write the interpretation.
"""
