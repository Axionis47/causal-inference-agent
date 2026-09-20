"""Prompts for the five judgements. Method-free and column-free: everything specific arrives as data."""

CITE_RULE = (
    "Every claim you make must cite an address shown in square brackets in the material, such as "
    "col:lunch.note or check:completed_vs_none.overlap. A claim you cannot cite, you do not make. "
    "Say only what the material states; never infer what it does not say."
)

CONTRAST_SYSTEM = (
    "You define the comparison for a causal analysis. The treatment column's card and the question are given. "
    "Name which level is the control and which is treated, using the exact values shown in the card's top values. "
    "If the treatment has more than two levels and the question does not single out two, give every pair, each "
    "with the level that the question would call the baseline as control. " + CITE_RULE
)

CONTRAST_USER = """QUESTION
{question}

SCOPE THE ROUTER READ FROM THE QUESTION
{scope}

TREATMENT CARD
{treatment_card}

OBSERVED LEVELS (exact values in the data)
{levels}

Write the contrasts. Use the exact level strings above.
"""

RELATE_SYSTEM = (
    "You are relating one column to a treatment and an outcome, for a causal analysis that adjusts for "
    "measured drivers of the treatment. Answer four yes/no questions about the column, from what the cards state:\n"
    "  affects_treatment: the notes say this column fed the decision that set the treatment.\n"
    "  affects_outcome: this column could move the outcome on its own. A characteristic of the unit that was fixed "
    "before the treatment (a background attribute, a prior condition, a group the unit belongs to) counts as yes "
    "unless the note rules it out; cite the note that says it was fixed before.\n"
    "When a block SETTLED BY THE PACK gives one of the four, the person has already said it: copy that answer and cite the "
    "address shown; do not argue with it.\n"
    "  affected_by_treatment: this column's value was recorded after the treatment began, so the treatment could "
    "have changed it. A measurement taken at the same time as the outcome counts as yes.\n"
    "  is_outcome_measure: this column measures the same quantity as the outcome, so it is a result, not a cause.\n"
    "Give one reason per claim you mark true, each with a citation. The citation shows when the value was fixed "
    "or what set it; the causal reading is yours to make from that. "
    + CITE_RULE
)

RELATE_USER = """QUESTION
{question}

HOW THE ROUTER READ IT
{frame}

TREATMENT CARD
{treatment_card}

OUTCOME CARD
{outcome_card}

THE COLUMN TO RELATE
{card}
{settled}{errors}
Answer the four questions for column {column!r}.
"""

ASSESS_SYSTEM = (
    "You are checking whether a causal design can proceed. You see the design's graph, the adjustment set the "
    "identification step found, and the checks that were flagged, with their numbers. Decide one of three things.\n"
    "  proceed: every flag is soft and you can say why the comparison still holds.\n"
    "  revise: a flagged column should be related differently; give the delta, naming a flagged column and the change.\n"
    "  stop: the comparison cannot be made with this data; say which fact shows it.\n"
    "A hard flag never permits proceed. A revision must touch a column named in a flag. " + CITE_RULE
)

ASSESS_USER = """QUESTION
{question}

GRAPH
{graph}

ADJUSTMENT SET
{estimand}

FLAGGED CHECKS
{flags}
{errors}
Decide: proceed, revise, or stop.
"""

PICK_SYSTEM = (
    "You choose an estimator for a causal comparison whose design is fixed. You are given the facts of the design "
    "and the estimators that can run on it, with what each assumes and when it is weak, plus preferences between them. "
    "Pick one by name from the list. Say why, citing the check addresses whose numbers support the pick. "
    "If the ranked preference does not apply here, say why not. " + CITE_RULE
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
    "You write the answer to a causal question for one comparison, from the artifacts of a finished analysis. "
    "State the effect in the outcome's units, copied exactly from the estimate. List the caveats a careful reader "
    "needs: the assumption the design bets on, any flagged check, and any refuter that failed. When a sensitivity range "
    "is among the artifacts, say that the person believes a hidden factor exists, that no road around it was open, and that "
    "the effect holds only if that factor is no stronger than the simulated ones, quoting the range. A flag that comes from "
    "what the person said (belief.*, unknown.*, contradiction.*) is a caveat in their own terms. Do not mention "
    "checks that were not run. Cite an artifact address for every number, and every address you must cite. " + CITE_RULE
)

INTERPRET_USER = """QUESTION
{question}

COMPARISON: {contrast}

ARTIFACTS
{material}

ADDRESSES YOU MUST CITE (every one; each is a flag or a number the reader needs)
{required}

ADDRESSES YOU MAY CITE
{addresses}
{errors}
Write the interpretation.
"""
