"""Prompts for the five judgements. Method-free and column-free: everything specific arrives as data."""

CITE_RULE = (
    "Every claim you make must cite an address shown in square brackets in the material, such as "
    "col:score.note or check:above_vs_below.density. A claim you cannot cite, you do not make. "
    "Say only what the material states; never infer what it does not say."
)

SCORE_SYSTEM = (
    "You are naming the score and the cutoff that decided who got a change, for a comparison of units just either "
    "side of that cutoff. The question, the router's reading, the dataset card, the change cards, the treatment card "
    "(when the hand-off named one), and every column card are given. Return:\n"
    "  column: the column holding the score the rule was applied to, or null if the notes state no cutoff rule on a numeric score.\n"
    "  cutoff: the cutoff value in the score's own units, exactly as the notes give it.\n"
    "  treated_side: whether units above or below the cutoff got the change.\n"
    "  cutoff_value_treated: true when a unit whose score equals the cutoff got the change (an 'at or above' or 'at or below' "
    "rule), false when the rule is strict.\n"
    "  takeup_column and takeup_level: the column that records whether each unit actually received the change and the exact "
    "value that means it did; null when the data records no such column and the change is the cutoff rule itself.\n"
    "If the hand-off named a treatment column that is not the score, that column is the take-up column unless the notes say "
    "it is something else. " + CITE_RULE
)

SCORE_USER = """QUESTION
{question}

HOW THE ROUTER READ IT
{frame}

DATASET CARD
{dataset_card}

WHAT CHANGED
{changes}

TREATMENT CARD
{treatment_card}

EVERY COLUMN
{cards}
{errors}
Name the score column, the cutoff, the treated side, the cutoff-value rule, and the take-up column and level or null.
"""

RELATE_SYSTEM = (
    "You are judging one column's standing at a cutoff, for a comparison of units just either side of it. Three yes/no "
    "questions, from what the cards state:\n"
    "  predetermined: the column's value was fixed before the score was set and the change decided, so units just either side "
    "of the cutoff should not differ on it. A characteristic measured later but describing something fixed earlier "
    "(a person's schooling, a county's population before the programme) counts, when the card says so.\n"
    "  affected_by_treatment: the column's value could have been changed by the treatment, so adjusting for it would remove "
    "part of the effect.\n"
    "  is_outcome_measure: another measure of the outcome, or a later outcome, or a fitted or derived version of one.\n"
    "A code, label, or identifier that names a unit, a place, or a category (a county code, a state number, a campus label) "
    "is not a characteristic: mark all three false for it.\n"
    "Give one reason per claim you mark true, each with a citation. When a block SETTLED BY THE PACK gives a claim, the "
    "person has already said it: copy that answer and cite the address shown. " + CITE_RULE
)

RELATE_USER = """QUESTION
{question}

THE COMPARISON
{frame}

THE COLUMN TO JUDGE
{card}
{settled}{errors}
Answer the three questions for column {column!r}.
"""

ASSESS_SYSTEM = (
    "You are checking whether a cutoff design can proceed. You see the design, the checks that were flagged with their "
    "numbers, and the score and dataset cards. Decide one of two things.\n"
    "  proceed: every flag is soft and you can say, for each one, why the comparison still holds; cite every flag's address.\n"
    "  stop: the comparison cannot be made with this data; say which fact shows it.\n"
    "A hard flag never permits proceed. A density flag or a covariate-continuity flag is not a number you can wave away: "
    "the notes decide. To proceed over one you must cite the note that says how the score was set and whether units could "
    "move it, or that says the covariate was fixed before the change; if no note says so, stop. A flag that says a test "
    "was uninformative or not computable is a caveat to carry, not a failure; cite it and say why the data cannot test it. "
    "A flag that comes from what the person said (belief.*, unknown.*, contradiction.*) is theirs to answer; you may only carry it "
    "as a caveat, never clear it. " + CITE_RULE
)

ASSESS_USER = """QUESTION
{question}

DESIGN SO FAR
{design}

FLAGGED CHECKS
{flags}

SCORE CARD, DATASET CARD, AND THE CARDS OF THE COVARIATES TESTED AT THE CUTOFF
{cards}
{errors}
Decide: proceed or stop.
"""

PICK_SYSTEM = (
    "You choose an estimator for a cutoff design whose design is fixed. You are given the facts of the design and the "
    "estimators that can run on it, with what each assumes and when it is weak, plus preferences. Pick one by name from the "
    "list. Say why, citing the check addresses whose numbers support the pick. " + CITE_RULE
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
    "You write the answer to a causal question for a cutoff design, from the artifacts of a finished analysis. State, in "
    "the outcome's units and copied exactly from the estimate: the effect, its robust interval, the bandwidth, and the "
    "effective rows on each side. Name what the effect is: the effect for units at the cutoff (a sharp design), the effect "
    "for units at the cutoff who took up the change because they crossed it (a fuzzy design), or the effect of crossing the "
    "cutoff whatever was taken up. Say in one sentence that the effect is local to units at the cutoff, in the score's units, "
    "and does not speak to units far from it. List the caveats a careful reader needs: the assumption the design bets on, "
    "every flagged check, every falsification that failed, and how the estimate moved when covariates or the polynomial "
    "order changed if that was run. Do not mention checks that were not run. Cite an artifact address for every number. " + CITE_RULE
)

INTERPRET_USER = """QUESTION
{question}

COMPARISON: {contrast}

ARTIFACTS
{material}

ADDRESSES YOU MAY CITE
{addresses}

ADDRESSES YOU MUST CITE
{required}
{errors}
Write the interpretation.
"""
