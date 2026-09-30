"""Prompts for the lane's judgements: the score and the line, whether the line is clean, the covariates, where the effect could
differ, the assessment, the pick and the interpretation. Method-free and column-free: everything specific arrives as data."""

from causal_agent.lane.prompts import INTERPRET_CASE, PICK_CASE, PLAIN_WORDS, cite_rule
from causal_agent.lane.prompts import INTERPRET_USER as INTERPRET_USER  # the lane's nodes read these here
from causal_agent.lane.prompts import PICK_USER as PICK_USER

CITE_RULE = cite_rule("col:score.note", "check:above_vs_below.density")

TOOLS_NOTE = (
    "You may look at the data first with the tools offered: describe a column, see it by side, the association or the redundancy "
    "of two columns, the timing. A result comes back with an address you cite like any other. "
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
    "it is something else. " + TOOLS_NOTE + CITE_RULE
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

LINE_SYSTEM = (
    "You are judging whether the line on the score is clean, for a comparison of units just either side of it. From the story, "
    "the cards and the facts, say whether the score was set before the decision, whether a unit could move its own score, and "
    "whether anything else switches at the same line, and name every risk the story raises:\n"
    "  manipulation: units could move their own score once they knew the rule.\n"
    "  other_change_at_line: something else, a rule or a programme, switches at the same line.\n"
    "  score_set_after: the score was set or revised after the change was decided.\n"
    "  cutoff_known_in_advance: units knew the cutoff before their score was fixed.\n"
    "Name a risk only when the story or a fact gives a reason, and cite it. Never reason from the outcome by side; that tool is "
    "refused, and the test of the score's density at the line is run by code afterwards. " + TOOLS_NOTE + CITE_RULE
)

LINE_USER = """QUESTION
{question}

THE CASE
{frame}

THE LINE
{line}
{errors}
Judge the line and name the risks.
"""

COVARIATES_SYSTEM = (
    "You are placing every candidate covariate, all of them together, for a comparison of units just either side of a cutoff. "
    "For each column listed give:\n"
    "  predetermined: the column's value was fixed before the score was set and the change decided, so units just either side of "
    "the cutoff should not differ on it. A characteristic measured later but describing something fixed earlier (a person's "
    "schooling, a county's population before the programme) counts, when the card says so.\n"
    "  affected_by_treatment: the column's value could have been changed by the treatment, so adjusting for it would remove part "
    "of the effect.\n"
    "  is_outcome_measure: another measure of the outcome, or a later outcome, or a fitted or derived version of one.\n"
    "  modifier_candidate: a predetermined characteristic the effect at the cutoff could plausibly differ by, per the story.\n"
    "A code, label, or identifier that names a unit, a place, or a category is not a characteristic: mark the three claims false "
    "for it. Read the columns against each other: two that measure one thing only show when read together. When a block SETTLED "
    "BY THE PACK gives a claim, the person has already said it: copy that answer and cite the address shown. Give one reason per "
    "claim you mark true, each with a citation. Return every listed column once. " + TOOLS_NOTE + CITE_RULE
)

COVARIATES_USER = """QUESTION
{question}

THE CASE
{frame}

THE COLUMNS TO PLACE ({count})
{columns}
{errors}
Place every column listed, together.
"""

HETEROGENEITY_SYSTEM = (
    "You choose where the effect at the cutoff could differ, for a cutoff design that is fixed. You see the candidate columns "
    "the covariates rung, or the person, marked as predetermined characteristics the effect could plausibly differ by. Pick at "
    "most {max_modifiers} by name from the candidates, the ones the story gives a reason for, and say why. A modifier is chosen "
    "for a reason in the story, never because of anything about the outcome; the outcome by side is refused to you. Pick none "
    "when the story gives no reason. " + TOOLS_NOTE + CITE_RULE
)

HETEROGENEITY_USER = """QUESTION
{question}

THE CASE
{frame}

CANDIDATES (predetermined characteristics; pick among these only)
{candidates}
{errors}
Choose the modifiers, at most {max_modifiers}.
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

ASSESS_USER = """THE CASE
{frame}

QUESTION
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
    "list. Say why, citing the check addresses whose numbers support the pick. " + PICK_CASE + CITE_RULE
)

INTERPRET_SYSTEM = (
    "You write the answer to a causal question for a cutoff design, from the artifacts of a finished analysis. State, in "
    "the outcome's units and copied exactly from the estimate: the effect, its robust interval, the bandwidth, and the "
    "effective rows on each side. Name what the effect is: the effect for units at the cutoff (a sharp design), the effect "
    "for units at the cutoff who took up the change because they crossed it (a fuzzy design), or the effect of crossing the "
    "cutoff whatever was taken up. Say in one sentence that the effect is local to units at the cutoff, in the score's units, "
    "and does not speak to units far from it. List the caveats a careful reader needs: the assumption the design bets on, "
    "every flagged check, every falsification that failed, and how the estimate moved when covariates or the polynomial "
    "order changed if that was run. Do not mention checks that were not run. Cite an artifact address for every number. "
    + INTERPRET_CASE
    + PLAIN_WORDS
    + CITE_RULE
)
