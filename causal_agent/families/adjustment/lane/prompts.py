"""Prompts for the lane's judgements: the pair, the mechanism, the roles, the post-treatment roles, the assessment, the pick and
the interpretation. Method-free and column-free: everything specific arrives as data."""

from causal_agent.lane.prompts import INTERPRET_CASE, PICK_CASE, PLAIN_WORDS, cite_rule
from causal_agent.lane.prompts import INTERPRET_USER as INTERPRET_USER  # the lane's nodes read these here
from causal_agent.lane.prompts import PICK_USER as PICK_USER

CITE_RULE = cite_rule("col:lunch.note", "check:completed_vs_none.overlap")

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

TOOLS_NOTE = (
    "You may look at the data first with the tools offered: describe a column, see it by arm, the association or the redundancy "
    "of two columns, the overlap cells, the timing. A result comes back with an address you cite like any other. "
)

MECHANISM_SYSTEM = (
    "You are reading how a treatment came to be assigned, for a causal analysis that adjusts for the measured drivers of the "
    "treatment. The pack states the kind of assignment; you fill what it leaves open, from the story and the cards:\n"
    "  drivers: the columns the decision or the offer looked at. Only columns listed under THE COLUMNS; none if the story names none.\n"
    "  offer_column, uptake_column: when being offered the change and taking it are two different columns, name both; else null.\n"
    "  self_selection: whether units could move their own assignment, by choosing after an offer or by acting on the rule.\n"
    "Read the story first; the cards and the data facts second. " + TOOLS_NOTE + CITE_RULE
)

MECHANISM_USER = """QUESTION
{question}

THE CASE
{frame}

TREATMENT CARD
{treatment_card}

THE COLUMNS
{columns}
{errors}
Fill the mechanism. The kind is {kind!r}, as the pack states it.
"""

ROLES_SYSTEM = (
    "You are placing every column fixed before the treatment, all of them together, for a causal analysis that adjusts for "
    "measured drivers of the treatment. For each column listed give:\n"
    "  affects_treatment: the story or the mechanism says this column fed the decision that set the treatment.\n"
    "  affects_outcome: this column could move the outcome on its own. A characteristic of the unit that was fixed before the "
    "treatment (a background attribute, a prior condition, a group the unit belongs to) counts as yes unless the note rules it "
    "out; cite the note that says it was fixed before.\n"
    "  affected_by_treatment: yes only if the treatment could have changed its value; for a column fixed before, no.\n"
    "  is_outcome_measure: this column measures the same quantity as the outcome.\n"
    "  stands_for: what outside the file this column stands in for, when the story says so; else null.\n"
    "  redundant_with: another listed column that carries the same information; nested_in: a listed column this one is a finer "
    "version of. Say so only on a redundancy fact or a pack line that states it, and cite it under links.\n"
    "  modifier_candidate: the effect could plausibly differ across this column's values, per the story or the person's word.\n"
    "Read the columns against each other: two that measure one thing, or nest in each other, only show when read together. "
    "When a block SETTLED BY THE PACK gives a claim, the person has already said it: copy that answer and cite the address shown. "
    "When a block THE LAST READING gives one, an earlier run read it so and nobody has confirmed it: keep that answer unless the "
    "cards give a reason to depart, and then list the claim under departures, citing what does. "
    "Give one reason per claim you mark true, each with a citation. Return every listed column once. " + TOOLS_NOTE + CITE_RULE
)

ROLES_USER = """QUESTION
{question}

THE CASE
{frame}

TREATMENT CARD
{treatment_card}

OUTCOME CARD
{outcome_card}

THE COLUMNS TO PLACE ({count})
{columns}
{errors}
Place every column listed, together.
"""

POST_ROLES_SYSTEM = (
    "You are placing every column set at or after the treatment, all of them together, for a causal analysis that adjusts for "
    "measured drivers of the treatment. None of these is adjusted for; the question is what each one is:\n"
    "  mediator: the treatment changed it and it moves the outcome; the person's named mediator is copied as such.\n"
    "  outcome_measure: another measurement of the same quantity as the outcome.\n"
    "  consequence_of_treatment: the treatment changed it and it does not move the outcome.\n"
    "  consequence_of_outcome: the outcome moved it.\n"
    "  background: recorded late but fixed before the treatment in fact (a background attribute), so a cause of the outcome.\n"
    "  unrelated: none of these.\n"
    "Never reason from the outcome by arm; that tool is refused. "
    "When a block SETTLED BY THE PACK gives a claim, copy it and cite the address shown; when THE LAST READING gives one, keep it "
    "unless the cards give a reason to depart, and then list the claim under departures. "
    "Cite for every column. Return every listed column once. " + TOOLS_NOTE + CITE_RULE
)

POST_ROLES_USER = """QUESTION
{question}

THE CASE
{frame}

TREATMENT CARD
{treatment_card}

OUTCOME CARD
{outcome_card}

THE COLUMNS TO PLACE ({count})
{columns}
{errors}
Place every column listed, together.
"""


ASSESS_SYSTEM = (
    "You are checking whether a causal design can proceed. You see the design's graph, the adjustment set the "
    "identification step found, and the checks that were flagged, with their numbers. Decide one of three things.\n"
    "  proceed: every flag is soft and you can say why the comparison still holds.\n"
    "  revise: a flagged column should be related differently; give the delta, naming a flagged column and the change.\n"
    "  stop: the comparison cannot be made with this data; say which fact shows it.\n"
    "A hard flag never permits proceed. A revision must touch a column named in a flag. " + CITE_RULE
)

ASSESS_USER = """THE CASE
{frame}

QUESTION
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
    "If the ranked preference does not apply here, say why not. " + PICK_CASE + CITE_RULE
)

INTERPRET_SYSTEM = (
    "You write the answer to a causal question for one comparison, from the artifacts of a finished analysis. "
    "State the effect in the outcome's units, copied exactly from the estimate. List the caveats a careful reader "
    "needs: the assumption the design bets on, any flagged check, and any refuter that failed. When a sensitivity range "
    "is among the artifacts, say that the person believes a hidden factor exists, that no road around it was open, and that "
    "the effect holds only if that factor is no stronger than the simulated ones, quoting the range. A flag that comes from "
    "what the person said (belief.*, unknown.*, contradiction.*) is a caveat in their own terms. Do not mention "
    "checks that were not run. Cite an artifact address for every number, and every address you must cite. " + INTERPRET_CASE + PLAIN_WORDS + CITE_RULE
)
