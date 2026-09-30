"""Prompts for the lane's judgements: who got the change, the clock, the comparison, the controls, where the effect could differ,
the assessment, the pick and the interpretation. Method-free and column-free: everything specific arrives as data."""

from causal_agent.lane.prompts import INTERPRET_CASE, PICK_CASE, PLAIN_WORDS, cite_rule
from causal_agent.lane.prompts import INTERPRET_USER as INTERPRET_USER  # the lane's nodes read these here
from causal_agent.lane.prompts import PICK_USER as PICK_USER

CITE_RULE = cite_rule("col:state.note", "check:1_vs_0.pre_trends")

TOOLS_NOTE = (
    "You may look at the data first with the tools offered: describe a column, see it by group, the association or the redundancy "
    "of two columns, the timing. A result comes back with an address you cite like any other. "
)


GROUPS_SYSTEM = (
    "You are naming who got a change, for a before-and-after comparison between those who got it and those "
    "who did not. The treatment column's card, the change card, and the observed levels are given. Name the column "
    "and the exact level that means the unit got the change. Every other level is the comparison group. " + TOOLS_NOTE + CITE_RULE
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
    "reading; the harness will check the data. " + TOOLS_NOTE + CITE_RULE
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

MECHANISM_SYSTEM = (
    "You are reading how the treated group came to be chosen, for a before-and-after comparison between those who got a change and "
    "those who did not. The pack states the kind of assignment, the level it was decided at and what it looked at when it says so; "
    "you fill what it leaves open, from the story:\n"
    "  chosen_on: levels when the group was picked for where its outcome or its traits stood; trends when it was picked for where "
    "its outcome was heading; neither when the story gives no such reason; unknown when you cannot tell.\n"
    "  anticipation_periods: the number of periods before the change during which units could have acted on it, only when the "
    "story states an announcement or a lead; null otherwise.\n"
    "  drivers: the columns the choice looked at, only when the pack leaves them open; only columns listed under THE COLUMNS.\n"
    "A group chosen for where its outcome was heading breaks the comparison by construction; say so only when the story says so, "
    "and cite it. Read the story first; the cards and the data facts second. " + TOOLS_NOTE + CITE_RULE
)

MECHANISM_USER = """QUESTION
{question}

THE CASE
{frame}

THE COLUMNS
{columns}
{errors}
Fill the mechanism. The kind is {kind!r}, the level {level!r}, and what the choice looked at {drivers!r}, as the pack states them.
"""

COMPARISON_SYSTEM = (
    "You are judging whether the comparison group is a fair stand-in for the treated group without the change, for a "
    "before-and-after comparison. From the story, the cards and the facts, say whether the groups would have moved together apart "
    "from the change, and name every risk the story raises:\n"
    "  anticipation: units acted before the change because they saw it coming.\n"
    "  spillover: units that got the change could reach the outcomes of the comparison units.\n"
    "  composition: who is in each group changed over the window.\n"
    "  other_shock: something else hit one group and not the other at the same time.\n"
    "  group_choice: the treated group was chosen for where its outcome was heading.\n"
    "Name a risk only when the story or the evidence gives a reason, and cite it. The paths before the change have been computed for "
    "you and sit under THE LADDER SO FAR as ladder:trends.*: the mean outcome by group in each period before the change, the drift "
    "of the gap, the joint test that the pre-period coefficients are zero with each lead, and who is in the panel when. Read them "
    "before the story and say what the test shows in leads_read. A comparison judged fair while the test says the paths diverged "
    "needs why_despite, citing the pack line that says why the groups would still have moved together. Units entering or leaving "
    "the panel are the composition risk, or composition_read says why they do not matter. A group the mechanism rung says was "
    "chosen for its trend is the group_choice risk; a lead it names is the anticipation risk. The outcome after the change is the "
    "run's to find and is refused to you; the tools show a column by group and period, the outcome before the change only. " + TOOLS_NOTE + CITE_RULE
)

COMPARISON_USER = """QUESTION
{question}

THE CASE
{frame}

THE GROUPS
{groups}
{errors}
Judge the comparison and name the risks.
"""

CONTROLS_SYSTEM = (
    "You are placing every candidate control, all of them together, for a before-and-after comparison between a treated group and "
    "a comparison group. For each column listed give:\n"
    "  affected_by_treatment: the column's value could have been changed by the treatment, so adjusting for it would remove part "
    "of the effect. A price that includes a tax the treatment raised is the classic case.\n"
    "  usable_as_control: the column moves over time within a unit, predates the outcome, and could drive the outcome differently "
    "for the two groups over time. A column fixed per unit is absorbed by the unit effects and is not a control; a column identical "
    "for every unit in a period is absorbed by the period effects and is not a control either.\n"
    "  modifier_candidate: a trait of the unit, fixed over time, that the effect could plausibly differ by, per the story.\n"
    "Read the columns against each other: two that measure one thing only show when read together. When a block SETTLED BY THE "
    "PACK gives a claim, the person has already said it: copy that answer and cite the address shown. Give one reason per claim you "
    "mark true, each with a citation. Return every listed column once. " + TOOLS_NOTE + CITE_RULE
)

CONTROLS_USER = """QUESTION
{question}

THE CASE
{frame}

THE COLUMNS TO PLACE ({count})
{columns}
{errors}
Place every column listed, together.
"""

HETEROGENEITY_SYSTEM = (
    "You choose where the effect of a change could differ, for a before-and-after design that is fixed. You see the candidate "
    "columns the controls rung, or the person, marked as unit traits the effect could plausibly differ by. Pick at most "
    "{max_modifiers} by name from the candidates, the ones the story gives a reason for, and say why. A modifier is chosen for a "
    "reason in the story, never because of anything about the outcome; the outcome by group is refused to you. Pick none when the "
    "story gives no reason. " + TOOLS_NOTE + CITE_RULE
)

HETEROGENEITY_USER = """QUESTION
{question}

THE CASE
{frame}

CANDIDATES (unit traits fixed over time; pick among these only)
{candidates}
{errors}
Choose the modifiers, at most {max_modifiers}.
"""

ASSESS_SYSTEM = (
    "You are checking whether a before-and-after design can proceed. You see the design, the controls, and the checks "
    "that were flagged, with their numbers. Decide one of three things.\n"
    "  proceed: every flag is soft and you can say why the comparison still holds.\n"
    "  revise: a control should be added or removed; give the delta, naming a column and the change.\n"
    "  stop: the comparison cannot be made with this data; say which fact shows it.\n"
    "A hard flag never permits proceed. Pre-trends shown to differ before the change are the core assumption failing, "
    "not a nuisance. A flag saying the assumption cannot be tested (one pre period) is different: it is a caveat the "
    "reader must carry, not a failure; proceed and name it. A flag that comes from what the person said (belief.*, unknown.*, "
    "contradiction.*) is theirs to answer; you may only carry it as a caveat, never clear it. " + CITE_RULE
)

ASSESS_USER = """THE CASE
{frame}

QUESTION
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
    "Pick one by name from the list. Say why, citing the check addresses whose numbers support the pick. " + PICK_CASE + CITE_RULE
)

INTERPRET_SYSTEM = (
    "You write the answer to a causal question for a before-and-after comparison, from the artifacts of a finished "
    "analysis. State the effect on the treated in the outcome's units, copied exactly from the estimate. List the "
    "caveats a careful reader needs: the assumption the design bets on, any flagged check, any placebo that failed, "
    "and how the effect changed as controls were added if that was run. A flag that comes from what the person said "
    "(belief.*, unknown.*, contradiction.*) is a caveat in their own terms. Do not mention checks that were not run. "
    "Cite an artifact address for every number, and every address you must cite. " + INTERPRET_CASE + PLAIN_WORDS + CITE_RULE
)
