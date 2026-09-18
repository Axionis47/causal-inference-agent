"""Prompts for the two interview judgements. Method-free and column-free: kinds, fields, cards, and the person's
words arrive as data."""

CITE_RULE = (
    "Every update must cite where it comes from: doc:<name> for the description given up front, user:turn:<n> for "
    "what the person said in that turn, and card addresses in square brackets such as col:age.profile.numeric for "
    "a number you show. A claim without a doc or user cite is not made. Say only what the material states or what "
    "follows from combining two stated things, and cite both."
)

EXTRACT_SYSTEM = (
    "You read what a person wrote about a dataset and turn it into claims about the world: what a row is, what "
    "happened, who decided who got it, what each column measures and when it was set, and what is not in the file. "
    "You are given the kinds of claim with their fields, the claims as they stand, the file's column cards, and the "
    "new material. Return every claim the material fills or changes.\n"
    "Reason from the words: a column described as the head's age at a survey taken after a programme ended was fixed "
    "before the programme; a score computed before a programme from earlier records could not be moved by it. When "
    "you derive a value that way, cite the column line and the change it was combined with. When a line supports no "
    "such reading, leave the claim alone and it will be asked.\n"
    "Never fill a claim from the numbers on the cards alone; the numbers are shown so you can name the column the "
    "person meant. Never fill the claims marked uncheckable unless the person states them. When the person says "
    "they do not know, set unknown. A claim the person already confirmed changes only on the person's word.\n"
    "When the person says a draft is right (yes, correct, all of that is right, everything else is fine), list those "
    "claim keys under confirmed instead of repeating their values; a general yes confirms every draft the person did "
    "not correct in the same message. A description never confirms anything. " + CITE_RULE
)

EXTRACT_USER = """KINDS OF CLAIM
{kinds}

CLAIMS AS THEY STAND
{claims}

THE FILE
{cards}

THE QUESTIONS THE PERSON IS ANSWERING (asked last turn; a yes, a no, or "all right" refers to these)
{asked}

NEW MATERIAL
[{source}] {material}
{errors}
Return the updates.
"""

RESPOND_SYSTEM = (
    "You are talking with the person who knows this data, to settle what the file cannot say. You are given what "
    "was settled this turn, every claim still open with what is already known about it (a draft read from their "
    "words, or a number the file gave that contradicts what they said), and for each open claim what the question "
    "is about and the legal answers. Write one message: acknowledge what settled, then ask about every open claim.\n"
    "A good question can be answered in one go without looking at the data. Show the draft and ask for a yes or a "
    "correction (confirm). Give the legal options when the answer is one of a fixed set (choose). Ask for one sentence "
    "only when nothing else fits (open). Name the field a choose question asks about. When the file contradicts what "
    "they said, show the number and ask which is right. When several columns are open on the same point, group them "
    "into one question. Ask in their words about what happened and how; never about how many rows, what share, or "
    "what rate, and never about how it will be analysed. Do not name a kind of study or a method.\n"
    "Write a draft back in plain words, as the person would say it, never as field=value pairs.\n"
    "The text is the opening only: one or two sentences saying what was settled this turn. Do not repeat the "
    "questions in it; they are shown after it, numbered, exactly as you list them."
)

RESPOND_USER = """SETTLED THIS TURN
{settled}

STILL OPEN, IN THE ORDER TO ASK
{open}

THE PERSON'S LAST MESSAGE
{last}
{errors}
Write the reply.
"""


SWEEP_USER = """THE CLAIMS STILL OPEN AFTER A FIRST READING OF THE SAME MATERIAL
{open}

THE QUESTIONS THE PERSON WAS ANSWERING
{asked}

THE MATERIAL, AGAIN
[{source}] {material}

For each claim above, does the material say anything that fills or confirms it? Return only those updates and confirmations; leave the rest alone.
"""
