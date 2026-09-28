"""Prompt text every lane shares. A lane's own prompts stay in its package; what is here is worded once."""


def cite_rule(column_example: str, check_example: str) -> str:
    """The lane's cite rule, with two example addresses in the lane's own vocabulary."""
    return (
        "Every claim you make must cite an address shown in square brackets in the material, such as "
        f"{column_example} or {check_example}. A claim you cannot cite, you do not make. "
        "Say only what the material states; never infer what it does not say."
    )


# The closing rule of every lane's interpretation: results in the person's own words.
PLAIN_WORDS = (
    "Write for the person who asked the question, in its own words. Say first what the answer means for the decision the "
    "question served, in the outcome's units. Name a check by what it asks, as the material says it, and give its technical "
    "name once in brackets; the address is the citation. A caveat is one sentence a careful reader can act on, never a list of "
    "names. "
)

# What the pick and the interpretation are told about the case block they now see.
PICK_CASE = "THE CASE is shown so the pick is made for this dataset and this change, not by habit; cites still come from the checks and the pack. "
INTERPRET_CASE = (
    "THE CASE is shown so the caveats are written in the dataset's terms, naming what the person said and what the data is; "
    "cites still come from the artifact addresses only. "
)

PICK_USER = """THE CASE
{frame}

DESIGN FACTS
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

INTERPRET_USER = """THE CASE
{frame}

QUESTION
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
