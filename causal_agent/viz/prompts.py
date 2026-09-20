"""The one viz judgement: which declared figure makes the point. No figure names in the prompt; they arrive as data."""

from causal_agent.common.prompts import CITE_ADDRESSES_RULE as CITE_RULE

PICK_SYSTEM = f"""You are choosing one figure to make a point to a person who is deciding whether an analysis design fits their
data. You are given the point, what is known about the data, and the figures that can be made from it, each with what it
shows and when it makes a point. Choose the one that makes this point best, and say why in one sentence. If none of them
makes the point, choose none and say what would. {CITE_RULE}"""

PICK_USER = """THE POINT TO MAKE
{point}

WHAT IS KNOWN
{context}

FIGURES THAT CAN BE MADE (name, what it shows, when it makes a point)
{figures}

NAMES YOU MAY CHOOSE: {names}, none
{errors}"""
