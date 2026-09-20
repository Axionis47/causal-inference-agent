"""Prompt text every lane shares. A lane's own prompts stay in its package; what is here is worded once."""


def cite_rule(column_example: str, check_example: str) -> str:
    """The lane's cite rule, with two example addresses in the lane's own vocabulary."""
    return (
        "Every claim you make must cite an address shown in square brackets in the material, such as "
        f"{column_example} or {check_example}. A claim you cannot cite, you do not make. "
        "Say only what the material states; never infer what it does not say."
    )
