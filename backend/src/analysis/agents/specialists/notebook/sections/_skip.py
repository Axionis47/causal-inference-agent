"""Helpers for rendering 'this section was skipped' placeholders.

When an upstream agent fails or is intentionally skipped, the section
that depends on its output should still appear in the notebook with a
clear explanation rather than crashing or rendering an empty header.
A complete-but-truthful notebook beats a half-rendered one.
"""

from nbformat.v4 import new_markdown_cell


def render_skipped_cell(
    section_name: str,
    *,
    reason: str,
    upstream_agent: str | None = None,
) -> list:
    """Build a single markdown cell saying the section was skipped.

    Args:
        section_name: Human-readable section title (e.g. "Exploratory Data Analysis").
        reason: One-sentence explanation of *why* the section was skipped.
        upstream_agent: Optional name of the agent whose output was missing.
    """
    parts = [f"## {section_name}\n"]
    parts.append("> **Section skipped.**\n>")
    parts.append(f"> {reason}\n>")
    if upstream_agent:
        parts.append(
            f"> Re-running the `{upstream_agent}` agent will populate this "
            f"section on the next iteration.\n"
        )
    return [new_markdown_cell("\n".join(parts))]
