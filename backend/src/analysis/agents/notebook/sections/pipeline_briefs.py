"""Pipeline briefs section: deterministic transcription of state.agent_briefs.

Every sealed specialist writes a typed AgentBrief on each dispatch
(status, headline, flags, raised_issues). The flags are decisions —
"the estimator declared METHOD_UNSTABLE", "ps_diagnostics declared
POOR_OVERLAP" — and must appear in the notebook verbatim, not
paraphrased by the intro / conclusions LLM.

This renderer is pure transcription. The companion `decisions.py`
section transcribes `state.decisions` (agent-pushed choices); this
section transcribes `state.agent_briefs` (agent-emitted flags).
Different semantics, different sources, separate tables.
"""

from nbformat.v4 import new_markdown_cell

from src.analysis.agents.base import AnalysisState
from src.domain.briefs import AgentBrief, Flag

from ._skip import render_skipped_cell

EM_DASH = "—"


def render_pipeline_briefs(state: AnalysisState) -> list:
    """Render the per-agent brief audit trail."""
    briefs = state.agent_briefs
    if not briefs:
        return render_skipped_cell(
            "Pipeline Issues & Flags",
            reason=(
                "No agent wrote a brief for this run. Either the pipeline "
                "did not dispatch any sealed specialists, or all of them "
                "refused before writing their typed return."
            ),
            upstream_agent=None,
        )

    ordered = _sort_briefs(briefs)

    md = "## Pipeline Issues & Flags\n\n"
    md += (
        "Each sealed specialist writes a typed brief on every dispatch. "
        "Below is the raw, verbatim record: which agents ran, their "
        "status, the flags they raised from the closed enum, and the "
        "issue strings they emitted. No paraphrasing.\n\n"
    )
    md += "| Agent | Status | Headline | Flags | Issues |\n"
    md += "|-------|--------|----------|-------|--------|\n"

    for brief in ordered:
        md += _row(brief) + "\n"

    return [new_markdown_cell(md)]


def _sort_briefs(briefs: dict[str, AgentBrief]) -> list[AgentBrief]:
    """Refused first, then flagged-done, then clean-done.

    Preserves insertion order within each group so the reader sees
    issues at the top and clean stages at the bottom.
    """
    refused: list[AgentBrief] = []
    flagged: list[AgentBrief] = []
    clean: list[AgentBrief] = []

    for brief in briefs.values():
        if brief.status == "refused":
            refused.append(brief)
        elif brief.flags:
            flagged.append(brief)
        else:
            clean.append(brief)

    return refused + flagged + clean


def _row(brief: AgentBrief) -> str:
    """One markdown table row, with pipes escaped to keep the table valid."""
    flag_cell = (
        ", ".join(f.value for f in brief.flags)
        if brief.flags else EM_DASH
    )
    issues_cell = (
        "<br>".join(_escape(issue) for issue in brief.raised_issues)
        if brief.raised_issues else EM_DASH
    )
    return (
        f"| {_escape(brief.agent)} | {brief.status} | "
        f"{_escape(brief.headline)} | {flag_cell} | {issues_cell} |"
    )


def _escape(s: str) -> str:
    """Escape pipes for markdown table cells (matches decisions.py)."""
    return s.replace("|", "\\|")
