"""Human and JSON rendering of exactly one `CliResultV1` (SYSTEM-CONTRACT §1.1).

There is no second result path: both renderings read the same committed result
document. Operational NDJSON goes to the log sink and never to stdout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from causal.shared.canonical import canonical_bytes

if TYPE_CHECKING:  # the result model lives in `main`; importing it here would cycle.
    from causal.cli.main import CliResultV1

__all__ = ["SUMMARIES", "render", "render_human", "render_json"]

# The allowlisted human summaries, one per permitted message key.
SUMMARIES: Final[dict[str, str]] = {
    "analysis.created": "Analysis created; intake ran to its boundary.",
    "status.read": "Committed status read.",
    "command.accepted": "Your answer was committed.",
    "interrupt.open": "The design run stopped and needs your input.",
    "stage.finished": "The design stage reached a terminal outcome.",
    "command.blocked": "The command was rejected; nothing was changed.",
    "command.failed": "The command failed.",
}
_ANSWER_COMMANDS: Final = ("select-table", "answer-context", "approve-design")


def render(result: CliResultV1, output_format: str) -> str:
    """One document for `--format json`, one beginner-readable block for `--format human`."""
    return render_json(result) if output_format == "json" else render_human(result)


def render_json(result: CliResultV1) -> str:
    """Exactly one canonical JSON `CliResultV1` document."""
    return canonical_bytes(result.model_dump(mode="json")).decode("utf-8")


def render_human(result: CliResultV1) -> str:
    """The same result as plain lines; identities only, never payload internals."""
    lines = [SUMMARIES[result.message_key], f"Status: {result.status}"]
    for label, value in (("Analysis", result.analysis_id), ("Stage run", result.stage_run_id)):
        if value is not None:
            lines.append(f"{label}: {value}")
    if result.interrupt is not None:
        held = result.interrupt
        lines.append(
            f"Open interrupt: {held.interrupt_kind or 'unknown'} {held.interrupt_id}"
            f" (hash {held.expected_interrupt_hash}, revision {held.expected_revision})")
    lines.extend(f"Artifact: {ref.artifact_id} {ref.content_hash}" for ref in result.artifacts)
    lines.extend(f"{key}: {value}" for key, value in sorted(result.message_args.items()))
    if result.error_code is not None:
        lines.append(f"Error: {result.error_code}")
    if result.blocker_event_id is not None:
        lines.append(f"Blocker event: {result.blocker_event_id}")
    if result.next_command_name is not None:
        lines.append(f"Next command: {_next_command(result)}")
    return "\n".join(lines)


def _next_command(result: CliResultV1) -> str:
    """The exact permitted next command, with the flags that bind it to this state."""
    parts = [f"causal {result.next_command_name}", result.analysis_id or ""]
    held = result.interrupt
    if held is not None and result.next_command_name in _ANSWER_COMMANDS:
        parts.append(
            f"--interrupt-id {held.interrupt_id}"
            f" --expected-interrupt-hash {held.expected_interrupt_hash}"
            f" --expected-revision {held.expected_revision}")
    if result.next_command_name == "run" and result.stage_run_id is not None:
        parts.append(f"--expected-stage-run {result.stage_run_id}")
    parts.append("--idempotency-key KEY")
    return " ".join(part for part in parts if part)
