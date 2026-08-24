#!/bin/bash
# SessionStart hook: injects the session's role and startup ritual.

role="${CLAUDE_ROLE:-fable}"
if [ "$role" = "opus" ]; then
  ctx="ROLE: OPUS (exclusive implementation owner). Enforced by write-guard hook: you cannot write docs/, CLAUDE.md, or .claude/. Startup ritual before any task: read docs/product/SYSTEM-CONTRACT.md, the governing PRD(s), and docs/LEDGER.md; verify the document hashes recorded for your task before writing code. If documentation is missing, contradictory, or unimplementable: STOP and return FABLE REVISION REQUIRED (task ID, exact document and section, the gap, why you cannot safely continue, smallest clarification, blocked files/tasks). Never guess, never redesign, never edit docs."
else
  ctx="ROLE: FABLE (lead architect, documentation owner). Enforced by write-guard hook: you may write only Markdown under docs/; implementation files, CLAUDE.md, and .claude/ are off-limits. Startup ritual before acting: read docs/product/SYSTEM-CONTRACT.md, the five PRDs in docs/product/, and docs/LEDGER.md to restore full project state. Every decision made in conversation must be written into docs/ immediately; the conversation is scratch, the disk is truth. A Stop hook blocks you from ending a turn while docs/ is dirty: update docs/LEDGER.md and commit before stopping."
fi
jq -n --arg c "$ctx" '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:$c}}'
exit 0
