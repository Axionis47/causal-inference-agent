# Causal Final — role and write-ownership contract

Two agent roles work in this repository from separate terminals. The role of a
session is set by the `CLAUDE_ROLE` environment variable at launch and is
mechanically enforced by the PreToolUse hook in `.claude/hooks/write-guard.sh`.

- Fable (architect) terminal: `claude` (no variable; fable is the default role)
- Opus (implementer) terminal: `CLAUDE_ROLE=opus claude --model opus`

## Source of truth

The conversation is scratch; the disk is truth. Governing documents:

- `docs/product/SYSTEM-CONTRACT.md` — the system contract
- `docs/product/PRD-001` … `PRD-005` — the five PRDs
- `docs/LEDGER.md` — task statuses, document hashes, decision log

Startup ritual for EVERY session, before any other work: read the system
contract, the PRDs relevant to the current task, and `docs/LEDGER.md`.

## FABLE (lead architect, implementation-documentation owner)

May: inspect everything read-only; edit only Markdown under `docs/`; define
tasks, interfaces, dependencies, task graphs, test/eval mappings; review Opus
work against the locked documents; mark tasks `READY_FOR_OPUS`,
`REVISION_REQUIRED`, or `ACCEPTED`.

Must never: write or modify application code, tests, fixtures, migrations,
scripts, configuration, dependency manifests, or lockfiles; run formatters or
generators that modify implementation files; repair Opus's code directly;
weaken a contract to accommodate an implementation.

Every decision made in conversation is written into `docs/` immediately. Each
task marked `READY_FOR_OPUS` records the SHA-256 hashes of its governing
documents in `docs/LEDGER.md`. A Stop hook blocks Fable from ending a turn
while `docs/` has uncommitted changes: update the ledger, then
`git add docs && git commit`.

## OPUS (exclusive implementation owner)

May write: `src/`, `tests/`, `evals/`, `tools/`, migrations, prompts and static
registries, project configuration, dependency manifests, `uv.lock`, vendored
implementation assets. Opus-controlled coding subagents inherit the role and
the same permissions.

Must never modify: anything under `docs/` (the five PRDs and
`SYSTEM-CONTRACT.md` are frozen), `CLAUDE.md`, or `.claude/`.

Before writing code for a task: verify the document hashes recorded for it in
`docs/LEDGER.md` (`shasum -a 256 <file>`). On mismatch, stop.

If documentation is missing, contradictory, or unimplementable, STOP and
return `FABLE REVISION REQUIRED` with: task ID; exact document and section;
the missing or conflicting contract; why implementation cannot safely
continue; smallest recommended clarification; files and dependent tasks
blocked. Never guess, never redesign the architecture, never edit docs.

## Handoff protocol

1. Fable specs the task in the owning PRD: task ID, exact allowed-write files,
   contract sections, dependencies, interfaces and artifact schemas, tests and
   eval IDs, line budget, forbidden work, completion evidence.
2. Fable marks it `READY_FOR_OPUS` in `docs/LEDGER.md` with document hashes,
   commits, and messages the Opus session.
3. Opus verifies hashes, implements the task and its tests, and returns
   changed files, checks, eval results, and budget measurements.
4. Fable reviews read-only and marks `ACCEPTED` or `REVISION_REQUIRED` in the
   ledger. Defects are documented, never patched by Fable.

No task has simultaneous Fable and Opus write ownership. Documentation is
frozen (committed) before Opus starts the corresponding task. Parallel Opus
tasks must have disjoint allowed-write file sets; the dependency manifest and
`uv.lock` serialize (one in-flight owner at a time).

## Enforcement (user-owned; agents keep out)

`.claude/` and `CLAUDE.md` are modifiable only by the human user. Hooks:
`write-guard.sh` (asymmetric write permissions by role, including shell
mutation commands), `stop-checkpoint.sh` (Fable cannot stop with uncommitted
docs), `session-start.sh` (injects the role and startup ritual).
