# Causal Final — working agreement (single-session model)

One Claude Code session both architects and implements this repository. The
earlier two-terminal Fable/Opus role split and its hook enforcement were
dissolved by the user on 2026-08-24; hooks are removed from
`.claude/settings.json` (the scripts remain in `.claude/hooks/` as inert
history).

## Source of truth

The conversation is scratch; the disk is truth. Governing documents:

- `docs/product/SYSTEM-CONTRACT.md` — the system contract (binding for the
  product: architecture, pins, complexity budgets, evaluation gates)
- `docs/product/PRD-001` … `PRD-005` — the five PRDs
- `docs/tasks/` — frozen per-task specs
- `docs/LEDGER.md` — task statuses, document hashes, decision log

Startup ritual for every session: read the system contract sections relevant
to current work, the current task specs, and `docs/LEDGER.md`.

## Working discipline (kept from the old model)

- Every architecture decision is written to `docs/` when made, and the ledger's
  decision log is append-only.
- Work lands as checkpoint commits: docs updated, ledger current, then commit.
- Implementation follows the frozen task specs in `docs/tasks/`; spec changes
  are made and committed before the code that depends on them.
- The SYSTEM-CONTRACT complexity budgets (§14.1) apply to all code written
  here; `tools/budget_check.py` measures them.
