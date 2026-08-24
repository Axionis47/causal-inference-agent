# Project Ledger

Single source of truth for project state. Any session restores full context by
reading `docs/product/SYSTEM-CONTRACT.md`, the five PRDs, and this file.
Fable owns this file; Opus reads it and verifies hashes from it.

Statuses: `DRAFT` → `READY_FOR_OPUS` → `IN_PROGRESS` → `REVIEW` →
`ACCEPTED` | `REVISION_REQUIRED` | `FABLE_REVISION_REQUIRED`

## Governing document hashes

Recomputed and updated by Fable at every doc freeze
(`shasum -a 256 docs/product/*.md`).

| Document | SHA-256 | Frozen at (commit) |
|---|---|---|
| SYSTEM-CONTRACT.md | `5822cfc06a3f650cb79b94eb532436c94509ccc511119440cb5f342efd50ef8a` | initial freeze |
| PRD-001-kaggle-intake-storage.md | `a69337d3bfeac33cc7c017da357df369e01fe2c00d2b8ade429b84eac6913188` | initial freeze |
| PRD-002-causal-design-harness.md | `d6c0ddeec666bdfd09f635212463f22410ea2291155febd9cb5e40c14885ae94` | initial freeze |
| PRD-003-runnable-frame-preparation.md | `931b6bace64b7a9732cd27d70cdcbc0d82ba44a07c0f78cf8e32c5107c411416` | initial freeze |
| PRD-004-estimation-diagnostics-judgment.md | `79bb716eec3574508c1daea8686848ed24b74fef2e016e99467eda4c3fb12642` | initial freeze |
| PRD-005-evidence-visualization-presentation.md | `6f00087cd29cfd052d61b305a3f2fe22333c25cef22d8b2c669b243aad5f5f37` | initial freeze |

## Task board

| Task ID | Title | Owning PRD | Depends on | Model | Status | Doc hashes verified | Notes |
|---|---|---|---|---|---|---|---|
| _none yet_ | | | | | | | |

## Decision log

Append-only. One line per decision: date, decision, why.

- 2026-08-24 — Repository initialized with two-terminal Fable/Opus write-ownership enforcement (hooks in `.claude/`). Rationale: mechanical enforcement of the role contract instead of honor-system.

## Checkpoint log

One line per checkpoint commit: date, commit subject, what state it freezes.

- _none yet_
