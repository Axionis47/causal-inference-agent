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
| T-001 | Repo scaffolding, uv lock, budget checker | shared (SC §14, §14.1) | — | opus | READY_FOR_OPUS | pending | spec `docs/tasks/T-001-repo-scaffolding.md` hash `1767ff9d46eb9ce69d2d672ec6c1246bd9685b12796598536901de70dc12ab70`; terminal causal-final-6a |
| T-002 | Shared contract kernel (canonical, hashing, envelopes) | shared (SC §2, §3, §8.2) | T-001 (env to run tests) | opus | READY_FOR_OPUS | pending | spec `docs/tasks/T-002-shared-contract-kernel.md` hash `8b553fbafd2e84520ed513018409f1d7de5936a646d2c867268a5739b04c9dec` (rev 3: D-006 test named in required list); terminal causal-final-de; maps to EV-SYS-001 |

## Decision log

Append-only. One line per decision: date, decision, why.

- 2026-08-24 — Repository initialized with two-terminal Fable/Opus write-ownership enforcement (hooks in `.claude/`). Rationale: mechanical enforcement of the role contract instead of honor-system.
- 2026-08-24 — D-001: Task specs live as one file per task under `docs/tasks/`, referenced from PRDs/contract by section; keeps frozen PRDs stable and hashes per-task.
- 2026-08-24 — D-002: Wave 1 = T-001 (scaffolding) ∥ T-002 (contract kernel); disjoint write sets, T-002 depends on T-001 only for the runnable environment.
- 2026-08-24 — D-003: The §14.1.1 budget checker lives at `tools/budget_check.py`, standard library only, assigned to the tests scope for counting (verification tooling, not production behavior). Open to revision if Opus shows a contradiction.
- 2026-08-24 — D-004: Canonical JSON = UTF-8, code-point-sorted keys, `(",", ":")` separators, NFC strings, no NaN/Inf, None preserved, RFC 3339 UTC timestamps with exactly 6 fractional digits and `Z`; hashes are bare 64-char lowercase hex.
- 2026-08-24 — D-005: Identity fields are non-empty strings ≤200 chars at the envelope layer; stricter per-identity formats are enforced at creation sites, not in shared contracts.
- 2026-08-24 — D-006: Canonicalization raises stable code `duplicate_key_after_normalization` when two mapping keys collide under NFC; silent collapse would break key-order determinism. (Raised by Opus during T-002 drafting.)
- 2026-08-24 — D-007: `payload_locator` is exempt from the 200-char identity cap; its own cap is 1024 chars. (Raised by Opus during T-002 drafting.)
- 2026-08-24 — Enforcement finding: write-guard hooks verified live in fresh sessions (headless probe denied correctly) but inert in sessions that predate the settings file or have not approved project hooks; every interactive session must run /hooks once (or restart and accept the hook prompt) and then verify behaviorally.

## Checkpoint log

One line per checkpoint commit: date, commit subject, what state it freezes.

- _none yet_
