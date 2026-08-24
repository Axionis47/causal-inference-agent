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
| T-001 | Repo scaffolding, uv lock, budget checker | shared (SC §14, §14.1) | — | single-session (D-008) | ACCEPTED | yes | commit `4f7cbaf`; 154 pkgs resolved; 36 tests; budget `within_budget` |
| T-002 | Shared contract kernel (canonical, hashing, envelopes) | shared (SC §2, §3, §8.2) | T-001 | single-session (D-008) | ACCEPTED | yes | commit `bd6c985`; canonical.py adopted from causal-final-de draft with D-006 fix; 40 tests; maps to EV-SYS-001; budget `within_budget` (shared 158, tests 576) |
| T-003 | OperationalEventV1 + NDJSON emitter | shared (SC §10.1) | T-002 | single-session (D-008) | ACCEPTED | yes | commit `7b96eeb`; 28 event names; 21 tests; maps to EV-SYS-005 |
| T-004 | Artifact-type registry + seeded registrations | shared (SC §3.1) | T-002 | single-session (D-008) | ACCEPTED | yes | commit `99d0132`; registry fail-closed on unsupported_schema; 2 rows seeded (D-012); 11 tests; maps to EV-SYS-001/EV-SYS-006 |
| T-005 | Persistence + §8.2 commit protocol | shared (SC §3, §4, §8) | T-002, T-003, T-004 | single-session (D-008) | ACCEPTED | yes | commit `c2fc732`; 14 dockerized integration tests (postgres:18.6 + MinIO); trace-flush gate deferred (D-020); maps to EV-SYS-001/EV-SYS-004 |

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
- 2026-08-24 — D-008: User dissolved the two-terminal Fable/Opus model. Single session architects and implements; hooks removed from settings (scripts kept inert in .claude/hooks/); ledger/checkpoint discipline retained. T-001/T-002 reassigned to the main session. Budget-checker path exclusions fixed as: docs/**, .claude/**, CLAUDE.md, .gitignore, .python-version, pyproject.toml, uv.lock, README.md.
- 2026-08-24 — D-009: `registries/` is a top-level declarative-scope directory for static registries.
- 2026-08-24 — D-010: Static registries are JSON, not YAML (stdlib-parseable; the pinned stack has no YAML parser and none is added).
- 2026-08-24 — D-011: Canonical component IDs: intake-coordinator, design-harness, preparation-harness, estimation-harness, presentation-coordinator, cli, runtime.
- 2026-08-24 — D-012: artifact-types registry file is append-only; each stage task appends its rows once the owning PRD is read, so no parent-type name is invented ahead of its PRD.
- 2026-08-24 — D-013: `Stage` enum includes `system` for shared/CLI/runtime components acting outside one stage.
- 2026-08-24 — D-014: OperationalEventV1 `versions` keys are the closed set {model, prompt, tool, registry, schema, validator, compiler, renderer}.
- 2026-08-24 — D-015: `token_usage` keys are the closed set {input, output, thinking, total}.
- 2026-08-24 — D-016: The emitter (not the model) enforces name registration and requires non-empty `required_eval_ids` on task./agent./tool./handoff. events.
- 2026-08-24 — D-017: Dev/test object store is MinIO via Docker; src/ depends only on an S3 client Protocol (no boto3 import in production code). Postgres runs as the pinned postgres:18.6 image.
- 2026-08-24 — D-018: Legal run-state transition map fixed from the §4 diagram; completed/failed/failed_observability are terminal at the stage-run level.
- 2026-08-24 — D-019: payload_locator format is `objects/{content_hash}`; the committer rejects disagreement.
- 2026-08-24 — D-020: §8.2's LangSmith flush gate is deferred to the tracing task; commit currently ends at reopen-validation + artifact.committed event. Deliberate, recorded, must be revisited when LangSmith lands.

## Checkpoint log

One line per checkpoint commit: date, commit subject, what state it freezes.

- 2026-08-24 — `4f7cbaf` T-001: environment (exact §14 pins, uv.lock), package skeleton, budget checker + 36 tests.
- 2026-08-24 — `bd6c985` T-002: canonical serialization + hashing + ArtifactEnvelopeV1/HandoffManifestV1 + 40 tests. Full suite 76 green.
- 2026-08-24 — Environment note: repo lives under ~/Documents (likely iCloud-synced); macOS set UF_HIDDEN on venv files, which makes Python 3.12 silently skip .pth files. Cleared with `chflags nohidden`; if imports break again after re-sync, re-run: `find .venv -flags +hidden -exec chflags nohidden {} +`
- 2026-08-24 — `7b96eeb` T-003: OperationalEventV1 + emitter, 21 tests.
- 2026-08-24 — `99d0132` T-004: artifact-type registry + seeded rows, 11 tests. Full suite 107 green; budget `within_budget` (shared 355, tests 769, declarative 35, 13 modules).
