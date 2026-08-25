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
| T-006 | Handoff persistence + acceptance gate | shared (SC §3, §3.1, §8.1) | T-005 | single-session (D-008) | ACCEPTED | yes | commit `ec5db6d`; fail-closed, all codes collected; 11 dockerized tests; maps to EV-SYS-006 |
| T-007 | Intake core: contracts, registrations, archive, profiler | PRD-001 (§5.5–§8) | T-004 | single-session (D-008) | ACCEPTED | yes | commit (see checkpoint log); 6 registry rows appended; 58 tests; maps to EV-P1-003/EV-P1-004 |
| T-008 | Intake capture + catalog + coordinator + handoff | PRD-001 (§4, §9–§13) | T-005, T-006, T-007 | single-session (D-008) | ACCEPTED | yes | commit `e613eed`; spec + amendments D-027..D-038; 34 tests; maps to EV-P1-001/002/005; budget `warning` (intake 1180/1200, coordinator 329/350 — both inside limits, near them) |
| T-009 | Design substrate: shared task envelope, design contracts, registry rows, design schema | PRD-002 (SC §5.2, §6.1, §21; §6, §12, §16–§18) | T-002, T-004, T-005, T-008 | single-session + opus subagents (D-041) | ACCEPTED | yes | commit `2c98730`; spec + Amendment 1 (D-044); 134 new tests, suite 349 green; budget `warning` only on pre-existing intake dims (design 601/2500, shared 982/2500, tests 2678/8000, declarative 641/3000, 28 modules) |
| T-010 | Shared Vertex gateway + LangSmith tracing | shared (SC §10.2–§10.4) | T-009 | single-session + opus subagents (D-041) | ACCEPTED | yes | commit (see checkpoint log); 44 new tests + 2 env-gated live smokes; closes the D-020 revisit (flush gate live in ArtifactCommitter, D-045); maps to EV-SYS-003/EV-SYS-005; shared 1425/2500 |
| T-011 | Method-pack manifests, triage, DesignContextManifest, tool surface, entry gate | PRD-002 (§4, §5, §9, §13, §15) | T-009 | single-session + opus subagents (D-041) | IN_PROGRESS | yes | spec frozen `docs/tasks/T-011-packs-triage-manifest-tools.md`; ColumnTriageRecord type D-046 |
| T-012 | Validation walls, ask gate, diagnostics, delivery capacity | PRD-002 (§10, §11, §14, §16.3, §13.5) | T-011 | single-session + opus subagents (D-041) | DRAFT | — | |
| T-013 | Agents, prompts, LangGraph graph, interrupts, renderer, coordinator, outcome + handoff | PRD-002 (§8, §9, §12.3, §19–§23) | T-010, T-012 | single-session + opus subagents (D-041) | DRAFT | — | |
| T-014 | CLI seven commands, runtime composition, live Kaggle adapter | shared (SC §1.1) + PRD-002 §11.1 | T-013 | single-session + opus subagents (D-041) | DRAFT | — | closes D-034 |

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
- 2026-08-24 — D-021: The handoff receiver names its expected outcomes explicitly per seam; "wrong outcome" is seam knowledge, not manifest knowledge.
- 2026-08-24 — D-022: Handoff gate events are caller-built via an event factory so the D-016 eval-ID rule stays satisfiable.
- 2026-08-24 — D-023: Intake-internal registrations use terminal status `committed`; outcome statuses live only on IntakeOutcome. KaggleCapture is restricted, readable only by intake-coordinator.
- 2026-08-24 — D-024: kaggle_ref normalizes to owner/slug; owner/slug or kaggle.com/datasets URL accepted.
- 2026-08-24 — D-025: Profiler hypotheses limited to two bounded deterministic rules (identifier, ±9…±9999 sentinel at ≥1% and min/max), always labelled hypothesis.
- 2026-08-24 — D-026: Archive safety limits are constructor parameters (10k entries, 512 MiB file, 2 GiB total, 200:1 ratio above 1 MiB).
- 2026-08-24 — D-027: IntakeOutcome required parents reduced to [QuestionRecord]; the full chain moves to optional and the coordinator enforces it for usable/partial. An early refusal has no chain to cite; the registry must stay committable fail-closed.
- 2026-08-24 — D-028: Raw bytes (archive, extracted source files) are content-addressed objects without envelopes; catalog.resources rows point at them. Only canonical-JSON payloads get artifact envelopes.
- 2026-08-24 — D-029: PRD-001's `artifacts` table is realized by shared `causal.artifacts`; the `catalog` schema owns runs, datasets, resources, source_field_index, and exactly five narrow views. Run operational state stays in `causal.stage_runs`.
- 2026-08-24 — D-030: Provider field classification is the static registry `registries/kaggle-field-classes.v1.json`; unlisted fields default to `operational` (fail-closed away from model context).
- 2026-08-24 — D-031: Deterministic identities: analysis_id = an-sha16(idempotency_key); dataset_id = kaggle:owner/slug@version; artifact_id = kind:analysis_id:hash16. Payloads carry no wall-clock; envelope timestamps come from an injected clock.
- 2026-08-24 — D-032: v0 semantic-map rules are conservative: column meaning from provider description, missing_sentinel from profiler hypotheses, everything else not_offered. Free-text-to-slot interpretation belongs to PRD-002.
- 2026-08-24 — D-033: Outcome rule — refused per PRD-001 §12; partial when any resource is unreadable/excluded/failed or no semantic field is evidenced; usable otherwise.
- 2026-08-24 — D-034: The live kaggle==2.2.4 adapter is deferred to the CLI/runtime task; T-008 ships the client Protocol and capture layer; tests use frozen fixture clients. Credentials exist only at adapter construction from the runtime secret source.
- 2026-08-24 — D-035: Re-running an incomplete analysis uses a new stage_run_id; deterministic artifact IDs make recommits §8.2 replay no-ops (PRD-001 §13 restart-from-boundary).
- 2026-08-24 — D-036: resources.parse_status is PRD-001's five values (parsed, excluded, unreadable, unsafe, failed); withheld classifications map to excluded with a reason.
- 2026-08-24 — D-037: The producer never persists the handoff manifest (the T-006 gate records at receipt). The coordinator's open_handoff rebuilds it deterministically from analysis_id + intake_outcome_artifact_id and refuses for refused/missing/mismatched outcomes. Measured-fact index rows point at TableProfile artifacts. (T-008 Amendment 1.)
- 2026-08-24 — D-038: Shared docker fixtures live in tests/conftest.py so all test packages reuse them. (T-008 Amendment 2.)
- 2026-08-24 — D-039: PRD-002 wave plan = T-009 (substrate) → T-010 (gateway+tracing, shared) → T-011 (manifests/triage/manifest-compiler/tools/entry gate) → T-012 (validators/ask gate/diagnostics/capacity) → T-013 (agents/prompts/LangGraph/renderer/coordinator/outcome/handoff) → T-014 (CLI/runtime/live Kaggle). Specs freeze one task at a time, immediately before that task's implementation, so later specs absorb earlier findings. Declarative-first rule: method packs, requirement/tool/capacity registries, and prompt templates land in the declarative scope, keeping `design` code within its 2,500 lines.
- 2026-08-24 — D-040: AgentTaskEnvelopeV1, AgentTaskResultV1, ContextRequirementV1, and ClaimV1 live in the shared scope (`src/causal/shared/envelope.py`): SC §5.2/§5.4/§6.1/§16.2 define them across PRD-002–005.
- 2026-08-24 — D-041: Build mode for the PRD-002 wave (user-directed): Opus subagents implement the frozen specs; the main session architects, reviews, runs gates, and commits. The D-008 single-session discipline is retained at the spec/review/commit boundary.
- 2026-08-24 — D-042: The dev machine has no Graphviz binary; the Graphviz 15.1.1 pin binds the future OCI image. The renderer must record the actual local Graphviz version in every output and its tests must skip (not fake) when `dot` is absent. No silent substitution.
- 2026-08-24 — D-043: Session environment check: Kaggle credentials (`~/.kaggle/kaggle.json`) and Google ADC are present; no LangSmith key is visible. Tracing reads `LANGSMITH_API_KEY` from the environment at runtime and the preflight fails closed to `failed_observability` when it is missing, per SC §10.2.
- 2026-08-24 — D-045: LangSmith 0.11.0 has no per-batch ingest acknowledgement. The flush gate's verifiable contract is: client constructed with tracing_error_callback (fires once per exhausted ingest attempt) + post-flush tracing-queue-drain check; either signal raises ObservabilityError(flush_unacknowledged). Limits (sampled-out batches invisible; detection per flush, not per run) are recorded at the flush site. Span create/close failures map to flush_unacknowledged. SDK retries in google-genai default to 5 attempts; the gateway pins attempts=1 explicitly per SC §10.4.
- 2026-08-24 — D-044: T-009 Amendment 1. RoleName has 18 values (17 roles + unknown); an approved DesignOutcome requires all five refs (design, frame contract, graph view, capacity check, approval) non-None; the registry gains an eighteenth design row `DesignApprovalDecision` so the SC §11.1 verbatim CLI decision commit does not fail closed, and `DesignApproval` lists it as an optional parent. Registry now 26 rows.

## Checkpoint log

One line per checkpoint commit: date, commit subject, what state it freezes.

- 2026-08-24 — `4f7cbaf` T-001: environment (exact §14 pins, uv.lock), package skeleton, budget checker + 36 tests.
- 2026-08-24 — `bd6c985` T-002: canonical serialization + hashing + ArtifactEnvelopeV1/HandoffManifestV1 + 40 tests. Full suite 76 green.
- 2026-08-24 — Environment note: repo lives under ~/Documents (likely iCloud-synced); macOS set UF_HIDDEN on venv files, which makes Python 3.12 silently skip .pth files. Cleared with `chflags nohidden`; if imports break again after re-sync, re-run: `find .venv -flags +hidden -exec chflags nohidden {} +`
- 2026-08-24 — `7b96eeb` T-003: OperationalEventV1 + emitter, 21 tests.
- 2026-08-24 — `99d0132` T-004: artifact-type registry + seeded rows, 11 tests. Full suite 107 green; budget `within_budget` (shared 355, tests 769, declarative 35, 13 modules).
- 2026-08-24 — `c2fc732` T-005: persistence + commit protocol, 14 dockerized tests. Suite 121.
- 2026-08-24 — `ec5db6d` T-006: handoff store + gate, 11 dockerized tests. Suite 132.
- 2026-08-24 — `87629ec` T-007: intake core (contracts/archive/profiler + 6 registry rows), 58 tests. Suite 181 green; budget `within_budget` (shared 745, intake 344, tests 1570, declarative 166; 18 modules).
- 2026-08-24 — `faad0c8`/`a26c270` T-008 spec frozen + two pre-implementation amendments (D-027..D-038).
- 2026-08-24 — `e613eed` T-008: Kaggle capture Protocol, field-class registry, evidence/semantic-map builders, catalog migration + 5 views, coordinator with idempotency/refusals/handoff opening, shared build_event/build_envelope. 34 tests; suite 215 green; budget `warning` (intake 1180/1200, shared 792/2500, tests 2050/8000, declarative 291/3000, largest module 329/350; 25 modules). Intake budget is nearly exhausted by design — PRD-001 is complete; nothing further lands in the intake scope except fixes.
- 2026-08-24 — Environment note: the UF_HIDDEN/.pth issue recurred mid-session and was cleared again with `find .venv -flags +hidden -exec chflags nohidden {} +`. MinIO fixture now retries on XMinioServerNotInitialized, not just connection refusal.
- 2026-08-24 — `2c98730` T-009: shared AgentTaskEnvelopeV1/AgentTaskResultV1/ContextRequirementV1/ClaimV1; design contracts/semantics/frame models; 18 design registry rows (26 total); 0004_design.sql (8 tables). 134 new tests; suite 349 green.
- 2026-08-24 — `ff4e7f8` T-010: gateway.py (250) + tracing.py (187) + committer flush gate (+6); 44 tests; suite 390 green + 2 gated live smokes. Vertex/LangSmith APIs introspected from installed pins, none guessed.
