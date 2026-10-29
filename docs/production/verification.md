# Component verification and release gates

The [ingestion test credibility audit](../audits/ingestion-test-credibility.md) records the original case-by-case findings and the repairs with deliberate defect checks. Current ingestion commands and isolation are documented in [containerization](containerization.md).

Status: Phase 1 ingestion and Phase 2 owned dataset browsing have isolated PostgreSQL/S3 tests, mutation controls, a real Playwright journey and a dedicated CI job. The broader matrix below remains planned tests for the [implementation sequence](implementation-plan.md). Test identifiers below name requirements, not existing tests or passing results. Requirements outside ingestion remain proposals; the ingestion checks do not establish authentication, profiling, review, chat or analysis readiness.

## Existing foundation

Current suites already cover [dataset API behavior](../../causal_agent/server/tests/test_datasets.py), [session journeys](../../causal_agent/server/tests/test_sessions.py), [profiling](../../causal_agent/profile/tests/test_profiler.py), [memory versions](../../causal_agent/memory/tests/test_versions.py), [desk transformations](../../causal_agent/desk/tests/test_transform.py), exploration gates and sandbox behavior, family engines, trace policy and generated contracts.

The web uses Vitest, React Testing Library and jsdom. [Rendered ingestion tests](../../web/src/pages/NewDataset.test.tsx) exercise form behavior. The real XHR adapter has separate tests, and Playwright exercises the built app against real HTTP/PostgreSQL/S3 in the disposable Compose smoke.

Keep useful existing tests and move them with their responsibilities. Replace assertions about removed routes/storage shapes in the implementation phase that removes those shapes. Do not retain tests solely to guarantee loading an obsolete layout.

## Test layers and placement

| Layer | Purpose | Proposed location and execution |
|---|---|---|
| Pure component rules | Owner policy, status transitions, checksum identities, diff/lineage and context applicability | Owning Python package `tests/`; pytest with temporary data |
| Real persistence integration | SQL constraints/transactions, leases, restart, publication, cleanup | Owning storage/dataset/runtime package `tests/integration/`; disposable PostgreSQL and private temp file root |
| API contracts | Principal enforcement, generated types, errors, request deduplication and response projection | `server/tests/`; real application operations with isolated persistence |
| React interaction | Upload/review/version acceptance, errors, keyboard behavior and private-state clearing | Feature-local `*.test.tsx`; Vitest plus React Testing Library/DOM environment added during implementation |
| Browser journey | Real frontend/API/database/file behavior across refresh and restart | Proposed `web/e2e/`; Playwright added during implementation |
| Real model acceptance | Actual desk, judgment, gate and workflow behavior | Existing eval/trace workflow and phase acceptance report; separately controlled model credentials/budget |
| Deployment exercises | Image, volume, login, sandbox and recovery behavior | Isolated deployment-check scripts/CI job introduced with phase 7 |

Test observable behavior rather than component internals, consistent with [Testing Library's principles](https://testing-library.com/docs/guiding-principles/). Browser cases should have isolated state and use user-visible controls, following [Playwright guidance](https://playwright.dev/docs/best-practices). Avoid snapshot tests for large pages and trivial tests that merely repeat an implementation.

Do not use an in-memory substitute to claim PostgreSQL transaction, constraint or locking behavior works. Do not use a mocked browser request to claim actual file persistence or ownership works. UI-only error-state tests can use typed response fixtures, but the integrated journeys use the real services.

## Component responsibility matrix

| ID / responsible component | Required assertions | Evidence layer |
|---|---|---|
| `IDENTITY` / identity + server | Missing/invalid identity rejected; owner derived server-side; two users cannot list/read/mutate/download/resume each other's resources; cross-owner payload references fail; production rejects development identity; logout/expiry clears access and UI state | Policy, API, database and browser |
| `STORAGE` / storage | Composite owner/dataset foreign keys; unique version/revision allocation under concurrency; rollback leaves no published partial record; checksum/file mismatch cannot publish; path/symlink confinement; database failure after file write is recoverable; disk-full failure is visible | Real PostgreSQL and filesystem integration |
| `INTAKE` / datasets | Streaming byte bound; malformed/empty/encoding/duplicate-header CSV handling; expired/consumed uploads; same filename across users; duplicate submission returns same operation; changed payload under same key conflicts; original checksum preserved | Unit, API, filesystem, browser |
| `PROFILE` / profile | Correct counts/types/missingness on known small inputs; no causal interpretations added; exact bytes/options/profiler determine profile; transformed version cannot reuse incorrect profile; cache isolation and eviction; parsing options preserve semantics | Existing profiler tests plus owned-record integration |
| `REVIEW` / datasets + review UI | Profile/sources correspond to selected version; stale hashes/revisions rejected; accept persisted across reload; review does not confirm all model claims or run analysis; keyboard/focus/loading/error behavior | API and rendered UI/browser |
| `CONVERSATION` / runtime + conversation UI | Separate histories/threads; monotonic message order; duplicate client message ID does not rerun a turn; two requests cannot mutate one conversation simultaneously; refreshing/polling/resuming uses the right owner and thread | API, real persistence and browser |
| `READ-WRITE` / repositories + API + frontend | Write through the real application command, restart, then read through the real API/inspector; fields, parents and citations agree; reads have no mutation side effects; pagination preserves order; same-owner wrong-chat/attempt references fail; context-aware cache/poll results cannot overwrite another chat; replacing a boundary removes both its old writer and readers | Real database/object integration, browser and targeted source-boundary review |
| `LEDGER` / lane + persistence | Separate attempt ledgers even when entry addresses match; atomic accepted change batches and revision checks; replay cannot duplicate history; rejected proposals cannot enter accepted history; completed results pin a ledger revision; chat journal links resolve to the correct attempt | Existing ledger rule tests and real transaction integration |
| `DESIGN-HISTORY` / families + results + persistence | Reasoning steps and candidate revisions resolve to their attempt, evidence and ledger revision; rejected alternatives remain inspectable; stopped analyses need no frozen design; frozen specification resolves to the decision that produced it; result inspector preserves these distinctions | Family adapter contracts, persistence integration and inspector tests |
| `CONTEXT-HISTORY` / desk + runtime + persistence | Every context selection retains exact version/profile/knowledge/source IDs; pointer belongs to its chat; desk decisions resolve to their input message and selection; stale admission fails; another chat's selection cannot be substituted even for the same owner | Contract and real transaction integration |
| `RESULT-PROVENANCE` / families + runtime + storage | Result resolves to the correct snapshot, candidate/frozen design, ledger revision and engine execution; worker attempts and engine fallbacks remain distinct; declined and failed outcomes do not fabricate estimates; citations resolve against their original immutable scope; artifact manifests cannot claim another attempt's files | Family contract, storage/API integration and result inspector tests |
| `TRACE-LIFECYCLE` / tracing + storage | Correct owner/chat/attempt/span links across turns and subprocesses; explicit payload/attachment policy; disabled or unavailable telemetry does not corrupt domain records; remote expiry is distinguished from missing application results; optional export records completeness; remote deletion retries independently from local cleanup | Trace-policy regression, integration and real LangSmith tree readback |
| `ARTIFACT-OWNERSHIP` / exploration + results + storage | Pre-analysis figures belong to the originating chat/attempt/version; analysis products also resolve to the request; another chat cannot acquire them through a dataset-wide listing; explicit reuse retains producer provenance | API and real persistence integration |
| `MEMORY` / memory | Source/status/confirmation rules retained; exact prior revision checked; simultaneous updates produce one accepted revision and one conflict; no lost updates; accepted version switch and journal commit together; unrelated conversations remain pinned | Pure rules and real transaction integration |
| `CONTEXT` / desk | Assembly excludes other datasets/users and unrelated chats; selected sources remain cited; human and deterministic facts remain distinguishable; hypothetical context cannot mutate durable memory; updates require appropriate re-review | Deterministic assembly tests, existing regression floor and real conversation |
| `LINEAGE` / datasets + validators | Stable IDs survive repeated filter/reorder/correction; duplicates in original remain distinct; rename vs derived variable identity; correct cell/column/row diff and null/type rules; immutable parents; branch selection; rejection of forged IDs/unsupported row creation; incomplete candidate never selected | Pure algorithms, real storage and browser |
| `EXPLORE` / explore | Input projection preserves existing restrictions; generated script cannot read other inputs/credentials or escape output area; failed/timed-out code recorded; validation failures cannot publish; execution manifest describes accepted code/output; declared and verified dependency claims distinguished | Existing gate/sandbox tests, target-platform container run and real model acceptance |
| `HANDOFF` / desk + runtime | Review/authorization matches snapshot/table/memory/family identity; stale or tampered input fails; repeated authorization request does not schedule duplicates; essential deficiency returns to clarification; a correction needs new authorization | Contract/integration plus real trace |
| `RUNTIME` / runtime | One active writer; expired lease cannot publish; duplicate job delivery publishes once; graceful shutdown and process death; recovery does not expand authorization; attempts/errors remain visible; deletion wins over late completion | Real processes/DB, failure injection and browser |
| `ADJUSTMENT` / adjustment family | Current frozen-design integrity, endpoint/engine rules and pre-effect boundaries remain; correct version/artifact namespace; supported sourced journey and honest diagnostics/limits | Existing real-engine tests plus real Gemini lane/integrated acceptance |
| `DID` / difference-in-differences family | Current panel/adoption/averaging policy, pre-treatment restrictions and estimator contracts remain; version identity retained; unsupported support/inference visible | Existing real-engine tests plus real Gemini acceptance |
| `RD` / discontinuity family | Score/cutoff/receipt/eligibility and window/inference contracts remain; correct input version; no hidden changes in policy during execution | Existing real-engine tests plus real Gemini acceptance |
| `RESULTS` / viz + results UI | Correct result/design/version displayed; figures resolve sources; numerical citations retain existing checks; safe rendering of user/generated text; tables/images download only through owned published files; failure/partial result is not shown as success | Existing figure/material tests, UI and real result readback |
| `FILES` / datasets + storage | Deletion hides dataset first; active work cannot republish; cleanup retry removes remaining records/files/checkpoints; another user's objects survive; referenced versions are not expired as orphans; artifact URLs cannot bypass authorization | API, real storage/process integration and browser |
| `WIRE` / server | Pydantic/OpenAPI/TypeScript agree; error codes/states consistent; UUID references owner-scoped; response schema excludes internal filesystem paths/secrets; schema generation has no model/job/production-storage side effects | Schema freshness, types and API tests |
| `BOUNDARY` / common + package owners | Import contracts, family registry and effect/reasoning separation retained; model/process/write/gate tracing entry points respected; no runtime writes into fixture directories; explicit scope reaches child processes | Existing architecture/trace checks and focused integration |
| `DEPLOY` / deployment + runtime | Non-root image starts with durable volumes; dev auth refused; readiness detects dependency failure; permitted sandbox runs on deployment OS; backup restore verifies manifests and resumes; observed load stays within chosen budgets | Built-image and restore exercises |
| `LIVE` / owner of changed desk/judgment/gate | Real Gemini exhibits intended behavior; trace contains prompt/record/gate/tool/process/file evidence; outcomes and limitations are read back and recorded | Real run, never a substituted scripted result |

Only add checks that protect an actual boundary or failure risk. Tests of labels, folder existence or private method call counts do not establish these requirements.

## Integrated journeys

Use small bounded fixtures for deterministic lifecycle tests and retained sourced datasets for real causal acceptance. Synthetic ownership/CSV cases are software checks, not evidence of causal validity.

| Journey | Scenario and pass condition |
|---|---|
| `J1` Owned ingestion | User A uploads CSV and reviews profile/sources; refresh and server restart preserve it; User B uploads same filename independently and cannot access A's dataset or profile |
| `J2` Conversations | Two conversations select one owned dataset; messages/checkpoints remain separate; one memory revision update cannot silently alter the other's authorized context; expired identity loses access |
| `J3` Lineage | V0 → correction V1 → filter V2; branch V3 from V0; inspect full changes and original IDs; reject stale candidate; select reviewed version in one conversation; original bytes and previous results remain unchanged |
| `J4` Analysis | Review exact context, authorize one request, observe progress/results and cited follow-up; interrupt a recoverable step and resume correctly; changed input/context cannot reuse the old permission; exercise each affected executable family |
| `J5` Deletion | Delete dataset with an active attempt, interrupt cleanup, restart and finish; old file URLs stop working; worker cannot recreate it; another user's dataset remains usable |
| `J6` Deployment | Start from empty persistent storage with real login configuration, complete intake, restart image, restore a backup to another empty installation and verify identity/data/version/checkpoint consistency |

Browser automation for deterministic intake/review uses real backend/storage with test identities supplied only by a test application factory. Never ship an impersonation endpoint or user-selection header. Existing scripted desk suites remain a regression floor. Model-driven `J2`–`J4` paths also need real acceptance; do not expand fakes as the proof of a changed desk/gate under the repository rule.

## Failure injection

Crash at meaningful boundaries: after staging bytes, before DB registration commit, after output files but before publication, after job claim, after external model return but before local commit, during version acceptance, and during deletion. Use subprocess termination or bounded integration hooks outside the production API. Each case asserts visible state, ownership, retry semantics and absence of duplicate published records.

Run race cases with two database connections/processes, not merely sequential calls: upload registration, memory revision update, conversation turn claim, job lease replacement and deletion versus publication. An external model call may have incurred cost before a crash; acceptance must not imply exactly-once external execution.

## Test isolation and evidence handling

- All tests write to their own database/schema and temporary roots. Read repository fixtures without modifying them. Hash-check fixture/source data before and after integration jobs where useful.
- Persistence tests explicitly use the real adapter; they must not accidentally inherit the existing `no_memory_writes` fixture and report a successful no-op save.
- Separate test users, file roots, browser contexts and job identities across workers. Tests cannot depend on execution order or an existing local `.env`.
- Credential-free suites use `LANGSMITH_TRACING=false` and deny external model calls. The controlled real-run acceptance environment retains tracing and approved credentials.
- Attach failed browser traces/screenshots, API errors, relevant manifests and DB assertions to CI evidence without exposing private source data or secrets. Live trace records follow the project's existing tracing policy and access controls.
- A skipped platform-dependent sandbox test is not a passed deployment test. Exercise the actual target runtime before release.

## Checks and CI gates

Existing commands, verified against the current Makefile:

```bash
LANGSMITH_TRACING=false make check
make build-web
```

`make check` runs Python/web lint and types, pytest, Vitest, schema generation and ladder generation. It does not run the web build; `make build-web` is separate. Generation commands also need freshness assertions in CI. The existing workflow checks generated OpenAPI, TypeScript and ladders against Git.

Add new jobs as their implementation lands:

| Gate | When | Required output |
|---|---|---|
| Existing static/unit/engine checks | Each implementation PR | Passing checks and generated files current |
| PostgreSQL/file integration | Every change to ownership, persistence, lifecycle or jobs | Required integration group passes with real DB; no optional silent skip |
| Rendered React tests | Changed feature PRs | User-visible behavior and error handling |
| Browser intake/review/ownership | Changes affecting these journeys | Real UI/API/storage journey passes |
| Process recovery/lineage/deletion | Changes affecting these boundaries | Failure/race invariants pass |
| Real model acceptance | Any changed desk/judgment/gate, and integrated release | Sourced case, observed behavior, trace readback, limits |
| Image/restore/load | Release candidate and affected deployment changes | Fresh install, actual sandbox, restart and measured restore/load results |

React/browser/database dependencies and the dedicated CI job are implemented for ingestion. Extending these checks to later milestones remains planned. Put Python integration suites inside `causal_agent/` so normal test discovery can include them; CI must provision their DB dependency and explicitly run the required group. Keep model runs separately controlled rather than embedding credentials or automatic paid calls in ordinary pytest.

For changes requiring real acceptance, follow [AGENTS.md](../../AGENTS.md): preserve the scripted suite with the smallest necessary adjustment, run Gemini through the changed behavior, read the trace and report what happened. A successful model response without valid evidence is not sufficient. No new scripted judgment/gate tests are a substitute for this bar.

## Release checklist

- Every phase's exit criteria and responsible test IDs have evidence, with no unresolved ownership, lineage, authorization or publication failures.
- Current UI/API/database/file structures agree; obsolete routes/shapes and duplicate mutable stores are gone; fixtures remain intact.
- All three families retain their contracts and affected workflows have real-run evidence; undeclared claims of endpoint coverage are absent.
- Real authentication works for hosted access; original owner checks also protect jobs, downloads, caches and checkpoints.
- CSV lineage, review decisions, context revisions and result citations remain connected after restart and restore.
- Deployment limits, expiry/deletion/backup policy and measured recovery expectations are recorded.
- Documentation and demo reflect shipped behavior; proposed features are not presented as complete.
