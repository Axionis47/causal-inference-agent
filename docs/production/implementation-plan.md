# Implementation plan: private workspaces, review and traceable data changes

The pre-consolidation inventories in this design are historical. For implemented reviewed context, messages, receipts, checkpoints and artifact ownership, use the [current persistence chapter](../components/09-persistence.md). Remaining proposals below concern hosted workspaces and broader lifecycle work.


Status: planned, 22 October 2026. This is the execution sequence for the [production design](README.md), [data model](data-model.md) and [CSV lineage specification](lineage.md). None of the implementation steps below is complete merely because this document exists. The [verification plan](verification.md) assigns tests to responsible components.

The [current application container baseline](containerization.md) can be built before these phases. Reuse that single application boundary throughout implementation. Phase 7 completes hosted identity, sandbox readiness, operational recovery and release validation; it is not the first opportunity to package the app.

## Scope and defaults

Build one application with clearly owned modules. One person maps to one user identity; each person can own multiple datasets and conversations. There is no sharing or organization/team permission model in the first release. Authentication initially uses an explicit local development provider. All resource access is owner-scoped immediately; public or multi-person deployment requires real login integration.

Use PostgreSQL for application records and a private object-store adapter for durable production files. A filesystem adapter can support local development and tests against the same file contract. Preserve uploaded CSV bytes. Every modification produces a new version with lineage, a new applicable profile and review. Pin conversations and analyses to explicit data and knowledge revisions. Keep the built frontend with the API in one application image; keep generated scripts in separate restricted sandboxes. A database service and persistent storage are deployment dependencies, not new application microservices.

Bring-your-own-key, arbitrary document ingestion, cross-user sharing, joins/aggregation that create rows, vector search, Kubernetes and independently scaled analysis workers are outside this increment. Keep operator-configured Gemini access and existing causal family policies. Complete the ownership and durable-job contracts before considering worker scaling.

Unselected deployment decisions are the real login provider, hosting destination and numerical operating budgets. These do not block local identity, ingestion or review work, but they block a hosted-release claim. Record the selected values before phase 7: maximum upload bytes/rows/columns, per-user disk quota, queued/active job limits, profile/sandbox/model time budgets, expiry/retention, recovery-point and recovery-time targets. Do not insert arbitrary advertised capacity figures.

## Chat identity is the connection point

The user's term **chat ID** maps to one canonical field, `conversation_id`, throughout the database, HTTP contracts, job scope and trace metadata. Do not introduce parallel `chat_id`, dataset-name session IDs and conversation IDs for the same application concept. The existing graph `thread_id` remains a distinct execution/checkpoint identifier with one registered conversation owner.

The chat ID retrieves a connected workspace; it is not the primary key of every child record. Messages, requests, attempts, designs and artifacts have their own identities so that several of each can coexist under one chat. The [data model](data-model.md) remains the authoritative record catalogue.

```text
User
  Dataset → Versions → Profiles / row and column lineage
  Conversation (one dataset; one stable conversation_id)
    Context selections → exact version + profile + knowledge revision + sources
    Messages / desk decisions / journal events / checkpoint ownership
    Exploration and transformation requests → jobs and attempts → products
    Analysis requests → authorized snapshots → jobs → attempts
      reasoning steps / candidate revisions / evidence ledger changes
      frozen designs → engine executions → published results
      artifacts → file manifest → registered object keys
      trace links → LangSmith spans
```

Dataset records can be shared by several chats belonging to the same person. They do not receive a fabricated conversation owner. A chat-generated product records its originating conversation, while `ArtifactUse` represents deliberate reuse. A reusable source or confirmed knowledge revision retains the originating message where applicable. Nothing is assigned to a chat merely because its files are near one another on disk.

### Keys and constraints before write code

| Scope | Required relationships and enforcement |
|---|---|
| User-owned dataset | Dataset references its owner; version/profile/source references preserve both owner and dataset |
| Conversation | References its owner and dataset; current context pointer must belong to this conversation |
| Direct chat children | Message, event and context selection reference `(owner_id, conversation_id)`; child IDs/sequences are independently unique |
| Analysis request | References its conversation and exact snapshot from the same conversation |
| Job/attempt | Job has one typed cause; attempt belongs to that job; its owner, conversation, dataset and request scope are fixed from the stored cause |
| Design/reasoning children | Reference the producing attempt and conversation; a candidate, ledger, frozen design, execution or result from another attempt cannot be substituted |
| Generated artifact | References originating conversation, attempt and exact data version; optional result/design/execution references must agree with that scope |
| File and trace | File metadata and trace links preserve the owned producing scope; storage keys or provider UUIDs alone grant no access |

For example, define a unique key on `(owner_id, conversation_id, id)` for an analysis request. A child referencing that request must use the matching composite foreign key. Where a descendant carries `conversation_id` for querying, enforce it against its parent rather than trusting the repeated value. Same-user, wrong-chat references must fail just as cross-user references do. Use attempt-qualified composite keys for candidate/design/ledger relations. Payload-internal citations additionally use the existing semantic resolvers against a fixed source scope.

Create an immutable execution-scope value after authorization: owner, dataset, conversation, context selection, version, knowledge revision, request/job/attempt identifiers as applicable. Derive it on the server from owned records. Pass it through worker calls and subprocess manifests. Do not accept an arbitrary client-supplied owner or discover scope from global configuration, a directory name or the latest run. Keep this application scope separate from the scientific handoff so adding ownership does not silently change causal contracts.

## Read and write paths are one deliverable

Each slice must replace both the mutation path and its corresponding reads. A new writer is not complete while the page, desk or worker still reconstructs its data from the old directories. Repository interfaces expose explicit commands and queries; a second copy of the business rules is not created for the database adapter.

| Capability | Write path | Read path | Integrity check |
|---|---|---|---|
| Upload/version/profile | Authenticated intake → allocate IDs and pending file records → write/verify bytes → publish version/profile metadata | Dataset library or worker → owned dataset/version/profile query → registered file → object adapter | Immutable bytes, matching version/hash/profile and owner; retries do not duplicate registration |
| Create chat | Owned dataset review → Conversation + initial context selection + checkpoint ownership | Chat list/detail by principal and conversation ID; dataset page queries chats linked to its dataset | Two chats on one dataset retain separate IDs, selections and checkpoints |
| Submit/answer a message | Principal + conversation ID + client submission ID → Message + desk-turn Job; worker writes its response under the same scope | Ordered, paginated message query for that owned conversation | Unique submission ID, stable sequence, no repeat model work on HTTP retry |
| Confirm/correct context | Memory gate → new sourced revision + context selection + journal event in one transaction | Desk assembles exact selected revision/sources; inspector reads those same records | Stale revision fails; other chats and completed snapshots do not silently change |
| Authorize an analysis | Admission + human action → immutable snapshot + request + job + journal | Worker loads its request/snapshot by stored job scope; UI lists requests for the chat | Every input belongs to the authorized scope; changed context requires fresh authorization |
| Reason and freeze | Family gates → steps, candidate revisions, ledger changes and sealed design through shared persistence boundary | Design inspector reads the attempt's history; engine reads the named frozen design | Same-attempt references and ordered revisions; rejects remain distinct from accepted changes |
| Run and publish results | Engine execution → output validation → private objects → result/artifact manifests + job completion | Result inspector and follow-up desk read the immutable result/version; files resolve through authorized manifests | Publication lease, exact inputs/specification and checksum checks; no folder discovery |
| Explore/transform | Chat-scoped request → restricted input projection → attempt → artifact or candidate version/change set → review before selection | Gallery reads originating artifacts plus explicit uses; lineage view follows version parents and transformation records | Pre-analysis artifacts need no fake analysis ID; version changes do not overwrite the original |
| Journal/progress/resume | Domain operations append events with mutations; runtime persists jobs/checkpoints | Paginated activity; job status; owned checkpoint for resume | Read endpoints do not trigger mutations or start new analyses |
| Tracing | Scoped instrumentation → LangSmith plus recorded trace links | Authorized operator query follows exact project/trace/span references | No latest-trace heuristic for a chat; telemetry loss does not change scientific outcomes |
| Deletion | Hide owned target, fence writers, enqueue local/object/remote cleanup | Status query reports pending/failed/complete cleanup; deleted targets cannot be served | Late worker cannot resurrect data; shared live references are respected |

### Proposed HTTP read and write boundary

Use `conversation_id` as the route parameter's canonical meaning. The route set below is a contract target, not implemented endpoints. Generate frontend types from server models; never hand-maintain another response schema.

| Operation | Mutation | Corresponding read |
|---|---|---|
| Create/list chats | `POST /api/conversations` with owned dataset/version/review references | `GET /api/conversations`, optionally filtered by dataset; `GET /api/conversations/{id}` |
| Messages | `POST /api/conversations/{id}/messages` with client submission ID | `GET /api/conversations/{id}/messages?cursor=...` |
| Context selection | Version selection and confirmation commands reference an expected current context/revision | `GET /api/conversations/{id}/context`; explicit revision history |
| Analysis | `POST /api/conversations/{id}/analyses` authorizes an exact snapshot | `GET /api/conversations/{id}/analyses`; `GET /api/analyses/{id}` |
| Reasoning/results | Internal scoped publication by the producing attempt | `GET /api/analysis-attempts/{id}/design-history`; `/result` |
| Artifacts | Internal scoped publication; explicit reuse command | `GET /api/conversations/{id}/artifacts`; `GET /api/files/{id}` after ownership checks |
| Activity/recovery | Domain operations append activity; `POST /api/conversations/{id}/resume` | `GET /api/conversations/{id}/events?cursor=...`; `GET /api/jobs/{id}` |

The conversation overview returns references and bounded summaries. It does not embed every message, file, trace and ledger revision into one ever-growing JSON object. Independent inspectors paginate their own child collections. API authorization applies even when a child endpoint uses only its child ID. Frontend caches include conversation/resource identity and clear on principal changes; polling one chat must never replace another chat's view.

## Consolidation and retirement map

One operation owns each business decision, one repository path persists it, and one API projection presents it. Shared lower-level code is extracted when multiple concrete consumers need the same behavior. Family-specific statistical judgments remain in their families.

| Current implementation | Reuse | Replace and remove from live runtime |
|---|---|---|
| [server/datasets.py](../../causal_agent/server/datasets.py) | Intake validation, deterministic profile projection, lifecycle intent | Global YAML registry and name-based file discovery with dataset/version repositories and file IDs |
| [server/sessions.py](../../causal_agent/server/conversations/manager.py), [transcript.py](../../causal_agent/server/conversations/commands.py) | Desk invocation/resume semantics and response projection intent | Dataset-name session dictionary, shared transcript JSONL and mutable metadata as authoritative history |
| [memory/records.py](../../causal_agent/memory/records.py), [ops.py](../../causal_agent/memory/ops.py), [store.py](../../causal_agent/memory/store.py) | Field contracts, confirmation/contradiction rules and deterministic checks | Mutable multi-file runtime writes with transactional revisions; fixture loading remains explicitly separate |
| [memory/journal.py](../../causal_agent/memory/journal.py) | Event meaning, actor and evidence references | Directory-allocated conversation IDs and JSONL persistence with ordered owned events |
| [memory/artifacts.py](../../causal_agent/memory/artifacts.py), [versions.py](../../causal_agent/memory/versions.py) | Validated artifact/version payloads where accurate | Dataset-wide discovery and folder-derived identity with owned records, lineage and registered manifests |
| [desk/pipeline.py](../../causal_agent/desk/pipeline.py), [lane/records.py](../../causal_agent/lane/records.py) | Family entry, scientific records and execution boundaries | Absolute paths as durable identity; wrap existing family outputs through one publication adapter, not three new persistence implementations |
| [server/views.py](../../causal_agent/server/views.py), [artifacts.py](../../causal_agent/server/artifacts.py) | Useful presentation/projection logic | Reads reconstructed by scanning run folders with owned relational queries and file resolution |
| [web/src/api.ts](../../web/src/api.ts), [store.ts](../../web/src/store.ts), [App.tsx](../../web/src/App.tsx) | Existing UI components and generated wire types where applicable | Name-keyed routes/cache and old session clients in the same slice as their replacement endpoints |
| [common/trace.py](../../causal_agent/common/trace.py) | Span, gate, process and artifact chokepoints | Incomplete correlation with scoped trace links and explicit export/deletion policy; no second tracing wrapper |

No dual-write period, fallback old-layout reader, duplicate route alias or permanent parallel transcript is part of this plan. Within a slice, switch its producer and all consumers together and remove superseded code/tests. Immutable result exports and recomputable read projections are allowed only when labeled and derived from the authoritative record; they are not second writable stores. Do not delete fixture datasets, current user originals or unrelated working-tree changes as cleanup.

Keep one authoritative document per purpose: [data-model.md](data-model.md) for records, this plan for execution/retirement, [verification.md](verification.md) for tests, [lineage.md](lineage.md) and [tracing.md](tracing.md) for their specialized contracts. Current-code component chapters change only after implementation changes. Amend these files rather than creating another competing master plan.

## Blockers and reporting

Report a blocker to the user when confirmed, with evidence, affected slice, independent work that can continue, and the smallest decision needed. Distinguish a failed prerequisite from ordinary work not yet implemented. Do not weaken isolation, bypass ownership or introduce a compatibility path to conceal a blocker.

| Item | Evidence/status | Impact and next action |
|---|---|---|
| Docker generated-code isolation | **Confirmed failure** in the recorded container smoke test: `bwrap: Can't mount proc on /newroot/proc: Operation not permitted` | Blocks containerized script-dependent exploration, transformations and diagnostics. Qualify a supported runner/host integration and read its real test output; report any required host privilege change before applying it. Relational schema/API work can proceed. |
| Container release completeness | Page/API/storage smoke passed, but no full containerized Gemini journey or mid-analysis recovery evidence | Release gate remains open. Test the complete supported journey after identity/storage/runner integration; do not count an image build as functional completion. |
| Database/object adapters and checkpoint choice | Planned, not implemented; no confirmed product failure | Phase 0/1 work: select and lock supported adapters, prove transactions/file publication and the checkpointer on the real database. Escalate only a discovered incompatibility or missing required access. |
| Hosted identity, cloud credentials and operating limits | Deployment configuration decisions remain unselected | Block hosted acceptance, not offline schema work. Request missing values only when they are needed for a concrete configured release. |
| Existing dirty working tree | Many pre-existing modifications, additions and deletions | Preserve them and record the implementation baseline. If a required edit conflicts with unrelated work, report the exact conflict rather than silently reverting it. |

For each phase below, track a single evidence entry: status, authoritative schema change, writer changed, readers changed, old path removed, focused test results, live trace when required, and any blocker. This is the work ledger; do not create overlapping implementation efforts for the same boundary.

## Working method for every phase

Each phase is a vertical slice: a user-visible capability with its frontend, API, backend, data, storage and verification delivered together.

1. Define the user action, expected result and failures. Name the component responsible for deciding each transition.
2. Define owned records, immutable identities and transaction boundaries. Add schemas for requests, responses and errors before consumers implement them.
3. Regenerate OpenAPI/TypeScript, implement backend operations and then connect the feature UI to the same contract. The two implementations can proceed independently once that contract is fixed.
4. Run focused component and integration checks; inspect the actual browser behavior. Run required real-model acceptance for any changed desk, judgment or gate behavior.
5. Complete the phase's exit criteria, update the source ownership map and remove superseded code/tests in the same change. Do not preserve duplicate APIs or old-layout loaders.

Move files when their responsibility is established. Avoid a repository-wide mechanical rename preceding the ownership changes. Preserve uncommitted work and fixture data. Each phase's review describes both what changed and which evidence establishes it.

## Responsible components

| Component | Owns | Must delegate |
|---|---|---|
| `identity` | Verified principal, authentication adapter, ownership policy | Data lookup to owner-scoped repositories; causal run permission to desk |
| `datasets` | Upload/registration/version/review lifecycle and deletion | Statistics to profile, semantic confirmation to memory/desk |
| `profile` | Parsing options, deterministic facts, profile identity and cache | Human meaning and causal assumptions to desk |
| `memory` | Revisioned sourced knowledge, confirmation and contradiction rules | Persistence mechanics to storage |
| `desk` | Relevant context, questions, admission and run authorization | Execution/recovery to runtime; methods to families |
| `runtime` | Conversation steps, job ownership, attempts, checkpoints and recovery | Causal decisions to desk/families |
| `storage` | Transactions, file publication, checksums and physical cleanup | Application permission and semantic rules to their owners |
| `lane` and each family | Existing workflow gates, frozen designs, estimates and diagnostics | User identity/session management to application layer |
| `explore` | Restricted script execution and transformation candidates | Version publication/review to datasets; claims to memory |
| `viz` | Shared figure specifications and validation | Scientific figure choices to family; display to frontend |
| `server` | Authentication dependency, HTTP schemas, route dispatch, response projection | Business decisions to the modules above |
| Frontend features | User actions, progress, review, results and failure presentation | Readiness, authorization and durable state to backend |

The target folder structures remain in the [organization plan](README.md#frontend-organization). Component tests live with their owning package/feature. Whole-user-journey tests cross those boundaries without transferring ownership to a generic utility module.

## Phase 0: establish the baseline and contracts

**Deliverable:** an implementation baseline and explicit interfaces before changing persistence.

- Inventory the actual dirty working tree and distinguish existing changes from this work. Record relevant commit and checks without reverting unrelated files.
- Classify application runtime files separately from fixture datasets and checked-in family declarations. Inventory the files that a later fresh installation or cutover will create and remove.
- Finalize the logical records in the data model, including transformation inputs, stable row/column identities, review records, checkpoint ownership and deletion records.
- Define contracts for `Principal`, owner-scoped repository operations, file publication, conversation execution scope and job claims. Storage adapters implement these interfaces; lower-level causal packages do not import the HTTP server.
- For each persistence boundary, inventory every current writer and reader, including frontend caches, desk material assembly, worker inputs, downloads and deletion. Assign a single replacement command/query pair and a removal check using the consolidation map above.
- Add import rules for the new packages and preserve current family/engine boundaries. Record the supported database/checkpointer package versions in dependency locks when implementation begins.
- Separate source-code configuration from storage roots and fixture lookup. Schema generation must not open a production database, start jobs or require model credentials.

**Exit:** baseline checks are recorded, unresolved failures are attributed, target ownership is explicit, and the first API contract can be implemented without guessing ownership or identity semantics. See tests `BOUNDARY` and `WIRE`.

## Phase 1: user identity and durable records

**Frontend:** an account boundary that loads the current identity and clears private cached state on sign-out or identity change. Local mode displays its development status; do not build a pretend login form.

**Backend/data:** internal user IDs, the principal interface, development identity provider, owner-scoped repository methods and PostgreSQL schema bootstrap. Composite foreign keys enforce owner/dataset consistency. Add the file-store interface with generated keys, byte/hash verification, pending/ready publication and deletion records. Establish durable jobs with attempt/lease tokens even if one local process executes them initially.

**Interface:** `GET /api/me`, common error envelope, UUID resource identifiers and expected-revision/request-id conventions. Authentication errors return 401; inaccessible resource IDs return 404; stale or busy operations return 409. Do not allow a request body/header to choose an owner in deployed mode.

**Exit:** tests using two principals prove isolation at API and repository boundaries; DB constraints reject cross-owner references; production configuration rejects development identity. Real database/file tests establish transactional rollback and incomplete-file invisibility. See `IDENTITY`, `STORAGE`, `RUNTIME`, `WIRE`.

## Phase 2: private ingestion, dataset library and review

**Frontend:** feature folders for datasets, intake and review. Provide upload progress, saved processing status, retry, profile preview, source notes, review acceptance and a separate start-conversation action. Cover empty, busy, error, expired-upload and stale-review states, keyboard focus and narrow-screen layout.

**Backend/data:** streamed upload with configured bounds; immutable V0; original checksum/parsing metadata; sidecar row identities; column identities; deterministic profile records; private sources and review records. Registration and profiling do not invoke a model. Cache scope includes owner, bytes, parser/profile options and profiler version. A repeated idempotency key with a different payload is a conflict.

**Interface:** upload/create/list/detail/progress/review endpoints from the organization plan. Add `GET /api/uploads/{id}` and `GET /api/jobs/{id}` for durable progress; upload retry uses the original operation identity or explicitly creates a new attempt. Profiles are only served through their owned dataset/version.

**Exit:** a user uploads a CSV, refreshes or restarts the app, reviews its saved profile and study notes, and sees the same owned dataset. Another user cannot see it. Profile/source changes invalidate stale review acceptance. Original bytes remain identical. See `INTAKE`, `PROFILE`, `REVIEW`, `IDENTITY`, browser journey `J1`.

## Phase 3: private conversations and localized context

**Frontend:** conversation routes keyed by conversation ID, multiple conversations per dataset, selected-version indicators, message history, pending questions, activity and resume. Dataset review acceptance is visibly separate from causal run authorization.

**Backend/data:** replace global dataset-name session lookup with conversation identity. Persist ordered messages, owned checkpoint/thread references and job state. Preserve data access through explicit execution scope: owner, conversation, version and knowledge revision. Propagate scope explicitly through threads and subprocess boundaries; never mutate global configuration to select the current user.

Persist immutable `ConversationContext` selections instead of repeatedly overwriting version/knowledge/source pointers with no history. Record typed `DeskDecision` outputs for framing, readiness and admission. Explicit exploration requests pin their data restrictions; results point to artifacts rather than relying on a dataset-wide shelf. Use the [consolidated model](data-model.md) as the source for record ownership and transaction boundaries.

Move mutable memory persistence to revisioned records while retaining `memory.ops` semantic authority. A new conversation deliberately selects dataset knowledge. Existing conversations pin revisions; concurrent writes use expected-revision checks. Context assembly loads only the admitted sources and relevant facts, not all account history.

**Interface:** conversation create/read/list, message submission with client ID, progress and resume. Use `POST /api/conversations/{id}/resume` for interrupted execution, not a new analysis. A new question creates a new conversation; `POST /api/conversations/{id}/analyses` is reserved for a newly authorized analysis request.

**Exit:** two conversations on one dataset retain separate messages/checkpoints; two users remain isolated; a restart resumes the correct owned checkpoint; a competing memory update causes an explicit conflict/review. Changed desk/context behavior is validated with a real conversation and trace inspection. See `CONVERSATION`, `MEMORY`, `CONTEXT`, `RUNTIME`, `J2`, `LIVE`.

## Phase 4: modifications with lineage and review

**Frontend:** transformation request, candidate preview, full change counts with paginated details, parent/branch history, context-impact report, accept/reject and explicit version selection. Keep the original downloadable through an owned file endpoint.

**Backend/data:** transformation/request/execution records; immutable candidate CSV; persisted row map and version columns; measured changes; re-profile; human review record. Accepting a candidate updates only the intended conversation and applicable memory revision atomically. Updating the dataset default is a separate explicit action. No analysis starts automatically.

**Interface:** `POST /api/datasets/{id}/transformations` pins input version, conversation and instruction; `GET /api/transformations/{id}` returns attempt/candidate/validation state; `GET /api/transformations/{id}/changes` returns measured differences; `POST /api/transformations/{id}/reviews` accepts/rejects an exact candidate/change-set hash. `POST /api/conversations/{id}/version-selection` records selection of a reviewed version. All parent references are owner/dataset checked.

Retain outcome restrictions and protected-column policy. First release supports same-source-row transformations; joins, new rows and aggregation require separate future contracts. Supported operations verify their declared effects. Arbitrary script provenance records what was executed without pretending to prove unverified semantic dependencies.

**Exit:** a two-step filter/correction traces retained rows and columns back to V0, original bytes are unchanged, stale acceptance fails, and prior analysis references remain fixed. Review of changed context and new analysis permission is verified in a real model journey. See `LINEAGE`, `EXPLORE`, `MEMORY`, `REVIEW`, `J3`, `LIVE`.

## Phase 5: analysis authorization, execution and results

**Frontend:** exact context review, explicit run instruction, progress, retained failed attempts, result and figure inspectors, source/version labels and follow-up discussion.

**Backend/data:** immutable context snapshot, authorizing message/user/time, analysis request, job/attempt records, frozen designs and published artifact manifest. Authorization and queued work commit together. Job claims use durable leases and token checks. A stale worker cannot publish after replacement or deletion. Resume/retry policies preserve existing input and authorization constraints; changed context requires a new authorized request.

Persist each analysis attempt's evidence ledger separately from chat messages and journal events. Retain typed evidence, accepted before/after changes and revision order. Publish results against a specific ledger revision. Require a conversation owner for chat-generated artifacts, including exploration products made before an analysis. See the [ownership hierarchy](hierarchy.md) and [ledger records](data-model.md).

Preserve `ReasoningStep`, candidate revisions and episode evidence through family adapters. Connect each frozen design to its candidate revision, freezing step and ledger revision. Record statistical `EngineExecution` separately from worker attempts and model episodes. `AnalysisResult` owns the immutable typed result; UI inspectors are projections. File membership and explicit cross-chat artifact use follow `ArtifactFile` and `ArtifactUse` relationships.

Family entry continues through the registry. Test all three executable families independently and retain explicit routing behavior for declared/unbuilt families. Do not change estimator selection, prompts or causal policy as an incidental cleanup. Any necessary behavior change is separately described and real-run validated.

**Exit:** every result resolves to an owned request, snapshot, exact table and memory revision. Deficiencies return to the desk. Retry cannot duplicate published results or alter an authorized design. Each affected executable family has real-run/trace evidence, with limitations reported. See `HANDOFF`, `RUNTIME`, `ADJUSTMENT`, `DID`, `RD`, `RESULTS`, `J4`, `LIVE`.

## Phase 6: files, deletion and operational limits

**Frontend:** authorized downloads, storage/processing errors, deletion progress and a clear effect statement before dataset deletion. Offer whole-dataset deletion; defer arbitrary historical-version pruning and standalone conversation deletion until source-reference retention is specified.

**Backend/data:** all file reads resolve owned published records. Source/raw/profile/context/lineage/results/checkpoint/cache retention have explicit policies. Mark deletion before cleanup, block publication, stop or fence active work, purge descendants and retry remaining removal. Bound disk use, abandoned uploads, job queues and script outputs. Maintain request/job/attempt trace IDs and useful error classifications without leaking file contents into ordinary logs.

**Exit:** deleting a dataset during processing cannot resurrect it; another user's files remain; interrupted cleanup finishes; orphan expiry preserves referenced data. Direct artifact URLs and directory traversal cannot bypass ownership. See `FILES`, `STORAGE`, `RUNTIME`, `J5`.

## Phase 7: deployment, real login and release

**Deliverable:** reproducible single-host deployment, real authentication integration for hosted access, operational runbook and evidence-backed release.

- Build the frontend and locked backend into the application image, running with a non-root identity. Separate DB volume, durable file volume and temporary work. Keep the current sandbox image distinct.
- Select the login provider; validate its identity/session boundary, expiry, logout and CSRF protection where cookies are used. Verify original local-mode ownership tests still apply to the production adapter.
- Specify the trusted sandbox launch boundary. Scripts get only their allowed inputs and output area, never DB/model credentials, full user directories or Docker administration access.
- Separate liveness from readiness. Readiness checks DB, configured durable storage and required execution dependencies. Establish graceful shutdown/draining and restart behavior.
- Restore database and files into an empty environment, verify manifests/checksums, and resume an interrupted conversation. Record achieved recovery times and data-loss window against selected targets.
- Run the integrated suite and live acceptance; publish the measured limits, operating instructions and known gaps. Test realistic concurrent users/files against the selected resource budgets, not an unqualified scalability claim.
- Perform a controlled fresh-schema cutover following the repository policy. Inventory runtime files and scope any reset separately; do not delete uncommitted work, original user uploads or fixture datasets as a side effect. Never add old-shape compatibility readers. A release fallback must restore a mutually consistent application/database/file backup, not run old code against the new schema.

**Exit:** a fresh installation completes the supported journey, production login isolates people, a restart and restore are demonstrated, and all release gates in the verification plan pass. See `DEPLOY`, `IDENTITY`, `LIVE`.

## Definition of complete

The work is complete when a person can sign in, upload/review their CSV, have private conversations, review traceable modifications, authorize a specific analysis, inspect cited results, resume interrupted work and delete their dataset with predictable cleanup. Every operation has an owner, typed boundary, durable state and failure behavior. The original CSV, profiles and lineage remain inspectable until explicit deletion.

Completion also requires generated API types and docs to match the implementation; no duplicate mutable stores or routes; meaningful component, real-storage, browser and live-model evidence; and a tested deployment/restore path. Folder cleanliness alone is not completion.

For the relational hierarchy specifically, completion means a principal and conversation ID can load the correct overview, messages, context selections, journal, authorized analyses, design reasoning, ledger changes, results and artifacts through bounded queries. Every returned product resolves to its exact parent and inputs. Two chats over the same dataset remain independent; explicit reuse is visible. These reads must still work after application restart, without old-layout discovery or provider trace availability.

## Deliverable record for each phase

Keep one short evidence record per completed phase under `docs/acceptance/`: affected code/contract identities, user journey, exact checks and results, storage/ownership observations, real-run trace references when required, and unresolved limitations. Update the current-code handbook only after implementation lands. Keep this plan's status separate from observed behavior.

## Implemented first boundary: ingestion only

The active milestone is [owned intake](../components/02-intake.md): form, structure validation, immutable CSV/context persistence and confirmation. Profiling, review, chats and the wider relational model remain future work. The earlier broader phases below/above must not be interpreted as implemented.
