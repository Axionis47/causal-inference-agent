# Component handoff contracts

## Purpose

A handoff transfers a request or accepted result between components.
It does not transfer permission to change the originating component's records.

Each contract identifies the sender, receiver, payload, validation, durable result, and failure behavior.
The IDs below are documentation references.
They are not new runtime identifiers.

## Handoff index

| ID | Sender → Receiver | Input | Durable output |
| --- | --- | --- | --- |
| H01 | Frontend → API | Authenticated HTTP request | Domain operation or explicit error |
| H02 | Intake → Storage and Database | Source bytes and submission ID | Dataset version and checksum |
| H03 | API → Queue → Worker | Frozen operation identity | Reserved and completed work records |
| H04 | Context preparation → Context service | Validated source-derived proposal | Draft context revision |
| H05 | Chat proposal → Context service | Explicit Apply request | Revision and chat selection |
| H06 | Chat context → Analysis | Applied brief and readiness fingerprint | Frozen analysis inputs |
| H07 | Coordinator → Scientific task | Scoped evidence and accepted prerequisites | Task receipt or blocker |
| H08 | Scientific task → Admission gate | Typed model proposal | Admitted claims and inactive candidates |
| H09 | Frozen causal account → Diagnostics | Permitted projection and declared question | Computation receipt and feasibility evidence |
| H10 | Frozen plan → Owner approval | Exact science and program hashes | Approval receipt and execution job |
| H11 | Worker → Sandbox runner | Program, projected bytes, runtime identity | Immutable execution receipt |
| H12 | Domain events → Frontend | Ordered persisted events | Derived terminal and inspector state |
| H13 | Saved analysis → Chat turn | Authorized artifact catalogue and hashes | Bound artifact discussion |

## H01: Frontend to API

**Sender:** React feature components through `web/src/lib/api/client.ts`.
**Receiver:** Authenticated FastAPI routes.

The request uses generated OpenAPI types.
Mutations require `X-Causal-Client: web` and an acceptable Origin when Origin is present.
The API resolves the session and owner.
Client-supplied record identities do not establish ownership.

The owning backend service validates the operation.
The response returns a saved view, an accepted background-operation identity, or an explicit error.
The frontend refreshes its derived state from the API.
It cannot set readiness, scientific acceptance, or approval hashes on the server.

Source: [HTTP boundary](../backend/main.py), [API client](../web/src/lib/api/client.ts).
See [API conventions](api-reference.md#request-conventions).

## H02: Intake to storage and database

**Sender:** Dataset upload service.
**Receiver:** CSV validator, private file store, and PostgreSQL.

The input contains title, filename, bytes, owner, and submission ID.
The validator checks encoding, headers, row shape, and configured limits.
Storage publishes original bytes under a server-generated key.
The database then commits the dataset and initial version together.

A matching repeated submission returns the saved identity.
A changed payload under the same submission ID conflicts.
A crash before metadata commit can leave an unreferenced file.
The same submission can reuse that immutable file.
Automatic orphan deletion is not implemented.

Source: [dataset service](../backend/datasets/service.py), [storage](../backend/storage.py).

## H03: API to queue and worker

**Sender:** Context, chat, or analysis service.
**Receiver:** Procrastinate and the capability's job handler.

The service stores frozen inputs and enqueues the operation in one database transaction.
The HTTP response does not wait for model completion.
The worker claims the saved operation and validates its compatibility.
It reserves attempts before external calls.

Duplicate delivery reuses saved work under the capability's ownership controls.
Interrupted or uncertain calls remain charged.
LangGraph stores chat and analysis position separately from domain records.
Context preparation uses its own bounded persisted steps.

Source: [queue setup](../backend/jobs.py), [worker](../backend/worker.py), capability `jobs.py` modules.

## H04: Preparation to context publication

**Sender:** Context preparation worker.
**Receiver:** Context publication service.

The proposal binds selected source bytes and the expected parent revision.
Validation checks statement structure, references, source quotes, and participants.
The worker can publish a draft only if the parent remains current.

A stale result cannot overwrite newer user edits.
The proposal and failure remain processing evidence.
A published draft does not become reviewed context automatically.

Source: [context jobs](../backend/context/jobs.py), [context service](../backend/context/service.py).

## H05: Chat proposal to applied context

**Sender:** The user's Apply changes action on a completed context-mode turn.
**Receiver:** `chat.service.apply_context` and `context.service.save_revision`.

The input identifies the conversation and proposal turn.
The service checks turn ownership, successful completion, context mode, and the bound base revision.
It rejects a stale proposal or an active competing turn.

The operation resolves proposal-local edit references.
It saves edits, the analysis brief, and the chat selection in one transaction.
A repeated Apply returns the same selected revision.
Earlier turns and analysis inputs remain unchanged.

Source: [chat service](../backend/chat/service.py).

## H06: Applied context to analysis launch

**Sender:** Analysis creation request from the conversation.
**Receiver:** `analysis.service.prepare`.

The request includes a submission ID, expected context revision, and readiness fingerprint.
An optional question must equal the applied brief question.
The route obtains `ConversationScopeReader` from Integration.
The reader checks ownership and locks the conversation on the caller session.
It returns the dataset version, selected revision, and exploration exposure count.
Analysis recomputes readiness and owns its preparation count.
It rejects stale inputs before enqueueing.

The saved input includes dataset checksum, context hash, brief, question, readiness receipt, and policy version.
Runtime construction verifies those bindings on initial work and recovery.
Later revisions cannot enter that run through retrieval.

Source: [analysis service](../backend/analysis/service.py), [readiness](../backend/analysis/readiness.py).

## H07: Agent to accepted handoff

**Sender:** One bounded analysis agent.
**Receiver:** A downstream agent with a private conversation.

The input packet contains the objective, frozen identities, compact evidence catalogue, accepted handoff references, and remaining allowance.
Native model calls and delegated reviews consume the owning agent's saved budget.
Tools retrieve existing frozen keys and validate small typed submissions.
The backend assembles the final study or design instead of asking downstream agents to rewrite it.

An immutable handoff stores its producer, schema version, input hash, dependencies, evidence, assumptions, and unresolved issues.
Consumers load it by producer identity and content hash.
Completed tool receipts and accepted handoffs survive worker interruption.
Progress events use stable keys and can be restored after the domain receipt commits.

Source: [orchestration](../backend/analysis/agents.py), [agent store](../backend/analysis/agent_store.py), [native loop](../backend/analysis/agent_runtime.py).

## H08: Proposal to claim admission

**Sender:** Scientific model task.
**Receiver:** Claim admission and typed handoff validation.

The model supplies a typed proposal with references and rule selections.
It does not author its own authority status.
Source admission restores the compatible frozen source statement.
Supported deductions require compatible premises and scope.

Other interpretations remain inactive candidates or explicit rejections.
Necessary assumptions can become relied upon only under the causal specification rules.
Raw proposals remain available for audit.
Only admitted claims enter downstream factual packets.

Source: [authority](../backend/analysis/authority.py), [packet assembly](../backend/analysis/packets.py).

## H09: Causal account to diagnostics

**Sender:** Accepted, frozen causal account.
**Receiver:** Diagnostic projection and computation path.

Each diagnostic declares a question and a consequence.
The harness constructs permitted input bytes before code runs.
It withholds target-period outcomes, known derived proxies, and unsupported timing scopes.
A supported earlier-outcome window requires explicit lineage and timing evidence.

The result records code, exposed inputs, observations, failure, and feasibility implications.
It cannot create causal premises or change the frozen graph.
If safe projection cannot be established, the diagnostic is blocked.

Source: [lineage](../backend/analysis/lineage.py), [design tools](../backend/analysis/agent_design.py).

## H10: Frozen plan to approval

**Sender:** Owner review of the displayed packet.
**Receiver:** `analysis.service.approve`.

The request contains `snapshot_hash` and `program_hash`.
The service checks ownership, current contract, status, and program identity.
The program hash binds the saved runtime identity.
The execution path checks the actual runner runtime before computation.
Approval and execution enqueueing occur under the locked operation.

A stale approval cannot authorize a newer program.
A repair creates a separate program revision and requires renewed approval.
The scientific snapshot remains fixed.
Explicit rejection preserves the packet and reason without execution.

Source: [approval service](../backend/analysis/service.py).

## H11: Worker to sandbox runner

**Sender:** Chat exploration or approved analysis execution.
**Receiver:** Shared sandbox adapter and separate runner service.

The request binds execution identity, code, permitted bytes, and runtime hash.
The runner enforces its local isolation and resource limits.
It returns outputs, runtime evidence, and an immutable receipt.

The caller validates outputs before publishing artifacts or computational facts.
A saved matching receipt supports recovery after a lost workflow checkpoint.
The runner does not decide which causal design to execute.
The application must establish that authorization before submission.

Source: [sandbox adapter](../backend/execution/runner.py), [runner](../docker/explore_worker.py).

## H12: Domain events to frontend

**Sender:** Committed analysis event records.
**Receiver:** SSE endpoint and analysis panel.

Each event has a run-local sequence number.
The client reconnects from its last received sequence.
The server replays later records without creating new scientific decisions.

The panel derives stage progress, handoff summaries, approval actions, and results from saved records. Historical checkpoints remain inspectable.
Connection status and last activity remain separate.
The workflow graph shows dependencies.
The causal graph shows accepted scientific relations.

Source: [event endpoint](../backend/analysis/routes.py), [analysis panel](../web/src/features/analysis/panel.tsx).

## H13: Saved analysis to chat

**Sender:** Saved analysis artifact catalogue.
**Receiver:** A newly submitted chat turn.

Chat requests `AnalysisDiscussionReader` from Integration.
Analysis captures the available catalogue and file hashes on the caller session.
Chat saves that binding in the turn submission transaction.
The model selects exact authorized analysis IDs and filenames.
The reader returns bound scientific views and validates selected file checksums.
Analysis owns historical fallback and computed-evidence classification.
Chat does not query Analysis records.
Later artifacts cannot silently enter the saved turn.

Validated execution facts can support numerical claims.
Arbitrary JSON, proposed explanations, and inactive assumptions cannot become computed evidence.
Frozen context and checkpoint artifacts remain discussable before execution exists.

Source: [read interfaces](../backend/handoffs.py), [analysis discussion](../backend/analysis/discussion.py), [integration adapters](../backend/integration/readers.py).

## H14: Prepared request to Model Access

**Sender:** Context, Chat, or Analysis model operation.
**Receiver:** Shared Model Access transport.

Context and Chat retain their structured-response transport.
New analysis agents use `PreparedToolRequest` and native Vertex function calling.
The prepared request records the system objective, complete provider conversation, tool declarations, model, output limit, and thinking settings.
The exact request is saved and charged before dispatch.

The provider's native content, function-call identities, opaque signatures, usage, and tool observations remain ordered across turns.
The application executes tools itself, without hidden SDK tool execution or automatic provider retries.
The configured `VALIDATED` mode constrains native function selection; backend validation still checks every argument and scientific claim.
Historical compact structured-response records remain readable under their original contracts.

Source: [model transport](../backend/model_access.py), [native loop](../backend/analysis/agent_runtime.py), [scientific schema policy](../backend/analysis/provider.py).

## H15: Application page to feature views

**Sender:** The application conversation page.
**Receiver:** Chat workspace and Analysis panel.

The page supplies one typed side-panel function to Chat.
Chat passes current conversation and context display state to that function.
The page renders Analysis with those values.
The validated `view` URL parameter selects Chat or Analysis.
Both views stay mounted so switching preserves unsent input and local inspection state.
A deep link with an analysis identity defaults to Analysis unless it explicitly selects Chat.
Context links retain `fromConversation` through section and column navigation.
That return destination does not change the conversation's saved context.
URL state, Query state, polling, and SSE remain with their existing owners.
No new global store or duplicate scientific state is created.

Source: [conversation page](../web/src/app/conversation-page.tsx), [Chat workspace](../web/src/features/chat/workspace.tsx).

## Maintenance rule

For each boundary change, update its input, writer, transaction, output, and failure behavior here.
Update the relevant API or domain contract at the same time.
Add a focused verification reference in the [verification guide](verification.md).
Do not describe a future decoupling as an existing dependency boundary.
