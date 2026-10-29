# Collaboration, desk chat and integration plan

Status: proposed implementation plan, written 27 October 2026. This document does not claim that the proposed study runtime, protocol review or scientific journal is implemented.

Read the [causal architecture](causal-agent-architecture.md) first. That document owns the scientific invariants and research rationale. This document owns how the builders collaborate and how the architecture becomes part of the existing desk, chat, storage and review experience. Existing behavior was checked against the current working tree, including uncommitted changes, and the [component handbook](../components/README.md), [source index](../components/source-index.md), [architecture](../architecture.md) and [gates](../gates.md).

## 1. Keep the two kinds of collaboration separate

**Building the product:** a coordinator assigns implementation tasks to specialist agents. They exchange proposals, code reviews, counterexamples and test evidence. Their shared records are the architecture plan, build board and decision log. They work in isolated Git worktrees.

**Using the product:** the person talks to one desk about one dataset and one primary causal question. Behind the desk, a study planner can invoke several specialist episodes. Their accepted decisions become scientific records and readable progress. The person does not manage specialist chats, copy messages between agents or approve every internal tool call.

A model thread is an execution detail. A conversation is a product record. A study revision is a scientific commitment. None is a substitute for either of the others.

The integration entry point is the reviewed context pack. Preserve existing intake, profiling and context review. Replace the early exclusive family decision with assessment of all relevant capabilities under one study specification. Preserve the desk as the only conversational interface for domain clarification, context correction and human authorization.

## 2. The product conversation, explicitly

### The normal journey

The chat header always shows the dataset version, selected context revision and primary question. An active-study card shows its own snapshot revision, stage and next required action. These identities can differ after a correction, and the interface must show that difference.

| Moment | What the person sees or says | What the system records and does |
|---|---|---|
| Discuss | "Can this data answer whether the course improved scores?" | The desk explains available evidence. Discussion alone does not authorize investigation or execution. |
| Prepare | The desk presents the proposed population, intervention, comparator, outcome, horizon, uncertainties and investigation budget. | Prepare a concrete investigation snapshot from the reviewed context. Existing valid context confirmations are reused; they are not asked again. |
| First commitment | **Authorize investigation** | Bind the exact context, dataset, question and investigation scope. Start outcome-restricted design work. This does not authorize target-effect fitting. |
| Investigate | "Reviewing how participation was assigned. Two causal accounts remain." | Publish safe progress from accepted decision events. Assess columns, accounts, identification, support and inference obligations. |
| Clarify | "Does attendance describe the period before assignment or attendance during the course?" | A specialist emits a typed `ContextDeficiency`; the desk asks the question with its scientific consequence. No specialist interviews the person directly. |
| Review protocol | A concrete portfolio appears with assumptions, procedures, code manifest, inference, diagnostics, sensitivities and failure rules. | Compile and validate the complete executable protocol without target-effect access. Record its integrity hash. |
| Second commitment | **Approve protocol and run** | Authorize the exact protocol hash and referenced dataset/snapshot. The executor may now access the permitted outcomes. |
| Execute | "Approved protocol is running. Diagnostics and all declared attempts will be retained." | Run only the approved procedures and their justified adaptive rules. No model chooses a favorable specification from their results. |
| Discuss results | "Why did these two estimates disagree?" | Explain the recorded targets, assumptions and results with citations. Preserve primary failures and limitations. Discussion does not reopen design. |

Do not turn every uncertainty into another approval screen. Batch related domain questions, retain unknown answers, and ask only when the answer changes admissibility, meaning or an authorized commitment. An unresolved issue can remain conditional if the relevant method contract permits it.

### A concrete desk exchange

> **Person:** Did completing the preparation course raise math scores?
>
> **Desk:** I have the reviewed account and dataset version 3. The proposed question concerns eligible students and the final math score. We still need to distinguish course completion from being offered a place. Review this study question and authorize its investigation.
>
> **Person:** Authorize the displayed investigation.
>
> **Desk:** I am checking the assignment and measurement accounts. Attendance has two possible interpretations that would change whether it can be used in this design. Does the column describe attendance before the course or during it?
>
> **Person:** During the course. Correct that in the context.
>
> **Desk:** The correction is saved. The earlier investigation remains attached to its original snapshot. Here is the updated snapshot and the affected candidate decisions. Authorize this revised investigation to continue.
>
> **Desk, after design:** The protocol is ready. It identifies the retained causal accounts, the requested target, the proposed primary and companion procedures, inference, sensitivity analyses and what happens if a procedure fails. Target-effect results have not been made available to the design agent by this workflow. Prior exposure is recorded separately. Review and approve this protocol to run it.
>
> **Person:** Approve protocol 2 and run it.
>
> **Desk, after execution:** The primary procedure failed its declared support requirement. One companion produced an estimate under a different retained assumption. The primary failure remains visible. Open Designs to inspect the distinction or ask about either result.

The wording above illustrates behavior; it is not a required script or keyword parser. Preserve the existing semantic interpretation path for free text. Explicit approval controls submit typed, version-bound commands. Free text requesting an analysis leads to the appropriate concrete review card rather than bypassing it.

### Corrections, questions and interruptions

| User action | Required behavior |
|---|---|
| Ask about a column, graph edge, design or result | Discuss the explicitly referenced item. Otherwise use the saved discussion focus. Opening an inspector alone does not change focus. |
| Correct a substantive fact | Commit an authorized source-context revision, preserve the old study snapshot and invalidate affected proposals. A new investigation requires a fresh authorization. |
| Ask a hypothetical | Create an explicitly hypothetical proposal. Do not overwrite durable dataset context or treat it as confirmed knowledge. |
| Ask status during background work | Answer from safe journal projections. Do not cancel the work or expose restricted diagnostics. |
| Pause investigation | Stop dispatching new actions at a safe boundary. Retain current receipts and remaining budget. Resume the same snapshot if it is still valid. |
| Cancel running execution | Fence new work, request supported cancellation, and preserve completed, cancelled and uncertain attempts. Never promise instantaneous interruption of an external numerical process. |
| Correct facts during execution | Stop/fence affected work and retain the original attempts. Resume the old version only if its relevant dependencies remain valid. A false or superseded causal account requires a new approved corrected or explicitly hypothetical protocol, with inherited exposure. Human choice cannot reactivate an invalid certificate. |
| Resume after a crash | Reconcile operation receipts before dispatch. Resume the existing authorized work only when its state is known; uncertain external completion remains visible. |
| Ask for a different estimator after results | Explain what it would answer. Any new computation creates a linked, result-informed study revision requiring a new protocol. |
| Change the primary causal question or dataset | Offer a linked new desk chat with a fresh study. Carry permitted context references and known exposure; do not silently change this chat's question. |
| Refresh this chat to newer dataset context | Show the context diff and select it explicitly. Existing studies continue to identify their original revisions. |

The original study question remains stable within a chat. Several investigation/protocol revisions and results can exist for that question, visibly linked by parent identity.

## 3. What changes at the current integration seams

These observations are verified current behavior; the right-hand column is proposed work.

| Current seam | Proposed integration |
|---|---|
| [Desk graph](../../causal_agent/desk/graph.py) interprets discussion, corrections and exploration together; `ChatState` contains `operation_id`. | Preserve semantic interpretation. Add study discussion and typed planning/execution operations without storing scientific authority in graph messages. |
| [Chat commands](../../causal_agent/memory/context/conversation.py) include one `authorize` action tied to a prepared `snapshot_id`. | Replace that action with `authorize_investigation` and `authorize_protocol`, each bound to its own immutable reviewed object. Remove the superseded action shape at cutover. |
| [Command acceptance](../../causal_agent/server/conversations/commands.py) rejects new commands while a conversation operation is running. Existing `pause` affects fast-track preparation. | Separate short desk-command execution from background-study lifecycle. Permit status/discussion while a study runs. Implement study pause/cancel separately from preparation controls. |
| [Runtime](../../causal_agent/server/conversations/runtime.py) authorizes one prepared family and immediately launches the complete pipeline. | Split investigation, protocol compilation/review and frozen execution. Preparation cannot implicitly authorize fitting. |
| [Pipeline](../../causal_agent/desk/pipeline.py) launches one family subprocess and reads its final result file. | Retain process isolation while adding typed progress/decision exchange with a parent broker. The final file is no longer the only progress channel. |
| [Analysis routes](../../causal_agent/server/conversations/routes.py) expose completed command records. | Project active, awaiting-review, stopped, failed and completed studies, with paged journal events. |
| [Chat UI](../../web/src/pages/Chat.tsx) polls, shows a single analysis review and lists completed analyses. | Keep chat as the entry surface; add stage-specific approval cards, active-study progress and Study/Models/Designs/Journey views. |

The current source account and single-family handoff are useful integration inputs, not constraints on the proposed scientific architecture. In particular, the new default moves outcome-derived DiD diagnostics behind final protocol authorization. Update declarations, prompts, gates, traces, tests and documentation together when implementing this change.

## 4. Authority, runtime and public interfaces

### Ownership across layers

Place shared study contracts and evidence addresses in `common`. Place scientific state transitions, authority rules, dependencies and context projections in `memory`. The shared study controller and capability broker belong in `lane`; method knowledge and engines remain under registered `families`. The desk owns human conversation and preparation. Server adapters own authenticated persistence, scheduling and delivery. Preserve the established import direction.

The parent application process supplies a repository port to the scientific state service. A child study worker receives a sealed input reference, limited evidence grants, a worker lease and a dedicated bidirectional IPC channel. It sends typed action intents and decision proposals. The parent validates and commits them, then acknowledges the accepted revision. Child workers do not import server repositories or receive database credentials.

The existing pipeline inherits its parent environment. Replace that behavior with an explicit environment allowlist when implementing the broker. Do not claim process isolation alone provides data or credential isolation.

Commit accepted events and their projection revision atomically. Include an operation ID, payload hash, expected dependencies and worker fencing token. Repeated delivery of the same request is idempotent; reuse with different content fails. A stale worker cannot commit after cancellation or a newer lease. Persist an intent receipt before external execution or disclosure and a completion receipt afterward. An unknown completion requires reconciliation, not an unrecorded retry.

### Minimum contracts

| Contract | Required content and purpose |
|---|---|
| `StudySnapshot` | Dataset/version/checksum, context revision, exact target, investigation envelope, provenance, known exposure and contract/library versions |
| `DecisionEvent` | Ordered sequence, scientific revision, actor, operation, evidence, dependencies, decision/change, allowed audience and trace reference |
| `ContextDeficiency` | Missing or contradictory requirement, affected candidates, supporting references and the consequence of leaving it unresolved |
| `ProtocolManifest` and its review projection | Exact protocol hash, target/accounts, procedures, justified adaptation, inference, diagnostics, sensitivities, fallbacks, reporting and limitations |
| `StudyAuthorization` | Actor, reviewed object identity, exact version/hash, source command and timestamp |
| `AnalysisBundle` | Original protocol and snapshot references, all planned attempts, results and nonresults, inference objects, artifacts and complete provenance |

Keep event sequence separate from scientific revision: attempts and disclosures can advance the former without accepting new scientific knowledge. Keep study, protocol, candidate, runtime attempt, checkpoint and artifact identities distinct.

### HTTP and chat projections

Extend the existing analysis list/detail routes to return the new study lifecycle. Keep one canonical route family rather than adding duplicate study aliases. Add `GET /api/conversations/{name}/analyses/{analysis_id}/events?after=<sequence>&limit=<n>` for incremental, permission-filtered journey events. Include the active study summary in the chat overview. Use polling first, including the existing polling mechanism, rather than making streaming infrastructure a prerequisite.

Extend typed references beyond columns/artifacts to studies, designs and decisions. Preserve explicit-reference precedence and enforce ownership for every resolution. Add study IDs and expected scientific revisions to study commands. Protocol approval must carry the exact displayed hash. A stale approval returns a structured conflict and the current review identity; it never authorizes a replacement.

Expose investigation status separately from conversation-command status. Keep one short mutating command active per chat and one active scientific job per chat by default. A job can dispatch several independent specialist episodes within its authorized budget. Waiting for human review releases worker capacity without ending the study. Serialize source-context commits and validate study dependencies at dispatch and commit. Discussion uses an audience-filtered projection and cannot automatically forward its full transcript into a design episode.

Each write carries `submission_id` and `revision_id`; the server computes and stores its canonical request hash. `authorize_investigation` additionally identifies the immutable study snapshot; `authorize_protocol` identifies `study_id`, the expected scientific revision and `protocol_hash`. Study pause, cancel and resume identify the study and active job; the broker resolves the current worker lease. A rejected command changes no scientific commitment. Return a typed stale-revision, unavailable-action or missing-context response rather than attempting semantic repair of the authorization.

Generate the OpenAPI and TypeScript contracts from server models. API/UI changes, backend consumers and generated files move together in one integration change.

## 5. Where the information lives

The [persistence chapter](../components/09-persistence.md) describes the current stores. The proposed study journal adds a scientific authority; it does not create another copy of dataset knowledge.

| Information | Authority and proposed handling |
|---|---|
| Original dataset and source account | Existing ingestion identities/checksums and immutable private S3 objects |
| Reviewed dataset knowledge | Existing PostgreSQL `context.revisions`, with source statements, predecessors and edit receipts |
| Chat messages, submitted commands and questions | Existing PostgreSQL `chat` records, including idempotent submission identity |
| Selected chat context, focus and active study pointer | `chat.state`, as a selection/projection, never a replacement for study history |
| Studies, scientific events, protocol versions, approvals and attempts | New PostgreSQL `analysis` records with owner/version constraints, append-only event semantics and transactionally maintained projections |
| Exposure/disclosure history | New records linked to actor, dataset lineage, study, evidence identity, audience and time; independent of any runtime checkpoint |
| Working model context | Ephemeral filtered material plus a persisted manifest of exact evidence references and pinned versions |
| Runtime recovery | Existing SQLite checkpoint facility extended with study/worker execution references; scientific authority remains in PostgreSQL |
| Registered graphs, reports, tables, code and result bundles | Existing owned artifact catalogue with private S3 bytes, checksums, source study/protocol IDs and disclosure checks |
| Local run directories | Proposed new-study scratch and staged immutable files until publication. Current durable local analysis products remain governed by the explicit cutover below. A staged file is not automatically a registered artifact. |
| Evaluation traces and observability | LangSmith spans linked to journal/attempt IDs; traces are evidence of execution, not the source of scientific authority |
| Procedure skills | Version-controlled source, reviewed release identity and study-pinned version |

Use separate tables for studies, revisions/events, protocols, authorizations, attempts and disclosures. Store complete review documents and sealed snapshots as immutable payloads or artifact references; queryable identity/status fields remain columns. Avoid treating the entire growing transcript as the scientific database.

The current implementation keeps authorized snapshots and assembled records under `data/conversations/<id>/analyses/<operation>/` on a durable application volume. The proposed new study shape moves authoritative identities/events to PostgreSQL and registered snapshot/result bytes to private S3. This is an explicit storage transition, not a description of current behavior. Publish and checksum-verify immutable objects before committing their catalogue references; reconcile interrupted publication idempotently. Make local new-study directories disposable only after durable publication and restart-recovery tests pass. Retire superseded local products through the scoped cutover policy, without adding compatibility readers or prematurely deleting unpublished work.

Every result publication uses the original study snapshot revision. The current runtime reads the selected context during publication; that must change before concurrent chat corrections are allowed, otherwise a result can be incorrectly labelled with a newer context revision.

A new chat, CSV name or model context cannot reset known exposure. Link registered copies by owner-scoped dataset identity, verified checksum and available transformation ancestry. When ancestry or external prior exposure is unknown, preserve that uncertainty. Do not claim complete CSV lineage is already implemented. Prior exposure declarations are source records, not evidence that somebody is objectively unexposed.

Record disclosure before serving protected bytes or URLs. Apply restrictions to raw values, prose, summaries, indexes, caches, ordering, errors and trace attachments. Existing `trace.write` can attach file contents; extend its policy so restricted content produces a protected reference until release. Preserve span identity and original timestamps. Log availability rather than claiming proof of reading.

Recovery reconciles journal receipts, referenced artifact hashes and worker leases. A missing committed artifact or ambiguous external completion stops visibly. Reconstructing an old display is read-only; executing from an old checkpoint creates no permission to restore an old exposure or authority head.

On schema cutover, follow the repository's no-compatibility rule: inventory affected analyses/checkpoints/runtime folders, retain a reviewed prior-exposure declaration where relevant, retire only superseded products, and remove old readers and aliases. Preserve original uploads, valid source-context products, fixture inputs and unrelated work. Do not silently erase knowledge of prior result access while retiring an obsolete artifact format.

## 6. Ergonomics of the integrated surface

The default chat answers five questions: what are we studying, which assumptions matter, what remains possible, what is happening now, and what needs the person.

Use four linked views alongside the transcript:

- **Study:** question, assignment and measurement timeline, column dispositions, context sources and deficiencies.
- **Models:** named causal accounts, graph type, latent nodes, consequential unresolved structure and differences between accounts.
- **Designs:** identification, implementation, support and inference separately, with retained and excluded candidates.
- **Journey:** accepted decisions, dependencies, corrections, approvals, attempts and protected-evidence placeholders.

The transcript receives concise milestones and questions. Raw agent conversations and repeated tool logs belong in the detailed journey. Repeated unchanged status should not create new chat messages. Progress can move backward when an assumption is corrected; display the invalidation and consequence rather than a steadily increasing confidence percentage.

Approval cards show the commitment, source evidence, consequences, remaining uncertainty and exact version. Expandable technical details explain algorithms and inference. Graph edits remain proposals until their authority requirements are met. Results keep the predeclared presentation order and visibly retain primary failures.

A request to reveal protected evidence is an explicit exposure transition. Its effect on subsequent design and inference status must be shown before release. The normal workflow avoids this complication by running outcome-derived diagnostics after final protocol authorization.

## 7. Organize the builders and their communication

Keep one coordinator and at most three specialists active simultaneously. The research departments become recurring ownership/review roles, rather than six agents simultaneously modifying shared contracts.

| Owner | Deliverable and boundary |
|---|---|
| Coordinator | Shared contracts, integration branch, dependency order, build board and resolved decision log |
| Scientific owner | Scientific invariants, formal/empirical distinctions, target comparison rules and method obligations |
| State/memory builder | Journal, authority transitions, approvals, exposure and filtered context construction |
| Harness builder | Broker, isolated workers, orchestration, checkpoints, dispatch and cancellation |
| Three method builders | Separate adjustment, DiD and discontinuity capabilities implementing the common contracts |
| Desk/visibility builder | Semantic desk integration, API projections, review controls and the four views |
| Independent evaluator | Contract, adversarial, numerical, real-source and human-review acceptance evidence |

Prepare a recorded baseline containing the intended current source changes. A worktree from `HEAD` alone omits uncommitted work. Keep the original checkout intact and exclude credentials and runtime output from shared snapshots. Each builder receives an isolated worktree from that baseline and an explicit writable ownership area. Shared contracts have one writer, the coordinator. Changes to them require a proposal before consumers implement against a new version.

Give each agent a clean brief: objective, accepted plan/contract version, relevant source sections and skills, owned decisions/files, consumed/produced interfaces, mandatory invariants, exclusions, dependencies and acceptance evidence. Include only the research needed for the role. Start fresh working contexts when roles change; durable decisions remain in the shared records.

Use direct agent messages for coordination, with a short structured form:

```text
Task and contract version:
Claim or requested change:
Evidence or failing example:
Affected owners and invariants:
Proposed resolution:
Required decision or next action:
```

Debate consequential issues through proposal, counterexample, evidence, resolution and a test. A majority vote cannot establish causal validity. Record unresolved scientific objections as limitations or blockers. The coordinator resolves engineering choices within the approved architecture; ask the person only when a proposed change alters scientific authority, exposure policy, supported scope or approval semantics. Ordinary patches do not require repeated architecture approval.

Maintain a build board with owner, deliverable, dependencies, status, blocker and verification links. Maintain a decision log with the disputed proposition, alternatives, rationale, decision authority and affected contract version. These are engineering records. They never become current-study causal evidence.

## 8. Delivery waves and acceptance

| Wave | Specialist slots alongside the coordinator | Exit evidence |
|---|---|---|
| A. Contract and challenge | Scientific owner, harness designer, evaluator | Reviewed contracts, authority map, source obligations and adversarial examples |
| B. First complete path | State/memory, harness, adjustment builder | One real-source study travels from desk preparation through both approvals to complete reporting, with interruption and failure paths |
| C. Method expansion | Adjustment, DiD, discontinuity builders | Each pack conforms to the same contracts with independent numerical checks and explicit unsupported cases |
| D. Chat and visibility | Desk/visibility, evaluator, scientific reviewer | Real backend states drive the conversation, review cards, graph comparison and journey |
| E. Acceptance and repair | Evaluator, affected builder, independent reviewer | Real-model traces and end-to-end evidence satisfy the release gates |

Build the minimum desk controls needed for wave B as part of the shared contract integration. Wave D completes the richer views; do not defer the approval boundary until after method implementation.

Every handoff includes implemented behavior, contract changes, test results, unresolved limitations, independent review findings and exact artifacts/trace links. Builders cannot approve their own acceptance. Integrate small completed changes and update owning documentation and generated declarations together.

Required integration cases include:

- Preparation and investigation authorization cannot trigger target-effect fitting.
- A protocol edited in another session rejects the old approval.
- Discussion remains possible during background work without entering the designer's restricted context.
- A correction during execution cannot relabel existing outputs with the new context revision.
- Cancel/resume preserves completed and uncertain attempts and never resets budgets.
- A specialist deficiency becomes one contextual desk question and stops the affected attempt.
- An old checkpoint cannot erase later exposure or authorization changes.
- A renamed dataset or reused result retains known lineage and exposure.
- A graph correction invalidates exactly the dependent decisions.
- Every planned primary, companion, fallback and diagnostic has a visible disposition.
- Cross-owner artifact, event and evidence access is rejected.
- Protocol and artifact hashes are checked before use; missing bytes fail visibly.
- Legitimate frozen adaptive algorithms execute; result-driven discretionary changes are rejected.

Use component tests and small contract/numerical inputs during development. Endpoint acceptance uses real-source data with the study account recorded. Changes to judgments, gates, prompts or the desk require real Gemini runs and trace readback under the repository policy, with the owner's model-run budget. Missing credentials or sandbox capability remain explicit acceptance blockers. A scripted test or attractive final estimate cannot substitute for that evidence.

Documentation delivery is complete when these two plan files are saved, linked and checked. Building the product is a separate execution phase governed by this plan and the [scientific architecture](causal-agent-architecture.md).
