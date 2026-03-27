# Chat and exploration contract

## Ownership

Chat owns conversations, ordered turns, replies, context proposals, and exploratory artifact manifests.
It reads applied context through the Context component.
It reads saved analyses through the Analysis artifact boundary.
It cannot alter the frozen scientific records of an existing run.

Source: [contracts](../backend/chat/contracts.py), [service](../backend/chat/service.py), [workflow](../backend/chat/graph.py).
User procedures are in the [user guide](user-guide.md#discuss-and-explore).

## Turn inputs

A conversation binds an exact dataset version and an optional initial context revision.
An applied context selection can change the revision used by future turns.
Existing turns retain their original inputs.

Each send request includes a submission ID, expected turn number, text, mode, and selected artifact IDs.
The service permits one active turn per conversation.
Stale turn numbers and conflicting repeated submissions fail.
A matching repeated submission returns the original turn.

| Mode | Permitted behavior | Saved context change |
| --- | --- | --- |
| `chat` | Discussion and explicitly requested exploration | None |
| `context` | Discussion and proposed keyed edits or analysis brief | Only through Apply changes |

A successful reply is not permission to apply its proposal automatically.
The Apply operation validates the original context binding again.
See [H05](component-contracts.md#h05-chat-proposal-to-applied-context).

## Workflow and recovery

Procrastinate delivers the saved turn to the worker.
LangGraph stores workflow position in PostgreSQL.
Domain records own the transcript, replies, and files.

The graph interprets the current request from a compact saved-analysis catalogue.
It selects exact analysis identities and files, then supplies only those frozen records to reply generation.
Selected analyses and exploratory artifacts have separate identity namespaces.
It starts exploration only for an explicit computational request.
It then composes, validates, and publishes the reply.
Model calls are reserved before execution.
Completed work can be reused after interruption.

A failed earlier turn cannot be retried after a later turn starts.
A completed reply and its exploratory artifact manifest publish atomically.
The application preserves failed attempts and consumed budgets.

## Exploration boundary

Exploration is an open-ended Python workflow for descriptive analysis and visualization.
Its inputs include the dataset, pinned context, and authorized selected artifacts.
It does not authorize a causal estimate or change source bytes.

The model can inspect inputs, submit code, inspect outputs, and submit a final product.
Python runs only through the separate sandbox runner.
The model must inspect generated figure pixels before it publishes that figure.
A prose correction can reuse a successful figure.

The runner cannot access application credentials, private source storage, or Docker controls.
Its restrictions are described in [security](security.md#generated-code-boundary).

## Limits

| Scope | Limit |
| --- | --- |
| Conversation | 60 turns |
| Conversation transcript | 240,000 characters |
| User message | 8,000 characters |
| Explicitly selected prior exploratory artifacts | Four per turn |
| Selected analysis files | Four per interpretation |
| Model calls | 24 per turn |
| Script executions | Eight per exploration |
| Input/output inspections | 16 per exploration |
| Final exploration submissions | Three |
| Script execution time | 60 seconds |
| Reply correction | One shared correction for schema, citation, and numerical failures |

A failed script consumes its allowance.
An interrupted model call does not restore budget.
The conversation limit causes an explicit stop, not silent transcript truncation.

## Evidence and replies

Context citations must resolve against the turn's pinned revision.
Analysis references must belong to the turn's authorized catalogue.
Every digit-written numerical claim needs a source key and value.
Each short numerical handle names one source value, including literals in a prespecified question.
The renderer resolves supported numeric placeholders, then validation checks the exact source and permitted rounding.
Worded quantities do not create numerical handles. A source containing several literals yields separate handles.

Computed facts, supplied context, and user statements retain separate basis metadata.
A number parsed from code can explain a code constant.
It cannot establish an observed numerical result.
A failed narrative gate does not erase an already validated computation.

A second invalid reply remains unpublished.
These checks do not verify every nonnumeric statement or prove scientific validity.

## Saved analysis artifacts

The service freezes the available analysis catalogue at turn submission.
The binding includes file hashes, snapshot hash, result hash or absence, and status.
Later execution cannot add evidence to that earlier turn.

Chat can discuss frozen context, checkpoint receipts, code, diagnostics, review rejections, and validated execution outputs.
A result does not need to exist before artifact discussion is useful.
The reply must distinguish preparation, approval, execution, and failure.

Only validated execution facts acquire computed-evidence authority.
Arbitrary JSON and inactive hypotheses remain distinct.
See [H13](component-contracts.md#h13-saved-analysis-to-chat).

## Frontend behavior

The frontend polls persisted turn status and displays complete replies.
Token streaming is deferred.
The URL owns the selected conversation.
The composer owns unsent text and mode.
Local state owns selected follow-up artifacts.

Enter sends a message.
Shift+Enter inserts a newline.
Input-method composition does not submit.
Failed saves retain the relevant form state where supported.
Saved conversations and artifacts reopen from the backend.

## Scope

Chat uses durable turns, immutable artifact manifests, explicit input bindings, and isolated computation.
Conversation deletion, cancellation, arbitrary external inputs, and dataset transformations are outside the current chat contract.
See [verification](verification.md#real-model-and-browser-acceptance) for connected acceptance procedures.
