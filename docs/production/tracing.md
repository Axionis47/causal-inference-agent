# LangSmith storage and application trace links

The pre-consolidation inventories in this design are historical. For implemented reviewed context, messages, receipts, checkpoints and artifact ownership, use the [current persistence chapter](../components/09-persistence.md). Remaining proposals below concern hosted workspaces and broader lifecycle work.


Status: current code/configuration inspected; proposed persistence links described separately. No trace was created, fetched, exported or deleted for this documentation update. Remote ingestion, workspace permissions and actual retention settings were not inspected.

## Two stores, with explicit links

The application keeps its own messages, knowledge, journals, designs, results and files. LangSmith receives execution telemetry: trace trees, model/tool inputs and outputs, validation failures, timings and attachments. The proposed application database stores links to that telemetry. A trace is not the database record of an analysis, a backup of its files, or a checkpoint for resuming it.

```text
Application                                    LangSmith
User → Conversation                            Tracing project
         → Job → JobAttempt → TraceLink ──────→ Trace (root run and child runs)
                    │                             ├── desk nodes and judgments
                    ├── ReasoningStep             ├── analysis-family subprocess
                    ├── EngineExecution           ├── model episodes and tool calls
                    └── Artifact → StoredFile     ├── validation gates and errors
                                      │           └── artifact attachment copies
                                      └───────────────┘
```

A conversation spans multiple turns and therefore can have multiple trace roots. A root trace can contain a whole specialist execution beneath the desk invocation. `TraceLink` records the relevant root and span, rather than assuming one chat equals one trace. A LangSmith run means one node/span; it is not the same identity as an application analysis or local run-directory name. LangSmith [threads](https://docs.langchain.com/langsmith/threads) group related traces using thread metadata; that grouping does not provide application access control.

## What the code stores today

| Concern | Inspected behavior | Source |
|---|---|---|
| Configuration | Local `.env` declares tracing enabled and project `causal-agent`; `.env.example` has the same project default. Runtime environment can override it. Compose defaults tracing off unless configured. | [example](../../.env.example), [Compose](../../compose.yaml) |
| Span records | `span` opens a LangSmith run with name, type, inputs, metadata and tags. `end` records outputs/errors; the SDK receives these records for transmission. | [trace.py](../../causal_agent/common/trace.py) |
| Model judgments | Structured calls record schema, system/user prompts, model/tier, typed answer and returned debug thought text/token counts when available. That debug text is not confirmed knowledge or a citable decision record. | [llm.py](../../causal_agent/common/llm.py) |
| Episodes/tools/gates | Bounded episodes, tool arguments/results, attempts and gate errors appear as nested spans. | [episode.py](../../causal_agent/lane/episode.py), [trace.gate](../../causal_agent/common/trace.py) |
| Chat grouping | Each session has a UUID thread ID. Graph configuration includes it plus dataset, analysis-journal ID, turn, phase and descriptive tags. New analysis creates a new thread. | [Session.cfg and new_analysis](../../causal_agent/server/conversations/manager.py) |
| Process continuity | `child_env` sends current run headers in `CAUSAL_TRACE_PARENT` and conversation ID in `CAUSAL_THREAD_ID`. Child code enters `trace.parent()`; lane invocation receives the thread ID. | [trace.py](../../causal_agent/common/trace.py), [pipeline](../../causal_agent/desk/pipeline.py), [exploration client](../../causal_agent/explore/client.py), [exploration server](../../causal_agent/explore/server.py) |
| File copies | `trace.write` writes the local file, then calls `artifact`. Nonempty files at most 15 MiB are attached as bytes. Larger/empty files retain metadata without a binary attachment. Text outputs show a preview truncated at 20,000 characters. These are application limits, not LangSmith service limits. | [trace.artifact/write](../../causal_agent/common/trace.py) |
| Reading traces | `make trace` uses the LangSmith client to query project roots and their child runs, then prints a tree. It does not read a local authoritative trace archive. | [evals.trace](../../causal_agent/evals/trace.py) |
| Local application storage | Transcripts, knowledge, journals, snapshots/results and checkpoints remain in their existing locations. Some episode/reasoning details also appear in result files, but this is not a complete local LangSmith export. | [storage inventory](../components/09-persistence.md) |
| Deletion | Current dataset/session cleanup targets local files and checkpoints. It has no remote LangSmith trace-deletion operation. | [datasets.delete_dataset](../../causal_agent/server/datasets.py), [SessionManager.delete](../../causal_agent/server/conversations/manager.py) |

LangSmith supports [binary trace attachments](https://docs.langchain.com/langsmith/upload-files-with-traces). Consequently a generated CSV, report, code file or picture routed through `trace.write` can exist in both local storage and LangSmith. The trace helper's truncation is not redaction: nested payloads and attached file bytes can contain data values or study context. This inspection did not establish a centralized payload-redaction policy or a durable local telemetry delivery queue.

Current configuration shows the application's intent to send telemetry, not proof that a particular trace was accepted. Actual service reachability, credentials, sampling, quotas and retention affect availability. No self-hosted LangSmith deployment is defined by the application's Compose file; the configured SDK endpoint determines the service destination.

## Additions to the consolidated model

| Record | Fields and relationships | Purpose |
|---|---|---|
| TraceLink | owned application scope, provider, endpoint/region identifier, workspace/project identity, trace ID, run/span ID, parent span ID when available, thread ID, job attempt, optional reasoning-step/engine-execution/artifact reference, timestamps, observed delivery/availability state | Navigate from an owned application operation to the corresponding provider trace/span |
| TraceExport | owned trace link/scope, export file ID/hash, format/schema version, export time, included traces/spans and attachments manifest, completeness and retention/deletion state | Optional deliberate archive when required; not an automatic copy of all provider traces |

Use one or more links per job attempt because work may span trace roots. Several application records can refer to spans within the same root. A pre-job intake trace can use a typed upload scope instead; do not invent a chat owner before a chat exists. Enforce the same owner/dataset/conversation constraints as the linked records. Store provider endpoint/project context so a bare trace UUID does not imply the wrong environment.

`JobAttempt` uses these links instead of an unstructured trace URL. Use IDs as canonical references; construct UI links through the provider adapter. Never put API keys, credentials or expiring attachment-download URLs in durable trace references. An artifact's provider attachment is linked through its artifact span; the authoritative bytes remain its `StoredFile`. Do not register provider copies as a second mutable artifact.

Instrumented operations should propagate opaque owner, dataset, conversation, message, analysis-request and job-attempt IDs where applicable, plus step/design IDs on relevant child spans. Preserve thread metadata on child runs as required by the [LangSmith thread documentation](https://docs.langchain.com/langsmith/threads). The current names/tags do not yet implement this full user-owned ID model. A live trace readback must verify propagation; code inspection alone is insufficient.

Delivery state must distinguish disabled/not submitted, submitted and remotely observed. Returning from an SDK call does not by itself prove remote durable availability. A tracing failure must not rewrite the application's scientific outcome; implement and test the operational failure policy without claiming current helpers already swallow every provider error. Trace links remain useful even after the remote trace expires, with availability reported honestly.

## Retention, deletion and access

LangSmith has its own [retention and deletion controls](https://docs.langchain.com/langsmith/data-purging-compliance). The actual project retention tier and workspace settings are unverified here. Do not promise indefinite history or assume local dataset deletion removes remote copies.

Before hosted user data, define the trace payload policy: which prompts, row values, source text, generated code and attachments may leave the app; any redaction; the debugging level; and retention. Enforce that policy at the tracing chokepoints. Application user ownership does not automatically constrain LangSmith workspace membership. Keep full operator traces restricted; an application-facing design inspector uses owned structured records. Any user-facing trace view must pass application authorization and deliberate payload filtering, not create public shared links by default.

Extend `DeletionOperation` with remote trace cleanup tasks and any registered local exports. Track submission, failure/retry and provider-confirmed or policy-governed completion separately from local cleanup. If a trace spans more than one operation, apply its complete recorded ownership before deciding deletion scope. Whole-dataset/account deletion must not remove another owner's traces. Dataset deletion can become locally complete while external cleanup remains pending, and the status must say so.

An optional export must declare whether it includes all spans, input/output payloads and attachment bytes. A JSON export with temporary provider URLs is not a durable archive of those files. Trace exports, if enabled, are owned storage products with explicit retention, authorization and deletion handling.

## Verification

Existing [trace-policy tests](../../causal_agent/common/tests/test_trace_policy.py) check chokepoints and [attachment tests](../../causal_agent/common/tests/test_trace_artifacts.py) cover empty/nonempty file behavior. Proposed integration coverage must also test owner/scope links, child-process thread propagation, separate trace roots across turns, redaction/attachment policy, disabled/unavailable tracing, incomplete exports, remote expiry and independently retried deletion. Read an actual LangSmith tree after implementing changes to verify nesting and metadata. No new model run was needed for this inventory.
