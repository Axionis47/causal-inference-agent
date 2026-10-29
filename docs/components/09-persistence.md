# 09. Artifact and persistence lifecycle

[Component map](README.md) · [Source ownership](source-index.md)

The active application has one dataset-context authority in PostgreSQL. SQLite holds graph execution checkpoints. Original uploads and registered artifact files live in private object storage. Local analysis files remain on the existing durable application volume.

| Product | Authoritative owner and lifecycle |
|---|---|
| Original CSV and original study account | Ingestion PostgreSQL/private S3, immutable source and checksum |
| Numerical profiles and original extraction evidence | Owned profile facts and immutable extraction audit records; extraction publishes revision 1 directly |
| Context revisions from extraction onward, and edit receipts | `context.revisions` and `context.receipts`, immutable published records; replacements retain predecessors |
| Pre-chat reviews | `context.reviews`, question, starting revision, draft edits, per-column dispositions and completed revision |
| Conversation selection and preparation | `chat.state`, exact selected revision, question, focus, readiness, pending questions and active operation |
| Messages, commands and questions | `chat.messages`, `chat.commands`, `chat.questions`; idempotent publication and structured answer identity |
| Exploration and causal artifacts | Shared owned request, attempt, result, artifact and file registrations in `chat`; private S3 with checksum-verified retrieval |
| Reused artifacts | `chat.artifact_selections`, explicit selection in another owned chat from the same dataset; original producer/version retained |
| Execution checkpoints | `.artifacts/web/checkpoints.sqlite`, operation reference only |
| Authorized analysis | `data/conversations/<id>/analyses/<operation>/`, immutable snapshot reference, handoff bytes, result and assembled record |
| Lane runtime | `.artifacts/runs/<run-id>/`, family-owned inputs, freeze records, diagnostics, results and figures |
| Exploration scratch/cache | `.artifacts/runs/explorations/<conversation>/<artifact>/`, disposable; never the catalogue |
| Fixture inputs | Existing `data/raw`, `data/claims`, `data/context` and fixture memory used by CLI evaluations; not active chat authority |

`server/context/repository.py` commits canonical knowledge under a dataset-version lock. `server/conversations/commands.py` publishes transcript and operation progress. `publication.py` validates and registers evidence after result assembly. Origin (exploration or analysis) is independent of format (figure, table, report or structured result). Registration and file reads enforce owner, conversation, dataset version and checksums.

A restart exposes interrupted execution. Resume uses existing receipts, task identities and completed artifacts. A started external attempt without a recoverable completed result stops visibly. It does not run again by accident. Shared volumes remain required for SQLite, input caches and analysis files in this single-application deployment.

The old filesystem conversation knowledge writer, transcript writer and checkpoint shape have been retired. There are no compatibility readers. Before applying the new version to an installation containing superseded chats, inventory and retire only those chats and their checkpoints/runtime folders. Preserve original uploads, valid Phase 4 products, fixture inputs and unrelated runtime work. Hosted account lifecycle, independent scaling and complete derived-table lineage remain separate production work.

Extraction, dataset browsing, pre-chat review and chat all use `context.revisions`. The extraction audit tables do not supply a separate current context. Preparation cannot overwrite an edited revision, and review opening does not create a copy.
