# Who owns each record, and where it lives

The pre-consolidation inventories in this design are historical. For implemented reviewed context, messages, receipts, checkpoints and artifact ownership, use the [current persistence chapter](../components/09-persistence.md). Remaining proposals below concern hosted workspaces and broader lifecycle work.


Status: proposed application hierarchy, with a code-verified inventory of current storage below. No database tables or runtime files have been moved. Start here for ownership; use the [data model](data-model.md) for record fields and constraints, and [lineage](lineage.md) for CSV edits.

## The application hierarchy

```text
Person / User
├── Uploads                         Staged files, uploader, checksum, processing state
├── Datasets
│   ├── Original CSV (version 0)     Immutable uploaded bytes
│   ├── Dataset versions            Parent version, transformation and measured changes
│   │   ├── Profile                 Statistics for these exact bytes and parsing options
│   │   └── Row/column lineage      Stable identities and change records
│   ├── Source documents            Study descriptions and their provenance
│   └── Knowledge revisions         Sourced dataset facts, confirmations and corrections
└── Chats / Conversations
    ├── Selected dataset/version    Reference to an owned dataset, not another CSV copy
    ├── Context selection history    Exact version, profile, knowledge and sources
    ├── Messages                    What the person and assistant said
    ├── Desk decisions              Question framing, clarification, readiness and routing
    ├── Journal events              What happened, who did it, inputs and outputs
    ├── Checkpoint thread           Where execution can resume
    ├── Exploration requests        Charts, tables and cited facts, even before analysis
    │   └── Jobs/attempts/results/artifacts
    ├── Transformation requests     Link to candidate dataset versions and their review
    └── Analysis requests            One chat can authorize several analyses
        ├── Authorized context snapshot
        ├── Jobs and attempts
        │   ├── Design process      Questions, candidates, evidence, checks and choices
        │   │   ├── Reasoning steps and bounded model episodes
        │   │   └── Candidate revisions and selection/rejection reasons
        │   ├── Evidence ledger     Accepted reasoning changes, with sources and reasons
        │   ├── Engine executions   Each calculation against a frozen specification
        │   └── Results/artifacts   Figures, tables, reports and diagnostics
        └── Frozen designs          Exact specifications used for estimation
```

This is an ownership and relationship hierarchy, not a requirement to put every record in a nested directory. Each chat selects one dataset in the first release. Several chats can reference the same version without copying it. Every resource belongs to one user; the API and database enforce that ownership on reads, writes and file downloads.

The [consolidated data model](data-model.md) is authoritative for all proposed record relationships, lifecycle states, transaction boundaries and sources of truth. In particular, `ConversationContext` preserves selection history, `DeskDecision` records pre-analysis decisions, `AnalysisResult` records the published outcome, and `ArtifactUse` records deliberate reuse without changing the original producer.

A dataset version answers “which table?”. A memory revision answers “which sourced knowledge?”. A chat answers “which conversation?”. An analysis request answers “what was authorized?”. An attempt answers “which execution produced this?”. These IDs must remain distinct.

## Where the analysis is designed

Design reasoning is a first-class part of each analysis attempt. The ledger records evidence changes, but it does not replace the whole design process. Preserve the proposed specifications, alternatives considered, checks, accepted decisions, reasons for rejection or stopping, and the final frozen specification.

The desk first frames the person's question and prepares authorized context. The specialist then determines which analysis that context can support. In the current application this primarily designs an analysis of an existing dataset, rather than assigning participants to a new prospective experiment.

```text
Chat: question → interview → context review and authorization
  Analysis attempt:
    admitted context → relationship/assumption reasoning → candidate designs
    → admissibility and support checks → method selection → frozen design
    → statistical calculation → diagnostics → interpretation
```

That sequence is illustrative; each family's generated ladder defines its actual order and stopping conditions. A deficient study can stop before any design is frozen.

| Design responsibility | Current implementation |
|---|---|
| Frame the question, collect missing context, prepare handoff | [Conversation desk](../components/06-desk.md), [readiness/handoff](../components/08-readiness-handoff.md) |
| Adjustment relationships, identification candidates, support, estimator and freeze | [adjustment reason.py](../../causal_agent/families/adjustment/lane/reason.py), [generated ladder](../ladders/adjustment.md) |
| Difference-in-differences comparison groups, controls, candidates and freeze | [diff-in-diff reason nodes](../../causal_agent/families/diff_in_diff/lane/reason/nodes.py), [generated ladder](../ladders/diff_in_diff.md) |
| Discontinuity score, locality, assumptions, candidates and freeze | [discontinuity reason nodes](../../causal_agent/families/discontinuity/lane/reason/nodes.py), [generated ladder](../ladders/discontinuity.md) |
| Record evidence gathered, structured answers and validation failures | [EpisodeLog and Try](../../causal_agent/lane/episode.py) |
| Preserve evidence/status changes | [Ledger](../../causal_agent/lane/ledger.py) |
| Seal the accepted execution specification | [FrozenDesign](../../causal_agent/common/contracts/frozen.py), extended by each family |
| Assemble the design process for later inspection | [records](../../causal_agent/lane/records.py): `reasoning_history`, `candidate_history`, `episodes`, `ledger`, `designs` |

The proposed database represents this explicitly as `ReasoningStep`, `DesignCandidate` and `CandidateRevision` under the producing attempt, alongside `AnalysisLedger` and `FrozenDesign`. Structured episode evidence and answer/validation logs attach to their step. The frontend can present a Design inspector with considered alternatives, evidence, decisions and final specifications. Persisting these records does not depend on showing every internal prompt in the chat.

## Four different histories

| Record | Question it answers | Scope and content |
|---|---|---|
| Message history | What was said? | Chat: ordered user/assistant/system messages, citations and client submission IDs |
| Operation journal | What happened? | Chat: ordered events such as confirmation, version selection, authorization, exploration, run completion and failure; actor and referenced objects |
| Knowledge revision | What do we know, and why? | Dataset: immutable sourced fields with status and revision ancestry; each chat pins an explicitly selected revision |
| Analysis evidence ledger | How did this analysis change its conclusions? | Analysis attempt: entries and accepted before/after changes, evidence addresses, status, author and reason |

The ledger is not the full conversation and is not private model chain-of-thought. It is the application's explicit, typed evidence and decision record. Rejected proposals and gate errors belong to attempt diagnostics; they must not appear as accepted ledger changes.

Shared dataset knowledge is deliberately reusable, but a model's analysis assumption does not automatically become shared knowledge. A chat's question, hypothetical alternatives, messages, authorization, journal and analysis ledger remain local to that chat. Dataset knowledge updates still pass the existing source/confirmation rules. Other chats retain their pinned revisions until a person selects an update. Reuse of another chat's artifact must be explicit and preserve its original producer and source version.

## Where these records live today

Paths below are defaults relative to the repository root, not the proposed layout. Configuration can override some roots. This inventory follows writers and schemas; it does not assert that a particular dataset has generated every file.

| Current record | Default location | Authoritative code |
|---|---|---|
| Dataset registry | `data/datasets.yaml` | [datasets.entries/write_entries](../../causal_agent/server/datasets.py) |
| Ingestion operation | PostgreSQL `ingestion.operations` | [ingestion service](../../causal_agent/server/ingestion/service.py) |
| Original uploaded CSV | Private S3 `users/<owner>/uploads/<operation>/original.csv` | [ingestion service](../../causal_agent/server/ingestion/service.py) |
| Saved profile and recomputable cache | `data/profiles/<name>.json`; `.artifacts/profiles/` | Retained fixture profiles; [profile.data](../../causal_agent/profile/data.py) |
| Current dataset knowledge and recorded source utterances | `data/memory/<name>/{meta,columns,fields}.yaml`, `said.jsonl` | [Memory schema](../../causal_agent/memory/records.py), [store.save/load](../../causal_agent/memory/store.py), [ops](../../causal_agent/memory/ops.py) |
| Page transcript | `data/web/<name>/transcript.jsonl` | [transcript.path/append/read](../../causal_agent/server/conversations/commands.py) |
| Current session and previous thread references | `data/web/<name>/meta.json` | [SessionManager.get/new_analysis](../../causal_agent/server/conversations/manager.py) |
| Conversation operation journal | `data/memory/<name>/analyses/<analysis-id>/journal.jsonl` | [Step and Journal](../../causal_agent/memory/journal.py) |
| Resume checkpoint | `.artifacts/web/checkpoints.sqlite`, keyed by graph thread | [SessionManager](../../causal_agent/server/conversations/manager.py), [Settings](../../causal_agent/server/settings.py) |
| Authorized input, memory snapshot and desk result wrapper | `data/memory/<name>/designs/<number>/`, including `memory.json`, `handoff.json`, `result.json`, `record.json` | [store.snapshot](../../causal_agent/memory/store.py), [pipeline](../../causal_agent/desk/pipeline.py) |
| Analysis evidence ledger | `ledger` inside the specialist result and run `artifacts.json`; includes `entries`, `revision`, `history` | [Ledger/Entry/Change/apply](../../causal_agent/lane/ledger.py), [records.artifacts/result/write](../../causal_agent/lane/records.py) |
| Analysis outputs and structured figures | `.artifacts/runs/<run-id>/`; design-folder `figures.json` and generated pictures under `viz/` | [lane records](../../causal_agent/lane/records.py), [visualization chapter](../components/15-visualization.md) |
| Exploration picture/table/facts and provenance | `data/memory/<name>/artifacts/<artifact-id>/` | [Artifact and shelf functions](../../causal_agent/memory/artifacts.py), [explore client](../../causal_agent/explore/client.py) |
| Transformed CSV and basic version provenance | `data/memory/<name>/versions/<version>/table.csv`, `version.json` and supporting files | [Version and shelf functions](../../causal_agent/memory/versions.py), [explore client](../../causal_agent/explore/client.py) |

The current name `analyses/<analysis-id>` means a conversation journal, whereas `designs/<number>` identifies a desk-authorized run. A lane may also produce multiple frozen candidate designs. The proposed records use separate conversation, analysis-request, attempt and frozen-design IDs to remove this ambiguity.

The main ownership gaps are concrete: sessions are indexed by dataset name; starting a new analysis appends a divider to the same dataset transcript; artifact metadata has dataset/version/design references without a first-class user or conversation owner; current memory files are overwritten on save. They are not a normalized, transactional revision history. The code already has useful journals, ledger histories and snapshots, but their relationships need explicit records and enforced ownership.

See the [full persistence inventory](../components/09-persistence.md) for deletion behavior, temporary files and cache limitations. `said.jsonl` contains source utterances retained in memory; it is not the page transcript. External traces help debug execution but are not the application's authoritative history.

## Database records and durable files

The proposed division is PostgreSQL for identities, ownership, messages, events, revisions, ledger updates, jobs and file metadata; a private file store for immutable CSVs, pictures, tables, reports and large manifests. A mounted durable directory can implement that file store initially. File keys are generated by the server and resolved through authorized metadata.

Use the [file strategy](data-model.md#file-strategy) as the single layout specification. Dataset files live under the user's datasets; chat products live under the user's conversations. A profile stays with its exact dataset version. An exploration picture belongs to its chat and points to the version used. An analysis figure additionally points to its request, producing attempt and design where applicable.

For a ledger, preserve the existing typed entry/change semantics. The proposed `AnalysisLedger` identifies its attempt; `LedgerChange` stores ordered accepted changes. Current entries are a projection of those changes. A JSON result export may include the ledger, but it is not a separately editable source of truth. Publication records the result's ledger revision so the UI can distinguish a running attempt from a completed result.

## Implementation and verification boundary

Implement ownership/repositories first, then dataset/version/profile persistence, then independent conversations and their messages/events/checkpoints, then analysis ledgers and artifact publication. Follow the [cross-layer implementation plan](implementation-plan.md). This work changes backend contracts, frontend routes and storage together; moving directories alone cannot establish ownership.

Required checks include two users with identical filenames; two chats referencing one version with separate histories and journals; two analyses whose ledger entry IDs may match without collision; a pre-analysis picture owned by the correct chat; explicit memory refresh; immutable original bytes through transformations; and restart/deletion behavior across database and files. These extend the [verification plan](verification.md). No real model run was needed to document the current storage writers. Runtime ownership and new persistence remain unimplemented.

The filesystem table above describes retained fixture/analysis stores. New ingestion writes original CSVs to private S3 and identities/context to PostgreSQL; it does not write staged upload folders, named profile files or YAML registry entries. See [current intake](../components/02-intake.md).
