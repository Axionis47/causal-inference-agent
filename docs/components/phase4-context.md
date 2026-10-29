# Phase 4: Organized dataset and column context

The dataset details page can prepare and display the original study account organized by dataset and column. This is a saved interpretation of supplied text, not analysis readiness, human confirmation or a causal graph. Missing fields display **Not supplied**. There is no approval step, editor or new conversation path.

## Ownership and contracts

- `causal_agent/memory/context/contracts.py` owns the typed statements, citations, relationships and responses. `extract.py` owns fill, validation, assessment and at most one revision.
- `causal_agent/server/context/` owns PostgreSQL publication and retrieval. The shared ingestion lookup enforces owned, ready datasets before any context access.
- `web/src/components/DatasetContext.tsx` owns the context side panel, lazy column details and source highlighting. It can be mounted later in chat without implementing chat now.

Every profiled column has an entry, even when unmentioned. Column statements cover meaning, values, measurement, timing, construction, missingness and qualifications. Dataset statements cover observations, population, organization, study process, measurement, supplied question and qualifications. Dataset statements can apply study-wide or reference particular column identities. Profile numbers stay in profiling storage.

Each assertion carries an ID, source-statement/interpretation distinction, exact quotations, Unicode code-point offsets and qualifications. A citation proves that the quoted passage exists, not the truth of its interpretation. A relationship kind also needs source support: assignment rules, receipt coding and programme descriptions are not promoted to causal-effect claims. Semantic assessment compares both source coverage and support. Contradictory or qualified source accounts remain qualified. No missing relationship is interpreted as absence.

Relationships are stored once and retrieved under all participating columns. Types are measurement, derivation, temporal order, membership, assignment rule, selection rule, association, causal influence, explicitly absent causal influence and unclassified. Participants may be columns, concepts or events. Multi-party alternatives, timing, conditions, directness and absence scope are retained without constructing graph edges. Unmeasured concepts/events carry their own citations.

## Execution and decision authority

Preparation requires a saved profile with a complete column identity index; a partial numerical profile can meet that requirement. No automatic profiling occurs.

1. Load the owned original text and identities, without reading the CSV.
2. Fill the typed context using the existing structured model wrapper.
3. Give the model the complete source as numbered lines. It selects a line and an exact quotation; code calculates and validates the stored character offsets. Repeated quotations within the selected line require a longer unique passage. There is no fuzzy matching. Participant kinds are derived from their referenced column/concept/event identities and traced; code does not ask the model to independently decide the same type twice. Unknown or ambiguous IDs still fail validation.
4. Assess support, mappings, coverage and qualifications with a separate model invocation. Each correction carries affected statement IDs, source line numbers and a specific reason. Omitted source information has no statement ID yet. Code checks assessment references before passing corrections to the extractor. Unsupported claims, incorrect mappings and lost material facts or qualifications block publication. Equivalent wording, alternative valid field placement and supported information appearing in both a column attribute and a relationship are nonblocking notes.
5. If necessary, revise once and reassess. At most four application model calls; provider transport retries remain bounded by the existing wrapper.
6. Publish only validated, assessed content in one transaction.

Empty/whitespace-only text produces an empty structure for every column without a model call. Unsupported statements must be removed rather than promoted. Invalid references/citations or uncorrected semantic assessment failures prevent publication and preserve the previous revision. Source text is never silently truncated. Reports are limited to 2 MiB plus contract-specific bounds. Extractor and assessor share one provider-schema implementation. It omits unsupported numeric/length bounds while preserving required collection minimums. Local Pydantic validation retains every bound. There is no background queue or running-job record.

There is one orchestrator, one extractor role and one assessor role. These are two prompts through the same model wrapper, not independently deployed agents. No routing, per-column model calls or tool selection is needed. Structural validation is shared by extraction and publication; the database writer does not maintain its own copy of the rules.

The model's trace may contain debugging thoughts, but thoughts are not stored as assertions or used as evidence. Extraction assessment is separate from scientific readiness. No family, graph, estimator, repair or exploration execution is part of this feature.

## Persistence and identity

Extraction publishes the first `context.revisions` record in the same transaction as its provenance. Review and chat update that revision chain through the shared gate. Dataset browsing reads that same chain, including human corrections and relationship changes. Opening a review does not copy or fork the extracted context.

| Table | Stored product |
| --- | --- |
| `context.revisions` | The single dataset/column/relationship knowledge chain, starting with extracted proposals and continuing through explicit review and corrections |
| `context.packs`, `context.column_contexts`, `context.relationships` | Immutable original extraction output and model/source provenance, retained only as audit history |
| `context.reviews`, `context.receipts` | Review drafts and idempotent update receipts referring to the canonical revision chain |

Owner/version composite foreign keys constrain relational parents and source context. The domain validates every statement and participant against the canonical column identities. Dataset and column responses project current `KnowledgeStatement` records, including status and source, instead of reconstructing a different extraction-only shape.

Preparation initializes context only when it is absent. Once context exists, even a changed extractor or profiler identity cannot reset it or create an independent pack. Reads and refreshes perform no model work. Revisions and extraction provenance reject updates at the database boundary. Original source text and CSV bytes remain unchanged; earlier context revisions remain available as history, and chats keep their explicit revision selections.

## HTTP and UI

| Request | Result |
| --- | --- |
| `POST /api/datasets/{id}/context` | Initialize missing context, or return its current revision |
| `GET /api/datasets/{id}/context` | Latest summary and all ordered column summaries, or `not_prepared` |
| `GET /api/datasets/{id}/contexts/{revision}` | Revision-specific canonical study statements, concepts, relationships and original source |
| `GET /api/datasets/{id}/contexts/{revision}/columns/{column}` | One column's statements, applicable study statements and relationships |

All reads use the current server principal. Unknown and other-owner resources return the same 404. Metadata is private/no-store. Missing profile and busy preparation are distinguishable 409s; unusable extraction is 502; dependency failure is 503. Responses are defined once in Python and exported through OpenAPI into frontend types.

The side panel lists columns in source order. Opening a column loads its context. Relationships link to other participant columns. Source buttons highlight exact passages in the unchanged original text, including Unicode and whitespace. Values render as text, never HTML. Refresh only reads saved records; stale requests cannot replace a newer dataset or column view. The column summary list is bounded by ingestion's 200-column limit.

## Consolidation and retirement inventory

The extraction-only dataset reader and the review-time copy into writable knowledge have been removed. There is one current context and one update gate. Extraction output remains immutable audit evidence, not another editable or browsable current context. See [context and chat persistence](09-persistence.md) for revision selection and the derived lane working view. Original uploads, supplied text, fixtures and unrelated checkout work remain preserved.

## Verification

See [Phase 4 evidence](../audits/phase4-context.md). `make test-context` runs extraction and real isolated storage cases; `make test-storage` runs Phases 1–4 with the same harness. `make test-ingestion-mutations` runs the shared deliberate-defect controls. `make container-smoke` runs real browser and replacement/persistence checks with explicit deterministic context fixtures. Real semantic acceptance is separate:

```bash
uv run python -m causal_agent.evals.context --csv data/raw/gov-transfers-uruguay/gov_transfers.csv --source data/context/gov_transfers.md --out .artifacts/context-acceptance/gov-transfers.json
```

That command makes real model calls and records a trace. It never mutates original inputs or application records. A passing scripted suite is not a certificate of semantic truth.
