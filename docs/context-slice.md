# Context contract

## Ownership and purpose

The Context component owns source material and keyed context revisions.
It records supplied statements, interpretations, review actions, and qualifications separately.
Preparation creates proposals for review.
It does not create an analysis authorization or prove causal claims.

Source: [contracts](../backend/context/contracts.py), [domain operations](../backend/context/domain.py), [service](../backend/context/service.py).
User procedures are in the [user guide](user-guide.md#prepare-and-review-context).

## Input limits

| Input | Default limit |
| --- | --- |
| Source format | UTF-8 `.txt`, `.md`, or a pasted note |
| One source | 1 MiB |
| Sources per dataset version | 10 |
| Combined preparation text | 100,000 decoded characters |
| Statements in one snapshot | 2,000 |
| Preparation model calls | Four, including failed or interrupted attempts |

Preparation rejects excess text instead of silently truncating the context pack.
Original source bytes, names, checksums, and decoded text remain available.
PDF and Word conversion remain deferred.

## Keyed vocabulary

| Namespace | Fields |
| --- | --- |
| `dataset` | `observations`, `population`, `organization`, `study_process`, `measurement`, `question`, `qualifications` |
| `columns.<column_id>` | `definition`, `concept`, `entity`, `units`, `codes`, `valid_ranges`, `missing_codes`, `method`, `measurement_source`, `determined_by`, `collection_time`, `reference_period`, `construction`, `derivation`, `missingness`, `qualifications` |
| `concepts.<concept_id>` | One described concept or event |
| `relationships.<relationship_id>` | One relationship with exact participants and qualifications |
| `sources.<source_id>` | Original source identity and retrieval metadata |

Dataset and column fields contain statement maps.
Several statements can describe the same field.
Qualified or conflicting accounts remain separate.
Empty fields record no supplied statement.
They do not establish that a relationship or property is absent.

The column UI groups the fields into six sections:
Meaning, Values, Measurement, Construction, Missingness, and Qualifications.
These display groups do not replace the stored keys.

## Statements and provenance

Each statement retains its value, basis, citations, qualifications, status, source, and review provenance.
Corrections can also retain `replaces`, `reason`, and explicit `conflicts_with` references.

| Dimension | Values or meaning |
| --- | --- |
| Basis | `source_statement` or `interpretation` |
| Review status | `proposed`, `reviewed`, `unresolved`, `withdrawn` |
| Authorship | Extraction, review, or message provenance |
| Qualification | Scope or condition attached to the statement |
| Conflict | An explicitly declared incompatible statement reference |

Authorship, review status, and factual basis are separate.
Reviewing an interpretation does not turn it into a supplied fact.
The backend does not infer every prose contradiction from `conflicts_with`.

Citations bind source UUID, one-based line, exact quote, and Unicode code-point offsets.
The end offset is exclusive.
The frontend highlights code points rather than UTF-16 positions.
Original CRLF and multibyte source content remain intact.

## Relationships

Participants reference columns, concepts, or events.
Participant roles can identify origin, destination, condition, or intermediate participation.
A multi-party relationship remains one record.

Optional kinds include measurement, derivation, temporal order, membership, assignment, selection, association, and causal influence.
The complete description remains authoritative context text.
A kind label does not replace the description or qualifications.

Directness can be direct, indirect, or unspecified.
An absence claim distinguishes a direct effect from any pathway.
A missing relationship does not establish either form of absence.

The `related_to` view indexes shared participation.
It is derived at retrieval time and omits withdrawn relationships.
It does not create causal arrows.

## Analysis brief

The optional `analysis_brief` belongs to the immutable context content.
It contains the finalized question and exact supporting references.

| Brief field | Meaning |
| --- | --- |
| `question` | The sole finalized question for analysis launch |
| `target` | Bound source columns and meaning references |
| `treatments` | Treatment identities, column bindings, meanings, and changes |
| `comparison` | References defining the requested comparison |
| `population`, `unit`, `timing` | Required study scope references |
| `assumptions` | References to explicit conditional context |
| `optional_unknowns` | Unresolved questions that do not supply facts |
| `source` | Server-recorded application provenance |

Several columns can represent one variable.
The brief does not contain a model-authored readiness flag.
Backend [readiness](analysis-slice.md#applied-context-boundary) derives that result.

Chat can propose the brief through Update context mode.
Only Apply changes saves the proposal and selects its new revision.
Proposal-local `edit:<index>` references resolve within that same Apply transaction.

## Exact retrieval

The shared operation is `read_context(db, version_id, revision_id, keys)`.
It requires an exact revision.
It does not call a model, change state, or select a newer revision.

Example keys:

```text
dataset.study_process
columns.<column_id>.definition
columns.<column_id>.collection_time.<statement_id>
relationships.<relationship_id>
sources.<source_id>
```

A whole-column read includes applicable dataset statements, direct relationships, and participant definitions.
A relationship read includes participant definitions.
Replacement reads retain predecessor references.
Expansion does not recursively traverse the complete relationship graph.

`requested_keys` records the request.
`included_keys` records dependency additions.
Empty slots outside those keys are projection placeholders.
Use the complete revision endpoint to inspect the full saved pack.

## Writes and recovery

Source, preparation, and revision submissions use submission UUIDs.
Preparation and revision writes require the expected parent.
Matching replays return the same saved identity.
Changed submissions and stale parents return conflicts.

A semantic correction creates a new statement identity and withdraws its predecessor in the new revision.
Older revisions remain unchanged.
Reviewing a column does not review its relationships.
An unchanged content and review state does not create another revision.

The worker reserves calls before contacting the provider.
It allows one extraction correction and reassessment cycle within four calls.
Completed steps remain available for recovery.
A stale preparation cannot overwrite a newer pack head.

See [H04 and H05](component-contracts.md#h04-preparation-to-context-publication) for publication handoffs.
See [API reference](api-reference.md) for routes.

## Reference lineage

The original project's keyed vocabulary, citations, and bounded preparation informed this component.
Version-owned PostgreSQL records replace mutable session context.
Preparation does not require a scientific question or profiling workflow.
Old-project files and stored data are not modified or silently imported.
