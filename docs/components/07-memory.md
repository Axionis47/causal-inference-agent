# 07. Dataset memory

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md) · [Evidence and discrepancies](evidence.md)

## Purpose and boundary

Owns durable, sourced knowledge about a dataset. It distinguishes observed data facts, human accounts, model drafts, unknowns and contradictions. Conversation progress, analysis ledgers and displayed transcripts are separate records.

**Durable** means saved so it remains after a conversation or server restart. **Sourced** means a field records where its information came from. A **draft** is a proposed fact awaiting the required confirmation. For example, the model may suggest what a column represents, but the memory keeps that suggestion distinguishable from the person's confirmed account. See [terms](terms.md) for how memory differs from a journal, ledger and checkpoint.

The active chat authority is now [reviewed PostgreSQL revisions](../../causal_agent/server/context/repository.py). `memory/context` owns the pure review/update contracts and projects bound canonical statements through `ops.apply`. The record, catalogue and gate descriptions below apply to that disposable working projection. `store.py` remains for fixture and CLI evaluation inputs, not active chat writes. See [persistence](09-persistence.md).

## Internal parts

| Part | Current implementation | Responsibility |
|---|---|---|
| Records | [Memory, Field, Column](../../causal_agent/memory/records.py) | Addressed field map and versions |
| Write/check operations | [ops](../../causal_agent/memory/ops.py), [checks](../../causal_agent/memory/checks.py) | Validate updates, consistency, data checks and derived fit |
| Catalogue | [catalogue](../../causal_agent/memory/catalogue.py), [fields](../../causal_agent/memory/fields.yaml), [check declarations](../../causal_agent/memory/checks.yaml) | Legal fields, value kinds, requirements and check policy |
| Views and facts | [claims](../../causal_agent/memory/claims.py), [views](../../causal_agent/memory/views.py), [facts](../../causal_agent/memory/facts.py), [matrix](../../causal_agent/memory/matrix.py), [table](../../causal_agent/memory/table.py), [probes](../../causal_agent/memory/probes.py) | Claim projections, dataset evidence, routing matrix and generic probes |
| Persistence | [store](../../causal_agent/memory/store.py) | Load/save, fixture seeding, design snapshots |

Artifact, version and journal storage are covered in [persistence](09-persistence.md).

## Inputs and outputs

`ops.apply(memory, updates)` consumes `Update` records and returns rejection reasons while applying accepted changes. Each `Field` retains value, status, source, source sentence (`said`), reason and evidence. Profile seeding supplies `Column` facts; fit/readiness and handoff consume derived views of memory.

## Contracts

| Boundary | Enforcement | Failure |
|---|---|---|
| Draft/update → field | `ops.apply`: address/field exists, source supplied, value coerces, referenced column exists | Rejection reason; rejected update is not applied |
| Model → knowledge | Model sources may only draft; uncheckable beliefs require a person/document source | Cannot self-confirm or overwrite a confirmed field |
| Verbatim story → memory | Person/document source, verbatim flag and `value == said` | Reader-produced paraphrases cannot become the verbatim story |
| Check → field status | Data and consistency checks retain evidence/refutation | Unknown or contradiction stays visible to admission policy |

Statuses are `empty`, `drafted`, `confirmed`, `refuted`, `unknown`, `contradiction`. Being settled does not mean every endpoint accepts a field. `Memory.set` is a lower-level primitive also used by seeding/checks; external semantic updates go through `ops.apply`.

## Decision authority

Code confirms applicable data facts and controls write policy. Human statements and sourced documents carry account evidence. Model proposals stay drafts. The lane ledger can return source-backed relationship drafts, but absent graph arrows do not create negative facts and confirmed claims cannot be replaced by model proposals.

## Dependencies

Imports common/profile layers. Receives family requirement objects from callers rather than importing family implementations. Supplies desk, handoff, artifact projections and visualization with sourced facts.

## State and storage

`data/memory/<name>/` contains `meta.yaml`, `columns.yaml`, `fields.yaml`, and `said.jsonl`. `version` tracks memory changes; `data_version` tracks the active table. `store.migrate` is the current function name for seeding memory from registered fixture/profile/claims data, not evidence of an old-analysis compatibility guarantee.

## Worked walkthrough

The Reader proposes assignment as a model draft → `ops.apply` records it as drafted → the desk shows its source sentence → a human confirmation passes through the same gate → checks can confirm, refute, or expose a contradiction → readiness reevaluates the endpoint obligations using those statuses.

## Verification

**Code-verified:** write gates, statuses and store layout. **Test-supported:** [memory tests](../../causal_agent/memory/tests/test_memory.py), [matrix](../../causal_agent/memory/tests/test_matrix.py), [facts](../../causal_agent/memory/tests/test_facts.py) and [returned relationship drafts](../../causal_agent/desk/tests/test_relations.py). **Unresolved:** the multi-file store is not documented as an atomic transaction; crash consistency across all memory files was not demonstrated here.
