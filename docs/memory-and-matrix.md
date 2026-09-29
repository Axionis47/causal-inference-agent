# The memory and the matrix

What is known about a file, with a status and a source on every field, and the grid that turns it into a design choice by code.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/matrix-diff-dark.svg">
  <img alt="The matrix before and after three answers: cells that changed are outlined, the instrument family is struck" src="diagrams/matrix-diff-light.svg">
</picture>

## A field

The memory is a map from an address to a `Field` ([memory/records.py:32](../causal_agent/memory/records.py)):

| part | what |
|---|---|
| `value` | the value, or nothing |
| `status` | empty, drafted, confirmed, refuted, unknown, contradiction |
| `source` | `data`, `user:turn:<n>`, `doc:<name>`, `model:<node>`, `code:<rule>` |
| `said` | the person's sentence it rests on, verbatim |
| `reason` | why the source gave this value: a judgement's one sentence, or the rule |
| `evidence` | the check addresses that touched it |

Addresses are `col:<key>.<field>` for a column and `claim:<kind>.<field>` for a claim about the world. The catalogue of kinds
is [memory/fields.yaml](../causal_agent/memory/fields.yaml): grain, sampling, change, assignment, measured (one per column),
missing, then the beliefs no data check can touch: unobserved, exclusion, spillover, trend_continues, cutoff_only, mediator.
Each kind lists its fields, their types and options, the check that can refute it, and the frame a question is written from.

From the checked-in run, one field as the pack renders it:

```
[col:lunch.when] fixed before the change · confirmed · user:turn:3 · said "the district sets it from household income before the exam"
```

## One write path

`ops.apply` ([memory/ops.py:109](../causal_agent/memory/ops.py)) is the only way a field changes. It takes `Update` objects and
returns the reason for every one it refused. The rules:

1. A write needs a source.
2. A `model:` source may only draft. A draft is never trusted by the matrix.
3. A belief is written only from `user:turn:` or `doc:`.
4. A confirmed field changes only by the person, or by `data` or a `code:` rule.
5. Values are coerced to the field's type: text, choice, bool, number, column, columns.
6. A change to `assignment.kind` reopens the fields that depended on it.

Escalation runs one way, said, drafted, confirmed, fact, and only the person or the data escalates. A contradiction is
remembered as one and never overwritten. That is [ADR 0003](adr/0003-a-persons-word-is-evidence.md) and the middle of
[ADR 0006](adr/0006-the-desk-operates-tools-reason-the-memory-escalates.md).

`ops.check` ([memory/ops.py:365](../causal_agent/memory/ops.py)) runs each kind's data check on drafted and confirmed claims. A
failure marks the field refuted and never touches its value. Then the consistency rules run: a column the offer looked at is set
before it, a score is set before, the outcome after, a before-column cannot be moved by the change.

## The matrix

`ops.fit` ([memory/ops.py:401](../causal_agent/memory/ops.py)) calls `table.compute` ([memory/table.py:43](../causal_agent/memory/table.py)),
which builds the grid of every family against every claim kind. Each family says what it needs in its own yaml:

```yaml
# families/adjustment/family.yaml
needs_claims:
  requires: [grain, sampling, change, assignment, measured, missing, unobserved, spillover]
  fits:
    assignment.kind: [lottery, own_choice, third_party, cutoff_rule, date_by_others]
```

One cell ([memory/table.py:16](../causal_agent/memory/table.py)):

| value | when |
|---|---|
| not needed | the kind is not in the family's `requires` |
| unknown | the claim is missing, empty or refuted, or a `fits` field is unset |
| does not fit | a `fits` field holds a value outside the allowed list |
| fits | otherwise |

Beside every cell is `set_by`, the address that decided it, so the chat can say why.

A family is **struck** by its first cell that does not fit, or by a failed probe. **Required** is the union of `requires` over
the surviving families, in catalogue order; belief kinds wait until assignment is settled. **Ready** means every required claim
is settled and at least one family survives ([memory/table.py:78](../causal_agent/memory/table.py)). Drafts do not count.

In the picture above, read from `docs/demo/interview/`, the memory at version 35 has the story in it: diff-in-diff,
interrupted series and synthetic control are already struck on grain, discontinuity on assignment. Two families are in play and
three beliefs are unknown. The person answers three questions. "No hidden factor" fits both; "no instrument" strikes instrument
on exclusion; "no spillover" fits three families. One family stands and the matrix says ready.

## The matrix as a record

`Matrix` ([memory/matrix.py:37](../causal_agent/memory/matrix.py)) keeps each cell with its value, its `set_by`, and the memory
version it took that value at. `update` keeps a cell's version when its value did not change. `diff` lists the cells that moved,
each as `family.kind: before -> after (set_by)`, and the desk writes that line to the journal as a `fit` step whenever a cell
moves ([desk/nodes/interview.py:78](../causal_agent/desk/nodes/interview.py)). The chat cites a cell as `matrix:<family>.<kind>`.

The record is written to `designs/<n>/matrix.json` at hand-off ([desk/nodes/run.py:54](../causal_agent/desk/nodes/run.py)), and
two designs of the same file are compared by their matrices in the brief after the second run.

## On disk

```
data/memory/<name>/
  meta.yaml  columns.yaml  fields.yaml  said.jsonl        the memory: what is known, and every sentence
  analyses/a<k>/journal.jsonl                             what one conversation did, step:<n>
  designs/<n>/  memory.json handoff.json brief.json frame.json decision.json matrix.json record.md
                result.json figures.json record.json      one design run: the snapshot, the pack, the result
  viz/pre/<id>/  designs/<n>/viz/<id>/                    pictures drawn on request
```

The memory outlives every conversation. Graph state is where the conversation is, never the source of truth.
