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
is [memory/fields.yaml](../causal_agent/memory/fields.yaml): the story, then grain, sampling, change, assignment, measured (one
per column), missing, then the beliefs no data check can touch: unobserved, exclusion, spillover, trend_continues, cutoff_only,
mediator. Each kind lists its fields, their types and options, the check that can refute it, and the frame a question is
written from.

A column's claim says more than what it measures and when. It also says how the column stands to the pair and to the other
columns: whether it fed the treatment, moves the outcome, was moved by the change, measures the outcome, is the same thing
as another column (`same_as`), sits inside a coarser one (`nested_in`), stands for something outside the file
(`stands_for`), and whether the effect could differ by it (`may_modify`). The mechanism can name an offer column and an
uptake column when the offer and the taking are two columns. A field of type `column_or_none` takes a column's name or the
word none, so "there is no such column" is a settled answer.

The `story` kind holds the person's account as they gave it, one verbatim field. Only the desk writes it, with the words
themselves, on the story turn or from the note that was mined; the Reader never sees it as a field to fill.

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
5. Values are coerced to the field's type: text, choice, bool, number, column, columns, column or none.
6. A change to `assignment.kind` reopens the fields that depended on it.
7. A verbatim kind is written only by the desk, and only with the words themselves.

Escalation runs one way, said, drafted, confirmed, fact, and only the person or the data escalates. A contradiction is
remembered as one and never overwritten. That is [ADR 0003](adr/0003-a-persons-word-is-evidence.md) and the middle of
[ADR 0006](adr/0006-the-desk-operates-tools-reason-the-memory-escalates.md).

`ops.check` ([memory/ops.py:365](../causal_agent/memory/ops.py)) runs each kind's data check on drafted and confirmed claims. A
failure marks the field refuted and never touches its value. Then the consistency rules run: a column the offer looked at is set
before it, a score is set before, the outcome after, a before-column cannot be moved by the change.

## The matrix

`ops.fit` ([memory/ops.py:401](../causal_agent/memory/ops.py)) calls `table.compute` ([memory/table.py:43](../causal_agent/memory/table.py)),
which builds the grid of every family against every claim kind. What a family requires is derived from its decisions: every
claim kind a decision in its `family.yaml` rests on. So ready means the inputs of every rung of the lane's reasoning are settled
before the lane starts. The one exception is declared beside the fits: the kinds the lane settles with its own question when
its own graph needs them, which the interview asks too but which never block.

```yaml
# families/adjustment/family.yaml
needs_claims:
  lane_settles: [exclusion, mediator]
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

**What is asked, and in what order.** A family's decisions in `family.yaml` say which addresses each rests on. The loader
turns that into what the family asks, and `ops.open` lists those fields when they are empty, beside the required ones. A
relation a decision rests on never blocks readiness: the person may say run without it, and the lane then reads it as
unknown. A field with `asked_when` is asked only when its condition holds: a column fixed before the change is not asked
whether the change moved it; a belief's column only once the belief exists; a cutoff only under a cutoff rule. The ask
composer ([desk/nodes/interview.py](../causal_agent/desk/nodes/interview.py), `first_gap`) walks the surviving families'
decisions in the order each lists them, what blocks readiness first, so a relation never stands in front of a required
claim, and asks every open field under one decision in one turn.

**Data facts before the run.** At every fit and at hand-off, [memory/facts.py](../causal_agent/memory/facts.py) computes,
by code, for the columns in play: each candidate by arm, its association with the outcome, the associations among
candidates, which pairs are redundant or nested, and every column's timing in one line. Each is a `probe:data.<name>` with
a number and no verdict, so the pack, the chat before the run and every lane judgement see the numbers a reasoning would
otherwise have to ask for. One rule, in code: no fact joins the outcome with the treatment.

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
