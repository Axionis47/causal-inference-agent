# 6. The desk operates, tools reason, the memory escalates

Three layers, one rule between them.

## The layers

**The desk is an operator.** One graph, no reasoning of its own. Per message: receive, classify (a claim, a question, a
drawing request, run, done), dispatch to a tool, gate what comes back, write it, ask the next thing. No node of the desk
calls a model except through a tool.

**A tool is one contract and one process.** The Reader turns words into claims with reasons. The Designer turns the
memory and the matrix into a design brief. A lane turns a pack into a result. The Drawer turns a request into a
picture with its numbers. The Explainer turns a question and the material into a cited answer. Each is asked one typed
question and answers with addresses; the desk gates the answer and stores it.

**The memory is what is known.** It outlives every conversation. A design folder is a snapshot of it plus what one run
made. The journal is what one conversation did. Graph state is where the conversation is, never the source of truth.

## The escalation rule

Four levels, one direction, one gate.

1. **said**: every sentence, verbatim, always kept, never interpreted.
2. **drafted**: a tool's reading of a sentence, a note, or the file. Carries the reason. Never trusted by the matrix.
3. **confirmed**: on the person's word or a data check, and only then. Carries the reason.
4. **fact**: what a lane takes from the pack, weighed by code.

A tool can only draft. Only the person or the data escalates. A contradiction is remembered as one, never overwritten.
A relationship between columns is a claim under the same rule, not lane state. Artifacts are evidence with an address;
they never set a field.

## What a decision rests on

The Designer writes one record per lane with these decisions filled, each with the claim addresses it rests on and the
reason. The lane's block is that record. The matrix strikes a family when a claim a decision rests on does not fit. The
interview asks for a claim only because a decision needs it.

Adjustment:

| decision | rests on |
|---|---|
| who is treated, versus whom | assignment.treatment_column, treated_level; sampling.how not by_arm or by_outcome |
| the road: back door, front door, instrument | unobserved.exists; exclusion.exists and column; mediator.exists and column |
| what enters the adjustment set | assignment.depends_on; measured.when before; moved_by_change false; stands_for |
| what is forbidden | measured.when at or after; moved_by_change true; measures_outcome true |
| the target: average or on the treated | assignment.kind; the question's scope |
| whether to run at all | the overlap probe; spillover.possible; missing.why by arm |

Diff-in-diff:

| decision | rests on |
|---|---|
| the groups | assignment.treatment_column, treated_level; level_column when it reached whole groups |
| the periods, long or wide | grain.panel, key_columns; change.date_column, period_value; measured.when for a wide pair |
| whether the comparison holds | trend_continues.believed; assignment.kind; sampling.how not by_outcome |
| the controls | measured.when before or varying; moved_by_change false; not constant within unit or period |
| where to cluster | assignment.level_column; grain.nesting |
| whether to run at all | a pre period exists; spillover.possible; one treated unit or many |

Discontinuity:

| decision | rests on |
|---|---|
| the score and the line | assignment.kind cutoff_rule, score_column, cutoff, treated_side, cutoff_value_treated |
| sharp or fuzzy | a treatment column that is not a strict function of the score |
| whether the line is clean | assignment.movable false; cutoff_only.believed; the score's when before |
| the covariates | measured.when before; moved_by_change false |
| whether to run at all | the density probe on both sides; sampling.how by_side allowed, by_outcome not |

Five claims carry every decision: assignment, change.when with its period column, measured.when and moved_by_change per
column, and the beliefs. The interview is built around those, in that order.

## The interview

A story, a readback, then gaps asked by decision. The person tells what happened, who got it and how that was decided,
what each column is and when it was set. The Reader drafts every claim it can, with reasons. The desk reads the story
back, grouped by the five claims, each line with the sentence it rests on, and asks whether it is right. What is still
open is asked because a decision needs it, and the question says which decision. The person can ask anything or ask
for a picture at any point.

## Where drawn pictures live

A picture is drawn by the Drawer from a request, stored with its code and its numbers, and cited by address:

    data/memory/<name>/viz/pre/<id>/            before any run: keyed to the memory version it was drawn from
    data/memory/<name>/designs/<n>/viz/<id>/    after a run: inside the design it belongs to

Each folder holds `request.json`, `code.py`, `figure.png`, `facts.json`, `artifact.json`. An artifact is deleted with its
dataset or its design and is never read back by code. The chat cites `artifact:<id>` and `artifact:<id>.<fact>`.

## What this replaces

The standalone routing graph (one function now), the figure-pick graph and the per-family pre-run figure registry (the
Drawer), the canned assumption string (the Designer's brief), and the reason a judgement gave that the write path
dropped (it travels with the field).
