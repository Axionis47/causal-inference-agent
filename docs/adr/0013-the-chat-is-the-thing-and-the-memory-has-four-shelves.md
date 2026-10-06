# 13. The chat is the thing, and the memory has four shelves

Decided 2026-10-06. ADR 0003, 0006, 0008 and 0012 stand: a person's word is evidence, the desk operates and tools reason, a
judgement is a bounded episode, the trace mirrors the graph. This one says what the conversation is, what the memory holds,
who reads which part of it, and what a tool that looks at the data may and may not do.

## The problem it answers

The desk asked an interview question on every turn from the first reply on. A person who wanted to look at the file first,
or say one thing about it, or ask what a family was, got the next question appended to every answer. The run had no
confirmation: a word in a fixed list launched the lane, and the page's button sent that word. The person's words were read
for intent by a literal match in the node that holds the interrupt, and read for claims by the model in the node after it.
The ready moment rebuilt the decision and the pack inline, duplicating five graph nodes. A picture was the only thing the
desk could make from the file on request, and it lived in two places by when it was made rather than one place by what it
was. Nothing could run code against the data and come back with a table. And the things the chat could cite were assembled
three times over, in the explainer before a run, in the material after one, and in the lane's resolver.

Each of these is one concern living in two homes. The fix is to give each concern one home and say which it is.

## The decision

**The chat is the thing.** One conversation from upload to after the run. Each turn is classified first, into one of four
kinds, before anything else reads it:

- `update`: the message says something about the data, answers what was asked, confirms a draft, or says they cannot say.
  Only this kind reaches the Reader's writes, and before the person has asked to run, what it reads is drafted.
- `explore`: the message asks to look at the data: a picture, a table, a count, a comparison. The exploration agent is called.
- `ask`: the message asks the desk something. The Explainer answers.
- `run`: the message asks for the analysis, or says yes to the design the desk showed.

A cheap decision model decides the kind, with a calibrated confidence; below its threshold the Reader decides the kind in its
own record. No interview question is asked until run intent has been read. Then the desk asks only what blocks the run, one
question per turn, until the matrix says ready; then it shows the design it would run; then the person's word launches it.
Run intent persists until the run or a new question. Only the run needs a confirmation. Nothing else asks permission.

**The memory has four shelves**, by what made the thing, not when:

| shelf | holds | address | written by |
|---|---|---|---|
| the understanding | `col:` and `claim:` fields with status, source, sentence, reason | `col:`, `claim:` | `ops.apply` alone |
| the artifacts | what the exploration agent made: facts, a table, a picture, the script, the summary | `artifact:<id>`, `.<fact>`, `.table`, `.figure` | the desk, on receipt |
| the experiments | one folder per design run: handoff, pack, result, record, figures | `estimate:`, `refute:`, `check:`, `design.`, `figure:` | the desk and the lane |
| the journal | the steps, in order, across the other three | `step:<n>` | `record` alone |

Every item carries the memory version and the data version it was made at. Version 0 of the data is the uploaded file; a
later agent may write version 1 and up, never in place. An item is pullable by its address from any later turn, and a shelf
is listable by kind, design and version. The chat's material is a view over that pull, not a third copy of the rules.

**One reader per shelf.** The interview reads the understanding and the probes. The handoff builder reads the understanding
only. The lane reads the handoff only. The exploration agent reads the table it is sent and, after a run, that run's lines.
The chat cites every shelf through the material, under the one-percent rule. The bridge between the artifacts and the
understanding is narrow: a fact the agent computed may confirm or refute a field as a data check would; the model may draft
from one; nothing from an artifact confirms a belief.

**The exploration agent** is a separate agent reached over A2A, with a card, in its own process, reading only what it is
sent. Inside it is a bounded episode with one tool, `run_code`, that runs a script in the sandbox and returns what it
printed and what it wrote. It answers one typed record: a summary, the step whose files are the answer, the facts it stands
behind, whether it drew, which table it wrote. Code gates the record: the facts exist and are numbers, the picture exists
when claimed, the table parses, every number in the summary is within one percent of a fact or of what it was told, and
before a design has run no sentence and no fact joins the outcome with the treatment. Before that freeze the table it is
sent has one of those two columns taken out, so the number cannot be computed, and the gate catches it being written from
memory. It never estimates the effect. It writes nothing but its own output folder. The drawing tool is one of its products.

**The invariants a test holds.** Nothing is cited without an address. Nothing reaches the understanding except through
`ops.apply`. Every artifact names what it was made from. Every model call, file write, re-prompt and process goes through
the chokepoints ADR 0012 names.

## What it costs

A process per exploration, started on demand and stopped after. A decision model loaded in the API process. The Reader's
prompt grows by one paragraph. The shortcut that confirmed every open draft on the word "run" goes, and with it the test
that proved it: the readback is now the first thing asked under run intent, one turn. The interview stays plain text until
the questionnaire lands. The ready moment runs through the routing nodes, so the trace shows `fit`, `decide`, `gate` and
`design` under it and nothing is built twice.

## How it is read

In the chat: no question until you ask to run. In the trace: a `decide` span first on every turn, then `read:` only on an
update, `explore` with the agent's episode nested under it, `gate:kind` and `gate:reply` where a record was refused. In the
memory folder: `artifacts/<id>/` beside `designs/<n>/` and `analyses/<k>/`. At the ready moment: `fit`, `decide`, `gate`,
`design` and `judge` under it, the judge's verdict in the reply's last lines and beside the fields on the page, and an edit
on the page arriving as `address = value` lines that no model reads. Built since: the design guideline for the page
([design.md](../design.md)) and the memory hygiene (every status change moves the version); the handoff view with the run
confirmation and the judge. Later, in order: the questionnaire; the data agent that writes new table versions with lineage.
