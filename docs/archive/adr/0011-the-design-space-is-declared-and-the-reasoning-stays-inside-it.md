# 11. The design space is declared, and the reasoning stays inside it

ADR 0008 and 0009 stand. ADR 0010 is proposed beside this one. This decision says what shapes a lane can run and how its
reasoning is held to them, which columns it reasons over, who decides each claim about a column, in what order, and how
the ladder it describes is the one the code runs. It was made when two real runs on one file gave the lane four columns
one day and six the next, judged none of the columns the design rested on, and changed the question's target by a rule
in code; and when an audit found the shapes the lane can run declared in five places, three of them constants, with
nothing proving the space covered and the reasoning finding out it had left it only at the pick, after every judgement
was spent.

## The principle, stated once

A lane reasons forward, within a space declared once, into one experiment shape; the code works that shape out. Six rules
hold it.

**Every column is carried.** Every non-constant, non-identifier column of the file is in the pack and is placed by the
lane. No frame list, no `in_play` cut, no code shortcut leaves one out. A column set aside is a cited ladder line, not
an absence.

**A causal claim is always a judgement.** Whether a column fed the treatment, moves the outcome, was moved by it or
measures it; whether the effect could differ by it; how the treatment was assigned and whether units chose; the road
among several; the adjustment set among several: each is a judgement in a bounded episode. The pack's word enters it as
evidence carrying its status, and a confirmed word is departed from only on a cited data fact. Code copies a word only
about the data's shape (which column, which level, when it was set, an id, a constant) and only when a check bears it
out. ADR 0009's sentence "a judgement where the pack leaves it open, and a fact where it does not" is withdrawn: a fact
is a data check's result, or a word about the data's shape a check bears out, never a word about cause.

**Reasoning over columns is a hierarchy.** Cheap placement of every column; the treatment side; the outcome side; the
graph by code; the road; a deep pass over the adjustment set only.

**The space is declared, and the reasoning is confined to it.** A lane's design space is one file of characteristics
with their supported values or ranges, the rung that decides each, the fact it is read from, and the unsupported values
with the reason a stop carries. The estimator and refuter entries are the allowed tuples over those characteristics; the
rules they already carry are that relation. A rung that decides a characteristic is offered only the values still
runnable given what the rungs below fixed, and its gate refuses any other with the declared reason. Every cell of the
space is proved covered by an estimator and a refuter before any run, or is reachable only through a declared
unsupported value. A file that can reach no cell stops, typed, before any model call. Growing the library is widening the
declaration and adding entries; the proof fails until the new shape is reachable and runnable.

**The ladder is declared, and the graph is built from it.** One file per lane names each rung, who decides it, what it
reads, what it writes, where it exits and whether it may stop. The graph is built from that file, so it cannot differ
from it; the docs and the diagram are generated from it; a proof over it shows every read is carried by the pack or
written by a rung below. Every characteristic of the space names the rung that decides it, and that rung writes it.

**Nothing loops back after a fact arrives.** The evidence a judgement rests on is computed by code before it, as ADR 0009
says, so every decision is made once. No revise loop on a flagged check; no second pick on a failed fit. A flag is
answered at the rung that owns its evidence; a fit failure is a typed stop the person sees. A correction is the person's
move: new words make a new design, which climbs the same ladder.

## What follows from it

**The pack carries every column; the interview still asks only what a decision rests on.** Asking and carrying are split.
The interview's scope stays what a family's decisions rest on; the pack's column list is the file's, filtered by code for
constants and identifiers. The frame's relevant columns are a hint to the interview and nothing to the lane.

**Status rides with every word, and the model cites the address, not the source.** The pack holds one map of facts,
every entry with its value, status, source and sentence: the memory's own shape, ADR 0010's. A rung sees each claim as
the word with its status: the person said it, with the turn; drafted from a note, unconfirmed; never asked. A block the
desk fills from words by code, and a belief known while drafted, go.

**A draft is a draft everywhere.** A drafted timing excludes nothing by itself; a drafted hidden factor is a flag, not a
node; a drafted road is a decline when the graph does not open it, not a stop.

**The hierarchy is column-invariant.** The rungs are fixed; the records are per column. A judgement reads a column
against the pair and the story, never against every other column: what is pairwise is computed by code and read only
for the adjustment set. Above a batch size the breadth rungs run as workers over batches with a code join, and the gate's
coverage rule runs on the join. At eight columns one call per breadth rung; at eighty, four in parallel.

**The target is what was asked.** A rule in code never changes the estimand. Self-selection is a threat the
interpretation must cite, every time the mechanism says units chose.

**The same policy holds in all three lanes.** Each lane declares its space from the design-space table it already has
and loses its loops. What goes and what moves:

| lane | the loops that go | the evidence that moves below the judgement |
|---|---|---|
| adjustment | the revise loop from the assessment to the graph; the second pick on a failed fit | the treatment's rate by column before the treatment side; overlap and balance per candidate set at the road |
| diff-in-diff | the revise loop from the assessment to the controls; the second pick | already below (the trends); the balance of candidate controls before the controls rung |
| discontinuity | the second pick | already below (the density test, the covariate balance, the width table) |

**From ADR 0010, what stands:** status on every fact; drafts to a judgement; declared reads; seals and replay. **What
falls:** `in_play` as the pack's column list; "confirmed means code decides"; a confirmed write of `feeds_treatment`
derived from `depends_on` by code; a fully confirmed column skipping the model; the block filled from confirmed facts.

**Amendments to ADR 0006.** Level 4 reads: a fact is a data check's result, or a word about the data's shape that a check
bears out; a word about cause is evidence to a judgement at every status. The decision tables keep their claims, read
with one sentence: a decision rests on a claim as evidence the lane's judgement weighs with its status; code never
derives the decision from the claim.

## What it costs

About three more model calls per run at eight columns, and bounded prompts at eighty. A space file and a ladder file per
lane, each held by tests: coverage, the graph equal to the declaration, every read carried, no dead key. Every stored
hand-off regenerated. Facts for every carried column at hand-off time, one pass over the file. The adjustment lane first,
then the other two, each change done only after a real-model run shows the ladder lines it promises.
