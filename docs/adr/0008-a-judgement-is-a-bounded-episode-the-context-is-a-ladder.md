# 8. A judgement is a bounded episode; the context is a ladder

ADR 0002 stands: code computes every fact, the model judges, the harness gates. This decision changes the shape of a
judgement and the order of the context it reads. It replaces the per-column relate call that every lane once made.

## The problem it answers

No stage held the whole story. A lane related one column at a time, in a one-shot call that saw the pack as a flat map
and no other column. It could not notice what a person notices when reading a problem in order: that the pair comes
first, that the mechanism that set the treatment decides what a column can be, that two columns measure one thing, that
a risk lives in the story rather than in any field. Checkability per step had been bought at the price of the reasoning.

## The decision

**The context is assembled in the order the reasoning reads it, before the run.** The pack renders the story whole, then
the pair, the mechanism, time, the columns with their relations to each other, the rows, the hidden factors, where the
effect could differ, the threats. Relations between columns are claims the interview asks because a decision rests on
them, and data facts about the columns in play are computed by code and ride with the probes. The person reads that
context back before saying run.

**A judgement may be a bounded episode.** The model reads its material, may ask code for facts about the data through a
fixed set of read-only tools, sees every fact it asked for as an addressed line, and answers in one typed record that
code gates. The budget of tool calls, the tool set and the shape of the answer are fixed by the caller, so the leash is
short and visible. One rule is code, not prompt: no tool joins the outcome with the treatment before the design is
frozen. Choosing a role or a modifier by peeking at the effect is the one thing this structure makes impossible.

**The design is climbed as a ladder.** Each lane lists its rungs in the order an analyst reads its design. A rung is
code where the pack settles it and an episode where it does not. A rung reads the rungs below it and never the ones
above. Every rung's lines have addresses (`ladder:<rung>.<field>`), so a higher rung, the report and the chat after can
cite them. The rungs that place columns place them all together.

**The lane never asks the person.** The matrix is the contract: what a family requires is derived from what its
decisions rest on, so ready means the inputs of every rung are settled before the lane starts. A rung that would not
guess says `unsure`; that is a flag the assessment answers and the interpretation cites, never a question. An `unsure`
on a claim the interview could have settled is also a decline that names the line `family.yaml` should gain, so the
next interview asks first. The one question a lane still asks is the one no interview could foresee, because it depends
on the lane's own graph: the road question, once a hidden factor closes the back door.

**Every lane hands back the risks of its own design.** A threats rung names them by code from the pack and from the
rungs below, and a judgement rung names the ones only the story can raise, from a fixed list and cited. Each is a flag.

## The three ladders

| adjustment | diff-in-diff | discontinuity |
|---|---|---|
| the pair | who got the change | the score and the line |
| the mechanism | the clock | the shape of the two sides |
| time | the shape of the panel | whether the line is clean |
| the pre-treatment roles, together | whether the comparison group is a fair stand-in | the covariates, together |
| the post-treatment roles, together | the controls, together | where the effect at the cutoff could differ |
| the graph | where the effect could differ, by a unit trait | the threats |
| the road | the threats | the window |
| where the effect could differ | where the errors cluster | |
| the threats | | |

Each ladder lives in its own family. What every ladder shares, the records that mean the same thing in every design and
the code that turns threats and unsure items into flags, lives in the harness.

## What it costs

An episode is more than one model call, and a rung that places every column at once reads a longer prompt than a call
per column did. The budgets in each lane's `checks.yaml` bound the first; the tests script every episode through the
same fake as before, so the second costs nothing in checkability. No rung names a column or a dataset; the design is
spoken of in general terms and the columns come only from the pack, as the core's own test enforces.
