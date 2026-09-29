# The pack and the addresses

A lane sees one thing: the `Handoff`, projected from the memory at hand-off time and written to `designs/<n>/handoff.json`. Every
claim a judgement makes cites an address the pack resolves. Addresses are the spine of the whole system: the chat, the lanes and the
figures all speak in them, and code checks each one.

## The pack

`Handoff` ([common/contracts/pack.py:314](../causal_agent/common/contracts/pack.py)) carries:

| part | what |
|---|---|
| the decision | family, specialist, outcome, treatment, scope, the assumption bet on, why this family over the others |
| the question | the text, the intent, the design id, the memory version it was projected from |
| the data | the CSV path, the dataset facts, grain, sampling, missing |
| the columns | one `ColumnBrief` per column that matters: its note, when it was set, what set it, whether it fed the treatment, moves the outcome, was moved by the change, measures the outcome, each with provenance |
| how the change happened | `change` and `assignment` as the person settled them |
| what the person believes | the `Belief`s, the addresses left unknown, the contradictions kept |
| the words | every sentence, as `Said`, with the turn it came from |
| the evidence | the probes, the claim snapshot |
| the family block | one `Design` subclass, filled by code from the memory: for adjustment the candidates to adjust for, the forbidden columns, the named instrument or mediator, the hidden confounding the person declared |
| the brief | the Designer's `DesignBrief`: the road, the target, one decision per decision the family lists, the threats, the sentence it bets on |

`render_context` ([pack.py:427](../causal_agent/common/contracts/pack.py)) turns that into the text every lane judgement reads
first, as addressed lines. From the checked-in run:

```
[change:1.note] a six-week test preparation course the school ran. It reached any enrolled student ... The decision or the offer depended on lunch, parental level of education.
[claim:unobserved] nothing outside the file drove both who got the change and the outcome. The offer rule only looked at lunch and parental level of education ... [user:turn:9]
[col:lunch.when] fixed before the change · confirmed · user:turn:3 · said "the district sets it from household income before the exam"
```

The memory wins over the frame wherever both speak. A note is read once, by `mine`, into drafts marked `doc:<name>`; the lanes never
read a note. Where a lane cannot apply something the pack says, it records a `Decline` and goes on: in the checked-in run the scope's
population filter was not in a form the code can apply, so every row was kept and the brief says so.

## The address grammar

| prefix | names | example |
|---|---|---|
| `col:<key>.<field>` | one column's field in the memory or the pack | `col:lunch.when` |
| `col:<key>.profile.<facet>` | a profiler fact | `col:math_score.profile.numeric` |
| `claim:<kind>.<field>` | a claim about the world | `claim:assignment.depends_on` |
| `change:1.note`, `dataset.note` | the change and the dataset as the pack renders them | `change:1.note` |
| `said:<turn>` | one sentence the person said | `said:5` |
| `user:turn:<n>` | the turn a field came from | `user:turn:3` |
| `probe:<family>.<name>` | a probe's result | `probe:adjustment.overlap` |
| `matrix:<family>.<kind>` | one cell of the matrix | `matrix:instrument.exclusion` |
| `step:<n>` | one step of the conversation's journal | `step:10` |
| `design.<...>`, `design.brief.<...>` | the frozen design and the brief | `design.estimand.adjustment_set` |
| `check:<contrast>.<name>` | a design check with its number | `check:completed_vs_none.balance.lunch` |
| `estimate:<contrast>.value`, `.ci`, `.n` | an estimate | `estimate:completed_vs_none.ci` |
| `refute:<contrast>.<name>.<field>` or `placebo:` | a falsification | `refute:completed_vs_none.placebo_treatment_refuter.p_value` |
| `decline:<stage>.<about>` | where the lane disagreed with the pack | `decline:load.scope_population_filter` |
| `figure:<id>`, `figure:<id>.<series>.<i>` | a run figure and one of its marks | `figure:balance_completed_vs_none` |
| `artifact:<id>`, `artifact:<id>.<fact>` | a drawn picture and one of its numbers | `artifact:3f9a1c2e.mean_completed` |
| `interpretation:<contrast>.answer`, `.caveat:<i>` | what the lane wrote | `interpretation:completed_vs_none.caveat:1` |

The one column `key()` and the normalisation live in [common/addresses.py](../causal_agent/common/addresses.py). Every package that
names a column goes through it.

## How a cite is checked

- **In a lane**, `Handoff.resolve` ([pack.py:482](../causal_agent/common/contracts/pack.py)) answers yes only for an address in
  `Handoff.addresses()`: the dataset facets, every column's fields and profile facets, every claim and its fields, every belief's
  fields, the probes, the sentences, the brief's lines. A cite outside that set is an error the judgement is re-prompted with. A
  lane's own artifacts (`check:`, `estimate:`, `refute:`) are added as the run makes them.
- **In the chat after a run**, the material ([desk/material.py:103](../causal_agent/desk/material.py)) is a list of addressed lines
  built from the run record, the memory, the journal and the pictures. The Explainer may cite only what is in it, and every number
  it states must match its address within one percent or appear in that line's text.
- **In a figure**, every `draws_on` address must resolve in the run ([lane/figures.py](../causal_agent/lane/figures.py)); a spec that
  fails becomes a `Decline`.
- **In the routing**, a decision's cites must resolve in the memory or the probes ([desk/nodes/decide.py:179](../causal_agent/desk/nodes/decide.py)).

An address is therefore a promise: whoever states it, code can find the thing and, when it is a number, read it back.

## Building a pack without a conversation

`desk/handoff.py` builds the pack at run time from the memory, the frame and the decision. `forced(...)` builds one with neither, for
tests and evals, and the command line writes it to disk:

```bash
uv run python -m causal_agent.desk.handoff students --family adjustment --outcome "math score" --treatment "test preparation course" --columns "lunch,parental level of education" --question "Did completing the prep course raise math scores?" -o handoff.json
```

The stored packs under `families/<name>/evals/handoffs/` freeze this contract for the evals.

Decided in [ADR 0004](adr/0004-the-pack-is-the-only-input-a-lane-sees.md).
