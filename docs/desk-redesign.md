# The desk, final design: memory, the frozen context pack, the hand-off point, the run, and the chat after

Status, 18 Sept: stages 0 and 1 landed (the first `Handoff`, the one builder, the lanes reading the pack, two new claims). Stage 2
landed the same day: `profile/` and `memory/` split out of `intake/`, the field catalogue final, one map from address to field with a
status and a source on every entry, roles as a view, the gated write path, the consistency rules, `open`, the store and the migration. The router is folded in too:
routing runs on the memory (`desk/route.py`), the per-family model verdicts are replaced by the fit over the memory, and the pack is
projected from a memory. Stage 3 landed on 19 Sept: the profile facts (`binary_like`, `bounds`, `role_hints`, the index line, the
disk cache), the pack with how every field was settled and the lanes reading the person's words, `identification_allowed` gone,
`viz/` with the figure contract, the three pre-viz functions, the viz subgraph, and `Figure.tsx`. Stage 4 landed the same day: `desk/graph.py`
replaces the interview and the chat; the question comes first and is validated against the file; one question per turn from what the
surviving families need, composed by code; `infer` gated by `apply`; contradictions; the lane in its own process on `designs/<n>/`;
CSV-only upload; `intake/` and `chat/` deleted. Stage 5 (convince with a figure, then-and-now designs) next.

## Context

The desk asked thirteen readback questions, stored the beliefs that matter as bare booleans, routed from a note the beliefs never
reached, and could not talk once "everything was settled". The fix is not more questions or a cleverer router. It is one memory of
the dataset, built through a conversation, from which every run's context pack is projected, which the lanes reason from with
agency, and which the person can keep editing after a run: what if this had not happened, here is something I forgot.

Principles that hold everywhere:

- Facts from code, judgements from one gated model call each, declarations in YAML. Never patch code per dataset.
- The person can say anything. Whether it changes the memory is the agent's gated judgement, weighed against the file's facts and
  what is already settled. A statement is evidence with a source, never an automatic write.
- Vague means a needed field is empty or drafted. Vague raises the next question. The hand-off point is when nothing a surviving
  family needs is vague.
- Every run's payload comes from the memory and nothing else. Using it well is each lane's job.
- Every artifact has an address. Nothing is stated that the run did not produce.

## 1. The memory

One memory per dataset, persisted under `data/memory/<name>/`, read and written only through `causal_agent/memory/`.

### The map: `fields.yaml`, one field per address

Everything known, or asked and not known, is one entry keyed by its address. Every entry carries a `value`, a `status`, a `source`,
the person's sentence it rests on (`said`), and the checks that touched it (`evidence`). Nothing else is stored about it. An
address absent from the map has not been spoken of: a wide file is a small memory until someone talks about a column.

```yaml
col:lunch.meaning:      {value: "standard or free/reduced; the district's low-income flag", status: confirmed, source: user:turn:2, said: "...verbatim..."}
col:lunch.stands_for:   {value: household income, status: drafted, source: model:infer}        # groups columns that measure one thing
col:lunch.proxy:        {value: proxy, status: drafted, source: model:infer}                    # exact | proxy | derived; derived_from: [cols]
col:lunch.when:         {value: before, status: confirmed, source: user:turn:4, evidence: [check:col:lunch.before_is_fixed]}
col:lunch.set_by:       {value: the district, status: confirmed, source: user:turn:2}
col:lunch.moved_by_change:  {value: false, status: drafted, source: model:infer}
col:lunch.measures_outcome: {value: false, status: drafted, source: model:infer}
claim:assignment.kind:  {value: own_choice, status: confirmed, source: user:turn:3, said: "..."}
claim:assignment.depends_on: {value: [lunch, parental level of education], status: confirmed, source: user:turn:3}
claim:unobserved.exists: {value: false, status: confirmed, source: user:turn:5, said: "the counsellor had nothing else to go on"}
```

Column fields: `meaning`, `stands_for`, `proxy`, `derived_from`, `when`, `set_by`, `moved_by_change`, `measures_outcome`. Dataset
fields: the kinds listed under the dataset record below. The catalogue `memory/fields.yaml` declares every field, its type, its
options, and which families need it.

Field statuses: `empty`, `drafted` (a model wrote it), `confirmed` (the person's word, judged and accepted), `refuted` (a data check
contradicts it), `unknown` (the person cannot say), `contradiction` (refuted twice, or two confirmed fields disagree). Sources:
`data`, `user:turn:<n>`, `doc:<name>`, `model:<node>`, `code:<rule>`.

### The file's facts: `columns.yaml` and `meta.yaml`

The profiler's facts on every column (`kind`, `distinct`, `nulls`, `top_values`, `varies_over`, `numeric`, `switch`) by key, and on the
dataset (rows, grain, time coverage, entity summary). Code, turn 0, never asked, never written to. Addressed as
`col:<key>.profile.<facet>` and `dataset.profile.<facet>`.

### Views, never stored

Roles (outcome, treatment, depends_on, score, unit, time, group, instrument, mediator) are computed from the dataset fields and the
question every time they are needed. So are what is in play, what is open, the family fit, and the pack. Change the map and every
view follows; nothing derived is written back to be kept in step.

Cross-column context is a short named list, never every pair: the grain and nesting (dataset fields), one concept measured by
several columns (`stands_for` and `proxy`), the assignment set (`assignment.depends_on`), every column against time (`varies_over`,
computed), one covariate driving another (optional, drafted from meanings, never asked).

### Dataset fields

### The person's words: `said.jsonl`

Every turn verbatim with its number. A field's `said` points at the sentence that set it. This is what a lane reads when it weighs
an inherited claim.

### Designs: `designs/<n>/`

A design is a frozen snapshot: `memory.json` (the map and the facts as they stood), `frame.json` (the question read), `decision.json`
(family, assumption, why, over), `handoff.json` (the pack as sent), and `run/` (the lane's artifacts). Designs are numbered per
dataset. A what-if forks the memory into design n+1 without touching design n; a correction edits the memory and makes design n+1
the same way. Then-and-now compares two designs by address.

### Operations: `causal_agent/memory/`

Code, each a function with tests:

- `seed(profile)`: facts for every column, `missing` when there are no gaps, `grain.panel` from the entity summary. No field is written
  until someone speaks; candidate roles (binary → treatment candidate, dated → time, id → unit) are profile facts, stage 3.
- `apply(memory, updates, gate)`: the one write path. An update names a field, a value, a status, a source, the said. The gate refuses a
  write with no source, a write to a confirmed field not from the person or a data check, a belief not from the person, a value
  outside the field's options, a column not in the file.
- `check(memory, df)`: the data facts. Existing `checks.py` plus the consistency rules: a `depends_on` column is `before`; a score is
  `before`; the outcome is `after`; a column marked `moved_by_change` is not `before`; the treated level appears in the column. A failed
  rule sets `refuted` on the field with the check address as evidence. It never overwrites a value.
- `probe(memory, df)`: the family probes. `probes.py` as today, plus overlap and by-group-over-time shared with the pre-viz.
- `roles(memory, outcome, treatment)`: a view, `{key: role}`, from the dataset fields and the frame. Never written.
- `fit(memory)`: the family grid (`table.py`), `required` per survivor read from `families.yaml`, computed per column only for columns
  whose role is in play.
- `open(memory)`: the vague fields a survivor needs, in priority order. This is the question engine's input.
- `snapshot(memory, n)`, `fork(memory)`, `render(memory)`: the design files and the text every prompt sees.
- `infer` is the one judgement over memory: given the message, the open fields, and the fields in play, return updates with
  reasons and the sentence they rest on. `apply` gates them. It also returns what the answer implies (a lottery implies no hidden
  factor) and what it contradicts.

`memory/` absorbs `intake/knowledge/claims.yaml` (renamed `fields.yaml`, the field catalogue with options, hints, frames, and the
family needs) and `intake/interview/{contracts,table,checks,probes}.py`. The profiler moves to `causal_agent/profile/`.

## 2. The context pack, frozen

`common.contracts.Handoff`, projected from one design's snapshot by `desk/handoff.py: build()`. It changes from the stage-1 shape
in four ways: `identification_allowed` is dropped (the lane finds what identifies); every brief field carries its status, source,
and said; the column's new fields are in the brief (`stands_for`, `proxy`, `measures_outcome`, `moved_by_change` renamed from
`affected_by_treatment`, and `role` from the roles view); and the pack carries the design number and the memory version.

```
Handoff
  question, intent, scope                     the question as read
  family, specialist, chosen_assumption, why, over                 the decision
  design_id, memory_version, pack_name, csv
  dataset: grain, nesting, sampling, missing, change, assignment    each field with status and source
  beliefs: {kind: Belief(value, what, why, column, status, source, said)}
  treated_level, control_level
  columns: [ColumnBrief]                      only columns whose role is in play; each field with status, source, said
  probes: [Probe]
  said: [Said]                                the sentences behind assignment, change, beliefs, and each in-play column
  unknowns: [address]                         fields the person could not settle
  contradictions: [address]                   fields the data refuted and the person kept
  design: AdjustmentDesign | DidDesign | RdDesign      a projection by code, for the lane's convenience, never a constraint
```

The family block stays what it is today minus `identification_allowed`. It is derived; a lane may disagree with it and say why.

Rendering keeps every address the lanes cite: `col:<key>.note` is the meaning, `col:<key>.when`, `.set_by`, `.moved`, `.measures_outcome`,
`.role`, `.profile.<facet>`; `claim:<key>[.<field>]`; `probe:<family>.<name>`; `change:1.note`; `dataset.*`; `said:<turn>`.

## 3. The hand-off point

Ready when, for at least one surviving family: every field it needs on every in-play column is settled (`confirmed`, `unknown`, or
`refuted` and then re-answered), every dataset field it needs is settled, no `contradiction` stands, and its probes pass. Then:

1. `decide`: one survivor is chosen by code; several, one gated call with `prefer_over`, citing memory addresses.
2. `convince`: the family's `Point` goes to the viz tool; the message says the design in one sentence, the assumption in the person's
   words, the evidence with addresses, the figure, and the struck families with one reason each. The right strip shows the grid.
3. The person accepts, asks, or changes something. A change goes through `infer` and `apply`, back to `fit`.
4. `handoff`: snapshot design n, write `handoff.json`, run.

## 3a. The hand-off point, as it looks

The students file, design 1, exactly what the adjustment lane receives. Every line has an address. Every field says how it was
settled. Nothing here was guessed by the lane.

```
HANDOFF design 1 of students · memory v7 · family adjustment → dowhy

QUESTION   Did completing the prep course raise math scores?
           intent effect_of_change · contrast switch · target average
DECISION   adjustment. Assumption: nothing beyond lunch and parental education decided who got a place. [decision.assumption]
           over diff_in_diff: one exam per student, nothing before the course [claim:grain.panel]
           over discontinuity: places were offered by a rule on two labels, not a line on a score [claim:assignment.kind]
           over instrument: no column pushed students in without touching marks [claim:exclusion]

DATASET    [dataset.note] one row per student, May 2026 exam, one school; every sitter kept; no gaps
           [claim:grain.row_is] one student's results                     confirmed · user:turn:1 · said:1
           [claim:grain.panel] false                                      confirmed · data
           [claim:sampling.how] whole                                     confirmed · user:turn:1 · said:1
           [claim:missing.why] none                                       confirmed · data
           [dataset.profile.rows] 1000 rows, 8 columns

CHANGE     [claim:change.what] a six-week prep course                    confirmed · user:turn:2 · said:2
           [claim:change.when] 1 March to 12 April 2026, before the exam  confirmed · user:turn:2
           [claim:assignment.kind] own_choice                             confirmed · user:turn:3 · said:3
           [claim:assignment.rule] "offered first to free-lunch students and to those whose parents hold no degree, then anyone who asked; taking it was up to them"
           [claim:assignment.depends_on] lunch, parental level of education   confirmed · user:turn:3
           [claim:assignment.treatment_column] test preparation course  confirmed · code:frame
           [claim:assignment.treated_level] completed                    confirmed · user:turn:3
           [claim:assignment.movable] true                               drafted · model:infer

BELIEFS    [claim:unobserved] nothing outside the file drove both        confirmed · user:turn:5 · said:5 "the counsellor had nothing else to go on"
           [claim:exclusion] no such column                              confirmed · user:turn:5
           [claim:spillover] students sit alone                          confirmed · user:turn:1 · said:1
           [claim:mediator] not asked (no hidden factor)                 empty

COLUMNS    (6 in play, 2 out of play: reading score, writing score measure the outcome)

[col:test_preparation_course] role treatment
  [col:test_preparation_course.note] whether the student completed the course     confirmed · user:turn:2
  [col:test_preparation_course.when] at the change                                  confirmed · code:role
  [col:test_preparation_course.profile.top_values] none=64%, completed=36%

[col:math_score] role outcome
  [col:math_score.note] the exam mark, 0 to 100                                     confirmed · user:turn:1
  [col:math_score.when] after the change                                            confirmed · code:role
  [col:math_score.profile.numeric] min 0, p50 66, max 100

[col:lunch] role depends_on
  [col:lunch.note] standard or free/reduced, the district's low-income flag         confirmed · user:turn:4 · said:4
  [col:lunch.stands_for] household income (proxy)                                   drafted · model:infer
  [col:lunch.when] before the change                                                confirmed · user:turn:4 · check:col:lunch.before_is_fixed passed
  [col:lunch.set_by] the district                                                   confirmed · user:turn:4
  role depends_on                                                                    view · claim:assignment.depends_on
  [col:lunch.moved] false                                                           confirmed · code:when=before
  [col:lunch.profile.by_arm] free/reduced 29% of completers, 40% of the rest

[col:parental_level_of_education] role depends_on           (same shape)
[col:gender] role candidate
  [col:gender.when] before the change                                               confirmed · user:turn:4 (tick)
  [col:gender.moved] false                                                          confirmed · code:when=before
  [col:gender.measures_outcome] false                                               drafted · model:infer
[col:race_ethnicity] role candidate                        (same shape)

PROBES     [probe:adjustment.arms] pass: 358 completed, 642 not; floor 20
           [probe:adjustment.overlap] pass: every lunch × parental-education cell has both arms; smallest cell 14 and 23

SAID       [said:1] "Each row is one student's results... every student who sat is in the file"
           [said:3] "The counsellor offered places first to... then anyone who asked. Whether they took it was up to them."
           [said:5] "The counsellor had nothing else to go on, honestly."

UNKNOWNS   none      CONTRADICTIONS   none

FAMILY BLOCK (derived by code; the lane may disagree and say why)
  candidates to adjust for: lunch, parental_level_of_education, gender, race_ethnicity
  never adjust for: reading_score, writing_score
  hidden confounding: no, per the person · uptake: voluntary after an offer · target: average
```

What the lane does with it: draws the graph with lunch and parental education as parents of both course and marks, gender and
race as parents of marks (the two drafted fields it may keep or revise, citing the brief), runs DoWhy identification, gets a backdoor
set, checks overlap and balance, picks an estimator, refutes, and writes the reading. It asks nothing, because nothing is vague.

What it would look like with one thing vague: `[claim:unobserved] something outside the file drove both · confirmed · said:5 "the
counsellor knew the kids"` and `[claim:mediator] empty`. The lane's identify finds no backdoor. Its assess judgement returns an ask,
one question keyed to `claim:mediator`, the desk shows it, the answer goes through `infer` and `apply`, design 1 is re-projected and
the lane resumes. That is stage 9; until then the lane reports backdoor with a sensitivity caveat that names the hidden factor.

## 3b. Updates after a run, as they look

Every update is an edit to the memory through `infer` and `apply`, then a new design. The run payload never comes from anywhere else.

**A correction.** "Actually lunch status was recorded in June, after the exam."

```
infer   → col:lunch.when = after · user:turn:9 · said:9   (contradicts confirmed 'before'; the reason is given, so it is an update)
check   → consistency: depends_on implies before → col:lunch.when refuted · evidence check:col:lunch.depends_on_before
desk    → "Lunch decided who got a place, so it was set before the offer. If June is when it was written down but the status was
           the district's from earlier, it is 'before'. Which is it?"        one question, keyed to col:lunch.when
person  → "written in June, but it was the district's flag from September"
infer   → col:lunch.when = before (kept) · col:lunch.set_by = "the district, from September" · user:turn:10
fit     → unchanged · no new design
```

**A what-if.** "What if it had not been voluntary, what if the counsellor just assigned it?"

```
fork    → design 2 = memory v7 with claim:assignment.kind = third_party · source user:turn:11 (what-if) · movable false
fit     → adjustment still survives; the family block's uptake becomes 'by a rule'
run     → design 2 runs; then-and-now:
           [run:1.effect] 5.62 [3.69, 7.55] · [run:2.effect] 5.58 [3.60, 7.51] · overlap unchanged
desk    → "Same design, same set; the answer barely moves. What changes is the caveat: with an assigned course the weak case is a
           rule with no one going the other way, and overlap says that did not happen here [probe:adjustment.overlap]."
```
Design 1 is untouched. The designs strip shows both with the one field that differed.

**New information.** "I found the counsellor's roster; the order was random. It is the offer_rank column."

```
infer   → claim:exclusion = true, column offer_rank · user:turn:12 · said:12
        → col:offer_rank.when = before; the roles view now reads offer_rank as the instrument
fit     → instrument now survives beside adjustment; prefer_over decides, or decide asks one question
design 3 → the lane sees exclusion with a column; identify returns an instrument estimand; the pick chooses it (stage 9) or, until
           then, backdoor with the instrument noted in the caveat
```

**General rule for every lane.** The pack is the whole payload. A lane that needs something not in it either derives it from the
table (a fact, addressed) or asks for it (stage 9). It never reads the CSV's neighbours, a note, or an earlier run.

## 4. Pre-viz

`causal_agent/viz/`: `spec.py` (`FigureSpec`, `Point`, `Figure`), `previz/{adjustment,diff_in_diff,discontinuity}.py`, `graph.py`
(the viz subgraph: `pick` judgement, `render` fact, `check` fact). The desk sends a `Point` to make, never a figure name; the tool may
answer `made=False` with why. Shown unasked once, at the ready moment. Otherwise on request, before or after the run. Every pre-viz
function returns the spec and the probe number from the same computation.

- adjustment: overlap of each `depends_on` column by arm.
- diff_in_diff: the outcome by group over time with the change marked.
- discontinuity: score density around the cutoff and the outcome means per bin.

`Figure.tsx` renders the spec kinds (bars, lines, points, density, interval) inline in a message (a turn carries an optional
`figure`) and, at stage 8, in the inspector. The layout is pure code in `web/src/figure.ts`, tested. `viz/figures.yaml` declares
what each function shows, when it makes a point, and what it needs settled; `candidates` is code over that, `pick` a judgement only
among several.

## 5. The run

`desk/pipeline.py` runs `python -m causal_agent.desk.pipeline designs/<n>/handoff.json --json-file out.json`: the lane subgraph
alone, on the pack alone. No router in the subprocess. The lane reads `Handoff` and the CSV, nothing else.

Each lane's design changes only as much as it takes to use the pack well:

- `load` takes the CSV path, the unit, the cluster, and the sample rule from the pack.
- The judgements that the pack settles become facts with the same checks: the contrast from `treated_level`, the group and the
  period from the panel block, the score and cutoff from the cutoff block. The model is asked only if the fact fails its checks.
- `relate` drafts only the fields the record left open, and marks them `model:relate`. Where the field says `before` and the
  roles view says `depends_on`, the edge is not a judgement.
- The beliefs and the said are in the material every assessment and interpretation reads. A lane may decline an inherited field,
  citing the check that contradicts it, and the run record says so.
- DoWhy: the adapter reads every estimand DoWhy returns (backdoor, frontdoor, instrument); the estimator pick chooses among what
  identifies, citing the belief it rests on; with a hidden factor kept and only backdoor available, a sensitivity refuter runs and
  the caveat says so. This is the lane's own follow-on and does not block the hand-off point.

## 6. Artifacts of a run, under `designs/<n>/run/`

| file | what | addresses |
|---|---|---|
| `handoff.json` | the pack exactly as the lane received it | every `col:*`, `claim:*`, `probe:*`, `said:*` |
| `record.md` | the decision: families in play, struck and why, the assumption | `decision.*` |
| `design.json`, `design.md` | the lane's frozen design: graph or panel or score, estimand, estimator, checks, refuters | `design.*`, `check:<c>.<name>` |
| `table.csv` and the lane's shaped tables (`panel.csv`, `canon.csv`, `bins.csv`) | what was fitted | none, files |
| `artifacts.json` | estimates, refutations or placebos, interpretations, feasibility | `estimate:*`, `refute:*` or `placebo:*`, `interpretation:*`, `feasibility.*` |
| `figures.json` | the lane's post-viz specs (stage 8) | `figure:<id>[.<series>.<i>]` |
| `report.md` | the reading, the caveats, the model's thoughts | none, prose |

`desk/material.py` renders all of these as lines with addresses and a table of numbers. It is what the chat cites after the run.

## 7. The chat after the run

One free-flowing conversation over the memory and the designs. `turn` is one gated judgement that classifies the message and
answers from the material:

- `answer`: what was found, why this design, what a check means. Cited. Numbers attached to addresses.
- `explain`: the same in the question's own terms, technical name second. `Interpretation.explained` per check, written by the lane.
- `show`: a `Point` to the viz tool; the figure comes back inline or `made=False` with why.
- `revise`: "I forgot, lunch was set after the offer". `infer` and `apply` on the memory; `check` and `fit` again; design n+1 is
  snapshotted and run; then-and-now against design n.
- `what_if`: "what if it had not been voluntary". `fork` the memory into design n+1 with the changed field; design n stays. The
  reply says what changed and what the new run found.
- `requestion`: a new causal question on the same memory; a new frame, a new design.
- `chat`: anything about the data or the process that changes nothing. Answered from the memory and the profile.
- `done`.

The right strip lists designs side by side, each with its family, effect, flags, and the fields that differed. The transcript keeps
every turn so a later `said` can point at it.

## 8. Organisation

```
causal_agent/
  common/       contracts.py (Handoff, ColumnBrief with statused fields, Belief, Said, Probe, family blocks, lane artifacts), llm.py, addresses.py
  profile/      profiler.py, cards.py, cache.py
  memory/       fields.yaml, checks.yaml, records.py (Field, Column, Memory: the map), ops.py (seed, apply, check, probe, fit, open, roles as a view), checks.py, probes.py, table.py, store.py (data/memory/<name>/ layout), tests/
  desk/         graph.py, state.py, contracts.py, nodes/{journey.py, frame.py, decide.py, after.py}, prompts/, handoff.py, material.py, pipeline.py, route.py, tests/
  viz/          spec.py, graph.py, previz/, postviz/, registry.py, tests/
  knowledge/    families.yaml (needs, fits, convince block, prefer_over), loader
  specialists/  dowhy/, did/, rd/ reading the pack
  server/       CSV-only upload, sessions over the desk graph, designs and figures routes
web/            Figure.tsx, the designs strip, chips from one question
```

Deleted when their stage lands: `intake/`, `router/`, `chat/`, `server/context.py`, note rendering. `data/context/*.md` become
optional docs mined once at turn 0. `data/claims/*.yaml` migrate into `data/memory/<name>/` by a one-off script.

## 9. Stages, in build order

Each stage is a commit or a few; the app works after every stage. Stages 2 to 5 are the hand-off point. Stage 6 is the chat after.

2. **The memory.** `causal_agent/memory/` with the records, the field catalogue, the store layout, and the operations `seed`,
   `apply`, `check` (with the consistency rules), `probe`, `fit`, `open`, `snapshot`, `fork`, `render`. `profile/` carved out.
   The existing interview and router keep working on top of it through thin adapters until stage 4. A migration script turns
   `data/claims/*.yaml` into `data/memory/<name>/`. Tests for every operation and every consistency rule.
3. **The pack, frozen, and the lanes on it.** `Handoff` and `ColumnBrief` take the statused fields; `identification_allowed` goes;
   `build()` projects from a memory snapshot; the lanes read the new field names, draft only open fields, and put the said in their
   material. Forced hand-offs regenerated. Profile facts `binary_like`, `role_hints`, the compact index line, the bounds and the cache.
4. **The journey to ready.** `desk/` replaces the interview and the chat entry points: CSV-only upload, then the first thing asked is
   the causal question, nothing before it. `frame` validates it against the file, by code where it can and by one judgement where it
   must: it asks about the effect of a change on an outcome; the outcome is a column in this file; the change is something that
   reached some rows and not others, or some time and not another, and the file can tell them apart; the scope the question implies
   is one the rows cover. A question that fails says which of these it fails and asks again; no interview starts on it. Then `next`
   from `open()`, `ask` one field per turn aware of what the survivors need, `infer` gated by `apply`, the journey payload, the
   grouped timing tick, contradictions raised as questions. Page: file picker, chips from one question, the grid with "asking because".
5. **Ready, convince, hand off, run.** `viz/` spec, `Point`, `Figure`, the three pre-viz functions and the viz subgraph; `Figure.tsx`;
   `decide` and `frame` moved from the router; `convince`; `handoff` snapshots design 1 and writes `handoff.json`; `desk/pipeline`
   runs the lane from it; `router/`, `chat/`, `intake/`, `server/context.py` deleted; designs routes on the server.
6. **The chat after.** `turn` with answer, explain, show, revise, what_if, requestion, chat, done; designs as snapshots and forks;
   then-and-now; `explained` in the lanes' interpretations; the designs strip on the page.
7. **Minimal asks and scale.** `asks_next` per assignment kind; `needs` text into `ask`; the shortlist fan-out above 200 columns.
8. **Post-viz and show after the run.** `viz/postviz/` per lane, `figures.json`, figures in material and the inspector.
9. **The lanes ask back, and DoWhy's other roads.** The lane runs in-process on the desk's checkpointer once the discontinuity crash
   is fixed; an `ask` output on identify and assess; the adapter reads every estimand; a sensitivity refuter; the mediator question.

## 10. Verification

- Every stage: `uv run pytest -q` and `npm test --prefix web` green.
- Stage 2: each operation has a unit test; each consistency rule refutes the right field and never overwrites a value; a claims file
  migrates and round-trips; `open()` on the students memory returns the timing tick and the two drafts and nothing else.
- Stage 3: the three lane suites pass on packs projected from memory; the forced hand-offs give the same estimates as before within
  tolerance; a brief renders every field with status and source; a contradiction in the pack reaches the lane's material.
- Stage 4, scripted conversations with a fake model (`desk/evals/cases.yaml` as answers, not contexts): students reaches ready in
  frame plus at most six questions; a question that is not causal, or names an outcome or a change the file does not carry, is
  refused with which test it failed and asked again, and no interview starts; a wrong timing answer is refuted by the
  file and asked again; "the person's word is evidence": a message that contradicts a confirmed field with no reason does not
  overwrite it and the desk says why.
- Stage 5: on students the ready message names adjustment, cites `probe:adjustment.overlap` and `probe:adjustment.arms`, carries one
  figure, lists the struck families with an address each; design 1 has `handoff.json`, `record.md`, and a run; on senate it names
  discontinuity with the density figure.
- Stage 6: after a run, "what if lunch was set after the offer" makes design 2, leaves design 1 untouched, and the reply compares
  the two by address; "explain the placebo" answers in the question's words with the number attached.
- End to end in the browser: upload `StudentsPerformance.csv`, ask the question, answer from chips, watch the grid strike families,
  accept the ready message, run, ask what the placebo means, ask what if, see design 2 beside design 1, ask for the overlap figure.
- Nothing under `.artifacts/`, `data/web/`, or `data/memory/` is committed except the shipped datasets' migrated memories.
