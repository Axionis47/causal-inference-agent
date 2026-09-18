# The desk, redesigned: CSV in, a routed context pack out

Status, 2026-09-18: stage 0 and stage 1 done (the pack: contracts, the one builder, the three lanes reading it, the two new claims, the forced hand-offs regenerated). Stage 2 next.

## Context

The financial_distress session on 18 Sept showed the whole problem. The upload form asks for a description, the
interview reads that description back as 13 confirm questions, the one real question is skipped and filled in
anyway, the answers that matter (hidden confounding: yes, an instrument: yes) are stored as bare booleans and
change nothing, the router decides from a rendered note that never mentions them, and once "everything is
settled" the chat cannot answer a question. After the run the explanation is "the placebo refuter passed".

Root cause: the interview's goal is a cell count (every claim confirmed), not "what does routing still need";
the router reads a lossy note instead of the claim table; the two never meet; and the page shows none of the
reasoning. The 17 Sept redesign goals (question after upload, one question per turn, router folded in,
previsualisation, designs as objects) name the fix. This plan is the systematic version of it, scaled to a few
hundred columns, with a code split that puts each concern in one package.

Facts confirmed by reading the code before the redesign began (18 Sept):

- No lane reads the claim table or the claims YAML; every lane reads rendered note cards only. The DoWhy lane
  discards DoWhy's frontdoor and IV estimands (`specialists/dowhy/adapter.py:46-52`).
- The router sees a one-line-per-column index plus `pack.digest()`; the prefilter fans out one model call per
  column above 150 columns; `frame` filters on votes, then family workers get full cards for the relevant set.
- The profiler is deterministic, no row cap, no column cap; `_co_missing` is O(cols² × rows) and unbounded.
- The interrupt payload has no `open_question`; chips come from `Reply.questions`; the composer re-quotes
  every answered question as prose; the gate demands every open claim be asked in one message.
- Nothing draws. rd writes `bins.csv` for a visualisation step that does not exist.
- `_key` is defined twice; `_pack_cache` exists in four modules and is invalidated by reaching into them.

## Goal, in one paragraph

Upload a CSV. The desk greets you and asks what you want to know. Each turn asks one thing, only what the
families still in play need, phrased in your world and aware of what the analysis will rest on. Every answer
can settle several cells; the file settles what it can on its own. The right-hand strip shows the journey: which
families survive, why the others are struck, why this question is being asked. When one family survives and its
needs are met, the desk decides, builds a context pack for that family's agent, and runs it. After the run the
desk explains the result and every check in the question's own terms, cites every number, and shows a figure
only when you ask for one.

## Principles (unchanged, restated)

- Facts from code, judgements from one gated model call each, declarations in YAML. Never patch code per dataset.
- One question per turn. The next question is chosen by code from the claim table, never by the model.
- No family or method word in a question; the model is told what the survivors need, and asks about the world.
- Every artifact has an address. Figures included. Nothing is stated the run did not produce.
- The lanes stay the place where library-vocabulary judgement happens (DoWhy identification, estimator, refuters).
  Wider lane reasoning (frontdoor, IV, forced covariates) is a separate later plan.

## Target layout

```
causal_agent/
  common/       contracts.py (Cited, Candidate, Scope, QuestionFrame, Handoff, Thought, artifact models), llm.py,
                addresses.py (the one `key()` and the address grammar)
  profile/      profiler.py (moved), cards.py (ColumnCard, DatasetCard, ColumnBrief, index line), cache.py, tests/
  ontology/     claims.yaml, checks.yaml, catalogue.py (loader), contracts.py (Claim, ClaimTable, Status,
                ProbeResult), table.py, checks.py, probes.py, seed.py, apply.py (the gated _apply), tests/
  knowledge/    families.yaml (registry; family_needs merged in from claims.yaml), loader, tests/  [stays]
  desk/         graph.py, state.py, contracts.py (Read, Question, Ask, Decision, AfterReply, RunRecord),
                nodes/{intake.py, frame.py, decide.py, after.py}, prompts/{intake.py, frame.py, decide.py, after.py},
                handoff.py, material.py, pipeline.py, __main__.py, evals/, tests/
  specialists/  dowhy/, did/, rd/ as today; each lane reads ColumnBrief from the handoff, gains a `figure` node
  viz/          spec.py (FigureSpec), previz/{adjustment,diff_in_diff,discontinuity}.py, postviz/{dowhy,did,rd}.py,
                registry.py, tests/
  server/       as today minus context.py; upload is CSV only; figure routes added
```

Deleted when their stage lands: `intake/` (after profile/ and ontology/ absorb it), `router/`, `chat/`,
`server/context.py`, `intake/interview/writer.py`'s note rendering. `data/context/*.md` stays as optional
attached docs for the shipped datasets (`docs:` in datasets.yaml); `data/claims/*.yaml` is the persisted ontology.

## The desk graph

```
START ─ profile ─ seed ─ greet ─(listen)─ read ─┬─ question ─ frame ─ infer ─ check ─ probe ─ fit ─ next ─┬─ ask ─(listen)─ read
                                                 │                                                        └─ ready ─ decide ─ convince ─(listen)─ read
                                                 │                                                                   run ─ handoff ─ run ─ brief ─ talk ─(interrupt)─ turn ─ …
                                                 ├─ answer  ─ infer ─ check ─ probe ─ fit ─ next
                                                 ├─ show    ─ previz ─ ask            (figure rides on the same payload)
                                                 ├─ chat    ─ reply ─ ask             (answers about the data, changes nothing)
                                                 ├─ run     ─ (ready? decide : ask)
                                                 └─ quit    ─ END
```

Facts: `profile`, `seed`, `check`, `probe`, `fit`, `next`, `handoff`, `run`, `brief`, `previz`.
Judgements, one gated call each: `read`, `frame`, `infer`, `ask`, `decide` (only when >1 family survives),
`reply`, `turn`. Interrupts: `listen` (before), `talk` (after). Retry policy as today; gates loop with errors
then fall back to templates.

### Nodes

- `profile`: `profile.profiler.profile(csv)` cached by file hash under `.artifacts/profiles/`. Builds cards.
- `seed`: `ontology.seed.seed(profile, docs)`. From the profile alone (source `data`, status confirmed):
  `missing` when no nulls; `grain.panel` from `entity_summary`; candidate keys as `grain.key_columns` draft;
  `measured.kind` per column. Binary and datetime columns are marked as treatment and time candidates on the
  cards. If a doc is attached, one `infer` call mines it with source `doc:<name>`; those claims are treated as the
  person's word (confirmed), never read back.
- `greet`: templated. "I have your file: N rows, M columns; it looks like one row per <grain guess>. What do
  you want to know? Ask it the way you would ask a colleague, for example: did X change Y."
- `read`: classifies the message: `question` (carries the causal question text), `answer` (to the open
  question, may carry more), `accept` (happy with what was shown: "looks right", "yep", "go" after the ready
  message), `show` (a figure request, names the family or figure), `chat` (asks about the data or the process),
  `run`, `quit`. Gate: kind in the set; a `question` carries text; an `answer` exists only when a question is
  open; `accept` exists only after `convince`. Tone is not a mode: every prompt writes in the person's register
  from the last exchanges; the desk asks only through `ask`, one thing at a time, and everything else is a
  `reply` that changes nothing.
- `frame`: from the question and the column index (one line per column, profile only): intent, outcome
  candidates, cause candidates, scope, relevant columns. Gate: named columns exist. `not_causal` intent goes to
  `reply` with a nudge, never to the interview. Writes `assignment.treatment_column` and the outcome's role as
  drafts the next `ask` confirms in one question ("You want the effect of <cause> on <outcome>, right?").
- `infer`: the current `extract` and its sweep, folded into one call with two jobs: (1) claim updates the
  message states, cited `user:turn:n`; (2) claims the answer *implies*, cited the same turn with `reason`
  (a lottery implies `unobserved.exists=false` and `exclusion.exists=false`; "everyone is in the file" implies
  `sampling.how=whole`). Gate: `ontology.apply.apply()` as today (cites resolve, no fill from numbers,
  uncheckable only from the person, confirmed changes only on the person's word) plus: an implied update must
  name the answer it follows from.
- `check`, `probe`: moved as is. Probes also return the pre-viz spec when asked (see Viz).
- `fit`: `ontology.table.compute()`, with two changes: `required` for `measured` covers only the frame's
  relevant columns; a family whose `fits` fail on the uncheckable answers is struck (adjustment struck when
  `unobserved.exists=true` and no `exclusion.column`; instrument kept only with `exclusion.column`).
- `next`: deterministic picker. Order: the frame confirmation; `assignment.kind`; the follow-up fields the kind
  demands (`asks_next` in claims.yaml: cutoff_rule → score_column, cutoff; own_choice, third_party →
  depends_on; date_by_others → date_column, period_value, group); outcome timing; unit and time columns for
  panel families; the uncheckable claims the survivors need; then timing of the remaining relevant columns as
  one grouped question pre-ticked from profile facts. Skips cells the file settled. Returns one `Ask` target:
  claim key, field, kind (confirm, choose, open, tick), options, and `because`: which surviving families need it.
- `ask`: writes one question for that target. Prompt carries: the target, the legal options, the draft if any,
  the `needs` text of the surviving families from families.yaml (so the question can say "the analysis will
  compare people who got it with people who did not, holding <depends_on> fixed; was anything else behind who got
  a place?"), and the last two exchanges. Gate: exactly one target, options exact, no method words, no counts
  or shares. Fallback: the template from the claim's `frame`.
- `decide`: when `status.ready`. One survivor: chosen by code, no model call; the assumption text comes from
  families.yaml `assumes`. Several: one call choosing among survivors with `prefer_over`, citing claim and probe
  addresses; gate: chosen in survivors, cites resolve. No survivor: `reply` says which families fell and why,
  and asks which claim to revisit. Writes `Decision` (chosen, assumption, why, over: {family: reason}).
- `convince`: the one place a figure appears unasked. Reruns the family's probes on the final table, sends the
  family's `Point` to the viz tool (it picks the figure and writes the caption, or says the point cannot be
  made), and writes the "ready to route" message by template: the design in one sentence, the assumption in your words, the evidence (figure,
  caption, probe numbers, each with its address), and the struck families with one reason each. Then `listen`.
  You accept (`accept` → run), ask about the evidence (`chat`), or change a claim (`answer`), which loops back
  through `fit`.
- `handoff`: on run. Builds the context pack (below) and writes `data/claims/<name>.yaml` and
  `.artifacts/web/<name>/handoff-<n>.json`.
- `run`, `brief`, `talk`, `turn`: as today, with the after-phase changes in stage 5.
- `previz`: runs the family's pre-viz tools on the frame's columns and the claims; the spec rides on the
  interrupt payload once and is not stored.
- `reply`: answers a `chat` message from the profile, the claims, and the status, cited; changes nothing.

### State

```python
class DeskState(TypedDict, total=False):
    dataset: str; csv: str; docs: dict[str, str]
    profile_id: str                         # file hash; the profile itself is loaded by the nodes
    question: str | None; frame: QuestionFrame | None
    claims: ClaimTable; probes: list[ProbeResult]; status: Status | None
    target: Ask | None                      # the one open question
    reply: Reply | None                     # text + the one Question for chips + journey + figures
    decision: Decision | None; handoff: Handoff | None
    runs: list[RunRecord]; exchanges: list[Exchange]; brief: str
    phase: Literal["before", "after"]; turn: int; after_turn: int
    messages: Annotated[list[dict], operator.add]; debug: Annotated[list[Thought], operator.add]
    # per-node error/attempt counters as today
```

### Interrupt payload (what the page gets)

```python
{"phase": "before", "text": str, "question": Question | None,            # one question, chips from its options
 "journey": {"families": {name: {"state": "in_play" | "struck", "why": str}},
             "asking_because": [family, ...], "settled": [claim key, ...], "open": [claim key, ...]},
 "figures": [FigureSpec], "ready": bool}
```
The page shows `text`, chips for `question`, and the journey in the right strip (StatusMatrix already draws the
grid; add the "why this question" line). The composer sends the chip answer as
`On "<question>": <answer>` exactly as today; free text stays free text.

## The ready moment: what it looks like

The message the page shows when `convince` runs, on the students file (every bracket is an address the page
renders as a mark; the figure is inline, drawn from its spec):

> **One design fits: compare students who completed the course with those who did not, holding lunch and
> parental education fixed.** It rests on one thing you told me: nothing outside the file, beyond those two,
> decided who got a place [claim:unobserved.exists].
>
> Why it fits. Both kinds of student exist in every lunch × parental education cell, so there is always someone
> to compare with [probe:adjustment.overlap]; 358 completed and 642 did not [probe:adjustment.arms].
>
> *(figure: overlap of completion by lunch and by parental education, two small bar panels)*
>
> Why not the others. Before-and-after comparison: the file has one exam per student, nothing before the course
> [claim:grain.panel]. Cutoff comparison: places were offered by a rule on lunch and parental education, not by
> a line on a score [claim:assignment.kind]. Nudge-based: you named nothing that pushed students in without
> touching their marks [claim:exclusion.exists].
>
> Say **run**, ask me about any of this, or tell me what is wrong.

Right strip at that moment: the family grid with one row lit and the others struck with their one-line reasons;
under it the settled claims; the "asking because" line is gone. After the run the same strip gains the runs and
files tabs it has today.

For diff_in_diff the figure is the outcome by group over time with the change date marked; for discontinuity
it is the score density around the cutoff with the outcome means per bin. The message template is per family in
`families.yaml` (`convince:` block: which probes and which pre-viz, in which order), read by code.

## How the profile is used

The profile is computed once and read everywhere. What each fact feeds:

| profile fact | where it is used |
|---|---|
| rows, columns, candidate keys, grain, entity summary | `greet` ("one row per student", "the same store over 52 weeks"); `seed` drafts `grain` and `grain.panel` |
| nulls per column, co-missing | `seed` settles `missing` when there are none; `check missing_by_arm`; the grouped timing question skips gap-free facts |
| kind per column, distinct, top values | the column index line for `frame`; `seed` marks candidates: binary columns as possible treatments, datetime or period columns as time, ids as units, numeric non-id measures as outcomes |
| top values of the treatment column | `next` offers the two levels as chips for `treated_level`; `check assignment_matches_data` |
| numeric stats | cutoff candidates: a numeric column with a rule-like mass on one side; `probe rows_by_side` |
| varies_over, switch profile (panels) | pre-ticks the timing question (a column that varies within a unit over time cannot be "fixed before"); `probe treated_before`, `treated_units` |
| time coverage, gaps | `probe pre_periods`; the diff_in_diff and series families are struck on the shape alone |
| sentinels, format issues | shown once in `greet`; `reply` answers "what is -999 in income" without a model call |

New profile facts the desk needs and the profiler does not yet compute (stage 2):

- `binary_like`: two non-null values; the treatment candidate list.
- `role_hints` per column: `treatment_candidate`, `time_candidate`, `unit_candidate`, `outcome_candidate`,
  `score_candidate` (numeric, many distinct, a visible mass or gap), from the facts above; the frame prompt sees
  them in the index line.
- `arms` given a treatment column: counts and the null share per arm; computed by `probe`, not the profiler,
  because it needs a claim.
- `overlap` given a treatment column and a set of columns: the cells and the share of cells with both arms;
  the adjustment pre-viz and the `probe:adjustment.overlap` number come from the same function in `viz/previz`.
- `by_group_over_time` given unit, time, treatment: the series the diff_in_diff pre-viz and `pre_periods` share.

The rule: a number the page shows and a number the desk cites come from the same deterministic function, so
the figure and the probe never disagree.

## The context pack (Handoff): fixed first, everything else builds toward it

The pack is what a lane receives. It must let the lane draw the graph, pick the estimand and estimator, choose
controls, run the right checks, and write the caveats without reading a note and without guessing. It carries
the person's own words beside every claim, and it names the alternatives the lane must weigh so no lane keeps
picking its one default analysis. One core, one family block. `common/contracts.py`:

```python
class ColumnBrief(BaseModel):
    name: str; key: str; kind: str
    role: Literal["outcome","treatment","depends_on","score","unit","time","group","instrument","candidate"]
    meaning: str | None; when: Literal["before","at","after","unknown"]; set_by: str | None
    affected_by_treatment: bool | None; varies_over: str | None
    profile: ColumnProfile                 # facts: nulls, distinct, top values, numeric, datetime
    def render(self) -> str                # [col:key.meaning] [col:key.when] [col:key.profile.*] — what the lanes read

class Belief(BaseModel):                   # an uncheckable claim, with the person's reason
    value: bool | None; what: str | None; why: str | None; column: str | None; status: str; said: str | None

class Said(BaseModel):                     # verbatim, so lane judgements see the human's reasons, not field values
    turn: int; about: str; text: str

class Handoff(BaseModel):
    # the question
    question: str; intent: Intent; scope: Scope          # contrast switch|dose|level_vs_level; target average|on_treated|conditional
    # the decision
    family: str; specialist: str; supported_now: bool
    chosen_assumption: str; why: str; over: dict[str, str]
    # the data
    pack_name: str; csv: str; docs: dict[str, str]
    grain: dict                # row_is, key_columns, panel
    sampling: dict; missing: dict
    # the columns that matter
    outcome: ColumnBrief; treatment: ColumnBrief | None; treated_level: str | None; control_level: str | None
    columns: list[ColumnBrief]
    # how the change happened
    change: dict               # what, to_whom, when, date_column, period_value
    assignment: dict           # kind, rule (person's words), depends_on, level_column, movable
    # what the person believes and could not say
    beliefs: dict[str, Belief] # unobserved, exclusion, spillover, trend_continues, cutoff_only
    unknowns: list[str]
    said: list[Said]
    # evidence
    probes: list[ProbeResult]
    claims: dict               # the full snapshot, statuses and sources, for the record
    # the family block
    design: AdjustmentDesign | DidDesign | RdDesign
```

Family blocks. Each field is something the lane infers from a note today or fixes to one default.

```python
class AdjustmentDesign(BaseModel):                       # dowhy
    adjustment_candidates: list[str]   # depends_on + every before-column not moved by the treatment
    forbidden: list[str]               # after/at columns, other outcome measures
    identification_allowed: list[Literal["backdoor","instrument","frontdoor"]]
                                       # backdoor always; instrument when beliefs.exclusion.column; frontdoor when a mediator is named
    instrument: str | None; mediator: str | None
    unobserved_confounding: bool | None  # True → a sensitivity refuter is mandatory and the caveat says so
    voluntary_uptake: bool             # own_choice → overlap expected; strict rule → the weak case
    target_units: str                  # from scope.target
    contrast: str                      # switch|dose|level_vs_level; dose → feasibility stop with reason

class DidDesign(BaseModel):                              # pyfixest
    unit: str; time: str; period_kind: Literal["date","integer"]
    change_period: str; treated_group: dict   # {column, level} or {cohort_column, adoption_periods}
    staggered: bool; never_treated_exists: bool | None; pre_periods: int | None; post_periods: int | None
    controls_allowed: list[str]        # before-columns and unit-constant columns; post-treatment varying forbidden
    cluster_level: str                 # unit or the group assignment happened at
    trend_belief: Belief; spillover: Belief

class RdDesign(BaseModel):                               # rdrobust
    score: str; cutoff: float; treated_side: Literal["above","below"]; cutoff_value_treated: bool | None
    score_fixed_before: bool | None; movable: bool | None
    takeup: dict | None                # {column, level} → fuzzy; None → sharp
    covariates_allowed: list[str]; cluster: str | None; sampled_by_side: bool
    cutoff_only: Belief                # nothing else switches at that line
```

Two claims the catalogue lacks and the pack needs: `assignment.level_column` (the level the change was
assigned at, for clustering) and an uncheckable `cutoff_only` belief required by discontinuity. One line each
in claims.yaml. A `mediator` claim is out of scope until the DoWhy lane's frontdoor path exists.

One builder, `desk/handoff.py: build(question, frame, decision, claims, probes, profile, said) -> Handoff`,
used by the desk at run time and by `desk/handoff.py --dataset --family` to write the forced hand-offs the
specialist evals use. No second copy anywhere.

Lane changes (one commit per lane, no new judgement): `_card(pack, key)` becomes `handoff.brief(key).render()`;
`load` stops touching the pack; dowhy `contrast` reads `treated_level` and `pick_estimator` sees
`identification_allowed` and `unobserved_confounding`; `relate` prompts get meaning and `when` from the brief;
did `groups`/`periods` read the `DidDesign` block and stop inferring axes; rd `score` reads the `RdDesign`
block. `pipeline.run` launches `python -m causal_agent.desk.pipeline <handoff.json> --json-file <out>`, which
runs the lane subgraph only.

## No duplicated router code

- `frame` moves from `router/nodes.py` into `desk/nodes/frame.py` with its prompt, contract (`QuestionFrame`),
  normalisation, and gate unchanged. It reads the profile index instead of the pack digest.
- `test_family` workers are deleted. Their job, needs versus material, is the deterministic fit table over the
  claim snapshot (`family_needs` and `fits` in families.yaml). The struck reasons come from the table.
- `decide` moves with its prompt and `FamilyDecision`; its verdicts input becomes the table's survivors and
  struck reasons; it runs only when more than one family survives.
- `gate`, `handoff`, `route_specialist`, `_pack_cache` in the router go. The lanes' three `_pack_cache` copies
  go with the pack read.
- `router/` is deleted in the same commit the desk's `frame` and `decide` land.

## Scaling to a few hundred columns

- Profiler: `_co_missing` over the 40 columns with the most nulls only; `_try_datetime` regex on the first
  1000 non-null values; profile cached by file hash. Candidate key pairs already bounded at 12.
- Column index line ≤ 25 tokens: `[col:key] name · kind · distinct · nulls · 3 examples`. Deterministic filter
  first: drop ids, constants, free text, null rate > 0.9.
- `frame` takes the whole index in one call up to `FRAME_WIDTH_BUDGET` (default 200 columns). Above it,
  `shortlist` fans out with `Send` in chunks of 100 columns (question + chunk → relevant names), then `frame`
  runs on the union. 500 columns is 5 calls, not 500.
- The interview never fans out per column. `measured` is required only for the relevant set, asked as one
  grouped tick question pre-ticked from `varies_over` and the assignment claim.
- Lanes' `relate` stays one call per relevant column, bounded by the frame (≤ ~15).

## Visualisation (`causal_agent/viz/`)

The figure is a fact; whether it convinces is a fact (the probe's pass or fail against its threshold); which
figure to show and what it means in the person's words is the one judgement. No model draws, and no model looks
at a picture to judge it.

- The desk calls the viz tool with a **point to make**, never with a figure name:

  ```python
  class Point(BaseModel):
      text: str                    # one sentence, the claim the figure must support, in the person's words:
                                   # "every lunch × parental-education cell has students who did and did not complete"
      family: str; columns: list[str]; cites: list[str]     # the claim and probe addresses the point rests on
      register: str                # the last two exchanges, so the caption matches the person's tone

  class Figure(BaseModel):
      made: bool                   # False when the numbers do not support the point
      spec: FigureSpec | None; caption: str; numbers: dict[str, float]; why_not: str | None
  ```
  Who sends a `Point`: `convince` by code (the family's `convince:` block in families.yaml gives the point
  template and the columns); `ask`, `reply`, `turn` and the lanes' `interpret` as an optional field on their
  reply when the model judges a figure would make its point (gate: the point cites addresses that resolve).
  The person's own "show me" becomes a `Point` too, written by `read`.
- `viz/graph.py`: the figure subgraph that serves a `Point`:
  `pick` (judgement: choose among the family's registered figures the one that makes this point, write the
  caption in the person's register, citing the numbers) → `render` (fact: the spec from the registered
  function) → `check` (fact where a rule exists: the point's numbers against the registered function's output,
  for example an overlap point needs the share of cells with both arms at 1.0; otherwise the gate is that every
  number in the caption matches) → return `Figure`. When the check fails, `made=False` with `why_not` and the
  numbers, and the caller says so instead of showing a picture. Fallback after retries: the family's default
  figure with a templated caption.
- `spec.py`: `FigureSpec` (id, kind: bars | lines | points | density | interval; title; caption; series with
  x/y and labels; marks such as a vertical line at the cutoff or change date; `numbers: dict[address, float]`).
  Addresses `figure:<id>` and `figure:<id>.<series>.<i>`. No drawing anywhere in Python.
- `previz/`: one module per family, pure pandas from the DataFrame and the claims: adjustment → overlap of each
  `depends_on` column by arm; diff_in_diff → outcome by group over time with the change marked; discontinuity →
  score density around the cutoff and outcome means per bin. `registry.PREVIZ[family]`.
- `postviz/`: one module per lane, reading the run directory csvs and artifacts: dowhy → overlap by arm on the
  adjustment set, effect with interval against the placebo; did → group trends and event-study coefficients; rd →
  binned outcome with fits (`bins.csv`) and the density test. Each lane's new `figure` node writes
  `figures.json`; `material.render` adds figure lines and numbers; `runs.py` serves it.
- Shown unasked exactly once: at the ready moment, by `convince`, for the chosen family. Otherwise only on
  request. Before the run: `read` kind `show` → `previz` → the spec rides on the payload once. After the run:
  `turn` kind `show` with a figure address; the brief lists what figures exist by name and does not render
  them. Page: one `Figure.tsx` SVG renderer for the spec kinds, opened inline in the message and in the
  inspector under `#runs/2/figures/<id>`.
- The pre-viz functions return both the spec and the probe number (`overlap` returns the cell table and the
  share of cells with both arms), so `probe` and `convince` call the same code.

## Post-run explanation in the question's terms (stage 5)

- Lane `interpret` prompts: for every check, refutation, and placebo, one sentence in the question's world
  ("if the course had done nothing, a made-up course should show nothing; it showed 0.2, so the 5.6 is not an
  artefact of the method") beside the technical line, both cited. `Interpretation` gains
  `explained: list[Explained(address, plain)]`.
- `material.render` uses each lane's `Design.render()` instead of `str(dict)[:600]`, and includes `explained`.
- `turn` prompt: answer from the question's point of view first, technical name second, every number attached.
- `brief`: opens with the answer in one sentence, then what was checked in plain words, then "ask me about any
  of these, or say show <figure>".

## Stages (each a commit or a few; the app works after every stage)

0. **Visibility.** `docs/desk-redesign.md`: this plan, kept current per stage with a status line each. The
   README of every package rewritten as the stage lands.
1. **The pack, first.** `ColumnBrief`, `Belief`, `Said`, the new `Handoff`, and the three design blocks in
   `common/contracts.py`; `assignment.level_column` and `cutoff_only` in claims.yaml; `desk/handoff.py` with the
   one builder and its CLI for forced hand-offs. Today's router `handoff` node calls the builder (from the claim
   YAML, the frame, and the decision) so the current interview and router already produce the new pack. The
   three lanes read briefs and their design block; the forced hand-offs under `specialists/*/evals/handoffs/`
   are regenerated; lane tests and evals pass. The interview asks the two new claims where their family
   survives. After this stage nothing downstream reads a note.
2. **Mechanical split.** `profile/` (profiler, cards, cache) and `ontology/` (claims.yaml, checks.yaml,
   catalogue, contracts, table, checks, probes, apply) carved out of `intake/`; `common/addresses.py` replaces
   both `_key` copies; `family_needs` moved into `families.yaml`. Tests move with their modules. No behaviour
   change; `intake/` re-exports until stage 5.
3. **Profile facts and the figure contract.** `binary_like`, `role_hints`, the compact index line, profiler
   bounds and the hash cache; `viz/spec.py`, `Point`, `Figure`, and `viz/previz/` for the three built families,
   each returning the spec and its probe number; `probe` calls them; the viz subgraph with its `pick` judgement
   and `check`. `Figure.tsx` renders the spec kinds. Nothing on the page shows a figure yet; the tests draw the
   specs from the shipped CSVs.
4. **Desk graph v1: the journey.** `desk/` replaces `chat/` and `intake/interview/` at the entry points:
   CSV-only upload, `greet`, `read`, `frame` confirmation as the first question, `next` + `ask` one per turn,
   `infer` with implied claims, `reply` for chat, journey payload. Page: `NewDataset.tsx` becomes a file picker
   and a name; `Composer` reads `question`; `StatusStrip` shows `asking_because`. The router still runs at `run`
   for this stage and hands off through the stage-1 builder. Target on the students file: frame + ≤ 6 questions
   to ready.
5. **Fold the router, and convince.** `frame` and `decide` moved (not copied) into the desk; `fit` strikes on
   beliefs; `convince` sends the family's `Point` to the viz tool and shows the figure inline, with the
   `convince:` block per family in families.yaml; `desk.pipeline` runs a lane from a hand-off file; `router/`,
   `chat/`, `intake/`, `server/context.py`, note rendering deleted; shipped datasets get `docs:` entries and are
   mined at turn 0; `langgraph.json` lists `desk` and the three lanes.
6. **Minimal asks and scale.** `asks_next` per assignment kind in claims.yaml; `measured` required only for the
   relevant set with the grouped tick question pre-ticked from the profile; `needs` text into the `ask` prompt;
   `shortlist` fan-out; `FRAME_WIDTH_BUDGET` replaces `ROUTER_WIDTH_BUDGET`.
7. **Plain-language after phase.** `explained` in the lanes' interpret; `Design.render()` in material; `turn`
   and `brief` rewritten; `Point` requests from `turn` and `interpret`.
8. **Post-viz and show.** `viz/postviz/` per lane; `figure` node per lane writing `figures.json`; figure lines
   in material and the runs route; `show` before and after the run; figures in the inspector.
9. (Follow-on plan, not here.) Designs as first-class forks; the DoWhy lane's frontdoor and IV paths and a
   `mediator` claim; forced covariate sets.

## Files that carry the change (by stage)

- 1: `causal_agent/profile/{profiler,cards,cache}.py` ← `intake/profiler.py`, `intake/pack.py`;
  `causal_agent/ontology/*` ← `intake/knowledge/*`, `intake/interview/{contracts,table,checks,probes}.py`, the
  `_apply`/`_coerce` half of `intake/interview/nodes.py`; `common/addresses.py`; `knowledge/families.yaml`.
- 2: `profile/profiler.py` (`binary_like`, `role_hints`, bounds), `profile/cards.py` (index line),
  `profile/cache.py`; `viz/spec.py`, `viz/previz/{adjustment,diff_in_diff,discontinuity}.py`, `viz/registry.py`;
  `ontology/probes.py` (calls previz); `web/src/components/Figure.tsx`.
- 3: `causal_agent/desk/{graph,state,contracts,handoff,material,pipeline}.py`, `desk/nodes/*`, `desk/prompts/*`
  (from `chat/*` and `intake/interview/{nodes,prompts}.py`); `server/{datasets,models,sessions}.py`;
  `web/src/pages/NewDataset.tsx`, `web/src/components/{Composer,QuestionList,StatusStrip}.tsx`, `web/src/types.ts`.
- 4: `common/contracts.py` (Handoff, ColumnBrief), `desk/nodes/decide.py` (`fit`, `decide`, `convince`),
  `knowledge/families.yaml` (`convince:` block), `specialists/{dowhy,did,rd}/nodes.py` (`_card`, `load`,
  `contrast`/`groups`/`score`), `specialists/*/evals/handoffs/*.json`, `desk/pipeline.py`, `langgraph.json`,
  `data/datasets.yaml`, `web/src/components/Message.tsx` (inline figure).
- 5: `ontology/claims.yaml` (`asks_next`), `ontology/table.py`, `desk/nodes/{intake,frame}.py`.
- 6: `specialists/*/prompts.py`, `common/contracts.py` (Interpretation), `desk/material.py`, `desk/prompts/after.py`.
- 7: `viz/postviz/*`, `specialists/*/graph.py` (+ `figure` node), `server/runs.py`, `server/app.py`,
  `web/src/selection.ts`, `web/src/components/inspector/FilesView.tsx`.

## Verification

- Per stage: `uv run pytest -q` green; the moved tests unchanged in stage 1. `npm test --prefix web` green.
- Stage 2 and 3, scripted: the desk evals (`desk/evals/cases.yaml`) rewritten as scripted *answers* instead of
  contexts: students (own choice after an offer → adjustment in ≤ 6 questions), senate (cutoff → discontinuity),
  card_krueger (date from above, panel → diff_in_diff), a not-causal question (no hand-off), a hidden-confounder
  yes with no instrument (no family survives, the desk says so). Evaluators: question count, no method word in
  any question, no claim confirmed without a user or data source, chosen family, hand-off fields present.
- Stage 3, the lanes: the three lane test suites pass against hand-off JSON of the new shape; the forced
  hand-offs under `specialists/*/evals/handoffs/` produce the same estimates as before within tolerance.
- Stage 2: for each shipped CSV, `role_hints` name the known treatment, time, and unit columns; each pre-viz
  function returns a spec whose numbers equal the probe number it reports (one test per family).
- Stage 4, the ready moment: on students the message names adjustment, cites `probe:adjustment.overlap` and
  `probe:adjustment.arms`, carries one figure, and lists the three struck families with a claim address each;
  every address resolves. On senate it names discontinuity with the density figure.
- Stage 5: a synthetic 500-column CSV (students plus 490 noise columns) frames in ≤ 6 model calls and the
  interview asks no more questions than on the 8-column file.
- End to end in the browser (preview tools): upload `StudentsPerformance.csv`, read the greeting, ask the
  question, answer the questions from chips, watch the journey strip strike families, run, ask "what does the
  placebo check mean for my question", then "show me the overlap", and confirm the figure renders and is cited.
- Nothing in `.artifacts/` or `data/web/` is committed; `data/claims/<name>.yaml` round-trips through
  `load_dataset_pack` and the server restart test in `server/tests/test_sessions.py`.
