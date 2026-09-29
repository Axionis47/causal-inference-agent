# desk

The one conversation from a CSV to a routed context pack, and back after the run. The graph is drawn and walked in
[docs/desk.md](../../docs/desk.md); the judgements and their gates are in [docs/gates.md](../../docs/gates.md).

- `graph.py`, `nodes/journey.py`, `nodes/after.py`, `prompts/journey.py`, `contracts.py`, `state.py` — the conversation. The first
  thing asked is the causal question; `read_question` frames it (one judgement) and validates it against the file by code: an effect
  of a change on an outcome, the outcome a column, the change a column the file can tell apart. A question that fails is refused
  with which test it failed. Then `check` (the data checks and the consistency rules), `probe_fit` (the families over the memory,
  counting only the columns in play), and `ask`, which composes the next thing by code: the map first, once per question (which
  families the file could answer it with, what each still needs, which are struck and why, and that the person may narrow the
  interview to the ones they care about); then the interview of ADR 0006, a story, a readback, and gaps asked by decision.
  The story turn (`Ask(kind="story")`, once per question, before any field question) asks what the change was, who could get it
  and how that was decided, what each column records and when it was set, and what one row is; the Reader drafts every claim it
  can from the answer. When a note was mined the story is skipped and the readback comes first. The readback (`readback.py`,
  `compose_readback`) reads the drafts back grouped by the five claims every decision rests on (the assignment, the change, the
  grain and the rows, each column's meaning and when, the beliefs), each line in the world's terms with the sentence it rests on,
  and asks whether it is right; a yes confirms every draft shown. It serves every confirm turn, including the drafts a run's graph
  leaves. What is still open is then asked because a decision needs it: `compose_ask` takes the first open field, finds the first
  family decision (`Family.decisions`, `rests_on` patterns matched with fnmatch) that rests on it among the surviving families,
  and asks every open field under that decision in one turn, prefixed "To settle <what the decision asks>, I need:"; a field no
  decision rests on is asked with the rest of its claim, or with the same field of every other column. A belief is asked as what
  the design would bet on ("The design will assume <the family's assumption>. <the kind's frame>?") with the person's own
  assignment rule and what it depended on beside it. Free text answers; "don't know" is allowed.
  `listen` takes the answer; `infer` is the Reader over it (`reader.py`, `read_words`: one prompt, `READ_SYSTEM`/`READ_USER`,
  over a message tagged `user:turn:<n>` or a note tagged `doc:<name>`, returning a `Reading`: the fields it fills with the words
  each rests on, the drafts confirmed, the ones unknown, and whether it asks the desk something or asks for a picture; the story
  is read into drafts, a note drafts and never confirms and never sets a belief). The gate is `memory.ops.apply`. `draw` (when
  the person asked for a picture: the drawing tool makes it from the file, the journal records an `explore` step, the caption
  and the picture come before the next thing asked; a picture settles nothing). `explain` is the Explainer before the run
  (`explainer.py`, `answer_from` and `gate`: one prompt, `ANSWER_SYSTEM`/`ANSWER_USER`, one contract `AfterReply`, one gate
  before and after the run; before it the material is `before_material`: `family:<name>` lines with each family's knowledge,
  `matrix:` cells, `probe:` results, `step:<n>`, `user:turn:<n>`, and every `claim:`/`col:` field, so every cite is an address
  in the material; only `answer` and `draw` are legal before the run and the gate refuses the rest; three tries, then the honest
  fallback). A refuted answer is asked again with the check that refuted it, under the decision that needs it; kept twice, it
  stands as a contradiction and reaches the lane. "run" while drafts are open takes them on the person's word. After the run:
  `brief`, `talk`, `turn` (the same Explainer over `material.render`: answer, revise, what_if, requestion, draw, done), with a
  revision going through the same gate and back to the checks, and a picture drawn into the design's folder and citable as
  `artifact:<id>`.
  Every step writes itself into the conversation's journal (`memory.journal`, through `nodes/shared.record`): the question read,
  each claim settled and on whose word, each explanation, the design, the run, the brief, each answer, each what-if, revision and
  new question, with the memory version at that moment, what it read, and what it left under `designs/<n>/`. A step's address is
  `step:<n>`; the Explainer reads the steps as lines and may cite one. The unit of the record is the design run: the steps that
  led to design n, the design, the run, and what was said about it until the next design.
- `designer.py` — the Designer: one judgement writes the `DesignBrief` (the road, the target, one decision per decision the family
  lists, the threats, the sentence it bets on); `check` gates it and `fallback` is the honest brief after three failures.
- `relations.py` — the run's graph absorbed back into the memory as drafts the person confirms on the next ask.
- `pipeline.py` — the lane in its own process on `designs/<n>/handoff.json`; `material.py` — everything a run left behind as lines
  with addresses, what the chat after cites.
- `nodes/question.py` — the question asked first and validated by code; `nodes/frame.py`, `nodes/decide.py`, `prompts/routing.py` — the routing; `route.py` is `route(question, dataset)`, the same nodes
  called in order without the interview, for the evals and the command line:
  `load` the memory; `mine` an attached document once into drafts through the Reader when the memory holds only the file's facts (a
  description never confirms and never sets a belief); `prefilter` on a wide table; `frame` the question; `fit` the families over the memory by code;
  `decide` among the ones that stand, one judgement, only when more than one does; `gate` the choice; `handoff`. The old router's
  per-family model verdicts are gone: what a family needs is read from the memory, and a belief not asked yet is listed unmet
  without striking the family, so the routing before the interview still works and the assumption bet on names it.
- `handoff.py`, the pack builder. `build(question, frame, decision, family, memory, probes, design_id)` projects the
`Handoff` a lane receives from the memory: every brief field with its status, source, and the sentence it rests on; the
beliefs; the person's words; the fields left unknown or in contradiction; the family block derived by code. The routing
calls it at run time. `forced(...)` makes one without a frame or a decision, for tests and evals, and the CLI writes those
to disk (`--mine` first reads the attached note into drafts when the memory holds only the file's facts):

```bash
uv run python -m causal_agent.desk.handoff students --family adjustment --outcome "math score" \
    --treatment "test preparation course" --columns "lunch,parental level of education" --question "..." -o handoff.json
```

The memory wins over the frame's reading wherever both speak. A note is read once, by `mine`, into drafts marked
`doc:<name>`; the lanes never read a note. Roles are a view (`memory.ops.roles`), computed from the memory and the frame.
