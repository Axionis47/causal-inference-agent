# desk

The one conversation from a CSV to a routed context pack, and back after the run. Built stage by stage; see
`docs/desk-redesign.md` for the plan and where it stands.

- `graph.py`, `nodes/journey.py`, `nodes/after.py`, `prompts/journey.py`, `contracts.py`, `state.py` — the conversation. The first
  thing asked is the causal question; `read_question` frames it (one judgement) and validates it against the file by code: an effect
  of a change on an outcome, the outcome a column, the change a column the file can tell apart. A question that fails is refused
  with which test it failed. Then `check` (the data checks and the consistency rules), `probe_fit` (the families over the memory,
  counting only the columns in play), `ask` (the map first, once per question: which families the file could answer it with,
  what each still needs, which are struck and why, and that the person may narrow the interview to the ones they care about;
  then one question, composed by code: the drafts to confirm in one go, the in-play columns in one tick, else one dataset
  field, with why it is asked), `listen`, `infer` (one judgement: what the message settles, with the
  person's words), and the gate in `memory.ops.apply`. A refuted answer is asked again with the check that refuted it; kept twice,
  it stands as a contradiction and reaches the lane. "run" while drafts are open takes them on the person's word. After the run:
  `brief`, `talk`, `turn` (answer, revise, requestion, done), with a revision going through the same gate and back to the checks.
- `pipeline.py` — the lane in its own process on `designs/<n>/handoff.json`; `material.py` — everything a run left behind as lines
  with addresses, what the chat after cites.
- `route.py`, `nodes/frame.py`, `nodes/decide.py`, `prompts/routing.py` — the routing, also usable alone (`desk.route`):
  `load` the memory; `mine` an attached document once into drafts when the memory holds only the file's facts (a description never
  confirms and never sets a belief); `prefilter` on a wide table; `frame` the question; `fit` the families over the memory by code;
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
