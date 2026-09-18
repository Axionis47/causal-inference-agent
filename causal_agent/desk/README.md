# desk

The one conversation from a CSV to a routed context pack, and back after the run. Built stage by stage; see
`docs/desk-redesign.md` for the plan and where it stands.

Today: the routing and the pack builder.

- `route.py`, `nodes/frame.py`, `nodes/decide.py`, `prompts/routing.py`, `state.py` — from a memory and a question to a hand-off:
  `load` the memory; `mine` an attached document once into drafts when the memory holds only the file's facts (a description never
  confirms and never sets a belief); `prefilter` on a wide table; `frame` the question; `fit` the families over the memory by code;
  `decide` among the ones that stand, one judgement, only when more than one does; `gate` the choice; `handoff`. The old router's
  per-family model verdicts are gone: what a family needs is read from the memory, and a belief not asked yet is listed unmet
  without striking the family, so the routing before the interview still works and the assumption bet on names it.
- `handoff.py`, the pack builder. `build(question, frame, decision, family, pack, claims, probes, said)` makes the
`Handoff` a lane receives; the router calls it at run time. `forced(...)` makes one without a frame or a decision, for
tests and evals, and the CLI writes those to disk:

```bash
uv run python -m causal_agent.desk.handoff students --family adjustment --outcome "math score" \
    --treatment "test preparation course" --columns "lunch,parental level of education" --question "..." -o handoff.json
```

The person's claims win over the frame's reading wherever both speak. A column with no measured claim takes its meaning
from the note card once, here, marked `doc:<name>`; the lanes never read a note.
