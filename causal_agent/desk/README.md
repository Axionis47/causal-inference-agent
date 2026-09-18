# desk

The one conversation from a CSV to a routed context pack, and back after the run. Built stage by stage; see
`docs/desk-redesign.md` for the plan and where it stands.

Today: `handoff.py`, the pack builder. `build(question, frame, decision, family, pack, claims, probes, said)` makes the
`Handoff` a lane receives; the router calls it at run time. `forced(...)` makes one without a frame or a decision, for
tests and evals, and the CLI writes those to disk:

```bash
uv run python -m causal_agent.desk.handoff students --family adjustment --outcome "math score" \
    --treatment "test preparation course" --columns "lunch,parental level of education" --question "..." -o handoff.json
```

The person's claims win over the frame's reading wherever both speak. A column with no measured claim takes its meaning
from the note card once, here, marked `doc:<name>`; the lanes never read a note.
