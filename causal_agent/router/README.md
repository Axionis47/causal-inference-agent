# router

Moved. The routing lives in `causal_agent/desk/route.py` and `desk/nodes/{frame,decide}.py`, and it runs on the memory:
`load ─ mine ─ (prefilter × N, wide only) ─ frame ─ fit ─ decide ─ gate ─ handoff ─ specialist:<family>`. `fit` is code over the
memory; the per-family model verdicts are gone; `decide` is one judgement, and only when more than one family stands.

This package keeps `graph.py` (a shim), `run.py` (the CLI the chat's pipeline calls), and the evals until the desk graph replaces
them at stage 4 of docs/desk-redesign.md.
