# common

Shared by every step. Change with care: a contract change is a version bump.

- `contracts.py` — the typed artifacts nodes exchange: QuestionFrame, FamilyVerdict, FamilyDecision, Handoff, Thought. Every claim carries the pack addresses it cites.
- `llm.py` — Gemini on Vertex AI, one place to build it. `structured()` returns the parsed contract and the model's thought summary. Tests swap the model with `set_llm()`.
