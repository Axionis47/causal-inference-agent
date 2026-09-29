# common

Shared by every step. Change with care: a contract change is a version bump.

- `contracts/` — the typed artifacts nodes exchange (`base`, `frame`, `brief`, `pack`, `artifacts`, `text`): QuestionFrame, FamilyVerdict, FamilyDecision, Thought, the lane artifacts, and the context pack: `Handoff` with one `ColumnBrief` per column that matters, the `Belief`s the person holds, the `Said` quotes, the `Probe`s, and one family block (`AdjustmentDesign`, `DidDesign`, `RdDesign`). A hand-off resolves its own addresses (`col:key.note`, `col:key.when`, `claim:*`, `probe:*`, `change:1.note`), so a lane's citations are checked against the pack alone.
- `addresses.py` — the one column `key()` and the address grammar. Every package that names a column goes through it.
- `llm.py` — Gemini on Vertex AI, one place to build it. `structured()` returns the parsed contract and the model's thought summary. Tests swap the model with `set_llm()`.
- `config.py` — the one reading of the environment: the model, the paths, the sandbox, the knobs.
- `prompts.py` — the cite rule and the prompt fragments every judgement shares.
