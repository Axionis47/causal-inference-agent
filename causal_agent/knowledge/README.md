# knowledge

What the router knows about downstairs. One entry per analysis family in `families.yaml`:
what it answers, what it needs from the data, where that evidence usually lives in a pack,
what it assumes, when it is weak, which families it is preferred over and why, which specialist runs it, and whether that specialist exists.

Written as method knowledge the model reads and judges the data against. Never as rules the harness evaluates.
Adding a lane is adding an entry here and a specialist under `specialists/`. The router does not change.

- `tests/` — the registry loads, required families exist, preferences render.
