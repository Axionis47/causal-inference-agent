# intake

From a CSV and a semantic note to a pack the router can cite. No model calls anywhere in this step.

- the profiler, the pack loader, and the dataset index now live in `causal_agent/profile/`; the claim catalogue, the claim table, the checks, the probes, and the family table in `causal_agent/memory/`. The modules here are shims until this package goes (stage 5 of docs/desk-redesign.md). CLI: `uv run python -m causal_agent.profile.profiler <csv> --entity <col> --time <col> -o <json>`.
- `pack.py` — joins a note (`data/context/<name>.md`) and a profile (`data/profiles/<name>.json`) into cards with addresses like `col:lunch.note`, `change:1.note`, `dataset.profile.grain`. Addresses are the citation system.
- `datasets.py` — the dataset index (`data/datasets.yaml`) and `load_dataset_pack(name)`.
- `tests/` — profiler behaviour on real and synthetic files; every pack loads with every column noted.

Inputs live in `data/`: `raw/` (the CSVs), `context/` (notes), `profiles/` (profiler output).

`interview/` is the chat that writes a note, a profile, a claims file, and the `datasets.yaml` entry for a new CSV from a short description and a few questions; see its README. `chat.py` is its CLI. Packs built that way carry `claim:` and `probe:` cards beside the note cards.
