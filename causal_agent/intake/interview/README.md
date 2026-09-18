# intake/interview — the chat in front of the router

Takes a CSV, a short description (a paragraph on the data and the change, one line per column), drafts the typed claims an analysis rests on, checks each one the file can check, asks the person only about what the file cannot say, and raises a **ready flag** when every claim the surviving families need is settled. Then it writes the pack and the router takes over. No human after that.

```
START ─ load ─ extract ─ check ─ probe ─ status ─ respond ─ listen ─┐
                 ▲                                                   │ a user turn (any number of claims, revisions, instructions)
                 └───────────────────────────────────────────────────┘
listen ─ run, when READY ─▶ write_pack ─ END
```

## The claims

`memory/fields.yaml` declares eleven kinds: `grain`, `sampling`, `change`, `assignment`, `measured` (one per column), `missing`, and five the file can never check, `unobserved`, `exclusion`, `spillover`, `trend_continues`, `cutoff_only` (asked only while the cutoff family survives). `assignment.level_column` names the group a change was assigned at, for clustering. Each kind has typed fields, the check that can refute it, and a question frame in world terms. `family_needs` says which kinds each family requires and which values fit; that block is read by code to build the table. No family or method word appears in a frame.

A claim carries a status (`empty`, `drafted`, `confirmed`, `refuted`, `unknown`, `contradiction`), a source (`doc:<name>`, `user:turn:<n>`, `data`), and evidence addresses. Drafted is not settled: what the model derived from the description is confirmed by the person before it counts.

## Facts, judgements, gates

- **Facts**: `load` (profile, seeded claims), `check` (`checks.py`: key uniqueness, the period column is dates or numbers, take-up by side of a cutoff, a column claimed fixed is constant within a unit, missing shares by arm), `probe` (`probes.py`: family disqualifiers, pandas only), `status` (`table.py`: every family against every kind; `ready` is a cell count), `write_pack`.
- **Judgements**, one call each: `extract` reads the new material into claim updates, reasoning from the column lines and the change together, citing both; `respond` writes one message with one question per open claim.
- **Gates**: an update needs a doc or user cite (numbers alone never make a claim); a confirmed claim changes only on the person's word; the uncheckable kinds are never read from a description; a confirm cannot settle a claim with a required field unset. A reply must cover every open claim and no settled one, use `confirm` only where a draft exists, list the exact options for `choose`, cite the check address for a refuted claim, never ask for a count or a share, never name a method or a family, and group columns when many are open. Three tries, then templated questions from the frames.

## The table and the flag

Rows are the claim kinds, columns the families, cells `✓ ✗ ? ·`. A family with a `✗` or a failed probe is struck out and its own claims are never asked. `required` is the union over survivors; `ready` when nothing required is open and one family survives. The uncheckable claims are asked last, once the assignment is settled.

## Run it

```bash
uv run python -m causal_agent.intake.chat data/raw/senate-incumbency/senate.csv --name senate2 --context notes.md --question "Does winning narrowly raise the next vote share?"
uv run langgraph dev   # graph "intake" in Studio
uv run pytest causal_agent/intake -q
```

`write_pack` writes `data/profiles/<name>.json`, `data/claims/<name>.yaml`, `data/context/<name>.md` (rendered from the claims in the three-heading format the pack parser reads), copies the CSV under `data/raw/<name>/` when it lives elsewhere, and adds the `datasets.yaml` entry. The pack then carries claim cards (`claim:assignment.kind`, `claim:col:margin.when`) and probe cards (`probe:discontinuity.rows_by_side`) beside the note cards, and the router cites them.

## Known limits

- The lanes still read the rendered note, not the claims. A later step points them at the claims directly.
- `entity` in the entry is set only for panels (`grain.panel` true); a cluster column for a one-row-per-unit file is not a claim yet, so the Senate rerun does not cluster by state.
- Probes cover one treated group and one cutoff. Multi-cutoff and staggered designs are not probed.
