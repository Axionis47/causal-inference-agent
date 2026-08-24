# T-007 — Intake deterministic core: contracts, registrations, archive safety, profiler

Status: ACCEPTED (single-session, D-008)
Contract sections: PRD-001 §5.5–§8 (classification, slots, availability),
§6 (measured facts), §12 (outcomes); SYSTEM-CONTRACT §3.1, §14.1 (intake
1,200-line allocation). Maps to EV-P1-003 (archive) and EV-P1-004
(profiling/availability) at the unit layer.

Split: T-007 = the deterministic, network-free core. T-008 (next) = Kaggle
capture layer (behind a client Protocol), catalog migration + views,
coordinator, IntakeOutcome commit, PRD-002 handoff.

## Files

- `registries/artifact-types.v1.json` (append 6 rows, D-012)
- `src/causal/intake/contracts.py`
- `src/causal/intake/archive.py`
- `src/causal/intake/profiler.py`
- `tests/intake/__init__.py`, `tests/intake/test_contracts.py`,
  `tests/intake/test_archive.py`, `tests/intake/test_profiler.py`
- `tests/shared/test_registry.py` (update row-count/lookup assertions)

## Decisions

- D-023: intake-internal artifact registrations use the single terminal
  status `committed`; PRD-declared outcome statuses exist only on
  `IntakeOutcome` (`usable`/`partial`/`refused`). Raw `KaggleCapture` is
  sensitivity `restricted` and readable only by `intake-coordinator`
  (PRD-001 §11.3: downstream never reads undifferentiated captures).
- D-024: `IntakeSubmissionV1.kaggle_ref` normalizes to `owner/slug`;
  accepted inputs are `owner/slug` or a `kaggle.com/datasets/owner/slug[...]`
  URL. Anything else is a validation error.
- D-025: profiler hypotheses (PRD-001 §6) are two bounded deterministic
  rules: identifier-hypothesis (cardinality == row count, integer or string
  column) and sentinel-hypothesis (a value from {±9, ±99, ±999, ±9999} is
  the column min or max and covers ≥1% of non-null rows). Labelled
  `hypothesis`, never provider evidence.
- D-026: archive safety limits are constructor parameters with defaults:
  max entries 10,000; max single-file 512 MiB; max total uncompressed 2 GiB;
  max compression ratio 200:1 (checked only above 1 MiB compressed).

## Registry rows appended (lineage per PRD-001 §10)

QuestionRecord (no parents) → KaggleCapture (parent QuestionRecord,
restricted) → SourceManifest (parent KaggleCapture) → TableProfile (parent
SourceManifest) → EvidenceBundle (required parent KaggleCapture; optional
SourceManifest, TableProfile) → SemanticMap (parent EvidenceBundle). All
produced by `intake-coordinator`; all but KaggleCapture readable by
`design-harness`, destination `design`.

## `contracts.py`

`ContextClass` (7 values, PRD §5.7); `SemanticStatus` (9 values, PRD §8)
with `AVAILABLE_STATUSES` = {evidenced, hinted, hypothesis};
`DATASET_SLOTS` (7 names, PRD §7.1); `COLUMN_SLOTS` (8 names, PRD §7.2);
`SemanticSlotV1` (status + optional value + evidence IDs; validators:
evidenced/hinted ⇒ value present and ≥1 evidence ID; hypothesis ⇒ value
present and NO evidence IDs; unavailable ⇒ no value, no evidence);
`DatasetSemanticsV1` / `ColumnSemanticsV1` (exactly the defined slot set —
a missing slot is a schema error, PRD §8); `IntakeSubmissionV1`
(schema_version literal, question 1–10,000 chars, optional context text,
normalized kaggle_ref per D-024, idempotency key). All frozen/strict/
extra-forbid, reusing shared `Identity`.

## `archive.py`

`ArchiveAdmission` dataclass: per-entry `ResourceDecision` (name,
classification, reason) + overall `safe`. Classifications per PRD §5.5:
`table` (csv/tsv/parquet), `document` (txt/md), `metadata` (json/yaml),
`unreadable` (xls/xlsx/pdf/images), `withheld` (notebooks/scripts),
`unsafe` (executables, pickles, macro workbooks, nested archives), plus
safety refusals: path traversal, absolute paths, symlinks, encrypted
entries, per-file/total/ratio bomb limits (D-026). Unsafe entries never
extract; `extract_admitted` returns bytes only for table/document/metadata
classes. Zip only in V1 (the PRD's downloaded archive format).

## `profiler.py`

`profile_table(data, media_type, profiler_version) -> dict` (canonical-ready
payload): PRD §6 measured facts — row/column counts, per-column physical
dtype, null count/rate, cardinality, numeric min/max/mean/std/quantiles
(.25/.5/.75), non-finite counts, categorical levels under a 20-level cap,
date min/max for temporal dtypes, duplicate-row count, single-column
uniqueness, constant/all-null flags, input-bytes SHA-256, profiler version,
and D-025 hypotheses. polars only; csv/tsv/parquet; identical bytes ⇒
identical payload ⇒ identical `content_hash` (replay determinism).

## Budget

intake package ≤ 500 logical lines this task (of its 1,200 allocation);
tests ≤ 550; registry delta declarative.
