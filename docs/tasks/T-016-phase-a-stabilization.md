# T-016 — Phase A: row identity, stabilization engine, impact, shared frame I/O

Status: frozen for implementation (isolated-worktree agent; merge by architect)
Owning PRD: PRD-003 §8, §9, §12 (row-stabilization subsections), as amended by §24
Depends on: T-015

## 1. Deliverables

### 1.1 `src/causal/preparation/stabilize.py` (≤ 300 logical)
- `source_row_id(csv_hash, row_number, row_content_hash)` per §24.4: the selected CSV
  artifact hash + 1-based parsed row number + a canonical content hash of the parsed row
  values (sha256 over the shared canonical JSON array of the row's values, in schema
  column order, via the pinned polars parse). Identical duplicate rows stay distinct via
  row number.
- §9.1 ordered disposition engine over a `pl.DataFrame`: (1) parser policy (rows polars
  cannot represent under the profile → `unusable_corrupt_record` — via typed parse
  errors, no byte scanner); (2) population eligibility; (3) timeframe eligibility;
  (4) required identity/grain; (5) method-required non-imputable roles (§9.3
  four-condition test); (6) duplicates/key collisions (§9.4 — byte-identical detection
  via row content hash + row number; conflicting duplicates on a required key without an
  approved resolution rule → `unresolved_conflict`); first applicable rule = primary
  disposition, later conditions = warnings.
- Eligibility/unusable rules evaluate ONLY registered rule kinds already named in the
  method-pack `eligibility_rule_vocabulary` (read the T-011 vocabulary; closed evaluator:
  set membership, numeric/date range, nonnull, timeframe window). An unknown rule kind
  raises the typed conflict path — never a silent skip.
- Retained-set freeze inputs: ordered retained `source_row_id` list (canonical JSON →
  object store), `row_set_hash` = sha256 of that canonical list, counts, unique units.
- `StabilizationRecordV1` assembly from the above (T-015 model).

### 1.2 `src/causal/preparation/impact.py` (≤ 200 logical)
- `DimensionImpactV1` computation across the pack's `deletion_impact_dimensions`
  (dimension = column levels; derived pre/post and cutoff-side dimensions from manifest
  role columns).
- Method-structure validation for all four packs from the preparation overlay: RCT arms
  (both arms retained, ≥ minimums), AIPW one-row-per-unit + observed treatment/outcome,
  DiD group-time cell support (per-cell minimums, pre/post counts), RDD both cutoff
  sides retained. Invalidation → `runnable` | `not_runnable` | conflict classification
  with stable codes.

### 1.3 `src/causal/shared/frames.py` (≤ 120 logical, shared scope)
Content-addressed frame artifacts: deterministic `pl.DataFrame` → CSV bytes
(`write_csv`, fixed formatting), sha256, `ObjectStore` put at `objects/{hash}`;
`read_frame(locator, schema)` re-applies the stored column dtypes and verifies the hash
on reopen. (PRD-004 reads the same frames; dtype fidelity comes from the frame
artifact's schema metadata, not CSV inference.)

### 1.4 Tests (lean — see §2 cap; ~420 logical, `tests/preparation/`)
- Hypothesis properties (narrow): identical CSV bytes → identical id set + `row_set_hash`;
  any value change → different row hash; duplicates distinct; exactly-one-disposition.
- Table-driven §9.1 ordering cases; §9.3 all-four-conditions matrix; §9.4 duplicate/
  collision cases incl. `unresolved_conflict`.
- Per-method structure fixtures (4 packs: pass + not_runnable + conflict each).
- frames round-trip with dtype fidelity + reopen-hash verification (docker MinIO).

## 2. Constraints
Budgets: preparation +≤500 this task; shared +≤120; tests +≤420 (D-073 test-headroom cap); modules +3. polars
allowed HERE (first preparation use); still no langgraph/model/CLI imports; no
cross-import from `causal.design`. Pre-code projection + one rethink per SC §14.1.2.

## 3. T-015 absorption notes
- The substrate models live exactly as committed at `762bb5f` (contracts 295, plans 203,
  packs 95, shared/receipts 97) — build against the code, and where this spec's field
  expectations differ from the committed models, the committed models win.
- There is NO migrations ledger table; nothing in this task touches migrations.
- Module count 56/68; largest allowed module 350; largest function 75.
