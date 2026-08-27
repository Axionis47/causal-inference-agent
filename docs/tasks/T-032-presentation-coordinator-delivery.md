# T-032 — Presentation coordinator, five gates, bundle, CLI delivery, EV-P5 rows

Status: frozen for implementation
Owning PRD: PRD-005 §6, §15, §16, §17.2, §19, §20; SC §1.1
Depends on: T-031

## 1. Deliverables

1. `src/causal/presentation/nodes.py` (≤ 250 logical): sequential coordinator
   (entry → manifest → curate → compile fan (sequential, plan order) → render →
   final validation → bundle commit), `run_presentation(deps, *, analysis_id,
   handoff ids, stage_run_id) -> PresentationRunResult`; the five §16 gates in
   order (gate machinery from T-030/T-031 validators; no later gate waives an
   earlier failure); §15 presentation-summary assembly (every substantive sentence
   cites a claim statement, qualification, or frozen artifact); `PresentationRunV1`
   row upkeep terminal on every exit; §17.2 typed status destinations
   (needs_template / needs_layout_revision / blocked / failed /
   failed_observability); §19 events flush-gated; PRD-002 CausalGraphView reused at
   its approved hash (placed, never rerendered).
2. CLI delivery (cli scope ≤ 500 binding; `cli/main.py` is AT the 350 cap):
   new module `src/causal/cli/present.py` (≤ 55) implementing
   `causal presentation ANALYSIS_ID --bundle-id ID --expected-bundle-hash HASH
   [--output-dir DIR]` per §20 — exact bundle + hash required (never implicit
   latest); `--output-dir` must be absent or empty; copy only committed bytes;
   verify every hash post-copy; failure → blocker, bundle untouched. Wire into
   main.py net-zero (trim to offset any added dispatch lines; main.py stays ≤350).
3. `runtime/composition.py`/`failures.py` (runtime ≤ 800 binding): `present()`
   dispatch — latest estimation run terminal `complete` with recorded PRD-005
   handoff → run presentation (revision bump idiom); status row + next command.
4. `evals/catalog.v1.yaml` (+~45 declarative): EV-P5-001..006 byte-faithful from
   PRD-005 §17.2's table, D-079 conventions; `_header` scoping note updated (no
   pending surfaces remain); test id-tuples extended.

## 2. Tests (≤ 300 logical)

E2e: scripted estimation `complete` fixture (T-026 machinery + a fake-gateway
curator) → `run_presentation` → `complete`/`complete_with_qualifications`, bundle
committed, summary cites resolve, delivery command exports to an empty dir with
verified hashes; occupied dir refused; wrong hash refused; unreportable claim
fixture → `blocked` at gate 1; needs_template path terminal + typed; rerun replay
(no duplicate artifacts); runtime dispatch + status next-command; qualification
placement (mandatory qualification beside its result — assert in summary/bundle).

## 3. Constraints

presentation ≤ 1,280 total after (+250; ceiling 1,500); cli ≤ 500; runtime ≤ 800;
declarative ≤ +45; tests ≤ 300; modules +2. Delivery never replans/recompiles/
rerenders. Wave 2 shares one rethink. Committed code wins; record deviations.
