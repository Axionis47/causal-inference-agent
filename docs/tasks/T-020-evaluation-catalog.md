# T-020 — Evaluation catalog: evals/catalog.v1.yaml

Status: frozen for implementation (freeze committed after T-019 acceptance)
Owning contract: SC §10.5 / §10.5.1 / §10.5.2, §15 #28–33; PRD eval tables
(PRD-001 EV-P1-001..005, PRD-002 §20.7 EV-P2-001..008, PRD-003 §18.7 EV-P3-001..007,
SC §10.5.1 EV-SYS-001..007 and EV-E2E rows)
Depends on: T-019

## 1. Deliverables

1. `evals/catalog.v1.yaml` (~420 declarative lines) — one `EvaluationRegistrationV1`
   row per registration for the existing surfaces (EV-SYS, EV-E2E, EV-P1, EV-P2,
   EV-P3; EV-P4/EV-P5 land with their waves — record that scoping in a header comment).
   Field set per SC §10.5 (id, kind, owner/boundary, fixture focus, trigger, hard pass
   condition, case budget per §10.5.2). Content is copied FAITHFULLY from the SC/PRD
   tables — no invented ids, no paraphrase that changes a pass condition. PRD-003 rows
   read through Amendment 2 (D-076): EV-P3-003/004 register as deterministic-compiler
   fixtures (grouping, coupling, coverage, fan-in rejection — no live model leg);
   EV-P3-007's restart leg registers as rerun replay (no duplicate artifacts, same
   terminal outcome); no preparation model surface exists — note this inline on each
   re-scoped row. Pilot findings D-060..D-071 map to case notes on the registrations
   they exercise (EV-SYS-003 seed/schema; EV-P2-001 intent vocabularies; EV-P2-002/004
   slot + evidence discipline; EV-SYS-006 handoff; D-071's two wall gaps as named
   future cases on EV-P2-004).
   **File format:** the contract names `.yaml`; the pinned stack has no YAML parser
   (D-010). Author the file as JSON-syntax content (valid YAML subset) so `json.loads`
   reads it; record this as the format decision in the header comment.
2. `tests/evals/test_catalog.py` (≤ 80 logical, D-075 rebalance) — parses the catalog
   with `json.loads`; asserts: every id unique and matching `EV-(SYS|E2E|P1|P2|P3)-\d+`;
   exact id sets per surface against the PRD tables (hardcoded expected tuples);
   `kind` in the closed §10.5 set; per-id case count ≤ 8 (non-e2e); every
   `required_eval_ids` value emitted anywhere in `src/causal/` (the EVAL_* constant
   tables in intake/design/preparation) exists in the catalog.

## 2. Constraints
Declarative +≤430 (≤3,500 total); tests ≤80 FIRM; ZERO production modules or lines.
No new dependency (no pyyaml). Pre-code projection; one rethink. Where this spec
conflicts with committed constants or tables, the committed code wins — record
deviations.
