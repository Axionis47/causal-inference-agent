# T-029 — Estimation runtime dispatch, CLI status/render, EV-P4 catalog rows

Status: frozen for implementation
Owning PRD: PRD-004 §20.6, §24 #49; SC §1.1 (status next-command), §10.5
Depends on: T-026..T-028

## 1. Deliverables

1. `runtime/failures.py` / `runtime/composition.py` (runtime ≤ +70; scope ≤ 800
   total): `estimate()` dispatch in the `prepare()` idiom — latest preparation run
   terminal `prepared` with a recorded PRD-004 handoff → run estimation at
   estimation_revision n (crash re-entry n+1 via artifact replay); broad-exception
   guard already generalizes — extend its coverage to the estimation call. `status()`
   gains the estimation stage row + exact next command for every non-terminal state
   (D-069b discipline).
2. `cli/render.py` (+≤ 20; cli ≤ 500 total): render `EstimationRunResult` through
   the existing render protocol (outcome, conflict code, next command). `cli/main.py`
   is at the 350 cap: any status-map addition must be net-zero there (value-map
   entries only; no new lines — restructure inside render.py if needed).
3. `evals/catalog.v1.yaml` (+~70 declarative): EV-P4-001..010 registered
   byte-faithful from PRD-004 §20.6, read through Amendment 1 (D-083): EV-P4-010's
   restart leg = rerun replay, concurrency clauses noted as upper bounds; kind/
   case_budget derivation per the D-079 conventions; `_header` scoping note updated
   (EV-P5 remains the only pending surface).
4. `tests/evals/test_catalog.py`: expected-id tuples extended to EV-P4; the
   required_eval_ids sweep now also covers `src/causal/estimation/` EVAL constants.

## 2. Tests (≤ 100 logical, incl. the catalog test delta)

Dispatch: prepared→estimation golden path via fixtures; not-yet-prepared refusal;
crash re-entry revision bump; status next-command strings for running/terminal
estimation states; CLI render snapshot for `complete` and `design_conflict`.

## 3. Constraints

Zero estimation-scope lines. runtime ≤ 800, cli ≤ 500 binding. tests ≤ 100.
Wave rethink shared. Committed code wins; record deviations.
