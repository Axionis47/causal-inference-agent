# T-035 — Real-data figure semantic mapping

Status: frozen for implementation
Owning PRDs: PRD-004 §17; PRD-005 §11–§14
Depends on: T-031, T-033, T-034

## 1. Finding

Visual inspection of the actual Groupon and state-panel deliveries exposed two semantic encoding
defects that dimension-only tests did not catch. The shared primary-interval builders place the
estimate in `y_value` and a row index in `x_value`, while `estimate_forest.v1` maps `x_value` to
the quantitative effect axis; the point therefore appears at zero while its interval is correct.
DiD group-time and cohort-event-time points use a constant series ID, so the line template joins
distinct groups into a sawtooth. Its numeric year/event-time field is also declared temporal,
which Vega interprets as epoch milliseconds.

## 2. Deliverables

1. Put every primary estimate in the forest's quantitative `x_value`, and use its contrast ID as
   the nominal series label, for all four method packs through one shared builder.
2. Give each DiD group/cohort its own series for group-time and event-time figures.
3. Declare the DiD trend template's numeric x-axis quantitative, preserving all frozen values and
   references.
4. Add semantic tests for the estimate coordinate, DiD series separation, and x-axis scale; then
   rerun and visually inspect the actual AIPW and DiD figures.

### Amendment 1 — builder-specific axis semantics

The corrected actual renders exposed one remaining semantic-label defect: the shared FigureData
default describes both axes as the outcome, so numeric year/event-time and nominal contrast axes
receive outcome labels. Registered builders must override those defaults for time, event time,
count, and contrast axes. Suppress the duplicate non-color group legend while retaining that
channel as the required redundant group encoding. This changes labels and legend presentation
only; it does not change a frozen value, unit conversion, estimator, diagnostic, or reference.

## 3. Constraints and verification

No estimator or diagnostic calculation changes. Production code net non-positive; tests no more
than +12 logical lines; declarative line count unchanged; no new module. Run focused presentation
and adapter tests, budget checker, ruff, mypy, and the full suite.
