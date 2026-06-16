# Data preparation: layering and the FrameRecipe

This is a design note, not yet a build. It records where data-shaping work
should live, why several 2026-06-15 lane fixes are stopgaps in the wrong layer,
and the shape of the missing piece (the FrameRecipe) that would let them move
home. It extends `00-architecture.md` (the build contract).

## 1. The problem

Running real Kaggle datasets through the live spine forced a set of fixes into
the deterministic method lanes: comma-tolerant numeric coercion, categorical
one-hot encoding, collinear-covariate dropping, high-missingness dropping, and
wide-format role recovery. They made 7 of 8 representative datasets reach a good
state, but they pushed work into a layer whose contract is "the lane does the
math, deterministically, and never decides design." Some of that work is not
math; it is normalization (a fact about the bytes) or preparation (a judgment
about structure). Left in the lanes it accretes as safety nets, and each new
dataset shape adds another one.

The contract still holds: **lanes never get agency.** What the work exposed is
that two owners are missing upstream, so the lanes are standing in for them.

## 2. The layering contract

Every data concern has exactly one owner. The verbs are the test: who *knows* a
fact, who *decides* on it, who *transforms* the frame, and who *fits*.

| concern | verb | owner today | owner it belongs to |
|---|---|---|---|
| thousands separators, stray currency, dtype | normalize | lane (`to_numeric`) | input-slice normalize step (S0) |
| which columns are categorical / id / high-missing | know | profile (already) | profile (ProfilingAgent, S2) |
| which columns are valid confounders (drop bad/missing controls) | decide | lane (`usable_covariates`) | the DAG / dossier (S3) |
| reshape wide->long, derive a column, join a file | transform | `resolve.py` heuristic | investigator FrameRecipe (new) |
| build the design matrix (encode categoricals for the fit) | prepare | lane (`encode_design`) | shared design-prep helper, fed by the profile |
| drop collinear columns, derive sharp/fuzzy take-up | fit | lane | the lane (correct) |

Two rows are genuinely lane-intrinsic and stay: collinearity is a property of a
specific design matrix, and sharp-vs-fuzzy take-up is part of the RDD estimator.
The rest is normalization or preparation that the lane is covering for.

## 3. The two missing layers

### 3a. Normalization completeness (ingest, sealed today)

`to_numeric` strips `1,234` -> `1234` inside the lane because the parquet the
input slice writes still carries the separators. The honest home is the
normalize step that produces the parquet: clean once, at S0, so every consumer
sees numbers. The input slice is sealed, so the lane patch stands until it
opens; when it does, the rule is "normalize at ingest, and the lane's
`to_numeric` becomes a belt-and-suspenders no-op."

### 3b. The FrameRecipe (the investigator's preparation half)

This is the real build, and it is the documented gap (`LIVE_AUDIT.md` item 5).
The investigator today ships a **dossier**: read-only understanding (column
roles, assumptions, open questions). It cannot change the frame. So a dataset
whose analyzable form differs from its stored form has no owner for the
transform, and the lane or `resolve.py` improvises:

- wide-format DiD (Card-Krueger `total_emp_feb` / `total_emp_nov`): role
  recovery is a heuristic in `resolve.py` ("the lone binary column outside the
  wide pair is the group"). That is a judgment with no owner.
- derived instrument (cigarettes `salestax = (taxs - tax) / cpi`): cannot run
  at all; the lane only reads existing columns.
- multi-file join (Rossmann sales joined to `store.csv` for the event date):
  cannot run; the input slice picks one winner file.

These are the same class: the analyzable frame must be *constructed*, and that
construction is a judgment the investigator should make and a transform the
runner should apply.

## 4. The FrameRecipe, sketched

A FrameRecipe is the investigator's second output, alongside the dossier. It is
a typed, deterministic, replayable description of how to turn the stored frame
into the analyzable frame. It is data, not code: a fixed vocabulary of ops, each
with provenance, so it can be shown, confirmed, vetoed, and re-applied without
re-running the agent.

```
FrameRecipe
  steps: list[FrameStep]            # applied in order, deterministic
  produces: dict[role -> column]    # the roles the recipe makes available
  provenance: str                   # why, in plain language (artifact, no CoT)

FrameStep (a closed union, no arbitrary code):
  ReshapeWideToLong(pre: [col], post: [col], id_vars: [col],
                    time_name, value_name)
  DeriveColumn(name, expr)          # expr from a whitelisted, safe grammar
  JoinFile(file, on, how, columns)  # within the job's downloaded manifest only
  Cast(col, to)                     # numeric/datetime/categorical
  Filter(predicate)                 # whitelisted comparisons only
```

Invariants (the same ones the rest of the slice keeps):

- **Deterministic and replayable.** No hidden state, no randomness. The same
  recipe on the same frame yields the same prepared frame, so a resumed or
  re-opened run reconstructs it exactly.
- **Closed vocabulary, sandboxed.** No `eval`, no free Python. `DeriveColumn`
  and `Filter` use a small safe grammar (arithmetic, comparisons, the existing
  columns). This is what `LIVE_AUDIT.md` means by "the sandboxed
  frame-preparation half."
- **Provenance as an artifact.** The recipe and a before/after column map are
  written under `analysis/investigator/`, like the dossier. No chain-of-thought.
- **Confirmable at the gate.** The plan gate already parks wide reshape behind
  `confirm_reshape` (`plan_critic/rules.py`, `resume.py` `_ACK_FIELDS`).
  Generalize that: any recipe step that changes the frame surfaces a
  confirmation item the human can accept or veto. Auto-approval still requires a
  fully resolved, high-confidence plan.

### Flow

```
S2  ProfilingAgent     knows: categorical, id, high-missing, wide-pair stems
S2b InvestigatorAgent  emits: dossier (roles) + FrameRecipe (transform)
S3  DesignDetection    reads the prepared schema (recipe.produces), not raw cols
S6  plan gate          surfaces recipe steps for confirmation
S6->S7 runner          applies the recipe deterministically, once, then the lane
S7  MethodLane         fits the prepared frame; no reshape/derive logic of its own
```

The runner applies the recipe; the agent only proposes it; the lane only fits.
That keeps every existing boundary: agents return an `AgentResult`, the runner
commits, the lane stays deterministic.

## 5. Migration

Nothing here reverts a current fix; it relocates ownership over time.

1. **Keep as lane safety nets, demote to belt-and-suspenders:** `to_numeric`,
   `usable_covariates`, `encode_design`. They stop being the primary owner once
   ingest normalizes and the DAG sets the adjustment set, but they stay so a
   mis-shaped frame degrades instead of crashing.
2. **Move the decision, not just the cleanup:** high-missingness exclusion is an
   adjustment-set choice; the DAG/dossier should drop those columns from the
   covariate set at S3, and the lane filter becomes redundant.
3. **Replace the heuristic with agency:** the wide-DiD role recovery in
   `resolve.py` is the first thing the FrameRecipe retires. The investigator
   recognizes the wide panel, names the group, and emits `ReshapeWideToLong`;
   `resolve.py` goes back to single-candidate promotion only.
4. **Leave alone:** `drop_collinear`, RDD take-up. Estimator numerics, correctly
   in the lane.

## 6. The one-line principle

Ingest normalizes the bytes; the profile knows the facts; the investigator
decides and prepares (dossier plus FrameRecipe); the DAG owns identification;
the lane fits, deterministically, and never decides. When a fix wants to live in
the lane but is really normalization or preparation, that is the signal a layer
above is missing, not that the lane needs agency.
