# Adjustment: fixed context to causal designs

The desk assembles and admits context through the shared matrix and this family's endpoint declarations. The lane reads that fixed handoff, builds a separate evidence ledger, identifies and checks rival analysis designs, freezes each valid design, and executes it with DoWhy or the installed statsmodels IV adapter. A lane never interviews the person or writes their durable memory.

```text
Desk: material + human clarification + matrix admission + explicit run authorization
  |
  +-- immutable handoff (context identity, dataset-byte digest, admitted requirements)
        |
        +-- load -> case -> pair -> seed
        +-- mechanism -> relate -> bounded probe
        +-- rival graphs -> identification endpoints -> support checks
        +-- eligible estimator selection -> integrity-sealed designs
        +-- execution for each frozen design and contrast
        +-- falsifications + separate sensitivities + post-freeze graph diagnostics
        +-- target-aware interpretation -> figures -> canonical result

Any missing context, unidentified target, unsupported method, inadequate support,
exhausted reasoning or execution failure is recorded with its specific reason.
The desk owns the conversation and any subsequent authorization.
```

## One declaration source

`lane/knowledge/endpoints.yaml` defines twelve executable semantic endpoints and one identification-only diagnostic, including their target, context requirements, assumptions, trajectory, checks, estimators and stop reasons. There is no universal set of exactly ten DAGs. The endpoint catalogue supplies the desk's decisions and the lane's admission checks. `rungs.yaml` defines the actual ladder; `space.yaml` describes its dimensions; `rivals.yaml` defines enumeration limits and identification methods, including default, minimal, maximal and efficient adjustment policies with road/set deduplication. [motifs.yaml](lane/knowledge/motifs.yaml) retains the ten requested reusable graph examples, natural mediation and modifier variants, and blocked adjustment traps. These are illustrative graphs with explicit evidence and assumptions, not extra endpoint eligibility rules. The independent former decision checklist and belief/refuter shapes are removed.

The executable classes are unadjusted ATE, backdoor ATE/ATT/ATC, general-adjustment ATE, binary Wald complier LATE, scalar homogeneous IV, multiple-instrument homogeneous IV, conditional homogeneous IV, linear front-door total effect, linear natural indirect effect, linear natural direct effect, linear baseline-modifier effects, and positively supported structural zero. See [the scope and validation guide](evals/README.md) for each class's conditions and limitations.

## Evidence and freeze

Handoff context stays read-only. The lane ledger records literal supplied relationships and timing, causal hypotheses, their source addresses, status and accepted revision history. A drafted model statement or a covariance does not become a confirmed causal fact. Hidden common causes remain in the graph. No unspecified edge can establish structural zero; that endpoint requires a positively supplied absence of every treatment-outcome causal path.

Before freeze, outcome access is restricted to univariate scale/shape description. Outcome associations, redundancies, cells and other relational diagnostics are refused, including outcome-covariate pairs. The lane can investigate unresolved non-outcome relationships within its declared budget. At most four unsettled structural entries yield sixteen rival graphs; at most eight executable designs may survive. Every admitted identification road for the requested target becomes a distinct candidate. Excess uncertainty is an explicit refusal.

Support checks and estimator eligibility happen before outcome fitting. Singleton eligible estimators are selected by code; other picks use exact offered names and design identities. Freeze seals the graph, certificate, target, context and ledger revision, complete-case row identities and data fingerprints, weight recipe, estimation parameters, inference and predeclared fallback. Execution cannot change them. Every fit checks integrity, and percentile bootstrap refits the complete estimator on resampled raw rows.

## Results and conversation

The result preserves all designs and candidates, accepted ledger history, candidate transitions and obligations, episode visits and rejected tries, execution attempts, estimates, falsifications, sensitivity grids, comparisons and terminal outcomes. Source context facts, admission and dataset-byte identity travel with the result. Every estimate and diagnostic is design-qualified; failed attempts remain visible even when a frozen fallback succeeds. A failed lead design is not replaced silently by a successful rival.

For a conditional target, freeze names every requested contrast, modifier and subgroup in `target.primary_summaries`. The overall effect is a secondary reference in `target.reference_summaries`. Completion requires every requested subgroup, including its inference, to succeed. A missing or unsupported subgroup remains an explicit failure and the overall mean cannot replace it.

Comparisons require identical complete target definitions, contrasts, scales and frozen analysis populations. Different complete-case populations or subgroup targets cannot receive an agreement score. Post-freeze independence diagnostics and falsifications can expose concerns but do not select another causal graph. Sensitivity returns ranges and simulated strength points, with no pass verdict. The desk can discuss these artifacts, clarify the pre-analysis context, and authorize a fresh run.

## Files

```text
family.yaml             family-level scope and matrix claim kinds
endpoints.yaml          authoritative reasoning endpoint declarations (under lane/knowledge)
design.py, handoff.py    advisory family block and fixed handoff construction
probes.py, overlap.py    desk-facing data support probes
postviz.py              design-qualified graph and comparable-estimate figures
lane/contracts.py       graph/certificate/candidate/frozen-design and judgement records
lane/reason.py          admission, ledger reasoning, rival identification and freeze
lane/support.py         pure row fingerprints and numerical comparison identities
lane/engine/            graph identification and frozen numerical adapters
lane/execute.py         fits, declared fallback, attempt records and diagnostics
lane/artifacts.py       interpretation, canonical result and reviewable report
lane/knowledge/         endpoints, motifs, rungs, space, estimators, checks, rivals,
                        falsifications and sensitivity declarations
evals/                  real-source cases, retained handoff and endpoint evaluators
tests/                  pure policy invariants and actual numerical engines
```

A run leaves `table.csv`, `result.json`, `artifacts.json`, `figures.json` and `report.md` under `.artifacts/runs/<dataset>-dowhy-<id>/`. Frozen designs are held in the result's `designs` map.

## Validation

```bash
LANGSMITH_TRACING=false uv run pytest causal_agent/families/adjustment/tests -q
GRPC_DNS_RESOLVER=native uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"
uv run python -m causal_agent.evals.trace --trace <trace-id> --thoughts
```

The generated evaluation datasets and their generators have been removed. Existing real-source fixtures remain; future evaluation uses Kaggle data with documented study context. Unit tests establish code invariants and numerical behavior. Real-source runs and inspected traces are required to establish endpoint acceptance coverage.

The first code-assisted selection increment gives `pick_estimator` three sandboxed Python calls when more than one method is eligible. Scripts can inspect treatment and identified adjustment variables on the admitted analysis rows. The outcome and unrelated columns are absent, and isolated execution blocks access to the full source table. Each selected method must cite a successful diagnostic result with nonempty, finite numerical facts. The existing frozen inference, fallback, falsification and sensitivity policies remain unchanged by this increment.
