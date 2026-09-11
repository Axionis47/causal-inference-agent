# T-038 verification

The standalone analysis library is implemented and verified at its public
boundary. The existing agent has not been redesigned. The user clarified that
legacy regression counts are not the acceptance criterion; verification here
focuses on the new capability contract and actual numerical behavior.

## Acceptance evidence

- 27 public-interface cases pass: dependency-free discovery and guidance,
  generated column/type/unit requirements, incomplete scientific context,
  unsupported configurations, input incompatibility, outcome leakage rejection,
  diagnostic obligations/applicability, frozen approval, stale data/configuration/
  implementation rejection, deterministic replay, and explicit failures.
- 24 method-boundary cases pass: randomized formula/covariance/adjustment,
  independent AIPW ATE/ATT scores, simultaneous/staggered DiD, sharp RDD,
  every advertised sensitivity, outcome-kind handling, source column names,
  diagnostic failure isolation, and supporting data prerequisites. Numerical
  expectations use arithmetic or direct reference-library calculations.
- The combined focused run is 51 passed, 38 deselected. The 16 warnings are
  statistical-library disclosures about deliberately saturated DiD fixtures;
  they are not omitted from the test output.
- The documented agent-independent example completes and recovers its known
  effect of 3.0. A successful primary remains available when a planned diagnostic
  fails; nonconverged outputs remain failed and are retained for audit.
- Ruff passes over source, tests, tools and root test configuration. Strict mypy
  passes 118 production source modules; colocated test code is excluded from
  production typing and remains explicitly counted as tests.
- A wheel builds with all method source, guidance and historical adapter resources.
  The built analysis assets were compared to the current source. Discovery and
  guidance import no statistical libraries, database clients or model services.

Commands for the focused acceptance run:

```sh
uv run pytest -q src/causal/analysis/tests/test_interface.py src/causal/analysis/methods -k 'test_interface or public or selected_ten_fold_sensitivity or selected_anticipation_sensitivity or reported_fold_leakage or simultaneous_primary_survives'
uv run ruff check src tests tools conftest.py
uv run mypy src/causal
uv build --wheel
```

The numerical library's accepted configurations and agent-facing operations are
explained in the [package documentation](../../src/causal/analysis/README.md).
Persisted estimation schemas and old numerical defaults remain historical
integration contracts; new retrieval does not advertise their unsupported branches.

## Budget accounting

The [machine-readable report](../../evals/reports/T-038-analysis-budget.json)
remains `blocked_complexity_budget`. No limits were raised. Colocated tests count
as tests; runtime guidance and resources count as declarative material.

| Dimension | Actual | Limit |
| --- | ---: | ---: |
| Analysis (retained estimation allocation) | 5,672 | 4,000 |
| Production total | 20,759 | 16,500 |
| Tests and tooling | 25,228 | 13,500 |
| Declarative | 5,419 | 4,900 |
| Grand total | 51,406 | 35,000 |
| Production modules | 118 | 86 |

Shared/design and individual module/function limits also remain exceeded. The
analysis allocation was 3,604 lines in the captured working-tree baseline;
this implementation adds 2,068 analysis lines. New boundary contracts and
retained legacy integration are both included; the refactor does not claim to
have eliminated that coexistence cost. The report also preserves previously
unassigned generated output paths instead of hiding them. This is functional
library acceptance, not a complexity-budget or product-release pass.

## Local environment and worktree

The host repeatedly marks the editable installation's `.pth` file hidden, which
Python skips. Same-call editable import after clearing that flag works, and
wheel packaging succeeds. No source-level workaround was added for this host
behavior. Source tests use the declared pytest `pythonpath`.

Existing unrelated working-tree changes and historical artifacts were preserved.
The frozen task plan was committed before implementation. Implementation changes
remain in the shared worktree to avoid bundling the user's pre-existing work into
a broad checkpoint commit.
