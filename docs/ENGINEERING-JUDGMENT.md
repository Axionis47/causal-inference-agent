# Engineering Judgment

This document explains four decisions that changed after contact with real
data, providers, or system behavior. Each case names the observed failure,
rejected alternatives, verification evidence, and remaining limitation.

The detailed implementation history is preserved in the append-only
[decision ledger](LEDGER.md). Decision identifiers below point readers to that
source record.

## Artifact design is the durable control layer

In an AI workflow, the model proposes and the artifact records. Prompts and model
responses are temporary; validated artifacts are the durable truth passed between
stages and reviewed by people.

Good artifact design makes a workflow testable through schemas and validators,
auditable through provenance and lineage, reproducible through hashes and
immutable revisions, safe across retries through explicit failure states, and
understandable through accurate quantities and units.

The general rule is: the model may propose; deterministic code validates; only
validated artifacts move downstream. Artifact design is part of the workflow's
correctness boundary, not merely a storage or formatting choice.

## 1. External APIs contradicted assumptions

**Context.** The intake and model layers depend on Kaggle and Vertex AI. Both
providers sit behind narrow adapters because provider details should not become
product contracts. The system still needs an exact dataset version, a stable
model profile, structured JSON, and sanitized failures.

**Observed failure.** The first live Kaggle call failed before download. Kaggle
had retired `GetDatasetStatus`, so the pinned client returned 404 for every
dataset. The first live Vertex call also failed. A seed derived as an unsigned
32-bit value exceeded the provider's signed integer range. A later live design
run returned `{}` repeatedly because the gateway sent `{"type": "object"}`
instead of the registered draft schema. Temperature zero made that bad result
consistent, not correct. These findings are recorded as D-060, D-061, and D-063
in the [ledger](LEDGER.md).

**Constraints.** Dataset identity could not become “latest available” at an
unknown time. Removing the model seed would weaken replay behavior. Parsing
free-form text after generation would bypass the structured-output boundary.
Provider exception bodies could not be copied into events because they may
contain credentials or request details.

**Options considered.** Keeping the retired endpoint and retrying it would turn
a permanent API change into repeated failure. Dropping the seed would avoid the
range error but lose a declared control. Sending a generic schema and trusting
prompt prose would make strict parsing an afterthought. Switching providers or
models on failure would hide the original execution contract.

**Decision.** The [Kaggle adapter](../src/causal/runtime/kaggle_live.py) now
resolves `currentVersionNumber` through an exact reference match on the dataset
list surface. Provider status is recorded as unknown because Kaggle no longer
offers that fact. The [Vertex gateway](../src/causal/shared/gateway.py) masks the
derived seed into a non-negative signed 31-bit range. The shared
[result schema builder](../src/causal/shared/agenttask.py) merges the actual
draft schema into the task result schema, including `$defs`. Provider failures
map to stable application codes, and exception bodies stay outside product
events.

**Verification.** The
[Kaggle adapter tests](../tests/runtime/test_kaggle_live.py) cover exact matches,
missing references, sanitization, and a gated live probe. The
[gateway tests](../tests/shared/test_gateway.py) pin the signed seed case and
provider failure taxonomy. The
[design graph test](../tests/design/test_graph.py) verifies that the full draft
schema reaches the gateway. Live smokes remain separately gated so deterministic
tests do not claim live-provider success.

**Remaining limitation.** A narrow adapter reduces the effect of API drift but
cannot prevent it. Release checks still need live probes for provider surfaces,
and those probes require credentials, network access, and cost approval.

## 2. The harness had evidence but did not give it to the model

**Context.** Intake captured a Kaggle data dictionary and committed it as
evidence. Design workers received an allowlist of evidence identifiers and had
to report whether each source supported a claim. The ask gate could question a
human only after available sources were exhausted.

**Observed failure.** The design harness loaded the evidence bundle, extracted
each identifier, and discarded the associated text. The prompt therefore named
the data dictionary but did not show its contents. Workers correctly reported
the source as not offered, and the ask gate questioned the user about facts the
system already held. On the NSW dataset, the omitted description contained
column meanings, units, treatment encoding, and measurement timing. A related
problem also hid the measured table profile from workers that had to infer
column scale and grain. These findings are recorded as D-102 and D-103 in the
[ledger](LEDGER.md).

**Constraints.** Passing every source to every task would create a different
failure. Large per-column evidence could overwhelm the prompt and leak sibling
context across worker scopes. The model's earlier guess could not count as
evidence for its later claim. The system also had to distinguish a missing
source from a source withheld by a harness budget.

**Options considered.** Automatically accepting semantic cards as evidence was
rejected because a model cannot corroborate itself. A new retrieval tool was
not justified while the relevant text was already loaded in memory. Rendering
only identifiers preserved a small prompt but removed the facts needed for the
task. Rendering all text without a total limit made prompt size depend on
dataset width.

**Decision.** The shared
[evidence block](../src/causal/shared/agenttask.py) now receives a mapping from
evidence identifier to text. Dataset-wide evidence reaches applicable tasks,
while column evidence reaches only tasks assigned that column. One item is
limited to 4,000 characters, and all evidence text in one task is limited to
24,000 characters. An item that cannot fit is rendered as withheld because the
task evidence budget is full. Measured table-profile facts also become
classified evidence for the tasks that need them.

**Verification.** The
[design graph tests](../tests/design/test_graph.py) inspect the prompt, compare
rendered identifiers with the envelope allowlist, and verify column scope. The
recorded live rerun changed the relevant source from not offered to evidenced,
reduced unnecessary requirement rows, and produced questions only for facts
the data dictionary could not settle. Validation still rejects citations
outside the task allowlist.

**Remaining limitation.** Evidence selection uses explicit scope and size
rules, not semantic retrieval. A much wider dataset may still withhold relevant
text after the total budget fills. A retrieval loop should be added only after
an evaluation shows that scoped hydration fails often enough to justify a new
permission, budget, trace, and failure surface.

## 3. “The row is the unit” must be an assertion

**Context.** Several supported estimators require a unit identifier. The unit
controls independence assumptions, cluster-aware uncertainty, and some
leave-one-unit-out sensitivities. Many cross-sectional CSV files contain one
row per person but no unique identifier column.

**Observed failure.** A real NSW table had no unique single column. It also had
byte-identical rows, so the data alone could not distinguish two similar people
from one duplicated person. The design correctly refused the method because no
unit identifier role was available. Simply observing that no column was unique
could not justify treating the row as the unit. A panel file that lost its
identifier would present the same structural symptom and require very different
standard errors. The audit and resolution are recorded as D-104 and D-105 in
the [ledger](LEDGER.md).

**Constraints.** Estimation code reads the unit column in multiple method
paths. One randomized sensitivity uses it as the leave-one-out key. Any repair
had to remain visible in the approved design, survive preparation, and reach
estimation through normal role resolution. It could not quietly weaken
clustering or convert duplicate measurements into independent units.

**Options considered.** Removing `unit_identifier` from method requirements
would replace an early refusal with a downstream missing-column error. Adding
per-estimator fallbacks would spread special cases across method code and risk
plain standard errors where cluster-aware errors were required. A sentinel in
`key_columns` would force every preparation and estimation consumer to learn a
second meaning. Inferring row identity from the lack of a unique column was
scientifically invalid.

**Decision.** Row identity is bound only when three facts hold together: no
committed unit role exists, the profile has no unique single column, and the
approved design intent explicitly states `one_row_per_unit`. The third fact is
the assertion. The [design harness](../src/causal/design/harness_nodes.py) adds
`__row_unit_id` to the role ledger, frame contract, and visible design
assumptions. The [stabilization stage](../src/causal/preparation/stabilize.py)
materializes it from source order before any repair can drop or reorder rows.
Estimation then uses the normal unit-role path with no fallback.

**Verification.** The
[design binding tests](../tests/design/test_graph.py) verify that every required
payload names the row unit. A strict read-back test in
[design validation](../tests/design/test_validators.py) confirms that the
harness-authored claim remains contract-valid. The
[stabilization test](../tests/preparation/test_stabilize.py) verifies source
order, idempotence, and the condition that activates materialization.

**Remaining limitation.** The harness adds this claim after the model payload
passes its main validation wall, so the synthetic concept is not part of the
model-authored causal context. The assumption is visible and human-approved,
but a future contract revision should make harness-authored claims pass an
equivalent explicit wall.

## 4. Delete unused infrastructure instead of advertising it

**Context.** The design specification originally registered eleven read-only
model tools. They covered inventory, semantic evidence, provenance, diagnostic
previews, and validation helpers. The surrounding envelopes and permission
types made the surface look available.

**Observed failure.** A live audit found no production caller for the tool
router. Allowed tool identifiers were not rendered into the prompts, and the
Vertex gateway had automatic function calling disabled. Workers instead
received harness-hydrated context. The registered tool subsystem therefore
added code, handlers, and tests without affecting one model decision. At the
same time, the design and shared scopes were close to their explicit complexity
budgets. This finding and the deletion are recorded as D-100 in the
[ledger](LEDGER.md).

**Constraints.** Wiring a real tool loop would require more than calling a
handler. It would need model-visible schemas, permission enforcement, call
budgets, trace events, sanitization, retry policy, correction behavior, and
evaluations for denied or malformed calls. Raising complexity limits would not
make the unused code useful. The current hydrated tasks already needed the same
evidence for deterministic walls.

**Options considered.** Keeping the subsystem “for the future” would continue
to advertise a capability that did not exist. Wiring it quickly would create a
second execution loop without evidence that the round trip improved quality or
prompt size. Raising line budgets would hide the immediate pressure while
preserving dead branches. Deleting unrelated validation or scientific rules
would reduce visible line counts at the cost of actual behavior.

**Decision.** The unused router, handlers, tests, and separate tool registry were
removed. The [task registry](../registries/design-tasks.v1.json) assigns every
design task a zero tool-call budget. This records the capability boundary
without claiming runtime support. Design prompts state that the model has no
tools. Context continues through the existing typed task envelope and evidence
renderer.

**Verification.** The
[tool registry tests](../tests/design/test_packs.py) verify that every design
task receives an empty tool allowlist and that invalid registrations fail. The
[complexity checker](../tools/budget_check.py) continues to measure production,
test, declarative, module, and function budgets. The normal graph tests prove
that design tasks still converge with hydrated evidence and no model-facing
tool loop.

**Remaining limitation.** Hydration is not always better than retrieval. A
future tool should return only when measured cases show that relevant context
cannot fit within task budgets or changes too often to hydrate safely. Its
return would require a complete permission model, observable call loop, bounded
failure policy, and evaluation cases before any prompt may advertise it.

Return to the [README](../README.md), review the system in
[Architecture](ARCHITECTURE.md), or inspect the task lifecycle in
[Harness and Evaluations](HARNESS-AND-EVALUATIONS.md).
