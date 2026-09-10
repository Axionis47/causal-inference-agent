# T-038 — Analysis capability library and execution boundary

Status: implementation in progress. Frozen implementation scope: user-approved
plan of 2026-09-09, replacing the earlier extraction-only proposal.

## Ownership

`src/causal/analysis` owns a standalone, versioned capability library for fixed
causal designs. Public `interface.py`, immutable `contracts.py`, and `runner.py`
expose guidance, assessment, data preflight, compilation and execution. Method
packages (`randomized`, `aipw`, `did`, `rdd`) own typed specifications, diagnostics,
guidance, verified numerical implementations and tests. `common` owns shared
numerical utilities. `integration` preserves historical persistence, reporting
contracts and application coordination. No agent redesign or new estimator.

Each method has one authoritative definition for supported choices, defaults,
requirements, diagnostics and exclusions. Retrieval uses exact method/topic
lookup (overview, configuration, diagnostics, diagnostic_details); schemas and
option lists are rendered from definitions. Scientific predicates are ordinary
Python. Retrieval and assessment import no statistical execution library,
database or model provider; numerical execution requires no LLM.

## Public boundary

- `list_methods()` returns capability summaries and versions.
- `retrieve_guidance(method, topic, draft, diagnostic_id=None)` returns relevant
  choices, requirements, exclusions, unresolved context and follow-up topics.
- `assess_specification(fixed_design, draft)` accepts incomplete proposals and
  returns localized missing-context, missing-data, unsupported-capability,
  contradictory-configuration or incompatible-data issues.
- `preflight(specification, data)` checks observable prerequisites without effects.
- `compile_plan(specification, preflight)` resolves mechanical defaults and freezes
  the configuration, data identity, diagnostic rules, sensitivity choices, seed,
  and capability version. Unknown scientific facts never become defaults.
- `execute(approved_plan, data)` verifies approval bound to the compiled hash,
  capability version and exact data; returns versioned evidence with every planned
  computation's terminal status, estimates, uncertainty, population, measurements,
  limitations, sensitivity results, plotting data and provenance.

Required/optional obligation and applicable/inapplicable/unresolved applicability
are separate diagnostic dimensions. Conditional triggers name their evaluation
stage. Compilation retains inapplicability reasons and blocks unresolved
prerequisites; execution performs all required applicable and selected optional
checks. Runtime failures remain explicit. Preflight means ready to attempt
estimation, never guaranteed convergence or causal validity. Technical retries
preserve the plan, data and seed. Caller owns human questions and approval.

## Checkpoints and acceptance

1. Reconcile existing partial moves and preserve unrelated work; repair imports,
   resources, packaging, colocated test discovery and local fixture ownership.
2. Establish method-owned verified definitions, guidance, assessment, preflight
   and compilation. Exclude no-op subgroup branches, unfinished sensitivities and
   inference configurations whose behavior is unsupported.
3. Connect existing verified numerical implementations; expose consistent evidence
   and document an agent-independent complete journey. Legacy records retain their
   schemas and meanings; legacy catalogs are historical integration resources,
   never an alternative capability source for the new boundary.

Verification covers behavioral honoring of every advertised choice, independent
numerical references, partial and invalid configurations, distinct diagnostic
states, explicit failures, stale data/configuration/version rejection, immutable
approval and a complete journey without an agent. Infrastructure integration tests
stay separate. No tests import builders from collected test modules.

Budget accounting includes colocated tests and guidance correctly; limits remain
unchanged. Existing broad debt from `/tmp/causal-t038-budget-before.json` remains
visible. The user's explicit implementation scope authorizes completing this
refactor while reporting that debt; no budget pass or release approval is implied.
The earlier 400-line extraction estimate is superseded by this approved larger
public boundary. Prefer direct typed models and ordinary functions; no generic
workflow engine, general rule language, parallel new estimators or search service.
